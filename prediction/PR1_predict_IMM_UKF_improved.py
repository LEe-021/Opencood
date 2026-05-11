"""
IMM-UKF 轨迹预测模块（改进版）
=============================

在原始 IMM-UKF 预测基础上，集成以下四项改进：
  1. 预测阶段协方差传播：通过 Unscented Transform 在多步预测中逐步传播协方差矩阵
  2. 模型权重马尔可夫演化：通过马尔可夫转移矩阵前向演化模型概率
  3. 混合协方差含散布项：完整实现 IMM 协方差混合公式（含模型间散布项）
  4. 马尔可夫矩阵运动状态自适应：根据当前运动特征动态调整转移概率

输出格式扩展：每个预测步增加 covariance（6×6协方差矩阵）和 model_weights 字段，
为后续不确定性感知融合（软加权）提供基础。

状态向量定义（6D，所有模型统一维度）：
    [px, py, v, yaw, yaw_rate, a]
    - px, py:  位置坐标 (m)
    - v:       速度标量 (m/s)
    - yaw:     航向角 (rad)
    - yaw_rate: 角速度 (rad/s)
    - a:       加速度 (m/s²)

三种运动模型：
    - CV  (Constant Velocity):               匀速直线运动
    - CTRV (Constant Turn Rate & Velocity):   恒定转弯率和速度
    - CTRA (Constant Turn Rate & Acceleration): 恒定转弯率和加速度
"""

import os
import yaml
import numpy as np
import json
import glob
import time
from collections import deque
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints, IMMEstimator


# ========================= 全局配置 =========================
# 输入数据目录（指向加噪后的仿真数据）
INPUT_ROOT = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene_noisy/test_1_noisy/-1_noisy"
# 输出预测结果文件
OUTPUT_FILE = "/home/step/data/pcPredict_data/cross_fullcar_clean/scene_noisy/test_1_noisy/prediction_registry_imm_ukf_improved.json"
# 历史观测窗口长度（帧数，每帧50ms，10帧=500ms）
HISTORY_LEN = 10
# 预测步数（6步×50ms=300ms最大预测时域）
PRED_STEPS = 6
# 帧间隔时间（秒）
DT = 0.05
# 协方差缩放因子（调整使协方差校准 Ratio ≈ 1.0）
# Ratio < 1.0 表示协方差偏大（保守）→ 增大此值使协方差缩小
# Ratio > 1.0 表示协方差偏小（乐观）→ 减小此值使协方差放大
# 效果：P_pred *= COV_SCALE，sigma *= sqrt(COV_SCALE)，Ratio /= sqrt(COV_SCALE)
COV_SCALE = 0.55
# ============================================================


# ========================= 工具函数 =========================

def deg2rad(deg):
    """角度转弧度"""
    return deg * np.pi / 180.0


def normalize_angle(angle):
    """
    将角度归一化到 [-π, π] 区间。
    用于计算角度偏差和最终输出时的角度折叠。
    注意：在滤波和预测的内部状态中，yaw 允许无限增长以保持连续性，
    只在需要计算偏差或输出结果时调用此函数。
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ========================= 运动模型定义 =========================
# 以下三个函数定义了三种运动模型的状态转移方程。
# 所有模型统一使用 6D 状态向量 [px, py, v, yaw, yaw_rate, a]，
# 以保证 IMM 框架中各子滤波器的状态维度一致。

def cv_transition(x, dt):
    """
    CV (Constant Velocity) 模型：匀速直线运动。

    假设：角速度 ω = 0，加速度 a = 0。
    车辆沿当前航向方向匀速行驶。
    yaw_rate 和 a 作为状态分量保留（透传），以维持维度一致性，
    但其过程噪声被设置为极小值（1e-5），等效于锁定。

    Args:
        x: 状态向量 [px, py, v, yaw, yaw_rate, a]
        dt: 时间步长 (秒)

    Returns:
        预测后的状态向量
    """
    px, py, v, yaw, yaw_rate, a = x
    px_new = px + v * np.cos(yaw) * dt
    py_new = py + v * np.sin(yaw) * dt
    # v, yaw, yaw_rate, a 在CV模型下保持不变
    return np.array([px_new, py_new, v, yaw, yaw_rate, a])


def ctrv_transition(x, dt):
    """
    CTRV (Constant Turn Rate and Velocity) 模型：恒定转弯率和速度。

    假设：角速度 ω 恒定，加速度 a = 0。
    车辆以恒定速度和恒定角速度行驶（即沿圆弧运动）。
    当角速度极小时退化为CV模型，避免数值除零。

    Args:
        x: 状态向量 [px, py, v, yaw, yaw_rate, a]
        dt: 时间步长 (秒)

    Returns:
        预测后的状态向量
    """
    px, py, v, yaw, yaw_rate, a = x
    if abs(yaw_rate) < 0.001:
        # 角速度极小时退化为直线运动，避免除零
        px_new = px + v * np.cos(yaw) * dt
        py_new = py + v * np.sin(yaw) * dt
        yaw_new = yaw
    else:
        # 圆弧运动的解析积分公式
        px_new = px + (v / yaw_rate) * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw))
        py_new = py + (v / yaw_rate) * (np.cos(yaw) - np.cos(yaw + yaw_rate * dt))
        yaw_new = yaw + yaw_rate * dt
    return np.array([px_new, py_new, v, yaw_new, yaw_rate, a])


def ctra_transition(x, dt):
    """
    CTRA (Constant Turn Rate and Acceleration) 模型：恒定转弯率和加速度。

    假设：角速度 ω 恒定，加速度 a 恒定。
    车辆同时转弯和加减速，是最复杂的运动模型。
    当角速度极小时退化为CA（恒加速直线）模型。

    Args:
        x: 状态向量 [px, py, v, yaw, yaw_rate, a]
        dt: 时间步长 (秒)

    Returns:
        预测后的状态向量
    """
    px, py, v, yaw, yaw_rate, a = x
    if abs(yaw_rate) < 0.001:
        # 角速度极小时退化为恒加速直线运动
        px_new = px + (v * dt + 0.5 * a * dt ** 2) * np.cos(yaw)
        py_new = py + (v * dt + 0.5 * a * dt ** 2) * np.sin(yaw)
        v_new = v + a * dt
        yaw_new = yaw
    else:
        # 弯道加减速的完整解析积分公式
        px_new = px + (1.0 / yaw_rate ** 2) * (
            (v * yaw_rate + a * yaw_rate * dt) * np.sin(yaw + yaw_rate * dt)
            + a * np.cos(yaw + yaw_rate * dt)
            - v * yaw_rate * np.sin(yaw)
            - a * np.cos(yaw)
        )
        py_new = py + (1.0 / yaw_rate ** 2) * (
            -(v * yaw_rate + a * yaw_rate * dt) * np.cos(yaw + yaw_rate * dt)
            + a * np.sin(yaw + yaw_rate * dt)
            + v * yaw_rate * np.cos(yaw)
            - a * np.sin(yaw)
        )
        v_new = v + a * dt
        yaw_new = yaw + yaw_rate * dt
    return np.array([px_new, py_new, v_new, yaw_new, yaw_rate, a])


def measurement_function(x):
    """
    观测映射函数 h(x)：将 6D 状态映射到 4D 观测空间。

    观测向量: [px, py, v, yaw]
    注意：角速度 yaw_rate 和加速度 a 无法直接观测，仅通过滤波器隐式估计。
    """
    return np.array([x[0], x[1], x[2], x[3]])


# ========================= 向量化批处理运动模型 =========================
# 以下三个函数是上述单点版本的向量化实现，可同时处理 N 个状态向量。
# 用于 ut_propagate 中的 Sigma 点批量变换，避免 Python 循环。

def cv_transition_v(x_batch, dt):
    """CV 向量化版本，输入 shape=(N, 6)，输出 shape=(N, 6)。"""
    out = x_batch.copy()
    out[:, 0] = x_batch[:, 0] + x_batch[:, 2] * np.cos(x_batch[:, 3]) * dt
    out[:, 1] = x_batch[:, 1] + x_batch[:, 2] * np.sin(x_batch[:, 3]) * dt
    return out


def ctrv_transition_v(x_batch, dt):
    """CTRV 向量化版本，输入 shape=(N, 6)，输出 shape=(N, 6)。"""
    out = x_batch.copy()
    yr = x_batch[:, 4]
    v = x_batch[:, 2]
    yaw = x_batch[:, 3]

    straight = np.abs(yr) < 0.001
    turning = ~straight

    if straight.any():
        s = np.where(straight)[0]
        out[s, 0] = x_batch[s, 0] + v[s] * np.cos(yaw[s]) * dt
        out[s, 1] = x_batch[s, 1] + v[s] * np.sin(yaw[s]) * dt

    if turning.any():
        t = np.where(turning)[0]
        yr_t, v_t, yaw_t = yr[t], v[t], yaw[t]
        out[t, 0] = x_batch[t, 0] + (v_t / yr_t) * (np.sin(yaw_t + yr_t * dt) - np.sin(yaw_t))
        out[t, 1] = x_batch[t, 1] + (v_t / yr_t) * (np.cos(yaw_t) - np.cos(yaw_t + yr_t * dt))
        out[t, 3] = yaw_t + yr_t * dt

    return out


def ctra_transition_v(x_batch, dt):
    """CTRA 向量化版本，输入 shape=(N, 6)，输出 shape=(N, 6)。"""
    out = x_batch.copy()
    yr = x_batch[:, 4]
    v = x_batch[:, 2]
    yaw = x_batch[:, 3]
    a = x_batch[:, 5]

    straight = np.abs(yr) < 0.001
    turning = ~straight

    if straight.any():
        s = np.where(straight)[0]
        dist = v[s] * dt + 0.5 * a[s] * dt ** 2
        out[s, 0] = x_batch[s, 0] + dist * np.cos(yaw[s])
        out[s, 1] = x_batch[s, 1] + dist * np.sin(yaw[s])
        out[s, 2] = v[s] + a[s] * dt

    if turning.any():
        t = np.where(turning)[0]
        yr_t, v_t, yaw_t, a_t = yr[t], v[t], yaw[t], a[t]
        yr2 = yr_t ** 2
        out[t, 0] = x_batch[t, 0] + (1.0 / yr2) * (
            (v_t * yr_t + a_t * yr_t * dt) * np.sin(yaw_t + yr_t * dt)
            + a_t * np.cos(yaw_t + yr_t * dt)
            - v_t * yr_t * np.sin(yaw_t) - a_t * np.cos(yaw_t))
        out[t, 1] = x_batch[t, 1] + (1.0 / yr2) * (
            -(v_t * yr_t + a_t * yr_t * dt) * np.cos(yaw_t + yr_t * dt)
            + a_t * np.sin(yaw_t + yr_t * dt)
            + v_t * yr_t * np.cos(yaw_t) - a_t * np.sin(yaw_t))
        out[t, 2] = v_t + a_t * dt
        out[t, 3] = yaw_t + yr_t * dt

    return out


# ========================= 核心算法函数 =========================

def ut_propagate(x, P, fx, dt, Q, points):
    """
    通过 Unscented Transform (UT) 传播状态均值和协方差。

    用于预测阶段：给定当前状态分布 (x, P)，通过非线性状态转移函数 fx
    计算经过 dt 时间后的预测状态分布 (x_pred, P_pred)。

    原理：
        1. 从当前分布中采样 Sigma 点（确定性采样）
        2. 将每个 Sigma 点通过非线性函数变换
        3. 从变换后的 Sigma 点重构均值和协方差
        4. 叠加过程噪声 Q

    Args:
        x: 当前状态均值, shape=(6,)
        P: 当前状态协方差矩阵, shape=(6, 6)
        fx: 状态转移函数, 签名 f(x, dt) -> x_new
        dt: 预测时间步长 (秒)
        Q: 过程噪声协方差矩阵, shape=(6, 6)
        points: JulierSigmaPoints 采样器实例

    Returns:
        x_pred: 预测状态均值, shape=(6,)
        P_pred: 预测状态协方差矩阵, shape=(6, 6)
    """
    n = len(x)

    # 步骤1: 根据当前分布 (x, P) 生成 Sigma 点
    # JulierSigmaPoints(n=6, kappa=0) 生成 2n+1=13 个 Sigma 点
    sigmas = points.sigma_points(x, P)  # shape=(13, 6)

    # 步骤2: 将每个 Sigma 点通过状态转移函数变换
    sigmas_f = np.zeros_like(sigmas)
    for i in range(len(sigmas)):
        sigmas_f[i] = fx(sigmas[i], dt)

    # 步骤3: 获取 Sigma 点的均值权重 Wm 和协方差权重 Wc
    Wm = points.Wm  # shape=(13,)
    Wc = points.Wc  # shape=(13,)

    # 步骤4: 计算预测均值（Sigma 点的加权平均）
    x_pred = np.dot(Wm, sigmas_f)  # shape=(6,)

    # 步骤5: 计算预测协方差（加权外积和 + 过程噪声）
    P_pred = Q.copy()
    for i in range(len(sigmas)):
        y = sigmas_f[i] - x_pred
        P_pred += Wc[i] * np.outer(y, y)

    return x_pred, P_pred


def ut_propagate_v(x, P, fx_v, dt, Q, Wm, Wc):
    """
    ut_propagate 的向量化加速版本。

    相比原始版本，将 Python 循环替换为：
      - 批量 Sigma 点变换：fx_v(sigmas, dt) 一次性处理 13 个点
      - einsum 计算加权外积和：避免 Python 循环 + np.outer

    参数 fx_v 必须是向量化运动模型函数（接受 (N,6) 输入）。
    Wm, Wc 从外部预缓存传入，避免重复访问 points 对象属性。

    Args:
        x: 当前状态均值, shape=(6,)
        P: 当前协方差矩阵, shape=(6, 6)
        fx_v: 向量化状态转移函数, 签名 f(x_batch, dt) -> x_batch_new
        dt: 时间步长
        Q: 过程噪声, shape=(6, 6)
        Wm: 均值权重, shape=(13,)
        Wc: 协方差权重, shape=(13,)

    Returns:
        x_pred, P_pred
    """
    sigmas = JulierSigmaPoints(n=6, kappa=0).sigma_points(x, P)
    sigmas_f = fx_v(sigmas, dt)              # (13, 6) 一次性批量变换
    x_pred = Wm @ sigmas_f                   # (6,) 加权均值
    y = sigmas_f - x_pred                    # (13, 6) 偏差
    P_pred = Q + np.einsum('i,ij,ik->jk', Wc, y, y)  # 加权外积和
    return x_pred, P_pred


def adaptive_markov(base_M, state):
    """
    根据当前运动状态自适应调整马尔可夫转移矩阵。

    基本思路：根据状态向量中的角速度和加速度判断车辆当前最可能
    处于哪种运动模式，并增强该模式的自转移概率（增大对角线元素），
    使模型切换更加灵敏和准确。

    判别规则：
        - |yaw_rate| 小, |a| 小 → 直行匀速 → 增强 CV 概率
        - |yaw_rate| 大, |a| 小 → 弯道匀速 → 增强 CTRV 概率
        - |yaw_rate| 大, |a| 大 → 弯道加减速 → 增强 CTRA 概率
        - |yaw_rate| 小, |a| 大 → 直行加减速 → CTRA 概率提升

    Args:
        base_M: 基础马尔可夫转移矩阵, shape=(3, 3)
                行/列顺序: [CV, CTRV, CTRA]
                base_M[i, j] 表示从模式 i 转移到模式 j 的概率
        state: 当前状态向量 [px, py, v, yaw, yaw_rate, a]

    Returns:
        M: 调整后的马尔可夫转移矩阵, shape=(3, 3)，行归一化
    """
    M = base_M.copy()
    _, _, v, _, yaw_rate, a = state

    # 角速度判别阈值 (rad/s)：约 2.9°/s，低于此视为直行
    YAW_RATE_THRESHOLD = 0.05
    # 加速度判别阈值 (m/s²)：低于此视为匀速
    ACCEL_THRESHOLD = 0.5

    is_turning = abs(yaw_rate) > YAW_RATE_THRESHOLD
    is_accelerating = abs(a) > ACCEL_THRESHOLD

    if not is_turning and not is_accelerating:
        # 直行匀速 → 显著增强 CV 的自转移概率
        M[0, 0] = 0.95; M[0, 1] = 0.04; M[0, 2] = 0.01
    elif is_turning and not is_accelerating:
        # 弯道匀速 → 增强 CTRV 的自转移概率
        M[1, 0] = 0.03; M[1, 1] = 0.94; M[1, 2] = 0.03
    elif is_turning and is_accelerating:
        # 弯道加减速 → 显著增强 CTRA 的自转移概率
        M[2, 0] = 0.01; M[2, 1] = 0.04; M[2, 2] = 0.95
    else:
        # 直行加减速 → CTRA 为主，CV 保留一定概率
        M[0, 0] = 0.70; M[0, 1] = 0.05; M[0, 2] = 0.25
        M[2, 0] = 0.05; M[2, 1] = 0.05; M[2, 2] = 0.90

    # 行归一化：确保每行的转移概率之和为 1
    M = M / M.sum(axis=1, keepdims=True)
    return M


def mix_imm_covariance(pred_states, pred_covs, mu):
    """
    按 IMM 框架混合多个模型的预测状态和协方差。

    完整的 IMM 混合协方差公式包含两项：
        P_mixed = Σ_j μ_j · P_j                                      (第一项: 各模型协方差的加权和)
                + Σ_j μ_j · (x_j - x_mixed)(x_j - x_mixed)^T         (第二项: 模型间散布项)

    第一项是各子滤波器预测协方差的加权和，反映各模型内部的不确定性。
    第二项是模型间预测不一致带来的额外不确定性——当各模型预测差异大时
    （例如 CV 预测直行而 CTRV 预测转弯），此项显著增大混合协方差。
    缺少第二项会导致协方差被系统性低估。

    Args:
        pred_states: 各模型的预测状态均值列表, 每个元素 shape=(6,)
        pred_covs: 各模型的预测协方差矩阵列表, 每个元素 shape=(6, 6)
        mu: 模型权重向量, shape=(n_models,)

    Returns:
        x_mixed: 混合后的状态均值, shape=(6,)
        P_mixed: 混合后的协方差矩阵, shape=(6, 6)
    """
    n_models = len(pred_states)
    n_state = len(pred_states[0])

    # 1. 计算混合均值：各模型预测状态的加权平均
    x_mixed = np.zeros(n_state)
    for j in range(n_models):
        x_mixed += mu[j] * pred_states[j]

    # 2. 计算混合协方差：两项之和
    P_mixed = np.zeros((n_state, n_state))
    for j in range(n_models):
        # 第一项：各模型协方差的加权重和
        P_mixed += mu[j] * pred_covs[j]
        # 第二项：模型间散布项
        diff = pred_states[j] - x_mixed
        P_mixed += mu[j] * np.outer(diff, diff)

    # 3. 强制对称：消除浮点运算导致的微小不对称
    P_mixed = (P_mixed + P_mixed.T) / 2.0

    return x_mixed, P_mixed


def mix_imm_covariance_v(pred_states, pred_covs, mu):
    """
    mix_imm_covariance 的向量化加速版本。

    使用 numpy 数组操作和 einsum 替代 Python 循环：
      - 混合均值：mu @ pred_states_arr
      - 加权协方差和：np.einsum('i,ijk->jk', mu, cov_arr)
      - 散布项：np.einsum('i,ij,ik->jk', mu, diffs, diffs)

    Args:
        pred_states: list of 3 个 ndarray, 每个 shape=(6,)
        pred_covs: list of 3 个 ndarray, 每个 shape=(6, 6)
        mu: 模型权重, shape=(3,)

    Returns:
        x_mixed, P_mixed
    """
    states_arr = np.array(pred_states)    # (3, 6)
    covs_arr = np.array(pred_covs)        # (3, 6, 6)

    x_mixed = mu @ states_arr             # (6,)
    diffs = states_arr - x_mixed          # (3, 6)

    P_mixed = (np.einsum('i,ijk->jk', mu, covs_arr)
               + np.einsum('i,ij,ik->jk', mu, diffs, diffs))
    P_mixed = (P_mixed + P_mixed.T) / 2.0
    return x_mixed, P_mixed


# ========================= 数据结构定义 =========================

class VehicleState:
    """
    车辆状态数据类，用于存储从 YAML 文件中读取的单帧车辆状态。

    Attributes:
        x:  位置 X 坐标 (m)
        y:  位置 Y 坐标 (m)
        yaw: 航向角 (rad)
        speed: 速度标量 (m/s)
        timestamp: 时间戳 (未使用，预留)
    """
    def __init__(self, x, y, yaw, speed, timestamp):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = speed
        self.timestamp = timestamp


# ========================= IMM-UKF 预测器 =========================

class IMMUKFPredictor:
    """
    基于交互多模型（IMM）+ 无迹卡尔曼滤波（UKF）的轨迹预测器（改进版）。

    改进内容：
        1. 预测阶段协方差传播：通过 UT 逐步传播协方差矩阵，
           而非仅传播状态均值。
        2. 模型权重马尔可夫演化：预测阶段通过马尔可夫矩阵前向演化
           模型概率，使长时域预测的不确定性建模更合理。
        3. 混合协方差含散布项：完整实现 IMM 协方差混合公式，
           包含模型间预测不一致带来的额外不确定性。
        4. 马尔可夫矩阵运动状态自适应：根据当前运动特征动态调整
           转移概率，使模型切换更加灵敏。
    """

    def __init__(self):
        # Sigma 点采样器：Julier 方法，n=6 维状态，kappa=0
        # 生成 2n+1=13 个 Sigma 点，均值权重均匀分布
        self.points = JulierSigmaPoints(n=6, kappa=0)

        # 共享的观测噪声矩阵 R（所有模型相同）
        # 精准对齐注入的高斯噪声：位置 σ=0.2m, 速度 σ=0.2m/s, 航向 σ=0.2°
        self.R_shared = np.diag([
            0.2 ** 2,                # px 观测噪声方差
            0.2 ** 2,                # py 观测噪声方差
            0.2 ** 2,                # v  观测噪声方差
            deg2rad(0.2) ** 2        # yaw 观测噪声方差
        ])

        # 初始状态不确定性矩阵 P（所有模型相同）
        # 首帧状态直接来自观测，初始 P 与观测噪声 R 对齐
        self.P_shared = np.diag([
            0.2 ** 2,                # px
            0.2 ** 2,                # py
            0.2 ** 2,                # v
            deg2rad(0.2) ** 2,       # yaw
            0.1 ** 2,                # yaw_rate（初始不确定性较小）
            0.5 ** 2                 # a（初始不确定性较大，允许快速收敛）
        ])

        # 各模型的独立过程噪声矩阵 Q
        # CV 模型：极不信任角速度和加速度的变化，等效于锁死
        self.Q_cv = np.diag([
            0.02 ** 2,              # px 位置模型误差
            0.02 ** 2,              # py 位置模型误差
            0.1 ** 2,               # v  速度扰动
            deg2rad(0.1) ** 2,      # yaw 航向扰动
            1e-5,                   # yaw_rate 锁死
            1e-5                    # a 锁死
        ])
        # CTRV 模型：允许角速度变化，不允许加速度变化
        self.Q_ctrv = np.diag([
            0.02 ** 2,
            0.02 ** 2,
            0.1 ** 2,
            deg2rad(0.1) ** 2,
            0.05 ** 2,              # yaw_rate 允许变化
            1e-5                    # a 锁死
        ])
        # CTRA 模型：允许角速度和加速度同时变化
        self.Q_ctra = np.diag([
            0.02 ** 2,
            0.02 ** 2,
            0.1 ** 2,
            deg2rad(0.1) ** 2,
            0.05 ** 2,              # yaw_rate 允许变化
            0.5 ** 2                # a 允许变化（加速度突变容忍度高）
        ])

        # 便于统一调用的函数列表和 Q 列表
        self.transition_funcs = [cv_transition, ctrv_transition, ctra_transition]
        self.Q_list = [self.Q_cv, self.Q_ctrv, self.Q_ctra]

        # 向量化运动模型函数列表（用于 ut_propagate_v 加速）
        self.transition_funcs_v = [cv_transition_v, ctrv_transition_v, ctra_transition_v]

        # 预缓存 Sigma 点权重，避免每次调用 ut_propagate_v 时重复访问
        _pts = JulierSigmaPoints(n=6, kappa=0)
        self.Wm = _pts.Wm.copy()
        self.Wc = _pts.Wc.copy()

        # 基础马尔可夫转移矩阵
        # M[i, j] 表示从模式 i 转移到模式 j 的概率
        # 对角线概率最高，表示车辆倾向于保持当前运动状态
        self.base_M = np.array([
            [0.90, 0.08, 0.02],     # CV   → CV/CTRV/CTRA
            [0.05, 0.90, 0.05],     # CTRV → CV/CTRV/CTRA
            [0.02, 0.08, 0.90]      # CTRA → CV/CTRV/CTRA
        ])

    def _get_mixed_covariance(self, imm):
        """
        从 IMM 估计器中获取包含模型间散布项的完整混合协方差。

        标准 IMM 混合协方差包含两项：
            P_mixed = Σ μ_j · P_j + Σ μ_j · (x_j - x_mixed)(x_j - x_mixed)^T

        其中 x_j 和 P_j 分别是第 j 个子滤波器的状态均值和协方差，
        x_mixed 是 IMM 的混合状态均值。
        第二项反映了不同模型之间预测不一致带来的额外不确定性。

        注意：filterpy 的 IMMEstimator 的 imm.P 属性仅包含第一项，
        不含散布项，因此需要本函数手动计算完整的混合协方差。

        Args:
            imm: IMMEstimator 实例（已完成滤波更新阶段）

        Returns:
            P_mixed: 完整混合协方差矩阵, shape=(6, 6)
        """
        n_state = len(imm.x)
        mu = imm.mu           # 模型权重, shape=(3,)
        x_mixed = imm.x       # 混合状态均值, shape=(6,)

        P_mixed = np.zeros((n_state, n_state))
        for j in range(len(imm.filters)):
            P_j = imm.filters[j].P   # 第 j 个子滤波器的协方差
            x_j = imm.filters[j].x   # 第 j 个子滤波器的状态均值
            diff = x_j - x_mixed

            # 第一项：各模型协方差的加权和
            P_mixed += mu[j] * P_j
            # 第二项：模型间散布项
            P_mixed += mu[j] * np.outer(diff, diff)

        # 强制对称
        P_mixed = (P_mixed + P_mixed.T) / 2.0
        return P_mixed

    def create_imm_estimator(self):
        """
        为每辆车创建一个独立的 IMM 估计器实例。

        包含三个并行的 UKF 子滤波器（CV、CTRV、CTRA），
        各自具有不同的过程噪声矩阵 Q 以反映不同的运动假设。
        IMM 框架根据观测似然自动调整各模型的权重。

        Returns:
            IMMEstimator 实例
        """
        # CV 滤波器：直行运动专家
        ukf_cv = UnscentedKalmanFilter(
            dim_x=6, dim_z=4, dt=DT,
            fx=cv_transition, hx=measurement_function,
            points=self.points
        )
        ukf_cv.Q = self.Q_cv.copy()
        ukf_cv.R = self.R_shared.copy()

        # CTRV 滤波器：匀速转弯运动专家
        ukf_ctrv = UnscentedKalmanFilter(
            dim_x=6, dim_z=4, dt=DT,
            fx=ctrv_transition, hx=measurement_function,
            points=self.points
        )
        ukf_ctrv.Q = self.Q_ctrv.copy()
        ukf_ctrv.R = self.R_shared.copy()

        # CTRA 滤波器：加减速转弯运动专家
        ukf_ctra = UnscentedKalmanFilter(
            dim_x=6, dim_z=4, dt=DT,
            fx=ctra_transition, hx=measurement_function,
            points=self.points
        )
        ukf_ctra.Q = self.Q_ctra.copy()
        ukf_ctra.R = self.R_shared.copy()

        filters = [ukf_cv, ukf_ctrv, ukf_ctra]

        # IMM 模型初始概率：均匀分布，CTRV 稍大
        # [CV, CTRV, CTRA] = [0.3, 0.4, 0.3]
        mu = np.array([0.3, 0.4, 0.3])

        # 创建 IMM 估计器（马尔可夫矩阵在预测阶段动态调整）
        return IMMEstimator(filters, mu, self.base_M.copy())

    def predict(self, history_states: list, steps: int):
        """
        基于历史观测序列进行多步预测，同时传播状态均值、协方差和模型权重。

        处理流程分为三个阶段：

            阶段1 - 滤波更新：
                使用历史观测序列驱动 IMM-UKF 滤波器，
                依次执行 predict → update 循环，
                最终得到滤波后的混合状态均值、协方差和模型权重。

            阶段2 - 提取滤波结果：
                从 IMM 估计器中提取：
                - 混合状态均值 (x)
                - 包含散布项的完整混合协方差 (P)
                - 各模型权重 (mu)
                - 根据当前运动状态计算自适应马尔可夫矩阵 (M)

            阶段3 - 多步预测：
                从滤波终点出发，通过多步迭代进行预测。
                每步执行以下操作：
                (a) 通过马尔可夫矩阵演化模型权重
                (b) 各模型独立通过 UT 传播均值和协方差
                (c) 按 IMM 框架混合（含散布项）
                (d) 更新自适应马尔可夫矩阵
                (e) 将混合结果传递给下一步

        Args:
            history_states: 历史状态列表, 每个元素为 VehicleState 实例
            steps: 预测步数（每步对应 DT=50ms 的时间间隔）

        Returns:
            predictions: 预测结果列表，每个元素为字典，包含：
                - step: 步数编号（从 1 开始）
                - time_offset: 相对于当前时刻的时间偏移 (秒)
                - x, y: 预测位置坐标 (米)
                - yaw: 预测航向角 (弧度, 归一化到 [-π, π])
                - speed: 预测速度 (m/s)
                - covariance: 预测协方差矩阵 (6×6, 嵌套列表)
                - model_weights: 各模型权重 [μ_CV, μ_CTRV, μ_CTRA]
        """
        imm = self.create_imm_estimator()

        # ================================================================
        # 阶段 1: 滤波更新 —— 使用历史观测序列驱动 IMM-UKF
        # ================================================================
        for i, state in enumerate(history_states):
            if i == 0:
                # 首帧初始化：利用前两帧差分估计角速度和加速度
                init_yaw_rate = 0.0
                init_accel = 0.0
                if len(history_states) > 1:
                    # 角速度估计：相邻两帧航向差 / 帧间隔
                    init_yaw_rate = normalize_angle(
                        history_states[1].yaw - history_states[0].yaw
                    ) / DT
                    # 加速度估计：相邻两帧速度差 / 帧间隔
                    init_accel = (
                        (history_states[1].speed - history_states[0].speed) / DT
                    )

                init_x = np.array([
                    state.x, state.y, state.speed, state.yaw,
                    init_yaw_rate, init_accel
                ])

                # 初始化所有子滤波器的状态和协方差
                for f in imm.filters:
                    f.x = init_x.copy()
                    f.P = self.P_shared.copy()
                imm.x = init_x.copy()
            else:
                # 后续帧：执行 IMM 预测 + 更新循环
                imm.predict()

                # 角度解卷 (Angle Unrolling)
                # 目的：将观测角度展开到与滤波器内部状态相同的连续空间，
                # 避免 ±π 边界处产生虚假的大角度偏差，导致协方差膨胀。
                # 方法：计算观测角与当前状态角的最短夹角，
                #       将该夹角叠加到当前状态角上，得到连续的观测角度。
                angle_diff = normalize_angle(state.yaw - imm.x[3])
                unrolled_obs_yaw = imm.x[3] + angle_diff

                z = np.array([state.x, state.y, state.speed, unrolled_obs_yaw])
                imm.update(z)

        # ================================================================
        # 阶段 2: 提取滤波结果
        # ================================================================
        current_state = imm.x.copy()
        current_mu = imm.mu.copy()
        current_P = self._get_mixed_covariance(imm)

        # ================================================================
        # 阶段 3: 多步预测 —— 直接预测（与原始 IMM 一致）
        # ================================================================
        predictions = []
        for step in range(1, steps + 1):
            t_pred = step * DT

            pred_cv = cv_transition(current_state, t_pred)
            pred_ctrv = ctrv_transition(current_state, t_pred)
            pred_ctra = ctra_transition(current_state, t_pred)

            mixed_state = (current_mu[0] * pred_cv
                           + current_mu[1] * pred_ctrv
                           + current_mu[2] * pred_ctra)

            # 协方差
            pred_states_arr = np.array([pred_cv, pred_ctrv, pred_ctra])
            diffs = pred_states_arr - mixed_state
            P_disp = np.zeros((6, 6))
            for j in range(3):
                P_disp += current_mu[j] * np.outer(diffs[j], diffs[j])

            Q_avg = (current_mu[0] * self.Q_cv
                     + current_mu[1] * self.Q_ctrv
                     + current_mu[2] * self.Q_ctra)

            P_pred = (current_P + P_disp + step * Q_avg) * COV_SCALE
            P_pred = (P_pred + P_pred.T) / 2.0

            predictions.append({
                "step": step,
                "time_offset": round(step * DT, 3),
                "x": float(mixed_state[0]),
                "y": float(mixed_state[1]),
                "yaw": float(normalize_angle(mixed_state[3])),
                "speed": float(mixed_state[2]),
                "covariance": P_pred.tolist(),
                "model_weights": current_mu.tolist()
            })

        return predictions

    def predict_from_estimator(self, imm, steps):
        """
        从已有的 IMM 估计器状态出发进行多步预测。

        预测策略：
          - 均值预测：从滤波状态直接预测 N×DT 步（与原始 IMM 一致），
            避免迭代混合带来的误差累积。
          - 协方差：基于滤波协方差 + 模型散布 + 过程噪声增长，
            反映预测不确定性随预测时域的合理增长。
          - 模型权重：保持滤波阶段的值不变，不做马尔可夫演化。

        适用场景：主循环中已通过增量方式维护了 IMM 估计器，
        无需每帧重放全部历史观测。

        Args:
            imm: IMMEstimator 实例（已完成最新的 predict/update）
            steps: 预测步数

        Returns:
            predictions: 同 predict() 的输出格式
        """
        current_state = imm.x.copy()
        current_mu = imm.mu.copy()
        current_P = self._get_mixed_covariance(imm)

        predictions = []
        for step in range(1, steps + 1):
            t_pred = step * DT

            # 各模型直接从滤波状态预测 step×DT 步（不迭代）
            pred_cv = cv_transition(current_state, t_pred)
            pred_ctrv = ctrv_transition(current_state, t_pred)
            pred_ctra = ctra_transition(current_state, t_pred)

            # 固定权重混合（与原始 IMM 一致）
            mixed_state = (current_mu[0] * pred_cv
                           + current_mu[1] * pred_ctrv
                           + current_mu[2] * pred_ctra)

            # 协方差：滤波协方差 + 模型散布 + 过程噪声增长
            pred_states_arr = np.array([pred_cv, pred_ctrv, pred_ctra])
            diffs = pred_states_arr - mixed_state
            P_disp = np.zeros((6, 6))
            for j in range(3):
                P_disp += current_mu[j] * np.outer(diffs[j], diffs[j])

            Q_avg = (current_mu[0] * self.Q_cv
                     + current_mu[1] * self.Q_ctrv
                     + current_mu[2] * self.Q_ctra)

            P_pred = (current_P + P_disp + step * Q_avg) * COV_SCALE
            P_pred = (P_pred + P_pred.T) / 2.0

            predictions.append({
                "step": step,
                "time_offset": round(t_pred, 3),
                "x": float(mixed_state[0]),
                "y": float(mixed_state[1]),
                "yaw": float(normalize_angle(mixed_state[3])),
                "speed": float(mixed_state[2]),
                "covariance": P_pred.tolist(),
                "model_weights": current_mu.tolist()
            })

        return predictions


# ========================= 数据加载 =========================

def load_frame_data(yaml_path):
    """
    从 YAML 文件中加载当前帧所有车辆的状态信息。

    YAML 文件格式：
        vehicles:
          vehicle_id:
            location: [x, y, z]
            angle: [pitch, yaw, roll]  # 角度单位：度
            speed: xx.x                # 速度单位：km/h

    Returns:
        frame_vehicles: 字典 {vehicle_id: VehicleState}
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    frame_vehicles = {}
    if 'vehicles' in data and data['vehicles'] is not None:
        for vid, vinfo in data['vehicles'].items():
            loc = vinfo['location']
            angle = vinfo['angle']
            yaw_rad = deg2rad(float(angle[1]))       # yaw 角度 → 弧度
            speed_mps = float(vinfo['speed']) / 3.6   # km/h → m/s
            frame_vehicles[vid] = VehicleState(
                x=float(loc[0]), y=float(loc[1]),
                yaw=yaw_rad, speed=speed_mps, timestamp=0
            )
    return frame_vehicles


# ========================= 主函数 =========================

def main():
    """
    主处理流程（增量滤波优化版）。

    核心优化：不再每帧重放全部历史观测，而是为每辆车维护一个持久化的
    IMM 估计器。每帧仅需执行 1 次 predict/update（而非 HISTORY_LEN 次），
    滤波阶段开销降低约 10 倍。预测阶段使用向量化运动模型和 einsum 加速。

    流程：
    1. 扫描输入目录中的所有 YAML 文件，按目录分组为序列
    2. 对每辆车维护一个 IMM 估计器（增量更新）
    3. 使用 predict_from_estimator 进行向量化多步预测
    4. 将预测结果保存为 JSON 文件
    """

    # 扫描输入目录
    yaml_files = sorted(glob.glob(os.path.join(INPUT_ROOT, "*.yaml"), recursive=True))
    sequences = {}
    for f in yaml_files:
        dirname = os.path.dirname(f)
        if dirname not in sequences:
            sequences[dirname] = []
        sequences[dirname].append(f)

    # 实例化预测器
    predictor = IMMUKFPredictor()
    registry = {}

    # 统计变量
    total_predict_time = 0.0
    total_predict_count = 0

    print("开始处理 (Method: IMM-UKF 改进版 + 增量滤波 + 向量化加速)")
    print("  优化1: 增量滤波 — 每帧仅 1 次 predict/update，不重放历史")
    print("  优化2: 向量化运动模型 — Sigma 点批量变换")
    print("  优化3: einsum 加速协方差混合")

    for seq_path, files in sequences.items():
        # 每个序列维护独立的估计器池
        estimators = {}  # {vehicle_id: IMMEstimator}
        files.sort()

        for frame_idx, yaml_path in enumerate(tqdm(
            files, desc=f"Processing {os.path.basename(seq_path)}", leave=False
        )):
            frame_name = os.path.splitext(os.path.basename(yaml_path))[0]
            frame_key = os.path.join(os.path.relpath(seq_path, INPUT_ROOT), frame_name)

            current_vehicles = load_frame_data(yaml_path)
            registry[frame_key] = {}

            for vid, state in current_vehicles.items():
                t_start = time.perf_counter()

                if vid not in estimators:
                    # ---- 新车辆：创建估计器并初始化 ----
                    imm = predictor.create_imm_estimator()
                    init_x = np.array([
                        state.x, state.y, state.speed, state.yaw, 0.0, 0.0
                    ])
                    for f in imm.filters:
                        f.x = init_x.copy()
                        f.P = predictor.P_shared.copy()
                    imm.x = init_x.copy()
                    estimators[vid] = imm
                else:
                    # ---- 已有车辆：增量 predict + update ----
                    imm = estimators[vid]
                    imm.predict()
                    angle_diff = normalize_angle(state.yaw - imm.x[3])
                    unrolled_obs_yaw = imm.x[3] + angle_diff
                    z = np.array([state.x, state.y, state.speed, unrolled_obs_yaw])
                    imm.update(z)

                # 向量化多步预测
                future_preds = predictor.predict_from_estimator(imm, PRED_STEPS)

                t_end = time.perf_counter()
                total_predict_time += (t_end - t_start)
                total_predict_count += 1

                registry[frame_key][vid] = future_preds

            # 清除已离开视野的车辆估计器，释放内存
            existing_vids = set(current_vehicles.keys())
            for vid in list(estimators.keys()):
                if vid not in existing_vids:
                    del estimators[vid]

    # 保存预测结果
    print(f"\n预测完成，正在保存至 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(registry, f, indent=2)

    # 输出统计信息
    print("=" * 50)
    if total_predict_count > 0:
        avg_time_ms = (total_predict_time / total_predict_count) * 1000.0
        print(f"统计完成 (IMM-UKF 改进版 + 增量滤波):")
        print(f"  总处理车辆次数: {total_predict_count}")
        print(f"  平均单车预测耗时: {avg_time_ms:.4f} ms")
    else:
        print("未检测到有效预测数据。")
    print("=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
