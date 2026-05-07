import os
import subprocess
import re
import sys

# ================= 1. 基础配置 =================
CONFIG_PATH = "model/pointpillar_early_fusion_cs_clean_pro/config.yaml"
RESULT_FILE = "ap_results_summary.txt"

CMD = [
    "python", "opencood/tools/inference.py", 
    "--model_dir", "model/pointpillar_early_fusion_cs_clean_pro", 
    "--fusion_method", "early"
]

# ================= 2. 定义测试矩阵 =================
experiments = [
    {
        "method": "CV_50ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CV_predict/test_1_compensated_50ms",
        "overhead": 50
    },
    {
        "method": "CV_100ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CV_predict/test_1_compensated_100ms",
        "overhead": 100
    },
    {
        "method": "CV_150ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CV_predict/test_1_compensated_150ms",
        "overhead": 150
    },
    {
        "method": "CV_200ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CV_predict/test_1_compensated_200ms",
        "overhead": 200
    },
    {
        "method": "CV_250ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CV_predict/test_1_compensated_250ms",
        "overhead": 250
    },
    {
        "method": "CV_300ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CV_predict/test_1_compensated_300ms",
        "overhead": 300
    },
    {
        "method": "CTRV_50ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRV_predict/test_1_compensated_50ms",
        "overhead": 50
    },
    {
        "method": "CTRV_100ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRV_predict/test_1_compensated_100ms",
        "overhead": 100
    },
    {
        "method": "CTRV_150ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRV_predict/test_1_compensated_150ms",
        "overhead": 150
    },
    {
        "method": "CTRV_200ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRV_predict/test_1_compensated_200ms",
        "overhead": 200
    },
    {
        "method": "CTRV_250ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRV_predict/test_1_compensated_250ms",
        "overhead": 250
    },
    {
        "method": "CTRV_300ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRV_predict/test_1_compensated_300ms",
        "overhead": 300
    },
    {
        "method": "CTRA_50ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRA_predict/test_1_compensated_50ms",
        "overhead": 50
    },
    {
        "method": "CTRA_100ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRA_predict/test_1_compensated_100ms",
        "overhead": 100
    },
    {
        "method": "CTRA_150ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRA_predict/test_1_compensated_150ms",
        "overhead": 150
    },
    {
        "method": "CTRA_200ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRA_predict/test_1_compensated_200ms",
        "overhead": 200
    },
    {
        "method": "CTRA_250ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRA_predict/test_1_compensated_250ms",
        "overhead": 250
    },
    {
        "method": "CTRA_300ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/CTRA_predict/test_1_compensated_300ms",
        "overhead": 300
    },
    {
        "method": "IMM_50ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/IMM_predict/test_1_compensated_50ms",
        "overhead": 50
    },
    {
        "method": "IMM_100ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/IMM_predict/test_1_compensated_100ms",
        "overhead": 100
    },
    {
        "method": "IMM_150ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/IMM_predict/test_1_compensated_150ms",
        "overhead": 150
    },
    {
        "method": "IMM_200ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/IMM_predict/test_1_compensated_200ms",
        "overhead": 200
    },
    {
        "method": "IMM_250ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/IMM_predict/test_1_compensated_250ms",
        "overhead": 250
    },
    {
        "method": "IMM_300ms",
        "val_dir": "pcPredict_data/cross_fullcar_clean/IMM_predict/test_1_compensated_300ms",
        "overhead": 300
    }
]

# ================= 3. 核心修改函数 =================
def modify_config_safe(config_path, new_val_dir, new_overhead):
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(config_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.startswith('validate_dir:'):
                f.write(f'validate_dir: {new_val_dir}\n')
            elif line.startswith('  async_overhead:'):
                f.write(f'  async_overhead: {new_overhead}\n')
            else:
                f.write(line)

# ================= 4. 主执行流程 =================
def main():
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write("========== 自动化评估结果汇总 ==========\n\n")
    
    ap_pattern = re.compile(r"The Average Precision at IOU 0\.3 is.*")
    total_exp = len(experiments)

    for i, exp in enumerate(experiments, 1):
        exp_name = exp["method"]
        val_dir = exp["val_dir"]
        overhead = exp["overhead"]
        
        # 1. 打印当前进度概览
        print(f"\n[{i}/{total_exp}] 正在准备: {exp_name} (时延: {overhead}ms)")
        modify_config_safe(CONFIG_PATH, val_dir, overhead)
        
        print(f"[{i}/{total_exp}] 正在运行推理，请稍候 ", end="", flush=True)
        
        ap_str = "未在输出中提取到 AP 结果。"
        
        try:
            # 2. 启动子进程，拦截输出
            process = subprocess.Popen(
                CMD, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1
            )
            
            line_count = 0
            # 3. 逐行静默读取，不打印原始日志
            for line in process.stdout:
                line_count += 1
                
                # 每读取到底层 20 行日志，在终端打印一个点号代表“程序活着并在处理”
                if line_count % 20 == 0:
                    print(".", end="", flush=True)
                
                # 边读边匹配结果
                match = ap_pattern.search(line)
                if match:
                    ap_str = match.group(0)
            
            process.wait()
            
            # 4. 运行结束，打印提取到的结果
            if process.returncode == 0:
                print(f"\n[{i}/{total_exp}] 提取成功: {ap_str}")
            else:
                print(f"\n[{i}/{total_exp}] 运行异常！状态码: {process.returncode}")
                ap_str = "运行失败，发生异常！"
                
        except Exception as e:
            print(f"\n[{i}/{total_exp}] 脚本执行报错: {e}")
            ap_str = "运行失败，发生致命错误！"
        
        # 写入汇总文件
        with open(RESULT_FILE, 'a', encoding='utf-8') as f:
            f.write(f"【测试组: {exp_name}】\n")
            f.write(f"验证集目录: {val_dir}\n")
            f.write(f"设置时延: {overhead} ms\n")
            f.write(f"检测结果: {ap_str}\n")
            f.write("-" * 50 + "\n")

    print(f"\n所有评估运行完毕！对比数据已保存在: {RESULT_FILE}")

if __name__ == "__main__":
    main()