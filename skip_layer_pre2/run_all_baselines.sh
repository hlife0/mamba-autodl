#!/bin/bash
# 顺序运行所有 baseline 实验，避免多 GPU 冲突

echo "开始运行 baseline 实验..."
echo "================================"

# 设置环境
export CUDA_HOME=/home/hlife/Mamba-experiment/.local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 激活 conda 环境
source /home/hlife/Mamba-experiment/setup_cuda_env.sh

# 运行实验（顺序执行，避免冲突）
echo ""
echo "1/4 运行 baseline_empty.py (cuda:0)..."
python skip_layer_pre2/baseline_empty.py --device="cuda:0"

echo ""
echo "2/4 运行 baseline_doc1.py (cuda:0)..."
python skip_layer_pre2/baseline_doc1.py --device="cuda:0"

echo ""
echo "3/4 运行 baseline_doc2.py (cuda:0)..."
python skip_layer_pre2/baseline_doc2.py --device="cuda:0"

echo ""
echo "4/4 运行 baseline_full.py (cuda:0)..."
python skip_layer_pre2/baseline_full.py --device="cuda:0"

echo ""
echo "================================"
echo "所有实验完成！"
