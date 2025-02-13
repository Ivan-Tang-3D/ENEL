#!/bin/bash

# 设置模型版本和标记
MODEL_VERSION=Pointllm-7B
TAG=general-v1.0
export PYTHONPATH="/ENEL:$PYTHONPATH"
# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=

# 日志目录
LOG_DIR="/ENEL/outputs/Train_stage2/"
mkdir -p "$LOG_DIR/eval_logs"  # 确保日志目录存在

# 日志文件路径
EVAL_MODEL_LOG="$LOG_DIR/eval_logs/eval_model_$MODEL_VERSION-$TAG.log"
EVAL_3DMMVET_LOG="$LOG_DIR/eval_logs/eval_3dmmvet_$MODEL_VERSION-$TAG.log"

# 运行第一个 Python 脚本并记录日志
echo "Running model_vqa.py..."
python pointllm/eval/model_vqa.py \
    --model-path $LOG_DIR \
    --model_name $LOG_DIR \
    --question-file /ENEL/data/question.jsonl \
    --point-folder /ENEL/data/points \
    --answers-file $LOG_DIR/evaluation/$MODEL_VERSION-$TAG.jsonl \
    --conv-mode vicuna_v1 \
    --num_beams 5 \
    > "$EVAL_MODEL_LOG" 2>&1

if [ $? -ne 0 ]; then
    echo "model_vqa.py failed. Check log at $EVAL_MODEL_LOG"
    exit 1
fi


echo "Running eval_3dmmvet.py..."
python pointllm/eval/eval_3dmmvet.py \
    --answers-file $LOG_DIR/evaluation/$MODEL_VERSION-$TAG.jsonl \
    --gt-file /ENEL/data/gt.jsonl \
    --output-file $LOG_DIR/evaluation/result-$MODEL_VERSION-$TAG.jsonl \
    --model gpt-4-0125-preview \
    --max_workers 16 \
    --times 5 \
    > "$EVAL_3DMMVET_LOG" 2>&1

if [ $? -ne 0 ]; then
    echo "eval_3dmmvet.py failed. Check log at $EVAL_3DMMVET_LOG"
    exit 1
fi

echo "Both scripts completed successfully!"
