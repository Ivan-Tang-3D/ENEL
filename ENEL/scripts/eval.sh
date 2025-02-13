
MODEL_NAME=
LOG_SUFFIX=
LOG_DIR="/ENEL/new_eval_logs"
LOG_EDIR="/ENEL/new_eval_logs"

export PYTHONPATH="/ENEL:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=1 python pointllm/eval/eval_objaverse.py --model_name $MODEL_NAME --task_type captioning --prompt_index 2 > $LOG_EDIR/try_obj_${LOG_SUFFIX}.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python pointllm/eval/eval_objaverse.py  --model_name $MODEL_NAME --task_type classification --prompt_index 0 > $LOG_EDIR/try_objcls_${LOG_SUFFIX}.log 2>&1 &

CLS_OBJAVERSE="${MODEL_NAME}/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json"
CAPTION_OBJAVERSE="${MODEL_NAME}/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json"
CLS_MODEL="${MODEL_NAME}/evaluation/ModelNet_classification_prompt0.json"



while [ ! -f "$CLS_OBJAVERSE" ]; do
    echo "等待 $CLS_OBJAVERSE 文件生成..."
    sleep 60  # 每60秒检查一次
done

echo "$CAPTION_OBJAVERSE 文件已生成，开始执行命令"
python pointllm/eval/evaluator.py --results_path $CLS_OBJAVERSE --model_type gpt-4-0613 --eval_type open-free-form-classification > $LOG_EDIR/try_objcls_caption_gpt_${LOG_SUFFIX}.log 2>&1 &

while [ ! -f "$CAPTION_OBJAVERSE" ]; do
    echo "等待 $CAPTION_OBJAVERSE 文件生成..."
    sleep 60  # 每60秒检查一次
done

echo "$CAPTION_OBJAVERSE 文件已生成，开始执行命令"
python pointllm/eval/evaluator.py --results_path $CAPTION_OBJAVERSE --model_type gpt-4-0613 --eval_type object-captioning > $LOG_EDIR/try_obj_caption_gpt_${LOG_SUFFIX}.log 2>&1 &


while [ ! -f "$CAPTION_OBJAVERSE" ]; do
    echo "等待 $CAPTION_OBJAVERSE 文件生成..."
    sleep 60  # 每60秒检查一次
done

echo "$CAPTION_OBJAVERSE 文件已生成，开始执行命令"
CUDA_VISIBLE_DEVICES=3 python pointllm/eval/traditional_evaluator.py --results_path $CAPTION_OBJAVERSE > $LOG_EDIR/try_obj_caption_trans_${LOG_SUFFIX}.log 2>&1 &

