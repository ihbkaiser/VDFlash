#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'


################Llava-onevision

#baseline
# CUDA_VISIBLE_DEVICES=0,1 python -m sparrow.evaluation.gen_baseline_answer_video \
#   --spec-model-path /root/autodl-tmp/model/sparrow \
#   --base-model-path /root/autodl-tmp/model/llava-onevision-qwen2-7b-ov-chat-hf \
#   --bench-name /root/project/Sparrow/sparrow/output/videodetailcaption/llava/baseline  \
#   --question-begin 0 \
#   --question-end 100 \
#   --model-id test_160_zhen \
#   --temperature 0.0 \
#   --data_path /root/autodl-tmp/datasets/VideoDetailCaption \
#   --task VideoDetailCaption \
#   --model_type qwen2_5_vl \
#   --frame_num 32 \
#   --data_num 3 \
#   >/root/project/Sparrow/sparrow/output/videodetailcaption/llava/baseline/test_160_zhen.log 2>&1

#sparrow   temp = 0  
# CUDA_VISIBLE_DEVICES=0,1 python -m sparrow.evaluation.gen_spec_answer_video \
#   --spec-model-path /root/autodl-tmp/model/sparrow \
#   --base-model-path /root/autodl-tmp/model/llava-onevision-qwen2-7b-ov-chat-hf \
#   --bench-name /root/project/Sparrow/sparrow/output/videodetailcaption/llava/sparrow  \
#   --question-begin 0 \
#   --question-end 100 \
#   --model-id test_zhen_32_new_27_1 \
#   --num-q 2\
#   --depth 3 \
#   --top-k 8 \
#   --total-token 30 \
#   --use_sparrow True \
#   --temperature 0.0 \
#   --data_path /root/autodl-tmp/datasets/VideoDetailCaption \
#   --task VideoDetailCaption \
#   --model_type llava_ov \
#   --frame_num 32 \
#   --data_num 3\
#   >/root/project/Sparrow/sparrow/output/videodetailcaption/llava/sparrow/test_zhen_32_new_27_1.log 2>&1










