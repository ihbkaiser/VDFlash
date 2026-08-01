#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

#baseline
# CUDA_VISIBLE_DEVICES=0,1 python -m sparrow.evaluation.gen_baseline_answer_video \
#   --spec-model-path  /root/autodl-tmp/model/sparrow \
#   --base-model-path /root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct \
#   --bench-name /root/project/Sparrow/sparrow/output/videodetailcaption/baseline  \
#   --question-begin 0 \
#   --question-end 100 \
#   --model-id test_160_zhen \
#   --temperature 0.0 \
#   --data_path /root/autodl-tmp/datasets/VideoDetailCaption \
#   --task VideoDetailCaption \
#   --model_type qwen2_5_vl \
#   --frame_num 160 \
#   --data_num 3 \
#   >/root/project/Sparrow/sparrow/output/videodetailcaption/baseline/test_160_zhen.log 2>&1

#sparrow  temp = 0  
# CUDA_VISIBLE_DEVICES=0,1 python -m sparrow.evaluation.gen_spec_answer_video \
#   --spec-model-path /root/autodl-tmp/model/sparrow \
#   --base-model-path /root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct \
#   --bench-name /root/project/Sparrow/sparrow/output/videodetailcaption/sparrow  \
#   --question-begin 0 \
#   --question-end 100 \
#   --model-id test_zhen_160_new_27_1 \
#   --num-q 2\
#   --depth 3 \
#   --top-k 8 \
#   --total-token 30 \
#   --use_sparrow True \
#   --temperature 0.0 \
#   --data_path /root/autodl-tmp/datasets/VideoDetailCaption \
#   --task VideoDetailCaption \
#   --model_type qwen2_5_vl \
#   --frame_num 160 \
#   --data_num 3\
#   >/root/project/Sparrow/sparrow/output/videodetailcaption/sparrow/test_zhen_160_new_27_1.log 2>&1



#vispec bs= 16  temp= 0   
# CUDA_VISIBLE_DEVICES=0,1 python -m sparrow.evaluation.gen_spec_answer_video \
#   --spec-model-path /root/autodl-tmp/model/ViSpec-Qwen2.5-VL-7B-Instruct-bs16 \
#   --base-model-path /root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct \
#   --bench-name /root/project/Sparrow/sparrow/output/videodetailcaption/vispec  \
#   --question-begin 0 \
#   --question-end 100 \
#   --model-id test_zhen_160_state_16_1 \
#   --num-q 2\
#   --depth 3 \
#   --top-k 8 \
#   --total-token 30 \
#   --use_vispec True \
#   --temperature 0.0 \
#   --data_path /root/autodl-tmp/datasets/VideoDetailCaption \
#   --task VideoDetailCaption \
#   --model_type qwen2_5_vl \
#   --frame_num 160 \
#   --data_num 3 \
#   >/root/project/Sparrow/sparrow/output/videodetailcaption/vispec/test_zhen_160_state_16_1.log 2>&1



# vispec_orig. bs= 8  temp= 0   
# CUDA_VISIBLE_DEVICES=0,1 python -m sparrow.evaluation.gen_spec_answer_video \
#   --spec-model-path /root/autodl-tmp/model/ViSpec-Qwen2.5-VL-7B-Instruct \
#   --base-model-path /root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct \
#   --bench-name /root/project/Sparrow/sparrow/output/videodetailcaption/vispec_orig  \
#   --question-begin 0 \
#   --question-end 100 \
#   --model-id test_zhen_160_1 \
#   --num-q 2\
#   --depth 3 \
#   --top-k 8 \
#   --total-token 30 \
#   --use_vispec True \
#   --temperature 0.0 \
#   --data_path /root/autodl-tmp/datasets/VideoDetailCaption \
#   --task VideoDetailCaption \
#   --model_type qwen2_5_vl \
#   --frame_num 160 \
#   --data_num 3 \
#   >/root/project/Sparrow/sparrow/output/videodetailcaption/vispec_orig/test_zhen_160_1.log 2>&1








