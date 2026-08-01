# Sparrow: Text-Anchored Window Attention with Visual-Semantic Glimpsing for Speculative Decoding in Video LLMs



## Requirements
You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Weights

| Base Model                                                                                    | Sparrow on Hugging Face                                                                                  |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)             | [Sparrow-Qwen2.5-VL-7B-Instruct](https://huggingface.co/fourss/Sparrow/tree/main/Sparrow-Qwen2.5-VL-7B-Instruct)     |
| [llava-hf/llava-onevision-7b-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-chat-hf)                   | [Sparrow-llava-onevision-7b-hf](https://huggingface.co/fourss/Sparrow/tree/main/Sparrow-llava-onevision-7b-hf)                   |


## Usage

We provide several pre-trained model checkpoints on Hugging Face (see the **Weights** section above). If you wish to use these, you can download them and skip the data generation and training sections, proceeding directly to **Evaluation**.



### Evaluation

Evaluate the inference speed of the model using both standard autoregressive decoding (baseline) and speculative decoding.


#### Baseline Speed Evaluation

```bash
python -m sparrow.evaluation.gen_baseline_answer_video \
  --base-model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --spec-model-path <path_to_your_model_directory> \
  --model-id test \
  --bench-name <path_to_baseline_results_folder> \
  --temperature=<value> \
  --data_path lmms-lab/VideoDetailCaption \
  --task VideoDetailCaption \
  --model_type qwen2_5_vl \
  --frame_num 160 \
  --data_num 3 \

```



**Parameters**:

  - `--bench-name`: The output directory for evaluation results.
  - `--spec-model-path`: Path to the directory containing the Sparrow model checkpoint. This can be a model directory downloaded from Hugging Face.
  - `--temperature`: Sampling temperature (e.g., `0.0` for greedy, `1.0` for stochastic).
  - `--data_path` : Path to the evaluation dataset.
  - `--task` : Type of evaluation task.
  - `frame_num` :  Number of frames extracted per video, balances model input information and memory usage.

#### Speculative Decoding Speed Evaluation

```bash
python -m sparrow.evaluation.gen_spec_answer_video \
  --base-model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --spec-model-path=<path_to_your_model_directory> \
  --model-id test \
  --bench-name <path_to_spec_results_folder> \
  --depth <value> \
  --top-k <value> \
  --total-token <value> \
  --use_sparrow True \
  --temperature <value>
  --data_path lmms-lab/VideoDetailCaption \
  --task VideoDetailCaption \
  --model_type qwen2_5_vl \
  --frame_num 160 \
  --data_num 3 \
```


**Specific Parameters**:

  - `--depth`: The depth of draft token tree.
  - `--top-k`: The width of draft token tree.
  - `--total-token`: Number of draft tokens selected from the draft tree to be verified by target model.






## Acknowledgements

We would like to acknowledge the foundational work of previous projects that inspired our approach, especially [EAGLE](https://github.com/SafeAILab/EAGLE), [ViSpec](https://github.com/KangJialiang/ViSpec), [MSD](https://github.com/Lyn-Lucy/MSD) and [SpecVLM](https://github.com/zju-jiyicheng/SpecVLM). 