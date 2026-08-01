"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""

import argparse
import json
import os
from transformers import AutoConfig, AutoTokenizer
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
import time

import shortuuid
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration

from ..model.kv_cache import initialize_past_key_values
from ..model.utils import *
from .mme_prompt import build_prompt
from accelerate import Accelerator
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name())
print("bfloat16 supported:", torch.cuda.is_bf16_supported())


def load_data(args):
    data = []
    with open(
        os.path.join(args.data_folder, "llava_mme.jsonl"), "r", encoding="utf-8"
    ) as f:
        lines = f.readlines()
        for l in lines:
            d = json.loads(l.strip())
            d["image"] = Image.open(
                open(
                    os.path.join(
                        args.data_folder,
                        "MME_Benchmark_release_version/MME_Benchmark",
                        d["image"],
                    ),
                    "rb",
                )
            )
            d["text"] = d["text"].partition("\n")[0]
            data.append(d)

    return Dataset.from_list(data).shuffle(seed=42).select(range(0, 100))
# def load_data(args):
#     data = []
#     with open(
#         os.path.join(args.data_folder, "llava_mme.jsonl"), "r", encoding="utf-8"
#     ) as f:
#         for line in f:  # 更省内存，避免 readlines()
#             d = json.loads(line.strip())
#             image_path = os.path.join(
#                 args.data_folder,
#                 "MME_Benchmark_release_version",
#                 d["image"]
#             )
#             d["image"] = Image.open(image_path).convert("RGB")
#             d["text"] = d["text"].partition("\n")[0]
#             data.append(d)

#     return Dataset.from_list(data).shuffle(seed=42).select(range(100))

# def load_data(args):
#     data = []
#     jsonl_path = os.path.join(args.data_folder, "llava_mme.jsonl")

#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     for l in lines:
#         d = json.loads(l.strip())
        
#         image_path = os.path.join(
#             args.data_folder,
#             "MME_Benchmark_release_version",
#             d["image"]
#         )
        
#         # ✅ 方法一：直接用路径打开（最简单）
#         # d["image"] = Image.open(image_path).convert("RGB")  # 建议 convert("RGB")

#         # 可选：增加异常处理防止某张图损坏导致整个程序崩溃
#         try:
#             d["image"] = Image.open(image_path).convert("RGB")
#         except Exception as e:
#             print(f"Error loading image {image_path}: {e}")
#             continue

#         d["text"] = d["text"].partition("\n")[0]
#         data.append(d)

#     return Dataset.from_list(data).shuffle(seed=42).select(range(0, 100))

# def load_data(args):
#     data = []
#     with open(
#         os.path.join(args.data_folder, "llava_mme.jsonl"), "r", encoding="utf-8"
#     ) as f:
#         lines = f.readlines()
#         for l in lines:
#             d = json.loads(l.strip())
#             d["image"] = Image.open(
#                 open(
#                     os.path.join(
#                         args.data_folder,
#                         "MME_Benchmark_release_version/MME_Benchmark",
#                         d["image"],
#                     ),
#                     "rb",
#                 )
#             )
#             d["text"] = d["text"].partition("\n")[0]
#             data.append(d)

#     return Dataset.from_list(data).shuffle(seed=42).select(range(0, 100))


def run_eval(
    base_model_path,
    spec_model_path,
    mlp_model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    args,
):

    data = load_data(args)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(data) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(data), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                spec_model_path,
                mlp_model_path,
                model_id,
                data.select(range(i, i + chunk_size)),
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    base_model_path,
    spec_model_path,
    mlp_model_path,
    model_id,
    data,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    args,
):
    # temperature = 0.0
    accelerator = Accelerator()
        # 总进程数量（所有GPU/节点的进程总数）
    total_processes = accelerator.num_processes
    print(f"总进程数（总rank数）: {total_processes}")



    # 本地rank（当前机器上的标识，范围：0 ~ local_processes-1）
    local_rank = accelerator.local_process_index
    print(f"当前进程本地rank: {local_rank}")

    # 是否为主进程（通常rank=0）
    is_main_process = accelerator.is_main_process
    print(f"是否为主进程: {is_main_process}")
    if args.use_ours:
        from ..model.spec_model_ours import SpecModel

        model = SpecModel.from_pretrained(
            base_model_path=base_model_path,
            spec_model_path=spec_model_path,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            # load_in_8bit=True,
            device_map="auto",
            num_q=args.num_q,
        )


        for name, param in model.named_parameters():
            print(f"{name}  {param.shape}  {param.device}")
    elif args.use_medusa:
        from ..model.spec_model_medusa import SpecModel

        model = SpecModel.from_pretrained(
            base_model_path=base_model_path,
            spec_model_path=spec_model_path,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            # load_in_8bit=True,
            device_map="auto",
        )
    elif args.use_msd:
        print("msdmsdmsdmsd")
        from ..model.spec_model_msd import SpecModel

        model = SpecModel.from_pretrained(
            base_model_path=base_model_path,
            spec_model_path=spec_model_path,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            # load_in_8bit=True,
            device_map="auto",
        )
    elif args.use_msd_yibu:
        print("msdmsdmsdmsd_yibu")
        from ..model.spec_model_msd_yibu import SpecModel

        model = SpecModel.from_pretrained(
            base_model_path=base_model_path,
            spec_model_path=spec_model_path,
            mlp_model_path= mlp_model_path,
            accelerator = accelerator,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            # load_in_8bit=True,
            device_map="cuda:0",
        )
    else:
        from ..model.spec_model import SpecModel

        model = SpecModel.from_pretrained(
            base_model_path=base_model_path,
            spec_model_path=spec_model_path,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            # load_in_8bit=True,
            device_map="auto",
        )
    if accelerator.local_process_index == 0:
        for name, param in model.named_parameters():
            print(f"{name}  {param.shape}  {param.device}")
    if accelerator.local_process_index == 1:
        for name, param in model.named_parameters():
            print(f"{name}  {param.shape}  {param.device}")
    if accelerator.local_process_index == 0:
        # tokenizer = AutoTokenizer.from_pretrained(
        #         base_model_path, use_fast=False
        #     )
        tokenizer = model.tokenizer
    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print("Check model training state:", model.training)

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print("CUDA VISIBLE DEVICES:", cuda_visible_devices)

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        turns = []
        idxs = []
        new_tokens = []
        wall_time = []

        model_inputs = build_prompt(data[0], args)
        model_inputs = model_inputs.to(accelerator.device)
        torch.cuda.synchronize()
        start_time = time.time()

        output_ids, new_token, idx = model.specgenerate(
            **model_inputs, temperature=temperature, log=True
        )
        # output_ids, new_token, idx, _ = model.baseline_forward(
        #     **model_inputs,
        #     logits_processor=logits_processor,
        # )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        if accelerator.local_process_index == 0:
            output_ids = output_ids[0][len(model_inputs["input_ids"][0]) :]

            output_ids[output_ids > tokenizer.vocab_size] = 0
            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
    print("Warmup done")

    for d in tqdm(data):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            acceptance_length = []

            # generate prompt
            model_inputs = build_prompt(d, args)
            model_inputs = model_inputs.to(accelerator.device)
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, accp_len = model.specgenerate(
                **model_inputs,
                temperature=temperature,
                log=True,
                return_acceptance_len=True,
            )
            # accp_len = [0]
            # output_ids, new_token, idx, _ = model.baseline_forward(
            # **model_inputs,
            # logits_processor=logits_processor,
            # )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(model_inputs["input_ids"][0]) :]
            if accelerator.local_process_index == 0:
                output_ids[output_ids > tokenizer.vocab_size] = 0
                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                acceptance_length.extend(accp_len)

                choices.append(
                    {
                        "index": i,
                        "turns": turns,
                        "idxs": idxs,
                        "new_tokens": new_tokens,
                        "wall_time": wall_time,
                        "acceptance_length": acceptance_length,
                    }
                )
        if accelerator.local_process_index == 0:
            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": d["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
    if accelerator.local_process_index == 0:
        reorg_answer_file(answer_file)


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec-model-path",
        type=str,
        default="down_checkpoints/LC70B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/home/lyh/weights/hf/llama2chat/70B/",
        help="1",
    )
    parser.add_argument(
        "--mlp-model-path",
        type=str,
        default="/home/lyh/weights/hf/llama2chat/70B/",
        help="1",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="sqa-llava-v1.6-vicuna-7b-fp16")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--answer-dir", type=str, help="The output answer directory.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=30,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    parser.add_argument("--image-fc", type=bool, default=False)
    parser.add_argument("--use-ours", type=bool, default=False)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--use-medusa", type=bool, default=False)
    parser.add_argument("--use-msd", type=bool, default=False)
    parser.add_argument("--use_msd_yibu", type=bool, default=False)

    parser.add_argument("--data-folder", type=str, default="/root/autodl-tmp/data/vispec_data/MME")

    args = parser.parse_args()

    args.model = args.base_model_path
    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    if args.answer_file:
        answer_file = args.answer_file
    elif args.answer_dir:
        answer_file = f"{args.answer_dir}/{args.model_id}.jsonl"
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.spec_model_path,
        args.mlp_model_path,
        args.model_id,
        args,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args,
    )
    
    
