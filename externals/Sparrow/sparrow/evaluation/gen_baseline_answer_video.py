"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""

import argparse
import json
import os
import av
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
import time

import shortuuid
from datasets import Dataset, load_dataset
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration

from ..model.choices import *
from ..model.kv_cache import initialize_past_key_values
from ..model.spec_model_sparrow import SpecModel
from ..model.utils import *

from .video_prompt import clip_input_video



def load_data(task, data_num, data_path):
    if task == "VideoDetailCaption":
       
        data_video = load_dataset(
                "/root/autodl-tmp/datasets/VideoDetailCaption",
                split="test",
                # cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
        
        def video_exists(example):
            video_path = os.path.join(video_dir, f"{example['video_name']}.mp4")
            return os.path.exists(video_path)

        video_dir = os.path.join(data_path, "Test_Videos")
        filtered_data = data_video.filter(video_exists)
        data_video = filtered_data
        # print("data_video",data_video)
    elif task == 'MVBench':
        data_video_1 = load_dataset(
                "/root/autodl-tmp/data/MVBench",
                'action_sequence',
                split="train",
                # cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))

        data_video_2 = load_dataset(
                "/root/autodl-tmp/data/MVBench",
                'action_prediction',
                split="train",
                # cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
        
        data_video = concatenate_datasets([data_video_1, data_video_2])
       
        data_video = data_video.shuffle(seed=42)
        

        def video_exists(example):
            video_path = os.path.join(video_dir, f"{example['video']}")
            return os.path.exists(video_path)
        
        video_dir = "/root/autodl-tmp/data/MVBench/video/star/Charades_v1_480"
        filtered_data = data_video.filter(video_exists)
        data_video = filtered_data
    elif task == 'MVLU':
        data_video = load_dataset(
                "",
                split="train",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
    elif task == 'LongVideoBench':
        data_video = load_dataset(
                "/root/autodl-fs/LongVideoBench",
                split="test",
                # cache_dir=cache_dir,
            ).shuffle(seed=24).select(range(data_num))
        
        def video_exists(example):
            video_path = os.path.join(video_dir, f"{example['video_path']}")
            if not os.path.exists(video_path):
                return False
           
            try:
                container = av.open(video_path)
                total_frames = container.streams.video[0].frames
                container.close()
                print("video_path",video_path)
                print("total_frames",total_frames)
                return total_frames > 300 and total_frames <= 1200
            except:
                return False

        video_dir = "/root/autodl-fs/LongVideoBench/videos"
        filtered_data = data_video.filter(video_exists)
        target_valid_num = 100
        data_video = filtered_data.select(range(target_valid_num))
       
        valid_sample_num = len(data_video)
       
    elif task == 'MMBench':
        data_video = load_dataset(
                "",
                split="train",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
    elif task == 'COCO_caption':
        cache_dir = ''
        os.makedirs(cache_dir, exist_ok=True)
        data_video = load_dataset(
                "",
                split="test",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(100))
    else:
        data_video = None

    # print(data_video)
    return data_video



def baseline_forward(
    input_ids,
    model,
    tokenizer,
    tree_choices,
    logits_processor=None,
    max_steps=2048,
    idx_layer = None,
    numbuer = None,
    **kwargs,
):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    # model.spec_layer.reset_kv()   modified

    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        try:
            tree_buffers = generate_tree_buffers(
                tree_choices,
                device=model.base_model.model.layers[-1].self_attn.q_proj.weight.device,
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                model.base_model.lm_head.weight.device
            )
        except:
            tree_buffers = generate_tree_buffers(
                tree_choices,
                device=model.base_model.language_model.model.layers[
                    -1
                ].self_attn.q_proj.weight.device,
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                model.base_model.language_model.lm_head.weight.device
            )
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        try:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model)
        except:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model.language_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    ################################
    inputs_embeds = None
    embed_weights = None
    special_image_mask = None
    if (
        model.base_model.config.architectures[0]
        == "LlavaOnevisionForConditionalGeneration"
    ):
        vision_feature_layer = kwargs.get("vision_feature_layer")
        vision_feature_select_strategy = kwargs.get(
            "vision_feature_select_strategy"
        )
        vision_aspect_ratio = kwargs.get(
            "vision_aspect_ratio"
        )
        pixel_values = kwargs.get("pixel_values")
        image_sizes = kwargs.get("image_sizes")
        pixel_values_videos = kwargs.get("pixel_values_videos")

        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else model.base_model.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else model.base_model.config.vision_feature_select_strategy
        )
        vision_aspect_ratio = (
            vision_aspect_ratio if vision_aspect_ratio is not None else model.base_model.config.vision_aspect_ratio
        )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = model.base_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = model.base_model.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_features, feature_lens = model.base_model.pack_image_features(
                image_features,
                image_sizes,
                image_newline=model.base_model.image_newline,
                vision_aspect_ratio=vision_aspect_ratio,
            )
            
            #MODIFIED
            #Delete image feature
            selected_indices = None 
            if selected_indices is not None:
                selected_indices = selected_indices.to(image_features.device)
                image_features = image_features[selected_indices]
                    

            n_image_tokens = (input_ids == model.base_model.config.image_token_index).sum().item()
            n_image_features = image_features.shape[0]

            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (
                (input_ids == model.base_model.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Video are simply embedded and further pooled to decrease seq len
        if pixel_values_videos is not None:
            video_features = model.base_model.get_video_features(
                pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            ) #[bs, n_frame*196,dim]
            image_newline = (
                model.base_model.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1).to(video_features.device)
            )
            video_features = torch.cat((video_features, image_newline), dim=1)
            video_features = video_features.flatten(0, 1) #[n_frame*196+1, dim]

            #Delete video feature
            selected_indices = None 
            if selected_indices is not None:
                selected_indices = selected_indices.to(video_features.device)
                video_features = video_features[selected_indices]

            n_video_tokens = (input_ids == model.base_model.config.video_token_index).sum().item()
            n_video_features = video_features.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            special_video_mask = (
                (input_ids == model.base_model.config.video_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)


    elif (
        model.base_model.config.architectures[0]
        == "Qwen2_5_VLForConditionalGeneration"
    ):
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        pixel_values_videos = kwargs.get("pixel_values_videos")
        video_grid_thw = kwargs.get("video_grid_thw")
        second_per_grid_ts = kwargs.get("second_per_grid_ts")

        if inputs_embeds is None:
            inputs_embeds = model.base_model.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(model.base_model.visual.dtype)
                image_embeds = model.base_model.visual(
                    pixel_values, grid_thw=image_grid_thw
                )
                n_image_tokens = (
                    (input_ids == model.base_model.config.image_token_id)
                    .sum()
                    .item()
                )
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == model.base_model.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask, image_embeds
                )

                special_image_mask = mask

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(
                    model.base_model.visual.dtype
                )
                video_embeds = model.base_model.visual(
                    pixel_values_videos, grid_thw=video_grid_thw
                )
                n_video_tokens = (
                    (input_ids == model.base_model.config.video_token_id)
                    .sum()
                    .item()
                )
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == model.base_model.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    video_mask, video_embeds
                )

                # special_video_mask = mask
                special_image_mask = mask
    ################################

    input_len = input_ids.shape[1]
    reset_tree_mode(model)

    outputs = model.base_model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        # **kwargs,
    )

    new_token = 0

    torch.cuda.synchronize()
    start_time = time.time()

    for idx in range(max_steps):
        if logits_processor is not None:
            logits = outputs.logits[:, -1]
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            input_id = torch.multinomial(probabilities, 1)
        else:
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = model.base_model(
            input_id, use_cache=True, past_key_values=past_key_values
        )
        input_ids = torch.cat([input_ids, input_id], dim=-1)

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if hasattr(tokenizer, 'eod_id') and tokenizer.eod_id in input_ids[0, input_len:].tolist():
            break
        # if new_token > 1024:
        #     break
        if new_token > 512:
            break
        
        new_token += 1
        # if input_ids.shape[1] > 1960:
        #     break
    torch.cuda.synchronize()
    end_time = time.time()

    return input_ids, new_token, idx, end_time - start_time 


def run_eval(
    base_model_path,
    spec_model_path,
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
    tree_choices,
):
    # data = load_data(args)
    data = load_data(args.task, args.data_num, args.data_path)

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
                model_id,
                data.select(range(i, i + chunk_size)),
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                tree_choices,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    base_model_path,
    spec_model_path,
    model_id,
    data,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    tree_choices,
):
    # temperature = 0.0

    model = SpecModel.from_pretrained(
        base_model_path=base_model_path,
        spec_model_path=spec_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto",
        attn_implementation = "sdpa",
    )

    print("Model devices:")
    for name, param in model.named_parameters():
        if param.device not in [torch.device("cpu"), torch.device("meta")]:
            print(f"{name}: {param.device}")

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print("Check model training state:", model.training)

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print("CUDA VISIBLE DEVICES:", cuda_visible_devices)

    # # warmup
    for _ in range(3):
        torch.manual_seed(0)

        turns = []
        idxs = []
        new_tokens = []
        wall_time = []

        # model_inputs = build_prompt(data[0], args)
        data_instance = data[0]
        model_inputs = clip_input_video(base_model_path, args.task, data_instance,frame_num = args.frame_num, model_type = args.model_type, data_path=args.data_path)
        

        torch.cuda.synchronize()
        start_time = time.time()

        output_ids, new_token, idx, _ = baseline_forward(
            **model_inputs,
            model=model,
            tokenizer=tokenizer,
            tree_choices=tree_choices,
            logits_processor=logits_processor,
        )

        torch.cuda.synchronize()
        total_time = time.time() - start_time
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


    croeet = 0 
    numbuer = 0
   
    for d in tqdm(data):
    
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            decode_time = []

            # model_inputs = build_prompt(d, args)
            print("d",d)
            data_instance = d 
            model_inputs = clip_input_video(base_model_path, args.task, data_instance,frame_num = args.frame_num, model_type = args.model_type, data_path=args.data_path)
            

            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, dec_time = baseline_forward(
                **model_inputs,
                model=model,
                tokenizer=tokenizer,
                tree_choices=tree_choices,
                logits_processor=logits_processor,
                numbuer=numbuer,
            )


            torch.cuda.synchronize()
            total_time = time.time() - start_time
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
            decode_time.append(dec_time)

            choices.append(
                {
                    "index": i,
                    "turns": turns,
                    "idxs": idxs,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                    "decode_time": decode_time,
                }
            )

        # Dump answers
        video_name = None
        if args.task == "VideoDetailCaption":
            video_name = d["video_name"]
            
        elif args.task == "MVBench":
            video_name = d["video"]
           
        elif args.task == 'LongVideoBench':
            video_name = d["id"]
           

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": video_name,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

     

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
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--model-id", type=str, default="sqa-llava-v1.6-vicuna-7b-fp16-baseline"
    )
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
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
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

    parser.add_argument("--data-folder", type=str, default="data/MME")

    ##################
    parser.add_argument('--data_path', type=str,
                        default='/data',
                        help='Path to the data directory')
    # Evaluation parameters
    parser.add_argument('--task', type=str, default='VideoDetailCaption',
                        choices=['VideoDetailCaption', 'MVBench', 'MVLU', 'LongVideoBench', 'MMBench'],
                        help='Evaluation task type')
    parser.add_argument('--frame_num', type=int, default=8,
                        help='Number of frames per video')
    parser.add_argument('--evaluation_num', type=int, default=1,
                        help='Number of evaluation samples')
    parser.add_argument('--data_num', type=int, default=100,
                        help='Number of data samples to load')
    parser.add_argument('--model_type', type=str, 
                        help='Number of data samples to load')
    
    parser.add_argument('--idx_layer', type=int, 
                        help='Number of layer to jianzhi')
    args = parser.parse_args()

    args.model = args.base_model_path
    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    args.tree_choices = eval(args.tree_choices)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.spec_model_path,
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
        args.tree_choices,
    )

    reorg_answer_file(answer_file)
