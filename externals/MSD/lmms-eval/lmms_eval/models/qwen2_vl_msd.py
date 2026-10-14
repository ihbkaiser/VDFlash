import base64
import json
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import load_video_decord


from eagle.model.ea_model import EaModel

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_vl_msd")
class Qwen2_VL_MSD(lmms):
    """
    Qwen2_VL Model
    "https://github.com/QwenLM/Qwen2-VL"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        max_pixels: int = 12845056,
        min_pixels: int = 3136,
        max_num_frames: int = 32,
        msd_model: str = None,
        use_msd: bool = False,
        msd_visual_mode: str = "full",
        msd_chain_mode: bool = False,
        log_draft_stats: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        if msd_visual_mode not in {"full", "cut"}:
            raise ValueError("msd_visual_mode must be 'full' or 'cut'")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # if use_flash_attention_2:
        #     self._model = Qwen2VLForConditionalGeneration.from_pretrained(
        #         pretrained,
        #         torch_dtype="auto",
        #         device_map=self.device_map,
        #         attn_implementation="flash_attention_2",
        #     ).eval()
        # else:
        #     self._model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map).eval()
        self._model, _ = EaModel.from_pretrained(
            base_model_path=pretrained,
            ea_model_path=msd_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            total_token=-1
        )
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model.eval()
        self.model.base_model.tie_weights()
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.use_msd = use_msd
        self.msd_visual_mode = msd_visual_mode
        self.msd_chain_mode = msd_chain_mode
        self.log_draft_stats = log_draft_stats
        self._diagnostic_sample_count = 0
        self._stats_handle = None
        if self.log_draft_stats:
            stats_path = Path(self.log_draft_stats)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            self._stats_handle = stats_path.open("a", encoding="utf-8")

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _target_rope_ids(self, input_ids, image_grid_thw=None, video_grid_thw=None):
        return self.model.base_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=None,
        )[0]

    def _target_forward(self, inputs_embeds, position_ids, attention_mask=None):
        outputs = self.model.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )
        return self.model.base_model.lm_head(outputs[0])

    def _target_visual_necessity(
        self, prompt_ids, prompt_embeds, target_ids, image_grid_thw=None, video_grid_thw=None
    ):
        prompt_len = prompt_ids.shape[1]
        generated = target_ids[:, prompt_len:]
        if generated.shape[1] == 0:
            return []
        token_embeds = self.model.base_model.model.embed_tokens(generated)
        full_embeds = torch.cat((prompt_embeds, token_embeds), dim=1)
        position_ids = self._target_rope_ids(target_ids, image_grid_thw, video_grid_thw)
        full_logits = self._target_forward(full_embeds, position_ids)

        seq_len = full_embeds.shape[1]
        min_value = torch.finfo(full_embeds.dtype).min
        masked = torch.full(
            (1, 1, seq_len, seq_len),
            min_value,
            dtype=full_embeds.dtype,
            device=full_embeds.device,
        )
        masked[0, 0] = torch.tril(torch.zeros((seq_len, seq_len), device=full_embeds.device))
        visual_positions = torch.nonzero(
            (prompt_ids[0] == 151655) | (prompt_ids[0] == 151656),
            as_tuple=False,
        ).flatten()
        if visual_positions.numel() > 0:
            masked[0, 0, prompt_len:, visual_positions] = min_value
        masked_logits = self._target_forward(
            full_embeds, position_ids, attention_mask=masked
        )

        first = max(prompt_len - 1, 0)
        last = seq_len - 1
        full_probs = torch.softmax(full_logits[:, first:last], dim=-1)
        masked_probs = torch.softmax(masked_logits[:, first:last], dim=-1)
        necessity = 0.5 * torch.abs(full_probs - masked_probs).sum(dim=-1)
        return [float(value) for value in necessity[0].detach().cpu()]

    def _write_draft_stats(self, rows):
        if self._stats_handle is None:
            return
        for row in rows:
            self._stats_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._stats_handle.flush()

    @torch.no_grad()
    def _collect_teacher_forced_stats(
        self,
        prompt_ids,
        prompt_embeds,
        target_ids,
        task,
        sample_id,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        prompt_len = prompt_ids.shape[1]
        target_necessity = self._target_visual_necessity(
            prompt_ids,
            prompt_embeds,
            target_ids,
            image_grid_thw,
            video_grid_thw,
        )
        generated_len = target_ids.shape[1] - prompt_len
        rows = []
        for generated_count in range(1, generated_len + 1):
            prefix_ids = target_ids[:, : prompt_len + generated_count]
            generated_prefix = target_ids[:, prompt_len : prompt_len + generated_count]
            prefix_embeds = torch.cat(
                (
                    prompt_embeds,
                    self.model.base_model.model.embed_tokens(generated_prefix),
                ),
                dim=1,
            )
            prefix_position_ids = self._target_rope_ids(
                prefix_ids, image_grid_thw, video_grid_thw
            )
            prefix_outputs = self.model.base_model.model(
                inputs_embeds=prefix_embeds,
                position_ids=prefix_position_ids,
                use_cache=False,
                output_attentions=False,
            )
            prefix_hidden = prefix_outputs[0]

            mode_data = {}
            for mode in ("full", "cut"):
                draft_tokens, draft_stats = self.model.diagnostic_chain(
                    prefix_ids, prefix_embeds, prefix_hidden, mode
                )
                draft_ids = draft_tokens[:, 1:]
                candidate_embeds = torch.cat(
                    (
                        prefix_embeds,
                        self.model.base_model.model.embed_tokens(draft_ids),
                    ),
                    dim=1,
                )
                candidate_ids = torch.cat((prefix_ids, draft_ids), dim=1)
                candidate_position_ids = self._target_rope_ids(
                    candidate_ids, image_grid_thw, video_grid_thw
                )
                target_logits = self._target_forward(
                    candidate_embeds, candidate_position_ids
                )
                first = prefix_embeds.shape[1] - 1
                target_logits = target_logits[:, first : first + len(draft_stats)]
                target_probs = torch.softmax(target_logits.float(), dim=-1)[0]
                records = []
                for depth, (draft_id, stat, target_prob) in enumerate(
                    zip(draft_ids[0], draft_stats, target_probs), start=1
                ):
                    q = stat["_draft_log_probs"].float().exp()
                    p = target_prob.float()
                    target_token = int(torch.argmax(p).item())
                    records.append(
                        {
                            "draft_token": int(draft_id.item()),
                            "draft_prob": float(stat["draft_prob"]),
                            "draft_margin": float(stat["draft_margin"]),
                            "draft_entropy": float(stat["draft_entropy"]),
                            "draft_attention_mass": stat["draft_attention_mass"],
                            "target_greedy_token": target_token,
                            "target_entropy": float(
                                -(p * torch.log(p.clamp_min(1e-12))).sum().item()
                            ),
                            "target_tv": float(0.5 * torch.abs(p - q).sum().item()),
                            "argmax_match": int(int(draft_id.item()) == target_token),
                        }
                    )
                mode_data[mode] = records

            for depth, (full, cut) in enumerate(
                zip(mode_data["full"], mode_data["cut"]), start=1
            ):
                token_string = self.tokenizer.decode(
                    [cut["draft_token"]], skip_special_tokens=False
                )
                necessity_index = min(generated_count, len(target_necessity)) - 1
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "task": str(task),
                        "t": generated_count,
                        "j": depth,
                        "token_string": token_string,
                        "token_string_full": self.tokenizer.decode(
                            [full["draft_token"]], skip_special_tokens=False
                        ),
                        "draft_top1_token_full": full["draft_token"],
                        "draft_top1_token_cut": cut["draft_token"],
                        "draft_prob_full": full["draft_prob"],
                        "draft_prob_cut": cut["draft_prob"],
                        "draft_margin_full": full["draft_margin"],
                        "draft_margin_cut": cut["draft_margin"],
                        "draft_entropy_full": full["draft_entropy"],
                        "draft_entropy_cut": cut["draft_entropy"],
                        "draft_attention_mass_full": full["draft_attention_mass"],
                        "target_greedy_token_full": full["target_greedy_token"],
                        "target_greedy_token_cut": cut["target_greedy_token"],
                        "target_entropy_full": full["target_entropy"],
                        "target_entropy_cut": cut["target_entropy"],
                        "target_tv_full": full["target_tv"],
                        "target_tv_cut": cut["target_tv"],
                        "argmax_match_full": full["argmax_match"],
                        "argmax_match_cut": cut["argmax_match"],
                        "necessity": target_necessity[necessity_index]
                        if target_necessity
                        else 0.0,
                    }
                )
        self._write_draft_stats(rows)
        return rows

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        
        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            messages = []
            processed_visuals = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        vr = decord.VideoReader(visual)
                        first_frame = vr[0].asnumpy()
                        height, width = first_frame.shape[:2]
                        # max_pixels = height * width
                        message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": self.max_pixels}, {"type": "text", "text": context}]})
                    elif isinstance(visual, Image.Image):  # Single image
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        image_content = []
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            from eagle.model.utils import get_input_embeds_qwen2vl
            input_embeds = get_input_embeds_qwen2vl(
                inputs.input_ids,
                getattr(inputs, "pixel_values", None),
                getattr(inputs, "image_grid_thw", None),
                self.model.base_model,
                getattr(inputs, "pixel_values_videos", None),
                getattr(inputs, "video_grid_thw", None),
            )
            # cont = self.model.generate(
            #     **inputs,
            #     eos_token_id=self.tokenizer.eos_token_id,
            #     pad_token_id=pad_token_id,
            #     do_sample=True if gen_kwargs["temperature"] > 0 else False,
            #     temperature=gen_kwargs["temperature"],
            #     top_p=gen_kwargs["top_p"],
            #     num_beams=gen_kwargs["num_beams"],
            #     max_new_tokens=gen_kwargs["max_new_tokens"],
            #     use_cache=self.use_cache,
            # )

            from eagle.model.utils import temp_cache

            def run_msd(mode):
                temp_cache.use_msd = True
                temp_cache.visual_mode = mode
                temp_cache.chain_mode = False
                return self.model.msdgenerate(
                    inputs.input_ids,
                    inputs_embeds=input_embeds.clone(),
                    temperature=0,
                    max_new_tokens=512,
                )

            if self.use_msd:
                target_ids = None
                if self.log_draft_stats or self._diagnostic_sample_count < 20:
                    target_ids = self.model.naivegenerate(
                        inputs.input_ids,
                        inputs_embeds=input_embeds.clone(),
                        temperature=0,
                        max_new_tokens=512,
                    )
                if self._diagnostic_sample_count < 20:
                    full_ids = run_msd("full")
                    cut_ids = run_msd("cut")
                    prompt_len = inputs.input_ids.shape[1]
                    decode_generated = lambda ids: self.tokenizer.decode(
                        ids[0, prompt_len:].tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    target_text = decode_generated(target_ids)
                    full_text = decode_generated(full_ids)
                    cut_text = decode_generated(cut_ids)
                    full_equal = full_text == target_text
                    cut_equal = cut_text == target_text
                    if not full_equal or not cut_equal:
                        token_debug = {}
                        for name, ids in (
                            ("target", target_ids),
                            ("full", full_ids),
                            ("cut", cut_ids),
                        ):
                            suffix = ids[0, prompt_len : prompt_len + 20].tolist()
                            token_debug[name] = {
                                "ids": suffix,
                                "pieces": [
                                    self.tokenizer.decode(
                                        [token_id],
                                        skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False,
                                    )
                                    for token_id in suffix
                                ],
                            }
                        raise AssertionError(
                            "MSD visual modes must match plain target greedy decoding: "
                            f"full_equal={full_equal}, cut_equal={cut_equal}, "
                            f"full_shape={tuple(full_ids.shape)}, cut_shape={tuple(cut_ids.shape)}, "
                            f"target_shape={tuple(target_ids.shape)}, "
                            f"target_text={target_text[:160]!r}, full_text={full_text[:160]!r}, "
                            f"cut_text={cut_text[:160]!r}, "
                            f"eos_token_id={self.tokenizer.eos_token_id}, "
                            f"eod_token_id={getattr(self.tokenizer, 'eod_id', None)}, "
                            f"token_debug={token_debug}"
                        )
                    self._diagnostic_sample_count += inputs.input_ids.shape[0]
                    output_ids = full_ids if self.msd_visual_mode == "full" else cut_ids
                else:
                    output_ids = run_msd(self.msd_visual_mode)

                if self.log_draft_stats:
                    stats_targets = target_ids if target_ids is not None else output_ids
                    for sample_index in range(inputs.input_ids.shape[0]):
                        self._collect_teacher_forced_stats(
                            inputs.input_ids[sample_index : sample_index + 1],
                            input_embeds[sample_index : sample_index + 1],
                            stats_targets[sample_index : sample_index + 1],
                            task,
                            f"{task}:{doc_id[sample_index]}",
                            image_grid_thw=getattr(inputs, "image_grid_thw", None),
                            video_grid_thw=getattr(inputs, "video_grid_thw", None),
                        )
            else:
                output_ids = self.model.naivegenerate(
                    inputs.input_ids,
                    inputs_embeds=input_embeds.clone(),
                    temperature=0,
                    max_new_tokens=512,
                )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]

            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # print(answers)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form

        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
