from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from .config import DFlashTrainConfig
from .target import Qwen25VLTargetAdapter, load_jsonl, validate_manifest_record


def _eos_token_ids(adapter: Qwen25VLTargetAdapter) -> set[int]:
    value = getattr(getattr(adapter.model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(adapter.processor.tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(token_id) for token_id in value}
    return {int(value)}


def prepare_responses(
    adapter: Qwen25VLTargetAdapter,
    records: list[dict],
    *,
    max_new_tokens: int,
    max_seq_length: int,
    output_path: str | Path,
) -> None:
    """Generate and persist exact target token IDs for a multimodal manifest."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    adapter.model.eval()
    eos_token_ids = _eos_token_ids(adapter)
    written = 0
    skipped_without_eos = 0
    skipped_length = 0
    with tmp_path.open("w") as writer:
        for index, record in enumerate(records):
            validate_manifest_record(record, require_target_response=False)
            messages = record["messages"]
            inputs = adapter.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **getattr(adapter.processor, "dflash_processor_kwargs", {}),
            )
            inputs = {
                key: value.to(adapter.device) if torch.is_tensor(value) else value
                for key, value in dict(inputs).items()
            }
            prompt_length = int(inputs["input_ids"].shape[1])
            if prompt_length >= max_seq_length:
                skipped_length += 1
                print(
                    f"[skip] {record.get('id', index)}: prompt has {prompt_length} tokens, "
                    f"leaving no response room under max_seq_length={max_seq_length}"
                )
                continue
            with torch.inference_mode():
                generated = adapter.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            response_ids = generated[0, prompt_length:].detach().cpu().tolist()
            if eos_token_ids and not any(token_id in eos_token_ids for token_id in response_ids):
                skipped_without_eos += 1
                print(
                    f"[skip] {record.get('id', index)}: target response reached max_new_tokens "
                    "without an EOS token"
                )
                continue
            if prompt_length + len(response_ids) > max_seq_length:
                skipped_length += 1
                print(
                    f"[skip] {record.get('id', index)}: clean sequence has "
                    f"{prompt_length + len(response_ids)} tokens, exceeding max_seq_length={max_seq_length}"
                )
                continue
            response_text = adapter.processor.tokenizer.decode(
                response_ids, skip_special_tokens=False
            )
            output = copy.deepcopy(record)
            output["target_response"] = {
                "token_ids": response_ids,
                "text": response_text,
                "generation": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "use_cache": True,
                    "eos_token_ids": sorted(eos_token_ids),
                },
            }
            output["provenance"] = adapter.target_provenance()
            writer.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{index + 1}/{len(records)}] {record.get('id', index)} -> {len(response_ids)} tokens")
    tmp_path.replace(output_path)
    print(
        f"[summary] prepared={written} skipped_without_eos={skipped_without_eos} "
        f"skipped_length={skipped_length}"
    )


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate DFlash target responses")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="JSONL manifest without target_response")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = DFlashTrainConfig.from_file(args.config)
    adapter = Qwen25VLTargetAdapter.from_pretrained(config)
    prepare_responses(
        adapter,
        load_jsonl(args.input),
        max_new_tokens=config.response_max_new_tokens,
        max_seq_length=config.max_seq_length,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
