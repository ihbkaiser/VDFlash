# LLaVA-only capture flag

## Goal

Allow the Qwen2.5-VL MSD depth-sweep launcher to capture only the LLaVA
multimodal feature cache without re-running or validating ShareGPT text
capture inputs.

## CLI contract

`train_qwen25vl_msd_depth_sweep.sh` accepts:

```text
--capture-target both|text|llava
```

The default is `both`, preserving existing launcher behavior. Any other value
is rejected before data or GPU work starts.

## Capture behavior

- `both` validates and captures ShareGPT first, then LLaVA.
- `text` validates and captures only ShareGPT.
- `llava` validates and captures only LLaVA.
- NVRTC configuration runs once whenever the capture phase has a selected
  target.
- Guards, directory creation, and subprocess calls apply only to the selected
  feature roots. In particular, `llava` does not require `SHAREGPT_JSONL` and
  does not inspect or modify `TEXT_FEATURE_ROOT`.

The flag controls only `--phase capture` and the capture portion of
`--phase all`. It does not change data preparation or MSD training semantics.
Training continues to require both text and visual feature caches.

## Verification

Launcher regression tests cover:

- the default still invokes text followed by visual capture;
- `--capture-target llava` succeeds without a ShareGPT manifest and invokes
  only the visual capture script;
- `--capture-target text` invokes only the text capture script;
- an invalid target is rejected before subprocess execution;
- Bash syntax remains valid.

