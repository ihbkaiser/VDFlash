# Findings — DDTree video cut-sweep

- The repository's DDTree paper target is block-diffusion DFlash, not GLANCE.
  Official DDTree constructs a tree from one block-drafter distribution pass
  and verifies the tree with an ancestor-only attention mask. The official
  implementation is for causal language models; this experiment must adapt the
  verification path to Qwen2.5-VL's 3-axis M-RoPE and video-prefill cache.
- Existing `scripts/run_dflash_qwen25vl3b_eval.py` uses a single linear proposal
  chain. It already defines four draft-context retention settings (100/50/25/0)
  but does not implement tree construction or tree verification.
- Historical raw DFlash files exist for 100 VidHalluc and 100 VideoChair
  records at all four retention settings. They are useful context only; they
  are not DDTree results and cannot substitute for this run.
- Available local records and video files: VideoDetailCaption 50/50 (about
  1.21 GB), VidHalluc 100/100 (about 490 MB), VideoChair 100/100 (about 66 MB).
- The local DFlash checkpoint exists (`training_state.pt`, about 3.53 GB), the
  Qwen2.5-VL target directory exists, and the draft config has depth 3 and
  block size 16.
- Workspace `.venv` contains PyTorch 2.1.2+cu121 and Transformers 4.52.4.
  CUDA is unavailable inside the sandbox, but an approved read-only hardware
  check outside it sees an RTX 3090 and an RTX A4000.
- Retention means masking visual hidden-state rows only for the drafter. The
  target continues to receive the full video, so any verified output change
  would indicate a correctness failure rather than an intended quality tradeoff.
- Historical VidHalluc/VideoChair summary script computes a word-overlap proxy
  it labels `CHAIR_i`; this is not an official CHAIR implementation. The new
  report must call it a lexical object-overlap proxy if reused.

## Sweep results

- The analysis uses local subsets (VDC 50, VidHalluc 100, VideoCHAIR 100), not
  the full benchmark releases. Coverage is VDC 400/400 and VidHalluc 800/800;
  VideoCHAIR has 799/800 speculative conditions because the final DDTree 0%
  row stalled during an RTX A4000/NVML fault. The single affected cell is
  provisional; all other condition cells have full coverage.
- Averaged across cuts, DDTree's end-to-end speedup against paired target AR is
  about 1.65× on VDC, 1.08× on VidHalluc and 2.02× on VideoCHAIR, compared with
  about 1.46×, 1.05× and 1.83× for DFlash. Paired-bootstrap intervals for the
  DDTree-minus-DFlash speedup contrast exclude zero at every cut in this local
  sample, but speedup is not quality-adjusted.
- DDTree also has higher tau-effective at all cuts. Token-exactness remains a
  separate guardrail and is far from universal on captioning. VidHalluc's
  BinaryQA/MCQ/Scene_Transition/Sequence_Ranking accuracy is reported
  separately; after correcting the natural-language Sequence_Ranking parser,
  AR and all speculative settings score 16/20 (80%) on that task and overall
  VidHalluc accuracy is 77–78%.
- Recomputed VidHalluc task annotations do not modify generated text, token
  IDs, hashes, timing or acceptance traces. The VideoCHAIR reference-vocabulary
  CHAIR-like figures remain lexical proxies, not official visual object
  hallucination scores.
