# TODO — HY-WorldPlay AR-distill Memory Baseline

## Immediate TODO

- [ ] Confirm remote machine access method
- [ ] Confirm Python environment path on remote machine
- [ ] Confirm CUDA / GPU availability with `nvidia-smi`
- [ ] Confirm `MODEL_PATH`
- [ ] Confirm `AR_DISTILL_ACTION_MODEL_PATH`
- [ ] Choose one real `pose_path` for the first baseline run
- [ ] Choose one reference image and prompt for I2V smoke testing
- [ ] Set a writable output directory for traces and generated videos

## Offline Trace TODO

- [ ] Run `scripts/analyze_ar_context.py` on one real pose file
- [ ] Save `jsonl` output
- [ ] Save markdown summary output
- [ ] Check invariants with `--strict`
- [ ] Review whether selected frames match the intended teacher rule

## Inference Trace TODO

- [ ] Set `HY_WORLDPLAY_CONTEXT_TRACE_JSONL`
- [ ] Run AR-distill short sequence (`29` or `61` frames)
- [ ] Save inference context trace
- [ ] Compare inference trace against offline trace
- [ ] Record any mismatch by chunk id

## Alignment TODO

- [ ] Compare inference path and training dataset path selection logic
- [ ] Document shared assumptions: first chunk, recent frames, FoV retrieval, 4-latent history unit
- [ ] Document differences caused by training sampling strategy
- [ ] Decide whether teacher trace should follow inference or training behavior

## Distillation Prep TODO

- [ ] Freeze V1 teacher trace schema
- [ ] Add one sample trace artifact to handoff package
- [ ] Decide whether V2 trace should include FoV scores
- [ ] Decide whether V2 trace should include rollout targets
- [ ] Prepare chunk-by-chunk comparison format for future student selection

## Smoke Test Ladder

- [ ] Short run: `29` frames
- [ ] Medium run: `61` frames
- [ ] Full run: `125` frames
- [ ] Record context length per chunk
- [ ] Record GPU memory issues / runtime stability

## Handoff Checklist

- [ ] Plan document updated
- [ ] TODO file updated with real paths
- [ ] One offline trace generated
- [ ] One inference trace generated
- [ ] Alignment conclusion written
