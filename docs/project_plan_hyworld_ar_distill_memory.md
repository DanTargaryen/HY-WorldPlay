# HY-WorldPlay AR-distill Memory Baseline Project Plan

## 1. Background

This project starts from **HY-World 1.5 / HY-WorldPlay** and uses its existing
**autoregressive distilled model** as the first teacher baseline for memory-aware
world-model experimentation.

The immediate goal is **not** to change the model architecture. Instead, the
first phase is to make the current teacher behavior observable, reproducible,
and easy to evaluate so that later agents can build sparse-memory students on a
solid baseline.

The core teacher context structure, as implemented today, is:

- **First-window frames**: fixed frames from the first chunk, typically `0,1,2,3`
- **FoV-retrieved frames**: historical frames selected by camera overlap / FoV
  retrieval
- **Recent frames**: the most recent temporal context before the current chunk

This matches the user's intended starting point:

> start from the HY-World 1.5 AR-distill model and test the current memory
> design first, where context consists of first-window frames, FoV-retrieved
> frames, and recent frames.

## 2. Project Goal

Build a **baseline testing and trace-export workflow** for HY-WorldPlay
AR-distill memory selection that can later support memory distillation into a
new sparse top-k student.

The project should answer the following questions clearly:

1. What frames does the current AR-distill teacher use as memory for each chunk?
2. How are those selected frames split into `first`, `recent`, and `fov`
   sources?
3. Is the selection logic used during inference consistent with the selection
   logic used during training data preparation?
4. Can we export those selections into a stable trace format for later student
   supervision?

## 3. Current Repository Facts

### 3.1 Inference Entry

The AR-distill teacher path is currently exercised through:

- `hyvideo/generate.py`
- `run.sh` with:
  - `--model_type ar`
  - `--few_step true`
  - `--num_inference_steps 4`
  - `--action_ckpt $AR_DISTILL_ACTION_MODEL_PATH`

### 3.2 Context Selection Logic

The memory selection logic is implemented in:

- `hyvideo/utils/retrieval_context.py`
- `hyvideo/pipelines/worldplay_video_pipeline.py`
- `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`

### 3.3 Current Default Teacher Rule

The current rule should be treated as the initial teacher spec:

- `chunk_latent_frames = 4`
- `memory_frames = 20`
- `temporal_context_size = 12`
- `pred_latent_size = 4`
- first chunk always contributes `0,1,2,3`
- current chunk must not be included in selected context

### 3.4 Existing Baseline Utilities Already Added

The repository already contains support for observing this baseline:

- `hyvideo/utils/context_trace.py`
- `scripts/analyze_ar_context.py`
- `docs/ar_distill_context_trace.md`
- optional inference hook in `hyvideo/pipelines/worldplay_video_pipeline.py`

## 4. Phase Plan

### Phase A — Rule Reproduction Baseline

Objective: reproduce teacher context selection **without loading model weights**.

Deliverables:

- a script that reads `pose_path` and reconstructs per-chunk context selection
- machine-readable output (`jsonl`)
- human-readable summary (`md` or terminal table)
- invariant checks:
  - first chunk included when expected
  - recent window included
  - current chunk excluded
  - selected frames deduplicated and sorted

Success criteria:

- repeated runs on the same pose file produce identical trace output
- the trace is understandable enough to serve as a supervision target later

### Phase B — Inference-Time Teacher Trace

Objective: log actual context selection during real AR-distill rollout.

Deliverables:

- optional hook controlled by env var
- one jsonl row per chunk during inference
- minimal runtime overhead when disabled

Success criteria:

- hook is fully disabled when env var is absent
- hook captures chunk-level selected frame indices on rank 0
- logged selection matches offline rule replay on the same pose sequence

### Phase C — Training vs Inference Alignment

Objective: compare training-side memory sampling with inference-side teacher
selection.

Key question:

- For future distillation, should teacher traces follow **inference behavior**
  or **training sampling behavior**?

Current working assumption:

- Use **inference-side selection** as the canonical teacher behavior
- Use training-side logic only to understand sampling bias and to design later
  trace generation workflows

Success criteria:

- a written comparison between the two paths
- a list of known differences and why they exist
- a decision note for later student supervision

### Phase D — Distillation-Ready Trace Schema

Objective: define the minimum supervision package for a first student.

V1 trace fields:

- `video_id`
- `pose_path`
- `chunk_id`
- `chunk_start_latent`
- `query_start_indices`
- `current_chunk_indices`
- `selected_frame_indices`
- `first_chunk_indices`
- `recent_indices`
- `fov_retrieved_indices`
- `selected_frame_source_tags`
- `context_length`

V2 optional extensions when GPU environment is ready:

- `fov_overlap_scores`
- `teacher rollout target`
- `teacher denoise target`
- `teacher hidden state / context features`

Success criteria:

- V1 trace is enough for **selection distillation**
- schema is stable across multiple sequences

### Phase E — Remote Execution Baseline

Objective: run a staged test ladder on the remote CUDA machine.

Recommended order:

1. offline trace only
2. short AR-distill smoke test at `29` frames
3. medium AR-distill smoke test at `61` frames
4. full standard test at `125` frames

For every run, collect:

- selected frame indices per chunk
- context length per chunk
- counts of first/recent/fov sources
- runtime
- GPU memory / OOM / cache issues

## 5. Recommended Evaluation Questions

The remote agent should explicitly answer:

1. Does AR-distill use the expected context decomposition?
2. Does FoV retrieval produce qualitatively different memory from simple recent
   windowing?
3. Which chunks rely most on FoV retrieval?
4. Does context length remain stable through longer rollouts?
5. Are there any mismatches between offline trace replay and actual inference
   trace?

## 6. Risks and Constraints

- The current local machine is not prepared for full execution:
  - no confirmed Python runtime in PATH for the project workflow
  - no confirmed CUDA runtime
  - no confirmed model checkpoints in local paths
- HunyuanVideo inference is heavyweight; smoke tests should start short
- Training-side memory sampling includes curriculum-like and outside-window
  logic, which may not exactly match inference behavior
- Student work should not begin until teacher trace quality is confirmed

## 7. Decision Defaults

Unless a later human overrides them, use these defaults:

- teacher baseline = **HY-WorldPlay AR-distill**
- canonical teacher behavior = **inference-side context selection**
- first student supervision = **selection-only distillation**
- frame unit = **latent frame**, not pixel frame
- first execution order = offline trace -> short inference trace -> alignment ->
  distillation prep

## 8. Expected Next Step After This Phase

Once the baseline is verified, the next project should be:

- export teacher traces for a real dataset
- compare teacher trace and future student top-k trace chunk-by-chunk
- add student-side sparse memory selection experiments
- optionally extend from selection-only to rollout-aware distillation
