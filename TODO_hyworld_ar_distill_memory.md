# TODO — HY-WorldPlay AR-distill Memory Baseline

> Remote agent instruction: **read this file first and execute the task exactly as described**.
>
> Human instruction to remote agent can be as simple as:
>
> `Read TODO_hyworld_ar_distill_memory.md and complete the task.`

## Fixed Paths

## Cluster and Session Rule

All real GPU work must run inside the target Kubernetes pods, not on a random
login shell.

Preferred GPU entry commands:

```bash
kubectl exec -it hzh-dev-juicef-2n-master-0 -- bash
kubectl exec -it hzh-dev-juicef-2n-worker-0 -- bash
```

Before any long-running task, create or attach a `tmux` session first. Do not
run long inference or tracing jobs in a fragile foreground shell.

Recommended pattern:

```bash
tmux new -s hyworld || tmux attach -t hyworld
```

## Network Proxy Rule

Before any network-dependent action such as model download, pip install, git
fetch, or HuggingFace access, export:

```bash
export http_proxy=http://192.168.48.17:18000
export https_proxy=http://192.168.48.17:18000
```

If a command fails and looks network-related, first verify that these proxy
variables are still present in the current shell or tmux session.


Use these paths as defaults unless a human explicitly overrides them.

### Code and model / weight root

```bash
PROJECT_ROOT=/mnt/pfs/users/huangzehuan/projects/linming/HY-WorldPlay
MODEL_ROOT=/mnt/pfs/users/huangzehuan/projects/linming
```

### Conda activation script

```bash
CONDA_ACTIVATE_SCRIPT=/mnt/pfs/users/huangzehuan/dev/miniconda3/bin/activate
```

### Data and outputs root

```bash
DATA_ROOT=/mnt/pfs/data/huangzehuan/datasets
TRACE_ROOT=/mnt/pfs/data/huangzehuan/datasets/hyworld_ar_distill_context_traces
OUTPUT_ROOT=/mnt/pfs/data/huangzehuan/datasets/hyworld_ar_distill_outputs
LOG_ROOT=/mnt/pfs/data/huangzehuan/datasets/hyworld_ar_distill_logs
REPORT_ROOT=/mnt/pfs/data/huangzehuan/datasets/hyworld_ar_distill_reports
```

Create directories if missing:

```bash
mkdir -p "$TRACE_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT" "$REPORT_ROOT"
```

## Absolute Path Rule

This task uses many fixed paths under `/mnt/pfs/...`.

**Rule:** any path beginning with `/` is an **absolute path** and must be used
exactly as written.

Do **not** prepend `PROJECT_ROOT`, current working directory, or any other path
prefix to absolute paths.

Example:

- correct: `/mnt/pfs/users/huangzehuan/dev/miniconda3/bin/activate`
- wrong: `/mnt/pfs/users/huangzehuan/projects/linming/users/huangzehuan/dev/miniconda3/bin/activate`

If you are unsure whether a path exists, check it directly with `ls -l <abs_path>`.

## Environment Activation Rule

Before any command involving Python, pip, conda, torch, torchrun, or model
execution, first run:

```bash
source "$CONDA_ACTIVATE_SCRIPT"
```

Then check whether a specific conda environment should be activated:

```bash
conda env list
```

If a project-specific environment exists, activate it and record its name in the
final report:

```bash
conda activate <env_name>
```

Do **not** use system python unless the conda activation path is unavailable and
this fact is explicitly recorded in the final report.

## Mission

The goal is **not** to change the model architecture.

The goal is to use the existing **HY-WorldPlay AR-distill model** as the first
teacher baseline and verify how its memory context is constructed for each
chunk.

The current teacher context is expected to consist of:

- first-window fixed frames
- FoV-retrieved historical frames
- recent temporal frames

## Current Teacher Rule To Verify

Treat the current inference rule as the initial teacher spec:

- `chunk_latent_frames = 4`
- `memory_frames = 20`
- `temporal_context_size = 12`
- `pred_latent_size = 4`
- first chunk context should include `0,1,2,3`
- current chunk itself must not appear in selected context

## Important Files To Read

Read these files before running anything:

- `docs/remote_agent_handoff_hyworld_ar_distill.md`
- `docs/project_plan_hyworld_ar_distill_memory.md`
- `docs/ar_distill_context_trace.md`
- `hyvideo/utils/context_trace.py`
- `scripts/analyze_ar_context.py`
- `hyvideo/pipelines/worldplay_video_pipeline.py`
- `hyvideo/utils/retrieval_context.py`
- `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`

## Task 1 — Confirm Environment

Run and save:

```bash
kubectl exec -it hzh-dev-juicef-2n-master-0 -- bash
# or
kubectl exec -it hzh-dev-juicef-2n-worker-0 -- bash

tmux new -s hyworld || tmux attach -t hyworld

export http_proxy=http://192.168.48.17:18000
export https_proxy=http://192.168.48.17:18000

cd "$PROJECT_ROOT"
source "$CONDA_ACTIVATE_SCRIPT"

{
  echo "===== BASIC ====="
  pwd
  git status --short
  git log -1 --oneline

  echo
  echo "===== ABSOLUTE PATH CHECK ====="
  ls -l "$CONDA_ACTIVATE_SCRIPT" || true

  echo
  echo "===== CONDA ====="
  which conda || true
  conda env list || true

  echo
  echo "===== PYTHON ====="
  which python || true
  python --version || true
  which python3 || true
  python3 --version || true

  echo
  echo "===== GPU ====="
  nvidia-smi || true

  echo
  echo "===== TORCH ====="
  python - <<'INNERPY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(i, props.name, props.total_memory / 1024**3, "GB")
INNERPY
} | tee "$LOG_ROOT/env_check.log"
```

If `python` does not exist after sourcing conda, try activating a concrete conda
environment and rerun the checks. Record the actual environment name in the
final report.

## Task 2 — Confirm Model Paths

Find and confirm:

```bash
MODEL_PATH=<path to HunyuanVideo-1.5 base model>
AR_DISTILL_ACTION_MODEL_PATH=<path to HY-WorldPlay AR-distill action safetensors>
```

Search under `MODEL_ROOT` if needed:

```bash
find "$MODEL_ROOT" -maxdepth 8 -type d -iname '*HunyuanVideo*' | head -50
find "$MODEL_ROOT" -maxdepth 10 -type f -iname '*.safetensors' | grep -i 'distill\|action\|ar' | head -50
```

Save confirmed paths:

```bash
{
  echo "MODEL_PATH=$MODEL_PATH"
  echo "AR_DISTILL_ACTION_MODEL_PATH=$AR_DISTILL_ACTION_MODEL_PATH"
  ls -la "$MODEL_PATH" || true
  ls -la "$AR_DISTILL_ACTION_MODEL_PATH" || true
} | tee "$LOG_ROOT/model_paths.log"
```

## Task 3 — Prepare One Real Test Case

Need one real:

- `TEST_IMAGE`
- `TEST_POSE`
- `TEST_PROMPT`

If the repo contains usable sample inputs, prefer them. Otherwise locate them
under `DATA_ROOT` or nearby project assets and record the final chosen paths.

Save them:

```bash
{
  echo "TEST_IMAGE=$TEST_IMAGE"
  echo "TEST_POSE=$TEST_POSE"
  echo "TEST_PROMPT=$TEST_PROMPT"
  ls -la "$TEST_IMAGE" || true
  ls -la "$TEST_POSE" || true
} | tee "$LOG_ROOT/test_inputs.log"
```

## Task 4 — Run Offline Context Trace First

Always activate conda first:

```bash
cd "$PROJECT_ROOT"
source "$CONDA_ACTIVATE_SCRIPT"
```

### 29-frame trace

```bash
python scripts/analyze_ar_context.py   --pose-path "$TEST_POSE"   --video-length 29   --jsonl-out "$TRACE_ROOT/offline_29f.jsonl"   --summary-out "$TRACE_ROOT/offline_29f.md"   --strict   2>&1 | tee "$LOG_ROOT/offline_trace_29f.log"
```

### 61-frame trace

```bash
python scripts/analyze_ar_context.py   --pose-path "$TEST_POSE"   --video-length 61   --jsonl-out "$TRACE_ROOT/offline_61f.jsonl"   --summary-out "$TRACE_ROOT/offline_61f.md"   --strict   2>&1 | tee "$LOG_ROOT/offline_trace_61f.log"
```

### 125-frame trace

```bash
python scripts/analyze_ar_context.py   --pose-path "$TEST_POSE"   --video-length 125   --jsonl-out "$TRACE_ROOT/offline_125f.jsonl"   --summary-out "$TRACE_ROOT/offline_125f.md"   --strict   2>&1 | tee "$LOG_ROOT/offline_trace_125f.log"
```

Check that:

- chunk 0 has no history context
- later chunks include first / recent / fov decomposition
- current chunk is excluded
- outputs are deterministic across repeated runs

## Task 5 — Run Short Real AR-distill Inference With Trace Hook

Always activate conda first:

```bash
cd "$PROJECT_ROOT"
source "$CONDA_ACTIVATE_SCRIPT"
```

Set trace output first:

```bash
export HY_WORLDPLAY_CONTEXT_TRACE_JSONL="$TRACE_ROOT/inference_29f.jsonl"
rm -f "$HY_WORLDPLAY_CONTEXT_TRACE_JSONL"
```

Run 29-frame AR-distill smoke test:

```bash
torchrun --nproc_per_node=1 hyvideo/generate.py   --prompt "$TEST_PROMPT"   --image_path "$TEST_IMAGE"   --resolution 480p   --aspect_ratio 16:9   --video_length 29   --seed 1   --rewrite false   --sr false   --save_pre_sr_video   --pose "$TEST_POSE"   --output_path "$OUTPUT_ROOT/smoke_29f"   --model_path "$MODEL_PATH"   --action_ckpt "$AR_DISTILL_ACTION_MODEL_PATH"   --few_step true   --num_inference_steps 4   --model_type ar   --height 480   --width 832   --use_vae_parallel false   --use_sageattn false   --use_fp8_gemm false   --transformer_resident_ar_rollout true   2>&1 | tee "$LOG_ROOT/inference_29f.log"
```

If 1 GPU fails due to memory, retry with `--nproc_per_node=4` or `8`, and log
exactly what changed.

## Task 6 — If 29 Frames Succeeds, Run 61 Then 125

Repeat the same process with:

- `video_length = 61`
- `video_length = 125`

and write traces to:

- `$TRACE_ROOT/inference_61f.jsonl`
- `$TRACE_ROOT/inference_125f.jsonl`

and outputs to:

- `$OUTPUT_ROOT/smoke_61f`
- `$OUTPUT_ROOT/full_125f`

## Task 7 — Compare Offline Trace vs Inference Trace

For each successful length (`29`, `61`, `125`), compare:

- chunk count
- `selected_frame_indices`
- `context_length`
- first / recent / fov decomposition

Need a per-chunk mismatch report if they differ.

## Task 8 — Compare Inference Logic vs Training Logic

Read and compare:

- `hyvideo/pipelines/worldplay_video_pipeline.py`
- `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`

Required conclusion:

- what is shared between training and inference selection
- what differs
- whether teacher traces for future distillation should follow inference or
  training behavior

Default expected answer unless evidence contradicts it:

- **teacher trace for distillation should follow inference behavior**
- training-side logic only describes how memory-like samples are constructed
  during training

## Runtime Discipline

- use the Kubernetes GPU pods listed above for actual GPU work
- use `tmux` before any long-running command
- export the proxy variables in every fresh shell
- source the conda activation script before any python / torchrun command
- do not assume the current shell already has the correct environment

## Deliverables

All final artifacts must be saved under:

```bash
/mnt/pfs/data/huangzehuan/datasets
```

Required deliverables:

- environment log
- model path log
- test input log
- offline trace jsonl
- offline summary markdown
- inference trace jsonl
- output video directory (if successful)
- final markdown report

## Final Report Format

Write final report to:

```bash
$REPORT_ROOT/final_report.md
```

Use this structure:

```markdown
# HY-WorldPlay AR-distill Context Baseline Report

## 1. Environment
- Host:
- Project root:
- Conda activate script:
- Conda environment name:
- Python:
- Torch:
- CUDA:
- GPU:

## 2. Model Paths
- MODEL_PATH:
- AR_DISTILL_ACTION_MODEL_PATH:

## 3. Test Inputs
- TEST_IMAGE:
- TEST_POSE:
- TEST_PROMPT:

## 4. Offline Trace Results
### 29f
- command:
- outputs:
- invariant status:
- notes:

### 61f
...

### 125f
...

## 5. Inference Trace Results
### 29f
- command:
- output dir:
- trace path:
- success or failure:
- runtime:
- GPU memory notes:

### 61f
...

### 125f
...

## 6. Offline vs Inference Alignment
- identical or not:
- mismatched chunks:
- likely reason:

## 7. Teacher Context Behavior Summary
- first frames behavior:
- recent frames behavior:
- fov retrieval behavior:
- context length behavior:

## 8. Training vs Inference Logic
- shared logic:
- differences:
- recommended teacher trace source:

## 9. Blockers
- missing dependency:
- missing model:
- OOM:
- invalid input:
- other:

## 10. Recommendation
- is teacher selection baseline ready:
- next minimal step:
- whether selection distillation can start:
```

## Do Not Do Yet

- do not redesign the transformer
- do not add a sparse student yet
- do not train a new model yet
- do not change teacher context rules yet
- do not assume training sampling equals teacher inference behavior without comparison

## Definition of Success

This task is successful if the agent can clearly answer:

1. For each chunk, what latent frames does the AR-distill teacher actually use?
2. Which of them come from first / recent / fov memory?
3. Does offline replay match actual inference trace?
4. Should later distillation use inference-side teacher trace as supervision?
