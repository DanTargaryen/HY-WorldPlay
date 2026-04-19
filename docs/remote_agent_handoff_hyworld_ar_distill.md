# Remote Agent Handoff — HY-WorldPlay AR-distill Context Baseline

## Mission

Use the existing HY-WorldPlay AR-distill model as the first teacher baseline and
verify how its memory context is built for each chunk.

The current task is **not** to change the model. The task is to:

1. replay teacher context selection offline
2. capture actual context selection during inference
3. compare the two
4. prepare traces for later student distillation

## Important Files

- `run.sh`
- `hyvideo/generate.py`
- `hyvideo/pipelines/worldplay_video_pipeline.py`
- `hyvideo/utils/retrieval_context.py`
- `hyvideo/utils/context_trace.py`
- `scripts/analyze_ar_context.py`
- `docs/ar_distill_context_trace.md`
- `docs/project_plan_hyworld_ar_distill_memory.md`
- `TODO_hyworld_ar_distill_memory.md`

## Baseline Teacher Rule

Treat the current AR-distill inference rule as canonical unless evidence shows
otherwise:

- first chunk fixed context: `0,1,2,3`
- recent window: `12` latent frames
- FoV-selected memory: chosen from historical 4-latent blocks
- memory budget: `20`
- current chunk excluded from context

## Commands To Run First

### 1. Offline trace

```bash
cd /Users/mumuxunzi/workspace/HY-WorldPlay
python scripts/analyze_ar_context.py   --pose-path <POSE_JSON>   --video-length 125   --jsonl-out outputs/context_traces/offline.jsonl   --summary-out outputs/context_traces/offline.md   --strict
```

### 2. Inference trace

```bash
export HY_WORLDPLAY_CONTEXT_TRACE_JSONL=outputs/context_traces/inference.jsonl
```

Then run the normal AR-distill command with real paths.

Start with a short run before trying 125 frames.

## What To Collect

For each run, collect:

- command used
- model paths used
- pose path used
- selected frame indices per chunk
- context length per chunk
- counts of first / recent / fov selections
- runtime and GPU memory issues

## Deliverables Back To Human

- offline trace jsonl
- inference trace jsonl
- one markdown summary table
- one note describing whether offline and inference traces match
- one note describing whether FoV retrieval seems meaningful beyond recent-only context

## Do Not Do Yet

- do not redesign the transformer
- do not add a sparse student yet
- do not mix teacher and student behavior in the same trace format
- do not treat training-side sampling as the final teacher behavior without comparison
