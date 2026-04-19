# AR-distill Context Trace Baseline

This document defines the first offline baseline for HY-WorldPlay AR-distill
teacher context selection. It does not run model weights or CUDA kernels; it only
replays the rule used by AR rollout to choose memory frames.

## Teacher Context Definition

For every AR chunk after chunk 0, the teacher context is selected at latent-frame
resolution:

- `first_chunk_indices`: fixed first chunk frames, normally `0,1,2,3`.
- `recent_indices`: the last `temporal_context_size=12` latent frames before the
  current query chunk.
- `fov_retrieved_indices`: older 4-frame history blocks selected by
  `select_aligned_memory_frames(...)` from FoV overlap.
- `selected_frame_indices`: deduplicated sorted union of the above, excluding
  the current chunk itself.

The default budget mirrors inference and memory training:

- `chunk_latent_frames=4`
- `memory_frames=20`
- `temporal_context_size=12`
- `pred_latent_size=4`

## Offline Trace Command

```bash
cd /Users/mumuxunzi/workspace/HY-WorldPlay
python scripts/analyze_ar_context.py \
  --pose-path assets/pose/your_pose.json \
  --video-length 125 \
  --jsonl-out outputs/context_traces/example.jsonl \
  --summary-out outputs/context_traces/example.md \
  --strict
```

Use `--num-latents` instead of `--video-length` when you already know the latent
length.

## JSONL Schema

Each row describes one AR chunk:

```json
{
  "video_id": "example",
  "pose_path": "assets/pose/example.json",
  "chunk_id": 2,
  "chunk_start_latent": 8,
  "query_start_indices": [8],
  "current_chunk_indices": [8, 9, 10, 11],
  "selected_frame_indices": [0, 1, 2, 3, 4, 5, 6, 7],
  "first_chunk_indices": [0, 1, 2, 3],
  "recent_indices": [0, 1, 2, 3, 4, 5, 6, 7],
  "fov_retrieved_indices": [],
  "selected_frame_source_tags": {"0": ["first", "recent"]},
  "context_length": 8
}
```

## Optional Inference Hook

Set `HY_WORLDPLAY_CONTEXT_TRACE_JSONL` before running `hyvideo/generate.py` to
append the actual AR rollout context selection from rank 0:

```bash
export HY_WORLDPLAY_CONTEXT_TRACE_JSONL=outputs/context_traces/inference.jsonl
bash run.sh
```

This hook is inactive when the environment variable is unset.

## Distillation Use

For the first student experiment, use only `selected_frame_indices` and
`selected_frame_source_tags` as selection-distillation supervision. Add teacher
rollout or denoise targets later after the GPU/weight environment is ready.
