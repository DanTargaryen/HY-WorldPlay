# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
"""Offline AR-distill context trace utilities.

The functions here mirror HY-WorldPlay AR rollout context selection without
running the model.  They make the teacher's reconstituted context memory
observable for later selection-distillation experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hyvideo.utils.retrieval_context import (
    generate_points_in_sphere,
    select_aligned_memory_frames,
)

DEFAULT_MEMORY_FRAMES = 20
DEFAULT_TEMPORAL_CONTEXT_SIZE = 12
DEFAULT_CHUNK_LATENT_FRAMES = 4
DEFAULT_PRED_LATENT_SIZE = 4
DEFAULT_FIRST_CHUNK_SIZE = 4


def video_length_to_num_latents(video_length: int) -> int:
    """Convert pixel-frame length to HY latent-frame count."""
    if video_length <= 0:
        raise ValueError(f"video_length must be positive, got {video_length}")
    return (int(video_length) - 1) // 4 + 1


def normalize_camera_center(w2c: np.ndarray) -> np.ndarray:
    """Normalize W2C poses to the first camera center, matching training data."""
    if w2c.ndim != 3 or w2c.shape[1:] != (4, 4):
        raise ValueError(f"w2c must have shape [T, 4, 4], got {w2c.shape}")
    c2w = np.linalg.inv(w2c)
    c0_inv = np.linalg.inv(c2w[0])
    c2w_aligned = np.array([c0_inv @ camera for camera in c2w])
    return np.linalg.inv(c2w_aligned)


def load_pose_json_latent_cameras(
    pose_path: str | Path,
    *,
    num_latents: int | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pose JSON into latent-frame W2C and intrinsic arrays.

    HY-WorldPlay maps latent index 0 to the first pose entry; latent index i>0
    maps to source frame ``4 * (i - 1) + 4``.
    """
    pose_path = Path(pose_path)
    with pose_path.open("r") as f:
        pose_json = json.load(f)
    pose_keys = list(pose_json.keys())
    if not pose_keys:
        raise ValueError(f"pose file is empty: {pose_path}")
    if num_latents is None:
        num_latents = (len(pose_keys) - 1) // 4 + 1
    if num_latents <= 0:
        raise ValueError(f"num_latents must be positive, got {num_latents}")

    w2c_list: list[np.ndarray] = []
    intrinsic_list: list[np.ndarray] = []
    for latent_idx in range(num_latents):
        pose_key_idx = 0 if latent_idx == 0 else 4 * (latent_idx - 1) + 4
        if pose_key_idx >= len(pose_keys):
            raise ValueError(
                f"pose file {pose_path} has {len(pose_keys)} entries, which is "
                f"too short for latent index {latent_idx} (needs key index {pose_key_idx})"
            )
        pose_entry = pose_json[pose_keys[pose_key_idx]]
        intrinsic = np.array(pose_entry["intrinsic"], dtype=np.float64)
        w2c = np.array(pose_entry["w2c"], dtype=np.float64)
        intrinsic = intrinsic.copy()
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        intrinsic_list.append(intrinsic)
        w2c_list.append(w2c)

    w2c_array = np.array(w2c_list)
    if normalize:
        w2c_array = normalize_camera_center(w2c_array)
    return w2c_array, np.array(intrinsic_list)


def classify_selected_frames(
    selected_frame_indices: list[int],
    *,
    query_start_indices: list[int],
    first_chunk_size: int = DEFAULT_FIRST_CHUNK_SIZE,
    temporal_context_size: int = DEFAULT_TEMPORAL_CONTEXT_SIZE,
) -> dict[str, Any]:
    """Split selected latent frames into first / recent / FoV buckets."""
    selected = sorted(set(int(i) for i in selected_frame_indices))
    first_set = set(range(0, int(first_chunk_size)))
    recent_set: set[int] = set()
    for query_start in query_start_indices:
        recent_set.update(range(max(0, query_start - temporal_context_size), query_start))

    first_indices = [idx for idx in selected if idx in first_set]
    recent_indices = [idx for idx in selected if idx in recent_set]
    fov_indices = [idx for idx in selected if idx not in first_set and idx not in recent_set]

    source_tags: dict[str, list[str]] = {}
    for idx in selected:
        tags: list[str] = []
        if idx in first_set:
            tags.append("first")
        if idx in recent_set:
            tags.append("recent")
        if idx not in first_set and idx not in recent_set:
            tags.append("fov")
        source_tags[str(idx)] = tags

    return {
        "selected_frame_indices": selected,
        "first_chunk_indices": first_indices,
        "recent_indices": recent_indices,
        "fov_retrieved_indices": fov_indices,
        "selected_frame_source_tags": source_tags,
    }


def build_context_trace_record(
    *,
    selected_frame_indices: list[int],
    chunk_id: int,
    chunk_start_latent: int,
    query_start_indices: list[int],
    current_chunk_indices: list[int],
    num_latents: int,
    memory_frames: int = DEFAULT_MEMORY_FRAMES,
    temporal_context_size: int = DEFAULT_TEMPORAL_CONTEXT_SIZE,
    pred_latent_size: int = DEFAULT_PRED_LATENT_SIZE,
    chunk_latent_frames: int = DEFAULT_CHUNK_LATENT_FRAMES,
    first_chunk_size: int = DEFAULT_FIRST_CHUNK_SIZE,
    pose_path: str | None = None,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON-serializable context trace row."""
    buckets = classify_selected_frames(
        selected_frame_indices,
        query_start_indices=query_start_indices,
        first_chunk_size=first_chunk_size,
        temporal_context_size=temporal_context_size,
    )
    return {
        "video_id": video_id,
        "pose_path": pose_path,
        "chunk_id": int(chunk_id),
        "chunk_start_latent": int(chunk_start_latent),
        "query_start_indices": [int(i) for i in query_start_indices],
        "current_chunk_indices": [int(i) for i in current_chunk_indices],
        "num_latents": int(num_latents),
        "chunk_latent_frames": int(chunk_latent_frames),
        "memory_frames": int(memory_frames),
        "temporal_context_size": int(temporal_context_size),
        "pred_latent_size": int(pred_latent_size),
        **buckets,
        "context_length": len(buckets["selected_frame_indices"]),
    }


def select_ar_context_for_chunk(
    w2c_list: np.ndarray,
    *,
    chunk_id: int,
    chunk_latent_frames: int = DEFAULT_CHUNK_LATENT_FRAMES,
    memory_frames: int = DEFAULT_MEMORY_FRAMES,
    temporal_context_size: int = DEFAULT_TEMPORAL_CONTEXT_SIZE,
    pred_latent_size: int = DEFAULT_PRED_LATENT_SIZE,
    first_chunk_size: int = DEFAULT_FIRST_CHUNK_SIZE,
    points_local: torch.Tensor | None = None,
    device: str | torch.device | None = None,
    pose_path: str | None = None,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Mirror ``ar_rollout`` context selection for one chunk."""
    num_latents = int(w2c_list.shape[0])
    chunk_start = int(chunk_id) * int(chunk_latent_frames)
    chunk_end = min(chunk_start + int(chunk_latent_frames), num_latents)
    if chunk_start >= num_latents:
        raise ValueError(
            f"chunk_id={chunk_id} starts at {chunk_start}, outside num_latents={num_latents}"
        )

    current_chunk_indices = list(range(chunk_start, chunk_end))
    query_start_indices = list(range(chunk_start, chunk_end, 4))
    if chunk_id == 0:
        return build_context_trace_record(
            selected_frame_indices=[],
            chunk_id=chunk_id,
            chunk_start_latent=chunk_start,
            query_start_indices=query_start_indices,
            current_chunk_indices=current_chunk_indices,
            num_latents=num_latents,
            memory_frames=memory_frames,
            temporal_context_size=temporal_context_size,
            pred_latent_size=pred_latent_size,
            chunk_latent_frames=chunk_latent_frames,
            first_chunk_size=first_chunk_size,
            pose_path=pose_path,
            video_id=video_id,
        )

    if points_local is None:
        points_local = generate_points_in_sphere(50000, 8.0)
    if device is not None:
        points_local = points_local.to(device)

    selected: list[int] = []
    for chunk_query_start in query_start_indices:
        selected += select_aligned_memory_frames(
            w2c_list,
            chunk_query_start,
            memory_frames=memory_frames,
            temporal_context_size=temporal_context_size,
            pred_latent_size=pred_latent_size,
            points_local=points_local,
            device=device,
        )
    current_set = set(current_chunk_indices)
    selected = sorted(idx for idx in set(selected) if idx not in current_set)

    return build_context_trace_record(
        selected_frame_indices=selected,
        chunk_id=chunk_id,
        chunk_start_latent=chunk_start,
        query_start_indices=query_start_indices,
        current_chunk_indices=current_chunk_indices,
        num_latents=num_latents,
        memory_frames=memory_frames,
        temporal_context_size=temporal_context_size,
        pred_latent_size=pred_latent_size,
        chunk_latent_frames=chunk_latent_frames,
        first_chunk_size=first_chunk_size,
        pose_path=pose_path,
        video_id=video_id,
    )


def trace_ar_contexts(
    w2c_list: np.ndarray,
    *,
    chunk_latent_frames: int = DEFAULT_CHUNK_LATENT_FRAMES,
    memory_frames: int = DEFAULT_MEMORY_FRAMES,
    temporal_context_size: int = DEFAULT_TEMPORAL_CONTEXT_SIZE,
    pred_latent_size: int = DEFAULT_PRED_LATENT_SIZE,
    first_chunk_size: int = DEFAULT_FIRST_CHUNK_SIZE,
    points_local: torch.Tensor | None = None,
    device: str | torch.device | None = None,
    pose_path: str | None = None,
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    """Trace all AR chunks for a latent pose sequence."""
    num_latents = int(w2c_list.shape[0])
    if num_latents % chunk_latent_frames != 0:
        raise ValueError(
            f"num_latents={num_latents} must be divisible by chunk_latent_frames={chunk_latent_frames}"
        )
    if points_local is None:
        points_local = generate_points_in_sphere(50000, 8.0)
    if device is not None:
        points_local = points_local.to(device)

    return [
        select_ar_context_for_chunk(
            w2c_list,
            chunk_id=chunk_id,
            chunk_latent_frames=chunk_latent_frames,
            memory_frames=memory_frames,
            temporal_context_size=temporal_context_size,
            pred_latent_size=pred_latent_size,
            first_chunk_size=first_chunk_size,
            points_local=points_local,
            device=device,
            pose_path=pose_path,
            video_id=video_id,
        )
        for chunk_id in range(num_latents // chunk_latent_frames)
    ]


def validate_context_record(record: dict[str, Any]) -> list[str]:
    """Return invariant violations for a trace record."""
    errors: list[str] = []
    selected = set(record["selected_frame_indices"])
    current = set(record["current_chunk_indices"])
    if selected & current:
        errors.append(f"selected current chunk frames: {sorted(selected & current)}")
    if record["chunk_id"] > 0:
        first = set(range(0, min(DEFAULT_FIRST_CHUNK_SIZE, record["chunk_start_latent"])))
        if first and not (selected & first):
            errors.append("missing first chunk context")
        recent = set(
            range(
                max(0, record["chunk_start_latent"] - record["temporal_context_size"]),
                record["chunk_start_latent"],
            )
        )
        missing_recent = sorted(recent - selected)
        if missing_recent:
            errors.append(f"missing recent context frames: {missing_recent}")
    if record["context_length"] != len(record["selected_frame_indices"]):
        errors.append("context_length does not match selected_frame_indices")
    return errors


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_context_summary(records: list[dict[str, Any]]) -> str:
    lines = [
        "chunk | start | context | first | recent | fov | selected",
        "--- | ---: | ---: | ---: | ---: | ---: | ---",
    ]
    for record in records:
        selected = ",".join(str(i) for i in record["selected_frame_indices"])
        lines.append(
            f"{record['chunk_id']} | {record['chunk_start_latent']} | "
            f"{record['context_length']} | {len(record['first_chunk_indices'])} | "
            f"{len(record['recent_indices'])} | {len(record['fov_retrieved_indices'])} | "
            f"{selected}"
        )
    return "\n".join(lines)
