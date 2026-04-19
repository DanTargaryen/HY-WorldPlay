from __future__ import annotations

import numpy as np

from hyvideo.utils.context_trace import (
    build_context_trace_record,
    classify_selected_frames,
    trace_ar_contexts,
    validate_context_record,
)


def _synthetic_w2c(num_latents: int) -> np.ndarray:
    w2c = np.repeat(np.eye(4)[None, ...], num_latents, axis=0)
    for idx in range(num_latents):
        w2c[idx, 0, 3] = idx * 0.05
        w2c[idx, 2, 3] = idx * 0.1
    return w2c


def test_classify_selected_frames_splits_first_recent_and_fov() -> None:
    out = classify_selected_frames(
        [0, 1, 2, 3, 8, 9, 10, 11, 16, 17],
        query_start_indices=[20],
        first_chunk_size=4,
        temporal_context_size=12,
    )
    assert out["first_chunk_indices"] == [0, 1, 2, 3]
    assert out["recent_indices"] == [8, 9, 10, 11, 16, 17]
    assert out["fov_retrieved_indices"] == []
    assert out["selected_frame_source_tags"]["0"] == ["first"]
    assert out["selected_frame_source_tags"]["16"] == ["recent"]


def test_trace_ar_contexts_excludes_current_chunk() -> None:
    records = trace_ar_contexts(
        _synthetic_w2c(32),
        chunk_latent_frames=4,
        memory_frames=20,
        temporal_context_size=12,
        pred_latent_size=4,
    )
    assert records[0]["selected_frame_indices"] == []
    for record in records[1:]:
        assert not set(record["selected_frame_indices"]) & set(record["current_chunk_indices"])
        assert validate_context_record(record) == []


def test_build_context_trace_record_has_expected_schema() -> None:
    record = build_context_trace_record(
        selected_frame_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        chunk_id=2,
        chunk_start_latent=8,
        query_start_indices=[8],
        current_chunk_indices=[8, 9, 10, 11],
        num_latents=32,
    )
    assert record["chunk_id"] == 2
    assert record["context_length"] == 8
    assert record["first_chunk_indices"] == [0, 1, 2, 3]
    assert record["recent_indices"] == [0, 1, 2, 3, 4, 5, 6, 7]
