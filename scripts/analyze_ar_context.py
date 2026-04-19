#!/usr/bin/env python3
"""Analyze HY-WorldPlay AR-distill teacher context selection offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyvideo.utils.context_trace import (  # noqa: E402
    format_context_summary,
    load_pose_json_latent_cameras,
    trace_ar_contexts,
    validate_context_record,
    video_length_to_num_latents,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace AR-distill context frames without running model weights.",
    )
    parser.add_argument("--pose-path", required=True, help="Path to pose JSON file")
    parser.add_argument(
        "--video-length",
        type=int,
        default=125,
        help="Pixel-frame video length; converted to latent count as (N-1)//4+1",
    )
    parser.add_argument(
        "--num-latents",
        type=int,
        default=None,
        help="Override latent-frame count directly",
    )
    parser.add_argument("--chunk-latent-frames", type=int, default=4)
    parser.add_argument("--memory-frames", type=int, default=20)
    parser.add_argument("--temporal-context-size", type=int, default=12)
    parser.add_argument("--pred-latent-size", type=int, default=4)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--jsonl-out", default=None, help="Write machine-readable JSONL")
    parser.add_argument("--summary-out", default=None, help="Write markdown summary")
    parser.add_argument(
        "--no-normalize-camera",
        action="store_true",
        help="Do not normalize W2C poses to the first camera",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if trace invariants are violated",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    num_latents = args.num_latents
    if num_latents is None:
        num_latents = video_length_to_num_latents(args.video_length)

    w2c, _ = load_pose_json_latent_cameras(
        args.pose_path,
        num_latents=num_latents,
        normalize=not args.no_normalize_camera,
    )
    records = trace_ar_contexts(
        w2c,
        chunk_latent_frames=args.chunk_latent_frames,
        memory_frames=args.memory_frames,
        temporal_context_size=args.temporal_context_size,
        pred_latent_size=args.pred_latent_size,
        pose_path=args.pose_path,
        video_id=args.video_id or Path(args.pose_path).stem,
    )

    violations: list[str] = []
    for record in records:
        for error in validate_context_record(record):
            violations.append(f"chunk {record['chunk_id']}: {error}")

    summary = format_context_summary(records)
    print(summary)
    if violations:
        print("\nInvariant violations:", file=sys.stderr)
        for error in violations:
            print(f"- {error}", file=sys.stderr)
        if args.strict:
            return 1

    if args.jsonl_out:
        write_jsonl(records, args.jsonl_out)
    if args.summary_out:
        path = Path(args.summary_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
