"""Build per-task train/eval EEF density maps and a single furthest blind-spot trail.

CPU-only. Eval recordings are optional: missing EEF data is treated as zero eval
coverage for that task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from blindspot_utils import (
    COVERAGE_NOTE,
    LIBERO_OBJECT_ENV_TASK_IDS,
    LIBERO_OBJECT_SUITE,
    build_histogram,
    compute_coverage_trail,
    env_task_id_to_dataset_index,
    find_blind_bins,
    load_eval_states,
    load_train_arrays,
    load_train_states,
    padded_range,
    pick_anchor_frame,
    plot_density_maps,
    rank_blind_bins,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", default="lerobot/libero")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("outputs/eval/object_recorded/recordings"),
        help="Directory containing libero_object_{id} recorded datasets",
    )
    parser.add_argument("--suite", default=LIBERO_OBJECT_SUITE)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument(
        "--anchor-mode",
        choices=["far", "near"],
        default="far",
        help="Anchor frame: far=farthest in-distribution demo from furthest point",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--n-trail-points", type=int, default=5, help="Trail points p1..p5")
    parser.add_argument("--state-dims", type=int, default=3, choices=[3, 6])
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="Env task ids in 0..9. Default: all libero_object tasks.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/blindspot/coverage"))
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def _task_ids(args: argparse.Namespace) -> list[int]:
    if args.suite != LIBERO_OBJECT_SUITE:
        raise ValueError(f"Only {LIBERO_OBJECT_SUITE} is supported")
    ids = list(args.task_ids) if args.task_ids else list(LIBERO_OBJECT_ENV_TASK_IDS)
    for env_task_id in ids:
        env_task_id_to_dataset_index(env_task_id, args.suite)
    return ids


def _init_xyz_for_trail(anchor: dict, train_positions: np.ndarray) -> np.ndarray:
    """Approximate object-init XYZ for histogram trail plotting (sim refines later)."""
    eef = np.asarray(anchor["eef"], dtype=np.float32).reshape(3)
    z = float(np.percentile(train_positions[:, 2], 15)) if len(train_positions) else eef[2]
    return np.array([eef[0], eef[1], z], dtype=np.float32)


def process_task(
    dataset: LeRobotDataset,
    env_task_id: int,
    args: argparse.Namespace,
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict:
    task_index = env_task_id_to_dataset_index(env_task_id, args.suite, meta=dataset.meta)
    train = load_train_states(
        dataset,
        task_index=task_index,
        stride=args.stride,
        state_dims=args.state_dims,
        arrays=train_arrays,
    )
    eval_root = args.eval_dir if args.eval_dir.exists() else None
    eval_table = load_eval_states(
        eval_root, env_task_id=env_task_id, stride=args.stride, state_dims=args.state_dims
    )

    low, high = padded_range(train.positions)
    train_hist, edges = build_histogram(train.positions, args.bins, low, high)
    eval_hist, _ = build_histogram(eval_table.positions, args.bins, low, high)
    blind_coords = find_blind_bins(train_hist, eval_hist)
    selected_coords, selected_centroids, train_dists, eval_dists, support_dists = rank_blind_bins(
        blind_coords,
        edges,
        train.positions,
        eval_table.positions,
        top_k=1,
        selection="deep",
    )

    if len(selected_coords) == 0:
        raise ValueError(f"No blind bins for task {env_task_id}")

    furthest_coord = selected_coords[0]
    furthest_centroid = selected_centroids[0]
    anchor = pick_anchor_frame(
        train, furthest_centroid, train_hist, edges, anchor_mode=args.anchor_mode
    )
    init_xyz = _init_xyz_for_trail(anchor, train.positions)
    trail = compute_coverage_trail(
        init_xyz,
        furthest_centroid,
        train_hist,
        eval_hist,
        edges,
        low,
        high,
        train.positions,
        eval_table.positions,
        n_points=args.n_trail_points,
    )

    trail_centroids = np.asarray([point["centroid"] for point in trail], dtype=np.float32)
    trail_support_dists = np.asarray(
        [point["nearest_support_distance"] for point in trail], dtype=np.float32
    )
    trail_labels = [str(point["label"]) for point in trail]

    out_dir = args.output_dir / f"task_{task_index}"
    plot_density_maps(
        train_hist,
        eval_hist,
        edges,
        out_dir,
        trail_centroids=trail_centroids,
        trail_support_dists=trail_support_dists,
        trail_labels=trail_labels,
        title=f"{args.suite} task {env_task_id} ({train.task})",
    )
    payload = {
        "suite": args.suite,
        "env_task_id": env_task_id,
        "task_index": task_index,
        "task": train.task,
        "note": COVERAGE_NOTE,
        "bins": args.bins,
        "anchor_mode": args.anchor_mode,
        "stride": args.stride,
        "state_dims": args.state_dims,
        "n_trail_points": args.n_trail_points,
        "n_train": len(train),
        "n_eval": len(eval_table),
        "n_blind_bins": int(len(blind_coords)),
        "eval_dir_exists": bool(eval_root),
        "range_low": low.tolist(),
        "range_high": high.tolist(),
        "edges": [edge.tolist() for edge in edges],
        "train_hist": train_hist.tolist(),
        "eval_hist": eval_hist.tolist(),
        "init_xyz_hint": init_xyz.tolist(),
        "furthest_point": {
            "bin_ijk": furthest_coord.tolist(),
            "centroid": furthest_centroid.tolist(),
            "nearest_train_distance": float(train_dists[0]),
            "nearest_eval_distance": float(eval_dists[0]),
            "nearest_support_distance": float(support_dists[0]),
            "anchor": anchor,
        },
        "trail": trail,
    }
    write_json(out_dir / "blind_spots.json", payload)
    print(
        f"task {env_task_id} ({train.task!r}): train={len(train)} eval={len(eval_table)} "
        f"blind_bins={len(blind_coords)} furthest_support_dist={support_dists[0]:.3f} -> {out_dir}"
    )
    return {
        "env_task_id": env_task_id,
        "task_index": task_index,
        "task": train.task,
        "n_train": len(train),
        "n_eval": len(eval_table),
        "n_blind_bins": int(len(blind_coords)),
        "furthest_support_distance": float(support_dists[0]),
        "output_dir": str(out_dir),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(args.train_dataset, video_backend=args.video_backend)
    train_arrays = load_train_arrays(dataset)
    summaries = [
        process_task(dataset, env_task_id, args, train_arrays=train_arrays)
        for env_task_id in _task_ids(args)
    ]
    write_json(
        args.output_dir / "summary.json",
        {
            "note": COVERAGE_NOTE,
            "train_dataset": args.train_dataset,
            "eval_dir": str(args.eval_dir),
            "tasks": summaries,
        },
    )
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
