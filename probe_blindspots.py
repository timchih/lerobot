"""State-swap pi0.5 probes at per-task blind-spot bins.

Holds task text and camera images fixed, interpolates observation.state from an
in-distribution anchor toward each selected blind-bin centroid, and measures
flow-matching action sensitivity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.lerobot_types import PolicyAction
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline

from blindspot_utils import (
    COVERAGE_NOTE,
    PROBE_LIMITATION,
    plot_probe_overlay,
    write_json,
)

WS = Path(os.environ.get("WS", "/vol/gpudata/tq422-vla"))
DEFAULT_MODEL = WS / "models" / "pi05_libero_finetuned_v044"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-dir", type=Path, default=Path("outputs/blindspot/coverage"))
    parser.add_argument("--train-dataset", default="lerobot/libero")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/blindspot/probes"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="Dataset task_index values (20-29). Default: every task_* dir in coverage-dir.",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=None,
        help="Optional cap on blind bins probed per task (for smoke tests).",
    )
    return parser.parse_args()


def _load_policy(
    model_path: Path, device: torch.device
) -> tuple[
    PI05Policy,
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    policy = PI05Policy.from_pretrained(str(model_path)).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(model_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocess, postprocess


def _predict_chunk(policy, preprocess, postprocess, frame: dict, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    batch = preprocess(dict(frame))
    with torch.inference_mode():
        action = policy.predict_action_chunk(batch)
        action = postprocess(action)
    return action[0].float().cpu()


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def _swap_state(frame: dict, state: np.ndarray) -> dict:
    swapped = dict(frame)
    tensor = torch.as_tensor(state, dtype=torch.float32)
    current = swapped.get("observation.state")
    if torch.is_tensor(current) and current.numel() == tensor.numel():
        tensor = tensor.reshape(current.shape).to(dtype=current.dtype)
    swapped["observation.state"] = tensor
    return swapped


def _episode_bounds(dataset: LeRobotDataset, frame_index: int) -> tuple[int, int]:
    frame = dataset[frame_index]
    episode = int(frame["episode_index"])
    start = int(dataset.meta.episodes["dataset_from_index"][episode])
    end = int(dataset.meta.episodes["dataset_to_index"][episode])
    return start, end


def _anchor_mae_vs_demo(
    dataset: LeRobotDataset, frame_index: int, pred: torch.Tensor
) -> dict[str, float | list[float]]:
    start, end = _episode_bounds(dataset, frame_index)
    horizon = min(int(pred.shape[0]), end - frame_index)
    if horizon <= 0:
        return {"mae": float("nan"), "horizon": 0, "per_dim": []}
    rows = dataset.hf_dataset.select(range(frame_index, frame_index + horizon))["action"]
    gt = torch.stack([torch.as_tensor(row) for row in rows]).float()
    err = (pred[:horizon, : gt.shape[1]] - gt).abs().mean(0)
    return {"mae": float(err.mean().item()), "horizon": horizon, "per_dim": err.tolist()}


def _iter_coverage_files(coverage_dir: Path, task_ids: list[int] | None) -> list[Path]:
    paths = sorted(coverage_dir.glob("task_*/blind_spots.json"))
    if task_ids is None:
        return paths
    wanted = {f"task_{tid}" for tid in task_ids}
    return [path for path in paths if path.parent.name in wanted]


def probe_task(
    coverage: dict,
    dataset: LeRobotDataset,
    policy,
    preprocess,
    postprocess,
    args: argparse.Namespace,
) -> dict:
    task_index = int(coverage["task_index"])
    out_dir = args.output_dir / f"task_{task_index}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_hist = np.asarray(coverage["train_hist"])
    edges = [np.asarray(edge) for edge in coverage["edges"]]
    spots = coverage["blind_spots"]
    if args.max_bins is not None:
        spots = spots[: args.max_bins]
    centroids = np.asarray([spot["centroid"] for spot in spots], dtype=np.float32).reshape(-1, 3)

    probe_xy: list[list[float]] = []
    probe_scores: list[float] = []
    bin_results = []

    for spot in spots:
        anchor_idx = int(spot["anchor"]["frame_index"])
        anchor_frame = dict(dataset[anchor_idx])
        if int(anchor_frame.get("task_index", task_index)) != task_index:
            raise ValueError(
                f"Anchor frame {anchor_idx} has task_index={anchor_frame.get('task_index')} "
                f"but coverage task_index={task_index}"
            )

        anchor_pred = _predict_chunk(policy, preprocess, postprocess, anchor_frame, seed=0)
        seed_deltas = [
            _mean_abs_diff(
                anchor_pred,
                _predict_chunk(policy, preprocess, postprocess, anchor_frame, seed=seed),
            )
            for seed in range(1, args.seeds + 1)
        ]
        seed_variation_mean = float(np.mean(seed_deltas)) if seed_deltas else 0.0
        demo_mae = _anchor_mae_vs_demo(dataset, anchor_idx, anchor_pred)

        probe_rows = []
        for step, state in enumerate(spot["probe_states"], start=1):
            state_np = np.asarray(state, dtype=np.float32)
            probe_frame = _swap_state(anchor_frame, state_np)
            pred = _predict_chunk(policy, preprocess, postprocess, probe_frame, seed=0)
            delta = _mean_abs_diff(pred, anchor_pred)
            probe_seed_deltas = [
                _mean_abs_diff(
                    pred,
                    _predict_chunk(policy, preprocess, postprocess, probe_frame, seed=seed),
                )
                for seed in range(1, args.seeds + 1)
            ]
            probe_seed_mean = float(np.mean(probe_seed_deltas)) if probe_seed_deltas else 0.0
            ratio = delta / seed_variation_mean if seed_variation_mean > 1e-8 else float("inf")
            in_blind_bin = step == len(spot["probe_states"])
            flag = bool(delta > 2.0 * seed_variation_mean and in_blind_bin)
            row = {
                "step": step,
                "alpha": step / len(spot["probe_states"]),
                "state": state_np.tolist(),
                "eef": state_np[:3].tolist(),
                "action_delta_vs_anchor": delta,
                "seed_variation": probe_seed_deltas,
                "seed_variation_mean": probe_seed_mean,
                "seed_variation_ratio": ratio,
                "in_blind_bin": in_blind_bin,
                "flow_matching_flag": flag,
            }
            probe_rows.append(row)
            probe_xy.append(state_np[:2].tolist())
            probe_scores.append(delta)

        bin_results.append(
            {
                "bin_ijk": spot["bin_ijk"],
                "centroid": spot["centroid"],
                "anchor": {
                    **spot["anchor"],
                    "mae_vs_demo": demo_mae,
                    "seed_variation": seed_deltas,
                    "seed_variation_mean": seed_variation_mean,
                },
                "probes": probe_rows,
            }
        )

    overlay = plot_probe_overlay(
        train_hist,
        edges,
        centroids,
        np.asarray(probe_xy, dtype=np.float32).reshape(-1, 2),
        np.asarray(probe_scores, dtype=np.float32),
        out_dir / "probe_map_xy.png",
        title=f"task {task_index}: {coverage.get('task', '')}",
    )
    payload = {
        "task_index": task_index,
        "env_task_id": coverage.get("env_task_id"),
        "task": coverage.get("task"),
        "note": COVERAGE_NOTE,
        "limitation": PROBE_LIMITATION,
        "n_flags": sum(
            int(row["flow_matching_flag"]) for item in bin_results for row in item["probes"]
        ),
        "overlay": str(overlay),
        "blind_spots": bin_results,
    }
    write_json(out_dir / "probe_results.json", payload)
    print(
        f"task {task_index}: {len(bin_results)} bins, "
        f"{payload['n_flags']} flow-matching flags -> {out_dir}"
    )
    return {
        "task_index": task_index,
        "task": coverage.get("task"),
        "n_bins": len(bin_results),
        "n_flags": payload["n_flags"],
        "output_dir": str(out_dir),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = _iter_coverage_files(args.coverage_dir, args.task_ids)
    if not files:
        raise FileNotFoundError(f"No blind_spots.json files under {args.coverage_dir}")

    device = torch.device(args.device)
    policy, preprocess, postprocess = _load_policy(args.model_path, device)
    dataset = LeRobotDataset(args.train_dataset, video_backend=args.video_backend)

    summaries = []
    for path in files:
        coverage = json.loads(path.read_text())
        summaries.append(probe_task(coverage, dataset, policy, preprocess, postprocess, args))

    write_json(
        args.output_dir / "summary.json",
        {
            "note": COVERAGE_NOTE,
            "limitation": PROBE_LIMITATION,
            "model_path": str(args.model_path),
            "n_flags_total": sum(item["n_flags"] for item in summaries),
            "tasks": summaries,
        },
    )
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
