"""Sim-positioned pi0.5 probes: move the target object along a p1→p5 blind trail.

One furthest blind point per task. Trail: p1 (support boundary) through p5 (furthest).
Open-loop action metrics at all five points; optional closed-loop eval rollouts at p1–p5.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.envs.factory import make_env_config, make_env_pre_post_processors
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS
from lerobot.lerobot_types import PolicyAction
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.relative_action_processor import bind_relative_anchor
from lerobot.utils.io_utils import write_video

from blindspot_sim_utils import (
    SimProbeEnv,
    build_policy_obs,
    interpolate_object_probes,
    plot_sim_probe_overlay,
    policy_view_image,
    save_probe_frame,
)
from blindspot_utils import (
    COVERAGE_NOTE,
    SIM_PROBE_NOTE,
    bin_centroids,
    dataset_index_to_env_task_id,
    env_task_id_from_task_text,
    find_blind_bins,
    write_json,
)

WS = Path(os.environ.get("WS", "/vol/gpudata/tq422-vla"))
DEFAULT_MODEL = WS / "models" / "pi05_libero_finetuned_v044"

TRAIL_ROLLOUT_VIDEO = {
    "p1_boundary": "p1_rollout",
    "p2": "p2_rollout",
    "p3": "p3_rollout",
    "p4": "p4_rollout",
    "p5_far": "p5_rollout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-dir", type=Path, default=Path("outputs/blindspot/coverage"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/blindspot/probes_sim"))
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--n-probes", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="Dataset task_index values (20-29). Default: every task_* dir in coverage-dir.",
    )
    parser.add_argument(
        "--run-eval-rollout",
        action="store_true",
        help="Run closed-loop eval rollouts at p1–p5 (lerobot-eval stack).",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Max env steps per rollout. Defaults to 280 for libero_object when --run-eval-rollout.",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save agentview mp4 per rollout (default: on when --run-eval-rollout).",
    )
    parser.add_argument("--no-save-videos", dest="save_videos", action="store_false")
    parser.set_defaults(save_videos=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--save-frames", action="store_true", default=True)
    parser.add_argument("--no-save-frames", dest="save_frames", action="store_false")
    parser.add_argument(
        "--skip-init-rollout",
        action="store_true",
        help="Skip eval rollout on the in-distribution init object position.",
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


def _predict_chunk(policy, preprocess, postprocess, raw_obs, task_desc, env_preprocessor, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    policy.reset()
    preprocess.reset()
    postprocess.reset()
    batch = build_policy_obs(raw_obs, task_desc, env_preprocessor, preprocess)
    with torch.inference_mode():
        action = policy.predict_action_chunk(batch)
        action = postprocess(action)
    return action[0].float().cpu()


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def _eef_from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float32).reshape(-1)[:3]


def _furthest_centroid(coverage: dict) -> np.ndarray:
    if "furthest_point" in coverage:
        return np.asarray(coverage["furthest_point"]["centroid"], dtype=np.float32)
    spots = coverage.get("blind_spots")
    if isinstance(spots, list) and spots:
        first = spots[0]
        if "centroid" in first:
            return np.asarray(first["centroid"], dtype=np.float32)
        if "furthest_point" in first:
            return np.asarray(first["furthest_point"]["centroid"], dtype=np.float32)
    raise KeyError("Coverage JSON missing furthest_point (re-run map_state_coverage.py)")


def _iter_coverage_files(coverage_dir: Path, task_ids: list[int] | None) -> list[Path]:
    paths = sorted(coverage_dir.glob("task_*/blind_spots.json"))
    if task_ids is None:
        return paths
    wanted = {f"task_{tid}" for tid in task_ids}
    return [path for path in paths if path.parent.name in wanted]


def _run_eval_rollout(
    sim_env: SimProbeEnv,
    policy,
    preprocess,
    postprocess,
    env_preprocessor,
    env_postprocessor,
    n_steps: int,
    videos_dir: Path | None,
    video_name: str,
    video_fps: int,
) -> dict[str, Any]:
    """Closed-loop rollout using the same policy loop as lerobot-eval."""
    from lerobot.utils.constants import ACTION

    policy.reset()
    preprocess.reset()
    postprocess.reset()
    bind_relative_anchor(policy, preprocess)
    sim_env._clear_episode_flags()
    obs = sim_env.env._format_raw_obs(sim_env.env._env.env._get_observations())
    reached = False
    success = False
    reward_sum = 0.0
    min_grip = float("inf")
    steps_taken = 0
    frames: list[np.ndarray] = [policy_view_image(obs, camera="image")]
    for _ in range(n_steps):
        if sim_env._task_env().done:
            break
        batch = build_policy_obs(obs, sim_env.task_description, env_preprocessor, preprocess)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = postprocess(action)
        transition = env_postprocessor({ACTION: action})
        step_action = transition[ACTION]
        if torch.is_tensor(step_action):
            step_action = step_action.detach().cpu().numpy()
        try:
            obs, reward, terminated, _truncated, info = sim_env.step(np.asarray(step_action).reshape(-1))
        except ValueError as exc:
            if "terminated episode" not in str(exc):
                raise
            break
        steps_taken += 1
        frames.append(policy_view_image(obs, camera="image"))
        reward_sum += float(reward)
        obj = sim_env.current_object_xyz()
        eef = _eef_from_obs(obs)
        min_grip = min(min_grip, float(np.linalg.norm(eef - obj)))
        reached = reached or min_grip < 0.05
        success = bool(info.get("is_success", False)) or success
        if terminated:
            break

    video_path = None
    if videos_dir is not None and len(frames) > 1:
        videos_dir.mkdir(parents=True, exist_ok=True)
        video_path = videos_dir / f"{video_name}.mp4"
        write_video(video_path, frames, fps=video_fps)

    return {
        "n_steps_requested": n_steps,
        "n_steps_taken": steps_taken,
        "reward_sum": reward_sum,
        "is_success": success,
        "gripper_reached_object": reached,
        "min_gripper_object_distance": min_grip if min_grip < float("inf") else None,
        "video": str(video_path) if video_path is not None else None,
    }


def probe_task(
    coverage: dict,
    policy,
    preprocess,
    postprocess,
    args: argparse.Namespace,
) -> dict:
    task_index = int(coverage["task_index"])
    task_text = str(coverage.get("task") or "")
    if task_text:
        env_task_id = env_task_id_from_task_text(task_text)
    else:
        env_task_id = int(coverage.get("env_task_id", dataset_index_to_env_task_id(task_index)))
    out_dir = args.output_dir / f"task_{task_index}"
    frames_dir = out_dir / "frames"
    videos_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = make_env_config("libero", task="libero_object", task_ids=[env_task_id])
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy.config)

    train_hist = np.asarray(coverage["train_hist"])
    eval_hist = np.asarray(coverage["eval_hist"])
    edges = [np.asarray(edge) for edge in coverage["edges"]]
    range_low = np.asarray(coverage["range_low"], dtype=np.float32)
    range_high = np.asarray(coverage["range_high"], dtype=np.float32)
    blind_coords = find_blind_bins(train_hist, eval_hist)
    all_blind_centroids = bin_centroids(blind_coords, edges)

    furthest_centroid = _furthest_centroid(coverage)

    rollout_steps = args.rollout_steps
    if rollout_steps is None:
        rollout_steps = TASK_SUITE_MAX_STEPS.get("libero_object", 280) if args.run_eval_rollout else 0
    save_videos = args.run_eval_rollout if args.save_videos is None else args.save_videos

    sim_env = SimProbeEnv(env_task_id=env_task_id)
    baseline_obs = sim_env.reset(init_state_id=args.init_state_id)
    init_xyz = sim_env.init_xyz
    assert init_xyz is not None
    if args.save_frames:
        save_probe_frame(frames_dir / "baseline.png", baseline_obs)

    baseline_pred = _predict_chunk(
        policy, preprocess, postprocess, baseline_obs, sim_env.task_description, env_preprocessor, seed=0
    )
    seed_deltas = [
        _mean_abs_diff(
            baseline_pred,
            _predict_chunk(
                policy,
                preprocess,
                postprocess,
                baseline_obs,
                sim_env.task_description,
                env_preprocessor,
                seed=seed,
            ),
        )
        for seed in range(1, args.seeds + 1)
    ]
    seed_variation_mean = float(np.mean(seed_deltas)) if seed_deltas else 0.0

    baseline_rollout = None
    if args.run_eval_rollout:
        baseline_rollout = _run_eval_rollout(
            sim_env,
            policy,
            preprocess,
            postprocess,
            env_preprocessor,
            env_postprocessor,
            rollout_steps,
            videos_dir if save_videos else None,
            "baseline_rollout",
            args.video_fps,
        )

    if (
        furthest_centroid[0] < range_low[0]
        or furthest_centroid[0] > range_high[0]
        or furthest_centroid[1] < range_low[1]
        or furthest_centroid[1] > range_high[1]
    ):
        raise ValueError(f"task {task_index}: furthest point out of workspace range")

    sim_env.reset(init_state_id=args.init_state_id)
    probe_positions = interpolate_object_probes(
        init_xyz,
        furthest_centroid[:2],
        n_probes=args.n_probes,
        train_hist=train_hist,
        eval_hist=eval_hist,
        edges=edges,
        range_low=range_low,
        range_high=range_high,
        blind_centroids=all_blind_centroids,
        sim_env=sim_env,
    )
    probe_rows = []
    probe_xy: list[list[float]] = []
    probe_scores: list[float] = []
    for step, item in enumerate(probe_positions, start=1):
        if len(item) == 3:
            xyz, role, placement_meta = item
        else:
            xyz, role = item
            placement_meta = {}
        if placement_meta.get("contact_overlap") or placement_meta.get("in_workspace") is False:
            print(
                f"task {task_index} skip {role}: "
                f"contact_overlap={placement_meta.get('contact_overlap')} "
                f"in_workspace={placement_meta.get('in_workspace')} "
                f"geom_gap={placement_meta.get('geom_gap')} "
                f"closest={placement_meta.get('closest_obstacle')}"
            )
            continue
        obs = sim_env.place_object(
            xyz,
            obstacle_adjustments=placement_meta.get("obstacle_adjustments"),
        )
        pred = _predict_chunk(
            policy, preprocess, postprocess, obs, sim_env.task_description, env_preprocessor, seed=0
        )
        delta = _mean_abs_diff(pred, baseline_pred)
        probe_seed_deltas = [
            _mean_abs_diff(
                pred,
                _predict_chunk(
                    policy,
                    preprocess,
                    postprocess,
                    obs,
                    sim_env.task_description,
                    env_preprocessor,
                    seed=seed,
                ),
            )
            for seed in range(1, args.seeds + 1)
        ]
        probe_seed_mean = float(np.mean(probe_seed_deltas)) if probe_seed_deltas else 0.0
        ratio = delta / seed_variation_mean if seed_variation_mean > 1e-8 else float("inf")
        in_blind_bin = role not in {"p1_boundary", "init", "boundary"}
        flag = bool(delta > 2.0 * seed_variation_mean and role == "p5_far")
        placed = sim_env.current_object_xyz()
        eef = _eef_from_obs(obs)
        row = {
            "step": step,
            "role": role,
            "placement": placement_meta,
            "object_xyz": placed.tolist(),
            "eef": eef.tolist(),
            "object_shift_distance": float(np.linalg.norm(placed - init_xyz)),
            "gripper_object_distance": float(np.linalg.norm(eef - placed)),
            "action_delta_vs_baseline": delta,
            "seed_variation": probe_seed_deltas,
            "seed_variation_mean": probe_seed_mean,
            "seed_variation_ratio": ratio,
            "in_blind_bin": in_blind_bin,
            "flow_matching_flag": flag,
        }
        if args.save_frames:
            frame_path = frames_dir / f"{role}.png"
            save_probe_frame(frame_path, obs)
            row["frame"] = str(frame_path)

        if args.run_eval_rollout and role in TRAIL_ROLLOUT_VIDEO:
            row["eval_rollout"] = _run_eval_rollout(
                sim_env,
                policy,
                preprocess,
                postprocess,
                env_preprocessor,
                env_postprocessor,
                rollout_steps,
                videos_dir if save_videos else None,
                TRAIL_ROLLOUT_VIDEO[role],
                args.video_fps,
            )

        probe_rows.append(row)
        probe_xy.append(placed[:2].tolist())
        probe_scores.append(delta)

    overlay = plot_sim_probe_overlay(
        train_hist,
        edges,
        furthest_centroid[:2].reshape(1, 2),
        ["furthest"],
        np.asarray(probe_xy, dtype=np.float32).reshape(-1, 2),
        np.asarray(probe_scores, dtype=np.float32),
        out_dir / "probe_map_xy.png",
        title=f"task {task_index}: {coverage.get('task', '')}",
    )
    task_description = sim_env.task_description
    target_joint = sim_env.joint_name
    sim_env.close()
    n_flags = sum(int(row["flow_matching_flag"]) for row in probe_rows)
    furthest_meta = coverage.get("furthest_point", {})
    payload = {
        "task_index": task_index,
        "env_task_id": env_task_id,
        "task": coverage.get("task"),
        "task_description": task_description,
        "note": COVERAGE_NOTE,
        "sim_note": SIM_PROBE_NOTE,
        "init_state_id": args.init_state_id,
        "init_object_xyz": init_xyz.tolist(),
        "target_joint": target_joint,
        "seed_variation_mean": seed_variation_mean,
        "probe_schedule": "anchor p5 (furthest) → p1 boundary → p2–p4 evenly spaced; eval rollouts at p1–p5",
        "run_eval_rollout": args.run_eval_rollout,
        "rollout_steps": rollout_steps,
        "baseline_rollout": baseline_rollout,
        "n_flags": n_flags,
        "overlay": str(overlay),
        "furthest_point": {
            "centroid": furthest_centroid.tolist(),
            "nearest_support_distance": furthest_meta.get("nearest_support_distance"),
            "object_shift_distance": float(np.linalg.norm(furthest_centroid[:2] - init_xyz[:2])),
        },
        "probes": probe_rows,
    }
    write_json(out_dir / "probe_results.json", payload)
    print(f"task {task_index}: {len(probe_rows)} trail points, {n_flags} flags -> {out_dir}")
    return {
        "task_index": task_index,
        "task": coverage.get("task"),
        "n_probes": len(probe_rows),
        "n_flags": n_flags,
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

    summaries = []
    for path in files:
        coverage = json.loads(path.read_text())
        summaries.append(probe_task(coverage, policy, preprocess, postprocess, args))

    write_json(
        args.output_dir / "summary.json",
        {
            "note": COVERAGE_NOTE,
            "sim_note": SIM_PROBE_NOTE,
            "model_path": str(args.model_path),
            "n_flags_total": sum(item["n_flags"] for item in summaries),
            "tasks": summaries,
        },
    )
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
