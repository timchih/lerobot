"""Nearest-demo expert replay at sim probe poses (counterfactual dataset smoke test).

For each probe point from ``probe_blindspots_sim`` results:
  1. Build (or load) a catalog of object XY at each LIBERO init_state.
  2. Pick the training episode whose init object XY is closest to the probe.
  3. Place the scene like the sim probe (including distractor shifts).
  4. Open-loop replay the demo actions; visualize alignment vs a rigid EEF XY warp.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from blindspot_expert_utils import (
    apply_probe_placement,
    build_episode_matches,
    build_init_catalog,
    load_episode_frames,
    load_init_catalog,
    nearest_episode_to_probe,
    plot_expert_alignment,
    replay_demo_actions,
    save_init_catalog,
    warp_demo_eef_xy,
)
from blindspot_sim_utils import SimProbeEnv
from blindspot_utils import dataset_index_from_meta, write_json
from lerobot.utils.io_utils import write_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-results",
        type=Path,
        required=True,
        help="probe_results.json from probe_blindspots_sim (one task).",
    )
    parser.add_argument("--train-dataset", default="lerobot/libero")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/blindspot/counterfactual_expert"))
    parser.add_argument("--train-dataset-root", type=Path, default=None)
    parser.add_argument("--n-inits", type=int, default=50)
    parser.add_argument("--roles", nargs="*", default=None, help="Probe roles to run (default: all in JSON).")
    parser.add_argument("--max-replay-steps", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--control-mode", default="relative", choices=["relative", "absolute"])
    return parser.parse_args()


def _task_index_from_probe(probe: dict, dataset: LeRobotDataset) -> int:
    if "task_index" in probe:
        return int(probe["task_index"])
    task_text = str(probe.get("task") or probe.get("task_description") or "")
    return dataset_index_from_meta(dataset.meta, task_text)


def _probe_rows(probe: dict, roles: list[str] | None) -> list[dict[str, Any]]:
    rows = list(probe.get("probes") or [])
    if roles:
        wanted = set(roles)
        rows = [row for row in rows if row.get("role") in wanted]
    return rows


def main() -> None:
    args = parse_args()
    probe_payload = json.loads(args.probe_results.read_text())
    env_task_id = int(probe_payload["env_task_id"])
    init_state_id = int(probe_payload.get("init_state_id", 0))

    ds_kwargs: dict[str, Any] = {"repo_id": args.train_dataset, "download_videos": False}
    if args.train_dataset_root is not None:
        ds_kwargs["root"] = str(args.train_dataset_root)
    dataset = LeRobotDataset(**ds_kwargs)
    task_index = _task_index_from_probe(probe_payload, dataset)

    out_dir = args.output_dir / f"task_{task_index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = out_dir / f"init_catalog_env{env_task_id}.json"

    sim_env = SimProbeEnv(env_task_id=env_task_id, control_mode=args.control_mode)
    try:
        if catalog_path.exists() and not args.refresh_catalog:
            catalog = load_init_catalog(catalog_path)
        else:
            catalog = build_init_catalog(sim_env, n_inits=args.n_inits)
            save_init_catalog(catalog_path, catalog, env_task_id)

        episode_matches = build_episode_matches(dataset, task_index, catalog)
        if not episode_matches:
            raise RuntimeError(f"No episodes matched for task_index={task_index}")

        rows_out: list[dict[str, Any]] = []
        for probe_row in _probe_rows(probe_payload, args.roles):
            role = str(probe_row["role"])
            probe_xyz = np.asarray(probe_row["object_xyz"], dtype=np.float32)
            probe_xy = probe_xyz[:2]
            placement = probe_row.get("placement") or {}
            obstacle_adjustments = placement.get("obstacle_adjustments")

            match = nearest_episode_to_probe(episode_matches, probe_xy)
            frames = load_episode_frames(dataset, task_index, match.episode_index)
            actions = [row["action"] for row in frames]
            demo_states = np.stack([row["state"] for row in frames], axis=0)
            demo_eef_xy = demo_states[:, :3]

            source_xyz = match.object_xyz
            delta_xy = probe_xy - match.object_xy
            warped_eef_xy = warp_demo_eef_xy(demo_eef_xy, delta_xy)

            apply_probe_placement(
                sim_env,
                probe_xyz,
                init_state_id=init_state_id,
                obstacle_adjustments=obstacle_adjustments,
            )
            replay = replay_demo_actions(sim_env, actions, max_steps=args.max_replay_steps)
            replay_eef_xy = replay["eef_path"][:, :3]

            plot_path = out_dir / f"alignment_{role}.png"
            plot_expert_alignment(
                plot_path,
                probe_xy,
                match.object_xy,
                demo_eef_xy,
                warped_eef_xy,
                replay_eef_xy,
                title=f"task {task_index} {role}: nearest ep {match.episode_index}",
            )

            video_path = out_dir / f"expert_replay_{role}.mp4"
            if len(replay["frames"]) > 1:
                write_video(str(video_path), replay["frames"], fps=args.video_fps)
            else:
                video_path = None

            row = {
                "role": role,
                "probe_object_xyz": probe_xyz.tolist(),
                "probe_object_xy": probe_xy.tolist(),
                "nearest_episode_index": match.episode_index,
                "nearest_init_state_id": match.init_state_id,
                "nearest_object_xy": match.object_xy.tolist(),
                "nearest_object_xyz": match.object_xyz.tolist(),
                "object_xy_gap": float(np.linalg.norm(match.object_xy - probe_xy)),
                "object_warp_delta_xy": delta_xy.tolist(),
                "eef_match_error_at_init": match.eef_match_error,
                "demo_length": len(actions),
                "warped_eef_final_xy": warped_eef_xy[-1, :2].tolist(),
                "replay_eef_final_xy": replay_eef_xy[-1, :2].tolist(),
                "warped_vs_replay_final_xy_error": float(
                    np.linalg.norm(warped_eef_xy[-1, :2] - replay_eef_xy[-1, :2])
                ),
                "alignment_plot": str(plot_path),
                "replay": {
                    k: v
                    for k, v in replay.items()
                    if k not in {"eef_path", "object_path", "frames"}
                },
                "video": str(video_path) if video_path is not None else None,
            }
            rows_out.append(row)
            print(
                f"{role}: ep={match.episode_index} init={match.init_state_id} "
                f"obj_gap={row['object_xy_gap']:.3f}m success={row['replay']['is_success']} "
                f"min_grip={row['replay']['min_gripper_object_distance']}"
            )

        payload = {
            "task_index": task_index,
            "env_task_id": env_task_id,
            "init_state_id": init_state_id,
            "probe_results": str(args.probe_results),
            "train_dataset": args.train_dataset,
            "catalog": str(catalog_path),
            "n_episode_matches": len(episode_matches),
            "note": (
                "Nearest-demo replay is a cheap counterfactual: actions are not re-planned. "
                "Orange path = demo EEF rigidly shifted by object Δxy (reference only)."
            ),
            "runs": rows_out,
        }
        write_json(out_dir / "counterfactual_expert.json", payload)
        print(f"Wrote {out_dir / 'counterfactual_expert.json'}")
    finally:
        sim_env.close()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
