"""Nearest-demo expert replay for counterfactual blind-spot poses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from blindspot_sim_utils import SimProbeEnv, policy_view_image
from lerobot.utils.io_utils import write_video


@dataclass(frozen=True)
class InitCatalogEntry:
    init_state_id: int
    object_xy: np.ndarray
    object_xyz: np.ndarray
    eef_xyz: np.ndarray


@dataclass(frozen=True)
class EpisodeMatch:
    episode_index: int
    init_state_id: int
    object_xy: np.ndarray
    object_xyz: np.ndarray
    eef_match_error: float


def build_init_catalog(sim_env: SimProbeEnv, n_inits: int = 50) -> list[InitCatalogEntry]:
    """Object and EEF pose at reset for each LIBERO init_state_id."""
    entries: list[InitCatalogEntry] = []
    for init_id in range(n_inits):
        sim_env.reset(init_state_id=init_id)
        obj = sim_env.current_object_xyz()
        eef = sim_env.current_eef_xyz()
        entries.append(
            InitCatalogEntry(
                init_state_id=init_id,
                object_xy=obj[:2].astype(np.float32),
                object_xyz=obj.astype(np.float32),
                eef_xyz=eef.astype(np.float32),
            )
        )
    return entries


def save_init_catalog(path: Path, entries: list[InitCatalogEntry], env_task_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_task_id": env_task_id,
        "entries": [
            {
                "init_state_id": e.init_state_id,
                "object_xy": e.object_xy.tolist(),
                "object_xyz": e.object_xyz.tolist(),
                "eef_xyz": e.eef_xyz.tolist(),
            }
            for e in entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def load_init_catalog(path: Path) -> list[InitCatalogEntry]:
    payload = json.loads(path.read_text())
    return [
        InitCatalogEntry(
            init_state_id=int(row["init_state_id"]),
            object_xy=np.asarray(row["object_xy"], dtype=np.float32),
            object_xyz=np.asarray(row["object_xyz"], dtype=np.float32),
            eef_xyz=np.asarray(row["eef_xyz"], dtype=np.float32),
        )
        for row in payload["entries"]
    ]


def nearest_init_by_object_xy(catalog: list[InitCatalogEntry], probe_xy: np.ndarray) -> InitCatalogEntry:
    probe_xy = np.asarray(probe_xy, dtype=np.float32).reshape(2)
    return min(catalog, key=lambda e: float(np.linalg.norm(e.object_xy - probe_xy)))


def match_init_by_eef(catalog: list[InitCatalogEntry], demo_eef_xyz: np.ndarray) -> InitCatalogEntry:
    demo_eef_xyz = np.asarray(demo_eef_xyz, dtype=np.float32).reshape(3)
    return min(catalog, key=lambda e: float(np.linalg.norm(e.eef_xyz - demo_eef_xyz)))


def _task_episode_table(dataset: Any, task_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    """One pass over the dataset for a task: episode_index, frame_index, state, action columns."""
    hf = dataset.hf_dataset.with_format(
        "numpy",
        columns=["task_index", "episode_index", "frame_index", "action", "observation.state"],
    )
    ti = np.asarray(hf["task_index"]).reshape(-1)
    mask = ti == task_index
    ei = np.asarray(hf["episode_index"])[mask].reshape(-1)
    fi = np.asarray(hf["frame_index"])[mask].reshape(-1)
    states = hf["observation.state"]
    actions = hf["action"]
    local_idx = np.where(mask)[0]
    return ei, fi, local_idx, states, actions


def list_task_episodes(dataset: Any, task_index: int) -> list[int]:
    ei, _, _, _, _ = _task_episode_table(dataset, task_index)
    return sorted({int(x) for x in ei})


def load_episode_frames(dataset: Any, task_index: int, episode_index: int) -> list[dict[str, Any]]:
    ei, fi, local_idx, states, actions = _task_episode_table(dataset, task_index)
    ep_mask = ei == episode_index
    order = np.argsort(fi[ep_mask])
    ep_fi = fi[ep_mask][order]
    ep_local = local_idx[ep_mask][order]
    rows: list[dict[str, Any]] = []
    for frame_index, idx in zip(ep_fi, ep_local, strict=True):
        rows.append(
            {
                "frame_index": int(frame_index),
                "action": np.asarray(actions[idx], dtype=np.float32).reshape(-1),
                "state": np.asarray(states[idx], dtype=np.float32).reshape(-1),
            }
        )
    return rows


def build_episode_matches(
    dataset: Any,
    task_index: int,
    catalog: list[InitCatalogEntry],
) -> list[EpisodeMatch]:
    """Map each training episode to the init_state whose reset EEF matches frame 0."""
    ei, fi, local_idx, states, _actions = _task_episode_table(dataset, task_index)
    frame0 = fi == 0
    matches: list[EpisodeMatch] = []
    for episode_index in sorted({int(x) for x in ei}):
        pick = frame0 & (ei == episode_index)
        if not np.any(pick):
            continue
        idx = int(local_idx[pick][0])
        eef0 = np.asarray(states[idx], dtype=np.float32).reshape(-1)[:3]
        init_entry = match_init_by_eef(catalog, eef0)
        matches.append(
            EpisodeMatch(
                episode_index=episode_index,
                init_state_id=init_entry.init_state_id,
                object_xy=init_entry.object_xy.copy(),
                object_xyz=init_entry.object_xyz.copy(),
                eef_match_error=float(np.linalg.norm(init_entry.eef_xyz - eef0)),
            )
        )
    return matches


def nearest_episode_to_probe(
    episode_matches: list[EpisodeMatch],
    probe_xy: np.ndarray,
) -> EpisodeMatch:
    probe_xy = np.asarray(probe_xy, dtype=np.float32).reshape(2)
    return min(episode_matches, key=lambda m: float(np.linalg.norm(m.object_xy - probe_xy)))


def warp_demo_eef_xy(states: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    """Rigid XY shift of recorded EEF positions (visual reference only)."""
    delta_xy = np.asarray(delta_xy, dtype=np.float32).reshape(2)
    warped = states.copy()
    warped[:, 0] += delta_xy[0]
    warped[:, 1] += delta_xy[1]
    return warped


def apply_probe_placement(
    sim_env: SimProbeEnv,
    probe_xyz: np.ndarray,
    init_state_id: int,
    obstacle_adjustments: list[dict[str, Any]] | None,
) -> np.ndarray:
    sim_env.reset(init_state_id=init_state_id)
    obs = sim_env.place_object(
        np.asarray(probe_xyz, dtype=np.float32),
        obstacle_adjustments=obstacle_adjustments,
    )
    return obs


def replay_demo_actions(
    sim_env: SimProbeEnv,
    actions: list[np.ndarray],
    max_steps: int | None = None,
) -> dict[str, Any]:
    sim_env._clear_episode_flags()
    eef_path: list[np.ndarray] = []
    obj_path: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    obs = sim_env.env._format_raw_obs(sim_env.env._env.env._get_observations())
    frames.append(policy_view_image(obs, camera="image"))
    eef_path.append(sim_env.current_eef_xyz())
    obj_path.append(sim_env.current_object_xyz())

    success = False
    reward_sum = 0.0
    min_grip = float("inf")
    steps = 0
    limit = len(actions) if max_steps is None else min(len(actions), max_steps)
    for step in range(limit):
        if sim_env._task_env().done:
            break
        action = np.asarray(actions[step], dtype=np.float32).reshape(-1)
        obs, reward, terminated, _truncated, info = sim_env.step(action)
        steps += 1
        frames.append(policy_view_image(obs, camera="image"))
        eef = sim_env.current_eef_xyz()
        obj = sim_env.current_object_xyz()
        eef_path.append(eef)
        obj_path.append(obj)
        min_grip = min(min_grip, float(np.linalg.norm(eef - obj)))
        reward_sum += float(reward)
        success = bool(info.get("is_success", False)) or success
        if terminated:
            break

    return {
        "n_steps_taken": steps,
        "reward_sum": reward_sum,
        "is_success": success,
        "gripper_reached_object": min_grip < 0.05,
        "min_gripper_object_distance": min_grip if min_grip < float("inf") else None,
        "eef_path": np.stack(eef_path, axis=0),
        "object_path": np.stack(obj_path, axis=0),
        "frames": frames,
    }


def plot_expert_alignment(
    out_path: Path,
    probe_xy: np.ndarray,
    source_object_xy: np.ndarray,
    demo_eef_xy: np.ndarray,
    warped_eef_xy: np.ndarray,
    replay_eef_xy: np.ndarray,
    title: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(*source_object_xy, c="tab:blue", s=120, marker="s", label="nearest demo object @ init", zorder=5)
    ax.scatter(*probe_xy, c="crimson", s=120, marker="*", label="probe object", zorder=6)
    ax.plot(demo_eef_xy[:, 0], demo_eef_xy[:, 1], color="tab:blue", alpha=0.35, linewidth=1.5, label="demo EEF (orig)")
    ax.plot(warped_eef_xy[:, 0], warped_eef_xy[:, 1], color="tab:orange", linewidth=2.0, label="demo EEF + object Δxy")
    ax.plot(replay_eef_xy[:, 0], replay_eef_xy[:, 1], color="tab:green", linewidth=2.0, label="replay EEF (sim)")
    ax.scatter(replay_eef_xy[-1, 0], replay_eef_xy[-1, 1], c="tab:green", s=40, marker="o")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
