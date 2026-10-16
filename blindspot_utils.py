"""Shared helpers for LIBERO-object coverage maps and state-swap probes.

Blind spots are defined relative to `lerobot/libero` demonstrations and one
eval recording run. pi0.5 may still have seen similar states during
pretraining or other finetuning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LIBERO_OBJECT_SUITE = "libero_object"
LIBERO_OBJECT_DATASET_TASK_OFFSET = 20
LIBERO_OBJECT_ENV_TASK_IDS = range(10)
LIBERO_OBJECT_DATASET_TASK_INDICES = range(20, 30)

COVERAGE_NOTE = (
    "Blind spots are relative to lerobot/libero demo coverage and this eval run "
    "only. pi0.5 may have seen broader data during pretraining/finetuning."
)
PROBE_LIMITATION = (
    "State-swap probes hold vision and language fixed while only proprioception "
    "changes. Results are a state-conditioned action sensitivity test, not "
    "physical execution failure at the perturbed pose."
)
SIM_PROBE_NOTE = (
    "Sim-positioned probes move the task target object in LIBERO while keeping "
    "task text, robot init pose, and distractors fixed. Images and proprioception "
    "are rendered from the new scene, unlike state-swap probes."
)
SELECTION_SCALES = ("min", "moderate", "far")
SCALE_MARKERS = {
    "min": ("o", "lime"),
    "moderate": ("s", "gold"),
    "far": ("x", "crimson"),
    "boundary": ("o", "lime"),
    "deep": ("x", "crimson"),
}
DIST_SUPPORT_CMAP = "Greens"

TRAIL_PLOT_LABELS = {
    "p1_boundary": "p1",
    "p2": "p2",
    "p3": "p3",
    "p4": "p4",
    "p5_far": "p5",
}


def trail_plot_label(role: str) -> str:
    return TRAIL_PLOT_LABELS.get(role, role)

_EEF_KEYS = (
    "robot_state/eef/pos",
    "observation.state.eef_pos",
    "observation.state.eef.pos",
)


LIBERO_OBJECT_ENV_LANGUAGES = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    3: "pick up the bbq sauce and place it in the basket",
    4: "pick up the ketchup and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
    7: "pick up the milk and place it in the basket",
    8: "pick up the chocolate pudding and place it in the basket",
    9: "pick up the orange juice and place it in the basket",
}
LIBERO_OBJECT_SLUG_TO_ENV = {
    "alphabet_soup": 0,
    "cream_cheese": 1,
    "salad_dressing": 2,
    "bbq_sauce": 3,
    "ketchup": 4,
    "tomato_sauce": 5,
    "butter": 6,
    "milk": 7,
    "chocolate_pudding": 8,
    "orange_juice": 9,
}


def env_task_id_from_task_text(task_text: str) -> int:
    """Map a dataset or language string to the LIBERO-object env task id."""
    name = task_text.strip().lower().replace(" ", "_")
    matches = [env_id for slug, env_id in LIBERO_OBJECT_SLUG_TO_ENV.items() if slug in name]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely map task text to env id: {task_text!r}")
    return matches[0]


def dataset_index_from_meta(meta: Any, task_text: str) -> int:
    """Find the lerobot/libero task_index whose language matches ``task_text``."""
    tasks = getattr(meta, "tasks", None)
    needle = task_text.strip().lower()
    if tasks is not None and hasattr(tasks, "iloc"):
        for idx in range(len(tasks)):
            row = tasks.iloc[idx]
            name = str(row.name if hasattr(row, "name") else idx).strip().lower()
            if name == needle or needle in name or name in needle:
                value = row["task_index"] if "task_index" in getattr(row, "index", []) else row
                return int(value)
    return env_task_id_to_dataset_index(env_task_id_from_task_text(task_text))


def env_task_id_to_dataset_index(env_task_id: int, suite: str = LIBERO_OBJECT_SUITE, meta: Any = None) -> int:
    if suite != LIBERO_OBJECT_SUITE:
        raise ValueError(f"Only {LIBERO_OBJECT_SUITE} is supported, got {suite!r}")
    if env_task_id not in LIBERO_OBJECT_ENV_TASK_IDS:
        raise ValueError(f"env_task_id must be in 0..9, got {env_task_id}")
    language = LIBERO_OBJECT_ENV_LANGUAGES[env_task_id]
    if meta is not None:
        return dataset_index_from_meta(meta, language)
    return env_task_id + LIBERO_OBJECT_DATASET_TASK_OFFSET


def dataset_index_to_env_task_id(task_index: int, suite: str = LIBERO_OBJECT_SUITE) -> int:
    if suite != LIBERO_OBJECT_SUITE:
        raise ValueError(f"Only {LIBERO_OBJECT_SUITE} is supported, got {suite!r}")
    env_task_id = task_index - LIBERO_OBJECT_DATASET_TASK_OFFSET
    if env_task_id not in LIBERO_OBJECT_ENV_TASK_IDS:
        raise ValueError(f"task_index {task_index} is not a {suite} task (expected 20-29)")
    return env_task_id


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _lookup_nested(mapping: dict[str, Any], path: str) -> Any | None:
    if path in mapping:
        return mapping[path]
    node: Any = mapping
    for part in path.replace(".", "/").split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def extract_eef_pos(frame: dict[str, Any], state_dims: int = 3) -> np.ndarray:
    """Return the first `state_dims` EEF coordinates from a train or eval frame."""
    if state_dims not in (3, 6):
        raise ValueError(f"state_dims must be 3 or 6, got {state_dims}")

    for key in _EEF_KEYS:
        value = _lookup_nested(frame, key)
        if value is None:
            continue
        pos = np.atleast_1d(_as_numpy(value)).reshape(-1)
        if pos.size >= state_dims:
            return pos[:state_dims].astype(np.float32)
        if pos.size >= 3 and state_dims == 3:
            return pos[:3].astype(np.float32)

    state = frame.get("observation.state")
    if state is not None:
        vec = np.atleast_1d(_as_numpy(state)).reshape(-1)
        if vec.size >= state_dims:
            return vec[:state_dims].astype(np.float32)

    raise KeyError(
        "Could not find end-effector position in frame. Tried "
        + ", ".join(_EEF_KEYS)
        + " and observation.state. Keys="
        + str(sorted(frame.keys()))
    )


def extract_full_state(frame: dict[str, Any]) -> np.ndarray:
    """Return the 8D LIBERO state when present, else pad EEF xyz."""
    state = frame.get("observation.state")
    if state is not None:
        vec = np.atleast_1d(_as_numpy(state)).reshape(-1)
        if vec.size >= 8:
            return vec[:8].astype(np.float32)
        padded = np.zeros(8, dtype=np.float32)
        padded[: vec.size] = vec
        return padded
    pos = extract_eef_pos(frame, state_dims=3)
    padded = np.zeros(8, dtype=np.float32)
    padded[: pos.size] = pos
    return padded


def task_name_from_meta(meta: Any, task_index: int) -> str:
    tasks = getattr(meta, "tasks", None)
    if tasks is None:
        return f"task_{task_index}"
    if hasattr(tasks, "iloc"):
        for idx in range(len(tasks)):
            row = tasks.iloc[idx]
            name = row.name if hasattr(row, "name") else str(idx)
            value = row["task_index"] if "task_index" in getattr(row, "index", []) else row
            try:
                if int(value) == task_index:
                    return str(name)
            except (TypeError, ValueError):
                continue
    return f"task_{task_index}"


@dataclass
class StateTable:
    positions: np.ndarray
    full_states: np.ndarray
    frame_indices: np.ndarray
    task_index: int
    task: str

    def __len__(self) -> int:
        return int(self.positions.shape[0])


def load_train_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load observation.state, task_index, and index columns once."""
    hf = dataset.hf_dataset
    table = hf.with_format("numpy", columns=["observation.state", "task_index", "index"])
    states = np.asarray(table["observation.state"], dtype=np.float32)
    if states.ndim == 1:
        states = np.stack([np.asarray(s, dtype=np.float32) for s in states])
    task_idx = np.asarray(table["task_index"]).reshape(-1)
    indices = np.asarray(table["index"]).reshape(-1)
    return states, task_idx, indices


def load_train_states(
    dataset: Any,
    task_index: int,
    stride: int = 1,
    state_dims: int = 3,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> StateTable:
    """Load demonstration EEF states for one dataset task_index without decoding video."""
    if arrays is None:
        arrays = load_train_arrays(dataset)
    states, task_idx, indices = arrays
    mask = task_idx == task_index
    if not np.any(mask):
        raise ValueError(f"No training frames found for task_index={task_index}")

    states = states[mask]
    indices = indices[mask]
    if stride > 1:
        states = states[::stride]
        indices = indices[::stride]

    positions = states[:, :state_dims].astype(np.float32)
    full_states = np.zeros((states.shape[0], 8), dtype=np.float32)
    width = min(8, states.shape[1])
    full_states[:, :width] = states[:, :width]
    return StateTable(
        positions=positions,
        full_states=full_states,
        frame_indices=indices.astype(np.int64),
        task_index=task_index,
        task=task_name_from_meta(dataset.meta, task_index),
    )


def _open_eval_dataset(recording_root: Path, env_task_id: int) -> Any | None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    candidates = [
        recording_root / f"{LIBERO_OBJECT_SUITE}_{env_task_id}",
        recording_root / f"{LIBERO_OBJECT_SUITE}_{env_task_id}" / "env_0",
    ]
    for root in candidates:
        if not root.exists():
            continue
        try:
            return LeRobotDataset(repo_id="eval_recording", root=str(root), download_videos=False)
        except Exception:
            continue
    return None


def load_eval_states(
    recording_root: Path | None,
    env_task_id: int,
    stride: int = 1,
    state_dims: int = 3,
) -> StateTable:
    """Load eval EEF states from a recorded LeRobot dataset, or an empty table."""
    task_index = env_task_id_to_dataset_index(env_task_id)
    empty = StateTable(
        positions=np.zeros((0, state_dims), dtype=np.float32),
        full_states=np.zeros((0, 8), dtype=np.float32),
        frame_indices=np.zeros((0,), dtype=np.int64),
        task_index=task_index,
        task=f"{LIBERO_OBJECT_SUITE}_{env_task_id}",
    )
    if recording_root is None:
        return empty

    npz_path = recording_root / f"{LIBERO_OBJECT_SUITE}_{env_task_id}" / "eef_states.npz"
    if npz_path.exists():
        payload = np.load(npz_path)
        positions = np.asarray(payload["positions"], dtype=np.float32)
        if stride > 1:
            positions = positions[::stride]
        full = np.zeros((positions.shape[0], 8), dtype=np.float32)
        full[:, : positions.shape[1]] = positions
        return StateTable(
            positions=positions[:, :state_dims],
            full_states=full,
            frame_indices=np.arange(len(positions), dtype=np.int64),
            task_index=task_index,
            task=str(payload["task"]) if "task" in payload else empty.task,
        )

    dataset = _open_eval_dataset(recording_root, env_task_id)
    if dataset is None:
        return empty

    positions: list[np.ndarray] = []
    full_states: list[np.ndarray] = []
    indices: list[int] = []
    for i in range(0, len(dataset), stride):
        try:
            frame = dataset.get_raw_item(i)
        except Exception:
            frame = dataset[i]
        try:
            positions.append(extract_eef_pos(frame, state_dims=state_dims))
            full_states.append(extract_full_state(frame))
            indices.append(int(frame.get("index", i)))
        except KeyError:
            continue

    if not positions:
        return empty
    return StateTable(
        positions=np.stack(positions).astype(np.float32),
        full_states=np.stack(full_states).astype(np.float32),
        frame_indices=np.asarray(indices, dtype=np.int64),
        task_index=task_index,
        task=empty.task,
    )


def padded_range(states: np.ndarray, pad_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    if states.size == 0:
        raise ValueError("Cannot compute a range from an empty state table")
    low = states.min(axis=0)
    high = states.max(axis=0)
    span = np.maximum(high - low, 1e-3)
    return low - pad_frac * span, high + pad_frac * span


def build_histogram(
    states: np.ndarray,
    bins: int,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """3D occupancy histogram over xyz (first 3 columns)."""
    xyz = states[:, :3] if states.size else np.zeros((0, 3), dtype=np.float32)
    edges = [
        np.linspace(float(low[i]), float(high[i]), bins + 1, dtype=np.float64) for i in range(3)
    ]
    hist, _ = np.histogramdd(xyz, bins=edges) if len(xyz) else (np.zeros((bins, bins, bins)), edges)
    return hist.astype(np.int64), edges


def find_blind_bins(train_hist: np.ndarray, eval_hist: np.ndarray) -> np.ndarray:
    """Return (K, 3) integer bin coordinates where both histograms are empty."""
    if train_hist.shape != eval_hist.shape:
        raise ValueError(f"Histogram shapes differ: {train_hist.shape} vs {eval_hist.shape}")
    coords = np.argwhere((train_hist == 0) & (eval_hist == 0))
    return coords.astype(np.int64)


def bin_centroids(coords: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centers = []
    for axis, edge in enumerate(edges):
        idx = coords[:, axis]
        centers.append(0.5 * (edge[idx] + edge[idx + 1]))
    return np.stack(centers, axis=1).astype(np.float32)


def nearest_point_distances(centroids: np.ndarray, support_pos: np.ndarray) -> np.ndarray:
    if len(centroids) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(support_pos) == 0:
        return np.full((len(centroids),), np.inf, dtype=np.float32)
    # Chunked pairwise min-distance to avoid a giant (K x N) matrix.
    dists = np.empty((len(centroids),), dtype=np.float32)
    chunk = 256
    support = support_pos[:, :3]
    for start in range(0, len(centroids), chunk):
        batch = centroids[start : start + chunk, :3]
        delta = batch[:, None, :] - support[None, :, :]
        dists[start : start + chunk] = np.sqrt((delta * delta).sum(axis=2)).min(axis=1)
    return dists


def nearest_train_distances(centroids: np.ndarray, train_pos: np.ndarray) -> np.ndarray:
    return nearest_point_distances(centroids, train_pos)


def nearest_support_distances(
    centroids: np.ndarray,
    train_pos: np.ndarray,
    eval_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-centroid min distance to train, eval, and combined support."""
    train_dists = nearest_point_distances(centroids, train_pos)
    eval_dists = nearest_point_distances(centroids, eval_pos)
    support_dists = np.minimum(train_dists, eval_dists)
    return train_dists, eval_dists, support_dists


def rank_blind_bins(
    coords: np.ndarray,
    edges: list[np.ndarray],
    train_pos: np.ndarray,
    eval_pos: np.ndarray,
    top_k: int,
    selection: str = "deep",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep `top_k` blind bins ranked by distance to train/eval support.

    `selection`:
      - ``deep``: farthest from any train or eval frame (strong unsupported probes)
      - ``boundary``: closest to training support (legacy boundary preference)
    """
    if selection not in {"deep", "boundary"}:
        raise ValueError(f"selection must be 'deep' or 'boundary', got {selection!r}")

    centroids = bin_centroids(coords, edges)
    train_dists, eval_dists, support_dists = nearest_support_distances(
        centroids, train_pos, eval_pos
    )
    if len(coords) == 0:
        return coords, centroids, train_dists, eval_dists, support_dists

    if selection == "deep":
        order = np.argsort(-support_dists)
    else:
        order = np.argsort(train_dists)
    keep = order[: min(top_k, len(order))]
    return (
        coords[keep],
        centroids[keep],
        train_dists[keep],
        eval_dists[keep],
        support_dists[keep],
    )


def _xy_separated(point_xy: np.ndarray, selected_xy: np.ndarray, min_sep: float) -> bool:
    if len(selected_xy) == 0:
        return True
    dists = np.linalg.norm(selected_xy - point_xy.reshape(1, 2), axis=1)
    return bool(np.all(dists >= min_sep))


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((0,), dtype=np.float32)
    order = np.argsort(values)
    ranks = np.empty((len(values),), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def rank_blind_bins_multiscale(
    coords: np.ndarray,
    edges: list[np.ndarray],
    train_pos: np.ndarray,
    eval_pos: np.ndarray,
    n_min: int = 1,
    n_moderate: int = 3,
    n_far: int = 1,
    min_bin_separation: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Select blind bins at min / moderate / far support-distance scales.

    Bins are ranked by ``nearest_support_distance`` (min of train and eval).
    Selected scenes are required to be at least ``min_bin_separation`` apart in XY.
    """
    centroids = bin_centroids(coords, edges)
    train_dists, eval_dists, support_dists = nearest_support_distances(
        centroids, train_pos, eval_pos
    )
    empty_scales: list[str] = []
    empty_pcts = np.zeros((0,), dtype=np.float32)
    if len(coords) == 0:
        return coords, centroids, train_dists, eval_dists, support_dists, empty_scales, empty_pcts

    order = np.argsort(support_dists)
    selected: list[int] = []
    scales: list[str] = []

    def _try_pick(candidates: np.ndarray, scale: str, needed: int) -> None:
        picked = 0
        for idx in candidates:
            i = int(idx)
            if i in selected:
                continue
            if not _xy_separated(centroids[i, :2], centroids[selected, :2] if selected else np.zeros((0, 2)), min_bin_separation):
                continue
            selected.append(i)
            scales.append(scale)
            picked += 1
            if picked >= needed:
                return

    _try_pick(order, "min", n_min)
    _try_pick(order[::-1], "far", n_far)

    remaining = [int(i) for i in order if int(i) not in selected]
    if remaining and n_moderate > 0:
        n = len(remaining)
        if n_moderate == 1:
            fractions = (0.50,)
        elif n_moderate == 2:
            fractions = (0.33, 0.66)
        else:
            fractions = tuple((k + 1) / (n_moderate + 1) for k in range(n_moderate))
        for frac in fractions:
            target = remaining[int(round(frac * (n - 1)))]
            loc = remaining.index(target)
            neighbors = remaining[loc:] + remaining[:loc][::-1]
            _try_pick(np.asarray(neighbors, dtype=np.int64), "moderate", 1)

    if not selected:
        return (
            coords[:0],
            centroids[:0],
            train_dists[:0],
            eval_dists[:0],
            support_dists[:0],
            empty_scales,
            empty_pcts,
        )

    keep = np.asarray(selected, dtype=np.int64)
    pcts = _percentile_ranks(support_dists)[keep]
    return (
        coords[keep],
        centroids[keep],
        train_dists[keep],
        eval_dists[keep],
        support_dists[keep],
        scales,
        pcts,
    )


def digitize_point(point: np.ndarray, edges: list[np.ndarray]) -> tuple[int, ...] | None:
    idxs = []
    for value, edge in zip(point[:3], edges, strict=True):
        bin_id = int(np.searchsorted(edge, value, side="right") - 1)
        if bin_id < 0 or bin_id >= len(edge) - 1:
            return None
        idxs.append(bin_id)
    return tuple(idxs)


def pick_anchor_frame(
    table: StateTable,
    centroid: np.ndarray,
    train_hist: np.ndarray,
    edges: list[np.ndarray],
    neighbor_pool: int = 32,
    anchor_mode: str = "far",
) -> dict[str, Any]:
    """Choose an in-distribution training frame to interpolate from.

    `anchor_mode`:
      - ``far``: farthest visited training frame from the blind-bin centroid
      - ``near``: high-density training frame closest to the centroid (legacy)
    """
    if anchor_mode not in {"far", "near"}:
        raise ValueError(f"anchor_mode must be 'far' or 'near', got {anchor_mode!r}")
    if len(table) == 0:
        raise ValueError("Cannot pick an anchor from an empty training table")

    dists = np.linalg.norm(table.positions[:, :3] - centroid[:3], axis=1)
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(table)):
        bin_id = digitize_point(table.positions[i], edges)
        count = int(train_hist[bin_id]) if bin_id is not None else 0
        if count > 0:
            candidates.append((float(dists[i]), count, i))

    if not candidates:
        raise ValueError("Could not find an in-distribution anchor frame with train coverage")

    if anchor_mode == "near":
        pool = sorted(candidates, key=lambda item: (item[0], -item[1]))[:neighbor_pool]
        _, best_count, best_i = max(pool, key=lambda item: (item[1], -item[0]))
    else:
        _, best_count, best_i = max(candidates, key=lambda item: (item[0], item[1]))

    return {
        "frame_index": int(table.frame_indices[best_i]),
        "state": table.full_states[best_i].tolist(),
        "eef": table.positions[best_i, :3].tolist(),
        "distance_to_centroid": float(dists[best_i]),
        "train_bin_count": int(best_count),
        "anchor_mode": anchor_mode,
        "task": table.task,
        "task_index": table.task_index,
    }


def interpolate_probes(
    anchor_state: np.ndarray,
    target_eef: np.ndarray,
    n_probes: int = 4,
    state_dims: int = 3,
) -> list[np.ndarray]:
    """Linearly shift xyz (or xyz+axis-angle) from the anchor toward the target."""
    if n_probes < 1:
        raise ValueError("n_probes must be >= 1")
    anchor = np.asarray(anchor_state, dtype=np.float32).reshape(-1)
    target = np.asarray(target_eef, dtype=np.float32).reshape(-1)
    probes = []
    for step in range(1, n_probes + 1):
        alpha = step / n_probes
        state = anchor.copy()
        state[:state_dims] = (1.0 - alpha) * anchor[:state_dims] + alpha * target[:state_dims]
        probes.append(state)
    return probes


def project_hist(hist: np.ndarray, axis: int) -> np.ndarray:
    return hist.sum(axis=axis)


def compute_coverage_trail(
    init_xyz: np.ndarray,
    furthest_centroid: np.ndarray,
    train_hist: np.ndarray,
    eval_hist: np.ndarray,
    edges: list[np.ndarray],
    range_low: np.ndarray,
    range_high: np.ndarray,
    train_pos: np.ndarray,
    eval_pos: np.ndarray,
    n_points: int = 5,
) -> list[dict[str, Any]]:
    """Histogram-only probe trail (p1 boundary → p5 furthest) for coverage plots."""
    from blindspot_sim_utils import compute_object_probe_positions

    init_xyz = np.asarray(init_xyz, dtype=np.float32).reshape(-1)
    furthest_xy = np.asarray(furthest_centroid[:2], dtype=np.float32).reshape(2)
    positions = compute_object_probe_positions(
        init_xyz,
        furthest_xy,
        train_hist,
        eval_hist,
        edges,
        range_low,
        range_high,
        n_probes=n_points,
    )
    centroids = np.stack(
        [np.array([xyz[0], xyz[1], xyz[2]], dtype=np.float32) for xyz, _role in positions],
        axis=0,
    )
    _train_d, _eval_d, support_dists = nearest_support_distances(centroids, train_pos, eval_pos)
    trail: list[dict[str, Any]] = []
    for (xyz, role), dist in zip(positions, support_dists, strict=True):
        trail.append(
            {
                "role": role,
                "label": trail_plot_label(role),
                "centroid": np.asarray(xyz, dtype=np.float32).tolist(),
                "nearest_support_distance": float(dist),
            }
        )
    return trail


def plot_density_maps(
    train_hist: np.ndarray,
    eval_hist: np.ndarray,
    edges: list[np.ndarray],
    output_dir: Path,
    trail_centroids: np.ndarray | None = None,
    trail_support_dists: np.ndarray | None = None,
    trail_labels: list[str] | None = None,
    title: str = "",
) -> dict[str, Path]:
    """Write XY/XZ heatmaps and a combined three-panel figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    blind = ((train_hist == 0) & (eval_hist == 0)).astype(np.float32)
    views = {
        "xy": (2, 0, 1, "x", "y"),
        "xz": (1, 0, 2, "x", "z"),
    }
    paths: dict[str, Path] = {}

    def _extent(axis_x: int, axis_y: int) -> list[float]:
        return [
            float(edges[axis_x][0]),
            float(edges[axis_x][-1]),
            float(edges[axis_y][0]),
            float(edges[axis_y][-1]),
        ]

    for name, (sum_axis, ax, ay, xlabel, ylabel) in views.items():
        train_img = project_hist(train_hist, sum_axis).T
        eval_img = project_hist(eval_hist, sum_axis).T
        blind_img = (project_hist(blind, sum_axis) > 0).astype(np.float32).T
        extent = _extent(ax, ay)
        for key, img, cmap in (
            (f"train_{name}", train_img, "Blues"),
            (f"eval_{name}", eval_img, "Oranges"),
            (f"blind_{name}", blind_img, "Greys"),
        ):
            fig, axis = plt.subplots(figsize=(5.2, 4.4))
            im = axis.imshow(img, origin="lower", aspect="auto", extent=extent, cmap=cmap)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.set_title(f"{title} {key}".strip())
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
            path = output_dir / f"{key}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
            paths[key] = path

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = [
        (project_hist(train_hist, 2).T, "Train XY", "Blues"),
        (project_hist(eval_hist, 2).T, "Eval XY", "Oranges"),
        ((project_hist(blind, 2) > 0).astype(np.float32).T, "Blind XY", "Greys"),
    ]
    extent = _extent(0, 1)
    dist_mappable = None
    for axis, (img, label, cmap) in zip(axes, panels, strict=True):
        im = axis.imshow(img, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
        if trail_centroids is not None and len(trail_centroids):
            dist_mappable = _scatter_trail_points(
                axis,
                trail_centroids,
                trail_support_dists,
                trail_labels,
                add_colorbar=False,
            )
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.88, 0.94])
    if dist_mappable is not None:
        pos = axes[-1].get_position()
        dist_cbar_ax = fig.add_axes([pos.x1 + 0.10, pos.y0, 0.012, pos.height])
        fig.colorbar(dist_mappable, cax=dist_cbar_ax, label="dist to support (m)")
    combined = output_dir / "combined.png"
    fig.savefig(combined, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["combined"] = combined
    return paths


def _scatter_trail_points(
    axis,
    centroids: np.ndarray,
    support_dists: np.ndarray | None,
    labels: list[str] | None = None,
    *,
    add_colorbar: bool = True,
) -> Any:
    """Scatter p1–p5 trail points; darker green = farther from train/eval support."""
    import matplotlib.pyplot as plt

    centroids = np.asarray(centroids, dtype=np.float32).reshape(-1, 3)
    if support_dists is None or len(support_dists) != len(centroids):
        axis.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="#39ff14",
            s=52,
            marker="o",
            edgecolors="black",
            linewidths=0.7,
            label="probe trail",
            zorder=5,
        )
        return None

    dists = np.asarray(support_dists, dtype=np.float32).reshape(-1)
    vmin = float(dists.min())
    vmax = float(dists.max())
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1e-3
    dist_cmap = plt.get_cmap(DIST_SUPPORT_CMAP)
    dist_norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, (xy, dist) in enumerate(zip(centroids, dists, strict=True)):
        color = dist_cmap(dist_norm(dist))
        axis.scatter(
            [xy[0]],
            [xy[1]],
            c=[color],
            s=56,
            marker="o",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
        label = labels[i] if labels is not None and i < len(labels) else f"p{i + 1}"
        axis.annotate(
            label,
            (float(xy[0]), float(xy[1])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="black",
            zorder=6,
        )
    sm = plt.cm.ScalarMappable(cmap=dist_cmap, norm=dist_norm)
    sm.set_array([])
    if add_colorbar:
        cbar = axis.figure.colorbar(sm, ax=axis, fraction=0.046, pad=0.08)
        cbar.set_label("dist to support (m)")
    return sm


def plot_probe_overlay(
    train_hist: np.ndarray,
    edges: list[np.ndarray],
    centroids: np.ndarray,
    probe_xy: np.ndarray,
    probe_scores: np.ndarray,
    output_path: Path,
    title: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = project_hist(train_hist, 2).T
    extent = [float(edges[0][0]), float(edges[0][-1]), float(edges[1][0]), float(edges[1][-1])]
    fig, axis = plt.subplots(figsize=(6.2, 5.0))
    im = axis.imshow(img, origin="lower", aspect="auto", extent=extent, cmap="Blues")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04, label="train count")
    if len(centroids):
        axis.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="black",
            s=36,
            marker="x",
            linewidths=1.5,
            label="blind bins",
        )
    if len(probe_xy):
        sc = axis.scatter(
            probe_xy[:, 0],
            probe_xy[:, 1],
            c=probe_scores,
            cmap="plasma",
            s=42,
            edgecolors="white",
            linewidths=0.4,
            label="probes",
        )
        fig.colorbar(sc, ax=axis, fraction=0.046, pad=0.08, label="action_delta_vs_anchor")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title or "Probe overlay")
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
