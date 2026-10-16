"""LIBERO sim helpers for object-repositioned blind-spot probes.

Moves the task target object while keeping task text, robot init pose, and
distractors fixed. Observations are rendered through the same eval stack as
``lerobot-eval``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blindspot_utils import SCALE_MARKERS, digitize_point, project_hist

LIBERO_OBJECT_TASK_SLUGS = (
    "alphabet_soup",
    "cream_cheese",
    "salad_dressing",
    "bbq_sauce",
    "ketchup",
    "tomato_sauce",
    "butter",
    "milk",
    "chocolate_pudding",
    "orange_juice",
)


def target_object_slug(task_name: str) -> str:
    """Parse a LIBERO-object task name or language string into an object slug."""
    name = task_name.strip().lower().replace(" ", "_")
    for prefix in ("pick_up_the_", "pick_up_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for suffix in ("_and_place_it_in_the_basket", "_and_place"):
        if suffix in name:
            name = name.split(suffix, 1)[0]
            break
    return name


def target_object_joint(task_name: str) -> str:
    return f"{target_object_slug(task_name)}_1_joint0"


def support_xy_mask(train_hist: np.ndarray, eval_hist: np.ndarray) -> np.ndarray:
    """2D mask: True where train or eval EEF occupancy exists at any height."""
    return ((train_hist > 0) | (eval_hist > 0)).any(axis=2)


def _xy_supported(
    xy: np.ndarray,
    support_mask: np.ndarray,
    edges: list[np.ndarray],
    table_z: float,
) -> bool:
    point = np.array([xy[0], xy[1], table_z], dtype=np.float32)
    idx = digitize_point(point, edges)
    if idx is None:
        return False
    return bool(support_mask[idx[0], idx[1]])


def _along_ray(init_xy: np.ndarray, xy: np.ndarray, unit: np.ndarray) -> float:
    return float(np.dot(np.asarray(xy, dtype=np.float32)[:2] - init_xy, unit))


def find_support_boundary_xy(
    init_xy: np.ndarray,
    target_xy: np.ndarray,
    support_mask: np.ndarray,
    edges: list[np.ndarray],
    table_z: float,
    max_dist: float = 0.45,
    n_samples: int = 256,
) -> np.ndarray:
    """Last supported XY point on the ray from init toward target."""
    init_xy = np.asarray(init_xy, dtype=np.float32).reshape(2)
    target_xy = np.asarray(target_xy, dtype=np.float32).reshape(2)
    delta = target_xy - init_xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return init_xy.copy()
    unit = delta / dist
    last_supported = init_xy.copy()
    for t in np.linspace(0.0, min(max_dist, dist * 1.5), n_samples):
        pt = init_xy + unit * t
        if _xy_supported(pt, support_mask, edges, table_z):
            last_supported = pt
        else:
            break
    return last_supported.astype(np.float32)


def _blind_candidates_on_ray(
    init_xy: np.ndarray,
    target_xy: np.ndarray,
    blind_centroids: np.ndarray,
    min_along: float,
    lateral_max: float = 0.06,
) -> list[np.ndarray]:
    if len(blind_centroids) == 0:
        return []
    init_xy = np.asarray(init_xy, dtype=np.float32).reshape(2)
    target_xy = np.asarray(target_xy, dtype=np.float32).reshape(2)
    delta = target_xy - init_xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return []
    unit = delta / dist
    picked: list[tuple[float, np.ndarray]] = []
    for centroid in blind_centroids:
        xy = np.asarray(centroid[:2], dtype=np.float32)
        along = _along_ray(init_xy, xy, unit)
        if along <= min_along:
            continue
        lateral = float(np.linalg.norm(xy - init_xy - unit * along))
        if lateral > lateral_max:
            continue
        picked.append((along, xy))
    picked.sort(key=lambda item: item[0])
    return [xy for _, xy in picked]


def _extrapolate_xy(boundary_xy: np.ndarray, target_xy: np.ndarray, scale: float) -> np.ndarray:
    boundary_xy = np.asarray(boundary_xy, dtype=np.float32).reshape(2)
    target_xy = np.asarray(target_xy, dtype=np.float32).reshape(2)
    return (boundary_xy + scale * (target_xy - boundary_xy)).astype(np.float32)


def _clamp_xy(
    xy: np.ndarray,
    range_low: np.ndarray,
    range_high: np.ndarray,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(2)
    return np.array(
        [
            np.clip(xy[0], range_low[0], range_high[0]),
            np.clip(xy[1], range_low[1], range_high[1]),
        ],
        dtype=np.float32,
    )


def _enforce_min_shift(prev_xy: np.ndarray, xy: np.ndarray, min_shift: float) -> np.ndarray:
    prev_xy = np.asarray(prev_xy, dtype=np.float32).reshape(2)
    xy = np.asarray(xy, dtype=np.float32).reshape(2)
    if float(np.linalg.norm(xy - prev_xy)) >= min_shift:
        return xy
    delta = xy - prev_xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return xy
    return (prev_xy + delta / dist * min_shift).astype(np.float32)


DEFAULT_TRAIL_POINTS = 5
DEFAULT_MIN_TRAIL_STEP = 0.07
DEFAULT_OBJECT_SEPARATION = 0.055
DEFAULT_MIN_GEOM_GAP = 0.045
DEFAULT_DISTRACTOR_STANDOFF_GAP = 0.10
DEFAULT_OBSTACLE_SHIFT_BUFFER = 0.03
DEFAULT_OBSTACLE_MIN_SHIFT = 0.05
DEFAULT_MAX_LATERAL_SHIFT = 0.12
DEFAULT_MAX_ALONG_RETREAT = 0.12
DEFAULT_MAX_ALONG_RETREAT_P5 = 0.25
DEFAULT_WORKSPACE_MARGIN = 0.015
TRAIL_ROLES = ("p1_boundary", "p2", "p3", "p4", "p5_far")


def _xy_in_workspace(
    xy: np.ndarray,
    range_low: np.ndarray,
    range_high: np.ndarray,
    margin: float = DEFAULT_WORKSPACE_MARGIN,
) -> bool:
    xy = np.asarray(xy, dtype=np.float32).reshape(2)
    return bool(
        range_low[0] + margin <= xy[0] <= range_high[0] - margin
        and range_low[1] + margin <= xy[1] <= range_high[1] - margin
    )


def _clamp_xy_to_workspace(
    xy: np.ndarray,
    range_low: np.ndarray,
    range_high: np.ndarray,
    margin: float = DEFAULT_WORKSPACE_MARGIN,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(2)
    return np.array(
        [
            np.clip(xy[0], range_low[0] + margin, range_high[0] - margin),
            np.clip(xy[1], range_low[1] + margin, range_high[1] - margin),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class SceneObjectInfo:
    """Movable scene prop (distractor or basket), excluding the pick target."""

    body_name: str
    slug: str
    joint_name: str
    init_xyz: np.ndarray
    init_quat: np.ndarray

    @property
    def init_xy(self) -> np.ndarray:
        return self.init_xyz[:2].astype(np.float32)

    @property
    def is_basket(self) -> bool:
        return "basket" in self.slug


@dataclass(frozen=True)
class SceneCollisionModel:
    """XY collision footprints for the target object and movable scene obstacles."""

    init_body_xy: np.ndarray
    target_rel_centers: np.ndarray
    target_half_extents: np.ndarray
    obstacle_aabbs: list[tuple[str, np.ndarray]]
    obstacle_init_xy: dict[str, np.ndarray]


def _unit_xy(delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32).reshape(2)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (delta / dist).astype(np.float32)


def _perpendicular(unit: np.ndarray) -> np.ndarray:
    unit = np.asarray(unit, dtype=np.float32).reshape(2)
    return np.array([-unit[1], unit[0]], dtype=np.float32)


def compute_trail_endpoints(
    init_xyz: np.ndarray,
    target_xy: np.ndarray,
    train_hist: np.ndarray,
    eval_hist: np.ndarray,
    edges: list[np.ndarray],
    range_low: np.ndarray,
    range_high: np.ndarray,
    blind_centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return support boundary XY, blind-bin centroid (p5), and ray unit (p1→p5)."""
    del blind_centroids  # p5 comes from the selected blind-bin centroid
    init = np.asarray(init_xyz, dtype=np.float32).reshape(-1)
    init_xy = init[:2]
    far_xy = _clamp_xy(target_xy, range_low, range_high)
    support_mask = support_xy_mask(train_hist, eval_hist)
    boundary_xy = find_support_boundary_xy(
        init_xy, far_xy, support_mask, edges, table_z=float(init[2])
    )
    unit = _unit_xy(far_xy - boundary_xy)
    return boundary_xy.astype(np.float32), far_xy.astype(np.float32), unit


def build_even_trail(
    boundary_xy: np.ndarray,
    far_xy: np.ndarray,
    z: float,
    n_points: int = DEFAULT_TRAIL_POINTS,
    min_step_shift: float = DEFAULT_MIN_TRAIL_STEP,
) -> list[tuple[np.ndarray, str]]:
    """Evenly space ``n_points`` from boundary (p1) to far (p5) with minimum spacing."""
    if n_points < 2:
        raise ValueError("trail requires at least 2 points")
    boundary_xy = np.asarray(boundary_xy, dtype=np.float32).reshape(2)
    far_xy = np.asarray(far_xy, dtype=np.float32).reshape(2)
    delta = far_xy - boundary_xy
    unit = _unit_xy(delta)
    span = float(np.linalg.norm(delta))
    min_span = min_step_shift * (n_points - 1)
    if span < min_span:
        far_xy = boundary_xy + unit * min_span
        span = min_span

    roles = TRAIL_ROLES if n_points == len(TRAIL_ROLES) else tuple(f"p{i + 1}" for i in range(n_points))
    trail: list[tuple[np.ndarray, str]] = []
    prev_xy = boundary_xy.copy()
    for i in range(n_points):
        frac = i / (n_points - 1)
        xy = boundary_xy + unit * (frac * span)
        if i > 0:
            xy = _enforce_min_shift(prev_xy, xy, min_step_shift)
        xyz = np.array([xy[0], xy[1], z], dtype=np.float32)
        trail.append((xyz, roles[i]))
        prev_xy = xy
    return trail


def agentview_contains(sim: Any, xyz: np.ndarray, margin_frac: float = 0.12) -> bool:
    """True when ``xyz`` projects inside the agentview frustum."""
    try:
        cam_id = int(sim.model.camera_name2id("agentview"))
    except (ValueError, KeyError, TypeError):
        return True
    cam_pos = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64)
    cam_mat = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    rel = np.asarray(xyz[:3], dtype=np.float64) - cam_pos
    cam = cam_mat.T @ rel
    if cam[2] >= 0:
        return False
    fovy = np.deg2rad(float(sim.model.cam_fovy[cam_id]))
    half = (fovy / 2.0) * (1.0 - margin_frac)
    px = float(np.arctan2(cam[0], -cam[2]))
    py = float(np.arctan2(cam[1], -cam[2]))
    return abs(px) <= half and abs(py) <= half


def clamp_far_to_agentview(
    sim: Any,
    boundary_xy: np.ndarray,
    far_xy: np.ndarray,
    z: float,
    margin_frac: float = 0.12,
) -> np.ndarray:
    """Pull ``far_xy`` back along the boundary→far ray until it is visible."""
    boundary_xy = np.asarray(boundary_xy, dtype=np.float32).reshape(2)
    far_xy = np.asarray(far_xy, dtype=np.float32).reshape(2)
    unit = _unit_xy(far_xy - boundary_xy)
    if agentview_contains(sim, np.array([far_xy[0], far_xy[1], z], dtype=np.float32), margin_frac):
        return far_xy
    lo, hi = 0.0, float(np.linalg.norm(far_xy - boundary_xy))
    best = boundary_xy.copy()
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        pt = boundary_xy + unit * mid
        if agentview_contains(sim, np.array([pt[0], pt[1], z], dtype=np.float32), margin_frac):
            best = pt
            lo = mid
        else:
            hi = mid
    return best.astype(np.float32)


def _is_scene_object_slug(slug: str, target_slug: str) -> bool:
    if slug == target_slug or slug in {"table", "floor", "robot0", "mount0"}:
        return False
    return any(token in slug for token in LIBERO_OBJECT_TASK_SLUGS) or "basket" in slug


def _geom_xy_half_extents(geom_type: int, geom_size: np.ndarray) -> np.ndarray:
    size = np.asarray(geom_size, dtype=np.float32).reshape(-1)
    if int(geom_type) == 6:
        return size[:2].astype(np.float32)
    if int(geom_type) == 7:
        return size[:2].astype(np.float32)
    radius = float(size[0]) if len(size) else 0.01
    return np.array([radius, radius], dtype=np.float32)


def _geom_xy_aabb(geom_pos: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    center = np.asarray(geom_pos[:2], dtype=np.float32)
    half = np.asarray(half_extents[:2], dtype=np.float32)
    return np.array(
        [center[0] - half[0], center[0] + half[0], center[1] - half[1], center[1] + half[1]],
        dtype=np.float32,
    )


def _aabb_edge_gap(a: np.ndarray, b: np.ndarray) -> float:
    overlap_x = min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0]))
    overlap_y = min(float(a[3]), float(b[3])) - max(float(a[2]), float(b[2]))
    if overlap_x > 0.0 and overlap_y > 0.0:
        return -min(overlap_x, overlap_y)
    gap_x = max(float(b[0] - a[1]), float(a[0] - b[1]), 0.0)
    gap_y = max(float(b[2] - a[3]), float(a[2] - b[3]), 0.0)
    return float(np.hypot(gap_x, gap_y))


def _target_aabbs_at_xy(model: SceneCollisionModel, body_xy: np.ndarray) -> list[np.ndarray]:
    body_xy = np.asarray(body_xy, dtype=np.float32).reshape(2)
    delta = body_xy - model.init_body_xy
    aabbs: list[np.ndarray] = []
    for rel_center, half in zip(model.target_rel_centers, model.target_half_extents, strict=True):
        center = rel_center + delta
        aabbs.append(_geom_xy_aabb(center, half))
    return aabbs


def _shift_aabb(aabb: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_xy, dtype=np.float32).reshape(2)
    return np.array(
        [aabb[0] + delta[0], aabb[1] + delta[0], aabb[2] + delta[1], aabb[3] + delta[1]],
        dtype=np.float32,
    )


def min_geom_gap_xy(
    model: SceneCollisionModel,
    body_xy: np.ndarray,
    obstacle_xy: dict[str, np.ndarray] | None = None,
) -> tuple[float, str | None]:
    """Minimum edge-to-edge XY gap between the target object and scene obstacles."""
    obstacle_xy = obstacle_xy or {}
    target_aabbs = _target_aabbs_at_xy(model, body_xy)
    best_gap = float("inf")
    closest: str | None = None
    for name, obstacle_aabbs in model.obstacle_aabbs:
        init_obs_xy = model.obstacle_init_xy[name]
        current_obs_xy = obstacle_xy.get(name, init_obs_xy)
        delta = current_obs_xy - init_obs_xy
        for target_aabb in target_aabbs:
            for obstacle_aabb in obstacle_aabbs:
                gap = _aabb_edge_gap(target_aabb, _shift_aabb(obstacle_aabb, delta))
                if gap < best_gap:
                    best_gap = gap
                    closest = name
    if not np.isfinite(best_gap):
        return float("inf"), None
    return best_gap, closest


def min_geom_gap_to_obstacle(
    model: SceneCollisionModel,
    body_xy: np.ndarray,
    obstacle_name: str,
    obstacle_xy: dict[str, np.ndarray] | None = None,
) -> float:
    """Edge-to-edge XY gap between the target and one named obstacle."""
    obstacle_xy = obstacle_xy or {}
    if obstacle_name not in model.obstacle_init_xy:
        return float("inf")
    target_aabbs = _target_aabbs_at_xy(model, body_xy)
    init_obs_xy = model.obstacle_init_xy[obstacle_name]
    current_obs_xy = obstacle_xy.get(obstacle_name, init_obs_xy)
    delta = current_obs_xy - init_obs_xy
    best_gap = float("inf")
    for name, obstacle_aabbs in model.obstacle_aabbs:
        if name != obstacle_name:
            continue
        for target_aabb in target_aabbs:
            for obstacle_aabb in obstacle_aabbs:
                gap = _aabb_edge_gap(target_aabb, _shift_aabb(obstacle_aabb, delta))
                best_gap = min(best_gap, gap)
    return best_gap if np.isfinite(best_gap) else float("inf")


def build_scene_collision_model(sim: Any, target_slug: str) -> SceneCollisionModel:
    """Snapshot geom footprints for the target object and all scene props."""
    model = sim.model
    init_body_xy = None
    target_rel_centers: list[np.ndarray] = []
    target_half_extents: list[np.ndarray] = []
    obstacle_aabbs: dict[str, list[np.ndarray]] = {}
    obstacle_init_xy: dict[str, np.ndarray] = {}

    for body_id in range(int(model.nbody)):
        body_name = model.body_id2name(body_id)
        if not body_name or not body_name.endswith("_main"):
            continue
        slug = body_name.replace("_1_main", "")
        if not _is_scene_object_slug(slug, target_slug) and slug != target_slug:
            continue
        body_xy = np.asarray(sim.data.body_xpos[body_id][:2], dtype=np.float32)
        if slug == target_slug:
            init_body_xy = body_xy.copy()
        geoms: list[np.ndarray] = []
        for geom_id in range(int(model.ngeom)):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            geom_name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
            if slug == target_slug:
                rel = np.asarray(sim.data.geom_xpos[geom_id][:2], dtype=np.float32) - body_xy
                target_rel_centers.append(rel)
                target_half_extents.append(
                    _geom_xy_half_extents(int(model.geom_type[geom_id]), model.geom_size[geom_id])
                )
                continue
            if slug != target_slug:
                geoms.append(
                    _geom_xy_aabb(
                        sim.data.geom_xpos[geom_id],
                        _geom_xy_half_extents(int(model.geom_type[geom_id]), model.geom_size[geom_id]),
                    )
                )
        if slug != target_slug and geoms:
            obstacle_aabbs[body_name] = geoms
            obstacle_init_xy[body_name] = body_xy.copy()

    if init_body_xy is None:
        raise ValueError(f"Could not find target object body for slug {target_slug!r}")
    if not target_rel_centers:
        target_rel_centers = [np.zeros(2, dtype=np.float32)]
        target_half_extents = [np.array([0.03, 0.03], dtype=np.float32)]

    return SceneCollisionModel(
        init_body_xy=init_body_xy,
        target_rel_centers=np.asarray(target_rel_centers, dtype=np.float32),
        target_half_extents=np.asarray(target_half_extents, dtype=np.float32),
        obstacle_aabbs=[(name, np.asarray(aabbs, dtype=np.float32)) for name, aabbs in obstacle_aabbs.items()],
        obstacle_init_xy=obstacle_init_xy,
    )


def object_joint_from_slug(slug: str) -> str:
    return f"{slug}_1_joint0"


def list_scene_objects(sim: Any, target_slug: str) -> list[SceneObjectInfo]:
    """Return movable distractors and basket (excluding the pick target)."""
    objects: list[SceneObjectInfo] = []
    for body_id in range(int(sim.model.nbody)):
        body_name = sim.model.body_id2name(body_id)
        if not body_name or not body_name.endswith("_main"):
            continue
        slug = body_name.replace("_1_main", "")
        if not _is_scene_object_slug(slug, target_slug):
            continue
        joint_name = object_joint_from_slug(slug)
        try:
            xyz, quat = get_object_qpos(sim, joint_name)
        except (ValueError, KeyError):
            continue
        objects.append(
            SceneObjectInfo(
                body_name=body_name,
                slug=slug,
                joint_name=joint_name,
                init_xyz=xyz.astype(np.float32),
                init_quat=quat.astype(np.float32),
            )
        )
    return objects


def list_other_object_xy(sim: Any, target_slug: str) -> list[tuple[str, np.ndarray]]:
    """Return distractor/basket object XY positions (excluding the pick target)."""
    objects: list[tuple[str, np.ndarray]] = []
    for body_id in range(int(sim.model.nbody)):
        name = sim.model.body_id2name(body_id)
        if not name or not name.endswith("_main"):
            continue
        slug = name.replace("_1_main", "")
        if not _is_scene_object_slug(slug, target_slug):
            continue
        xy = np.asarray(sim.data.body_xpos[body_id][:2], dtype=np.float32)
        objects.append((name, xy))
    return objects


def has_object_overlap_contacts(sim: Any, target_slug: str, penetration_tol: float = 1e-4) -> bool:
    """True when the target object geoms penetrate another scene object (not floor/table)."""
    target_prefix = f"{target_slug}_1_"
    allowed = {"floor", "table", "robot0", "mount0"}
    for contact_idx in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_idx]
        if float(contact.dist) > -penetration_tol:
            continue
        geom1 = sim.model.geom_id2name(int(contact.geom1)) or ""
        geom2 = sim.model.geom_id2name(int(contact.geom2)) or ""
        if target_prefix not in geom1 and target_prefix not in geom2:
            continue
        other = geom2 if target_prefix in geom1 else geom1
        if any(token in other for token in allowed):
            continue
        return True
    return False


def _is_basket_body(body_name: str | None) -> bool:
    return body_name is not None and "basket" in body_name.lower()


def compute_obstacle_shift(
    target_xy: np.ndarray,
    obstacle_xy: np.ndarray,
    gap: float,
    target_gap: float,
    *,
    shift_buffer: float = DEFAULT_OBSTACLE_SHIFT_BUFFER,
    min_shift: float = DEFAULT_OBSTACLE_MIN_SHIFT,
) -> np.ndarray:
    """Push a distractor away from the target until ``target_gap`` AABB clearance."""
    target_xy = np.asarray(target_xy, dtype=np.float32).reshape(2)
    obstacle_xy = np.asarray(obstacle_xy, dtype=np.float32).reshape(2)
    delta = obstacle_xy - target_xy
    if float(np.linalg.norm(delta)) < 1e-6:
        delta = np.array([1.0, 0.0], dtype=np.float32)
    direction = _unit_xy(delta)
    clearance_needed = max(target_gap - gap + shift_buffer, min_shift)
    return (direction * clearance_needed).astype(np.float32)


def _enforce_distractor_standoff(
    target_xy: np.ndarray,
    collision_model: SceneCollisionModel,
    scene_objects: list[SceneObjectInfo],
    obstacle_xy: dict[str, np.ndarray],
    range_low: np.ndarray,
    range_high: np.ndarray,
    standoff_gap: float = DEFAULT_DISTRACTOR_STANDOFF_GAP,
    max_iters_per_obj: int = 10,
) -> None:
    """Push every non-basket distractor to ``standoff_gap`` from the target for gripper access."""
    target_xy = np.asarray(target_xy, dtype=np.float32).reshape(2)
    for obj in scene_objects:
        if obj.is_basket:
            continue
        for _ in range(max_iters_per_obj):
            gap = min_geom_gap_to_obstacle(collision_model, target_xy, obj.body_name, obstacle_xy)
            if gap >= standoff_gap:
                break
            obs_xy = obstacle_xy[obj.body_name]
            shift = compute_obstacle_shift(target_xy, obs_xy, gap, standoff_gap)
            obstacle_xy[obj.body_name] = _clamp_xy_to_workspace(obs_xy + shift, range_low, range_high)


def _apply_probe_scene(
    sim: Any,
    target_joint: str,
    target_xyz: np.ndarray,
    target_quat: np.ndarray | None,
    scene_objects: list[SceneObjectInfo],
    obstacle_xy: dict[str, np.ndarray],
) -> None:
    for obj in scene_objects:
        xy = obstacle_xy[obj.body_name]
        _apply_object_qpos(
            sim,
            obj.joint_name,
            np.array([xy[0], xy[1], obj.init_xyz[2]], dtype=np.float32),
            quat=obj.init_quat,
        )
    _apply_object_qpos(sim, target_joint, target_xyz, quat=target_quat)


def _obstacle_adjustments(
    scene_objects: list[SceneObjectInfo],
    obstacle_xy: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    adjustments: list[dict[str, Any]] = []
    for obj in scene_objects:
        placed_xy = obstacle_xy[obj.body_name]
        if float(np.linalg.norm(placed_xy - obj.init_xy)) < 1e-5:
            continue
        adjustments.append(
            {
                "body": obj.body_name,
                "slug": obj.slug,
                "joint": obj.joint_name,
                "init_xy": obj.init_xy.tolist(),
                "placed_xy": placed_xy.tolist(),
                "z": float(obj.init_xyz[2]),
                "quat": obj.init_quat.tolist(),
            }
        )
    return adjustments


def resolve_probe_placement(
    raw_xy: np.ndarray,
    collision_model: SceneCollisionModel,
    scene_objects: list[SceneObjectInfo],
    support_xy: np.ndarray,
    z: float,
    sim: Any,
    target_slug: str,
    target_joint: str,
    target_quat: np.ndarray | None,
    range_low: np.ndarray,
    range_high: np.ndarray,
    min_geom_gap: float = DEFAULT_MIN_GEOM_GAP,
    max_target_retreat: float = DEFAULT_MAX_ALONG_RETREAT_P5,
    max_iters: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve collisions by displacing distractors or retreating the target from EEF support."""
    raw_xy = np.asarray(raw_xy, dtype=np.float32).reshape(2)
    support_xy = np.asarray(support_xy, dtype=np.float32).reshape(2)
    range_low = np.asarray(range_low, dtype=np.float32).reshape(-1)
    range_high = np.asarray(range_high, dtype=np.float32).reshape(-1)
    target_xy = _clamp_xy_to_workspace(raw_xy, range_low, range_high)
    obstacle_xy = {obj.body_name: obj.init_xy.copy() for obj in scene_objects}
    objects_by_body = {obj.body_name: obj for obj in scene_objects}

    collision_strategy = "clear"
    target_retreat = 0.0
    closest: str | None = None
    geom_gap = float("inf")

    for _ in range(max_iters):
        geom_gap, closest = min_geom_gap_xy(collision_model, target_xy, obstacle_xy)
        target_xyz = np.array([target_xy[0], target_xy[1], z], dtype=np.float32)
        _apply_probe_scene(sim, target_joint, target_xyz, target_quat, scene_objects, obstacle_xy)
        contact_overlap = has_object_overlap_contacts(sim, target_slug)
        in_workspace = _xy_in_workspace(target_xy, range_low, range_high)
        if geom_gap >= min_geom_gap and not contact_overlap and in_workspace:
            collision_strategy = "clear"
            break

        if _is_basket_body(closest):
            away_unit = _unit_xy(target_xy - support_xy)
            step = max(min_geom_gap - geom_gap + 0.01, 0.02)
            if target_retreat + step > max_target_retreat:
                break
            target_xy = _clamp_xy_to_workspace(target_xy + away_unit * step, range_low, range_high)
            target_retreat += step
            collision_strategy = "retreat_from_support"
            continue

        if closest is None or closest not in objects_by_body:
            break

        obs_xy = obstacle_xy[closest]
        shift = compute_obstacle_shift(
            target_xy,
            obs_xy,
            geom_gap,
            DEFAULT_DISTRACTOR_STANDOFF_GAP,
        )
        obstacle_xy[closest] = _clamp_xy_to_workspace(obs_xy + shift, range_low, range_high)
        collision_strategy = "displace_obstacle"

    _enforce_distractor_standoff(
        target_xy,
        collision_model,
        scene_objects,
        obstacle_xy,
        range_low,
        range_high,
    )
    target_xyz = np.array([target_xy[0], target_xy[1], z], dtype=np.float32)
    _apply_probe_scene(sim, target_joint, target_xyz, target_quat, scene_objects, obstacle_xy)

    contact_overlap = has_object_overlap_contacts(sim, target_slug)
    in_workspace = _xy_in_workspace(target_xy, range_low, range_high)
    geom_gap, closest = min_geom_gap_xy(collision_model, target_xy, obstacle_xy)
    collision_unresolved = geom_gap < min_geom_gap or contact_overlap or not in_workspace
    meta = {
        "raw_xy": raw_xy.tolist(),
        "adjusted_xy": target_xy.tolist(),
        "support_xy": support_xy.tolist(),
        "target_retreat_from_support": float(target_retreat),
        "collision_strategy": collision_strategy,
        "obstacle_adjustments": _obstacle_adjustments(scene_objects, obstacle_xy),
        "distractor_standoff_gap": DEFAULT_DISTRACTOR_STANDOFF_GAP,
        "min_geom_gap": min_geom_gap,
        "geom_gap": float(geom_gap),
        "closest_obstacle": closest,
        "in_workspace": bool(in_workspace),
        "collision_unresolved": bool(collision_unresolved),
        "contact_overlap": bool(contact_overlap),
        "visible_in_agentview": bool(
            agentview_contains(sim, np.array([target_xy[0], target_xy[1], z], dtype=np.float32))
        ),
    }
    return target_xy.astype(np.float32), meta


def _refine_trail_point(
    sim_env: SimProbeEnv,
    raw_xy: np.ndarray,
    role: str,
    collision_model: SceneCollisionModel,
    scene_objects: list[SceneObjectInfo],
    support_xy: np.ndarray,
    z: float,
    target_slug: str,
    range_low: np.ndarray,
    range_high: np.ndarray,
    min_geom_gap: float,
    max_target_retreat: float = DEFAULT_MAX_ALONG_RETREAT,
) -> tuple[np.ndarray, dict[str, Any]]:
    sim_env.restore_baseline_scene()
    adj_xy, meta = resolve_probe_placement(
        raw_xy,
        collision_model,
        scene_objects,
        support_xy,
        z,
        sim_env.sim,
        target_slug,
        sim_env.joint_name,
        sim_env.init_quat,
        range_low,
        range_high,
        min_geom_gap=min_geom_gap,
        max_target_retreat=max_target_retreat,
    )
    meta["role"] = role
    return adj_xy.astype(np.float32), meta


def refine_probe_trail(
    sim_env: SimProbeEnv,
    boundary_xy: np.ndarray,
    far_xy: np.ndarray,
    z: float,
    range_low: np.ndarray,
    range_high: np.ndarray,
    train_hist: np.ndarray,
    eval_hist: np.ndarray,
    edges: list[np.ndarray],
    init_xyz: np.ndarray,
    n_points: int = DEFAULT_TRAIL_POINTS,
    min_step_shift: float = DEFAULT_MIN_TRAIL_STEP,
    min_geom_gap: float = DEFAULT_MIN_GEOM_GAP,
    max_lateral_shift: float = DEFAULT_MAX_LATERAL_SHIFT,
    max_along_retreat: float = DEFAULT_MAX_ALONG_RETREAT,
) -> list[tuple[np.ndarray, str, dict[str, Any]]]:
    """Anchor p5 first, then place p1 on the support boundary and p2–p4 evenly toward p5."""
    target_slug = target_object_slug(sim_env.task_name)
    collision_model = build_scene_collision_model(sim_env.sim, target_slug)
    scene_objects = list_scene_objects(sim_env.sim, target_slug)
    init_xy = np.asarray(init_xyz, dtype=np.float32).reshape(-1)[:2]
    range_low = np.asarray(range_low, dtype=np.float32).reshape(-1)
    range_high = np.asarray(range_high, dtype=np.float32).reshape(-1)
    support_mask = support_xy_mask(train_hist, eval_hist)

    coarse_boundary = np.asarray(boundary_xy, dtype=np.float32).reshape(2)
    raw_p5 = _clamp_xy_to_workspace(far_xy, range_low, range_high)
    raw_p5 = clamp_far_to_agentview(sim_env.sim, coarse_boundary, raw_p5, z)
    p5_support_xy = find_support_boundary_xy(init_xy, raw_p5, support_mask, edges, table_z=float(z))

    p5_xy, p5_meta = _refine_trail_point(
        sim_env,
        raw_p5,
        TRAIL_ROLES[-1],
        collision_model,
        scene_objects,
        p5_support_xy,
        z,
        target_slug,
        range_low,
        range_high,
        min_geom_gap,
        max_target_retreat=max(DEFAULT_MAX_ALONG_RETREAT_P5, max_along_retreat),
    )

    boundary_xy = find_support_boundary_xy(
        init_xy, p5_xy, support_mask, edges, table_z=float(z)
    ).astype(np.float32)
    trail = build_even_trail(boundary_xy, p5_xy, z, n_points=n_points, min_step_shift=min_step_shift)
    unit = _unit_xy(p5_xy - boundary_xy)

    init_state_id = int(getattr(sim_env, "_init_state_id", 0))
    refined: list[tuple[np.ndarray, str, dict[str, Any]]] = []
    prev_xy = boundary_xy.copy()
    for xyz, role in trail:
        raw_xy = xyz[:2].copy()
        if role == TRAIL_ROLES[-1]:
            adj_xy = p5_xy
            meta = dict(p5_meta)
        else:
            support_xy = find_support_boundary_xy(init_xy, raw_xy, support_mask, edges, table_z=float(z))
            adj_xy, meta = _refine_trail_point(
                sim_env,
                raw_xy,
                role,
                collision_model,
                scene_objects,
                support_xy,
                z,
                target_slug,
                range_low,
                range_high,
                min_geom_gap,
                max_target_retreat=max_along_retreat,
            )
        if role != TRAIL_ROLES[0] and float(np.linalg.norm(adj_xy - prev_xy)) < min_step_shift:
            pushed_xy = prev_xy + unit * min_step_shift
            support_xy = find_support_boundary_xy(init_xy, pushed_xy, support_mask, edges, table_z=float(z))
            adj_xy, meta = _refine_trail_point(
                sim_env,
                pushed_xy,
                role,
                collision_model,
                scene_objects,
                support_xy,
                z,
                target_slug,
                range_low,
                range_high,
                min_geom_gap,
                max_target_retreat=max_along_retreat,
            )
        refined.append((np.array([adj_xy[0], adj_xy[1], z], dtype=np.float32), role, meta))
        prev_xy = adj_xy
    sim_env.reset(init_state_id=init_state_id)
    return refined


def compute_object_probe_positions(
    init_xyz: np.ndarray,
    target_xy: np.ndarray,
    train_hist: np.ndarray,
    eval_hist: np.ndarray,
    edges: list[np.ndarray],
    range_low: np.ndarray,
    range_high: np.ndarray,
    n_probes: int = DEFAULT_TRAIL_POINTS,
    blind_centroids: np.ndarray | None = None,
    min_step_shift: float = DEFAULT_MIN_TRAIL_STEP,
) -> list[tuple[np.ndarray, str]]:
    """Histogram-only trail (p1 boundary → p5 far). Use ``prepare_probe_trail`` with sim for refinement."""
    boundary_xy, far_xy, _unit = compute_trail_endpoints(
        init_xyz,
        target_xy,
        train_hist,
        eval_hist,
        edges,
        range_low,
        range_high,
        blind_centroids=blind_centroids,
    )
    z = float(np.asarray(init_xyz, dtype=np.float32).reshape(-1)[2])
    return build_even_trail(boundary_xy, far_xy, z, n_points=n_probes, min_step_shift=min_step_shift)


def prepare_probe_trail(
    sim_env: SimProbeEnv,
    init_xyz: np.ndarray,
    target_xy: np.ndarray,
    train_hist: np.ndarray,
    eval_hist: np.ndarray,
    edges: list[np.ndarray],
    range_low: np.ndarray,
    range_high: np.ndarray,
    blind_centroids: np.ndarray | None = None,
    n_points: int = DEFAULT_TRAIL_POINTS,
    min_step_shift: float = DEFAULT_MIN_TRAIL_STEP,
) -> list[tuple[np.ndarray, str, dict[str, Any]]]:
    """Compute coverage endpoints then refine with sim visibility and collision checks."""
    boundary_xy, far_xy, _unit = compute_trail_endpoints(
        init_xyz,
        target_xy,
        train_hist,
        eval_hist,
        edges,
        range_low,
        range_high,
        blind_centroids=blind_centroids,
    )
    z = float(np.asarray(init_xyz, dtype=np.float32).reshape(-1)[2])
    return refine_probe_trail(
        sim_env,
        boundary_xy,
        far_xy,
        z,
        range_low,
        range_high,
        train_hist,
        eval_hist,
        edges,
        init_xyz,
        n_points=n_points,
        min_step_shift=min_step_shift,
    )


def interpolate_object_probes(
    init_xyz: np.ndarray,
    target_xy: np.ndarray,
    n_probes: int = DEFAULT_TRAIL_POINTS,
    train_hist: np.ndarray | None = None,
    eval_hist: np.ndarray | None = None,
    edges: list[np.ndarray] | None = None,
    range_low: np.ndarray | None = None,
    range_high: np.ndarray | None = None,
    blind_centroids: np.ndarray | None = None,
    sim_env: SimProbeEnv | None = None,
) -> list[tuple[np.ndarray, str]] | list[tuple[np.ndarray, str, dict[str, Any]]]:
    """Return object probe trail; pass ``sim_env`` for collision/visibility refinement."""
    if sim_env is not None and train_hist is not None and eval_hist is not None and edges is not None:
        return prepare_probe_trail(
            sim_env,
            init_xyz,
            target_xy,
            train_hist,
            eval_hist,
            edges,
            np.asarray(range_low if range_low is not None else [-np.inf, -np.inf, -np.inf]),
            np.asarray(range_high if range_high is not None else [np.inf, np.inf, np.inf]),
            blind_centroids=blind_centroids,
            n_points=n_probes,
        )
    if train_hist is not None and eval_hist is not None and edges is not None:
        return compute_object_probe_positions(
            init_xyz,
            target_xy,
            train_hist,
            eval_hist,
            edges,
            np.asarray(range_low if range_low is not None else [-np.inf, -np.inf, -np.inf]),
            np.asarray(range_high if range_high is not None else [np.inf, np.inf, np.inf]),
            n_probes=n_probes,
            blind_centroids=blind_centroids,
        )
    init = np.asarray(init_xyz, dtype=np.float32).reshape(-1)
    target = np.array([target_xy[0], target_xy[1], init[2]], dtype=np.float32)
    return [
        ((1.0 - alpha) * init[:3] + alpha * target, f"step_{step}")
        for step in range(1, n_probes + 1)
        for alpha in [step / n_probes]
    ]


def filter_feasible_bins(
    centroids: np.ndarray,
    init_object_xyz: np.ndarray,
    range_low: np.ndarray,
    range_high: np.ndarray,
    min_shift: float = 0.0,
) -> np.ndarray:
    """Return indices of centroids whose XY is in-range and far enough from init."""
    if len(centroids) == 0:
        return np.zeros((0,), dtype=np.int64)
    xy = centroids[:, :2]
    init_xy = np.asarray(init_object_xyz, dtype=np.float32).reshape(-1)[:2]
    in_range = (
        (xy[:, 0] >= range_low[0])
        & (xy[:, 0] <= range_high[0])
        & (xy[:, 1] >= range_low[1])
        & (xy[:, 1] <= range_high[1])
    )
    shift = np.linalg.norm(xy - init_xy, axis=1)
    keep = np.where(in_range & (shift >= min_shift))[0]
    return keep.astype(np.int64)


def scale_min_shift(scale: str) -> float:
    if scale == "far":
        return 0.10
    if scale == "moderate":
        return 0.05
    return 0.0


def _joint_qpos_adr(sim: Any, joint_name: str) -> tuple[int, int]:
    jid = sim.model.joint_name2id(joint_name)
    qpos_adr = int(sim.model.jnt_qposadr[jid])
    dof_adr = int(sim.model.jnt_dofadr[jid])
    return qpos_adr, dof_adr


def get_object_qpos(sim: Any, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    qpos_adr, _ = _joint_qpos_adr(sim, joint_name)
    qpos = np.asarray(sim.data.qpos[qpos_adr : qpos_adr + 7], dtype=np.float32)
    return qpos[:3].copy(), qpos[3:7].copy()


def _apply_object_qpos(
    sim: Any,
    joint_name: str,
    xyz: np.ndarray,
    quat: np.ndarray | None = None,
) -> None:
    """Write a free-joint pose and refresh kinematics without advancing physics."""
    qpos_adr, dof_adr = _joint_qpos_adr(sim, joint_name)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    if quat is None:
        quat = np.asarray(sim.data.qpos[qpos_adr + 3 : qpos_adr + 7], dtype=np.float64)
    else:
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
    sim.data.qpos[qpos_adr : qpos_adr + 3] = xyz
    sim.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
    sim.data.qvel[dof_adr : dof_adr + 6] = 0.0
    sim.forward()


def set_object_qpos(
    env: Any,
    joint_name: str,
    xyz: np.ndarray,
    quat: np.ndarray | None = None,
    *,
    physics_settle: bool = False,
) -> Any:
    """Write a free-joint object pose into the live LIBERO sim and refresh obs."""
    from lerobot.envs.libero import get_libero_dummy_action

    inner = env._env
    sim = inner.env.sim
    _apply_object_qpos(sim, joint_name, xyz, quat=quat)
    raw = inner.regenerate_obs_from_state(inner.get_sim_state())
    if physics_settle:
        for _ in range(getattr(env, "num_steps_wait", 10)):
            raw, _, _, _ = inner.step(get_libero_dummy_action())
    return env._format_raw_obs(raw)


class SimProbeEnv:
    """Single-task LIBERO env with object teleport helpers."""

    def __init__(
        self,
        env_task_id: int,
        observation_height: int = 360,
        observation_width: int = 360,
        control_freq: int = 20,
        control_mode: str = "relative",
    ):
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv

        suite = benchmark.get_benchmark_dict()["libero_object"]()
        self.env = LiberoEnv(
            task_suite=suite,
            task_id=env_task_id,
            task_suite_name="libero_object",
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            observation_height=observation_height,
            observation_width=observation_width,
            init_states=True,
            episode_index=0,
            n_envs=1,
            control_freq=control_freq,
            control_mode=control_mode,
        )
        self.env_task_id = env_task_id
        self.task_name = self.env.task
        self.task_description = self.env.task_description
        self.joint_name = target_object_joint(self.task_name)
        self.init_xyz: np.ndarray | None = None
        self.init_quat: np.ndarray | None = None
        self._init_state_id: int = 0
        self._baseline_sim_state: np.ndarray | None = None

    @property
    def sim(self) -> Any:
        assert self.env._env is not None
        return self.env._env.env.sim

    def _inner_env(self) -> Any:
        assert self.env._env is not None
        return self.env._env

    def _task_env(self) -> Any:
        return self._inner_env().env

    def _clear_episode_flags(self) -> None:
        """Allow closed-loop rollouts after a prior episode ended in success."""
        task_env = self._task_env()
        task_env.done = False
        task_env.timestep = 0

    def _cache_baseline_scene(self) -> None:
        inner = self._inner_env()
        self._baseline_sim_state = np.asarray(inner.get_sim_state(), dtype=np.float64).copy()

    def restore_baseline_scene(self) -> None:
        """Restore robot, distractors, and basket to the settled init-state snapshot."""
        if self._baseline_sim_state is None:
            raise RuntimeError("Call reset() before restore_baseline_scene()")
        inner = self._inner_env()
        inner.regenerate_obs_from_state(self._baseline_sim_state)
        self._clear_episode_flags()

    def reset(self, init_state_id: int = 0) -> dict[str, Any]:
        self._init_state_id = int(init_state_id)
        self.env.init_state_id = self._init_state_id
        obs, _info = self.env.reset()
        self.init_xyz, self.init_quat = get_object_qpos(self.sim, self.joint_name)
        self._cache_baseline_scene()
        self._clear_episode_flags()
        return obs

    def place_object(
        self,
        xyz: np.ndarray,
        quat: np.ndarray | None = None,
        obstacle_adjustments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if quat is None:
            quat = self.init_quat
        self.restore_baseline_scene()
        if obstacle_adjustments:
            for adj in obstacle_adjustments:
                placed_xy = np.asarray(adj["placed_xy"], dtype=np.float32).reshape(2)
                obj_z = float(adj.get("z", xyz[2]))
                obj_quat = adj.get("quat")
                quat_arr = np.asarray(obj_quat, dtype=np.float32) if obj_quat is not None else None
                set_object_qpos(
                    self.env,
                    str(adj["joint"]),
                    np.array([placed_xy[0], placed_xy[1], obj_z], dtype=np.float32),
                    quat=quat_arr,
                    physics_settle=False,
                )
        return set_object_qpos(self.env, self.joint_name, xyz, quat=quat, physics_settle=False)

    def current_object_xyz(self) -> np.ndarray:
        xyz, _ = get_object_qpos(self.sim, self.joint_name)
        return xyz

    def current_eef_xyz(self) -> np.ndarray:
        return np.asarray(self.sim.data.site_xpos[self._eef_site_id()], dtype=np.float32)

    def _eef_site_id(self) -> int:
        for name in ("gripper0_grip_site", "robot0_grip_site"):
            try:
                return int(self.sim.model.site_name2id(name))
            except (ValueError, KeyError, TypeError):
                continue
        raise KeyError("Could not find gripper site in LIBERO sim")

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self.env.step(np.asarray(action, dtype=np.float32).reshape(-1))

    def close(self) -> None:
        self.env.close()


def _unsqueeze_robot_state(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _unsqueeze_robot_state(value) for key, value in node.items()}
    if isinstance(node, np.ndarray) and node.ndim >= 1:
        return node[None, ...]
    return node


def build_policy_obs(
    raw_obs: dict[str, Any],
    task_desc: str,
    env_preprocessor: Any,
    preprocessor: Any,
) -> dict[str, Any]:
    """Match the eval observation path: preprocess → LiberoProcessorStep → policy."""
    from lerobot.envs.utils import preprocess_observation

    batched = dict(raw_obs)
    if "robot_state" in batched:
        batched["robot_state"] = _unsqueeze_robot_state(batched["robot_state"])
    observation = preprocess_observation(batched)
    observation["task"] = [task_desc]
    observation = env_preprocessor(observation)
    observation = preprocessor(observation)
    return observation


def policy_view_image(raw_obs: dict[str, Any], camera: str = "image") -> np.ndarray:
    """Return the 180-flipped uint8 image the VLA sees (H, W, C)."""
    image = np.asarray(raw_obs["pixels"][camera])
    return np.ascontiguousarray(image[::-1, ::-1])


def save_probe_frame(path: Path, raw_obs: dict[str, Any], camera: str = "image") -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(policy_view_image(raw_obs, camera=camera)).save(path)
    return path


def plot_sim_probe_overlay(
    train_hist: np.ndarray,
    edges: list[np.ndarray],
    target_xy: np.ndarray,
    target_scales: list[str],
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
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    im = axis.imshow(img, origin="lower", aspect="auto", extent=extent, cmap="Blues")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04, label="train count")
    used: set[str] = set()
    for xy, scale in zip(target_xy, target_scales, strict=True):
        marker, color = SCALE_MARKERS.get(scale, ("x", "crimson"))
        label = f"target ({scale})" if scale not in used else None
        used.add(scale)
        scatter_kw: dict[str, Any] = {
            "c": color,
            "s": 70,
            "marker": marker,
            "linewidths": 1.5,
            "label": label,
            "zorder": 4,
        }
        if marker != "x":
            scatter_kw["edgecolors"] = "black"
        axis.scatter([xy[0]], [xy[1]], **scatter_kw)
    if len(probe_xy):
        sc = axis.scatter(
            probe_xy[:, 0],
            probe_xy[:, 1],
            c=probe_scores,
            cmap="plasma",
            s=36,
            edgecolors="white",
            linewidths=0.4,
            label="probe steps",
            zorder=3,
        )
        fig.colorbar(sc, ax=axis, fraction=0.046, pad=0.08, label="action_delta_vs_baseline")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title or "Sim probe overlay")
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
