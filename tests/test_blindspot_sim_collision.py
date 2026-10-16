import numpy as np

from blindspot_sim_utils import SceneCollisionModel, _aabb_edge_gap, min_geom_gap_xy


def test_aabb_edge_gap_separated():
    a = np.array([0.0, 0.1, 0.0, 0.1], dtype=np.float32)
    b = np.array([0.2, 0.3, 0.0, 0.1], dtype=np.float32)
    assert np.isclose(_aabb_edge_gap(a, b), 0.1)


def test_aabb_edge_gap_overlap():
    a = np.array([0.0, 0.2, 0.0, 0.2], dtype=np.float32)
    b = np.array([0.1, 0.3, 0.1, 0.3], dtype=np.float32)
    assert _aabb_edge_gap(a, b) < 0.0


def test_min_geom_gap_xy_uses_target_translation():
    model = SceneCollisionModel(
        init_body_xy=np.array([0.0, 0.0], dtype=np.float32),
        target_rel_centers=np.array([[0.0, 0.0]], dtype=np.float32),
        target_half_extents=np.array([[0.05, 0.05]], dtype=np.float32),
        obstacle_aabbs=[
            (
                "basket_1_main",
                np.array([[0.2, 0.4, 0.0, 0.2]], dtype=np.float32),
            )
        ],
        obstacle_init_xy={"basket_1_main": np.array([0.3, 0.1], dtype=np.float32)},
    )
    gap_far, _ = min_geom_gap_xy(model, np.array([0.0, 0.0], dtype=np.float32))
    gap_near, _ = min_geom_gap_xy(model, np.array([0.1, 0.0], dtype=np.float32))
    assert gap_far > gap_near
    assert np.isclose(gap_far, 0.15)
    assert np.isclose(gap_near, 0.05)


def test_min_geom_gap_xy_respects_obstacle_displacement():
    model = SceneCollisionModel(
        init_body_xy=np.array([0.0, 0.0], dtype=np.float32),
        target_rel_centers=np.array([[0.0, 0.0]], dtype=np.float32),
        target_half_extents=np.array([[0.05, 0.05]], dtype=np.float32),
        obstacle_aabbs=[
            (
                "milk_1_main",
                np.array([[0.08, 0.18, -0.02, 0.08]], dtype=np.float32),
            )
        ],
        obstacle_init_xy={"milk_1_main": np.array([0.13, 0.03], dtype=np.float32)},
    )
    gap_init, _ = min_geom_gap_xy(model, np.array([0.0, 0.0], dtype=np.float32))
    gap_shifted, _ = min_geom_gap_xy(
        model,
        np.array([0.0, 0.0], dtype=np.float32),
        obstacle_xy={"milk_1_main": np.array([0.25, 0.03], dtype=np.float32)},
    )
    assert gap_shifted > gap_init


def test_compute_obstacle_shift_pushes_away_from_target():
    from blindspot_sim_utils import compute_obstacle_shift

    shift = compute_obstacle_shift(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.1, 0.0], dtype=np.float32),
        gap=-0.01,
        target_gap=0.045,
    )
    assert shift[0] > 0.0
    assert np.isclose(shift[1], 0.0)


def test_enforce_distractor_standoff_pushes_nearby_props():
    from blindspot_sim_utils import SceneObjectInfo, _enforce_distractor_standoff

    model = SceneCollisionModel(
        init_body_xy=np.array([-0.15, 0.13], dtype=np.float32),
        target_rel_centers=np.array([[0.0, 0.0]], dtype=np.float32),
        target_half_extents=np.array([[0.04, 0.04]], dtype=np.float32),
        obstacle_aabbs=[
            (
                "milk_1_main",
                np.array([[-0.19, -0.11, 0.01, 0.11]], dtype=np.float32),
            )
        ],
        obstacle_init_xy={"milk_1_main": np.array([-0.15, 0.06], dtype=np.float32)},
    )
    milk = SceneObjectInfo(
        body_name="milk_1_main",
        slug="milk",
        joint_name="milk_1_joint0",
        init_xyz=np.array([-0.15, 0.06, 0.07], dtype=np.float32),
        init_quat=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    )
    target_xy = np.array([-0.15, 0.13], dtype=np.float32)
    obstacle_xy = {"milk_1_main": milk.init_xy.copy()}
    gap_before = min_geom_gap_xy(model, target_xy, obstacle_xy)[0]
    _enforce_distractor_standoff(
        target_xy,
        model,
        [milk],
        obstacle_xy,
        np.array([-0.3, -0.3, 0.0], dtype=np.float32),
        np.array([0.3, 0.3, 0.3], dtype=np.float32),
        standoff_gap=0.10,
    )
    gap_after = min_geom_gap_xy(model, target_xy, obstacle_xy)[0]
    assert gap_after > gap_before
    assert gap_after >= 0.10 - 1e-4
