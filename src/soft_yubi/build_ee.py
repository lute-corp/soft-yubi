"""Compose the 20-wide EE-native state from glove poses and jaw readings."""

import numpy as np
from scipy.spatial.transform import Rotation

ARMS = ("left", "right")
# Per arm: xyz, rot6d, gripper.
BIMANUAL_EE_DIM = 2 * 10

# The tool tip (TCP) in the glove's `hand_root` frame: the point where the two pads first meet.
TIP_XYZ = np.array([0.07839114949, 0.0, -0.0003999999363])
TIP_ROT = Rotation.from_euler("xyz", [0.0, 1.570796327, 0.0])


def quat_to_rotate6d(q: np.ndarray, scalar_first: bool = True) -> np.ndarray:
    """(..., 4) quaternion -> (..., 6): the rotation matrix's first two columns, row-major."""
    mat = Rotation.from_quat(q, scalar_first=scalar_first).as_matrix()
    return mat[..., :, :2].reshape((*mat.shape[:-2], 6))


def build_bimanual_ee_states(
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    jaw: dict[str, np.ndarray],
) -> np.ndarray:
    """Compose the (N, 20) EE matrix: each hand's recorded `hand_root` moved to the tool tip.

    One transform per hand, moving position *and* orientation. Turning one without the
    other is the silent failure — the poses stay orthonormal and describe a different
    grasp::

        p_tip = p_hand + R_hand * TIP_XYZ
        R_tip = R_hand * TIP_ROT

    Both blocks stay in the frame the poses were recorded in, and both are absolute.

    Args:
        poses: Per side, `(positions (N, 3), quaternions (N, 4) xyzw)` for `hand_root`.
        jaw: Per side, the recorded jaw channel, passed through unchanged.

    Returns:
        The (N, 20) bimanual EE matrix, `[left(10), right(10)]`.

    Raises:
        ValueError: If a pose array is malformed or the two hands disagree on length.
    """
    lengths = {len(poses[s][0]) for s in ARMS} | {len(jaw[s]) for s in ARMS}
    if len(lengths) != 1:
        raise ValueError(
            f"the two hands disagree on frame count: { {s: (len(poses[s][0]), len(jaw[s])) for s in ARMS} }"
        )
    n = lengths.pop()

    blocks = []
    for side in ARMS:
        positions, quats = poses[side]
        if quats.shape[1] != 4 or positions.shape[1] != 3:
            raise ValueError(
                f"{side}: expected (N,3) positions and (N,4) quaternions, got {positions.shape}/{quats.shape}"
            )
        # scalar_first=False because ROS stores quaternions as x, y, z, w.
        hand_rot = Rotation.from_quat(quats, scalar_first=False)
        tcp_pos = positions + hand_rot.apply(TIP_XYZ)
        tcp_rot6d = quat_to_rotate6d((hand_rot * TIP_ROT).as_quat(scalar_first=True))
        blocks.append(np.hstack([tcp_pos, tcp_rot6d, jaw[side].reshape(-1, 1)]))

    rows = np.hstack(blocks).astype(np.float64)
    if rows.shape != (n, BIMANUAL_EE_DIM):
        raise ValueError(f"composed rows have shape {rows.shape}, expected {(n, BIMANUAL_EE_DIM)}")
    return rows


def shift_for_action(states: np.ndarray) -> np.ndarray:
    """`action[t] = state[t + 1]`, with the last row repeated.

    Args:
        states: The per-frame state rows.

    Returns:
        The same rows shifted one step earlier.
    """
    actions = np.empty_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions
