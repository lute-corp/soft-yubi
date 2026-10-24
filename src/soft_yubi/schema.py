"""What the converter reads from a yubi bag, and what it writes to LeRobot."""

from lerobot.datasets.utils import DEFAULT_FEATURES
import numpy as np

from soft_yubi.build_ee import ARMS
from soft_yubi.build_ee import BIMANUAL_EE_DIM

ORIGIN_FRAME = "quest_origin"

OBS_STATE = "observation.state"
ACTION = "action"

# The independently-failing streams behind an EE row. The two poses come from the Quest
# bridge and the jaw from the DYNAMIXEL bus, so one can stall while the other does not.
EE_SOURCES = ("pose_left", "pose_right", "jaw")

ARM_EE_NAMES = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"]
EE_NAMES = [f"{side}_{name}" for side in ARMS for name in ARM_EE_NAMES]

GRIPPER_TOPIC = "/joint_states"

CAMERA_TOPICS = {
    # yubi-sw's portable variant (USB camera) or its stationary variant (RealSense; older
    # yubi-sw versions publish image_rect_raw).
    "observation.images.top": (
        "/center_camera/image_raw/compressed",
        "/camera/camera/color/image_raw/compressed",
        "/camera/camera/color/image_rect_raw/compressed",
    ),
    "observation.images.wrist_left": ("/left_camera/image_raw/compressed",),
    "observation.images.wrist_right": ("/right_camera/image_raw/compressed",),
}


def is_fresh_key(camera_key: str) -> str:
    """The freshness column for a camera: `cam`, not `images`, which lerobot reads as an image."""
    return f"is_fresh.cam.{camera_key.rsplit('.', 1)[-1]}"


def is_fresh_state_key(source: str) -> str:
    """The freshness column for one stream behind `observation.state`."""
    return f"is_fresh.state_ee.{source}"


def is_fresh_action_key(source: str) -> str:
    """The freshness column for one stream behind `action`."""
    return f"is_fresh.action_ee.{source}"


def hand_root_frame(side: str) -> str:
    """The TF frame the Quest bridge publishes this glove's pose as."""
    return f"{side}_hand_root"


def build_features(camera_shapes: dict[str, tuple[int, int, int]], fps: int) -> dict[str, dict]:
    """The LeRobot v3 feature map for a converted glove episode.

    Only these columns are written. lerobot casts the data parquet to this schema on
    load and extra columns raise `CastError`, so the set is closed rather than tidy.

    Args:
        camera_shapes: (H, W, C) per camera key, from the bag's own frames.
        fps: The episode's frame rate, stamped into each camera's info block.

    Returns:
        The feature map.
    """
    # Shapes are tuples, not lists: lerobot's validate_frame compares the declared shape
    # against np.shape(value), which is a tuple, with ==. A list silently fails every frame.
    vector = {"dtype": "float32", "shape": (BIMANUAL_EE_DIM,), "names": EE_NAMES}
    features: dict[str, dict] = {**DEFAULT_FEATURES, OBS_STATE: dict(vector)}
    for key, shape in camera_shapes.items():
        features[key] = {
            "dtype": "video",
            "shape": tuple(shape),
            "names": ["height", "width", "channel"],
            "info": {"video.fps": fps},
        }
    features[ACTION] = dict(vector)
    flag = {"dtype": "bool", "shape": (1,), "names": None}
    for source in EE_SOURCES:
        features[is_fresh_state_key(source)] = dict(flag)
        features[is_fresh_action_key(source)] = dict(flag)
    for key in camera_shapes:
        features[is_fresh_key(key)] = dict(flag)
    return features


def build_frame(state: np.ndarray, action: np.ndarray, images: dict, fresh: dict[str, bool], task: str) -> dict:
    """One `LeRobotDataset.add_frame` payload.

    Args:
        state: The (20,) EE state row.
        action: The (20,) action row, `state[t + 1]`.
        images: Decoded (H, W, 3) uint8 frames keyed by camera feature name.
        fresh: Per is_fresh feature key, whether that row was measured or held.
        task: The task prompt.

    Returns:
        The frame dict.
    """
    frame = {
        OBS_STATE: np.asarray(state, dtype=np.float32),
        ACTION: np.asarray(action, dtype=np.float32),
        "task": task,
    }
    frame.update(images)
    frame.update({key: np.array([value], dtype=bool) for key, value in fresh.items()})
    return frame
