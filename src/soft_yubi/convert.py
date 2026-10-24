"""Convert one raw yubi MCAP into a single-episode LeRobot v3 dataset."""

from collections.abc import Iterator
import json
import logging
from pathlib import Path
import shutil
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from soft_yubi import schema
from soft_yubi.build_ee import ARMS
from soft_yubi.build_ee import build_bimanual_ee_states
from soft_yubi.build_ee import shift_for_action
from soft_yubi.mcap_reader import Bag
from soft_yubi.mcap_reader import Transforms
from soft_yubi.mcap_reader import joint_index
from soft_yubi.mcap_reader import read_bag
from soft_yubi.mcap_reader import resolve_camera_topics
from soft_yubi.synchronize import Series
from soft_yubi.synchronize import Timeline
from soft_yubi.synchronize import build_timeline
from soft_yubi.synchronize import resample
from soft_yubi.synchronize import segment_span

logger = logging.getLogger(__name__)

# Frames per camera encoder queue: ~200 MB at 640x480x3.
QUEUE_LEN = 217
SUCCESS_MARKER = "_SUCCESS"


def _hand_root_edges(tf_edges: dict[tuple[str, str], Transforms], bag: Path) -> dict[str, Transforms]:
    """Locate `quest_origin -> <side>_hand_root` per hand.

    Raises:
        ValueError: If either pose is not a direct edge, which means the TF tree
            changed and this converter's no-FK assumption no longer holds.
    """
    found = {}
    for side in ARMS:
        edge = (schema.ORIGIN_FRAME, schema.hand_root_frame(side))
        transforms = tf_edges.get(edge)
        if transforms is None:
            raise ValueError(
                f"{bag}: no {edge[0]} -> {edge[1]} transforms on /tf. yubi-sw's Quest bridge "
                "publishes them while the controllers are tracked; check it was running and the "
                "controllers were on."
            )
        found[side] = transforms
    return found


def convert_bag(bag: Path, dest: Path, *, task: str, fps: int, vcodec: str, encoder_threads: int) -> Path:
    """Convert one `.mcap` into one single-episode LeRobot v3 dataset at `dest`.

    Idempotent: a `dest` carrying `_SUCCESS` is complete and returned as is; anything
    else at `dest` is a partial episode from an interrupted run and is discarded.

    Args:
        bag: The `.mcap` file. `meta.json` must sit beside it.
        dest: The episode dataset directory to write.
        task: The task prompt written into every frame.
        fps: The synchronisation grid rate.
        vcodec: Video codec passed to lerobot.
        encoder_threads: SVT-AV1 `lp=` per camera encoder.

    Returns:
        `dest`.

    Raises:
        ValueError: If `meta.json` is missing or carries other than one segment, a
            required topic is absent, the segment does not overlap the timeline, or
            the TF tree does not publish both hand poses as direct edges.
    """
    if (dest / SUCCESS_MARKER).exists():
        logger.debug("%s already converted, skipping", bag.name)
        return dest
    if dest.exists():
        logger.warning("%s: discarding a partial episode from an interrupted run", bag.name)
        shutil.rmtree(dest)

    meta_path = bag.parent / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"{bag}: no meta.json beside the bag; yubi-core writes it when a recording is saved")
    meta = json.loads(meta_path.read_text())

    segments = meta.get("segments") or []
    if not segments:
        raise ValueError(
            f"{bag}: meta.json has no segment, so there is no span to convert. Accept the "
            "episode once before saving the recording."
        )
    if len(segments) > 1:
        raise ValueError(
            f"{bag}: meta.json has {len(segments)} segments; one is expected. Several segments "
            "come from a rejected take and its retry, or from subtask labels, in one recording. "
            "Save one accepted episode per recording."
        )
    segment = segments[0]

    try:
        camera_topics = resolve_camera_topics(bag, schema.CAMERA_TOPICS)
    except ValueError as exc:
        raise ValueError(
            f"{exc}. Record with yubi-sw (portable or stationary variant), which publishes the wrist "
            "cameras and the top camera."
        ) from exc
    contents = read_bag(bag, camera_topics=list(camera_topics.values()), scalar_topics=[schema.GRIPPER_TOPIC])

    # The pose edges are consumed streams, so they belong in the bounds: read_bag keeps
    # /tf out of `series`, and without them a Quest bridge that starts after the cameras
    # puts tick 0 before its first message.
    edges = _hand_root_edges(contents.tf_edges, bag)
    pose_series = {f"tf:{side}_hand_root": Series(ts_ns=e.ts_ns, values=None) for side, e in edges.items()}
    timeline = build_timeline({**contents.series, **pose_series}, fps)
    logger.debug(
        "%s: %d ticks at %d fps over %.2fs", bag.name, len(timeline), fps, len(timeline) * timeline.frame_spacing_s
    )

    poses, pose_fresh = _hand_root_poses(edges, timeline)
    jaw, jaw_fresh = _jaw_values(contents, timeline, bag)
    states = build_bimanual_ee_states(poses, jaw)
    # Kept per stream rather than reduced to one flag: the poses and the jaw fail
    # independently, so a single bool could not say which stalled.
    source_fresh = {"pose_left": pose_fresh["left"], "pose_right": pose_fresh["right"], "jaw": jaw_fresh}

    cameras = {key: resample(contents.series[topic], timeline) for key, topic in camera_topics.items()}
    camera_shapes = {key: contents.jpegs[topic].shape for key, topic in camera_topics.items()}

    lo, hi = segment_span(timeline, segment["start_time"], segment["end_time"])
    if hi < lo:
        raise ValueError(
            f"{bag}: the segment does not overlap the synchronized timeline "
            f"[{int(timeline.grid_ns[0])}, {int(timeline.grid_ns[-1])}]"
        )

    # Shift within the segment, not across the whole bag: the segment is a sub-window of
    # the recording, so a global shift would make the final action a state from after the
    # segment ended.
    segment_states = states[lo : hi + 1]
    segment_actions = shift_for_action(segment_states)
    state_fresh = {k: v[lo : hi + 1] for k, v in source_fresh.items()}
    action_fresh = {k: shift_for_action(v) for k, v in state_fresh.items()}
    frames = (
        schema.build_frame(
            segment_states[i],
            segment_actions[i],
            {key: contents.jpegs[topic].decode(int(cameras[key][0][lo + i])) for key, topic in camera_topics.items()},
            {
                **{schema.is_fresh_state_key(k): bool(state_fresh[k][i]) for k in schema.EE_SOURCES},
                **{schema.is_fresh_action_key(k): bool(action_fresh[k][i]) for k in schema.EE_SOURCES},
                **{schema.is_fresh_key(key): bool(cameras[key][1][lo + i]) for key in camera_topics},
            },
            task=task,
        )
        for i in range(hi - lo + 1)
    )
    _write_episode(dest, schema.build_features(camera_shapes, fps), frames, fps, vcodec, encoder_threads)
    logger.debug("%s: wrote %d frames", bag.name, hi - lo + 1)
    return dest


def _write_episode(
    dest: Path, features: dict[str, dict], frames: Iterator[dict], fps: int, vcodec: str, encoder_threads: int
) -> None:
    """Stream frames into a new LeRobotDataset at `dest`, then mark it complete.

    A failure removes `dest`, so only a directory carrying `_SUCCESS` is ever a whole episode.
    """
    dataset = LeRobotDataset.create(
        repo_id=dest.name,
        root=dest,
        fps=fps,
        features=features,
        use_videos=True,
        vcodec=vcodec,
        streaming_encoding=True,
        encoder_threads=encoder_threads,
        encoder_queue_maxsize=QUEUE_LEN,
    )
    try:
        for frame in frames:
            _await_queue_capacity(dataset, QUEUE_LEN)
            dataset.add_frame(frame)
        dataset.save_episode()
        dataset.writer.stop_image_writer()
        dataset.finalize()
    except BaseException:
        logger.exception("%s failed; discarding the partial episode", dest.name)
        dataset.clear_episode_buffer()
        dataset.writer.stop_image_writer()
        dataset.finalize()
        shutil.rmtree(dest, ignore_errors=True)
        raise
    (dest / SUCCESS_MARKER).touch()


def _await_queue_capacity(dataset: LeRobotDataset, limit: int) -> None:
    """Block until every camera's encoder queue has room for one more frame.

    Args:
        dataset: The dataset being written.
        limit: Queue depth to stay below.
    """
    encoder = getattr(dataset.writer, "_streaming_encoder", None)
    if encoder is None:
        return
    queues: dict[str, object] = getattr(encoder, "_frame_queues", {})
    threads: dict[str, object] = getattr(encoder, "_threads", {})
    while any(q.qsize() >= limit for q in queues.values()):
        if any(not t.is_alive() for t in threads.values()):
            return
        time.sleep(0.005)


def _hand_root_poses(
    edges: dict[str, Transforms],
    timeline: Timeline,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    """Resample each hand's `hand_root` pose onto the grid."""
    poses, fresh = {}, {}
    for side, transforms in edges.items():
        pick, is_fresh = resample(Series(ts_ns=transforms.ts_ns, values=None), timeline)
        poses[side] = (transforms.pos[pick], transforms.quat[pick])
        fresh[side] = is_fresh
    return poses, fresh


def _jaw_values(contents: Bag, timeline: Timeline, bag: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Resample each hand's recorded jaw channel onto the grid.

    Raises:
        ValueError: If the topic was not decoded into positions, or its channel names
            do not resolve to exactly one per hand (a single-glove recording).
    """
    jaw = contents.series[schema.GRIPPER_TOPIC]
    if jaw.values is None or jaw.values.ndim != 2:
        raise ValueError(f"{bag}: {schema.GRIPPER_TOPIC} was not decoded into positions")
    names = contents.joint_names.get(schema.GRIPPER_TOPIC)
    if not names:
        raise ValueError(f"{bag}: {schema.GRIPPER_TOPIC} carries no channel names")
    pick, is_fresh = resample(jaw, timeline)
    return {side: jaw.values[pick, joint_index(names, side)] for side in ARMS}, is_fresh
