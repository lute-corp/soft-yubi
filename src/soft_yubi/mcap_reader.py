"""Read a yubi rosbag2 MCAP into per-topic :class:`~soft_yubi.synchronize.Series`."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import io
import logging
from pathlib import Path

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from PIL import Image

from soft_yubi.synchronize import Series

logger = logging.getLogger(__name__)

TF_TOPICS = ("/tf", "/tf_static")


@dataclass(frozen=True)
class JpegStore:
    """Undecoded JPEG payloads for one camera topic, addressed by message index."""

    payloads: list[bytes]

    def decode(self, index: int) -> np.ndarray:
        """Decode one frame to an (H, W, 3) uint8 RGB array.

        Args:
            index: Message index, as returned by
                :func:`~soft_yubi.synchronize.resample`.

        Returns:
            The decoded frame.
        """
        with Image.open(io.BytesIO(self.payloads[index])) as img:
            return np.asarray(img.convert("RGB"))

    @property
    def shape(self) -> tuple[int, int, int]:
        """(H, W, C) of the first frame. Every frame on a topic shares it."""
        return self.decode(0).shape


@dataclass(frozen=True)
class Transforms:
    """One TF edge's pose over time: translation, rotation (xyzw), and timestamps."""

    ts_ns: np.ndarray
    pos: np.ndarray
    quat: np.ndarray

    def series(self) -> Series:
        """This edge as a :class:`Series`, for resampling onto the grid."""
        return Series(ts_ns=self.ts_ns, values=None)


@dataclass(frozen=True)
class Bag:
    """One bag's decoded contents, in the shapes the converter consumes.

    Attributes:
        series: Per topic, its timestamps (and positions, for joint topics). Camera
            topics appear here too, carrying timestamps only — their payloads live in
            `jpegs`.
        jpegs: Per camera topic, the undecoded frames.
        tf_edges: Per `(parent_frame, child_frame)`, that edge's pose over time.
        joint_names: Per joint topic, its `name` field.
    """

    series: dict[str, Series]
    jpegs: dict[str, JpegStore]
    tf_edges: dict[tuple[str, str], Transforms]
    joint_names: dict[str, list[str]]


def resolve_camera_topics(path: Path, candidates: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Pick each camera's topic from the bag's summary: the one candidate that has messages.

    Raises:
        ValueError: If the bag has no summary, or a camera has no candidate with messages or more than one.
    """
    with path.open("rb") as handle:
        summary = make_reader(handle).get_summary()
    if summary is None or summary.statistics is None:
        raise ValueError(f"{path}: no summary to resolve camera topics from")
    counts = summary.statistics.channel_message_counts
    present = {ch.topic for cid, ch in summary.channels.items() if counts.get(cid, 0) > 0}
    resolved = {}
    for key, topics in candidates.items():
        found = [t for t in topics if t in present]
        if len(found) != 1:
            images = sorted(t for t in present if t.endswith("/compressed"))
            raise ValueError(
                f"{path}: {key} needs exactly one of {list(topics)} with messages, found {found}; "
                f"image topics in this bag: {images}"
            )
        resolved[key] = found[0]
    return resolved


def read_bag(
    path: Path,
    *,
    camera_topics: Sequence[str],
    scalar_topics: Sequence[str],
) -> Bag:
    """Walk an MCAP once, returning the pieces the converter needs.

    Args:
        path: The `.mcap` file.
        camera_topics: Compressed-image topics; payloads are kept undecoded.
        scalar_topics: `sensor_msgs/JointState` topics, decoded eagerly into arrays.

    Returns:
        The bag's contents. See :class:`Bag`.

    Raises:
        ValueError: If a requested topic carries no messages.
    """
    wanted = set(camera_topics) | set(scalar_topics) | set(TF_TOPICS)
    stamps: dict[str, list[int]] = {t: [] for t in wanted}
    payloads: dict[str, list[bytes]] = {t: [] for t in camera_topics}
    scalars: dict[str, list[list[float]]] = {t: [] for t in scalar_topics}
    joint_names: dict[str, list[str]] = {}
    edges: dict[tuple[str, str], list[tuple[int, list[float], list[float]]]] = {}

    with path.open("rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages(topics=sorted(wanted)):
            topic = channel.topic
            stamps[topic].append(message.log_time)
            if topic in payloads:
                payloads[topic].append(bytes(decoded.data))
            elif topic in scalars:
                scalars[topic].append(list(decoded.position))
                joint_names.setdefault(topic, list(decoded.name))
            else:
                _collect_transforms(edges, decoded, message.log_time)

    series: dict[str, Series] = {}
    for topic in sorted(wanted - set(TF_TOPICS)):
        if not stamps[topic]:
            raise ValueError(f"{path}: topic {topic!r} carries no messages; was its publisher running?")
        values = np.asarray(scalars[topic], dtype=np.float64) if topic in scalars else None
        series[topic] = Series(ts_ns=np.asarray(stamps[topic], dtype=np.uint64), values=values)

    jpegs = {topic: JpegStore(payloads=payloads[topic]) for topic in camera_topics}
    tf_edges = {
        edge: Transforms(
            ts_ns=np.asarray([t for t, _, _ in rows], dtype=np.uint64),
            pos=np.asarray([p for _, p, _ in rows], dtype=np.float64),
            quat=np.asarray([q for _, _, q in rows], dtype=np.float64),
        )
        for edge, rows in edges.items()
    }
    logger.debug("%s: %d topics, %d tf edges", path.name, len(series), len(tf_edges))
    return Bag(series=series, jpegs=jpegs, tf_edges=tf_edges, joint_names=joint_names)


def joint_index(names: Iterable[str], side: str) -> int:
    """Index of one hand's jaw channel in `/joint_states`, matched on the side prefix.

    Both channels are named `<side>_right_finger_joint` — the actuated jaw is the
    *right* finger on both grippers — so a substring test for the side picks the wrong
    one. Matching the leading `"<side>_"` is the only safe form.

    Args:
        names: The `name` field of a `/joint_states` message.
        side: `"left"` or `"right"`.

    Returns:
        The channel's index.

    Raises:
        ValueError: If the side does not name exactly one channel. A single-glove
            recording looks like this, and converting one would leave half of the
            20-wide row unmeasured.
    """
    matches = [i for i, name in enumerate(names) if name.startswith(f"{side}_")]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {side!r} channel in /joint_states, got {matches} from "
            f"{list(names)}. Both grippers' encoders must publish (left_* and right_*); a "
            "single-gripper recording cannot be converted."
        )
    return matches[0]


def _collect_transforms(
    edges: dict[tuple[str, str], list[tuple[int, list[float], list[float]]]],
    decoded: object,
    log_time: int,
) -> None:
    """Bucket one TFMessage's transforms into per-edge rows."""
    for transform in decoded.transforms:
        edge = (transform.header.frame_id, transform.child_frame_id)
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        edges.setdefault(edge, []).append(
            (
                log_time,
                [translation.x, translation.y, translation.z],
                [rotation.x, rotation.y, rotation.z, rotation.w],
            )
        )
