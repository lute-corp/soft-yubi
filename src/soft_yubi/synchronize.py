"""Zero-order-hold resampling of asynchronous topics onto one fixed-rate timeline.

1. Build a uniform grid spanning the recording.
2. For each topic independently, take the last message at or before each tick
   (a backward as-of join).
3. Mark each row `is_fresh` when that pick differs from the previous row's.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class Series:
    """One topic's messages: sorted timestamps plus whatever payload was kept.

    `ts_ns` holds MCAP *log* time, not the ROS `header.stamp`. The two disagree
    badly on this hardware — the hand cameras stamp their headers from a clock
    roughly 0.5 s off the recorder's — and log time is the one clock every topic
    shares.
    """

    ts_ns: np.ndarray
    values: Any

    def __post_init__(self) -> None:
        if self.ts_ns.ndim != 1:
            raise ValueError(f"ts_ns must be 1D, got shape {self.ts_ns.shape}")
        if len(self.ts_ns) == 0:
            raise ValueError("ts_ns is empty; a topic with no messages cannot be synchronized")
        if not np.all(np.diff(self.ts_ns.astype(np.int64)) >= 0):
            raise ValueError("ts_ns must be sorted ascending")


@dataclass(frozen=True)
class Timeline:
    """The uniform grid every topic is resampled onto."""

    grid_ns: np.ndarray
    fps: int

    def __len__(self) -> int:
        return len(self.grid_ns)

    @property
    def frame_spacing_s(self) -> float:
        """Seconds between ticks, derived from the grid rather than from `fps`.

        `build_timeline` truncates the step to whole nanoseconds, so at 30 fps the
        real spacing is 33_333_333 ns, not 1/30 s. Deriving it here keeps the
        `timestamp` column consistent with the grid.
        """
        return (NS_PER_S // self.fps) / NS_PER_S


def build_timeline(series: dict[str, Series], fps: int) -> Timeline:
    """Build the fixed-rate grid spanning every topic.

    `start` is the MAX of each topic's first timestamp, so no tick precedes any stream
    and the backward as-of join in `resample` cannot yield a leading null. `last` is the
    MAX of each topic's last. The step is integer-divided.

    Args:
        series: Every topic that participates in the timeline, keyed by topic name.
        fps: Target frame rate of the output grid.

    Returns:
        The grid, as timestamps in nanoseconds.

    Raises:
        ValueError: If `series` is empty or `fps` is not positive.
    """
    if not series:
        raise ValueError("series is empty; nothing to synchronize")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    start = max(int(s.ts_ns[0]) for s in series.values())
    last = max(int(s.ts_ns[-1]) for s in series.values())
    step = NS_PER_S // fps
    return Timeline(grid_ns=np.arange(start, last + 1, step, dtype=np.uint64), fps=fps)


def resample(series: Series, timeline: Timeline) -> tuple[np.ndarray, np.ndarray]:
    """Resample one topic onto the grid with a backward as-of join.

    Takes the last message at or before each tick, with no tolerance, so a value is held
    for as long as the topic stays silent. `is_fresh` compares the identity of the source
    message rather than its value, and row 0 is always fresh.

    Args:
        series: The topic to resample.
        timeline: The grid to resample onto.

    Returns:
        `(pick, is_fresh)`. `pick[i]` indexes into `series.values` for tick `i`;
        `is_fresh[i]` is False when tick `i` reuses tick `i - 1`'s message.

    Raises:
        ValueError: If the timeline starts before the topic's first message, which
            `build_timeline`'s choice of `start` is supposed to make impossible.
    """
    pick = np.searchsorted(series.ts_ns, timeline.grid_ns, side="right") - 1
    if pick[0] < 0:
        raise ValueError(
            "timeline starts before this topic's first message; build_timeline should have "
            "prevented this by taking start = max of per-topic minima"
        )
    is_fresh = np.empty(len(pick), dtype=bool)
    is_fresh[0] = True
    np.not_equal(pick[1:], pick[:-1], out=is_fresh[1:])
    return pick, is_fresh


def segment_span(timeline: Timeline, start_time_s: float, end_time_s: float) -> tuple[int, int]:
    """Grid indices covered by a `meta.json` segment, inclusive at both ends.

    Segment bounds are wall-clock seconds; the span is every tick with
    `start <= grid_ns <= end`.

    Args:
        timeline: The grid to index into.
        start_time_s: Segment start, epoch seconds.
        end_time_s: Segment end, epoch seconds.

    Returns:
        `(lo, hi)` inclusive grid indices. `hi < lo` when the segment does not
        overlap the timeline at all.
    """
    lo_ns = int(start_time_s * NS_PER_S)
    hi_ns = int(end_time_s * NS_PER_S)
    lo = int(np.searchsorted(timeline.grid_ns, lo_ns, side="left"))
    hi = int(np.searchsorted(timeline.grid_ns, hi_ns, side="right")) - 1
    return lo, hi
