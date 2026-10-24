"""Quality rules for filtering episodes before merge.

Each rule is a predicate over an `EpisodeData` dict and returns a list of valid frame ranges. An
empty list means the episode fails the rule. Current rules either pass the
full episode `[(0, length)]` or reject it entirely `[]`; the range
representation is kept for forward compatibility with sub-range acceptance.

Thresholds are expressed in physical units (seconds, same-units/second) and
converted to frames inside `evaluate` using `episode["fps"]`. A fps change in
recording config therefore does NOT silently retune rules.
"""

from abc import ABC
from abc import abstractmethod
from fnmatch import fnmatch
import logging
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)

# Glob selectors scoping a freshness rule to some `is_fresh.*` columns.
ALL_FRESHNESS = ("is_fresh.*",)
EE_FRESHNESS = ("is_fresh.state_ee.*", "is_fresh.action_ee.*")


class EpisodeData(TypedDict):
    """Shape of the input passed to `Rule.evaluate`.

    `dataset` is duck-typed: rules only read `dataset.hf_dataset`.
    """

    episode_index: int
    length: int
    start_idx: int
    end_idx: int
    fps: float
    dataset: object


class Rule(ABC):
    """Base class for quality rules."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, episode: EpisodeData) -> list[tuple[int, int]]:
        """Return valid frame ranges; empty list rejects the episode."""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


def freshness_columns(dataset: object, start_idx: int, end_idx: int) -> dict[str, np.ndarray]:
    """Every `is_fresh.*` column over `[start_idx, end_idx)`, as 1-D bool.

    Args:
        dataset: A LeRobotDataset, duck-typed to its `hf_dataset`.
        start_idx: First frame of the region to read.
        end_idx: One past its last frame.

    Returns:
        Column name to per-frame flags. Empty when the episode carries no such columns,
        which every non-UMI capture does; whether that is an error is the caller's to
        decide.
    """
    columns = [c for c in dataset.hf_dataset.column_names if c.startswith("is_fresh.")]
    if not columns:
        return {}
    rows = dataset.hf_dataset[start_idx:end_idx]
    return {c: np.asarray(rows[c], dtype=bool).reshape(-1) for c in columns}


def _scored_freshness(episode: EpisodeData, patterns: tuple[str, ...]) -> dict[str, np.ndarray]:
    """`freshness_columns` over an episode's scored region, narrowed to `patterns`.

    Args:
        episode: The episode to read.
        patterns: Glob selectors; a column is kept if it matches any of them.

    Returns:
        The selected columns.

    Raises:
        ValueError: If the episode carries no freshness columns, or none match
            `patterns`. A selector matching nothing would otherwise pass every
            episode without evaluating anything.
    """
    columns = freshness_columns(episode["dataset"], episode["start_idx"], episode["end_idx"])
    if not columns:
        raise ValueError(
            f"episode {episode['episode_index']} carries no is_fresh columns for a freshness rule to score"
        )
    selected = {c: v for c, v in columns.items() if any(fnmatch(c, p) for p in patterns)}
    if not selected:
        raise ValueError(
            f"episode {episode['episode_index']}: no is_fresh column matches {patterns}; it has {sorted(columns)}"
        )
    return selected


class FreshnessStallRule(Rule):
    """Reject episodes where any stream was held for more than ``max_held_s`` in a row.

    A held frame repeats the previous value, so its own delta is zero and the jump
    lands on the frame that resumes. When the hand was moving slowly that jump is
    indistinguishable from ordinary motion, so a stall cannot be found from the actions.
    """

    def __init__(self, max_held_s: float = 0.1, columns: tuple[str, ...] = ALL_FRESHNESS):
        super().__init__(name="freshness_stall")
        self.max_held_s = max_held_s
        self.columns = tuple(columns)

    def evaluate(self, episode: EpisodeData) -> list[tuple[int, int]]:
        max_held_frames = self.max_held_s * episode["fps"]
        for column, fresh in _scored_freshness(episode, self.columns).items():
            run = self.longest_run(~fresh)
            if run > max_held_frames:
                logger.debug(
                    "Episode %d: %s held for %d frames > %.1f",
                    episode["episode_index"],
                    column,
                    run,
                    max_held_frames,
                )
                return []
        return [(0, episode["length"])]

    @staticmethod
    def longest_run(flags: np.ndarray) -> int:
        """The longest run of consecutive True in `flags`.

        Sentinel zeros at both ends so a run touching the start or the end of the
        episode still shows up as a rising/falling pair; the diffs then alternate rise,
        fall, rise, fall, and each run's length is one fall minus its rise.
        """
        edges = np.flatnonzero(np.diff(np.r_[0, flags.view(np.int8), 0]))
        return int((edges[1::2] - edges[0::2]).max()) if len(edges) else 0

    def __str__(self) -> str:
        return f"FreshnessStallRule(max_held_s={self.max_held_s}, columns={self.columns})"


class FreshnessFractionRule(Rule):
    """Reject episodes where any stream was measured on ``min_fresh_fraction`` or fewer ticks.

    Catches what the stall rule misses: a source publishing just under the grid rate
    never holds for long, but is interpolated throughout the episode.
    """

    def __init__(self, min_fresh_fraction: float = 0.95, columns: tuple[str, ...] = ALL_FRESHNESS):
        super().__init__(name="freshness_fraction")
        self.min_fresh_fraction = min_fresh_fraction
        self.columns = tuple(columns)

    def evaluate(self, episode: EpisodeData) -> list[tuple[int, int]]:
        for column, fresh in _scored_freshness(episode, self.columns).items():
            fraction = float(fresh.mean())
            if fraction <= self.min_fresh_fraction:
                logger.debug(
                    "Episode %d: %s fresh for %.4f of frames <= %s",
                    episode["episode_index"],
                    column,
                    fraction,
                    self.min_fresh_fraction,
                )
                return []
        return [(0, episode["length"])]

    def __str__(self) -> str:
        return f"FreshnessFractionRule(min_fresh_fraction={self.min_fresh_fraction}, columns={self.columns})"
