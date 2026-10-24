"""Quality-filter converted single-episode datasets.

A glove episode is all demonstration, so every rule scores the whole episode.
"""

import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from soft_yubi.rules import EE_FRESHNESS
from soft_yubi.rules import EpisodeData
from soft_yubi.rules import FreshnessFractionRule
from soft_yubi.rules import FreshnessStallRule
from soft_yubi.rules import Rule

logger = logging.getLogger(__name__)

UMI_RULES: list[Rule] = [
    FreshnessStallRule(max_held_s=0.098),
    FreshnessFractionRule(min_fresh_fraction=0.95, columns=EE_FRESHNESS),
]


def filter_episodes(episode_dirs: list[Path], rules: list[Rule]) -> list[Path]:
    """The episodes that pass every rule, in their original order.

    Args:
        episode_dirs: Single-episode LeRobot datasets.
        rules: Non-empty list of quality rules.

    Returns:
        The passing episode directories.

    Raises:
        ValueError: If `rules` is empty, a dataset holds other than one episode, or every
            episode fails.
    """
    if not rules:
        raise ValueError("filter_episodes requires at least one rule")
    logger.info("Filtering %d episodes with %s", len(episode_dirs), ", ".join(str(r) for r in rules))

    passing = []
    for episode_dir in episode_dirs:
        dataset = LeRobotDataset(repo_id=episode_dir.name, root=episode_dir)
        if dataset.meta.total_episodes != 1:
            raise ValueError(f"Expected a single-episode dataset at {episode_dir}, got {dataset.meta.total_episodes}")
        length = dataset.meta.total_frames
        episode: EpisodeData = {
            "episode_index": 0,
            "length": length,
            "start_idx": 0,
            "end_idx": length,
            "fps": float(dataset.meta.fps),
            "dataset": dataset,
        }
        # Every failing rule, not just the first, so the log says everything that was wrong.
        failed = [rule.name for rule in rules if not rule.evaluate(episode)]
        if failed:
            logger.info("Rejected %s (failed %s)", episode_dir.name, ", ".join(failed))
        else:
            passing.append(episode_dir)

    logger.info("Quality filter kept %d / %d episodes", len(passing), len(episode_dirs))
    if not passing:
        raise ValueError(f"All {len(episode_dirs)} episodes failed the quality filter")
    return passing
