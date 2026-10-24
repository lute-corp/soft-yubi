"""Merge single-episode LeRobot datasets into one."""

import json
import logging
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets as lerobot_merge
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Quantile levels recomputed over pooled data. Must cover every quantile key
# lerobot emits into stats.json so we fully overwrite the (wrongly) averaged
# values.
_POOLED_QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def merge_episode_datasets(
    dataset_dirs: list[Path],
    output_dir: Path,
) -> LeRobotDataset:
    """Merge multiple single-episode LeRobot datasets into one.

    Args:
        dataset_dirs: Paths to individual episode dataset directories.
        output_dir: Where to write the merged dataset.
    """
    if not dataset_dirs:
        raise ValueError("No dataset directories provided")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    datasets = []
    for d in dataset_dirs:
        logger.debug("Loading %s", d.name)
        datasets.append(LeRobotDataset(repo_id=d.name, root=d))

    logger.debug("Merging %d datasets into %s", len(datasets), output_dir)
    merged_dataset = lerobot_merge(
        datasets=datasets,
        output_repo_id=output_dir.name,
        output_dir=output_dir,
    )

    logger.info(
        "Merged: %d episodes, %d total frames",
        merged_dataset.meta.total_episodes,
        merged_dataset.meta.total_frames,
    )

    recompute_pooled_quantiles(output_dir)
    return merged_dataset


def recompute_pooled_quantiles(dataset_dir: Path) -> None:
    """Overwrite the vector features' quantiles in stats.json with exact ones over the pooled frames.

    lerobot's merge averages per-episode quantiles, which collapses q01..q99 for a feature
    that is constant within each episode but varies across them, and the quantile norm then
    blows up. Image/video and scalar columns are left as lerobot wrote them.
    """
    meta_dir = dataset_dir / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    stats_path = meta_dir / "stats.json"
    stats = json.loads(stats_path.read_text())

    parquet_paths = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")

    schema = pq.read_schema(parquet_paths[0])
    dense_features = [
        name
        for name, feature in info["features"].items()
        if feature["dtype"] not in ("image", "video")
        and name in stats
        and name in schema.names
        and is_list_column(schema.field(name).type)
    ]
    if not dense_features:
        logger.debug("No dense features to recompute quantiles for in %s", dataset_dir)
        return

    logger.debug(
        "Recomputing exact pooled quantiles over %d parquet files for features: %s",
        len(parquet_paths),
        dense_features,
    )

    pooled: dict[str, list[np.ndarray]] = {name: [] for name in dense_features}
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=dense_features)
        for name in dense_features:
            pooled[name].append(parquet_col_to_np(table.column(name)))

    total_bytes = sum(arr.nbytes for chunks in pooled.values() for arr in chunks)
    logger.debug("Pooled %.1f MB of feature data for quantile recomputation", total_bytes / 1e6)

    for name, chunks in pooled.items():
        all_values = np.concatenate(chunks, axis=0)
        for q_key, q_level in _POOLED_QUANTILES.items():
            stats[name][q_key] = np.quantile(all_values, q_level, axis=0).tolist()

    stats_path.write_text(json.dumps(stats, indent=4))
    logger.debug("Wrote pooled quantiles to %s", stats_path)


def parquet_col_to_np(col: pa.ChunkedArray) -> np.ndarray:
    """Extract an (N, D) float32 array from a parquet list-typed column."""
    combined = col.combine_chunks()
    flat = combined.values.to_numpy(zero_copy_only=False)
    is_fixed = pa.types.is_fixed_size_list(combined.type)
    width = combined.type.list_size if is_fixed else len(flat) // len(combined)
    return flat.reshape(-1, width).astype(np.float32, copy=False)


def is_list_column(col_type: pa.DataType) -> bool:
    """Check whether a pyarrow column type is list-like (list, fixed-size list, or large list)."""
    return pa.types.is_list(col_type) or pa.types.is_fixed_size_list(col_type) or pa.types.is_large_list(col_type)
