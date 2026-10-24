"""Convert a directory of raw yubi MCAPs into one quality-filtered LeRobot v3 dataset.

    uv run run_pipeline --src /data/raw/<task> --out /data/datasets/<task> --task "<task>"

1. convert every `<src>/*/*.mcap` (with its `meta.json`) into a single-episode dataset
   under `<out parent>/.<out name>.episodes/`; a re-run reuses finished episodes,
2. drop episodes that fail the UMI freshness rules,
3. merge the rest into `<out>`, with exact pooled quantiles in `meta/stats.json`.

A bag that cannot be converted is logged and left out, and the run exits 1 after writing the
dataset. Each camera gets its own SVT-AV1 encoder, so keep::

    jobs x cameras x encoder_threads <= nproc
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool
import logging
import os
from pathlib import Path
import shutil
import sys
import warnings

import datasets
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from soft_yubi import schema
from soft_yubi.convert import convert_bag
from soft_yubi.filter import UMI_RULES
from soft_yubi.filter import filter_episodes
from soft_yubi.merge import merge_episode_datasets

logger = logging.getLogger("soft_yubi")

DEFAULT_FPS = 30
DEFAULT_VCODEC = "libsvtav1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="directory holding one subdirectory per recording")
    parser.add_argument("--out", type=Path, required=True, help="the merged dataset to create; must not exist")
    parser.add_argument("--task", required=True, help="task prompt written into every frame")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--vcodec", default=DEFAULT_VCODEC)
    parser.add_argument("--encoder-threads", type=int, default=2, help="SVT-AV1 lp= per camera")
    parser.add_argument("--jobs", type=int, default=None, help="bags at once (default: CPU budget)")
    parser.add_argument("--verbose", action="store_true", help="also log per-recording progress")
    return parser.parse_args(argv)


def plan_concurrency(encoder_threads: int, n_cameras: int) -> int:
    """How many bags to convert at once without oversubscribing the CPU."""
    return max(1, (os.cpu_count() or 1) // max(1, encoder_threads * n_cameras))


def convert_all(bags: list[Path], work: Path, task: str, args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    """Convert every bag in parallel.

    A worker dying (BrokenProcessPool, e.g. OOM) propagates and aborts the run; the finished
    episodes stay in `work` for the next run. Any other failure is the bag's own and skips it.

    Returns:
        The episode directories in bag order, and one `"<bag>: <error>"` line per skipped bag.
    """
    jobs = args.jobs or plan_concurrency(args.encoder_threads, len(schema.CAMERA_TOPICS))
    logger.info("Converting %d bag(s), %d at a time (lp=%d)", len(bags), jobs, args.encoder_threads)
    episodes: dict[Path, Path] = {}
    failures = []
    # The redirect keeps log lines from breaking up the progress bar.
    with ProcessPoolExecutor(max_workers=jobs) as pool, logging_redirect_tqdm():
        futures = {
            pool.submit(
                convert_bag,
                bag,
                work / bag.stem,
                task=task,
                fps=args.fps,
                vcodec=args.vcodec,
                encoder_threads=args.encoder_threads,
            ): bag
            for bag in bags
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="converting", unit="bag"):
            bag = futures[future]
            exc = future.exception()
            if exc is None:
                episodes[bag] = future.result()
                continue
            if isinstance(exc, BrokenProcessPool):
                raise exc
            # Our own errors already say what is wrong; anything else keeps its type for context.
            reason = str(exc).removeprefix(f"{bag}: ")
            failures.append(f"{bag}: {reason if isinstance(exc, ValueError) else f'{type(exc).__name__}: {reason}'}")
            logger.warning("Skipping %s", failures[-1])
    return [episodes[bag] for bag in bags if bag in episodes], failures


def log_failures(failures: list[str]) -> None:
    logger.error("%d bag(s) left out:\n%s", len(failures), "\n".join(failures))


def setup_logging(verbose: bool) -> None:
    """Our own INFO lines only: lerobot and Hugging Face datasets log and draw progress bars for
    every episode they write or load, and lerobot's stats emit a harmless numpy warning for the
    bool `is_fresh` columns. Forked workers inherit all of this."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    datasets.disable_progress_bars()
    warnings.filterwarnings("ignore", message="Converting input from bool", category=RuntimeWarning)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    out: Path = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    # Silences SVT-AV1's per-encoder banner; inherited by the workers.
    os.environ.setdefault("SVT_LOG", "1")

    mcaps = sorted(args.src.glob("*/*.mcap"))
    bags = [b for b in mcaps if (b.parent / "meta.json").exists()]
    failures = [
        f"{b}: no meta.json beside the bag; yubi-core writes it when a recording is saved"
        for b in mcaps
        if b not in bags
    ]
    if not bags:
        raise FileNotFoundError(
            f"no <recording>/*.mcap with a meta.json beside it under {args.src}; point --src at the "
            "folder holding one subdirectory per yubi-core recording"
        )
    task = args.task
    logger.info("Task %r: %d bag(s) under %s", task, len(bags), args.src)

    work = out.parent / f".{out.name}.episodes"
    partial = out.parent / f".{out.name}.partial"
    work.mkdir(parents=True, exist_ok=True)
    episodes, convert_failures = convert_all(bags, work, task, args)
    failures += convert_failures
    if not episodes:
        log_failures(failures)
        raise ValueError(f"no bag under {args.src} converted; see the errors above")

    passing = filter_episodes(episodes, UMI_RULES)
    shutil.rmtree(partial, ignore_errors=True)
    merge_episode_datasets(passing, partial)
    partial.rename(out)
    shutil.rmtree(work)
    logger.info("Wrote %s", out)

    if failures:
        log_failures(failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
