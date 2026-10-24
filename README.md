<h1 align="center">Robot-Matched Soft Fingers for UMI Data Collection:<br>Lessons from 100 Hours of Pick and Place</h1>

<!-- TODO: authors and affiliations -->

<p align="center">
  <a href="https://lute-corp.github.io/soft-yubi-site/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project page"></a>
  <!-- TODO: arXiv link -->
  <a href="#"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/LuteCorp/soft-yubi-100h"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Data-soft--yubi--100h-yellow" alt="soft-yubi-100h dataset"></a>
</p>

Turn recordings from the [YUBI](https://github.com/airoa-org/yubi-sw) handheld grippers into a
quality-filtered [LeRobot](https://github.com/huggingface/lerobot) v3.0 dataset, in the same
end-effector action space as a robot carrying the YUBI gripper.

This is the pipeline that produced the
[`soft-yubi-100h`](https://huggingface.co/datasets/LuteCorp/soft-yubi-100h) dataset: running it on
the same raw recordings reproduces those episodes, except gripper values outside [0, 1]. That dataset was
recorded with a yubi-sw version that publishes the gripper as a calibrated 0–1 opening, and its
gripper values were clipped to [0, 1].

```
uv run run_pipeline --src <recordings> --out <dataset> --task "<prompt>"
```

## Requirements

**Hardware**: a yubi-sw rig, either variant.

- two YUBI grippers with the standard fingers and pads. The tool-tip offset below is fixed to this
  geometry; a modified gripper needs a new `TIP_XYZ` / `TIP_ROT` in `build_ee.py`;
- a Meta Quest, with a controller mounted on each gripper as in yubi-sw's `yubi_description`;
- a USB wrist camera on each gripper, and a top camera: a USB center camera (portable variant) or a
  RealSense (stationary variant).

**Gripper tips**: [`hardware/`](hardware/) holds the CAD (STEP) for the wide gripper tips used to
record `soft-yubi-100h`: `tip_wide_l.step` (left) and `tip_wide_r.step` (right).

**Software**: Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). The AV1 encoder ships with
the PyAV wheel; no system ffmpeg is needed.

## Recording

Record with [yubi-sw](https://github.com/airoa-org/yubi-sw) and its yubi-core backend, following
their README, with these settings:

1. **Either variant.** The top camera is read from `/center_camera/image_raw/compressed`
   (portable) or `/camera/camera/color/image_raw/compressed` (stationary; `image_rect_raw` in older
   yubi-sw versions), whichever the recording has. `soft-yubi-100h` was recorded with the portable
   variant's USB center camera; stationary recordings have a different top-camera view.
2. **Calibrate both encoders** ("Capture MIN" in the device-setup window). The gripper value is the
   jaw angle measured from that closed position.
3. **One accepted episode per recording.** Each recording's `meta.json` must hold exactly one
   segment, which is the span that gets converted. Recordings with no segment, or with several (a
   rejected take and its retry, subtask labels), are skipped.
4. Keep yubi-core's default MCAP storage.

Before collecting at scale, run one test recording through the pipeline.

## Usage

```
git clone <this repo> && cd soft-yubi
uv sync
uv run run_pipeline --src /data/raw/<task> --out /data/datasets/<task> --task "<task>"
```

- `--src`: one folder per task, holding one subdirectory per yubi-core recording, each with its
  `.mcap` and `meta.json`:
  ```
  /data/raw/<task>/
    26-09-18-08-17-04-339dbade-.../
      26-09-18-08-17-04-339dbade-..._0.mcap
      meta.json
    26-09-18-08-17-12-cd8664e0-.../
    ...
  ```
- `--out`: the dataset to create; it must not exist yet.
- `--task`: the language prompt stored with every frame.
- `--jobs`, `--encoder-threads`: parallelism. Each recording runs one AV1 encoder per camera; keep
  `jobs x 3 x encoder_threads` at or below the CPU count (the default picks this for you).

The raw recordings are only read, never modified. Finished episodes are cached next to `--out`
(`.<name>.episodes/`), so an interrupted run resumes where it stopped.

A recording that cannot be converted is skipped and listed at the end with the reason and the fix,
for example one with no top camera or with two segments. The dataset is still written
from the rest, and the command exits with status 1.

## What it does

1. **Convert** each recording into one episode:
   - read the Quest hand poses (`/tf`, `quest_origin -> {left,right}_hand_root`), the jaw encoders
     (`/joint_states`) and the three cameras;
   - resample every stream onto a 30 fps grid, taking each stream's latest message at every tick;
     `is_fresh.*` marks whether that message was new or repeated from the previous tick;
   - keep the span of the recording's segment;
   - move each hand pose to the gripper's tool tip and build the 20-dim state; the action is the
     next frame's state.
2. **Filter** out episodes where a stream stalled for more than 0.098 s, or where a hand pose or
   jaw stream was fresh on 95% of frames or fewer.
3. **Merge** the remaining episodes into one dataset, with exact quantiles over all frames in
   `meta/stats.json`.

## Output

One LeRobot v3.0 dataset at 30 fps:

| Feature | Shape | Description |
|---|---|---|
| `observation.state` | (20,) | `[left_xyz(3), left_rot6d(6), left_gripper(1), right_xyz(3), right_rot6d(6), right_gripper(1)]` |
| `action` | (20,) | the next frame's `observation.state`; the last frame repeats |
| `observation.images.top` | 480x640x3 | top camera (USB center camera or RealSense), AV1 |
| `observation.images.wrist_left` | 480x640x3 | left gripper camera, AV1 |
| `observation.images.wrist_right` | 480x640x3 | right gripper camera, AV1 |
| `is_fresh.*` | (1,) | per stream: newly measured at this frame, or held from the previous one |

Conventions:

- **Position** is the gripper's tool tip, the point where the two pads meet, in metres. It is
  0.0784 m ahead of the Quest-tracked `hand_root` and does not move as the jaw opens.
- **Frame**: poses are in `quest_origin`, the Quest's tracking frame. It is set by the headset and
  can change between sessions, so use relative actions, or map poses into the robot's frame, when
  deploying on a robot.
- **rot6d** is the first two columns of the tool-tip rotation matrix, flattened row-major.
- **Gripper** is the jaw value yubi-sw publishes on `/joint_states`, unchanged: the jaw angle in
  radians from the calibrated closed position (0 = closed).

## License

[PolyForm Noncommercial License 1.0.0](LICENSE).
