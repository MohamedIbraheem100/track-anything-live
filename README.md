# track-anything-live

Draw a box around something in the webcam, and the program keeps tracking it
while it moves.

![demo](demo.gif)

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The SAM 2 tracker needs PyTorch. On a machine with an NVIDIA GPU:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

CPU-only torch also works, but SAM 2 is too slow to be useful on CPU there, so
use `--tracker mosse` instead.

## Running

```
python main.py                     # webcam, SAM 2
python main.py --tracker mosse      # correlation-filter tracker, no GPU
python main.py --source clip.mp4    # run on a video file
python main.py --record out.mp4     # also write the annotated video
```

When the window opens, drag a rectangle over the object and press Enter.
Press `r` to pick a different object, `q` to quit.

## How it works

`main.py` is the capture loop: read a frame, ask the tracker for the object's
new position, draw it, show the frame. The trackers are in `trackers.py`.

**SAM 2** (`sam2`) is Meta's Segment Anything 2, the small `sam2.1_t` checkpoint,
loaded through Ultralytics. I use it as a tracker by prompting it with a box:
the first box comes from the user, and every frame after that I prompt it with
the bounding box of the previous frame's mask. It gives back a segmentation mask,
which is drawn as the green overlay. On my laptop (RTX 3050, 4 GB) it runs around
20 fps at 384 px input with fp16 once CUDA has warmed up. The checkpoint (~75 MB)
downloads on the first run.

**MOSSE** (`mosse`) is the correlation-filter tracker from OpenCV
(`cv2.legacy.TrackerMOSSE_create`). It builds a filter from the first patch and
updates it online. No model, a few hundred fps on the CPU. It drifts when the
object rotates or gets covered, but it's a useful lightweight comparison.

## Limits

- One object at a time.
- If the object leaves the frame entirely, SAM 2 can latch onto the background.
  Press `r` to re-select.
- MOSSE doesn't cope well with the object changing size.

## Tests

```
pytest
```

Runs MOSSE on a small synthetic clip and checks it keeps up with the target,
plus a couple of unit tests for the mask-to-box helper.
