import numpy as np
import pytest

import trackers


def synthetic_clip(n=30):
    """A textured patch sliding across a dark background."""
    rng = np.random.default_rng(0)
    s = 70
    patch = rng.integers(40, 255, (s, s, 3), dtype=np.uint8)
    frames, boxes = [], []
    for i in range(n):
        x, y = 30 + i * 3, 80
        img = np.full((240, 360, 3), 15, np.uint8)
        img[y:y + s, x:x + s] = patch
        frames.append(img)
        boxes.append((x, y, s, s))
    return frames, boxes


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    iw = max(0, min(ax + aw, bx + bw) - ix)
    ih = max(0, min(ay + ah, by + bh) - iy)
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def test_mosse_stays_on_target():
    frames, boxes = synthetic_clip()
    tk = trackers.create("mosse")
    tk.init(frames[0], boxes[0])
    scores = []
    for f, gt in zip(frames[1:], boxes[1:]):
        ok, box, _ = tk.update(f)
        scores.append(iou(box, gt) if ok and box else 0.0)
    assert np.mean(scores) > 0.6


def test_mask_box():
    m = np.zeros((100, 100), bool)
    m[20:41, 30:71] = True
    assert trackers.mask_box(m) == (30, 20, 40, 20)


def test_mask_box_empty():
    assert trackers.mask_box(np.zeros((10, 10), bool)) is None


def test_bad_name():
    with pytest.raises(ValueError):
        trackers.create("orb")
