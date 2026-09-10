import cv2
import numpy as np


def create(name):
    if name == "sam2":
        return Sam2Tracker()
    if name == "mosse":
        return MosseTracker()
    raise ValueError(f"unknown tracker: {name}")


class MosseTracker:
    """OpenCV MOSSE correlation filter. No model, runs on the CPU."""

    def __init__(self):
        self.t = None

    def init(self, frame, box):
        self.t = cv2.legacy.TrackerMOSSE_create()
        self.t.init(frame, tuple(int(v) for v in box))

    def update(self, frame):
        ok, box = self.t.update(frame)
        if not ok:
            return False, None, None
        x, y, w, h = [int(v) for v in box]
        return True, (x, y, w, h), None


class Sam2Tracker:
    """SAM 2 used as a tracker: re-prompt it every frame with the previous box."""

    def __init__(self, weights="sam2.1_t.pt", imgsz=384):
        import torch
        from ultralytics import SAM

        torch.backends.cudnn.benchmark = True
        self.model = SAM(weights)
        self.imgsz = imgsz
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.box = None

    def _segment(self, frame, box):
        x, y, w, h = box
        kw = dict(bboxes=[[x, y, x + w, y + h]], imgsz=self.imgsz,
                  device=self.device, verbose=False)
        if self.device != "cpu":
            kw["quantize"] = "fp16"  # half precision, ~2x faster on the GPU
        res = self.model(frame, **kw)
        masks = res[0].masks
        if masks is None or len(masks) == 0:
            return None
        m = masks.data[0].cpu().numpy().astype(bool)
        if m.shape != frame.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (frame.shape[1], frame.shape[0])) > 0
        return m

    def init(self, frame, box):
        self.box = tuple(int(v) for v in box)
        m = self._segment(frame, self.box)
        if m is not None:
            self.box = mask_box(m) or self.box

    def update(self, frame):
        m = self._segment(frame, self.box)
        if m is None:
            return False, self.box, None
        box = mask_box(m)
        if box is None:
            return False, self.box, None
        self.box = box
        return True, box, m


def mask_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())
