import cv2

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


class Trail:
    """Keeps recent object centres so the path can be drawn."""

    def __init__(self, length=50):
        self.length = length
        self.points = []

    def add(self, box):
        x, y, w, h = box
        self.points.append((x + w // 2, y + h // 2))
        self.points = self.points[-self.length:]

    def clear(self):
        self.points = []

    def draw(self, frame):
        for i in range(1, len(self.points)):
            cv2.line(frame, self.points[i - 1], self.points[i], GREEN, 2)


def draw_mask(frame, mask):
    tinted = frame.copy()
    tinted[mask] = GREEN
    cv2.addWeighted(tinted, 0.4, frame, 0.6, 0, frame)


def draw_box(frame, box, ok):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN if ok else RED, 2)


def draw_hud(frame, tracker, fps, fps_avg, ok, frame_no):
    lines = [
        tracker.upper(),
        f"fps {fps:4.0f}  (avg {fps_avg:.0f})",
        "TRACKING" if ok else "LOST",
        f"frame {frame_no}",
    ]
    cv2.rectangle(frame, (0, 0), (185, 22 * len(lines) + 8), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        colour = RED if text == "LOST" else WHITE
        cv2.putText(frame, text, (8, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    hint = "1 SAM2   2 MOSSE   r re-select   space pause   q quit"
    cv2.putText(frame, hint, (8, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
