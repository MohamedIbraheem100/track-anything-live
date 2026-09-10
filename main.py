import argparse
import time

import cv2

import trackers

WIN = "tracker"


def draw(frame, ok, box, mask, fps, name):
    if mask is not None:
        green = frame.copy()
        green[mask] = (0, 200, 0)
        cv2.addWeighted(green, 0.4, frame, 0.6, 0, frame)
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 200, 0) if ok else (0, 0, 255), 2)
    label = f"{name} | {fps:.0f} fps | {'tracking' if ok else 'lost'}"
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def pick_box(frame):
    box = cv2.selectROI(WIN, frame, showCrosshair=False)
    return box if box[2] > 0 and box[3] > 0 else None


def main():
    ap = argparse.ArgumentParser(description="Track an object picked in the first frame.")
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--tracker", default="sam2", choices=["sam2", "mosse"])
    ap.add_argument("--record", help="save the annotated video to this path")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"could not read from {args.source}")

    box = pick_box(frame)
    if box is None:
        return
    tk = trackers.create(args.tracker)
    tk.init(frame, box)

    writer = None
    if args.record:
        fps_out = cap.get(cv2.CAP_PROP_FPS)
        fps_out = fps_out if 1 < fps_out < 120 else 20.0
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_out, (w, h))

    fps, last = 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracked, box, mask = tk.update(frame)

        now = time.time()
        dt = now - last
        last = now
        if dt > 0:
            inst = 1 / dt
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

        draw(frame, tracked, box, mask, fps, args.tracker)
        if writer is not None:
            writer.write(frame)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            new_box = pick_box(frame)
            if new_box is not None:
                tk = trackers.create(args.tracker)
                tk.init(frame, new_box)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
