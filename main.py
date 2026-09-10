import argparse
import time

import cv2

import overlay
import trackers

WIN = "track-anything-live"
KEYS = {ord("1"): "sam2", ord("2"): "mosse"}


def pick_box(frame):
    box = cv2.selectROI(WIN, frame, showCrosshair=False)
    return tuple(int(v) for v in box) if box[2] > 0 and box[3] > 0 else None


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

    name = args.tracker
    tk = trackers.create(name)
    tk.init(frame, box)
    last_box = box
    trail = overlay.Trail()

    writer = None
    if args.record:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        out_fps = src_fps if 1 < src_fps < 120 else 20.0
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 out_fps, (w, h))

    fps = fps_avg = 0.0
    prev = time.time()
    frame_no = 0
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1

            tracked, box, mask = tk.update(frame)
            if box is not None:
                last_box = box
            if tracked and box is not None:
                trail.add(box)

            now = time.time()
            fps = 1.0 / (now - prev) if now > prev else fps
            fps_avg = fps if fps_avg == 0 else 0.9 * fps_avg + 0.1 * fps
            prev = now

            if mask is not None:
                overlay.draw_mask(frame, mask)
            trail.draw(frame)
            if box is not None:
                overlay.draw_box(frame, box, tracked)
            overlay.draw_hud(frame, name, fps, fps_avg, tracked, frame_no)

            if writer is not None:
                writer.write(frame)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            new_box = pick_box(frame)
            if new_box is not None:
                last_box = new_box
                tk = trackers.create(name)
                tk.init(frame, new_box)
                trail.clear()
        elif key in KEYS and KEYS[key] != name:
            name = KEYS[key]
            tk = trackers.create(name)
            tk.init(frame, last_box)
            trail.clear()

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
