"""
demo.py
────────
Main entry point for the live self-driving perception demo.

Usage
-----
  # Webcam
  python demo.py --source 0

  # Video file
  python demo.py --source data/samples/highway.mp4

  # Save output
  python demo.py --source 0 --save output.avi

  # Disable TTS (silent mode, e.g. in a quiet lab)
  python demo.py --source 0 --no-tts
"""

import argparse
import time
import cv2

from src.pipeline import (
    YOLOPv2, DepthEstimator, preprocess,
    Fuser, Navigator, TTSEngine, draw,
)
from src.pipeline.navigation import Severity

# ── Model paths ────────────────────────────────────────────────────────────────
YOLO_ONNX  = "models/yolopv2_int8.onnx"
DEPTH_ONNX = "models/depth_anything_v2_small_int8.onnx"


def parse_args():
    p = argparse.ArgumentParser(description="Self-driving perception demo")
    p.add_argument("--source",   default="0",  help="0=webcam, or path to video file")
    p.add_argument("--save",     default=None, help="Path to save output video (optional)")
    p.add_argument("--no-tts",   action="store_true", help="Disable voice output")
    p.add_argument("--conf",     type=float, default=0.45, help="Detection confidence threshold")
    p.add_argument("--width",    type=int, default=1280, help="Display window width")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load models ──────────────────────────────────────────────────────────
    print("\nLoading models…")
    detector = YOLOPv2(YOLO_ONNX, conf_thresh=args.conf)
    depth    = DepthEstimator(DEPTH_ONNX, skip_frames=3)
    fuser    = Fuser()
    nav      = Navigator(cooldown_s=2.5)
    tts      = TTSEngine() if not args.no_tts else None

    print("All models loaded. Starting pipeline…\n")

    if tts:
        tts.speak("Self-driving perception active. Let's see how badly you drive.")

    # ── Open video source ────────────────────────────────────────────────────
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    # ── Output writer (optional) ─────────────────────────────────────────────
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.save, fourcc, 20,
                                 (args.width, int(args.width * 9 / 16)))

    # ── FPS counter ──────────────────────────────────────────────────────────
    fps_buf   = []
    frame_idx = 0

    cv2.namedWindow("Self-Driving Perception", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Self-Driving Perception", args.width, int(args.width * 9 / 16))

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────
    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 1. Preprocess
        yolo_tensor, depth_tensor, orig_shape = preprocess(frame)

        # 2. YOLOPv2 — detection + lane + drivable (every frame)
        boxes, da_mask, lane_mask = detector.infer(yolo_tensor, orig_shape)

        # 3. Depth — fires background thread every 3rd frame, returns cached map
        depth.update(depth_tensor)
        depth_map = depth.get_depth_map()

        # 4. Fuse into scene state
        state = fuser.fuse(boxes, da_mask, lane_mask, depth.get_depth_at_box)

        # 5. Navigation decision
        event = nav.process(state)

        # 6. Speak (non-blocking)
        if tts and event:
            priority = (event.severity == Severity.CRITICAL)
            tts.speak(event.instruction, priority=priority)

        # 7. Draw overlay
        vis = draw(frame, da_mask, lane_mask, depth_map, state, event)

        # 8. FPS overlay
        elapsed = time.perf_counter() - t0
        fps_buf.append(1.0 / max(elapsed, 1e-6))
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)
        cv2.putText(vis, f"FPS {fps:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 9. Display
        cv2.imshow("Self-Driving Perception", vis)

        if writer:
            resized = cv2.resize(vis, (args.width, int(args.width * 9 / 16)))
            writer.write(resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


if __name__ == "__main__":
    main()
