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
import os
import tempfile
import time
import cv2
from tqdm import tqdm

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
    p.add_argument("--conf",     type=float, default=0.6, help="Detection confidence threshold")
    p.add_argument("--depth-skip", type=int, default=6,
                   help="Run depth inference every N frames (higher = faster)")
    p.add_argument("--depth-size", default=None,
                   help="Depth input size WxH (requires re-exported depth ONNX), e.g. 560x336")
    p.add_argument("--width",    type=int, default=1280, help="Display window width")
    return p.parse_args()


def _parse_size(size_str: str) -> tuple[int, int]:
    parts = size_str.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError("Expected format WxH, e.g. 560x336")
    w, h = int(parts[0]), int(parts[1])
    return w, h


def _open_capture(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")
    return cap


def _get_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 1e-3 else 20.0


def _record_webcam(cap) -> str:
    fps = _get_fps(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
    print("Recording webcam. Press 'q' to stop and process...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cv2.destroyWindow("Recording")
    return tmp_path


def _process_video(
    cap,
    detector,
    depth,
    fuser,
    nav,
    tts,
    writer,
    show,
    window_w,
    fps,
    progress,
    tts_min_gap_s=1.2,
    depth_size=None,
):
    fps_buf = []
    frame_idx = 0
    last_tts_frame = -1_000_000

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = None
    if progress and total > 0:
        pbar = tqdm(total=total, unit="frame", desc="Processing")

    if show:
        cv2.namedWindow("Self-Driving Perception", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Self-Driving Perception", window_w, int(window_w * 9 / 16))

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        yolo_tensor, depth_tensor, orig_shape = preprocess(frame, depth_size=depth_size)
        boxes, da_mask, lane_mask = detector.infer(yolo_tensor, orig_shape)

        depth.update(depth_tensor)
        depth_map = depth.get_depth_map()

        state = fuser.fuse(boxes, da_mask, lane_mask, depth.get_depth_at_box)
        event = nav.process(state)

        if tts and event:
            priority = (event.severity == Severity.CRITICAL)
            min_gap_frames = int(tts_min_gap_s * max(fps, 1.0))
            if priority or (frame_idx - last_tts_frame) >= min_gap_frames:
                tts.speak(event.instruction, priority=priority)
                last_tts_frame = frame_idx

        vis = draw(frame, da_mask, lane_mask, depth_map, state, event)

        elapsed = time.perf_counter() - t0
        fps_buf.append(1.0 / max(elapsed, 1e-6))
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)
        cv2.putText(vis, f"FPS {fps:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if writer:
            writer.write(vis)

        if show:
            cv2.imshow("Self-Driving Perception", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if pbar is not None:
            pbar.update(1)

    if show:
        cv2.destroyWindow("Self-Driving Perception")
    if pbar is not None:
        pbar.close()


def main():
    args = parse_args()

    depth_size = None
    if args.depth_size:
        depth_size = _parse_size(args.depth_size)
        if depth_size[0] % 14 != 0 or depth_size[1] % 14 != 0:
            raise ValueError("Depth size must be divisible by 14 (both W and H)")

    # ── Load models ──────────────────────────────────────────────────────────
    print("\nLoading models…")
    detector = YOLOPv2(YOLO_ONNX, conf_thresh=args.conf)
    depth    = DepthEstimator(DEPTH_ONNX, skip_frames=args.depth_skip)
    fuser    = Fuser()
    nav      = Navigator(cooldown_s=2.5)
    tts      = TTSEngine() if not args.no_tts else None

    print("All models loaded. Starting pipeline…\n")

    if tts:
        tts.speak("Self-driving perception active. Let's see how badly you drive.")

    # ── Open video source ────────────────────────────────────────────────────
    src = int(args.source) if args.source.isdigit() else args.source

    if args.save:
        # Offline processing only; no real-time display.
        cap = _open_capture(src)
        tmp_path = None
        if isinstance(src, int):
            tmp_path = _record_webcam(cap)
            cap.release()
            cap = _open_capture(tmp_path)

        fps = _get_fps(cap)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

        _process_video(
            cap,
            detector,
            depth,
            fuser,
            nav,
            tts,
            writer,
            False,
            args.width,
            fps,
            True,
            depth_size=depth_size,
        )

        cap.release()
        writer.release()
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        print(f"\nSaved processed video to {args.save}")
    else:
        cap = _open_capture(src)
        fps = _get_fps(cap)
        _process_video(
            cap,
            detector,
            depth,
            fuser,
            nav,
            tts,
            None,
            True,
            args.width,
            fps,
            False,
            depth_size=depth_size,
        )
        cap.release()
        print("\nDemo ended.")


if __name__ == "__main__":
    main()
