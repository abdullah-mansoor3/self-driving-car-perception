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
from pathlib import Path

from src.pipeline import (
    YOLOPv2, DepthEstimator, preprocess_yolo, preprocess_depth,
    Fuser, Navigator, TTSEngine, draw,
)
from src.pipeline.navigation import Severity

# ── Model paths ────────────────────────────────────────────────────────────────
YOLO_ONNX  = "models/yolopv2_int8.onnx"
DEPTH_ONNX = "models/depth_anything_v2_small_int8.onnx"


def _fallback_model_path(preferred: str) -> str | None:
    if preferred.endswith("_int8.onnx"):
        fp32 = preferred.replace("_int8.onnx", ".onnx")
        return fp32 if Path(fp32).exists() else None
    return None


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
    p.add_argument("--profile",  action="store_true",
                   help="Print periodic per-stage timing stats")
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
    profile=False,
):
    fps_buf = []
    frame_idx = 0
    last_tts_frame = -1_000_000
    t_acc = {"prep": 0.0, "yolo": 0.0, "depthprep": 0.0, "fuse": 0.0, "draw": 0.0, "total": 0.0}
    t_count = 0

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

        t1 = time.perf_counter()
        yolo_tensor, orig_shape = preprocess_yolo(frame)
        t_acc["prep"] += time.perf_counter() - t1

        depth_tensor = None
        if depth.should_infer_next():
            td = time.perf_counter()
            depth_tensor = preprocess_depth(frame, depth_size=depth_size)
            t_acc["depthprep"] += time.perf_counter() - td

        t2 = time.perf_counter()
        boxes, da_mask, lane_mask = detector.infer(yolo_tensor, orig_shape)
        t_acc["yolo"] += time.perf_counter() - t2

        if depth_tensor is not None:
            depth.update(depth_tensor)
        else:
            depth.mark_frame_processed()
        depth_map = depth.get_depth_map()

        t3 = time.perf_counter()
        state = fuser.fuse(boxes, da_mask, lane_mask, depth.get_depth_at_box)
        event = nav.process(state)
        t_acc["fuse"] += time.perf_counter() - t3

        if tts and event:
            priority = (event.severity == Severity.CRITICAL)
            min_gap_frames = int(tts_min_gap_s * max(fps, 1.0))
            if priority or (frame_idx - last_tts_frame) >= min_gap_frames:
                tts.speak(event.instruction, priority=priority)
                last_tts_frame = frame_idx

        t4 = time.perf_counter()
        vis = draw(frame, da_mask, lane_mask, depth_map, state, event)
        t_acc["draw"] += time.perf_counter() - t4

        elapsed = time.perf_counter() - t0
        t_acc["total"] += elapsed
        t_count += 1
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

        if profile and t_count % 120 == 0:
            denom = float(t_count)
            print(
                f"[profile] avg ms: total={1000*t_acc['total']/denom:.1f} "
                f"prep={1000*t_acc['prep']/denom:.1f} "
                f"yolo={1000*t_acc['yolo']/denom:.1f} "
                f"depthprep={1000*t_acc['depthprep']/denom:.1f} "
                f"fuse={1000*t_acc['fuse']/denom:.1f} "
                f"draw={1000*t_acc['draw']/denom:.1f}"
            )

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
    try:
        detector = YOLOPv2(YOLO_ONNX, conf_thresh=args.conf)
    except Exception as e:
        fallback = _fallback_model_path(YOLO_ONNX)
        if not fallback:
            raise
        print(f"[WARN] Failed to load {YOLO_ONNX}: {e}")
        print(f"[WARN] Falling back to {fallback}")
        detector = YOLOPv2(fallback, conf_thresh=args.conf)

    try:
        depth = DepthEstimator(DEPTH_ONNX, skip_frames=args.depth_skip)
    except Exception as e:
        fallback = _fallback_model_path(DEPTH_ONNX)
        if not fallback:
            raise
        print(f"[WARN] Failed to load {DEPTH_ONNX}: {e}")
        print(f"[WARN] Falling back to {fallback}")
        depth = DepthEstimator(fallback, skip_frames=args.depth_skip)
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
            profile=args.profile,
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
            profile=args.profile,
        )
        cap.release()
        print("\nDemo ended.")


if __name__ == "__main__":
    main()
