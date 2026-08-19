"""
main.py
-------
Entry point.
  1. Starts thread readers for cam1 and cam2.
  2. Runs a 25 FPS loop pulling frames from both.
  3. Runs YOLO processing on each frame.
  4. Resizes and combines (concatenates) frames into a single window.
"""

import cv2
import time
import signal
import sys

import config
from camera_stream import CameraStream
from frame_processor import process_frame
from utils.fps_counter import FPSCounter

streams = {}
running = True

def handle_shutdown(sig=None, frame_arg=None):
    global running
    running = False

def main():
    global running

    # 1. Start camera threads
    for name, url in config.CAMERAS.items():
        streams[name] = CameraStream(
            name=name, src=url, reconnect_delay=config.RECONNECT_DELAY
        ).start()

    print("Warming up camera connections...")
    time.sleep(2)

    signal.signal(signal.SIGINT, handle_shutdown)

    fps_counters = {name: FPSCounter(report_every=2.0) for name in streams}
    frame_ids = {name: 0 for name in streams}

    next_tick = time.time()

    print(f"Running main loop at target {config.TARGET_FPS} fps. "
          f"Press 'q' in the display window (or Ctrl+C) to quit.")

    try:
        while running:
            processed_frames = []

            for name, cam in streams.items():
                frame = cam.read()
                if frame is None:
                    continue

                frame_ids[name] += 1
                frame = process_frame(name, frame, frame_ids[name])

                reported = fps_counters[name].tick()
                if reported is not None:
                    print(f"[{name}] processing @ {reported:.1f} fps (target {config.TARGET_FPS})")

                # Resize each frame to uniform dimensions for concatenation
                resized_frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
                processed_frames.append(resized_frame)

            # --- Combine and Display in One Screen ---
            if config.DISPLAY and len(processed_frames) > 0:
                # If both camera streams are present, stack them side-by-side
                if len(processed_frames) == 2:
                    combined_frame = cv2.hconcat(processed_frames)
                else:
                    combined_frame = processed_frames[0]

                cv2.imshow("Multi-Camera View", combined_frame)

            if config.DISPLAY and cv2.waitKey(1) & 0xFF == ord("q"):
                running = False

            # Fixed-rate scheduling to target 25 FPS
            next_tick += config.FRAME_INTERVAL
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.time()

    finally:
        print("Shutting down...")
        for cam in streams.values():
            cam.stop()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()