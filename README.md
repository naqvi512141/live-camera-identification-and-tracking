# Multi-Camera Live Person Detection, Tracking & Re-Identification Pipeline

A real-time pipeline that connects to multiple RTSP camera streams, detects people
in each frame, tracks them frame-to-frame, and recovers persistent identity across
occlusions/pose changes using Re-Identification (ReID). Built incrementally:
capture → multithreading → detection (YOLO) → tracking (ByteTrack) → identity
persistence (ReID) → combined display.

---

## 1. What this system does

- Connects to **2 (or more) RTSP camera streams simultaneously**, each read in
  its own background thread so a slow/dead camera never blocks the others.
- Runs a **fixed-rate 25 fps processing loop**, decoupled from each camera's
  native frame rate.
- Runs **YOLOv8n** on every frame to detect people (bounding box + confidence).
- Runs **ByteTrack** to assign a short-term track ID to each detected person,
  keeping the same ID as they move within a camera's view frame-to-frame.
- Runs **OSNet (via torchreid)** to compute an appearance embedding for any
  *newly appearing* track, and matches it against a shared **gallery** of
  previously-seen people — recovering the same global ID even after a person
  was briefly undetected (e.g. bent down, walked behind an obstacle, left and
  re-entered frame).
- **Combines both camera views into a single window** (side-by-side) for
  monitoring.

### What it can do right now
Detect people live on 2 camera feeds, draw a labeled box with a confidence
score and a **persistent ID** around each, and keep that ID stable across
short disappearances — within reliable limits (see [Section 7](#7-known-limitations)).

### What it does NOT do (yet)
- Cross-camera identity matching is architecturally possible (one shared
  gallery) but not yet tested/tuned — currently focused on within-camera
  persistence.
- No database/storage of sightings — everything is in-memory and resets on
  restart.
- No crowd-scale handling — see limitations.

---

## 2. Architecture

```
┌─────────────┐   ┌─────────────┐
│  Camera 1    │   │  Camera 2    │        config.py
│  (RTSP)      │   │  (RTSP)      │        - camera URLs
└──────┬──────┘   └──────┬──────┘        - target fps, display settings
       │ read()          │ read()
       ▼                 ▼
┌─────────────────────────────────┐
│  camera_stream.py                │
│  CameraStream (1 thread/camera)  │  ← always exposes latest frame,
│  auto-reconnect on drop          │    never blocks on network I/O
└──────────────┬───────────────────┘
               │ .read() → latest frame
               ▼
┌──────────────────────────────────────────┐
│  main.py                                   │
│  fixed-rate loop @ 25fps                   │
│  pulls latest frame per camera each tick   │
└──────────────┬─────────────────────────────┘
               │ frame
               ▼
┌──────────────────────────────────────────────────┐
│  frame_processor.py                                │
│  1. YOLOv8n  → person detections (box, confidence) │
│  2. ByteTrack → per-camera short-term track_id     │
│  3. IF track_id is new → crop → OSNet embedding    │
│     → gallery.match_or_register() → global_id      │
│  4. draw box + global_id on frame                  │
└──────────────┬─────────────────────────────────────┘
               │ annotated frame (per camera)
               ▼
┌──────────────────────────────────────────┐
│  main.py                                   │
│  resize both frames to same size           │
│  cv2.hconcat() → single combined frame     │
│  cv2.imshow("Multi-Camera View", ...)      │
└────────────────────────────────────────────┘

               shared across all cameras:
               ┌─────────────────────────┐
               │  gallery.py               │
               │  Gallery: global_id ↔      │
               │  embedding, cosine match    │
               └─────────────────────────┘
```

**Design principle:** each file has exactly one job. Adding YOLO, ByteTrack,
and ReID never required changing `camera_stream.py` or the capture threading
logic — everything new only touched `frame_processor.py` plus two small new
files (`reid_extractor.py`, `gallery.py`). Keep this separation as the system
grows.

---

## 3. File structure

```
ive_camera_capture/
├── config.py             # camera URLs, target fps, display settings — edit this to add/remove cameras
├── camera_stream.py       # CameraStream: threaded RTSP reader, one thread per camera, auto-reconnect
├── frame_processor.py     # YOLO detection + ByteTrack + ReID gallery lookup, draws boxes/IDs
├── reid_extractor.py      # loads OSNet once, exposes get_embedding(crop)
├── gallery.py             # Gallery class: cosine-similarity identity matching + pruning
├── main.py                # entry point: starts camera threads, runs 25fps loop, combines & displays frames
├── utils/
│   ├── __init__.py
│   └── fps_counter.py     # measures ACTUAL achieved fps per camera
├── requirements.txt
└── README.md               # this file
```

---

## 4. Setup

### 4.1 Virtual environment (do this first, before installing anything)

```bash
cd ive_camera_capture
python3 -m venv venv
source venv/bin/activate        # every new terminal session needs this again
```

In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `./venv/bin/python`.

### 4.2 Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` should contain:
```
opencv-python>=4.8.0
ultralytics>=8.0.0
lap>=0.4.0
torchreid>=0.2.5
scipy>=1.10.0
```

> `torchreid` has some undeclared dependencies depending on version/platform.
> If you hit `ModuleNotFoundError` for `yacs`, `gdown`, `tensorboard`, or
> `h5py` on first run, just `pip install` whichever one is named and re-run —
> this is a known packaging gap in the library, not a bug in this project.

### 4.3 Verify install

```bash
python -c "import cv2; print(cv2.__version__)"
python -c "from torchreid.reid.utils import FeatureExtractor; print('ok')"
```

Both should print without errors.

### 4.4 Configure cameras

Edit `config.py`:

```python
CAMERAS = {
    "cam1": "rtsp://<user>:<pass>@<ip1>:554/Streaming/Channels/101",
    "cam2": "rtsp://<user>:<pass>@<ip2>:554/Streaming/Channels/101",
}
```

> **Security note:** credentials are currently plain text here. Fine while
> prototyping — before this goes into production, move them into environment
> variables or a `.env` file instead.

### 4.5 Run

```bash
python main.py
```

A single combined window opens showing both cameras side-by-side, with boxes
and persistent IDs drawn on any detected person. Press `q` (window focused)
or `Ctrl+C` to stop.

---

## 5. Tuning parameters (where they live, what they do)

| Parameter | File | Effect |
|---|---|---|
| `TARGET_FPS` | `config.py` | Processing loop rate. Higher = more responsive, more CPU load. |
| `CONFIDENCE_THRESHOLD` | `frame_processor.py` | Minimum YOLO confidence to count as a detection. Lower = catches more (e.g. partially occluded people) but more false positives. |
| `SIMILARITY_THRESHOLD` | `gallery.py` | Cosine similarity cutoff for "same person" in ReID. Higher = stricter (more duplicate IDs, fewer wrong merges). Lower = looser (fewer duplicate IDs, more risk of merging two different people). **Tune this empirically for your environment/lighting/clothing — 0.65 is a starting point, not a guarantee.** |
| `GALLERY_TIMEOUT` | `gallery.py` | Seconds before an unseen gallery entry is dropped. Prevents unbounded growth and very-old false matches. |
| `DISPLAY_WIDTH` / `DISPLAY_HEIGHT` | `config.py` | Resize target before combining feeds — must match for `cv2.hconcat` to work. |

---

## 6. Troubleshooting quick-reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Package not actually installed into the active venv | `which python` to confirm venv is active, then `python -m pip install -r requirements.txt` |
| `TypeError: ...got an unexpected keyword argument` | Local file doesn't match expected class/function signature (often from re-typing code) | Compare the file's method signature against this README/source carefully — usually a typo or dropped parameter |
| A method silently "does nothing" / thread never starts | Indentation bug — a method accidentally nested inside another method instead of being a sibling class method | Check indentation level matches sibling methods (`_connect`, `read`, `stop`, etc.) exactly |
| `ModuleNotFoundError: No module named 'scipy'` (or `yacs`, `gdown`, etc.) | `torchreid` has undeclared dependencies | `pip install <missing package>`, repeat as needed, then add to `requirements.txt` |
| `ModuleNotFoundError: No module named 'torchreid.utils'` | This torchreid version nests utils one level deeper | Use `from torchreid.reid.utils import FeatureExtractor` |
| `No route to host` on RTSP connect | **Network problem, not code.** Your machine has no network path to the camera's IP right now | `ping <camera_ip>` to confirm. Check you're on the same LAN/VPN as the cameras — not a Python/OpenCV issue |
| fps well below `TARGET_FPS` once YOLO/ReID are added | Expected — inference cost on CPU adds up across 2 cameras × 3 models (YOLO, ByteTrack, OSNet) | Confirmed via `FPSCounter` output; see Section 7 for what's tunable |
| Same person keeps getting a new ID after bending/occlusion | This is exactly what ByteTrack alone cannot solve (motion/IoU only, no appearance memory) — is the reason ReID was added | Confirm ReID gallery is wired in (`frame_processor.py` should import `gallery.py`); tune `SIMILARITY_THRESHOLD` if it's still not recovering IDs |

---

## 7. Known limitations

- **Detection & tracking accuracy drops with crowding/overlap.** Heavily
  overlapping people can cause missed detections (occlusion) or ID switches
  (ByteTrack's IoU-based matching gets ambiguous when boxes overlap heavily).
- **ReID gallery lookup is O(n)** — compares a new embedding against every
  existing gallery entry. Fine for small numbers of people; will add
  measurable per-frame cost as the gallery grows.
- **ReID false-match risk grows with gallery size.** A fixed similarity
  threshold becomes statistically more likely to wrongly match two different
  people as the number of stored identities increases — inherent to this
  lightweight single-threshold design, not a bug.
- **Reliable range on current hardware/models (estimate, not measured):**
  roughly 2–8 well-spaced people per camera behaves reliably; 10–15 is usable
  but expect occasional ID switches/false ReID matches; dense crowds (20+,
  tightly packed) are outside what this lightweight stack (`yolov8n` +
  `bytetrack` + `osnet_x0_25`, chosen for real-time CPU speed) is suited for.
  **Worth validating with a real multi-person test once cameras are reachable.**
- **No persistence across restarts** — the gallery is in-memory only.
- **Credentials are plain text in `config.py`** — acceptable for prototyping,
  should move to environment variables before any production use.

---

## 8. Roadmap / natural next steps

1. Validate ID persistence and multi-person behavior with real people once
   camera network connectivity is confirmed.
2. Tune `SIMILARITY_THRESHOLD` and `CONFIDENCE_THRESHOLD` empirically against
   observed false-match / missed-detection rates.
3. If cross-camera identity matching becomes the goal, no architecture change
   is needed (gallery is already shared) — just testing/tuning against real
   footage of the same person crossing between cam1 and cam2.
4. Consider moving credentials to `.env` before any deployment beyond
   local testing.
