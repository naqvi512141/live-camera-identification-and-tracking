"""
frame_processor.py
-------------------
YOLO detects -> ByteTrack assigns short-term track_id (per camera) ->
whenever a track_id is NEW this session, run ReID and consult the
shared Gallery to recover a persistent global_id.
"""

import cv2
from ultralytics import YOLO

from reid_extractor import get_embedding
from gallery import Gallery

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5

_models = {}
_gallery = Gallery()                 # ONE shared gallery across all cameras
_known_track_ids = {}                # cam_name -> set of track_ids we've already ReID'd
_track_to_global = {}                # cam_name -> {track_id: global_id}

def _get_model(cam_name):
    if cam_name not in _models:
        _models[cam_name] = YOLO("yolov8n.pt")
    return _models[cam_name]

def process_frame(cam_name, frame, frame_id):
    model = _get_model(cam_name)
    _known_track_ids.setdefault(cam_name, set())
    _track_to_global.setdefault(cam_name, {})

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )[0]

    person_count = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None

        if track_id is None:
            continue  # tracker hasn't confirmed this detection into a track yet

        # --- Only run ReID when this track_id is new to us ---
        if track_id not in _known_track_ids[cam_name]:
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            embedding = get_embedding(crop)

            if embedding is not None:
                global_id = _gallery.match_or_register(embedding)
                _track_to_global[cam_name][track_id] = global_id
                _known_track_ids[cam_name].add(track_id)
            else:
                global_id = track_id  # fallback if crop was empty/invalid
        else:
            global_id = _track_to_global[cam_name].get(track_id, track_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # --- CHANGED: Larger text settings for ID and confidence ---
        text = f"ID {global_id} | {confidence:.2f}"
        font_scale = 1.2       # Increased from 0.6 -> Makes text significantly larger
        thickness = 3          # Increased from 2   -> Makes text bolder and clearer
        
        # Adjust position slightly higher so large text doesn't hit the box line
        text_y = max(y1 - 15, 30)

        cv2.putText(
            frame,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
        )
        person_count += 1

    # --- CHANGED: Larger text settings for top overlay ---
    cv2.putText(
        frame,
        f"{cam_name} | people: {person_count}",
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,                   # Increased scale from 0.7
        (0, 200, 255),
        3,                     # Increased thickness from 2
    )

    return frame