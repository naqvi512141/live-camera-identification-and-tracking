"""
frame_processor.py
-------------------
Pipeline per frame:
  1. YOLOv8n (GPU) detects people
  2. ByteTrack assigns a short-term, per-camera track_id
  3. Track LIFECYCLE management: tracks not seen for a grace period
     are purged from "active" bookkeeping, freeing their global_id
     for correct re-matching on return (see _cleanup_dead_tracks)
  4. On a NEW, stable, good-quality track -> ReID match/register
     against the shared Gallery using embedding + color histogram,
     excluding IDs already active on OTHER currently-visible tracks
  5. On an ALREADY-known track -> periodically enrich its identity's
     embedding bank and color profile with fresh good-quality crops
  6. Draw a box + label in a color derived deterministically from the
     global ID (same ID = same color, always)

"Good quality" crop = not heavily overlapping another box, not
touching the frame edge (partial body), and large enough to be a
reliable ReID reference.
"""

import cv2
import colorsys
from ultralytics import YOLO

from reid_extractor import get_embedding
from color_signature import get_color_signature
from gallery import Gallery

# --- detection / tracking settings ---
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU = 0.75              # raised from default 0.7 -- helps avoid merging two close people

# --- ReID quality-gate settings ---
MIN_TRACK_AGE = 3            # frames a track must persist before we trust it for ReID
MIN_CROP_WIDTH = 40
MIN_CROP_HEIGHT = 100
EDGE_MARGIN = 5               # px -- box this close to frame border = likely partial body
OVERLAP_IOU_SKIP = 0.3        # skip ReID if box overlaps another this much (contaminated crop)
UPDATE_EVERY_N_FRAMES = 15    # ~twice a second at 25fps -- periodic bank enrichment

# --- track lifecycle ---
MISSED_FRAME_GRACE = 8        # ~0.3s at 25fps -- tolerate brief detection blips before purging

_models = {}
_gallery = Gallery()
_known_track_ids = {}         # cam_name -> set(track_id) already ReID'd
_track_to_global = {}         # cam_name -> {track_id: global_id}
_track_seen_count = {}        # cam_name -> {track_id: consecutive frames seen}
_track_missed_count = {}      # cam_name -> {track_id: consecutive frames missed}
_color_cache = {}             # identity -> (B, G, R)


def _get_model(cam_name):
    if cam_name not in _models:
        _models[cam_name] = YOLO("yolo11x.pt")
    return _models[cam_name]


def _active_global_ids(exclude_cam=None, exclude_track=None):
    """Every global_id currently owned by a live, already-known track,
    across ALL cameras -- used so ReID never assigns an ID that's
    actively in use by a different, currently-visible person."""
    active = set()
    for cam, mapping in _track_to_global.items():
        for tid, gid in mapping.items():
            if cam == exclude_cam and tid == exclude_track:
                continue
            active.add(gid)
    return active


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _touches_edge(x1, y1, x2, y2, frame_w, frame_h):
    return (x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN
            or x2 >= frame_w - EDGE_MARGIN or y2 >= frame_h - EDGE_MARGIN)


def _cleanup_dead_tracks(cam_name, current_track_ids):
    """Remove bookkeeping for tracks ByteTrack hasn't reported for
    MISSED_FRAME_GRACE frames, freeing their global_id from the
    'active' exclusion set. The gallery's stored embeddings are
    NOT touched here -- only this per-camera active bookkeeping
    resets, so the person can still be correctly re-recognized."""
    missed = _track_missed_count.setdefault(cam_name, {})
    mapping = _track_to_global.setdefault(cam_name, {})
    known = _known_track_ids.setdefault(cam_name, set())
    seen = _track_seen_count.setdefault(cam_name, {})

    for tid in list(mapping.keys()):
        if tid in current_track_ids:
            missed[tid] = 0
        else:
            missed[tid] = missed.get(tid, 0) + 1
            if missed[tid] >= MISSED_FRAME_GRACE:
                mapping.pop(tid, None)
                known.discard(tid)
                seen.pop(tid, None)
                missed.pop(tid, None)


def _color_for_id(identity):
    """Deterministic, well-separated color per identity, derived from
    the ID itself (golden-angle hue spacing) so consecutive IDs never
    look visually similar, and the same ID always gets the same color."""
    if identity in _color_cache:
        return _color_cache[identity]
    try:
        n = int(identity)
    except (TypeError, ValueError):
        return (160, 160, 160)   # neutral gray for provisional "?N" labels

    hue = (n * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    color = (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV
    _color_cache[identity] = color
    return color


def _draw_label(frame, x1, y1, x2, y2, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 1.0, 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    ly1 = max(y1 - th - baseline - 8, 0)
    ly2 = max(y1, th + baseline + 8)

    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 10, ly2), color, -1)   # filled bg for readability
    cv2.putText(frame, text, (x1 + 5, ly2 - baseline - 4),
                font, font_scale, (0, 0, 0), font_thickness)


def process_frame(cam_name, frame, frame_id):
    model = _get_model(cam_name)
    _known_track_ids.setdefault(cam_name, set())
    _track_to_global.setdefault(cam_name, {})
    _track_seen_count.setdefault(cam_name, {})

    frame_h, frame_w = frame.shape[:2]

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=CONFIDENCE_THRESHOLD,
        iou=NMS_IOU,
        classes=[PERSON_CLASS_ID],
        device=0,          # use the GPU
        verbose=False,
    )[0]

    all_boxes = [tuple(map(int, b.xyxy[0])) for b in results.boxes]
    current_track_ids = {int(b.id[0]) for b in results.boxes if b.id is not None}
    _cleanup_dead_tracks(cam_name, current_track_ids)

    person_count = 0

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = all_boxes[i]
        confidence = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None
        if track_id is None:
            continue

        _track_seen_count[cam_name][track_id] = _track_seen_count[cam_name].get(track_id, 0) + 1
        crop_w, crop_h = (x2 - x1), (y2 - y1)

        overlapping = any(_iou((x1, y1, x2, y2), other) >= OVERLAP_IOU_SKIP
                           for j, other in enumerate(all_boxes) if j != i)
        on_edge = _touches_edge(x1, y1, x2, y2, frame_w, frame_h)
        big_enough = crop_w >= MIN_CROP_WIDTH and crop_h >= MIN_CROP_HEIGHT
        good_quality = (not overlapping) and (not on_edge) and big_enough

        # --- DIAGNOSTIC PRINT ---
        if not good_quality:
            print(f"[{cam_name}] track {track_id} REJECTED: "
                  f"overlap={overlapping} edge={on_edge} size=({crop_w}x{crop_h})")

        if track_id not in _known_track_ids[cam_name]:
            stable_enough = _track_seen_count[cam_name][track_id] >= MIN_TRACK_AGE

            if stable_enough and good_quality:
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                embedding = get_embedding(crop)
                color_hist = get_color_signature(crop)

                if embedding is not None:
                    exclude = _active_global_ids(exclude_cam=cam_name, exclude_track=track_id)
                    global_id = _gallery.match_or_register(embedding, color_hist, exclude_ids=exclude)
                    _track_to_global[cam_name][track_id] = global_id
                    _known_track_ids[cam_name].add(track_id)
                else:
                    global_id = f"?{track_id}"
            else:
                global_id = f"?{track_id}"   # not confident yet -- shown, not finalized
        else:
            global_id = _track_to_global[cam_name].get(track_id, track_id)
            # periodically enrich this identity's bank and color with fresh crops
            if good_quality and _track_seen_count[cam_name][track_id] % UPDATE_EVERY_N_FRAMES == 0:
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                embedding = get_embedding(crop)
                color_hist = get_color_signature(crop)

                if embedding is not None:
                    _gallery.update(global_id, embedding, color_hist=color_hist)

        color = _color_for_id(global_id)
        _draw_label(frame, x1, y1, x2, y2, f"ID {global_id} | {confidence:.2f}", color)
        person_count += 1

    cv2.putText(frame, f"{cam_name} | people: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    return frame