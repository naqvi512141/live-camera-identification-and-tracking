"""
gallery.py
-----------
Maintains the set of "people we've seen so far". Each identity stores
a small BANK of recent embeddings (not one frozen average) so it can
recognize a person across different poses/angles seen over time, along
with a color signature of their torso clothing.

match_or_register(): used when a track is NEW — compares against the
gallery using a weighted fusion of deep ReID embedding similarity and
clothing color histogram similarity. Optionally excludes IDs already
active on other live tracks this frame (prevents two visible people
ever sharing one ID).

update(): used on an ALREADY-matched, stable track — adds another
good-quality embedding to that identity's bank and refreshes its color
signature periodically.
"""

import numpy as np
import time
from collections import deque
from color_signature import color_similarity

# Weight balancing between deep embedding and clothing color similarity
EMBEDDING_WEIGHT = 0.7   # deep embedding trusted more; color as a supporting signal
COLOR_WEIGHT = 0.3

# Combined weighted cutoff for "same person"
FUSED_THRESHOLD = 0.68

# Drop a gallery entry if unseen for this long (seconds) -- prevents
# unbounded growth and very-old false matches.
GALLERY_TIMEOUT = 300

# How many recent embeddings to keep per identity.
EMBEDDING_BANK_SIZE = 8


class Gallery:
    def __init__(self):
        self._next_id = 1
        self._entries = {}   # global_id -> {"bank": deque, "color": hist, "last_seen": ts}

    def _cosine_similarity(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _prune_stale(self):
        now = time.time()
        stale = [gid for gid, e in self._entries.items()
                 if now - e["last_seen"] > GALLERY_TIMEOUT]
        for gid in stale:
            del self._entries[gid]

    def _best_score(self, embedding, gid):
        bank = self._entries[gid]["bank"]
        scores = sorted((self._cosine_similarity(embedding, e) for e in bank), reverse=True)
        top_k = scores[:4] if len(scores) >= 4 else scores
        return sum(top_k) / len(top_k)

    def match_or_register(self, embedding, color_hist, exclude_ids=None):
        """
        exclude_ids: global_ids currently active on OTHER live tracks
        this frame -- never matched against, forcing a fresh ID
        instead of stealing an ID that's genuinely in use elsewhere.
        """
        self._prune_stale()
        exclude_ids = exclude_ids or set()

        best_id, best_score = None, -1.0
        best_emb_score, best_col_score = 0.0, 0.0

        for gid in self._entries:
            if gid in exclude_ids:
                continue
            emb_score = self._best_score(embedding, gid)
            col_score = color_similarity(color_hist, self._entries[gid]["color"])
            fused = (EMBEDDING_WEIGHT * emb_score) + (COLOR_WEIGHT * col_score)

            if fused > best_score:
                best_score = fused
                best_id = gid
                best_emb_score = emb_score
                best_col_score = col_score

        # --- DIAGNOSTIC PRINT ---
        if best_id is not None:
            print(f"[gallery] closest match: ID {best_id} @ fused {best_score:.3f} "
                  f"(emb: {best_emb_score:.3f}, color: {best_col_score:.3f}, threshold {FUSED_THRESHOLD})")

        if best_id is not None and best_score >= FUSED_THRESHOLD:
            self._entries[best_id]["bank"].append(embedding)
            self._entries[best_id]["color"] = color_hist   # update reference color signature
            self._entries[best_id]["last_seen"] = time.time()
            return best_id

        new_id = self._next_id
        self._next_id += 1
        self._entries[new_id] = {
            "bank": deque([embedding], maxlen=EMBEDDING_BANK_SIZE),
            "color": color_hist,
            "last_seen": time.time(),
        }
        return new_id

    def update(self, global_id, embedding, color_hist=None):
        """Add another embedding and refresh the color histogram for an
        ALREADY-KNOWN identity's bank without running matching. Called
        periodically while a track is stable so the identity accumulates
        multiple poses/angles over time."""
        if global_id in self._entries:
            self._entries[global_id]["bank"].append(embedding)
            if color_hist is not None:
                self._entries[global_id]["color"] = color_hist
            self._entries[global_id]["last_seen"] = time.time()