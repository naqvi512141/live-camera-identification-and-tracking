"""
gallery.py
-----------
Maintains identities seen so far. Each identity stores a small BANK of
recent embeddings (not one blended average) for robustness, and the
matcher refuses to assign an ID that's already actively in use by
someone else currently on screen.
"""

import numpy as np
import time
from collections import deque

SIMILARITY_THRESHOLD = 0.72       # raised from 0.65 — retune empirically for your room
GALLERY_TIMEOUT = 300
EMBEDDING_BANK_SIZE = 5            # keep last N embeddings per identity


class Gallery:
    def __init__(self):
        self._next_id = 1
        self._entries = {}   # global_id -> {"bank": deque, "last_seen": ts}

    def _cosine_similarity(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _prune_stale(self):
        now = time.time()
        stale = [gid for gid, e in self._entries.items()
                 if now - e["last_seen"] > GALLERY_TIMEOUT]
        for gid in stale:
            del self._entries[gid]

    def _best_score(self, embedding, gid):
        # compare against EVERY stored embedding for this identity,
        # take the best match — more robust than one blended average
        bank = self._entries[gid]["bank"]
        return max(self._cosine_similarity(embedding, e) for e in bank)

    def match_or_register(self, embedding, exclude_ids=None):
        """
        exclude_ids: global_ids currently active on OTHER live tracks
        this frame — never matched against, forcing a fresh ID instead
        of stealing someone else's.
        """
        self._prune_stale()
        exclude_ids = exclude_ids or set()

        best_id, best_score = None, -1.0
        for gid in self._entries:
            if gid in exclude_ids:
                continue
            score = self._best_score(embedding, gid)
            if score > best_score:
                best_score, best_id = score, gid

        if best_id is not None and best_score >= SIMILARITY_THRESHOLD:
            bank = self._entries[best_id]["bank"]
            bank.append(embedding)
            self._entries[best_id]["last_seen"] = time.time()
            return best_id

        new_id = self._next_id
        self._next_id += 1
        self._entries[new_id] = {
            "bank": deque([embedding], maxlen=EMBEDDING_BANK_SIZE),
            "last_seen": time.time(),
        }
        return new_id

    def update(self, global_id, embedding):
        """Add an additional embedding to an ALREADY-KNOWN identity's bank,
        without re-running matching. Called periodically while a track is
        stable, so the identity accumulates multiple poses/angles over time
        instead of being frozen at one first-seen snapshot."""
        if global_id in self._entries:
            self._entries[global_id]["bank"].append(embedding)
            self._entries[global_id]["last_seen"] = time.time()