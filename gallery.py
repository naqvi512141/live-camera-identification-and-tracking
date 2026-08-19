"""
gallery.py
-----------
Maintains the set of "people we've seen so far", each represented by
an embedding. New embeddings are compared against this gallery using
cosine similarity to decide: "same person as before" vs "new person".
"""

import numpy as np
import time

# How similar two embeddings must be (cosine similarity, range -1 to 1,
# 1 = identical direction) to be considered the SAME person.
# Higher = stricter (fewer false merges, more new/duplicate IDs).
# Lower  = looser (fewer duplicate IDs, more risk of merging two
# different people). 0.6-0.7 is a reasonable starting point — you WILL
# want to tune this by testing, note it down as you experiment.
SIMILARITY_THRESHOLD = 0.65

# Drop a gallery entry if we haven't matched it in this long (seconds).
# Prevents the gallery growing forever and prevents very old
# appearances from wrongly matching someone new much later.
GALLERY_TIMEOUT = 300  # 5 minutes


class Gallery:
    def __init__(self):
        self._next_id = 1
        # global_id -> {"embedding": vector, "last_seen": timestamp}
        self._entries = {}

    def _cosine_similarity(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _prune_stale(self):
        now = time.time()
        stale = [gid for gid, e in self._entries.items()
                 if now - e["last_seen"] > GALLERY_TIMEOUT]
        for gid in stale:
            del self._entries[gid]

    def match_or_register(self, embedding):
        """
        Compare `embedding` against the gallery.
        Returns an existing global_id if similar enough to someone
        already known, otherwise registers a new global_id.
        """
        self._prune_stale()

        best_id = None
        best_score = -1.0

        for gid, entry in self._entries.items():
            score = self._cosine_similarity(embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_id = gid

        if best_id is not None and best_score >= SIMILARITY_THRESHOLD:
            # Same person as before — refresh their stored embedding with
            # a running average, so it adapts to lighting/angle over time
            # rather than freezing on one snapshot.
            old = self._entries[best_id]["embedding"]
            updated = 0.7 * old + 0.3 * embedding
            self._entries[best_id]["embedding"] = updated
            self._entries[best_id]["last_seen"] = time.time()
            return best_id

        # No good match — genuinely new person
        new_id = self._next_id
        self._next_id += 1
        self._entries[new_id] = {"embedding": embedding, "last_seen": time.time()}
        return new_id