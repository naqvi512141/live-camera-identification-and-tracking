"""
utils/fps_counter.py
---------------------
Measures the ACTUAL achieved fps of the main processing loop, so you
can verify you're really hitting the target rather than assuming it.
"""

import time


class FPSCounter:
    def __init__(self, report_every=1.0):
        self.report_every = report_every
        self._count = 0
        self._window_start = time.time()
        self.current_fps = 0.0

    def tick(self):
        """Call once per processed frame/loop iteration."""
        self._count += 1
        elapsed = time.time() - self._window_start
        if elapsed >= self.report_every:
            self.current_fps = self._count / elapsed
            self._count = 0
            self._window_start = time.time()
            return self.current_fps
        return None