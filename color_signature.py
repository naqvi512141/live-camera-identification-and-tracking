"""
color_signature.py
--------------------
Computes a color histogram "signature" for the torso region of a
detected person, and compares two signatures for similarity. Used
as a complementary signal alongside the deep ReID embedding --
color is cheap and pose-invariant, but can be fooled by two people
wearing similar clothes; the deep embedding is the opposite tradeoff.
Combining both is more robust than either alone.
"""

import cv2
import numpy as np


def get_color_signature(crop):
    """
    Extracts an HSV histogram from the TORSO region only (middle
    third of the crop, vertically) -- avoids head (skin tone, hair,
    not clothing) and legs/feet (often background-contaminated at
    crop edges), focusing on the most clothing-representative area.
    """
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    torso = crop[int(h * 0.25):int(h * 0.65), :]  # middle third, roughly shoulders-to-waist
    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    # Hue + Saturation histogram (ignore Value/brightness -- lighting
    # changes shouldn't count as a different color)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def color_similarity(hist_a, hist_b):
    """Returns a similarity score in [0, 1], higher = more similar."""
    if hist_a is None or hist_b is None:
        return 0.0
    score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return max(0.0, float(score))  # correlation can go slightly negative; clamp