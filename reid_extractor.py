"""
reid_extractor.py
-------------------
Wraps a pretrained OSNet ReID model. Given a cropped person image,
returns a fixed-length embedding vector (from your notes: the point
in "embedding space" representing this person's appearance).
"""

from torchreid.reid.utils import FeatureExtractor

# Loaded ONCE at import time — same reasoning as the YOLO model:
# this is expensive to load, cheap to call repeatedly.
_extractor = FeatureExtractor(
    model_name="osnet_x0_25",
    model_path="",     # empty = auto-download pretrained ImageNet+ReID weights
    device="cpu",       # switch to "cuda" here if you confirm a GPU is available
)


def get_embedding(crop):
    """
    crop: a BGR image (numpy array) of a single detected person,
          e.g. frame[y1:y2, x1:x2]
    returns: a 1D embedding vector (numpy array)
    """
    if crop is None or crop.size == 0:
        return None
    # FeatureExtractor accepts a list of images, returns a batch of embeddings
    features = _extractor([crop])
    return features[0].cpu().numpy()