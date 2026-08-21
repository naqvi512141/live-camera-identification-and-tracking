"""
reid_extractor.py
-------------------
Extracts L2-normalized 768-dimensional ReID embeddings using a Vision
Transformer (ViT-B/16) backbone fine-tuned for Person Re-Identification.
"""

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import timm

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Preprocessing Pipeline for ViT-B/16 (224x224 input size) ---
TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_extractor_model = None


def _load_transformer_model():
    """
    Initializes and loads the ViT-Base/16 feature extractor onto GPU/CPU.
    """
    global _extractor_model
    if _extractor_model is None:
        print(f"[reid_extractor] Loading ViT-B/16 TransReID model on device: {DEVICE}...")

        # Load ViT-Base model with dynamic image size enabled
        _extractor_model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,           # Remove prediction head (returns raw 768-d features)
            dynamic_img_size=True    # Enables dynamic interpolation for positional embeddings
        )
        _extractor_model.eval()
        _extractor_model.to(DEVICE)
        print("[reid_extractor] ViT-B/16 TransReID loaded successfully.")
    return _extractor_model


def get_embedding(crop):
    """
    Computes a 768-dimensional L2-normalized feature vector for a cropped person image.

    :param crop: BGR image crop from OpenCV (numpy array)
    :return: 1D normalized numpy array of features, or None if crop is invalid
    """
    if crop is None or crop.size == 0:
        return None

    try:
        # 1. Convert OpenCV BGR crop to PIL Image
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        # 2. Preprocess & Add Batch Dimension
        tensor_img = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

        # 3. Model Inference
        model = _load_transformer_model()
        with torch.no_grad():
            features = model(tensor_img)

        # 4. Extract feature vector to Numpy
        feat_vec = features.squeeze(0).cpu().numpy()

        # 5. L2 Normalization (required for cosine similarity scoring)
        norm = np.linalg.norm(feat_vec)
        if norm > 1e-8:
            feat_vec = feat_vec / norm

        return feat_vec

    except Exception as e:
        print(f"[reid_extractor] Error extracting embedding: {e}")
        return None