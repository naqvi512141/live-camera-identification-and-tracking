"""
config.py
---------
Configuration settings loaded safely from environment variables.
"""

import os
from dotenv import load_dotenv

# Load key-value pairs from the local .env file
load_dotenv()

CAMERAS = {
    "cam1": os.getenv("CAM1_URL", "rtsp://user:pass@IP:PORT/stream"),
    "cam2": os.getenv("CAM2_URL", "rtsp://user:pass@IP:PORT/stream"),
}

TARGET_FPS = 25
FRAME_INTERVAL = 1.0 / TARGET_FPS
RECONNECT_DELAY = 5

DISPLAY = True
DISPLAY_HEIGHT = 480
DISPLAY_WIDTH = 640