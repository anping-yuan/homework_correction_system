"""
全流程测试脚本
测试作业批改系统的完整流水线：预处理 → OCR → 批改 → 渲染
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import process_single_image
from src.utils.helpers import load_config, ensure_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"


def find_images(d