"""
摄像头采集模块
负责调用摄像头实时采集作业图像，支持单帧抓拍和连续采集。
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class CameraCapture:
    """摄像头采集器"""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
        return True

    def capture_frame(self) -> Optional[np.ndarray]:
        """抓拍一帧图像"""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def save_frame(self, frame: np.ndarray, file_path: str) -> bool:
        """保存图像到文件"""
        return cv2.imwrite(file_path, frame)

    def close(self) -> None:
        """关闭摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()