"""
摄像头采集模块
负责调用摄像头实时采集作业图像，支持单帧抓拍和连续采集。
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class CameraCapture:
    """摄像头采集器"""

    def __init__(self, camera_id: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        self.camera_id = camera_id
        self.target_width = width
        self.target_height = height
        self.cap: Optional[cv2.VideoCapture] = None

    @property
    def is_opened(self) -> bool:
        """摄像头是否已打开"""
        return self.cap is not None and self.cap.isOpened()

    def open(self) -> bool:
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.cap = None
            return False
        if self.target_width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        if self.target_height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        return True

    def set_resolution(self, width: int, height: int) -> bool:
        """设置摄像头分辨率"""
        self.target_width = width
        self.target_height = height
        if self.is_opened:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return True
        return False

    def get_resolution(self) -> Tuple[int, int]:
        """获取当前摄像头分辨率"""
        if not self.is_opened:
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def capture_frame(self) -> Optional[np.ndarray]:
        """抓拍一帧图像"""
        if not self.is_opened:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def capture_multiple(self, count: int, skip_frames: int = 5) -> List[np.ndarray]:
        """连续抓拍多帧图像"""
        frames = []
        for _ in range(count):
            for _ in range(skip_frames):
                self.cap.grab()
            frame = self.capture_frame()
            if frame is not None:
                frames.append(frame)
        return frames

    def save_frame(self, frame: np.ndarray, file_path: str) -> bool:
        """保存图像到文件"""
        if frame is None or frame.size == 0:
            return False
        return cv2.imwrite(file_path, frame)

    def close(self) -> None:
        """关闭摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        if not self.open():
            raise RuntimeError(f"无法打开摄像头 (camera_id={self.camera_id})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False