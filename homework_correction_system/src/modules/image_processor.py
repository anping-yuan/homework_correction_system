"""
图像预处理模块
负责对采集到的作业图像进行预处理操作，包括灰度化、二值化、去噪、纠偏等。
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImageProcessor:
    """图像预处理器"""

    def __init__(self):
        pass

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """转换为灰度图"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def binarize(self, image: np.ndarray, method: str = "otsu") -> np.ndarray:
        """二值化处理"""
        gray = self.to_grayscale(image)
        if method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary

    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """去噪处理"""
        return cv2.fastNlMeansDenoisingColored(image, None, strength, 10, 7, 21)

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """纠偏处理：校正倾斜图像"""
        gray = self.to_grayscale(image)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        else:
            angle = -angle
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def resize(self, image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        """调整图像尺寸"""
        if width is None and height is None:
            return image
        h, w = image.shape[:2]
        if width is not None and height is not None:
            return cv2.resize(image, (width, height))
        elif width is not None:
            ratio = width / w
            return cv2.resize(image, (width, int(h * ratio)))
        else:
            ratio = height / h
            return cv2.resize(image, (int(w * ratio), height))

    def process(self, image: np.ndarray) -> np.ndarray:
        """执行完整预处理流水线"""
        image = self.denoise(image)
        image = self.deskew(image)
        return image