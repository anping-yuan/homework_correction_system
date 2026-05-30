"""
图像预处理模块
负责对采集到的作业图像进行预处理操作，包括灰度化、二值化、去噪、纠偏等。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class ImageProcessor:
    """图像预处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.denoise_strength = self.config.get("denoise_strength", 10)
        self.binarize_method = self.config.get("binarize_method", "otsu")
        self.resize_width = self.config.get("resize_width", None)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """转换为灰度图"""
        if image is None or image.size == 0:
            return image
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
        if image is None or image.size == 0:
            return image
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
        return cv2.fastNlMeansDenoisingColored(image, None, strength, 10, 7, 21)

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """纠偏处理：校正倾斜图像"""
        if image is None or image.size == 0:
            return image
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
        if abs(angle) < 0.5:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def resize(self, image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        """调整图像尺寸"""
        if width is None and height is None:
            return image
        if image is None or image.size == 0:
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

    def enhance_contrast(self, image: np.ndarray, clip_limit: float = 2.0,
                         tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """使用CLAHE增强图像对比度"""
        if image is None or image.size == 0:
            return image
        gray = self.to_grayscale(image)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_channel = clahe.apply(l_channel)
            lab = cv2.merge([l_channel, a_channel, b_channel])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    def sharpen(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """图像锐化处理"""
        if image is None or image.size == 0:
            return image
        kernel = np.array([
            [0, -1, 0],
            [-1, 4 + strength, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        result = cv2.filter2D(image, -1, kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """旋转图像指定角度"""
        if image is None or image.size == 0:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def find_paper_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """检测试卷纸张的四角轮廓"""
        if image is None or image.size == 0:
            return None
        gray = self.to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        for cnt in contours[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if 4 <= len(approx) <= 6:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                return box.astype(np.float32)
        return None

    def warp_perspective(self, image: np.ndarray, src_points: np.ndarray,
                         target_width: int = 1200, target_height: int = 1600) -> np.ndarray:
        """对图像进行透视变换，校正拍摄角度"""
        if image is None or image.size == 0:
            return image
        dst_points = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ], dtype=np.float32)
        src_points = src_points.astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, matrix, (target_width, target_height))

    def auto_correct_and_crop(self, image: np.ndarray,
                               target_width: int = 1200,
                               target_height: int = 1600) -> Optional[np.ndarray]:
        """自动检测纸张并校正裁剪"""
        if image is None or image.size == 0:
            return None
        contour = self.find_paper_contour(image)
        if contour is None:
            return None
        return self.warp_perspective(image, contour, target_width, target_height)

    def split_regions(self, image: np.ndarray, positions: List[List[int]]) -> List[np.ndarray]:
        """根据位置信息从图像中切割出多个区域"""
        if image is None or image.size == 0:
            return []
        regions = []
        for pos in positions:
            if len(pos) == 4:
                x1, y1, x2, y2 = map(int, pos)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    regions.append(image[y1:y2, x1:x2])
        return regions

    def process(self, image: np.ndarray) -> np.ndarray:
        """执行完整预处理流水线：去噪 → 纠偏"""
        if image is None or image.size == 0:
            return image
        image = self.denoise(image)
        image = self.deskew(image)
        return image

    def process_with_config(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """根据配置执行可定制的预处理流水线"""
        if image is None or image.size == 0:
            return image
        cfg = config or self.config

        if cfg.get("denoise", True):
            strength = cfg.get("denoise_strength", self.denoise_strength)
            image = self.denoise(image, strength=strength)

        if cfg.get("deskew", True):
            image = self.deskew(image)

        if cfg.get("enhance_contrast", False):
            clip = cfg.get("contrast_clip_limit", 2.0)
            image = self.enhance_contrast(image, clip_limit=clip)

        if cfg.get("sharpen", False):
            strength = cfg.get("sharpen_strength", 1.0)
            image = self.sharpen(image, strength=strength)

        if cfg.get("binarize", False):
            method = cfg.get("binarize_method", self.binarize_method)
            image = self.binarize(image, method=method)

        if cfg.get("resize_width") is not None or cfg.get("resize_height") is not None:
            w = cfg.get("resize_width")
            h = cfg.get("resize_height")
            image = self.resize(image, width=w, height=h)

        return image