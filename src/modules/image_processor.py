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

    def bilateral_denoise(self, image: np.ndarray,
                          d: int = 9, sigma_color: float = 75,
                          sigma_space: float = 75) -> np.ndarray:
        """双边滤波去噪 — 保边去噪，比 NLM 更好地保护文字笔迹边缘。

        Args:
            d: 滤波窗口直径，9 对作业图片效果最佳
            sigma_color: 颜色空间标准差，越大则越大的颜色差异被平滑
            sigma_space: 坐标空间标准差，越大则越远的像素互相影响
        """
        if image is None or image.size == 0:
            return image
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def normalize_illumination(self, image: np.ndarray,
                                blur_kernel: int = 61,
                                strength: float = 0.85) -> np.ndarray:
        """光照归一化 / 去阴影 — 消除手机拍摄造成的光照不均。

        原理：将图像除以自身的高斯模糊版本，抵消大范围光照变化，
        保留文字等高频细节。这是 Adobe Scan / CamScanner 的核心技术。

        Args:
            blur_kernel: 高斯核大小，越大去除越均匀的光照变化；61 对 A4 作业合适
            strength: 混合强度，0=原图 1=完全归一化，0.85 保留一些自然感
        """
        if image is None or image.size == 0:
            return image

        is_color = len(image.shape) == 3

        if is_color:
            # LAB 色彩空间：只在亮度通道上做归一化
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            l_channel = lab[:, :, 0]
            # 大核高斯模糊获得光照估计
            l_blur = cv2.GaussianBlur(l_channel, (blur_kernel, blur_kernel), 0)
            # 归一化：I' = I / blur(I) * mean(blur(I))
            l_blur = np.where(l_blur < 1, 1, l_blur)  # 防除零
            l_norm = l_channel / l_blur * np.mean(l_blur)
            l_norm = np.clip(l_norm, 0, 255)
            # 混合原图和归一化结果
            l_mix = l_channel * (1 - strength) + l_norm * strength
            lab[:, :, 0] = np.clip(l_mix, 0, 255)
            lab = lab.astype(np.uint8)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            img_f = image.astype(np.float32)
            blur = cv2.GaussianBlur(img_f, (blur_kernel, blur_kernel), 0)
            blur = np.where(blur < 1, 1, blur)
            norm = img_f / blur * np.mean(blur)
            norm = np.clip(norm, 0, 255)
            mix = img_f * (1 - strength) + norm * strength
            return np.clip(mix, 0, 255).astype(np.uint8)

    def unsharp_mask(self, image: np.ndarray,
                     sigma: float = 1.5, amount: float = 1.2,
                     threshold: int = 0) -> np.ndarray:
        """USM 锐化 — 比简单 kernel 锐化更自然，不产生光晕伪影。

        Args:
            sigma: 高斯模糊标准差，控制锐化半径
            amount: 锐化强度，1.0=原图，1.5=强锐化
            threshold: 像素差值小于此值的区域不锐化，避免噪点增强
        """
        if image is None or image.size == 0:
            return image
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        if threshold > 0:
            diff = cv2.absdiff(image, blurred)
            mask = cv2.threshold(
                cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else diff,
                threshold, 255, cv2.THRESH_BINARY
            )[1]
            if len(image.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
            return np.where(mask > 0, sharpened, image).astype(np.uint8)
        else:
            result = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
            return np.clip(result, 0, 255).astype(np.uint8)

    # ═══════════════════════════════════════════════════════════════
    #  进阶方法：MSRCR 光照归一化 / Sauvola 二值化 / 形态学 / 高反差保留
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _ssr(img_float: np.ndarray, sigma: float) -> np.ndarray:
        """单尺度 Retinex"""
        blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
        blur = np.where(blur < 1, 1, blur)
        return np.log(img_float) - np.log(blur)

    def msrcr(self, image: np.ndarray,
              sigma_list: List[float] = None,
              restore_factor: float = 0.1) -> np.ndarray:
        """MSRCR 多尺度 Retinex 光照归一化。

        比简单的高斯除模糊更精细：多尺度处理同时保留局部细节（小 sigma）
        和全局光照（大 sigma），色彩恢复避免灰化。

        Args:
            sigma_list: 多尺度高斯核列表，默认 [15, 80, 250]
            restore_factor: 色彩恢复强度，0.1 对文档图片最佳
        """
        if image is None or image.size == 0:
            return image
        if sigma_list is None:
            sigma_list = [15, 80, 250]

        img_float = image.astype(np.float32) + 1.0  # +1 防 log(0)

        if len(image.shape) == 3:
            # 分通道 MSR
            b, g, r = cv2.split(img_float)
            msr_b = sum(self._ssr(b, s) for s in sigma_list) / len(sigma_list)
            msr_g = sum(self._ssr(g, s) for s in sigma_list) / len(sigma_list)
            msr_r = sum(self._ssr(r, s) for s in sigma_list) / len(sigma_list)
            msr = cv2.merge([msr_b, msr_g, msr_r])

            # 色彩恢复：CR = log(α·I) - log(sum(I))
            img_sum = np.sum(img_float, axis=2, keepdims=True)
            img_sum = np.where(img_sum < 1, 1, img_sum)
            cr = np.log(restore_factor * img_float) - np.log(img_sum)
            msrcr = msr * cr
        else:
            msr = sum(self._ssr(img_float, s) for s in sigma_list) / len(sigma_list)
            msrcr = msr

        # 归一化到 [0, 255]
        msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX)
        return msrcr.astype(np.uint8)

    def sauvola_binarize(self, image: np.ndarray,
                          window_size: int = 15,
                          k: float = 0.12,
                          r: float = 128.0) -> np.ndarray:
        """Sauvola 自适应二值化 — 手写文字专用。

        比 OTSU 更适合光照不均的文档：对每个像素用局部窗口的均值和
        标准差计算阈值，阴影区自动降低阈值。

        公式: T = m * (1 + k * (s/R - 1))

        Args:
            window_size: 局部窗口大小，15~21 推荐，不小于15防断笔画
            k: 敏感度，0.1~0.15，越小越敏感（更多像素判为文字）
            r: 标准差动态范围，128 是标准值
        """
        if image is None or image.size == 0:
            return image

        gray = self.to_grayscale(image).astype(np.float64)

        # 局部均值
        mean = cv2.blur(gray, (window_size, window_size))
        # 局部平方均值
        mean_sq = cv2.blur(gray * gray, (window_size, window_size))
        # 局部标准差
        var = mean_sq - mean * mean
        var = np.maximum(var, 0)
        std = np.sqrt(var)

        # Sauvola 阈值
        threshold = mean * (1.0 + k * (std / r - 1.0))

        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
        return binary

    def morphological_clean(self, image: np.ndarray,
                             kernel_size: int = 3) -> np.ndarray:
        """形态学开运算 — 修复断笔 + 消除背景噪点。

        先腐蚀（填白点/连断笔）→ 再膨胀（恢复粗细/去黑噪点）。
        """
        if image is None or image.size == 0:
            return image
        is_gray = len(image.shape) == 2
        if not is_gray:
            gray = self.to_grayscale(image)
        else:
            gray = image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # 开运算 = 先腐蚀后膨胀
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        if not is_gray:
            return cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
        return opened

    def high_pass_enhance(self, image: np.ndarray,
                           radius: float = 0.6,
                           opacity: float = 0.5) -> np.ndarray:
        """高反差保留增强 — 比 USM 更自然的锐化方式。

        提取图像高频成分（高反差保留），以叠加模式混合回原图，
        不会产生 USM 的白边光晕。

        Args:
            radius: 高反差保留半径，0.5~0.8 匹配手写细笔画
            opacity: 叠加强度，0.4~0.6 推荐
        """
        if image is None or image.size == 0:
            return image

        # 高斯模糊 → 高反差 = 原图 - 模糊
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        blurred = cv2.GaussianBlur(gray, (0, 0), radius)
        highpass = gray - blurred

        # 叠加模式：result = base + (2 * highpass * base / 255)
        # 简化：原图 + 高频 * opacity
        enhanced = gray + highpass * opacity
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        if len(image.shape) == 3:
            # 只在亮度通道做高反差保留
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            l_blur = cv2.GaussianBlur(l_channel, (0, 0), radius)
            l_highpass = l_channel - l_blur
            l_enhanced = l_channel + l_highpass * opacity
            lab[:, :, 0] = np.clip(l_enhanced, 0, 255).astype(np.uint8)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def linear_contrast_stretch(self, image: np.ndarray,
                                 low_pct: float = 2.0,
                                 high_pct: float = 2.0) -> np.ndarray:
        """线性对比度拉伸 — 裁掉直方图两端极值，拉伸到全范围。

        Args:
            low_pct: 低端裁切百分比
            high_pct: 高端裁切百分比
        """
        if image is None or image.size == 0:
            return image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        h, w = gray.shape
        n = h * w
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cumsum = np.cumsum(hist)
        low_val = np.searchsorted(cumsum, n * low_pct / 100.0)
        high_val = np.searchsorted(cumsum, n * (1.0 - high_pct / 100.0))
        stretched = np.clip((gray.astype(np.float32) - low_val) / max(high_val - low_val, 1) * 255, 0, 255)
        if len(image.shape) == 3:
            stretched = cv2.cvtColor(stretched.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return stretched.astype(np.uint8)

    # ═══════════════════════════════════════════════════════════════
    #  双模式预处理管线
    # ═══════════════════════════════════════════════════════════════

    def enhance_for_vlm(self, image: np.ndarray) -> np.ndarray:
        """VLM 预处理管线 — 送视觉模型。

        流程: 光照归一化 → 双边滤波保边去噪 → CLAHE对比度增强 → USM锐化
        """
        if image is None or image.size == 0:
            return image

        # Step 1: 光照归一化 — 消除手机拍摄阴影
        image = self.normalize_illumination(image)

        # Step 2: 双边滤波 — 保边去噪
        image = self.bilateral_denoise(image, d=7, sigma_color=25, sigma_space=8)

        # Step 3: CLAHE 对比度增强
        image = self.enhance_contrast(image, clip_limit=1.5, tile_grid_size=(8, 8))

        # Step 4: USM 锐化
        image = self.unsharp_mask(image, sigma=1.0, amount=0.8, threshold=3)

        return image

    def enhance_document(self, image: np.ndarray) -> np.ndarray:
        """文档增强管线 — 给人看。与 enhance_for_vlm 相同流程。

        流程: 光照归一化 → 双边滤波保边去噪 → CLAHE对比度增强 → USM锐化
        """
        return self.enhance_for_vlm(image)

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

    @staticmethod
    def stitch_vertical(image_paths: List[str], target_width: int = 1200) -> np.ndarray:
        """将多张图片纵向拼接为一张长图

        用于长题目分多张照片拍摄后合成完整试卷的场景。
        所有图片会被缩放到相同宽度后上下拼接，中间加一条分割线便于
        后续切题 API 区分不同页面。

        Args:
            image_paths: 图片文件路径列表（按从上到下的顺序）
            target_width: 统一缩放到的目标宽度（像素），0 表示不缩放

        Returns:
            拼接后的 BGR 图像 (numpy ndarray)
        """
        images = []
        for p in image_paths:
            # 使用 np.fromfile + cv2.imdecode 代替 cv2.imread，
            # 因为 cv2.imread 在 Windows 上无法处理中文路径
            img_array = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图片: {p}")
            images.append(img)

        if len(images) == 1:
            return images[0]

        # 统一缩放到相同宽度
        if target_width > 0:
            resized = []
            for img in images:
                h, w = img.shape[:2]
                if w != target_width:
                    ratio = target_width / w
                    img = cv2.resize(img, (target_width, int(h * ratio)))
                resized.append(img)
            images = resized

        # 图片之间加一条白色分隔线（帮助切题 API 识别页面边界）
        separator_height = 20
        sep = np.ones((separator_height, target_width, 3), dtype=np.uint8) * 255
        # 分隔线中间画一条虚线标记
        cv2.line(sep, (50, separator_height // 2),
                 (target_width - 50, separator_height // 2), (200, 200, 200), 1)

        # 纵向拼接
        parts = []
        for i, img in enumerate(images):
            parts.append(img)
            if i < len(images) - 1:
                parts.append(sep)

        stitched = np.vstack(parts)
        return stitched