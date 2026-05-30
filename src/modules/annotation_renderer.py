"""
批注渲染模块
负责将批改结果以可视化批注的形式渲染到作业图像上。
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


def _find_chinese_font() -> Optional[str]:
    """查找系统中可用的中文字体"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    return None


class AnnotationRenderer:
    """批注渲染器"""

    def __init__(self, font_path: Optional[str] = None, font_size: int = 24):
        self.font_path = font_path or _find_chinese_font()
        self.font_size = font_size

        self.correct_color = (0, 255, 0)
        self.wrong_color = (255, 0, 0)
        self.comment_color = (0, 0, 255)

    def _get_pil_font(self, size: int = None) -> ImageFont.FreeTypeFont:
        """获取PIL字体对象，用于中文文字渲染"""
        font_size = size or self.font_size
        if self.font_path and os.path.exists(self.font_path):
            return ImageFont.truetype(self.font_path, font_size)
        return ImageFont.load_default()

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """将PIL Image转换为OpenCV格式（numpy数组）"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """将OpenCV图像转换为PIL Image格式"""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    def draw_correct_mark(self, image: np.ndarray, position: Tuple[int, int], radius: int = 15) -> np.ndarray:
        """绘制正确标记（绿色对勾）"""
        result = image.copy()
        x, y = position
        pts = np.array([
            [x - radius, y],
            [x - radius // 3, y + radius],
            [x + radius, y - radius // 2]
        ], np.int32)
        cv2.polylines(result, [pts], isClosed=False, color=self.correct_color, thickness=3)
        return result

    def draw_wrong_mark(self, image: np.ndarray, position: Tuple[int, int], radius: int = 15) -> np.ndarray:
        """绘制错误标记（红色叉号）"""
        result = image.copy()
        x, y = position
        r = radius
        cv2.line(result, (x - r, y - r), (x + r, y + r), self.wrong_color, thickness=3)
        cv2.line(result, (x + r, y - r), (x - r, y + r), self.wrong_color, thickness=3)
        return result

    def draw_text_annotation(self, image: np.ndarray, text: str, position: Tuple[int, int],
                             color: Tuple[int, int, int] = None) -> np.ndarray:
        """绘制文字批注（支持中文）"""
        if not text:
            return image
        color = color or self.comment_color
        x, y = position
        pil_image = self._cv2_to_pil(image)
        draw = ImageDraw.Draw(pil_image)
        font = self._get_pil_font()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = max(0, x - text_w // 2)
        text_y = max(0, y - text_h - 10)
        text_w = min(text_w, image.shape[1] - text_x)
        text_h = min(text_h, image.shape[0] - text_y)
        padding = 4
        bg_x1 = max(0, text_x - padding)
        bg_y1 = max(0, text_y - padding)
        bg_x2 = min(image.shape[1], text_x + text_w + padding)
        bg_y2 = min(image.shape[0], text_y + text_h + padding)
        bg_color = (240, 240, 240, 200)
        overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)
        pil_image = pil_image.convert("RGBA")
        pil_image = Image.alpha_composite(pil_image, overlay)
        pil_image = pil_image.convert("RGB")
        draw = ImageDraw.Draw(pil_image)
        draw.text((text_x, text_y), text, font=font, fill=color)
        return self._pil_to_cv2(pil_image)

    def draw_bounding_box(self, image: np.ndarray, box: Tuple[int, int, int, int],
                          color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        """绘制边界框"""
        x1, y1, x2, y2 = box
        return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    def render_score(self, image: np.ndarray, score: float, total: float,
                     position: Tuple[int, int]) -> np.ndarray:
        """渲染分数"""
        if total == 0:
            score_text = f"得分: {score:.1f}"
        else:
            score_text = f"得分: {score:.1f} / {total:.1f}"
        ratio = score / total if total > 0 else 0
        if ratio >= 0.9:
            score_color = (0, 180, 0)
        elif ratio >= 0.6:
            score_color = (255, 165, 0)
        else:
            score_color = (255, 0, 0)
        return self.draw_text_annotation(image, score_text, position, color=score_color)

    def render_all(self, image: np.ndarray, corrections: List[Dict]) -> np.ndarray:
        """根据批改结果渲染所有批注"""
        result = image.copy()
        total_score = 0.0
        total_max = 0.0
        last_position = (50, 50)
        for correction in corrections:
            position = correction.get("position", (0, 0))
            is_correct = correction.get("is_correct", True)
            comment = correction.get("comment", "")
            score = correction.get("score", 0)
            max_score = correction.get("max_score", 0)
            if is_correct:
                result = self.draw_correct_mark(result, position)
            else:
                result = self.draw_wrong_mark(result, position)
            if comment:
                comment_pos = (position[0] + 40, position[1])
                result = self.draw_text_annotation(result, comment, comment_pos)
            total_score += score
            total_max += max_score
            last_position = position
        if total_max > 0:
            score_pos = (last_position[0], last_position[1] + 60)
            result = self.render_score(result, total_score, total_max, score_pos)
        return result