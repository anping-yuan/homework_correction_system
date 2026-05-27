# 单元测试
import pytest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modules.image_processor import ImageProcessor
from src.modules.annotation_renderer import AnnotationRenderer


class TestImageProcessor:

    def setup_method(self):
        self.processor = ImageProcessor()

    def test_to_grayscale_color_image(self):
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[:, :] = [128, 64, 32]
        gray = self.processor.to_grayscale(color_image)
        assert len(gray.shape) == 2

    def test_to_grayscale_grayscale_image(self):
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        gray = self.processor.to_grayscale(gray_image)
        assert len(gray.shape) == 2

    def test_binarize_otsu(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        image[30:70, 30:70] = 200
        binary = self.processor.binarize(image, method="otsu")
        assert binary.shape == image.shape
        assert binary.max() <= 255

    def test_resize_by_width(self):
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        resized = self.processor.resize(image, width=200)
        assert resized.shape[1] == 200
        assert resized.shape[0] == 100


class TestAnnotationRenderer:

    def setup_method(self):
        self.renderer = AnnotationRenderer()

    def test_draw_bounding_box(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.renderer.draw_bounding_box(image, (10, 10, 50, 50), (0, 255, 0))
        assert result.shape == image.shape