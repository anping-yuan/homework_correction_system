import pytest
import numpy as np
import cv2
import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modules.camera_capture import CameraCapture
from src.modules.image_processor import ImageProcessor
from src.modules.annotation_renderer import AnnotationRenderer
from src.modules.llm_grader import LLMGrader
from src.modules.aliyun_ocr import AliyunOCR


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

    def test_enhance_contrast_color(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.processor.enhance_contrast(image)
        assert result.shape == image.shape

    def test_enhance_contrast_gray(self):
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = self.processor.enhance_contrast(image)
        assert result.shape == image.shape

    def test_sharpen(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.processor.sharpen(image, strength=1.5)
        assert result.shape == image.shape

    def test_rotate(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = self.processor.rotate(image, 45.0)
        assert result.shape == image.shape

    def test_process_with_config(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        config = {
            "denoise": True,
            "deskew": True,
            "enhance_contrast": True,
            "sharpen": True,
        }
        result = self.processor.process_with_config(image, config)
        assert result.shape == image.shape

    def test_process_with_config_no_denoise(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        config = {"denoise": False, "deskew": False}
        result = self.processor.process_with_config(image, config)
        assert result.shape == image.shape

    def test_process_with_none_image(self):
        result = self.processor.process(None)
        assert result is None

    def test_enhance_contrast_with_none(self):
        result = self.processor.enhance_contrast(None)
        assert result is None

    def test_find_paper_contour_with_rectangular_image(self):
        image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 50), (350, 250), (0, 0, 0), 2)
        contour = self.processor.find_paper_contour(image)
        assert contour is not None

    def test_find_paper_contour_empty_image(self):
        result = self.processor.find_paper_contour(None)
        assert result is None

    def test_warp_perspective(self):
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        src = np.array([[50, 50], [350, 60], [340, 240], [40, 250]], dtype=np.float32)
        result = self.processor.warp_perspective(image, src, 200, 300)
        assert result.shape == (300, 200, 3)

    def test_auto_correct_and_crop_no_contour(self):
        image = np.ones((300, 400, 3), dtype=np.uint8) * 128
        result = self.processor.auto_correct_and_crop(image)
        assert result is None

    def test_split_regions(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        positions = [[10, 10, 50, 50], [60, 60, 100, 100]]
        regions = self.processor.split_regions(image, positions)
        assert len(regions) == 2
        assert regions[0].shape == (40, 40, 3)

    def test_split_regions_empty_image(self):
        result = self.processor.split_regions(None, [[0, 0, 10, 10]])
        assert result == []

    def test_denoise(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.processor.denoise(image)
        assert result.shape == image.shape

    def test_deskew(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.processor.deskew(image)
        assert result.shape == image.shape

    def test_process_with_resize(self):
        image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        config = {"denoise": False, "deskew": False, "resize_width": 150}
        result = self.processor.process_with_config(image, config)
        assert result.shape[1] == 150

    def test_process_with_binarize(self):
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        config = {"denoise": False, "deskew": False, "binarize": True}
        result = self.processor.process_with_config(image, config)
        assert result.shape == image.shape


class TestAnnotationRenderer:

    def setup_method(self):
        self.renderer = AnnotationRenderer()

    def test_draw_bounding_box(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.renderer.draw_bounding_box(image, (10, 10, 50, 50), (0, 255, 0))
        assert result.shape == image.shape

    def test_draw_correct_mark(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.renderer.draw_correct_mark(image, (100, 100))
        assert result.shape == image.shape

    def test_draw_wrong_mark(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.renderer.draw_wrong_mark(image, (100, 100))
        assert result.shape == image.shape

    def test_draw_text_annotation(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.renderer.draw_text_annotation(image, "正确", (100, 50))
        assert result.shape == image.shape

    def test_render_score(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.renderer.render_score(image, 85.0, 100.0, (100, 50))
        assert result.shape == image.shape

    def test_render_score_zero_total(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.renderer.render_score(image, 0.0, 0.0, (100, 50))
        assert result.shape == image.shape

    def test_render_all(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        corrections = [
            {"position": (50, 80), "is_correct": True, "comment": "正确", "score": 5, "max_score": 5},
            {"position": (50, 150), "is_correct": False, "comment": "错误", "score": 0, "max_score": 5},
        ]
        result = self.renderer.render_all(image, corrections)
        assert result.shape == image.shape

    def test_render_all_no_scores(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        corrections = [
            {"position": (50, 80), "is_correct": True, "comment": "", "score": 0, "max_score": 0},
        ]
        result = self.renderer.render_all(image, corrections)
        assert result.shape == image.shape

    def test_draw_text_annotation_empty_text(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.renderer.draw_text_annotation(image, "", (100, 50))
        assert result is image


class TestCameraCapture:

    def test_init_default(self):
        camera = CameraCapture()
        assert camera.camera_id == 0
        assert camera.is_opened is False

    def test_init_with_params(self):
        camera = CameraCapture(camera_id=1, width=1920, height=1080)
        assert camera.camera_id == 1
        assert camera.target_width == 1920
        assert camera.target_height == 1080

    def test_set_resolution(self):
        camera = CameraCapture()
        result = camera.set_resolution(1280, 720)
        assert result is False
        assert camera.target_width == 1280
        assert camera.target_height == 720

    def test_get_resolution_not_opened(self):
        camera = CameraCapture()
        w, h = camera.get_resolution()
        assert w == 0
        assert h == 0

    def test_save_frame_with_none(self):
        camera = CameraCapture()
        result = camera.save_frame(None, "dummy.jpg")
        assert result is False


class TestLLMGrader:

    def setup_method(self):
        self.grader = LLMGrader(api_key="test-key", model="gpt-4")

    def test_init(self):
        assert self.grader.api_key == "test-key"
        assert self.grader.model == "gpt-4"
        assert self.grader.temperature == 0.3

    def test_load_prompt(self):
        custom_prompt = "自定义批改提示词"
        self.grader.load_prompt(custom_prompt)
        assert self.grader.grading_prompt == custom_prompt

    def test_build_evaluation_context_no_reference(self):
        ctx = self.grader.build_evaluation_context("1+1=?", "2")
        assert "题目：1+1=?" in ctx
        assert "学生答案：2" in ctx
        assert "参考答案" not in ctx

    def test_build_evaluation_context_with_reference(self):
        ctx = self.grader.build_evaluation_context("1+1=?", "2", "2")
        assert "参考答案：2" in ctx

    def test_grade_empty_answer(self):
        result = self.grader.grade("1+1=?", "")
        assert result["is_correct"] is False
        assert result["score"] == 0
        assert result["comment"] == "未作答"

    def test_grade_blank_answer(self):
        result = self.grader.grade("1+1=?", "   ")
        assert result["is_correct"] is False
        assert result["score"] == 0

    def test_calculate_total_score(self):
        results = [
            {"max_score": 10, "score": 8},
            {"max_score": 5, "score": 5},
            {"max_score": 10, "score": 0},
        ]
        earned, total = self.grader.calculate_total_score(results)
        assert earned == 13
        assert total == 25

    def test_calculate_total_score_empty(self):
        earned, total = self.grader.calculate_total_score([])
        assert earned == 0
        assert total == 0

    def test_generate_feedback_empty(self):
        result = self.grader.generate_feedback([])
        assert isinstance(result, str)

    @patch("src.modules.llm_grader.requests.post")
    def test_call_llm_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"is_correct": true, "score": 10, "max_score": 10, "comment": "正确"}'}}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.grader._call_llm([{"role": "user", "content": "test"}])
        assert result["is_correct"] is True
        assert result["score"] == 10

    @patch("src.modules.llm_grader.requests.post")
    def test_call_llm_json_in_code_block(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {
                    "content": '```json\n{"is_correct": false, "score": 0, "max_score": 5, "comment": "错误"}\n```'}}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.grader._call_llm([{"role": "user", "content": "test"}])
        assert result["is_correct"] is False
        assert result["score"] == 0

    def test_parse_llm_response_valid_json(self):
        result = LLMGrader._parse_llm_response('{"is_correct": true, "score": 8, "max_score": 10, "comment": "不错"}')
        assert result["is_correct"] is True
        assert result["score"] == 8

    def test_parse_llm_response_json_with_text(self):
        result = LLMGrader._parse_llm_response('额外文字 {"is_correct": false, "score": 3} 额外文字')
        assert result["is_correct"] is False
        assert result["score"] == 3

    def test_parse_llm_response_invalid(self):
        result = LLMGrader._parse_llm_response("完全无法解析的内容")
        assert result["is_correct"] is False
        assert result["score"] == 0

    def test_batch_grade(self):
        questions = [
            {"question": "1+1=?", "student_answer": "2", "max_score": 5},
            {"question": "2+2=?", "student_answer": "4", "max_score": 5},
        ]
        with patch.object(self.grader, "grade") as mock_grade:
            mock_grade.side_effect = [
                {"is_correct": True, "score": 5, "max_score": 5, "comment": "正确"},
                {"is_correct": False, "score": 0, "max_score": 5, "comment": "错误"},
            ]
            results = self.grader.batch_grade(questions)
            assert len(results) == 2
            assert results[0]["is_correct"] is True
            assert results[1]["is_correct"] is False


class TestAliyunOCR:

    def test_init(self):
        ocr = AliyunOCR("test-id", "test-secret")
        assert ocr.access_key_id == "test-id"
        assert ocr.access_key_secret == "test-secret"

    def test_get_text_regions_from_questions(self):
        ocr = AliyunOCR("test-id", "test-secret")
        ocr_result = {
            "questions": [
                {
                    "questionInfo": [{"content": "计算 1+1"}],
                    "answerInfo": [{"content": "2"}],
                    "pos": [10, 20, 100, 80],
                }
            ]
        }
        regions = ocr.get_text_regions(ocr_result)
        assert len(regions) == 1
        assert regions[0]["question_text"] == "计算 1+1"
        assert regions[0]["answer_text"] == "2"
        assert regions[0]["index"] == 1

    def test_get_text_regions_from_prism(self):
        ocr = AliyunOCR("test-id", "test-secret")
        ocr_result = {
            "prism_wordsInfo": [
                {"word": "测试文字", "pos": [0, 0, 50, 20]},
            ]
        }
        regions = ocr.get_text_regions(ocr_result)
        assert len(regions) == 1
        assert regions[0]["question_text"] == "测试文字"

    def test_get_text_regions_empty(self):
        ocr = AliyunOCR("test-id", "test-secret")
        regions = ocr.get_text_regions({})
        assert regions == []

    def test_parse_result_from_questions(self):
        ocr = AliyunOCR("test-id", "test-secret")
        ocr_result = {
            "questions": [
                {
                    "questionInfo": [{"content": "计算 1+1"}],
                    "answerInfo": [{"content": "2"}],
                }
            ]
        }
        text = ocr.parse_result(ocr_result)
        assert "[题目] 计算 1+1" in text
        assert "[答案] 2" in text

    def test_parse_result_from_prism(self):
        ocr = AliyunOCR("test-id", "test-secret")
        ocr_result = {
            "prism_wordsInfo": [
                {"word": "测试"},
                {"word": "文字"},
            ]
        }
        text = ocr.parse_result(ocr_result)
        assert "测试 文字" in text

    def test_parse_result_raw_data(self):
        ocr = AliyunOCR("test-id", "test-secret")
        text = ocr.parse_result({"raw_data": "原始数据"})
        assert text == "原始数据"


class TestHelpers:

    def test_setup_logging(self):
        from src.utils.helpers import setup_logging
        logger = setup_logging(log_level="DEBUG")
        assert logger is not None
        assert logger.level == 10

    def test_ensure_dir(self):
        from src.utils.helpers import ensure_dir
        test_dir = os.path.join(os.path.dirname(__file__), "test_temp_dir")
        ensure_dir(test_dir)
        assert os.path.exists(test_dir)
        os.rmdir(test_dir)

    def test_save_and_load_json(self):
        from src.utils.helpers import save_json, load_json
        test_path = os.path.join(os.path.dirname(__file__), "test_temp.json")
        data = {"key": "value", "number": 42}
        save_json(data, test_path)
        loaded = load_json(test_path)
        assert loaded == data
        os.remove(test_path)

    def test_get_supported_formats(self):
        from src.utils.helpers import get_supported_formats
        formats = get_supported_formats()
        assert ".jpg" in formats
        assert ".png" in formats