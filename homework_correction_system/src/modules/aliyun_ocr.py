import os
import json
import base64
import time
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("homework_correction.aliyun_ocr")

from alibabacloud_ocr_api20210707.client import Client as OCRClient
from alibabacloud_ocr_api20210707.models import (
    RecognizeEduPaperCutRequest,
    RecognizeEduQuestionOcrRequest,
    RecognizeGeneralRequest,
    RecognizeHandwritingRequest,
)
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions

_SDK_AVAILABLE = True

def _encode_image_base64(image_data: bytes) -> str:
    return base64.b64encode(image_data).decode("utf-8")


def _load_and_encode(image_path: str) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()


class AliyunOCR:
    def __init__(
            self,
            access_key_id: str,
            access_key_secret: str,
            endpoint: str = "ocr-api.cn-hangzhou.aliyuncs.com",
            connect_timeout: int = 10,
            read_timeout: int = 30,
    ):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = endpoint
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._client: Optional[OCRClient] = None

    @property
    def client(self) -> OCRClient:
        if not _SDK_AVAILABLE:
            raise RuntimeError("阿里云OCR SDK未安装，请运行: pip install alibabacloud_ocr_api20210707")
        if self._client is None:
            config = Config(
                access_key_id=self.access_key_id,
                access_key_secret=self.access_key_secret,
                endpoint=self.endpoint,
                connect_timeout=self.connect_timeout * 1000,
                read_timeout=self.read_timeout * 1000,
            )
            self._client = OCRClient(config)
        return self._client

    def _call_ocr_api(self, request_obj, retry: int = 3) -> Dict[str, Any]:
        runtime = RuntimeOptions()
        last_error = None
        for attempt in range(retry):
            try:
                response = self.client.recognize_edu_paper_cut(
                    request_obj
                )
                return self._parse_response(response)
            except Exception as e:
                last_error = e
                logger.warning(f"OCR API 调用失败 (尝试 {attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    time.sleep(0.5 * (2 ** attempt))
        raise last_error

    def _call_question_ocr_api(self, request_obj, retry: int = 3) -> Dict[str, Any]:
        runtime = RuntimeOptions()
        last_error = None
        for attempt in range(retry):
            try:
                response = self.client.recognize_edu_question_ocr_with_options(
                    request_obj, runtime
                )
                return self._parse_response(response)
            except Exception as e:
                last_error = e
                logger.warning(f"题目OCR API 调用失败 (尝试 {attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    time.sleep(0.5 * (2 ** attempt))
        raise last_error

    def _call_general_ocr_api(self, request_obj, retry: int = 3) -> Dict[str, Any]:
        runtime = RuntimeOptions()
        last_error = None
        for attempt in range(retry):
            try:
                response = self.client.recognize_general_with_options(
                    request_obj, runtime
                )
                return self._parse_response(response)
            except Exception as e:
                last_error = e
                logger.warning(f"通用OCR API 调用失败 (尝试 {attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    time.sleep(0.5 * (2 ** attempt))
        raise last_error

    def _call_handwriting_api(self, request_obj, retry: int = 3) -> Dict[str, Any]:
        runtime = RuntimeOptions()
        last_error = None
        for attempt in range(retry):
            try:
                response = self.client.recognize_handwriting_with_options(
                    request_obj, runtime
                )
                return self._parse_response(response)
            except Exception as e:
                last_error = e
                logger.warning(f"手写OCR API 调用失败 (尝试 {attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    time.sleep(0.5 * (2 ** attempt))
        raise last_error

    @staticmethod
    def _parse_response(response) -> Dict[str, Any]:
        body = response.body
        if hasattr(body, "to_map"):
            result = body.to_map()
        else:
            result = dict(body)
        data_str = result.get("Data", "{}")
        if isinstance(data_str, str):
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return {"raw_data": data_str}
        return data_str or {}

    def recognize_text(self, image_path: str) -> Dict:
        image_base64 = _load_and_encode(image_path)
        request = RecognizeGeneralRequest(body=image_base64)
        return self._call_general_ocr_api(request)

    def recognize_handwriting(self, image_path: str) -> Dict:
        image_base64 = _load_and_encode(image_path)
        request = RecognizeHandwritingRequest(body=image_base64)
        return self._call_handwriting_api(request)

    def recognize_formula(self, image_path: str) -> Dict:
        image_base64 = _load_and_encode(image_path)
        request = RecognizeEduQuestionOcrRequest(
            body=image_base64,
            need_rotate=False,
            need_sort_page=True,
        )
        return self._call_question_ocr_api(request)

    def paper_cut(self, image_path: str, subject: str = "general") -> Dict:
        image_base64 = _load_and_encode(image_path)  # 现在是 bytes
        request = RecognizeEduPaperCutRequest(
            body=image_base64,
            cut_type="question",
        )
        request.image_type = "scan"
        response = self._call_ocr_api(request)
        return response

    def question_ocr(
            self, image_path: str, need_rotate: bool = False, need_sort_page: bool = True
    ) -> Dict:
        image_base64 = _load_and_encode(image_path)
        request = RecognizeEduQuestionOcrRequest(
            body=image_base64,
            need_rotate=need_rotate,
            need_sort_page=need_sort_page,
        )
        return self._call_question_ocr_api(request)

    def paper_cut_from_bytes(self, image_bytes: bytes, subject: str = "general") -> Dict:
        image_base64 = _encode_image_base64(image_bytes)
        request = RecognizeEduPaperCutRequest(
            body=image_base64,
            cut_type="question",  # ✅ 修复：固定合法值
        )
        request.image_type = "scan"  # ✅ 修复：必填参数
        response = self._call_ocr_api(request)
        return response

    def question_ocr_from_bytes(
            self, image_bytes: bytes, need_rotate: bool = False, need_sort_page: bool = True
    ) -> Dict:
        image_base64 = _encode_image_base64(image_bytes)
        request = RecognizeEduQuestionOcrRequest(
            body=image_base64,
            need_rotate=need_rotate,
            need_sort_page=need_sort_page,
        )
        return self._call_question_ocr_api(request)

    def get_text_regions(self, ocr_result: Dict) -> List[Dict]:
        regions = []
        questions = ocr_result.get("questions", [])
        for idx, q in enumerate(questions):
            question_info = q.get("questionInfo", [])
            question_text = " ".join(
                item.get("content", "") for item in question_info
            )
            answer_info = q.get("answerInfo", [])
            answer_text = " ".join(
                item.get("content", "") for item in answer_info
            )
            position = q.get("pos", [])
            region = {
                "index": idx + 1,
                "question_text": question_text,
                "answer_text": answer_text,
                "position": position,
                "raw_info": q,
            }
            regions.append(region)

        if not regions:
            prisms = ocr_result.get("prism_wordsInfo", []) or ocr_result.get("prism_wordsInfo", [])
            for idx, word_info in enumerate(prisms):
                pos = word_info.get("pos", [])
                region = {
                    "index": idx + 1,
                    "question_text": word_info.get("word", ""),
                    "answer_text": "",
                    "position": pos,
                    "raw_info": word_info,
                }
                regions.append(region)

        return regions

    def parse_result(self, ocr_result: Dict) -> str:
        questions = ocr_result.get("questions", [])
        if questions:
            lines = []
            for q in questions:
                question_info = q.get("questionInfo", [])
                answer_info = q.get("answerInfo", [])
                q_text = " ".join(item.get("content", "") for item in question_info)
                a_text = " ".join(item.get("content", "") for item in answer_info)
                lines.append(f"[题目] {q_text}")
                if a_text:
                    lines.append(f"[答案] {a_text}")
            return "\n".join(lines)

        words = ocr_result.get("prism_wordsInfo", []) or ocr_result.get("prism_wordsInfo", [])
        if words:
            return " ".join(w.get("word", "") for w in words)

        raw = ocr_result.get("raw_data", "")
        return raw if isinstance(raw, str) else json.dumps(ocr_result, ensure_ascii=False)

    def process_paper_pipeline(
            self,
            image_path: str,
            subject: str = "general",
            max_workers: int = 5,
    ) -> List[Dict]:
        logger.info(f"执行试卷切题, 学科: {subject}")
        cut_result = self.paper_cut(image_path, subject=subject)
        regions = self.get_text_regions(cut_result)
        logger.info(f"切题完成, 共识别出 {len(regions)} 个题目区域")

        logger.info(f"使用 {max_workers} 个线程并发执行题目OCR")
        processed = [None] * len(regions)

        def ocr_task(idx: int, region: Dict):
            position = region.get("position", [])
            if position and len(position) == 4:
                import cv2
                import numpy as np
                img = cv2.imread(image_path)
                if img is not None:
                    x1, y1, x2, y2 = position
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(img.shape[1], int(x2))
                    y2 = min(img.shape[0], int(y2))
                    if x2 > x1 and y2 > y1:
                        cropped = img[y1:y2, x1:x2]
                        temp_path = f"{image_path}_region_{idx}.jpg"
                        cv2.imwrite(temp_path, cropped)
                        try:
                            ocr_data = self.question_ocr(temp_path)
                            region["ocr_result"] = ocr_data
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        processed[idx] = region
                        return
            try:
                ocr_data = self.question_ocr(image_path)
                region["ocr_result"] = ocr_data
            except Exception as e:
                logger.warning(f"题目 {idx + 1} OCR 失败: {e}")
                region["ocr_result"] = {}
            processed[idx] = region

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(ocr_task, i, regions[i]): i
                for i in range(len(regions))
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"题目 {idx + 1} 处理异常: {e}")

        final_regions = []
        for i, r in enumerate(processed):
            if r is not None:
                final_regions.append(r)
            elif i < len(regions):
                regions[i]["ocr_result"] = {}
                final_regions.append(regions[i])

        logger.info(f"题目OCR处理完成, 共 {len(final_regions)} 题")
        return final_regions