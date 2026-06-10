"""
阿里云OCR模块
负责调用阿里云视觉智能开放平台进行文字识别。
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import base64

import logging
from alibabacloud_ocr_api20210707.client import Client as OcrClient
from alibabacloud_ocr_api20210707 import models as ocr_models
from alibabacloud_tea_openapi import models as open_api_models

# 获取当前模块的 logger（约定用法：__name__ 自动拿模块名）
logger = logging.getLogger(__name__)


class AliyunOCR:
    """阿里云OCR识别器"""

    def __init__(
            self,
            config_path: Optional[str] = None,
            access_key_id: Optional[str] = None,
            access_key_secret: Optional[str] = None,
            region: Optional[str] = None
    ):
        #先读取config.json 文件获取配置信息
        # 自动往上级目录找,一直找到根目录为止然后返回一个路径对象
        current_file = Path(__file__).resolve()
        # 获取项目根目录 parent能返回上一级目录 3级返回到根目录
        project_root = current_file.parent.parent.parent
        # 拼接成绝对路径 config_path 是一个path对象
        if config_path is None:
            config_path = project_root / "config" / "config.json"
        else:
            config_path = Path(config_path)

        # 判断文件是否存在

        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        aliyun_config = config.get("aliyun", {})
        self.access_key_id = (
            #  第一种是传参时 传入了
            access_key_id
            # 第二种是读取config.json 文件的配置 ,没有就是空
            or aliyun_config.get("access_key_id","")
            # 第三种是环境变量
            or os.environ.get("ALIYUN_ACCESS_KEY_ID")
        )
        self.access_key_secret = (
            access_key_secret
            or aliyun_config.get("access_key_secret","")
            or os.environ.get("ALIYUN_ACCESS_KEY_SECRET")
        )
        # region 指定阿里云 API 请求发到哪个地域的机房，最终影响 AcsClient 连接的 API 域名：
        self.region = (
            region
            or aliyun_config.get("region","cn-hangzhou")
            or os.environ.get("ALIYUN_REGION")
            or "cn-hangzhou"
        )
        if not self.access_key_id or not self.access_key_secret:
            raise ValueError("没有找到阿里云AccessKey,请配置.")

        # 读取超时配置
        connect_timeout = aliyun_config.get("connect_timeout", 10)
        read_timeout = aliyun_config.get("read_timeout", 30)

        # 组装配置包
        sdk_config = open_api_models.Config(
            access_key_id = self.access_key_id,
            access_key_secret = self.access_key_secret,
            connect_timeout = connect_timeout * 1000,
            read_timeout = read_timeout * 1000,
        )
        # 存储连接得服务器
        sdk_config.endpoint = f"ocr-api.{self.region}.aliyuncs.com"

        # 讲配置包传递给客户端
        self.client = OcrClient(sdk_config)

        logger.info(f"阿里云OCR服务初始化成功（region: {self.region}）")

    # 将图片转换为base64编码,便于后续调用api
    def _image_to_base64(self,image_path: str) -> str:
        """将图片文件编码为base64 字符串"""
        # 图片 / 视频 / 音频都是二进制文件，必须用 rb 打开，用普通文本模式 r 会直接报错
        with open(image_path, "rb") as f:
            # 先read变成全部2进制数据 在b64encode编码 转换成Base64编码的字节串(bytes类型)
            # decode("utf-8") 转换成字符串
            return base64.b64encode(f.read()).decode("utf-8")


    def recognize_text(self, image_path: str) -> Dict:
        """调用ocr api 识别图像中的文字内容 返回原始结果字典"""
        # 读取图片 rb为读取为2进制bytes类型
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        # RecognizeGeneralRequest通用文字识别请求类
        request = ocr_models.RecognizeGeneralRequest()
        request.body = image_bytes
        # 发送请求等待响应
        response = self.client.recognize_general(request)
        # 把响应数据转成Python字典
        result = response.body.to_map()

        logger.info("通用文字识别完成")
        return result

    def recognize_edu_paper_cut(self, image_path: str) -> Dict:
        """试卷切题：将整张试卷切分为单道题目区域"""
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        request = ocr_models.RecognizeEduPaperCutRequest()
        request.body = image_bytes
        request.cut_type = "question"
        request.image_type = "scan"

        response = self.client.recognize_edu_paper_cut(request)
        result = response.body.to_map()

        logger.info("试卷切题完成")
        return result

    def get_question_regions(self, paper_cut_result: Dict) -> List[Dict]:
        """从切题结果中提取每道题的区域坐标和题号"""
        data_str = paper_cut_result.get("Data", "{}")
        data = json.loads(data_str)

        regions = []
        for page in data.get("page_list", []):
            for subject in page.get("subject_list", []):
                ids = subject.get("ids", [])
                question_no = ids[0] if ids else 0
                content_infos = subject.get("content_list_info", [])
                if content_infos:
                    pos = content_infos[0].get("pos", [])
                    if pos:
                        x1 = pos[0].get("x", 0)
                        y1 = pos[0].get("y", 0)
                        x2 = pos[2].get("x", 0)
                        y2 = pos[2].get("y", 0)
                    else:
                        x1 = y1 = x2 = y2 = 0
                else:
                    x1 = y1 = x2 = y2 = 0

                regions.append({
                    "question_no": question_no,
                    "text": subject.get("text", ""),
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                })
        return regions

    def recognize_edu_question_ocr(self, image_path: str) -> Dict:
        """题目识别：对单道题目进行精细OCR"""
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        request = ocr_models.RecognizeEduQuestionOcrRequest()
        request.body = image_bytes

        response = self.client.recognize_edu_question_ocr(request)
        result = response.body.to_map()

        logger.info("题目识别完成")
        return result

    def recognize_handwriting(self, image_path: str) -> Dict:
        """识别手写文字"""
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        request = ocr_models.RecognizeHandwritingRequest()
        request.body = image_bytes

        response = self.client.recognize_handwriting(request)
        result = response.body.to_map()

        logger.info("手写文字识别完成")
        return result

    def recognize_formula(self, image_path: str) -> Dict:
        """识别公式内容"""
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        request = ocr_models.RecognizeFormulaRequest()
        request.body = image_bytes

        response = self.client.recognize_formula(request)
        result = response.body.to_map()

        logger.info("公式识别完成")
        return result

    def get_text_regions(self, ocr_result: Dict) -> List[Dict]:
        """从OCR结果中提取文字区域信息
        data把它从字符串转为字典后的keys有
        content	str	所有识别到的文字拼成的完整段落	"有一项是符合题目要求的。
        prism_wordsInfo	list[dict]	每一个文字块的详细信息（坐标+文字）
        height	int	算法矫正后的图片高度（像素）
        width	int	算法矫正后的图片宽度（像素）
        orgHeight	int	原始图片高度
        orgWidth	int	原始图片宽度
        prism_version	str	识别引擎版本号
        """
        data_str = ocr_result.get("Data","{}")
        data = json.loads(data_str)

        regions = []
        for word_info in data.get("prism_wordsInfo",[]):
            regions.append({
                "x": word_info.get("x",0),
                "y": word_info.get("y",0),
                "width": word_info.get("width",0),
                "height": word_info.get("height",0),
                "text": word_info.get("word","")
            })
        return regions


    def parse_result(self, ocr_result: Dict) -> str:
        """将OCR结果解析为纯文本"""
        data_str = ocr_result.get("Data","{}")
        data = json.loads(data_str)
        return data.get("content","")


if __name__ == '__main__':
    # 临时打开 DEBUG 级别，调试时看见更多信息
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s | %(name)s | %(message)s"
    )

    ocr = AliyunOCR()