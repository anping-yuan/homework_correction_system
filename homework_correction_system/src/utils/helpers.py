"""
工具函数模块
提供文件操作、日志记录、图像格式转换等通用工具。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger("homework_correction")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """加载JSON配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(dir_path: str) -> None:
    """确保目录存在，不存在则创建"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """保存数据为JSON文件"""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path: str) -> Dict[str, Any]:
    """从JSON文件加载数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_supported_formats() -> List[str]:
    """获取支持的图像格式列表"""
    return [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]