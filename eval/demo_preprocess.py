"""
图片预处理效果演示 — 独立运行，可视化对比每一步效果

用法: python eval/demo_preprocess.py [图片路径]
     不传路径则自动找 data/output 下第一张原始图片
"""
import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.modules.image_processor import ImageProcessor

# ── 配置：使用与 VL 预处理相同的参数 ──
CONFIG = {
    "denoise": True,       "denoise_strength": 10,
    "deskew": True,
    "enhance_contrast": True, "contrast_clip_limit": 2.0,
    "sharpen": True,
    "binarize": False,
}


def add_label(img: np.ndarray, text: str, font_scale: float = 1.2) -> np.ndarray:
    """在图片顶部添加标签条"""
    h, w = img.shape[:2]
    bar_h = 40
    labeled = np.ones((h + bar_h, w, 3), dtype=np.uint8) * 240
    labeled[bar_h:, :] = img
    cv2.putText(labeled, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    return labeled


def main(image_path: str = None):
    processor = ImageProcessor(config=CONFIG)

    # 找图片
    if image_path is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "output"
        originals = sorted(data_dir.glob("*_original.jpg"))
        if not originals:
            originals = sorted(data_dir.glob("*.jpg"))[:1]
        if not originals:
            input_img = Path(__file__).resolve().parent.parent / "data" / "input" / "img.png"
            if input_img.exists():
                originals = [input_img]
        if not originals:
            print("[WARN] 没有找到测试图片，请手动指定路径")
            return
        image_path = str(originals[0])

    print(f"[测试图片] {image_path}")
    print(f"   文件大小: {os.path.getsize(image_path)/1024:.0f} KB")

    # 读取原图
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print("[ERROR] 无法读取图片")
        return

    h, w = img.shape[:2]
    print(f"   尺寸: {w}x{h}")

    # ── 预处理增强管线（4步）──
    steps = []  # [(label, image)]

    # Step 0: 原图
    steps.append((f"① 原图 ({w}x{h})", img.copy()))

    # Step 1: 光照归一化（去阴影）
    illum = processor.normalize_illumination(img.copy())
    steps.append(("② 光照归一化 (去阴影)", illum))
    img = illum

    # Step 2: 双边滤波（保边去噪）
    bilateral = processor.bilateral_denoise(img.copy(), d=7, sigma_color=25, sigma_space=8)
    steps.append(("③ 双边滤波 (保边去噪)", bilateral))
    img = bilateral

    # Step 3: CLAHE + USM
    clahe = processor.enhance_contrast(img.copy(), clip_limit=1.5, tile_grid_size=(8, 8))
    usm = processor.unsharp_mask(clahe, sigma=1.0, amount=0.8, threshold=3)
    steps.append(("④ CLAHE + USM锐化", usm))

    # ── 拼接对比图：2 行 x 2 列 ──
    thumb_h = 500
    thumbnails = []
    for label, step_img in steps:
        sh, sw = step_img.shape[:2]
        ratio = thumb_h / sh
        tw, th = int(sw * ratio), thumb_h
        thumb = cv2.resize(step_img, (tw, th))
        thumbnails.append(add_label(thumb, label, font_scale=0.7))

    row1 = np.hstack(thumbnails[:2])
    row2 = np.hstack(thumbnails[2:4])
    max_w = max(row1.shape[1], row2.shape[1])
    for row in [row1, row2]:
        if row.shape[1] < max_w:
            pad = np.ones((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8) * 240
            row = np.hstack([row, pad])
    comparison = np.vstack([row1, row2])

    # 顶部大标题
    title_h = 50
    title_bar = np.ones((title_h, comparison.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_bar, f"图片预处理流水线: 光照归一化→双边滤波→CLAHE→USM锐化 | {os.path.basename(image_path)}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2, cv2.LINE_AA)
    final = np.vstack([title_bar, comparison])

    # 保存
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "preprocess_demo.png"
    cv2.imwrite(str(out_path), final)
    print(f"\n[OK] 对比图已保存: {out_path}")
    print(f"   尺寸: {final.shape[1]}x{final.shape[0]}")

    return str(out_path)


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    result = main(img_path)
    if result:
        print("\n查看方式: 直接打开 eval/preprocess_demo.png")
