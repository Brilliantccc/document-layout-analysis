"""
预处理脚本：把所有图片 resize 到 640×640 存到新目录

⚠ 已不再需要（保留备用）。

RT-DETR 走 RTDetrImageProcessor、Faster R-CNN 走 torchvision 的
GeneralizedRCNNTransform，两者都会在读图后自行 resize + 归一化，
config 的路径常量也已指向原始 1025×1025 数据。所以：
  - 训练不再依赖 data/preprocessed_640/
  - 而且 RTDetrImageProcessor 会无条件把图缩到 IMG_SIZE，
    预先缩到 640 只影响插值质量、不会产生尺度变化，纯属多一次有损重采样

如果以后想固定某个分辨率做快速实验，可以用这个脚本预生成一份缓存，
但记得同步改 config 的 TRAIN_ANN / TRAIN_IMG_DIR 等路径常量。
"""
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from config import (
    TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR,
    TRAIN_ANN, VAL_ANN, TEST_ANN,
    DATA_DIR,
)
from utils import imread, imwrite


def preprocess_split(img_dir, ann_path, out_dir, target_size=640):
    """预处理一个 split：resize 图片 + 调整 bbox"""
    os.makedirs(out_dir, exist_ok=True)

    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    new_images = []

    for img_info in tqdm(coco['images'], desc=f"处理 {out_dir}"):
        img_id = img_info['id']
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = imread(img_path)
        if image is None:
            print(f"  跳过: {img_path}")
            continue

        orig_h, orig_w = image.shape[:2]
        new_h, new_w = target_size, target_size

        # resize
        if orig_h != new_h or orig_w != new_w:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 保存（用新文件名，避免覆盖原图）
        new_filename = img_info['file_name']
        out_path = os.path.join(out_dir, new_filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        imwrite(out_path, image)

        # 更新图片信息
        new_img_info = img_info.copy()
        new_img_info['width'] = new_w
        new_img_info['height'] = new_h
        new_images.append(new_img_info)

    # 更新 COCO JSON（调整 bbox）
    new_anns = []
    img_id_map = {img['id']: img for img in new_images}

    for ann in tqdm(coco['annotations'], desc="调整标注"):
        img_id = ann['image_id']
        if img_id not in img_id_map:
            continue

        orig_img = images[img_id]
        new_img = img_id_map[img_id]
        scale_x = new_img['width'] / orig_img['width']
        scale_y = new_img['height'] / orig_img['height']

        new_ann = ann.copy()
        x, y, w, h = ann['bbox']
        new_ann['bbox'] = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]
        new_ann['area'] = (w * scale_x) * (h * scale_y)
        new_anns.append(new_ann)

    new_coco = {
        'images': new_images,
        'annotations': new_anns,
        'categories': coco['categories'],
    }

    out_ann = os.path.join(out_dir, '..', os.path.basename(ann_path))
    os.makedirs(os.path.dirname(out_ann), exist_ok=True)
    with open(out_ann, 'w', encoding='utf-8') as f:
        json.dump(new_coco, f, ensure_ascii=False)

    print(f"  完成: {len(new_images)} 张图片, {len(new_anns)} 个标注")
    print(f"  图片: {out_dir}")
    print(f"  标注: {out_ann}")


if __name__ == "__main__":
    # 固定随机种子
    import random
    random.seed(42)
    np.random.seed(42)

    out_base = os.path.join(DATA_DIR, "preprocessed_640")

    print("=" * 60)
    print("预处理 DocLayNet → 640×640")
    print("=" * 60)

    preprocess_split(TRAIN_IMG_DIR, TRAIN_ANN,
                     os.path.join(out_base, "train", "images"),
                     target_size=640)

    preprocess_split(VAL_IMG_DIR, VAL_ANN,
                     os.path.join(out_base, "val", "images"),
                     target_size=640)

    preprocess_split(TEST_IMG_DIR, TEST_ANN,
                     os.path.join(out_base, "test", "images"),
                     target_size=640)

    print("\n完成！在 config.py 中切换路径：")
    print(f'  TRAIN_IMG_DIR = os.path.join(DATA_DIR, "preprocessed_640", "train", "images")')
    print(f'  TRAIN_ANN = os.path.join(DATA_DIR, "preprocessed_640", "train.json")')
    print("  （val/test 同理）")