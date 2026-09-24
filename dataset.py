"""
DocLayNet 数据集加载

两条独立的数据通路：
  - FRCNN   : DocLayNetDataset  → 输出 [0,1] 的 CHW tensor + 项目格式 GT，
              resize/归一化交给 torchvision 的 GeneralizedRCNNTransform
  - RT-DETR : RTDetrDataset     → 在 __getitem__ 里用 RTDetrImageProcessor 做
              resize/归一化（12 个 worker 并行），同时输出项目格式 GT

项目格式 GT 的约定（全项目唯一真相）：
  boxes  = xyxy 绝对像素 @原始 1025×1025
  labels = 1-based（1~11），与 COCO category_id、LAYOUT_CLASSES 一致
HF 需要的（归一化 cxcywh + 0-based class_labels）只在 model.RTDetrLayoutDetector 里换算。
"""
import os
import json
import math
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    TRAIN_ANN, VAL_ANN, TEST_ANN,
    TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR,
    BATCH_SIZE, NUM_WORKERS, PERSISTENT_WORKERS, PREFETCH_FACTOR,
    RTDETR_IMG_SIZE,
)
from utils import imread

# 注意：GeneralizedRCNNTransform 内部已做 ImageNet 归一化，FRCNN 的 dataset 只需输出 [0,1]


# ==================== 数据增强 ====================

class LayoutAugmentation:
    """文档版面数据增强（纯 OpenCV，bbox 同步变换）—— FRCNN 用"""

    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, image, target):
        if not self.is_train:
            return image, target

        boxes = target['boxes'].clone()
        labels = target['labels'].clone()

        # 空标注：确保 shape 为 [0, 4]
        if len(boxes) == 0:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            return image, target

        img_h, img_w = image.shape[:2]

        # 随机缩放（多尺度训练）：短边随机到 [480, 800]
        if random.random() < 0.8:
            target_size = random.randint(480, 800)
            scale = target_size / min(img_h, img_w)
            if scale != 1.0:
                new_h, new_w = int(img_h * scale), int(img_w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                boxes[:, 0] *= scale
                boxes[:, 1] *= scale
                boxes[:, 2] *= scale
                boxes[:, 3] *= scale
                img_h, img_w = new_h, new_w

        # 随机水平翻转
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            x1, x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
            boxes[:, 0] = img_w - x2
            boxes[:, 2] = img_w - x1

        # 颜色抖动
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

        if random.random() < 0.3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(0.8, 1.2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 裁掉越界框
        boxes[:, 0].clamp_(0, img_w)
        boxes[:, 1].clamp_(0, img_h)
        boxes[:, 2].clamp_(0, img_w)
        boxes[:, 3].clamp_(0, img_h)

        # 过滤无效框
        valid = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
        target['boxes'] = boxes[valid]
        target['labels'] = labels[valid]

        return image, target


class RTDetrLayoutAugmentation:
    """RT-DETR 的数据增强 —— 在原始 1025 空间的 numpy RGB + xyxy 上做。

    必须自己做：RTDetrImageProcessor 只有 resize/rescale/normalize，没有任何随机增强。

    随机裁剪默认关闭：DocLayNet 的 Text 占 47% 的框，裁剪会切断文本行产生标签噪声。
    只在确认过拟合（train loss 远低于 val）时通过 --rand_crop 打开。
    """

    def __init__(self, is_train=True, hflip_p=0.5, color_p=0.5, rand_crop_p=0.0):
        self.is_train = is_train
        self.hflip_p = hflip_p
        self.color_p = color_p
        self.rand_crop_p = rand_crop_p

    def __call__(self, image, boxes, labels):
        """image: (H,W,3) uint8 RGB; boxes: (N,4) float32 xyxy; labels: (N,) int64"""
        if not self.is_train or len(boxes) == 0:
            return image, boxes, labels

        H, W = image.shape[:2]

        # 水平翻转 —— 版面分析里唯一"无损"的几何增强
        if random.random() < self.hflip_p:
            image = np.ascontiguousarray(image[:, ::-1])
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = W - x2
            boxes[:, 2] = W - x1

        # 颜色抖动 —— 模拟扫描/拍照的亮度、对比度、色偏
        if random.random() < self.color_p:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(0.75, 1.25)   # 饱和度
            hsv[:, :, 2] *= random.uniform(0.80, 1.20)   # 明度
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # random resized crop —— 真正的多尺度（默认关闭）
        if self.rand_crop_p > 0 and random.random() < self.rand_crop_p:
            image, boxes, labels = self._rand_resized_crop(image, boxes, labels)

        return image, boxes, labels

    @staticmethod
    def _rand_resized_crop(image, boxes, labels):
        """面积 [0.6,1.0]、长宽比 [0.8,1.25] 的随机裁剪。
        保留条件：框裁剪后剩余面积 >= 原面积 30%；否则丢弃（避免半截框污染 GT）。
        """
        H, W = image.shape[:2]
        for _ in range(10):
            area = H * W * random.uniform(0.6, 1.0)
            ar = math.exp(random.uniform(math.log(0.8), math.log(1.25)))
            cw = int(round(math.sqrt(area * ar)))
            ch = int(round(math.sqrt(area / ar)))
            if cw > W or ch > H or cw < 2 or ch < 2:
                continue
            x0 = random.randint(0, W - cw)
            y0 = random.randint(0, H - ch)
            x1, y1 = x0 + cw, y0 + ch

            ix1 = np.maximum(boxes[:, 0], x0)
            iy1 = np.maximum(boxes[:, 1], y0)
            ix2 = np.minimum(boxes[:, 2], x1)
            iy2 = np.minimum(boxes[:, 3], y1)
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            orig = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = inter >= 0.3 * np.maximum(orig, 1e-6)
            if keep.sum() == 0:
                continue

            crop = image[y0:y1, x0:x1]
            b = boxes[keep].copy()
            b[:, [0, 2]] -= x0
            b[:, [1, 3]] -= y0
            return np.ascontiguousarray(crop), b, labels[keep]
        return image, boxes, labels


# ==================== 数据集 ====================

def _load_coco(json_path):
    """读 COCO JSON，返回 (images dict, img_to_anns dict, sorted img_ids, 标注数)

    注意 img_to_anns 只包含【有标注的】图片（defaultdict 按需建 key）。
    要拿"所有图片的标注"，必须遍历 img_ids 再用 .get()，见 _build_gt_index。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = {img['id']: img for img in coco['images']}
    img_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    n_ann = len(coco['annotations'])
    return images, img_to_anns, sorted(images.keys()), n_ann


def _build_gt_index(img_to_anns, img_ids):
    """把标注摊平成 numpy，key 必须覆盖【所有】图片，并【去掉完全重复的框】。

    两件事都是有原因的，不要"简化"掉：

    1) 遍历 img_ids 而不是 img_to_anns.items()：DocLayNet train 里有 27 张零标注图，
       它们不在 img_to_anns 里。漏掉会让 __getitem__ 在 self.gt[img_id] 上 KeyError，
       而且因为 DataLoader 带 shuffle，崩会发生在第 1 个 epoch 中途，极难定位。

    2) 按 (x,y,w,h,category) 去重：pierreguillou/DocLayNet-base 这份转换里标注被
       大量复制 —— train 的 681,481 个标注去重后只剩 91,136 个（86.6% 是重复），
       其中 Table 98.0%、Formula 96.7%、Picture 94.4%、Text 86.6% 是重复，而
       Page-footer 只有 24.1%。
       不去重的后果是致命的：同一个位置堆着几百个完全相同的 GT，模型在那里给
       1 个框只能匹配 1 个，其余全算漏检，该类 AP 被压到接近 0 —— 实测 FRCNN 和
       RT-DETR 的 Table AP 分别只有 0.01 / 0.34，而重复最少的 Page-footer 高达
       0.69 / 0.41。两个架构给出同样的天花板和排序，正是因为这是 GT 的属性。

    返回 {img_id: (boxes[N,4] float32 xyxy, labels[N] int64 1-based)}
    """
    empty = (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))
    gt = {}
    n_dup = 0
    for img_id in img_ids:
        anns = img_to_anns.get(img_id) or []
        if not anns:
            gt[img_id] = empty
            continue
        arr = np.array([a['bbox'] for a in anns], dtype=np.float32)
        cats = np.array([a['category_id'] for a in anns], dtype=np.int64)
        valid = (arr[:, 2] > 0) & (arr[:, 3] > 0)     # 过滤 w/h <= 0 的无效框
        arr, cats = arr[valid], cats[valid]

        # 去重：同一张图上 (x,y,w,h,类别) 完全相同的只保留一份。
        # 按 (bbox, 类别) 去重而不是只按 bbox，避免把不同类别的同位置框合并掉。
        if len(arr):
            stacked = np.concatenate([arr, cats[:, None]], axis=1)
            keep = np.unique(stacked, axis=0, return_index=True)[1]
            keep.sort()                                # 保持原始顺序，便于复现
            n_dup += len(arr) - len(keep)
            arr, cats = arr[keep], cats[keep]

        if len(arr):
            boxes = np.stack([arr[:, 0], arr[:, 1], arr[:, 0] + arr[:, 2],
                              arr[:, 1] + arr[:, 3]], axis=1)
        else:
            boxes = np.zeros((0, 4), np.float32)
        gt[img_id] = (boxes, cats)

    if n_dup:
        print(f"  [去重] 丢弃 {n_dup} 个完全重复的标注框"
              f"（pierreguillou/DocLayNet-base 的转换把标注复制了多份，不去重会摧毁 AP）")
    return gt


class DocLayNetDataset(Dataset):
    """DocLayNet COCO 格式数据集（Faster R-CNN 用）

    输出 image: [3,H,W] float32 [0,1]（归一化交给 GeneralizedRCNNTransform）
         target: {'boxes' xyxy 像素, 'labels' 1-based, 'image_id'}
    """

    def __init__(self, json_path, img_dir, is_train=True, subset=None):
        self.img_dir = img_dir
        self.is_train = is_train
        self.aug = LayoutAugmentation(is_train)

        self.images, self.img_to_anns, self.img_ids, n_ann = _load_coco(json_path)
        if subset:
            self.img_ids = self.img_ids[:subset]
        n_empty = sum(1 for i in self.img_ids if not self.img_to_anns.get(i))
        print(f"  加载 {len(self.img_ids)} 张图片, {n_ann} 个标注"
              + (f", {n_empty} 张无标注" if n_empty else ""))

        # 预先把标注摊平成 numpy，省掉每 epoch 的 Python 循环
        self.gt = _build_gt_index(self.img_to_anns, self.img_ids)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = imread(img_path)
        if image is None:
            raise RuntimeError(f"无法读取: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        b, l = self.gt[img_id]
        target = {
            'boxes': torch.as_tensor(b.copy(), dtype=torch.float32).reshape(-1, 4),
            'labels': torch.as_tensor(l.copy(), dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
        }

        if self.is_train:
            image, target = self.aug(image, target)

        image = torch.tensor(np.ascontiguousarray(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image, target


class RTDetrDataset(Dataset):
    """RT-DETR 数据集

    processor 放在 __getitem__ 里跑，让 resize + ImageNet 归一化在 DataLoader 的
    多个 worker 中并行完成（放 collate_fn 会在主进程串行执行，直接阻塞 GPU）。

    输出:
        pixel_values : [3,S,S] float32，已归一化（processor 产物）
        boxes        : [N,4] float32 xyxy 像素 @原始分辨率
        labels       : [N] int64 1-based
        image_id     : [1] int64
        orig_size    : [2] int64 (H, W)，eval 时把预测框映射回原图要用
    """

    def __init__(self, json_path, img_dir, is_train=True, processor=None,
                 img_size=None, subset=None, aug_kwargs=None):
        assert processor is not None, "RTDetrDataset 必须传入 processor（构造一次后复用）"
        self.img_dir = img_dir
        self.is_train = is_train
        self.processor = processor
        self.img_size = img_size or RTDETR_IMG_SIZE
        self.aug = RTDetrLayoutAugmentation(is_train, **(aug_kwargs or {}))

        self.images, self.img_to_anns, self.img_ids, n_ann = _load_coco(json_path)
        if subset:
            self.img_ids = self.img_ids[:subset]
        n_empty = sum(1 for i in self.img_ids if not self.img_to_anns.get(i))
        print(f"  加载 {len(self.img_ids)} 张图片, {n_ann} 个标注 (img_size={self.img_size})"
              + (f", {n_empty} 张无标注" if n_empty else ""))

        # 预先把标注摊平成 numpy：否则每张图每 epoch 都要对 ~100 个标注做 Python 循环
        self.gt = _build_gt_index(self.img_to_anns, self.img_ids)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        from PIL import Image

        img_id = self.img_ids[idx]
        img_info = self.images[img_id]

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        bgr = imread(img_path)
        if bgr is None:
            raise RuntimeError(f"无法读取: {img_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = image.shape[:2]

        b, l = self.gt[img_id]
        boxes = b.copy()
        labels = l.copy()

        if self.is_train:
            image, boxes, labels = self.aug(image, boxes, labels)

        # 只把图像交给 processor。标注不走 processor —— 项目格式 GT 是唯一真相，
        # HF 的 0-based + 归一化 cxcywh 换算全部封在 model.RTDetrLayoutDetector 里。
        # 这样 labels 的 1-based/0-based 转换和坐标空间转换都只在一个地方发生。
        # 注意 processor 要的是 uint8 0-255，不是除过 255 的 float。
        out = self.processor(
            images=Image.fromarray(np.ascontiguousarray(image)),
            return_tensors='pt',
            size={'height': self.img_size, 'width': self.img_size},
        )

        return {
            'pixel_values': out['pixel_values'][0],
            'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([H0, W0]),
        }


def collate_fn(batch):
    """Faster R-CNN 的自定义 collate（需要图片列表）"""
    return tuple(zip(*batch))


def rtdetr_collate_fn(batch):
    """RT-DETR 的 collate。固定 resize 到 S×S，batch 内尺寸一致，直接 stack，无需 padding。"""
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch], dim=0),
        'targets': [{k: b[k] for k in ('boxes', 'labels', 'image_id', 'orig_size')}
                    for b in batch],
    }


# ==================== DataLoader 工厂 ====================

def build_rtdetr_processor(model_name_or_path, img_size):
    """构造 RTDetrImageProcessor 并固定输出尺寸。

    必须在 DataLoader 的 worker fork 之前构造一次，然后传给 Dataset 复用；
    不要在 worker 里各建一份。processor 是普通 dataclass，可 pickle。
    """
    from transformers import RTDetrImageProcessor
    from utils import hf_local_first
    # hf_local_first：连不上 HF 时自动用本地缓存（AutoDL 上 huggingface.co 常不可达）
    processor = hf_local_first(RTDetrImageProcessor.from_pretrained, model_name_or_path)
    processor.size = {'height': img_size, 'width': img_size}
    processor.do_pad = False        # 尺寸已统一，不需要 pixel_mask
    return processor


def _make_loader(dataset, batch_size, shuffle, detector, drop_last=False, num_workers=None):
    nw = NUM_WORKERS if num_workers is None else num_workers
    kw = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        collate_fn=rtdetr_collate_fn if detector == 'rtdetr' else collate_fn,
        pin_memory=True,
        drop_last=drop_last,
    )
    # num_workers=0 时不能设 persistent_workers / prefetch_factor，DataLoader 会报错
    if nw > 0:
        kw['persistent_workers'] = PERSISTENT_WORKERS
        kw['prefetch_factor'] = PREFETCH_FACTOR
    return DataLoader(dataset, **kw)


def _make_dataset(split, is_train, detector, processor, img_size, subset, aug_kwargs=None):
    ann, img_dir = {
        'train': (TRAIN_ANN, TRAIN_IMG_DIR),
        'val': (VAL_ANN, VAL_IMG_DIR),
        'test': (TEST_ANN, TEST_IMG_DIR),
    }[split]
    if detector == 'rtdetr':
        return RTDetrDataset(ann, img_dir, is_train=is_train, processor=processor,
                             img_size=img_size, subset=subset, aug_kwargs=aug_kwargs)
    return DocLayNetDataset(ann, img_dir, is_train=is_train, subset=subset)


def get_train_loader(batch_size=None, num_workers=None, detector=None, processor=None,
                     img_size=None, subset=None, no_aug=False, rand_crop=0.0):
    from config import DETECTOR
    detector = detector or DETECTOR
    batch_size = batch_size or BATCH_SIZE
    aug_kwargs = {'hflip_p': 0.0, 'color_p': 0.0} if no_aug else {'rand_crop_p': rand_crop}
    ds = _make_dataset('train', True, detector, processor, img_size, subset, aug_kwargs)
    return _make_loader(ds, batch_size, True, detector, drop_last=True,
                        num_workers=num_workers)


def get_val_loader(batch_size=None, num_workers=None, detector=None, processor=None,
                   img_size=None, subset=None):
    from config import DETECTOR
    detector = detector or DETECTOR
    batch_size = batch_size or BATCH_SIZE
    ds = _make_dataset('val', False, detector, processor, img_size, subset)
    return _make_loader(ds, batch_size, False, detector, num_workers=num_workers)


def get_test_loader(batch_size=None, num_workers=None, detector=None, processor=None,
                    img_size=None, subset=None):
    from config import DETECTOR
    detector = detector or DETECTOR
    batch_size = batch_size or BATCH_SIZE
    ds = _make_dataset('test', False, detector, processor, img_size, subset)
    return _make_loader(ds, batch_size, False, detector, num_workers=num_workers)


if __name__ == "__main__":
    import sys
    from config import DETECTOR, RTDETR_MODEL_NAME

    detector = sys.argv[1] if len(sys.argv) > 1 else DETECTOR
    print(f"=== 自检: detector={detector} ===")
    if detector == 'rtdetr':
        proc = build_rtdetr_processor(RTDETR_MODEL_NAME, RTDETR_IMG_SIZE)
        loader = get_train_loader(batch_size=2, num_workers=0, detector='rtdetr',
                                  processor=proc)
        batch = next(iter(loader))
        pv = batch['pixel_values']
        print(f"  pixel_values: {tuple(pv.shape)} {pv.dtype} "
              f"值域=[{pv.min():.3f}, {pv.max():.3f}]")
        for i, t in enumerate(batch['targets']):
            print(f"  target[{i}]: boxes{tuple(t['boxes'].shape)} "
                  f"labels范围=[{t['labels'].min()},{t['labels'].max()}] "
                  f"orig_size={t['orig_size'].tolist()} image_id={t['image_id'].tolist()}")
    else:
        loader = get_train_loader(batch_size=2, num_workers=0, detector='frcnn')
        images, targets = next(iter(loader))
        print(f"  batch images: {len(images)}, shape: {images[0].shape}")
        print(f"  target[0]: {targets[0]['boxes'].shape[0]} boxes, "
              f"labels范围=[{targets[0]['labels'].min()},{targets[0]['labels'].max()}]")
    print("  OK")
