"""
工具函数
"""
import os
import json
import numpy as np
import cv2
import torch


# ==================== 图片读取（中文路径兼容） ====================

def imread(path, flags=cv2.IMREAD_COLOR):
    """
    兼容中文路径的图片读取
    解决 cv2.imread 在 Windows 中文路径下返回 None 的问题
    """
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite(path, img, params=None):
    """
    兼容中文路径的图片写入
    """
    ext = os.path.splitext(path)[1]
    result, buf = cv2.imencode(ext, img, params)
    if result:
        buf.tofile(path)
    return result


# ==================== 通用工具 ====================

class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_iou(boxes1, boxes2):
    """
    计算两组框之间的 IoU
    boxes1: [N, 4] (x1, y1, x2, y2) — tensor 或 numpy array
    boxes2: [M, 4] (x1, y1, x2, y2) — tensor 或 numpy array
    返回: [N, M] IoU 矩阵（tensor）
    """
    if not isinstance(boxes1, torch.Tensor):
        boxes1 = torch.tensor(boxes1, dtype=torch.float32)
    if not isinstance(boxes2, torch.Tensor):
        boxes2 = torch.tensor(boxes2, dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou


# ==================== COCO 格式工具 ====================

def load_coco_json(json_path):
    """加载 COCO JSON 标注文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    return coco


def coco_bbox_to_xyxy(bbox):
    """
    COCO bbox [x, y, w, h] → [x1, y1, x2, y2]
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_coco_bbox(bbox):
    """
    [x1, y1, x2, y2] → COCO bbox [x, y, w, h]
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


# ==================== HuggingFace 加载工具 ====================

def hf_local_first(fn, *args, **kwargs):
    """优先正常加载 HuggingFace 资产，网络不通时自动回退到本地缓存。

    为什么需要：AutoDL 上 huggingface.co 常常不可达（[Errno 101] Network is unreachable），
    而模型/processor 其实早就缓存在 ~/.cache/huggingface 里了。但 transformers 默认会
    先发 HEAD 请求检查更新，网络不通时会重试 5 次（几十秒）然后抛错 —— 训练启动就卡死
    在那里。这个包一层把"网络不通"变成"用本地缓存"，其余异常照常抛出，不掩盖真问题。

    对应的 env 开关（推荐在训练前设上，省掉那次 30 秒重试）：
        export HF_HUB_OFFLINE=1
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        network_down = any(k in msg for k in
                           ('connection', 'unreachable', 'offline', 'resolve', 'timeout'))
        if not network_down:
            raise
        print("  [提示] 连不上 HuggingFace，改用本地缓存（local_files_only=True）")
        return fn(*args, local_files_only=True, **kwargs)


# ==================== 可视化工具 ====================

# DocLayNet 11 类颜色表（BGR）
CATEGORY_COLORS = {
    "Caption": (255, 0, 0),        # 红
    "Footnote": (0, 255, 0),       # 绿
    "Formula": (0, 0, 255),        # 蓝
    "List-item": (255, 255, 0),    # 青
    "Page-footer": (255, 0, 255),  # 品红
    "Page-header": (0, 255, 255),  # 黄
    "Picture": (128, 0, 0),        # 深红
    "Section-header": (0, 128, 0), # 深绿
    "Table": (0, 0, 128),          # 深蓝
    "Text": (128, 128, 0),         # 深青
    "Title": (128, 0, 128),        # 紫
}


def draw_detections(image, boxes, labels, scores=None, class_names=None, threshold=0.5):
    """
    在图片上绘制检测结果

    Args:
        image: numpy array (H, W, 3) BGR
        boxes: list of [x1, y1, x2, y2]
        labels: list of int (类别ID)
        scores: list of float (置信度，可选)
        class_names: dict {id: name} (可选)
        threshold: 置信度阈值

    Returns:
        annotated image
    """
    img = image.copy()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        if scores is not None and scores[i] < threshold:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]

        # 获取类别名和颜色
        if class_names and label in class_names:
            name = class_names[label]
        else:
            name = str(label)

        color = CATEGORY_COLORS.get(name, (0, 255, 0))

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 标签文字
        text = name
        if scores is not None:
            text += f" {scores[i]:.2f}"

        # 文字背景
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ==================== 统计工具 ====================

def print_dataset_stats(coco_json_path):
    """打印数据集统计信息"""
    coco = load_coco_json(coco_json_path)

    images = coco.get('images', [])
    annotations = coco.get('annotations', [])
    categories = coco.get('categories', [])

    print(f"图片数量: {len(images)}")
    print(f"标注数量: {len(annotations)}")
    print(f"类别数量: {len(categories)}")

    # 统计每个类别的标注数
    cat_counts = {}
    for ann in annotations:
        cat_id = ann['category_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1

    print("\n各类别标注数:")
    for cat in sorted(categories, key=lambda c: c['id']):
        cat_id = cat['id']
        count = cat_counts.get(cat_id, 0)
        print(f"  {cat.get('name', cat_id)}: {count}")


if __name__ == "__main__":
    from config import TRAIN_ANN
    if os.path.exists(TRAIN_ANN):
        print("=== 训练集统计 ===")
        print_dataset_stats(TRAIN_ANN)
    else:
        print(f"标注文件不存在: {TRAIN_ANN}")
        print("请先运行 download_data.py 下载数据集")
