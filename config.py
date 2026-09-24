"""
文档版面分析系统配置文件
数据集: DocLayNet-base (6910 train / 648 val / 499 test, 原始 1025×1025)
主模型: RT-DETR (RT-DETR-R50vd, HuggingFace transformers, COCO+Objects365 预训练)
基线:   Faster R-CNN (ResNet50-FPN) via torchvision
"""
import os
import torch

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# COCO 标注文件（原始 1025×1025 坐标）
TRAIN_ANN = os.path.join(DATA_DIR, "train.json")
VAL_ANN = os.path.join(DATA_DIR, "val.json")
TEST_ANN = os.path.join(DATA_DIR, "test.json")

# 图片目录（原始 1025×1025，由 processor / torchvision transform 缩放到 IMG_SIZE）
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")

# 模型保存路径
MODEL_DIR = os.path.join(BASE_DIR, "runs")


# ==================== 类别配置 ====================
# 项目内部约定：标签一律 1-based（1~11），与 COCO category_id 一致
LAYOUT_CLASSES = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item",
    5: "Page-footer", 6: "Page-header", 7: "Picture",
    8: "Section-header", 9: "Table", 10: "Text", 11: "Title",
}
# torchvision Faster R-CNN 需要 0=背景 + 11 个前景类
NUM_CLASSES = len(LAYOUT_CLASSES) + 1  # 12

# ---- RT-DETR 的 0-based 标签映射（HF 侧只用这一套）----
# 换算关系：project_label(1-based) = hf_label(0-based) + LABEL_OFFSET
LABEL_OFFSET = 1
RTDETR_ID2LABEL = {k - LABEL_OFFSET: v for k, v in sorted(LAYOUT_CLASSES.items())}
RTDETR_LABEL2ID = {v: k for k, v in RTDETR_ID2LABEL.items()}
RTDETR_NUM_CLASSES = len(RTDETR_ID2LABEL)  # 11（RT-DETR 没有独立的背景类）

# 分类损失权重（平方根反频率，平滑类别不均衡）——仅 Faster R-CNN 使用
# [背景, Caption, Footnote, Formula, List-item, Page-footer, Page-header,
#  Picture, Section-header, Table, Text, Title]
CLASS_WEIGHTS = [1.0, 0.88, 2.25, 0.40, 0.39, 1.11, 1.17, 0.54, 0.76, 0.25, 0.17, 3.08]

# ==================== 训练配置（RTX 3090 24GB） ====================
BACKBONE = 'resnet50'       # 'resnet50' 或 'swin_t'（仅 Faster R-CNN 用）
DETECTOR = 'rtdetr'         # 'rtdetr' | 'frcnn'（仅决定默认值，实际以命令行 --detector 为准）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- RT-DETR（主模型）---
RTDETR_MODEL_NAME = "PekingU/rtdetr_r50vd_coco_o365"   # 升级备选: PekingU/rtdetr_v2_r50vd
RTDETR_IMG_SIZE = 800           # {640, 800, 1024}；1024 为原生分辨率（零插值损失）
RTDETR_NUM_QUERIES = 300        # 保持预训练值。3.73% 的图超过 300 框，实测召回不足再提到 600
# ⚠ batch=8 @800px 在 24GB 的 3090 上会 OOM（实测崩在 decoder self-attention，
#   要再要 1.64 GiB）。原因是 decoder 的 query 数不是 300，还要加 denoising 的
#   query，注意力矩阵比预期大得多。故用 batch=4 + 累积 2 保持有效 batch=8，
#   这样不需要改 LR（线性缩放规则下有效 batch 与 LR 是配套的）。
RTDETR_BATCH_SIZE = 4
RTDETR_ACCUM_STEPS = 2          # 有效 batch = 4 x 2 = 8
RTDETR_BASE_LR = 0.0001         # RT-DETR 官方 recipe
RTDETR_BACKBONE_LR_SCALE = 0.1  # backbone lr = 1e-5
RTDETR_WEIGHT_DECAY = 0.0001
RTDETR_GRAD_CLIP_NORM = 0.1     # 官方值（注意不是 5.0）
RTDETR_WARMUP_STEPS = 1000      # 线性 warmup，≈0.58 epoch @ bs4（按 micro-batch 计）
RTDETR_BACKBONE_FREEZE_EPOCHS = 1   # 只冻第 1 个 epoch，给新 head 一个缓冲
RTDETR_EPOCHS = 60
RTDETR_EARLY_STOP_PATIENCE = 12


# 衰减点用【epochs 的比例】定义，而不是写死 epoch 数 ——
# 否则 `--epochs 80` 时衰减点还停在 36/51，最后 29 个 epoch 全卡在最低 LR。
RTDETR_LR_DECAY_FRACS = [(0.6, 0.3), (0.85, 0.3)]   # (占 epochs 的比例, 衰减倍率)


def rtdetr_decay_steps(epochs):
    """按总 epoch 数算出阶梯衰减点。1e-4 → 3e-5 → 9e-6"""
    return [(max(1, round(f * epochs)), m) for f, m in RTDETR_LR_DECAY_FRACS]


RTDETR_LR_DECAY_STEPS = rtdetr_decay_steps(RTDETR_EPOCHS)

# --- Faster R-CNN（基线，按 backbone 分配）---
FRCNN_SWIN_BATCH_SIZE = 4
FRCNN_SWIN_BASE_LR = 0.0002
FRCNN_SWIN_WARMUP_EPOCHS = 3
FRCNN_SWIN_WEIGHT_DECAY = 0.05
FRCNN_BATCH_SIZE = 8
FRCNN_BASE_LR = 0.001
FRCNN_WARMUP_EPOCHS = 1
FRCNN_WEIGHT_DECAY = 0.0001
FRCNN_EPOCHS = 60
FRCNN_LR_DECAY_STEPS = [(40, 0.3), (52, 0.3)]   # R50-FPN 收敛慢，给两段衰减

WEIGHT_DECAY = 0.0001       # 通用默认值


def resolve_train_params(detector=None, backbone=None):
    """
    按【实际使用的】detector/backbone 返回训练超参。

    之所以要这个函数：config 里的 DETECTOR 是静态默认值，`python train.py --detector frcnn`
    并不会改它。早期代码里 train.py 一半判断用命令行 detector_name、一半用 config.DETECTOR，
    导致命令行切换检测器时会串到另一套调度上。这里统一以调用方传入的 detector 为准。

    返回 dict 的 key 对所有分支保持一致（train.py 会无条件读取），因此新增字段时
    三个分支都要补，否则 KeyError。
    """
    detector = detector or DETECTOR
    backbone = backbone or BACKBONE

    if detector == 'rtdetr':
        return dict(
            detector='rtdetr',
            batch_size=RTDETR_BATCH_SIZE,
            base_lr=RTDETR_BASE_LR,
            epochs=RTDETR_EPOCHS,
            backbone_lr_scale=RTDETR_BACKBONE_LR_SCALE,
            weight_decay=RTDETR_WEIGHT_DECAY,
            grad_clip_norm=RTDETR_GRAD_CLIP_NORM,
            optimizer='adamw',
            lr_schedule='step',             # step 级 warmup + 阶梯衰减
            amp_dtype='bf16',               # VFL 的 BCEWithLogits 在 fp16 下易溢出
            accum_steps=RTDETR_ACCUM_STEPS,
            lr_decay_steps=list(RTDETR_LR_DECAY_STEPS),
            warmup_steps=RTDETR_WARMUP_STEPS,
            warmup_epochs=0,
            backbone_freeze_epochs=RTDETR_BACKBONE_FREEZE_EPOCHS,
            num_queries=RTDETR_NUM_QUERIES,
            img_size=RTDETR_IMG_SIZE,
            model_name_or_path=RTDETR_MODEL_NAME,
            early_stop_patience=RTDETR_EARLY_STOP_PATIENCE,
            # 早停不能早于最后一个衰减点，否则会在 LR 还没降下来时就停掉
            early_stop_min_epoch=RTDETR_LR_DECAY_STEPS[-1][0],
        )

    if backbone == 'swin_t':
        return dict(
            detector='frcnn',
            batch_size=FRCNN_SWIN_BATCH_SIZE,
            base_lr=FRCNN_SWIN_BASE_LR,
            epochs=FRCNN_EPOCHS,
            backbone_lr_scale=1.0,
            weight_decay=FRCNN_SWIN_WEIGHT_DECAY,
            grad_clip_norm=5.0,
            optimizer='adamw',
            lr_schedule='epoch',            # epoch 级 warmup（step 级对 FRCNN 收益不明显）
            amp_dtype='fp16',
            lr_decay_steps=list(FRCNN_LR_DECAY_STEPS),
            warmup_steps=0,
            warmup_epochs=FRCNN_SWIN_WARMUP_EPOCHS,
            backbone_freeze_epochs=0,
            num_queries=None,
            img_size=None,                  # torchvision transform 自己按 min_size=800 缩放
            model_name_or_path=None,
            early_stop_patience=20,
            early_stop_min_epoch=0,
        )

    return dict(
        detector='frcnn',
        batch_size=FRCNN_BATCH_SIZE,
        base_lr=FRCNN_BASE_LR,
        epochs=FRCNN_EPOCHS,
        backbone_lr_scale=1.0,
        weight_decay=FRCNN_WEIGHT_DECAY,
        grad_clip_norm=5.0,
        optimizer='sgd',
        lr_schedule='epoch',
        amp_dtype='fp16',
        lr_decay_steps=list(FRCNN_LR_DECAY_STEPS),
        warmup_steps=0,
        warmup_epochs=FRCNN_WARMUP_EPOCHS,
        backbone_freeze_epochs=0,
        num_queries=None,
        img_size=None,
        model_name_or_path=None,
        early_stop_patience=15,
        early_stop_min_epoch=0,
    )


# 默认值：config.DETECTOR/BACKBONE 选中的那一套（dataset.py 等仍直接引用 BATCH_SIZE）
_DEFAULT_PARAMS = resolve_train_params()
BATCH_SIZE = _DEFAULT_PARAMS['batch_size']
BASE_LR = _DEFAULT_PARAMS['base_lr']
EPOCHS = _DEFAULT_PARAMS['epochs']
WARMUP_EPOCHS = _DEFAULT_PARAMS['warmup_epochs']
BACKBONE_LR_SCALE = _DEFAULT_PARAMS['backbone_lr_scale']
LR_DECAY_STEPS = _DEFAULT_PARAMS['lr_decay_steps']
BACKBONE_FREEZE_EPOCHS = _DEFAULT_PARAMS['backbone_freeze_epochs']
NUM_QUERIES = _DEFAULT_PARAMS['num_queries']
IMG_SIZE = _DEFAULT_PARAMS['img_size']
EARLY_STOP_PATIENCE = _DEFAULT_PARAMS['early_stop_patience']

# 数据加载
NUM_WORKERS = 12         # 多开 worker 加速 CPU 预处理（AutoDL 通常 16 核）
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 6      # 多预取减少 GPU 等待

# ==================== 后处理 ====================
DETECTION_SCORE_THRESH = 0.5
# 评估时扫描的置信度阈值（RT-DETR 的分数分布与 FRCNN 不同，固定 0.5 可能滤掉真阳性）
EVAL_SCORE_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5]


if __name__ == "__main__":
    print(f"默认检测器: {DETECTOR} (backbone={BACKBONE})")
    for det in ['rtdetr', 'frcnn']:
        p = resolve_train_params(det, 'resnet50')
        accum = p.get('accum_steps', 1)
        print(f"\n[{det}] batch={p['batch_size']} x accum={accum} "
              f"(有效 {p['batch_size'] * accum}) lr={p['base_lr']} epochs={p['epochs']} "
              f"opt={p['optimizer']} amp={p['amp_dtype']} clip={p['grad_clip_norm']}")
        print(f"  lr_schedule={p['lr_schedule']} decay={p['lr_decay_steps']} "
              f"warmup_steps={p['warmup_steps']} warmup_epochs={p['warmup_epochs']}")
        print(f"  img_size={p['img_size']} queries={p['num_queries']} "
              f"freeze={p['backbone_freeze_epochs']} model={p['model_name_or_path']}")
        print(f"  patience={p['early_stop_patience']} (最早 epoch {p['early_stop_min_epoch']})")

    print(f"\n类别: {RTDETR_NUM_CLASSES} 类 (RT-DETR 0-based) / {NUM_CLASSES} 类 (FRCNN 含背景)")
    print(f"  RT-DETR id2label = {RTDETR_ID2LABEL}")

    print("\n通用配置:")
    for k, v in {
        'NUM_WORKERS': NUM_WORKERS, 'PERSISTENT_WORKERS': PERSISTENT_WORKERS,
        'PREFETCH_FACTOR': PREFETCH_FACTOR, 'DETECTION_SCORE_THRESH': DETECTION_SCORE_THRESH,
        'EVAL_SCORE_THRESHOLDS': EVAL_SCORE_THRESHOLDS,
    }.items():
        print(f"  {k:25s} = {v}")

    print("\n数据路径:")
    for k, v in {'TRAIN_ANN': TRAIN_ANN, 'TRAIN_IMG_DIR': TRAIN_IMG_DIR}.items():
        print(f"  {k:25s} = {v}  {'OK' if os.path.exists(v) else '!! 不存在'}")
