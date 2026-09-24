# RT-DETR-R50vd Document Layout Analysis (DocLayNet, 11 classes)

在 DocLayNet-base 上微调的文档版面元素检测模型。端到端检测，无需 anchor 和 NMS，
识别 11 类版面元素（Text / Title / Table / Picture / Formula / List-item / Caption /
Footnote / Section-header / Page-header / Page-footer）。

## 效果

DocLayNet-base **test** 集（499 页 / 6,349 个标注框，COCO 标准评估）：

| 指标 | 值 |
|------|-----|
| mAP@0.5 | **0.626** |
| mAP@[.5:.95] | **0.491** |
| mAP@0.75 | 0.533 |

逐类 AP@0.5：

| 类别 | AP | 类别 | AP |
|------|-----|------|-----|
| Text | 0.765 | Formula | 0.794 |
| Page-footer | 0.850 | Caption | 0.467 |
| Section-header | 0.757 | Picture | 0.419 |
| Table | 0.740 | Title | 0.669 |
| Page-header | 0.616 | Footnote | 0.226 |
| List-item | 0.583 | | |

> `Footnote` / `Title` 在 test 集里只有 47 / 52 个框，这两个数字波动较大。

## ⚠ 两个必须注意的点

### 1. transformers 版本必须 4.48 ~ 4.49

本权重是用 transformers **4.49** 训练并导出的。**transformers 5.x 重构了 RT-DETR 的
内部结构，键名变了**：

| 4.49 | 5.x |
|------|-----|
| `model.encoder.encoder.0.layers.0.self_attn.out_proj` | `model.encoder.aifi.0.layers.0.self_attn.o_proj` |
| `model.decoder.layers.0.fc1` / `fc2` | `model.decoder.layers.0.mlp.fc1` / `fc2` |
| `class_embed.0` | `model.decoder.class_embed.0` |

用 5.x 加载会报大量 missing / unexpected keys（**不会报错退出，而是静默随机初始化**），
所以务必：

```bash
pip install "transformers>=4.48,<4.50"
```

### 2. 标签是 0-based，不是 COCO 的 1-based

`config.json` 里的 `id2label` 是 **0-based**：

| 模型输出 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 类别 | Caption | Footnote | Formula | List-item | Page-footer | Page-header | Picture | Section-header | Table | Text | Title |

而 DocLayNet 的 COCO 标注里 `category_id` 是 **1-based**（Caption=1 … Title=11）。
评估时要把模型输出的 label **+1** 才能和 GT 对上，否则 AP 会接近 0。

## 用法

### 原生 transformers

```python
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

repo = "Brilliantccc/rtdetr-r50vd-doclaynet-layout"
model = RTDetrForObjectDetection.from_pretrained(repo).eval()
processor = RTDetrImageProcessor.from_pretrained(repo)

image = Image.open("page.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")      # 内部缩放到 1024×1024

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_object_detection(
    outputs, target_sizes=[image.size[::-1]], threshold=0.5
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(model.config.id2label[label.item()], f"{score:.3f}", [round(v) for v in box.tolist()])
```

输入尺寸由 `preprocessor_config.json` 固定为 **1024×1024**（训练时的设置），
无需手动指定。注意该预处理链是 `do_normalize=False`（只做 /255），与 DETR 系列
通常的 ImageNet 归一化不同 —— 这是基座权重自带的配置，请勿改动。

### 配合项目代码

完整训练 / 评估 / 推理脚本见 [document-layout-analysis](https://github.com/Brilliantccc/document-layout-analysis)，
其中 `inference.py` 可直接加载本权重。

## 训练细节

| 项 | 值 |
|----|-----|
| 基座 | [`PekingU/rtdetr_r50vd_coco_o365`](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365)（COCO 53.1% AP / O365 55.3% AP） |
| 数据集 | DocLayNet-base：6,910 训练页 / 648 验证页 / 499 测试页 |
| 输入尺寸 | 1024×1024（原始分辨率，零插值损失） |
| 优化器 | AdamW，lr 1e-4（backbone 1e-5），weight decay 1e-4，grad clip 0.1 |
| batch | 2 × 梯度累积 4（有效 batch 8） |
| epochs | 60，线性 warmup 1000 步，阶梯衰减 ×0.3 @ epoch 36 / 51 |
| 精度 | bf16 |
| 硬件 | 单卡 RTX 3090 24GB，约 13 分钟 / epoch |
| 选模型 | 按验证集 mAP@0.5 选 best（最终 best 出自 epoch 31） |

**与论文基线的可比性**：DocLayNet-base 是完整 DocLayNet 的约 **1/10**
（6,910 vs 69,375 训练页）。论文 Table 2 的基线（mAP@[.5:.95] 0.72 ~ 0.77，R101 级
backbone）用的是完整数据集，因此本模型的 0.491 **不是同口径对比**。

## ⚠ 数据说明（想复现训练的话必读）

`pierreguillou/DocLayNet-base` 这份转换**把标注复制了多份**：681,481 个标注去重后
只剩 91,136 个（**86.6% 是重复**），其中 Table 98.0% / Formula 96.7% / Picture 94.4% /
Text 86.6% 是重复。有一页的 929 个 Table 框完全一模一样。

**不去重直接训练会被严重污染**：同一位置堆着几百个完全相同的 GT，模型在那里给 1 个框
只能匹配 1 个、其余全算漏检，该类 AP 被压到接近 0（实测不去重时 Table AP 只有 0.34，
去重后 0.74）。复现时务必先按 `(x, y, w, h, 类别)` 去重。

## 引用与许可

- **模型权重**：MIT（同项目）
- **训练数据**：[DocLayNet](https://github.com/DS4SD/DocLayNet)（IBM Research），
  CDLA-Permissive-1.0。使用本模型请一并遵守该数据集的许可与引用要求
- **基座权重**：[PekingU/rtdetr_r50vd_coco_o365](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365)，Apache-2.0
- **RT-DETR 论文**：Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection*, arXiv:2304.08069
