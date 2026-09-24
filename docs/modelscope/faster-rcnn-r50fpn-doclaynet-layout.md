# Faster R-CNN R50-FPN Document Layout Analysis (DocLayNet, 11 classes)

在 DocLayNet-base 上训练的文档版面元素检测模型（torchvision Faster R-CNN +
ResNet50-FPN 主干），识别 11 类版面元素。是
[document-layout-analysis](https://github.com/Brilliantccc/document-layout-analysis) 项目
的阶段1基线，用于和 RT-DETR 主模型对比。

## 效果

DocLayNet-base **test** 集（499 页 / 6,349 个标注框，COCO 标准评估）：

| 指标 | 值 |
|------|-----|
| mAP@0.5 | **0.695** |
| mAP@[.5:.95] | **0.477** |
| mAP@0.75 | 0.531 |

逐类 AP@0.5：

| 类别 | AP | 类别 | AP |
|------|-----|------|-----|
| Page-footer | 0.882 | Table | 0.773 |
| Page-header | 0.784 | Text | 0.736 |
| Section-header | 0.780 | Formula | 0.694 |
| Picture | 0.675 | List-item | 0.614 |
| Title | 0.657 | Caption | 0.574 |
| Footnote | 0.474 | | |

> `Footnote` / `Title` 在 test 集里只有 47 / 52 个框，这两个数字波动较大。

**和 RT-DETR 主模型的对比**：两者整体接近（RT-DETR mAP@[.5:.95] 0.491 /
本模型 0.477）。本模型在 `Footnote`（0.474 vs 0.226）和 `Picture`（0.675 vs 0.419）
上明显更好 —— 原因很可能是它的 FPN 从 stride 4 起（4/8/16/32），而 RT-DETR 的
`feat_strides=[8,16,32]` 最细的一层就粗一倍，正好卡在小目标上。

## 用法

权重文件是 `frcnn_r50fpn_doclaynet.pth`，用项目代码加载：

```bash
git clone https://github.com/Brilliantccc/document-layout-analysis
cd document-layout-analysis
pip install -r requirements.txt
```

```python
import torch
from model import LayoutDetector

ckpt = torch.load("frcnn_r50fpn_doclaynet.pth", map_location="cpu", weights_only=False)
model = LayoutDetector(detector="frcnn", backbone="resnet50", pretrained=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

或直接用项目里的推理脚本：

```bash
python inference.py --weights frcnn_r50fpn_doclaynet.pth --input page.png
```

## ⚠ 标签这里是 1-based（和 RT-DETR 版相反）

torchvision 的检测头约定 `0 = 背景`，本项目把 11 个前景类放在 **1 ~ 11**，
与 DocLayNet 的 COCO `category_id` **完全一致**：

| 模型输出 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 类别 | Caption | Footnote | Formula | List-item | Page-footer | Page-header | Picture | Section-header | Table | Text | Title |

所以本模型的输出可以直接和 COCO 标注对齐；而同项目的 RT-DETR 版输出是 0-based，
需要 +1。**两者不要混用。**

## 训练细节

| 项 | 值 |
|----|-----|
| 模型 | torchvision Faster R-CNN + ResNet50-FPN（COCO 预训练） |
| 数据集 | DocLayNet-base：6,910 训练页 / 648 验证页 / 499 测试页 |
| 输入 | 800×800（`GeneralizedRCNNTransform` 默认 `min_size=800`） |
| 优化器 | SGD，lr 1e-3，momentum 0.9，weight decay 1e-4 |
| batch | 8 |
| epochs | 60，warmup 1 epoch，阶梯衰减 ×0.3 @ epoch 40 / 52 |
| 精度 | fp16（`GradScaler`） |
| 硬件 | 单卡 RTX 3090 24GB |
| 选模型 | 按验证集 mAP@0.5 选 best（最终 best 出自 epoch 56） |
| 其它 | RPN anchors 5 尺度 × 3 长宽比；`box_nms_thresh=0.1`；`box_detections_per_img=1500` |

**与论文基线的可比性**：DocLayNet-base 是完整 DocLayNet 的约 **1/10**
（6,910 vs 69,375 训练页）。论文 Table 2 的基线用的是完整数据集，因此本模型的
0.477 **不是同口径对比**。

## ⚠ 数据说明（想复现训练的话必读）

`pierreguillou/DocLayNet-base` 这份转换**把标注复制了多份**：681,481 个标注去重后
只剩 91,136 个（**86.6% 是重复**）。不去重直接训练会被严重污染：同一位置堆着几百个
完全相同的 GT，模型给 1 个框只能匹配 1 个、其余全算漏检，该类 AP 被压到接近 0。
复现时务必先按 `(x, y, w, h, 类别)` 去重 —— 实测去重前后 Table 的 AP 从 0.01 变成 0.77。

## 引用与许可

- **模型权重**：MIT（同项目）
- **训练数据**：[DocLayNet](https://github.com/DS4SD/DocLayNet)（IBM Research），
  CDLA-Permissive-1.0。使用本模型请一并遵守该数据集的许可与引用要求
- **实现**：torchvision（BSD-3-Clause）
