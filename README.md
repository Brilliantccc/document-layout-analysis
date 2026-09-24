# 项目3：文档版面分析系统

在 DocLayNet 上训练的文档版面元素检测系统，识别 11 类版面元素。
主模型 RT-DETR-R50vd，基线 Faster R-CNN R50-FPN，两者可对比。

DocLayNet-base test 集（499 页 / 6,349 个标注框）：

| 模型 | 输入 | mAP@0.5 | mAP@[.5:.95] | mAP@0.75 |
|------|------|:-------:|:------------:|:--------:|
| RT-DETR-R50vd（主模型） | 1024 | 0.626 | **0.491** | 0.533 |
| Faster R-CNN R50-FPN（基线） | 800 | 0.695 | **0.477** | 0.531 |

权重已发布到 ModelScope：[rtdetr-r50vd-doclaynet-layout](https://www.modelscope.cn/models/Brilliantccc/rtdetr-r50vd-doclaynet-layout) ·
[faster-rcnn-r50fpn-doclaynet-layout](https://www.modelscope.cn/models/Brilliantccc/faster-rcnn-r50fpn-doclaynet-layout)

## 功能特性

- ✅ 11 类版面元素检测（Text / Title / Table / Picture / Formula / List-item / Caption / Footnote / Section-header / Page-header / Page-footer）
- ✅ RT-DETR（端到端，无需 anchor 和 NMS）+ Faster R-CNN 基线，同一套数据与评估可直接对比
- ✅ COCO 标准评估：mAP@0.5 / mAP@[.5:.95] / mAP@0.75 + 逐类 AP + 多阈值扫描
- ✅ 训练前 pipeline 验证脚本，把静默错挡在几小时的训练之前
- ✅ 支持多分辨率训练（640 / 800 / 1024）
- ✅ 断点续训（含全局步数对齐）、早停、自动版本号目录

## 技术栈

| 层级 | 技术 |
|------|------|
| 主模型 | RT-DETR-R50vd，HuggingFace `transformers`，从 COCO+Objects365 预训练权重微调 |
| 基线 | Faster R-CNN ResNet50-FPN，torchvision |
| 训练 | PyTorch 2.1 + bf16，AdamW（RT-DETR）/ SGD（FRCNN），梯度累积 |
| 数据 | 原生 COCO 格式，`RTDetrImageProcessor` / `GeneralizedRCNNTransform` 直接消费 |
| 评估 | pycocotools（COCO 标准口径） |

选 RT-DETR 的理由：属于 DETR 家族（端到端、无需 NMS），带 query selection 和多尺度
hybrid encoder，收敛远快于原始 DETR；权重是 Apache-2.0，与本项目 MIT 许可兼容。

## 数据集

[DocLayNet-base](https://huggingface.co/datasets/pierreguillou/DocLayNet-base)
（IBM Research DocLayNet 的子集），原始分辨率 1025×1025：

```
data/
├── train.json          # COCO 标注（坐标是 1025 空间）
├── val.json
├── test.json
├── train/images/       # 6910 张 1025×1025 PNG
├── val/images/         # 648 张
└── test/images/        # 499 张
```

### ⚠ 标注被复制了多份（最重要的一条）

`pierreguillou/DocLayNet-base` 这份转换**把标注复制成了多份**：train 的 681,481 个标注
去重后只剩 **91,136** 个，**86.6% 是重复**。最极端的一页有 929 个 Table 框完全一模一样。

**不去重会摧毁 AP。** 同一位置堆着几百个完全相同的 GT，模型在那里给 1 个框只能匹配
1 个、其余全算漏检，该类 AP 被压到接近 0。实测（去重前后）：

| 类别 | 重复率 | 去重前 AP@0.5 | 去重后 |
|------|:------:|:-------------:|:------:|
| Table | 98.0% | 0.34 | 0.74 |
| Formula | 96.7% | 0.19 | 0.79 |
| Picture | 94.4% | 0.12 | 0.42 |
| Page-footer | 24.1% | 0.41 | 0.85 |

**最有说服力的证据**：第一版训练时，torchvision FRCNN 和 RT-DETR 这两个完全不同的架构
（连输入归一化都不同）给出了**同样的天花板和同样的类间排序** —— 这说明问题在 GT 而不在
模型。`dataset._build_gt_index` 已按 `(x, y, w, h, 类别)` 去重，加载时会打印丢弃了多少个。

> **任何数据统计都必须先去重。** 项目早期基于污染数据得出的结论（每页 98.6 个框、
> p99=453、短边 <32px 只占 21%）全部是错的，正确值是每页 13.2 个框、短边 <32px 占 62%。

### 类别分布（去重后）

| ID | 类别 | train 数量 | 占比 | 中位短边 |
|----|------|-----------:|-----:|---------:|
| 10 | Text | 42995 | 47.18% | 44.3 px |
| 4 | List-item | 15632 | 17.15% | 22.5 px |
| 8 | Section-header | 11836 | 12.99% | 14.5 px |
| 5 | Page-footer | 6076 | 6.67% | 12.1 px |
| 6 | Page-header | 4900 | 5.38% | 13.0 px |
| 9 | Table | 2985 | 3.28% | 165.8 px |
| 3 | Formula | 2036 | 2.23% | 38.7 px |
| 1 | Caption | 1883 | 2.07% | 18.2 px |
| 7 | Picture | 1863 | 2.04% | 208.0 px |
| 2 | Footnote | 488 | 0.54% | 12.9 px |
| 11 | Title | 442 | 0.48% | 20.3 px |

### 输入分辨率是主要杠杆

**短边 <32px 的框占比**：640 → 70.0%，800 → 62.2%，1024 → 56.4%。

**这个任务的主体是小目标检测**，所以分辨率比超参更重要。1024 是原生分辨率（零插值损失），
本项目的主模型就用 1024。

## 训练效果

DocLayNet-base **test** 集（499 页 / 6,349 框），COCO 标准评估：

| 模型 | 输入 | mAP@0.5 | mAP@[.5:.95] | mAP@0.75 |
|------|------|:-------:|:------------:|:--------:|
| RT-DETR-R50vd | 1024 | 0.626 | **0.491** | 0.533 |
| Faster R-CNN R50-FPN | 800 | 0.695 | **0.477** | 0.531 |

两个模型整体打平：FRCNN 在宽松阈值上高 0.07，RT-DETR 在严格指标上高 0.014。

### 逐类 AP@0.5

| 类别 | test 样本数 | RT-DETR | Faster R-CNN |
|------|:-----------:|:-------:|:------------:|
| Text | 3002 | **0.765** | 0.736 |
| List-item | 962 | 0.583 | **0.614** |
| Section-header | 873 | 0.757 | **0.780** |
| Page-footer | 408 | 0.850 | **0.882** |
| Page-header | 311 | 0.616 | **0.784** |
| Table | 252 | 0.740 | **0.773** |
| Formula | 150 | **0.794** | 0.694 |
| Caption | 149 | 0.467 | **0.574** |
| Picture | 143 | 0.419 | **0.675** |
| Title | 52 | **0.669** | 0.657 |
| Footnote | 47 | 0.226 | **0.474** |

> ⚠ `Footnote`(47) / `Title`(52) 在 test 里样本极少，单个数波动很大 —— 引用时请连样本数一起给。

### 两个模型的差异在哪

`Footnote`（0.226 / 0.474）和 `Picture`（0.419 / 0.675）差距最大。原因很可能是**特征层数**：
torchvision 的 FPN 从 stride 4 起（4/8/16/32），而 RT-DETR 的 `feat_strides=[8,16,32]`
最细的一层就粗一倍 —— 正好卡在小目标上。

### 为什么低于论文的 0.73

论文 Table 2 的基线（**注意是 mAP@[.5:.95]，不是 mAP@0.5**）：Mask R-CNN R50 0.724 /
R101 0.735 / Faster R-CNN R101 0.734 / YOLOv5x6 0.768。它们的训练集是**完整 DocLayNet**
的 69,375 页，而本项目用的是 1/10 子集：

```
论文       69375 / 6489 / 4994   (train/val/test)
base 子集   6910 /  648 /  499
```

**数据量差 10 倍、backbone 是 R50 对 R101** —— 这个差距里数据量占主导，不是实现问题。
参考上界：Docling 的 Heron-101（RT-DETRv2-R101，15 万文档）在原始 DocLayNet 上是 0.699。

## 模型下载

预训练模型权重已发布到 ModelScope：

- **主模型 RT-DETR-R50vd**：[Brilliantccc/rtdetr-r50vd-doclaynet-layout](https://www.modelscope.cn/models/Brilliantccc/rtdetr-r50vd-doclaynet-layout)
  （原生 transformers 格式，171 MB）
- **基线 Faster R-CNN R50-FPN**：[Brilliantccc/faster-rcnn-r50fpn-doclaynet-layout](https://www.modelscope.cn/models/Brilliantccc/faster-rcnn-r50fpn-doclaynet-layout)
  （`.pth`，166 MB，已去掉优化器状态）

下载后可直接推理，不需要先训练：

```bash
# 主模型（目录形式）
python inference.py --weights ./rtdetr-r50vd-doclaynet-layout --input page.png

# 基线（.pth 形式）
python inference.py --weights ./frcnn_r50fpn_doclaynet.pth --input page.png
```

RT-DETR 那份是**原生 transformers 目录**，也可以不用本项目代码：

```python
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
model = RTDetrForObjectDetection.from_pretrained("Brilliantccc/rtdetr-r50vd-doclaynet-layout")
processor = RTDetrImageProcessor.from_pretrained("Brilliantccc/rtdetr-r50vd-doclaynet-layout")
```

> ⚠ **必须用 transformers 4.48 ~ 4.49。** 5.x 重构了 RT-DETR 的内部结构、键名全变了
> （如 `encoder.encoder.0...out_proj` → `encoder.aifi.0...o_proj`），加载 4.49 存的权重会
> 大量 missing 且**不报错**（静默随机初始化）。代码里已加版本检查会直接拦住。
>
> ⚠ **标签口径两个模型不同**：RT-DETR 版的输出是 **0-based**（0=Caption … 10=Title），
> Faster R-CNN 版是 **1-based**（1=Caption … 11=Title，与 COCO `category_id` 一致）。
> 评估时 RT-DETR 需要 +1 才能和 GT 对上。

## 环境要求

```
镜像: PyTorch 2.1.0 / Python 3.10 / CUDA 12.1
GPU:  RTX 3090 (24GB)（训练用；推理 CPU 也能跑）
CPU:  15 vCPU
```

两条硬约束，见 `requirements.txt`：

- **`numpy < 2`** —— torch 2.1.x 是针对 numpy 1.x 编译的，配 numpy 2.x 会
  `Failed to initialize NumPy: _ARRAY_API not found`，之后任何 torch↔numpy 互操作
  都直接 `RuntimeError: Numpy is not available`
- **`transformers` 4.48 ~ 4.49** —— 5.x 要求 torch>=2.4，且改了 RT-DETR 的键名

## 安装

```bash
export HF_ENDPOINT=https://hf-mirror.com        # AutoDL 常连不上 huggingface.co
pip install -r requirements.txt
pip check                                        # 确认没把 numpy 顶到 2.x

# 预下载基座权重（约 170MB）
huggingface-cli download PekingU/rtdetr_r50vd_coco_o365
export HF_HUB_OFFLINE=1                          # 之后可断网；不设的话 transformers
                                                 # 启动时会去连 HF 检查更新，AutoDL 上会重试 5 次再崩
```

若只用发布好的权重做推理，装 `torch torchvision transformers pillow opencv-python` 即可，
不需要 pycocotools。

## 快速开始

### 1. 验证 pipeline（**正式训练前务必先跑**）

```bash
python verify_pipeline.py                 # Phase 0+1，约 1 分钟
python verify_pipeline.py --overfit 8     # 额外做过拟合测试，约 5 分钟
```

会检查 HF 的 API 行为、坐标往返无损、label 映射、空标注图、以及**LR 阶梯衰减在真实训练
循环里是否真的生效**，并过拟合 8 张图确认 loss 能降下去、eval 召回 ≥95%。

> 这一步不能省。本项目踩过的静默错（`sigmoid` 套两层把预测框锁死在 (0.5,0.731)、缺
> no-object 损失导致所有 query 都判前景、`validate()` 漏了 `model.eval()` 会让 60 个
> epoch 全部白跑）都是"训练能跑完、指标极低"的类型，靠看 loss 曲线发现不了。

### 2. 训练

```bash
# 主模型：RT-DETR，1024px。800px 下 batch 8 会 OOM（崩在 decoder self-attention），
# 故用 batch 2 + 累积 4 保持有效 batch 8，这样不需要改 LR
python train.py --detector rtdetr --epochs 60 --img_size 1024 \
    --batch_size 2 --accum_steps 4

# 基线：Faster R-CNN R50-FPN（torchvision 自动缩到 800×800）
python train.py --detector frcnn --epochs 60 --batch_size 8

# 断点续训（自动恢复 global_step 与优化器状态）
python train.py --resume runs/rtdetr/v1/latest.pth

# 冒烟：只用前 8 张图
python train.py --detector rtdetr --subset 8 --val_subset 32 --test_subset 32 --epochs 3
```

约 13 分钟/epoch @1024px（RT-DETR），60 epoch 约 13 小时。输出到 `runs/<detector>/v<N>/`，
每轮存 `latest.pth`，按 val mAP@0.5 存 `best.pth`。

常用可调项：`--rand_crop 0.3`（开随机裁剪增强，默认关）、`--num_queries 100`、
`--backbone_lr_scale 1.0`、`--nms_thresh 0.3`（仅 FRCNN）。

### 3. 评估

```bash
python evaluate.py --weights runs/rtdetr/v1/best.pth --split test
# 也支持直接评估发布好的原生 HF 目录
python evaluate.py --weights ./rtdetr-r50vd-doclaynet-layout --split test
```

在 5 个置信度阈值上各评估一次，取 mAP@0.5 最高的作为主结果，打印逐类 AP，
写进 `eval_results_<split>_<checkpoint名>.json`。

### 4. 推理

```bash
python inference.py --weights runs/rtdetr/v1/best.pth --input test.jpg
python inference.py --weights runs/rtdetr/v1/best.pth --input ./test_images/ --output_dir ./results/
```

`--detector` 会从 checkpoint 或权重目录自动判断，不用手动指定。

## 项目结构

```
项目3_文档版面分析系统/
├── config.py            # 超参与路径；resolve_train_params() 按 detector 返回对应一套
├── model.py             # RTDetrLayoutDetector（格式转换层）+ FasterRCNNLayoutDetector
├── dataset.py           # RTDetrDataset / DocLayNetDataset 两条数据通路 + 标注去重
├── train.py             # 训练骨架（LR 调度 / 断点续训 / 早停 / AMP / 分组 LR / 梯度累积）
├── evaluate.py          # pycocotools 评估 + 阈值扫描 + 逐类 AP
├── inference.py         # 单张/批量推理 + 可视化；支持 .pth 与原生 HF 目录
├── verify_pipeline.py   # 训练前的 pipeline 验证（务必先跑）
├── utils.py             # imread/imwrite/compute_iou/draw_detections/AverageMeter
├── preprocess.py        # 可选的图片预缩放（当前训练不依赖，保留备用）
├── requirements.txt
├── data/                # 数据集（不提交）
├── runs/                # 训练输出（不提交）
└── docs/
    └── 项目计划.md       # 完整开发记录：踩过的坑、数据事实、目标口径更正
```

## 实现要点

### 项目格式 ↔ HF 格式的边界

全项目内部只认一种 GT 表示：**xyxy 绝对像素 @原始分辨率 + 1-based 标签（1~11）**。
HF 需要的归一化 `cxcywh` + 0-based `class_labels` 只在 `model.RTDetrLayoutDetector`
内部换算，`dataset.py` 不把标注交给 processor。

这样 `evaluate.py` 的 GT 收集、`inference.py` 的 `draw_detections`、
`utils.CATEGORY_COLORS` 都不需要知道 HF 的存在，1-based ↔ 0-based 的转换和坐标空间
转换各自只发生在一个地方 —— **少一个转换点就少一类 off-by-one**。

### 交付数字以 `evaluate.py` 为准

`train.py` 里自写的 `compute_map` 会**系统性低报**：RT-DETR 上给 ~0.54 而标准口径是
0.626（差 0.09），FRCNN 上只差 0.012。原因是标准口径按 **(图像, 类别)** 截断前 100 个框，
而 RT-DETR 的 300 个 query 会集中输出到每页的主导类上，被截掉的全是低分 FP。

训练过程中用它看趋势没问题（同一口径、系统性偏移不影响排序），但**对外报的数字必须用
`evaluate.py`**。

### LR 阶梯衰减

step 级调度（RT-DETR）：线性 warmup 1000 步 → 恒定 → 衰减点按 **epochs 的比例**
（0.6 / 0.85）自动计算，而不是写死 epoch 数 —— 否则 `--epochs 80` 时衰减点还停在 36/51，
最后 29 个 epoch 全卡在最低 LR。FRCNN 走 epoch 级调度。

`verify_pipeline.py` 的 Phase 3 会跑真实 `train_one_epoch` 逐 step 记录 LR，断言走出一条
完整的台阶 —— 因为本项目在这个位置出过两次"代码看着对、调度静默不生效"的问题。

## 已知局限与后续方向

1. **增强偏弱**：当前只有水平翻转 + 颜色抖动，缺几何/缩放/裁剪。模型在第 10 个 epoch
   就收敛、之后 25 个 epoch 验证集完全不涨而训练损失仍在降，是典型的正则不足。
   代码里已有 `--rand_crop`（带"裁剪后剩余面积 <30% 的框丢弃"的保护，避免切断文本行），
   下一轮值得开。
2. **`--num_queries 300` 偏大**：去重后每页真实只有 13.2 个框，300 个 query 里 200 个
   连标准指标都不看，纯属浪费算力。
3. **RT-DETR 最细特征层是 stride 8**：`feat_strides=[8,16,32]`，比 FRCNN 的 stride 4 粗，
   这正是它在 `Footnote`/`Picture` 上落后 FRCNN 的可能原因。改它需要动预训练 encoder 的
   输入投影，不是改配置能解决的。
4. 可加 EMA（HF 未实现，需自己包一层，代价是显存 ×2）；或换
   `PekingU/rtdetr_v2_r50vd`（同架构，mAP 高约 1 点）。
5. **`do_normalize` 值得一验**：实测这个 checkpoint 的 `RTDetrImageProcessor` 是
   `do_normalize=False`（白底文档 255/0 只做 `/255`）。当前代码刻意跟随 HF 官方 demo 的
   默认值，没去"修正"它；但 PaddleDetection 原版预处理是带 ImageNet 均值的，
   改 `build_rtdetr_processor` 一处即可 A/B。

## 详细文档

完整的开发记录（每个坑的定位过程、数据事实、目标口径更正、超参选择依据）见
[docs/项目计划.md](docs/项目计划.md)。

## 许可证

MIT License

训练数据 [DocLayNet](https://github.com/DS4SD/DocLayNet)（IBM Research）采用
CDLA-Permissive-1.0，使用本项目权重时请一并遵守该数据集的许可与引用要求。
