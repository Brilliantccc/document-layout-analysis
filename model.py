"""
文档版面分析模型

主模型: RT-DETR (HuggingFace transformers, RT-DETR-R50vd, COCO+Objects365 预训练)
基线:   Faster R-CNN (ResNet50/Swin-T + FPN) via torchvision

RTDetrLayoutDetector 是一个【格式转换层】：对外沿用项目的调用约定
    forward(images, targets) 收 项目格式 GT（xyxy 像素 @原图 + 1-based 标签）
    训练返回 loss_dict，eval 返回 list[{'boxes','labels','scores'}]
HF 需要的 归一化 cxcywh + 0-based class_labels 全部封在 wrapper 内部。
这样 evaluate.py / inference.py / utils.draw_detections 都不需要知道 HF 的存在。
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    NUM_CLASSES, CLASS_WEIGHTS,
    RTDETR_ID2LABEL, LABEL_OFFSET,
    RTDETR_MODEL_NAME, RTDETR_NUM_QUERIES, RTDETR_IMG_SIZE,
)


# ==================== RT-DETR（主模型） ====================

class RTDetrLayoutDetector(nn.Module):
    """HuggingFace RT-DETR 包装层。

    训练: forward(pixel_values=[B,3,S,S], targets=list[项目格式 GT]) -> {loss_vfl, loss_bbox, loss_giou}
    eval: forward(pixel_values, targets=None)                    -> list[{'boxes' xyxy@原图, 'labels' 1-based, 'scores'}]

    eval 时传入 targets 是为了拿 orig_size 把框缩放回原始分辨率；不传则返回 S 空间的坐标。
    """

    def __init__(self, num_classes=None, pretrained=True, num_queries=None,
                 model_name_or_path=None, img_size=None, id2label=None):
        super().__init__()
        from transformers import RTDetrForObjectDetection, RTDetrConfig

        self.img_size = img_size or RTDETR_IMG_SIZE
        self.num_queries = num_queries or RTDETR_NUM_QUERIES
        self.model_name_or_path = model_name_or_path or RTDETR_MODEL_NAME
        id2label = id2label or RTDETR_ID2LABEL
        label2id = {v: k for k, v in id2label.items()}

        if pretrained:
            # ignore_mismatched_sizes=True 会重建 class_embed（COCO 80 类 → 11 类）
            # hf_local_first：连不上 HF 时自动用本地缓存，不让整个训练卡在重试上
            from utils import hf_local_first
            self.hf_model = hf_local_first(
                RTDetrForObjectDetection.from_pretrained,
                self.model_name_or_path,
                id2label=id2label, label2id=label2id,
                num_queries=self.num_queries,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = RTDetrConfig.from_pretrained(
                self.model_name_or_path,
                id2label=id2label, label2id=label2id,
                num_queries=self.num_queries,
            )
            self.hf_model = RTDetrForObjectDetection(cfg)

        n = self.hf_model.config.num_labels
        assert n == len(id2label), f"num_labels={n} 与 id2label({len(id2label)}) 不一致"

        self._processor = None

    # ---- 让 train.py 的 backbone 冻结逻辑（hasattr(model,'backbone')）继续可用 ----
    @property
    def backbone(self):
        return self.hf_model.model.backbone

    # ---- processor 懒加载：它只用于 eval 后处理，训练路径不需要 ----
    @property
    def processor(self):
        if self._processor is None:
            from dataset import build_rtdetr_processor
            self._processor = build_rtdetr_processor(self.model_name_or_path, self.img_size)
        return self._processor

    # ------------------------------------------------------------------
    def _targets_to_hf(self, targets):
        """项目格式 GT (xyxy 像素 @原图, 1-based) → HF 格式 (归一化 cxcywh, 0-based)

        归一化坐标是尺度无关的：原图 (x1,y1,x2,y2) 除以原图 H0/W0 得到的归一化值，
        与 processor 把图缩放到 S×S 后再归一化的结果完全一致。
        """
        hf = []
        for t in targets:
            boxes = t['boxes'].float()
            if 'orig_size' in t:
                hs = t['orig_size'].to(boxes.device).float()
                H, W = hs[0], hs[1]
            else:
                H = W = float(self.img_size)
            n = boxes.shape[0]
            if n == 0:
                hf.append({
                    'class_labels': torch.zeros((0,), dtype=torch.long, device=boxes.device),
                    'boxes': torch.zeros((0, 4), dtype=torch.float32, device=boxes.device),
                })
                continue

            x1, y1, x2, y2 = boxes.unbind(-1)
            # 先裁到图像范围，保证 cxcywh 合法（原数据有 0.03% 的框越界，最大坐标 1025.1）
            x1 = x1.clamp(0, W)
            y1 = y1.clamp(0, H)
            x2 = x2.clamp(0, W)
            y2 = y2.clamp(0, H)
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            hf.append({
                # 1..11 → 0..10。HF 的 processor 不会做这个偏移，必须自己减
                'class_labels': (t['labels'].long() - LABEL_OFFSET),
                'boxes': torch.stack([cx, cy, bw, bh], dim=-1),
            })
        return hf

    @staticmethod
    def _aggregate_loss(loss_dict):
        """把 RT-DETR 的 18 个 loss key（6 层 decoder × 3 项 + aux）聚成 3 个。

        sum 保持不变，但 tqdm/日志不会刷 18 行。
        必须过滤非 'loss_' 前缀的 key（如 cardinality_error），
        它们不带梯度，混进 sum().backward() 会直接报错。
        """
        agg = {}
        for k, v in loss_dict.items():
            if not k.startswith('loss_'):
                continue
            if 'vfl' in k:
                base = 'loss_vfl'
            elif 'bbox' in k:
                base = 'loss_bbox'
            elif 'giou' in k:
                base = 'loss_giou'
            else:
                base = k
            agg[base] = agg.get(base, 0.0) + v
        return agg

    def forward(self, images, targets=None, orig_sizes=None):
        """eval 时把框映射回原始分辨率的优先顺序：
        orig_sizes > targets[i]['orig_size'] > 不做映射（返回 S 空间坐标）
        """
        if self.training and targets is not None:
            out = self.hf_model(pixel_values=images,
                                labels=self._targets_to_hf(targets))
            return self._aggregate_loss(out.loss_dict)

        # ---- eval ----
        out = self.hf_model(pixel_values=images)
        B = images.shape[0]
        S = self.img_size
        # 先按 S×S 反算出 S 空间的像素框，再乘回原图尺寸。
        # （直接传原图 (H0,W0) 也可以，但 processor 是无脑 resize 到 S×S，
        #   统一走"S 空间 → 缩放"这条路对非正方图也成立。）
        results = self.processor.post_process_object_detection(
            out, threshold=0.01,      # 这里只做粗筛，真正的阈值由调用方/evaluate.py 决定
            target_sizes=torch.tensor([[S, S]] * B, device=images.device),
            use_focal_loss=True,
        )

        scaled = []
        for i, r in enumerate(results):
            boxes = r['boxes']
            size = None
            if orig_sizes is not None:
                size = orig_sizes[i]
            elif targets is not None and 'orig_size' in targets[i]:
                size = targets[i]['orig_size']
            if size is not None:
                H0, W0 = size.to(boxes.device).float()
                boxes = boxes * torch.stack([W0 / S, H0 / S, W0 / S, H0 / S])
            scaled.append({
                'boxes': boxes,
                'scores': r['scores'],
                'labels': r['labels'] + LABEL_OFFSET,   # 0..10 → 1..11
            })
        return scaled

    def save_pretrained(self, output_dir, processor=None):
        """把 HF config + processor 存进 run 目录，让 evaluate/inference 不必联网。

        processor 传进来就复用（train.py 已经构造过一份，避免重复建/重复下载）。
        """
        os.makedirs(output_dir, exist_ok=True)
        self.hf_model.config.save_pretrained(output_dir)
        (processor or self.processor).save_pretrained(output_dir)


# ==================== Swin-T Backbone + FPN（给 Faster R-CNN 用） ====================

class SwinTBackbone(nn.Module):
    """Swin-T Transformer backbone，输出多尺度特征图供 FPN 使用"""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import swin_t, Swin_T_Weights

        weights = Swin_T_Weights.DEFAULT if pretrained else None
        swin = swin_t(weights=weights)

        self.stage0 = nn.Sequential(swin.features[0], swin.features[1])
        self.stage1 = nn.Sequential(swin.features[2], swin.features[3])
        self.stage2 = nn.Sequential(swin.features[4], swin.features[5])
        self.stage3 = nn.Sequential(swin.features[6], swin.features[7])

        self.out_channels = 96

    def forward(self, x):
        c0 = self.stage0(x)
        c1 = self.stage1(c0)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        return {
            '0': c0.permute(0, 3, 1, 2),
            '1': c1.permute(0, 3, 1, 2),
            '2': c2.permute(0, 3, 1, 2),
            '3': c3.permute(0, 3, 1, 2),
        }


class ResNetBackbone(nn.Module):
    """ResNet backbone，输出 4 个 stage 的特征图供 FPN 使用。

    注意不要用 `body.avgpool = None; body.fc = None` 的写法拿中间特征 ——
    torchvision 的 ResNet._forward_impl 是无条件调用 self.avgpool(x) 的，
    置 None 会直接 TypeError: 'NoneType' object is not callable。
    这里显式拆出各 stage，和 SwinTBackbone 保持同一种写法。
    """

    def __init__(self, pretrained=True, arch='resnet50'):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        m = resnet50(weights=weights)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4
        self.out_channels = 2048

    def forward(self, x):
        x = self.stem(x)
        c0 = self.layer1(x)
        c1 = self.layer2(c0)
        c2 = self.layer3(c1)
        c3 = self.layer4(c2)
        return {'0': c0, '1': c1, '2': c2, '3': c3}


def _build_backbone(backbone_name, pretrained):
    if backbone_name == 'swin_t':
        body = SwinTBackbone(pretrained=pretrained)
        in_channels_list = [96, 192, 384, 768]
        out_channels = 256
    else:
        body = ResNetBackbone(pretrained=pretrained, arch=backbone_name)
        in_channels_list = [256, 512, 1024, 2048]
        out_channels = 256

    from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )

    class BackboneWithFPN(nn.Module):
        def __init__(self, body, fpn, out_channels):
            super().__init__()
            self.body = body
            self.fpn = fpn
            self.out_channels = out_channels
        def forward(self, x):
            return self.fpn(self.body(x))

    return BackboneWithFPN(body, fpn, out_channels)


# ==================== Faster R-CNN 版本（基线） ====================

def _make_weighted_forward(original_forward, class_weights):
    from torchvision.ops import boxes as box_ops

    def weighted_forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            labels_cat = torch.cat(labels, dim=0)
            regression_targets_cat = torch.cat(regression_targets, dim=0)

            device = class_logits.device
            weights = class_weights[:class_logits.shape[1]].to(device)

            # 注意：weight= 已经施加了类别权重，再乘一次 sample_weights 等于按平方加权。
            # CLASS_WEIGHTS 是"平方根反频率"，所以最终等效于按反频率（inverse frequency）加权。
            per_sample_loss = F.cross_entropy(class_logits, labels_cat, weight=weights, reduction='none')
            sample_weights = weights[labels_cat]
            loss_classifier = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp(min=1)

            sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
            labels_pos = labels_cat[sampled_pos_inds_subset]
            N, _ = class_logits.shape
            box_regression_flat = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

            loss_box_reg = F.smooth_l1_loss(
                box_regression_flat[sampled_pos_inds_subset, labels_pos],
                regression_targets_cat[sampled_pos_inds_subset],
                beta=1 / 9, reduction="sum",
            )
            loss_box_reg = loss_box_reg / labels_cat.numel()

            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            for i in range(len(boxes)):
                inds = scores[i] > self.score_thresh
                boxes[i], scores[i], labels[i] = boxes[i][inds], scores[i][inds], labels[i][inds]
                keep = box_ops.batched_nms(boxes[i], scores[i], labels[i], self.nms_thresh)
                keep = keep[:self.detections_per_img]
                boxes[i], scores[i], labels[i] = boxes[i][keep], scores[i][keep], labels[i][keep]
            result = [{"boxes": b, "labels": l, "scores": s} for b, s, l in zip(boxes, scores, labels)]

        return result, losses

    return weighted_forward


class FasterRCNNLayoutDetector(nn.Module):
    """Faster R-CNN 基线（torchvision）。

    transform 未指定 min_size/max_size，走 torchvision 默认的 min_size=800,
    max_size=1333；本数据集图片是 1025×1025，因此实际输入为 800×800。
    """

    def __init__(self, num_classes=None, pretrained=True, detections_per_img=1500,
                 nms_thresh=None, backbone='resnet50'):
        super().__init__()
        self.num_classes = num_classes or NUM_CLASSES
        self.backbone_name = backbone

        backbone_with_fpn = _build_backbone(backbone, pretrained)

        from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.ops import MultiScaleRoIAlign

        anchor_gen = AnchorGenerator(
            sizes=((32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

        roi_pool_size = 14 if backbone == 'swin_t' else 7
        roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=roi_pool_size, sampling_ratio=2)

        model = FasterRCNN(
            backbone=backbone_with_fpn,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_gen,
            rpn_pre_nms_top_n_train=8000,
            rpn_pre_nms_top_n_test=6000,
            rpn_post_nms_top_n_train=4000,
            rpn_post_nms_top_n_test=3000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.5,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=512,
            rpn_positive_fraction=0.5,
            box_roi_pool=roi_pool,
            box_detections_per_img=detections_per_img,
            box_nms_thresh=nms_thresh or 0.1,
            box_score_thresh=0.01,
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        self.backbone = model

        self.register_buffer('class_weights', torch.tensor(CLASS_WEIGHTS, dtype=torch.float32))

        import types
        self.backbone.roi_heads.forward = types.MethodType(
            _make_weighted_forward(self.backbone.roi_heads.forward, self.class_weights),
            self.backbone.roi_heads
        )

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.backbone(images, targets)
        else:
            return self.backbone(images)


# ==================== 统一入口 ====================

def LayoutDetector(num_classes=None, pretrained=True, detections_per_img=1500,
                   nms_thresh=None, backbone='resnet50', detector='rtdetr',
                   num_queries=None, model_name_or_path=None, img_size=None):
    """
    工厂函数：根据 detector 参数选择模型
    detector='rtdetr' → RT-DETR（主模型，需 COCO+Objects365 预训练权重）
    detector='frcnn'  → Faster R-CNN（基线）
    """
    if detector == 'rtdetr':
        return RTDetrLayoutDetector(
            num_classes=num_classes, pretrained=pretrained, num_queries=num_queries,
            model_name_or_path=model_name_or_path, img_size=img_size,
        )
    return FasterRCNNLayoutDetector(
        num_classes=num_classes, pretrained=pretrained,
        detections_per_img=detections_per_img,
        nms_thresh=nms_thresh, backbone=backbone,
    )


if __name__ == "__main__":
    import sys
    det = sys.argv[1] if len(sys.argv) > 1 else 'frcnn'
    if det == 'rtdetr':
        m = RTDetrLayoutDetector(pretrained=False,
                                 model_name_or_path=RTDETR_MODEL_NAME)
        n_backbone = sum(p.numel() for n, p in m.named_parameters() if 'backbone' in n)
        n_total = sum(p.numel() for p in m.parameters())
        print(f"rtdetr: {n_total/1e6:.1f}M params (backbone {n_backbone/1e6:.1f}M), "
              f"num_labels={m.hf_model.config.num_labels}, queries={m.num_queries}")
    else:
        m = LayoutDetector(pretrained=False, detector='frcnn')
        print(f"frcnn: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
