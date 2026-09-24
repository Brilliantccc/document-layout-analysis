"""
文档版面分析训练脚本

主模型: RT-DETR (HuggingFace transformers)
基线:   Faster R-CNN (torchvision)

两种检测器的差异被收敛到 params['detector'] 的几个分支里，训练骨架（LR 调度、
checkpoint 约定、断点续训、早停、AMP、分组 LR）是共用的。
"""
import os
import time
import argparse
from datetime import datetime

import random
import numpy as np
import torch
from tqdm import tqdm

# 固定随机种子（可复现）
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)

# CUDA 加速全局设置
torch.backends.cudnn.benchmark = True          # 自动选最快卷积算法
torch.backends.cuda.matmul.allow_tf32 = True   # TF32 加速矩阵乘法（Ampere+）
torch.backends.cudnn.allow_tf32 = True         # TF32 加速卷积（Ampere+）
# Flash Attention（PyTorch 2.1+ 可用，Ampere+ GPU）
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
except AttributeError:
    pass  # PyTorch < 2.0 不支持

from config import (
    DEVICE, LAYOUT_CLASSES, DETECTOR, BACKBONE,
    MODEL_DIR, resolve_train_params,
)
from model import LayoutDetector
from dataset import (
    get_train_loader, get_val_loader, get_test_loader, build_rtdetr_processor,
)
from utils import AverageMeter


# ==================== 学习率调度 ====================

def get_lr(epoch, base_lr, warmup_epochs, decay_steps=None):
    """epoch 级 LR 调度（线性 warmup → constant → 阶梯衰减），FRCNN 用
    decay_steps: [(epoch, factor), ...]
    """
    if decay_steps is None:
        decay_steps = []
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    lr = base_lr
    for decay_epoch, decay_factor in decay_steps:
        if epoch >= decay_epoch:
            lr *= decay_factor
    return lr


def get_lr_step(global_step, base_lr, warmup_steps=0, decay_steps=None):
    """step 级 LR 调度（线性 warmup → constant → 阶梯衰减），RT-DETR 用
    decay_steps: [(step, factor), ...]
    """
    if decay_steps is None:
        decay_steps = []
    if global_step < warmup_steps:
        return base_lr * global_step / warmup_steps
    lr = base_lr
    for decay_step, decay_factor in decay_steps:
        if global_step >= decay_step:
            lr *= decay_factor
    return lr


def set_optimizer_lr(optimizer, lr):
    """按各参数组自己的 lr_scale 分配 LR（RT-DETR 的 backbone 组更小）。
    lr_scale 在构建优化器时打标，见 train()。
    """
    for pg in optimizer.param_groups:
        pg['lr'] = lr * pg.get('lr_scale', 1.0)


def current_lr(optimizer):
    """当前生效的学习率（多组时取最大的那个，即 transformer/head 的 LR）"""
    return max(pg['lr'] for pg in optimizer.param_groups)


# ==================== 训练 & 验证 ====================

def _use_cuda(device):
    """DEVICE 是字符串（'cuda' / 'cpu'），不能直接用 device.type"""
    return str(device).startswith('cuda')


def _unpack_batch(batch, is_rtdetr, device):
    """把 DataLoader 的一个 batch 搬到 GPU。

    RT-DETR 的 collate 返回 dict（pixel_values + targets），
    FRCNN 的 collate 返回 tuple（图片列表 + target 列表）。
    """
    if is_rtdetr:
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                   for t in batch['targets']]
        return pixel_values, None, targets
    images, targets = batch
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return None, images, targets


def train_one_epoch(model, loader, optimizer, device, epoch, scaler, params,
                    global_step=0):
    model.train()
    epochs = params['epochs']
    steps_per_epoch = len(loader)
    is_rtdetr = params['detector'] == 'rtdetr'
    step_lr = params['lr_schedule'] == 'step'
    amp_dtype = torch.bfloat16 if params['amp_dtype'] == 'bf16' else torch.float16
    grad_clip = params['grad_clip_norm']
    # 梯度累积：显存放不下大 batch 时用它换有效 batch（RT-DETR 在 800px 下
    # batch=8 就会 OOM，故 config 里给的是 batch=4 + accum=2）
    accum = max(1, int(params.get('accum_steps', 1)))
    use_cuda = _use_cuda(device)

    # step 级调度：衰减点从 epoch 换算成 step
    decay_steps = [(e * steps_per_epoch, f) for e, f in params['lr_decay_steps']]

    loss_meters = {'total': AverageMeter()}
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{epochs} [Train]',
                ncols=140, bar_format='{l_bar}{bar:20}{r_bar}')

    optimizer.zero_grad(set_to_none=True)
    skipped = 0
    for step, batch in enumerate(pbar):
        # step 级 LR（RT-DETR）；FRCNN 在 train() 里按 epoch 设
        if step_lr:
            lr = get_lr_step(global_step, base_lr=params['base_lr'],
                             warmup_steps=params['warmup_steps'],
                             decay_steps=decay_steps)
            set_optimizer_lr(optimizer, lr)

        pixel_values, images, targets = _unpack_batch(batch, is_rtdetr, device)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_cuda):
            loss_dict = model(pixel_values, targets) if is_rtdetr else model(images, targets)
            total_loss = sum(loss_dict.values()) / accum

        # 数值不稳时跳过该 batch，避免污染参数（bf16/fp16 下偶发 inf/nan）
        if not torch.isfinite(total_loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            continue

        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (step + 1) % accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1

        n = pixel_values.shape[0] if is_rtdetr else len(images)
        loss_meters['total'].update(total_loss.item() * accum, n)
        for k, v in loss_dict.items():
            if k not in loss_meters:
                loss_meters[k] = AverageMeter()
            loss_meters[k].update(v.item(), n)
        pbar.set_postfix({'loss': f'{loss_meters["total"].avg:.4f}',
                          'lr': f'{current_lr(optimizer):.2e}',
                          **({'skip': skipped} if skipped else {})})

    return {k: m.avg for k, m in loss_meters.items()}, global_step


@torch.no_grad()
def validate(model, loader, device, epoch, epochs=None, params=None):
    """验证：算 mAP（+ FRCNN 额外的 val loss）。

    RT-DETR 不算 val loss：它的训练 forward 要对 6 层 decoder 各跑一次
    Hungarian matching（含 GPU→CPU 同步），代价接近训练一步；而模型选择只看 mAP。
    """
    is_rtdetr = params is not None and params['detector'] == 'rtdetr'
    amp_dtype = torch.bfloat16 if (params and params['amp_dtype'] == 'bf16') else torch.float16
    use_cuda = _use_cuda(device)

    all_preds, all_gts = [], []
    loss_meters = {'total': AverageMeter()}

    # 必须切 eval：RT-DETR 的 wrapper 在 train 模式下返回 loss 字典，
    # 只有 eval 模式才返回预测。漏掉这一行会让 zip(outputs, gts) 迭代到字典的
    # 字符串 key，报 "string indices must be integers" —— 而且因为 train() 里
    # train_one_epoch 结束时模型停在 train 模式，这个错会在每轮验证必现。
    model.eval()

    pbar = tqdm(loader, desc=f'Epoch {epoch}/{epochs or epoch} [Val]',
                ncols=140, bar_format='{l_bar}{bar:20}{r_bar}')
    for batch in pbar:
        if is_rtdetr:
            pixel_values = batch['pixel_values'].to(device)
            targets_gpu = [{k: v.to(device) for k, v in t.items()}
                           for t in batch['targets']]
            outputs = model(pixel_values, targets_gpu)   # 传 targets 以拿 orig_size 反缩放
            gts = batch['targets']
            if not (isinstance(outputs, list) and outputs
                    and isinstance(outputs[0], dict) and 'boxes' in outputs[0]):
                raise RuntimeError(
                    "validate() 拿到的是 loss 而不是预测 —— 说明模型没切到 eval() 模式。"
                    "检查 wrapper 的 self.training 分支")
        else:
            images, targets = batch
            images_gpu = [img.to(device) for img in images]
            targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # val loss：GeneralizedRCNN 只在 training=True 时返回 loss
            was_frozen = (not next(model.backbone.parameters()).requires_grad
                          if hasattr(model, 'backbone') else False)
            if was_frozen:
                for p in model.backbone.parameters():
                    p.requires_grad = True
            model.train()
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_cuda):
                loss_dict = model(images_gpu, targets_gpu)
            if was_frozen:
                for p in model.backbone.parameters():
                    p.requires_grad = False
            loss_meters['total'].update(sum(loss_dict.values()).item(), len(images_gpu))
            for k, v in loss_dict.items():
                if k not in loss_meters:
                    loss_meters[k] = AverageMeter()
                loss_meters[k].update(v.item(), len(images_gpu))

            model.eval()
            outputs = model(images_gpu)
            gts = targets

        for out, tgt in zip(outputs, gts):
            all_preds.append({'boxes': out['boxes'].float().cpu(),
                              'labels': out['labels'].cpu(),
                              'scores': out['scores'].float().cpu()})
            all_gts.append({'boxes': tgt['boxes'].float().cpu(),
                            'labels': tgt['labels'].cpu()})

    metrics = compute_map(all_preds, all_gts)
    if is_rtdetr:
        return metrics
    out = {k: m.avg for k, m in loss_meters.items()}
    out.update(metrics)
    return out


def compute_map(predictions, ground_truths, iou_threshold=0.5):
    """
    计算 mAP@0.5、P、R、F1、各类别 AP
    参考项目2 compute_fast_map，使用 IoU 缓存加速

    predictions: [{'boxes' xyxy 像素, 'labels' 1-based, 'scores'}]
    ground_truths: [{'boxes' xyxy 像素, 'labels' 1-based}]
    """
    from utils import compute_iou
    import numpy as np

    all_classes = set()
    for gt in ground_truths:
        if len(gt['labels']) > 0:
            all_classes.update(gt['labels'].cpu().tolist())

    aps = []
    per_class_ap = {}
    total_tp = total_fp = total_fn = 0

    for cls in sorted(all_classes):
        cls_pred_by_img = {}
        cls_gt_by_img = {}

        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            p_mask = pred['labels'].cpu() == cls
            if p_mask.any():
                cls_pred_by_img[img_idx] = {
                    'boxes': pred['boxes'].cpu()[p_mask],
                    'scores': pred['scores'].cpu()[p_mask],
                }
            g_mask = gt['labels'].cpu() == cls
            if g_mask.any():
                cls_gt_by_img[img_idx] = {'boxes': gt['boxes'].cpu()[g_mask]}

        if not cls_gt_by_img:
            continue

        # 预计算 IoU 矩阵
        iou_cache = {}
        for img_idx in cls_pred_by_img:
            if img_idx in cls_gt_by_img:
                pb = cls_pred_by_img[img_idx]['boxes']
                gb = cls_gt_by_img[img_idx]['boxes']
                if len(pb) > 0 and len(gb) > 0:
                    iou_cache[img_idx] = compute_iou(pb, gb)

        # 收集预测并按 score 排序
        all_cls_preds = []
        for img_idx, p in cls_pred_by_img.items():
            for j in range(len(p['boxes'])):
                all_cls_preds.append((img_idx, j, p['scores'][j].item()))

        if not all_cls_preds:
            total_fn += sum(len(g['boxes']) for g in cls_gt_by_img.values())
            continue

        all_cls_preds.sort(key=lambda x: x[2], reverse=True)

        # 匹配
        tp_list, fp_list = [], []
        matched_mask = {idx: torch.zeros(len(cls_gt_by_img[idx]['boxes']), dtype=torch.bool)
                        for idx in cls_gt_by_img}

        for img_idx, pred_idx, score in all_cls_preds:
            if img_idx not in cls_gt_by_img or img_idx not in iou_cache:
                tp_list.append(0); fp_list.append(1); continue

            ious_row = iou_cache[img_idx][pred_idx]
            mask = matched_mask[img_idx]
            masked_ious = ious_row.clone()
            masked_ious[mask] = -1.0
            best_iou, best_gt_idx = masked_ious.max(0)

            if best_iou.item() >= iou_threshold:
                tp_list.append(1); fp_list.append(0)
                mask[best_gt_idx.item()] = True
            else:
                tp_list.append(0); fp_list.append(1)

        # AP（全点插值）
        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        precision_arr = tp_cum / (tp_cum + fp_cum + 1e-6)
        total_gt = sum(len(g['boxes']) for g in cls_gt_by_img.values())
        recall_arr = tp_cum / (total_gt + 1e-6)

        for i in range(len(precision_arr) - 2, -1, -1):
            precision_arr[i] = max(precision_arr[i], precision_arr[i + 1])
        r = np.concatenate(([0.0], recall_arr, [1.0]))
        p = np.concatenate(([1.0], precision_arr, [0.0]))
        ap = np.sum((r[1:] - r[:-1]) * p[1:])

        aps.append(ap)
        per_class_ap[cls] = ap

        total_tp += tp_cum[-1]
        total_fp += fp_cum[-1]
        total_fn += total_gt - tp_cum[-1]

    mAP = np.mean(aps) if aps else 0.0
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_ap': per_class_ap,
    }


# 逐类 AP 的紧凑标签（截断前 4 字符会让 Page-footer/Page-header 撞名）
SHORT_CLASS_NAMES = {
    'Caption': 'Capt', 'Footnote': 'Foot', 'Formula': 'Form', 'List-item': 'List',
    'Page-footer': 'PFoot', 'Page-header': 'PHead', 'Picture': 'Pict',
    'Section-header': 'Sect', 'Table': 'Tabl', 'Text': 'Text', 'Title': 'Titl',
}


def format_per_class_ap(per_class_ap):
    """紧凑打印 per-class AP —— DocLayNet 的均值被稀有类拖累，必须逐类看"""
    return ' '.join(
        f'{SHORT_CLASS_NAMES[LAYOUT_CLASSES[c]]}:{per_class_ap.get(c, 0):.2f}'
        for c in sorted(LAYOUT_CLASSES)
    )


# ==================== 主训练流程 ====================

def save_checkpoint(path, model, optimizer, epoch, val_loss, mAP, global_step,
                    extra=None):
    """统一保存。global_step 必须一起存：step 级 LR 调度靠它续训对齐。"""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'mAP': mAP,
        'global_step': global_step,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def train(args=None):
    resume_path = args.resume if args and args.resume else None
    detector_name = args.detector if args and args.detector else DETECTOR
    # RT-DETR 的 backbone 由 HF checkpoint 决定，BACKBONE 配置无效
    if detector_name == 'rtdetr':
        backbone_name = 'resnet50'
    else:
        backbone_name = args.backbone if args and args.backbone else BACKBONE

    # 超参：按【实际的】detector/backbone 取，而不是 config.DETECTOR（见 config.resolve_train_params）
    params = resolve_train_params(detector_name, backbone_name)
    batch_size = args.batch_size if args and args.batch_size else params['batch_size']
    base_lr = args.lr if args and args.lr else params['base_lr']
    epochs = args.epochs if args and args.epochs else params['epochs']
    warmup_epochs = (args.warmup_epochs if args and args.warmup_epochs
                     else params['warmup_epochs'])
    img_size = (args.img_size if args and args.img_size else params['img_size'])
    model_name_or_path = (args.model_name_or_path if args and args.model_name_or_path
                          else params['model_name_or_path'])
    accum_steps = (args.accum_steps if args and args.accum_steps
                   else params.get('accum_steps', 1))
    # backbone LR 默认是 base_lr 的 0.1 倍（官方 recipe，针对"域相同"的微调）。
    # 自然图像 -> 文档是大幅域迁移，backbone 需要更多适配，所以做成可调。
    if args and args.backbone_lr_scale is not None:
        params['backbone_lr_scale'] = args.backbone_lr_scale
    if args and args.num_queries is not None:
        params['num_queries'] = args.num_queries
    # 衰减点跟着 epochs 走：--epochs 80 时衰减落在 epoch 48/68，而不是写死的 36/51
    if detector_name == 'rtdetr':
        from config import rtdetr_decay_steps
        params['lr_decay_steps'] = rtdetr_decay_steps(epochs)
        params['early_stop_min_epoch'] = params['lr_decay_steps'][-1][0]
    # 命令行覆盖后的值要同步回 params，否则调度仍用 config 里的基准
    params.update(base_lr=base_lr, epochs=epochs, warmup_epochs=warmup_epochs,
                  img_size=img_size, model_name_or_path=model_name_or_path,
                  batch_size=batch_size, accum_steps=accum_steps)
    is_rtdetr = detector_name == 'rtdetr'

    if resume_path:
        output_dir = os.path.dirname(resume_path)
        checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint.get('mAP', 0.0)
        # 检测器/backbone 不匹配会得到很难懂的 state_dict 报错，这里提前拦住
        ckpt_det = checkpoint.get('detector')
        if ckpt_det and ckpt_det != detector_name:
            raise SystemExit(
                f"检测器不匹配: {resume_path} 是 {ckpt_det!r} 训练的，"
                f"当前是 {detector_name!r}。请加 --detector {ckpt_det}")
        print(f"断点续训: {resume_path} (从 epoch {start_epoch} 继续)")
    else:
        checkpoint = None
        det_dir = os.path.join(MODEL_DIR, detector_name)  # runs/rtdetr/ 或 runs/frcnn/
        os.makedirs(det_dir, exist_ok=True)
        existing = [d for d in os.listdir(det_dir)
                    if os.path.isdir(os.path.join(det_dir, d)) and d.startswith('v')]
        next_ver = max([int(d[1:]) for d in existing if d[1:].isdigit()] or [0]) + 1
        output_dir = os.path.join(det_dir, f"v{next_ver}")
        os.makedirs(output_dir, exist_ok=True)
        start_epoch = 1
        best_mAP = 0.0

    print("=" * 68)
    print("文档版面分析系统 - 训练")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"版本: {output_dir}")
    print(f"设备: {DEVICE}")
    print(f"Detector: {detector_name}" + (f"  (backbone {backbone_name})" if not is_rtdetr else ""))
    if is_rtdetr:
        print(f"预训练权重: {model_name_or_path}")
        print(f"输入尺寸: {img_size}  queries={params['num_queries']}")
    print(f"Batch Size: {batch_size} x accum {accum_steps} "
          f"(有效 {batch_size * accum_steps})")
    print(f"学习率: {base_lr} (backbone ×{params['backbone_lr_scale']})  "
          f"optimizer={params['optimizer']}  wd={params['weight_decay']}")
    print(f"AMP: {params['amp_dtype']}  梯度裁剪: {params['grad_clip_norm']}")
    print(f"Epochs: {epochs}")
    if params['lr_schedule'] == 'step':
        print(f"预热: {params['warmup_steps']} steps (线性)")
        print(f"阶梯衰减: {params['lr_decay_steps']} (epoch, 倍率)")
    else:
        print(f"预热: {warmup_epochs} epochs")
        print(f"阶梯衰减: {params['lr_decay_steps']} (epoch, 倍率)")
    if params['backbone_freeze_epochs']:
        print(f"Backbone 冻结: 前 {params['backbone_freeze_epochs']} epochs")
    print(f"早停: patience={params['early_stop_patience']}, "
          f"最早 epoch {params['early_stop_min_epoch']}")
    print("=" * 68)

    # data processor：必须在 DataLoader fork worker 之前构造，供所有 loader 复用
    processor = (build_rtdetr_processor(model_name_or_path, img_size)
                 if is_rtdetr else None)
    train_subset = args.subset if args and args.subset else None
    val_subset = args.val_subset if args and args.val_subset else None

    print("\n加载数据集...")
    train_loader = get_train_loader(
        batch_size, detector=detector_name, processor=processor, img_size=img_size,
        subset=train_subset, no_aug=bool(args and args.no_aug),
        rand_crop=(args.rand_crop if args and args.rand_crop else 0.0))
    val_loader = get_val_loader(batch_size, detector=detector_name,
                                processor=processor, img_size=img_size,
                                subset=val_subset)

    # 模型：续训时优先从 run 目录读架构（保证不联网、且结构完全一致）
    arch_src, load_pretrained = model_name_or_path, True
    if resume_path and is_rtdetr and os.path.exists(os.path.join(output_dir, 'config.json')):
        arch_src, load_pretrained = output_dir, False
    elif resume_path:
        load_pretrained = False   # 权重会被 state_dict 覆盖，没必要下载预训练权重
    print("构建模型...")
    model = LayoutDetector(detector=detector_name, pretrained=load_pretrained,
                           backbone=backbone_name,
                           nms_thresh=args.nms_thresh if args else None,
                           num_queries=params['num_queries'],
                           model_name_or_path=arch_src, img_size=img_size).to(DEVICE)
    if not is_rtdetr:
        model = model.to(memory_format=torch.channels_last)  # Tensor Core 加速卷积
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {n_params:.1f}M" + ("" if is_rtdetr else " (channels_last)"))

    # 优化器
    scale = params['backbone_lr_scale']
    if is_rtdetr or params['optimizer'] == 'adamw':
        # HF 参数名 hf_model.model.backbone.* 含 'backbone' → 低 LR 组；
        # encoder/decoder/class_embed 等 → 高 LR 组。
        backbone_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (backbone_params if 'backbone' in name else other_params).append(p)
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': base_lr * scale},
            {'params': other_params, 'lr': base_lr},
        ], weight_decay=params['weight_decay'], betas=(0.9, 0.999))
        group_scales = [scale, 1.0]
        print(f"  分组 LR: backbone={base_lr * scale:.2e}, "
              f"其余={base_lr:.2e} ({len(backbone_params)}/{len(other_params)} 个参数张量)")
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=base_lr,
            momentum=0.9, weight_decay=params['weight_decay'])
        group_scales = [1.0]

    # bf16 用不上 GradScaler（指数范围与 fp32 相同）；CPU 上也没有可 scale 的
    use_amp = params['amp_dtype'] == 'fp16' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 加载 checkpoint（断点续训）
    steps_per_epoch = len(train_loader)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # global_step 必须一起恢复：step 级调度靠它定位 warmup/衰减点，
        # 从 0 重算会让 warmup 重来一遍、衰减点整体后移（旧 checkpoint 无此字段，按已训 epoch 估算）
        global_step = checkpoint.get('global_step', (start_epoch - 1) * steps_per_epoch)
        print(f"已加载模型和优化器状态 (global_step={global_step})")
    else:
        global_step = 0

    # lr_scale 在 load_state_dict 之后重新打标：
    # load_state_dict 会用 checkpoint 里的 param_groups 整体替换，丢掉这里自定义的 key
    for pg, s in zip(optimizer.param_groups, group_scales):
        pg['lr_scale'] = s
    set_optimizer_lr(optimizer, base_lr)

    # 把 HF config + processor 落盘，让 evaluate/inference 不需要连 HuggingFace
    if is_rtdetr and not resume_path:
        model.save_pretrained(output_dir, processor=processor)
        print(f"已保存模型 config / processor 到 {output_dir}")

    # 写进 checkpoint 的元信息，续训/评估时校验与重建
    meta = {'detector': detector_name, 'backbone': backbone_name,
            'img_size': img_size, 'num_queries': params['num_queries'],
            'model_name_or_path': model_name_or_path}

    # 训练
    patience_counter = 0
    print(f"\n开始训练... (steps/epoch={steps_per_epoch})")
    start_time = time.time()

    freeze_epochs = params['backbone_freeze_epochs']
    for epoch in range(start_epoch, epochs + 1):
        # Backbone 冻结/解冻
        if freeze_epochs > 0 and hasattr(model, 'backbone'):
            if epoch <= freeze_epochs:
                for p in model.backbone.parameters():
                    p.requires_grad = False
                if epoch == start_epoch or (epoch - 1) % freeze_epochs == 0:
                    print(f"  [Epoch {epoch}] Backbone 冻结中...")
            elif epoch == freeze_epochs + 1:
                for p in model.backbone.parameters():
                    p.requires_grad = True
                print(f"  [Epoch {epoch}] Backbone 解冻，全部参数参与训练")

        # FRCNN 走 epoch 级调度；RT-DETR 在 train_one_epoch 里按 step 调
        if params['lr_schedule'] == 'epoch':
            set_optimizer_lr(optimizer, get_lr(epoch - 1, base_lr, warmup_epochs,
                                               params['lr_decay_steps']))

        train_losses, global_step = train_one_epoch(model, train_loader, optimizer,
                                                    DEVICE, epoch, scaler, params,
                                                    global_step)
        val = validate(model, val_loader, DEVICE, epoch, epochs, params)

        elapsed = time.time() - start_time
        print(f"\n  Epoch {epoch}/{epochs} Summary ({elapsed/60:.1f} min cumulative):")
        loss_str = ' | '.join(f'{k}: {v:.4f}' for k, v in train_losses.items() if k != 'total')
        print(f"  Train - Loss: {train_losses['total']:.4f} | {loss_str}")
        val_loss_str = f"Loss: {val['total']:.4f} | " if 'total' in val else ""
        # ⚠ P/R/F1 是在"全部 num_queries 个预测"这个点上算的，不是工作点：
        #   每页真实只有 ~13 个框，300 个预测下 P 的理论上限就只有 13/300≈0.04。
        #   所以 P=0.04 可能是满分，别被它吓到 —— 看 mAP 和 R，权威 P/R 用 evaluate.py。
        print(f"  Val   - {val_loss_str}mAP@0.5: {val['mAP']:.4f} | "
              f"(全预测点) P: {val['precision']:.3f} R: {val['recall']:.3f} "
              f"F1: {val['f1']:.3f}")
        # 打印实际生效的 LR（不是 base_lr），阶梯衰减才看得出来
        lr_str = f"{current_lr(optimizer):.6f}"
        if len(optimizer.param_groups) > 1:
            lr_str += f"  (backbone {optimizer.param_groups[0]['lr']:.6f})"
        print(f"  LR: {lr_str}")
        # 均值会被 Footnote(0.28%)/Title(0.15%) 这类稀有类拖累，逐类看才知道真实水平
        print(f"  per-class AP: {format_per_class_ap(val['per_class_ap'])}")

        save_checkpoint(os.path.join(output_dir, "latest.pth"), model, optimizer,
                        epoch, val.get('total', 0.0), val['mAP'], global_step,
                        extra=meta)

        if val['mAP'] > best_mAP:
            best_mAP = val['mAP']
            patience_counter = 0
            save_checkpoint(os.path.join(output_dir, "best.pth"), model, optimizer,
                            epoch, val.get('total', 0.0), val['mAP'], global_step,
                            extra=meta)
            print(f"  * New best model saved! mAP@0.5: {best_mAP:.4f}")
        elif epoch >= params['early_stop_min_epoch']:
            # 最后一个衰减点之前不累计 patience，否则会在 LR 还没降下来时就早停
            patience_counter += 1

        if patience_counter >= params['early_stop_patience']:
            print(f"\n  Early stopping! No improvement for "
                  f"{params['early_stop_patience']} epochs")
            save_checkpoint(os.path.join(output_dir, "latest.pth"), model, optimizer,
                            epoch, val.get('total', 0.0), val['mAP'], global_step,
                            extra=dict(meta, early_stopped=True))
            break
        print("-" * 68)

    # 最终测试（使用 test 集）
    test_loader = get_test_loader(batch_size, detector=detector_name,
                                  processor=processor, img_size=img_size,
                                  subset=args.test_subset if args and args.test_subset else None)
    print("\n" + "=" * 68)
    print("  Final test (test set)...")
    print("=" * 68)
    model.load_state_dict(torch.load(os.path.join(output_dir, "best.pth"),
                                     map_location=DEVICE,
                                     weights_only=False)['model_state_dict'])
    test = validate(model, test_loader, DEVICE, epochs, epochs, params)
    print(f"\n  Test: mAP@0.5={test['mAP']:.4f} | P={test['precision']:.4f} | "
          f"R={test['recall']:.4f} | F1={test['f1']:.4f}")
    print(f"  Test per-class AP: {format_per_class_ap(test['per_class_ap'])}")

    total_time = time.time() - start_time
    print(f"\n  训练完成! 耗时: {total_time/3600:.2f} 小时")
    print(f"  模型保存在: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="断点续训 checkpoint 路径，如 runs/rtdetr/v1/latest.pth")
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None, choices=["resnet50", "swin_t"])
    parser.add_argument("--detector", type=str, default=None, choices=["rtdetr", "frcnn"])
    parser.add_argument("--img_size", type=int, default=None,
                        help="RT-DETR 的输入尺寸，默认取 config.RTDETR_IMG_SIZE")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="RT-DETR 预训练权重或本地 run 目录")
    parser.add_argument("--accum_steps", type=int, default=None,
                        help="梯度累积步数（显存不够时用来换有效 batch）")
    parser.add_argument("--backbone_lr_scale", type=float, default=None,
                        help="backbone 的 LR 倍数（默认 0.1）。域差异大时可提到 1.0")
    parser.add_argument("--num_queries", type=int, default=None,
                        help="解码器 query 数（默认 300）。去重后每页真实只有 ~13 个框，"
                             "300 里绝大多数是浪费；降到 100 可省解码算力并可能提升 precision")
    parser.add_argument("--nms_thresh", type=float, default=None,
                        help="仅 FRCNN。NMS 的 IoU 阈值，默认 0.1（比 torchvision 的 0.5 激进"
                             "得多，版面块相邻时可能压掉真阳性）。recall 异常低时试 0.3~0.5")
    # ---- 冒烟/调试用 ----
    parser.add_argument("--subset", type=int, default=None,
                        help="只用前 N 张训练图（过拟合 8 张图的 pipeline 验证）")
    parser.add_argument("--val_subset", type=int, default=None,
                        help="只用前 N 张验证图，加速冒烟测试")
    parser.add_argument("--test_subset", type=int, default=None,
                        help="只用前 N 张测试图，加速冒烟测试")
    parser.add_argument("--no_aug", action="store_true", help="关闭数据增强")
    parser.add_argument("--rand_crop", type=float, default=0.0,
                        help="random resized crop 概率，默认 0（会切断文本行）")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
