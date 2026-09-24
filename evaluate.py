"""
文档版面分析评估脚本
使用 COCO API 计算 mAP@0.5、mAP@0.5:0.95、各类别 AP

推理只跑一遍（不设阈值），然后对多个置信度阈值分别过滤评估 ——
RT-DETR 的分数分布与 Faster R-CNN 不同，固定 0.5 可能滤掉真阳性。
"""
import os
import sys
import json
import argparse

import torch
from tqdm import tqdm

from config import (
    LAYOUT_CLASSES, DETECTION_SCORE_THRESH, EVAL_SCORE_THRESHOLDS,
    RTDETR_IMG_SIZE, RTDETR_NUM_QUERIES,
)
from model import LayoutDetector, RTDetrLayoutDetector
from dataset import get_val_loader, get_test_loader


@torch.no_grad()
def generate_predictions(model, loader, device, detector='frcnn'):
    """对数据集推理，生成 COCO 格式预测（不做阈值过滤，阈值在评估阶段再扫）

    返回 (predictions, gt_annotations)；predictions 里 category_id 已是 1-based。
    """
    model.eval()
    predictions = []
    gt_annotations = []
    is_rtdetr = detector == 'rtdetr'

    pbar = tqdm(loader, desc='推理中', ncols=120)
    for batch in pbar:
        if is_rtdetr:
            pixel_values = batch['pixel_values'].to(device)
            targets_gpu = [{k: v.to(device) for k, v in t.items()}
                           for t in batch['targets']]
            outputs = model(pixel_values, targets_gpu)   # 传 targets 以拿 orig_size 反缩放
            targets = batch['targets']
        else:
            images, targets = batch
            outputs = model([img.to(device) for img in images])

        for output, target in zip(outputs, targets):
            img_id = target['image_id'].item()
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                # 丢掉退化框：面积为 0 的框会让 COCOeval 的 IoU 算出 0/0，
                # 进而污染 precision 数组（这也是它提前返回、stats 为空的嫌疑之一）
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                predictions.append({
                    'id': len(predictions),
                    'image_id': img_id,
                    'category_id': int(label),   # 1-based，与 LAYOUT_CLASSES / GT 一致
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score': float(score),
                })

            # 收集 GT
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            for box, label in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box
                gt_annotations.append({
                    'id': len(gt_annotations),
                    'image_id': img_id,
                    'category_id': int(label),   # 1-based
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'area': float((x2 - x1) * (y2 - y1)),
                    'iscrowd': 0,
                })

    return predictions, gt_annotations


def _per_class_ap(coco_eval, categories):
    """从 COCOeval 里取 IoU=0.5 / area=all / maxDets=100 的逐类 AP。

    eval['precision'] 形状 [T, R, K, A, M]，K 轴顺序 = sorted(catIds)，
    与 categories（已按 id 升序）一一对应。
    """
    p = coco_eval.eval['precision'][0, :, :, 0, 2]     # [R, K]，-1 表示无有效点
    out = {}
    for i, cat in enumerate(categories):
        col = p[:, i]
        valid = col > -1
        out[cat['name']] = float(col[valid].mean()) if valid.any() else 0.0
    return out


def evaluate_coco(predictions, gt_annotations, categories, verbose=False):
    """使用 pycocotools 评估"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # 构建 COCO GT（包含所有图片，包括无 GT 的图）
    coco_gt = COCO()
    pred_img_ids = set(a['image_id'] for a in predictions)
    gt_img_ids = set(a['image_id'] for a in gt_annotations)
    all_img_ids = sorted(pred_img_ids | gt_img_ids)
    coco_gt.dataset = {
        'images': [{'id': i} for i in all_img_ids],
        'annotations': gt_annotations,
        'categories': categories,
    }
    coco_gt.createIndex()

    if not predictions:
        print("警告: 没有预测结果!")
        return {}

    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()

    # ⚠ 必须调 summarize()：这个版本的 pycocotools 里 stats 是由 summarize() 填充的，
    #   单靠 accumulate() 它会是空的 list（用合成样本验证过：必然 mAP=1.0 的输入
    #   也会得到 stats 长度 0）。经典实现里 accumulate() 就会写 stats，所以这一点
    #   很容易踩坑。它的表格输出我们用 redirect_stdout 压掉 —— 下面的逐类 AP 更有用。
    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO() if not verbose else sys.stdout):
        coco_eval.summarize()

    # 兜底：万一某个版本的 summarize() 也没填 stats，把原因打出来而不是崩在 stats[0]。
    # 直接读 stats[0] 会报 IndexError: list index out of range，看不出原因，
    # 所以这里把 pycocotools 实际看到的参数打出来。
    # 注意用 len() 判空而不是 `if not stats` —— stats 是 numpy 数组时，
    # `not array_12元素` 会抛 "truth value of an array ... is ambiguous"。
    stats = getattr(coco_eval, 'stats', None)
    if stats is None or len(stats) == 0:
        import pycocotools
        p = coco_eval.params
        ev = getattr(coco_eval, 'evalImgs', None)
        print("\n警告: COCOeval.stats 为空，本阈值无法得到指标。诊断信息:")
        print(f"  pycocotools = {getattr(pycocotools, '__version__', '?')}")
        print(f"  GT    : {len(coco_gt.dataset['images'])} 图 / "
              f"{len(gt_annotations)} 标注 / {len(coco_gt.dataset['categories'])} 类")
        print(f"  预测  : {len(predictions)} 个")
        print(f"  参数  : imgIds={len(p.imgIds)} catIds={len(p.catIds)} "
              f"iouThrs={len(p.iouThrs)} recThrs={len(p.recThrs)} "
              f"areaRng={p.areaRng} maxDets={p.maxDets}")
        print(f"  evalImgs={0 if not ev else len(ev)}"
              + (f"  非 None 比例={sum(x is not None for x in ev) / len(ev):.2f}" if ev else ""))
        return {}

    return {
        'mAP@0.5:0.95': float(coco_eval.stats[0]),
        'mAP@0.5': float(coco_eval.stats[1]),
        'mAP@0.75': float(coco_eval.stats[2]),
        'mAP@0.5_small': float(coco_eval.stats[3]),
        'mAP@0.5_medium': float(coco_eval.stats[4]),
        'mAP@0.5_large': float(coco_eval.stats[5]),
        'per_class_ap@0.5': _per_class_ap(coco_eval, categories),
        'num_predictions': len(predictions),
    }


def sweep_thresholds(predictions, gt_annotations, categories, thresholds):
    """在多个置信度阈值上评估，返回 {threshold: results}"""
    out = {}
    for t in thresholds:
        filtered = [p for p in predictions if p['score'] >= t]
        print(f"\n--- score_threshold = {t} ({len(filtered)} 个预测) ---")
        out[t] = evaluate_coco(filtered, gt_annotations, categories)
    return out


def main(args):
    if not os.path.exists(args.weights):
        print(f"错误: 权重文件不存在: {args.weights}")
        sys.exit(1)

    print(f"加载模型: {args.weights}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 两种权重形态都支持：
    #   - 目录：ModelScope 上发布的原生 transformers 格式（config.json + model.safetensors）
    #   - .pth：训练时保存的 checkpoint
    if os.path.isdir(args.weights):
        _check_hf_version_for_native_weights()
        from transformers import RTDetrConfig
        cfg = RTDetrConfig.from_pretrained(args.weights)
        detector, backbone = 'rtdetr', 'resnet50'
        img_size = args.img_size
        if img_size is None:
            pc = os.path.join(args.weights, 'preprocessor_config.json')
            if os.path.exists(pc):
                with open(pc, encoding='utf-8') as f:
                    img_size = (json.load(f).get('size') or {}).get('height')
        print(f"detector={detector} (原生 HF 目录) num_queries={cfg.num_queries} "
              f"img_size={img_size}")
        model = RTDetrLayoutDetector(pretrained=True, num_queries=cfg.num_queries,
                                     model_name_or_path=args.weights,
                                     img_size=img_size).to(device)
        processor = model.processor
        run_dir = os.path.dirname(os.path.abspath(args.weights))
        model.eval()
        return _run_eval(args, model, processor, img_size, detector, device, run_dir)

    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    run_dir = os.path.dirname(args.weights)

    # checkpoint 里记录了训练时的 detector，优先用它（避免命令行忘了传 --detector）
    detector = args.detector or checkpoint.get('detector') or 'frcnn'
    backbone = args.backbone or checkpoint.get('backbone') or 'resnet50'
    print(f"detector={detector} backbone={backbone}")

    if detector == 'rtdetr':
        img_size = args.img_size or checkpoint.get('img_size') or RTDETR_IMG_SIZE
        num_queries = checkpoint.get('num_queries') or RTDETR_NUM_QUERIES
        # 训练时已把 HF config + processor 存进 run 目录 → 不需要联网
        src = run_dir if os.path.exists(os.path.join(run_dir, 'config.json')) \
            else checkpoint.get('model_name_or_path')
        model = RTDetrLayoutDetector(pretrained=False, num_queries=num_queries,
                                     model_name_or_path=src, img_size=img_size).to(device)
        processor = model.processor
    else:
        model = LayoutDetector(backbone=backbone, detector='frcnn',
                               pretrained=False).to(device)
        processor, img_size = None, None

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"epoch: {checkpoint.get('epoch', 'N/A')}, "
          f"mAP@0.5: {checkpoint.get('mAP', 'N/A')}")

    return _run_eval(args, model, processor, img_size, detector, device, run_dir)


def _check_hf_version_for_native_weights():
    """加载原生 HF 权重前检查 transformers 版本。

    5.x 重构了 RT-DETR 的内部结构、键名全变了，加载 4.49 存的权重会大量 missing
    且**不会报错**（静默随机初始化），结果就是模型输出全是垃圾却查不出原因。
    所以这里直接把版本卡住。
    """
    import transformers
    major = int(transformers.__version__.split('.')[0])
    if major >= 5:
        raise SystemExit(
            f"加载原生 HF 权重需要 transformers 4.x（当前 {transformers.__version__}）。\n"
            "  transformers 5.x 改了 RT-DETR 的内部键名，会静默加载失败。\n"
            "  请执行: pip install 'transformers>=4.48,<4.50'"
        )


def _run_eval(args, model, processor, img_size, detector, device, run_dir):
    """两种权重形态共用的评估流程（推理一遍 → 阈值扫描 → 出报告）"""
    # 数据
    loaders = (get_test_loader if args.split == 'test' else get_val_loader)
    loader = loaders(batch_size=args.batch_size, detector=detector,
                     processor=processor, img_size=img_size)

    # 推理（一遍）
    predictions, gt_annotations = generate_predictions(model, loader, device, detector)
    print(f"\n预测: {len(predictions)} 个检测, GT: {len(gt_annotations)} 个标注")

    categories = [{'id': k, 'name': v} for k, v in sorted(LAYOUT_CLASSES.items())]

    # 阈值扫描
    thresholds = sorted(set([args.score_threshold] + EVAL_SCORE_THRESHOLDS))
    sweep = sweep_thresholds(predictions, gt_annotations, categories, thresholds)

    # 选 mAP@0.5 最高的阈值作为主结果
    best_t = max((t for t, r in sweep.items() if r), key=lambda t: sweep[t]['mAP@0.5'])
    results = {
        'split': args.split,
        'weights': args.weights,
        'detector': detector,
        'best_score_threshold': best_t,
        'best': sweep[best_t],
        'sweep': {str(t): {k: v for k, v in r.items() if k != 'per_class_ap@0.5'}
                  for t, r in sweep.items()},
    }

    # 打印逐类 AP（DocLayNet 的均值被 Footnote/Title 这类稀有类拖累，必须逐类看）
    print(f"\n{'=' * 68}")
    print(f"最佳阈值 = {best_t}")
    b = sweep[best_t]
    print(f"  mAP@0.5      = {b['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95 = {b['mAP@0.5:0.95']:.4f}")
    print(f"  mAP@0.75     = {b['mAP@0.75']:.4f}")
    print("  per-class AP@0.5:")
    for name, ap in b['per_class_ap@0.5'].items():
        print(f"    {name:16s} {ap:.4f}")
    print("=" * 68)

    # 文件名带上 checkpoint 名字，否则 best.pth / latest.pth 两次评估会互相覆盖
    stem = os.path.splitext(os.path.basename(args.weights))[0]
    output_path = os.path.join(run_dir, f'eval_results_{args.split}_{stem}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存到: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--backbone", type=str, default=None, choices=["resnet50", "swin_t"])
    parser.add_argument("--detector", type=str, default=None, choices=["rtdetr", "frcnn"],
                        help="默认从 checkpoint 的元信息读取")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--score_threshold", type=float, default=DETECTION_SCORE_THRESH)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
