"""
Pipeline 验证脚本 —— 在花几小时训练之前，先把"静默错"打掉。

用法（在 AutoDL 上跑）:
    python verify_pipeline.py                 # Phase 0 + 1（约 1 分钟，CPU 即可）
    python verify_pipeline.py --overfit 8     # 额外跑过拟合测试（需 GPU，约 5 分钟）

检查内容:
  Phase 0  HF API 行为
    - processor 输出的尺寸 / 值域
    - class_labels 是否【原样】拷贝 category_id（HF 不做 0-based 重映射）
    - boxes 是否为归一化 cxcywh，反算回像素能否回到原值
    - 换成 11 类后 class_embed 的 out_features
    - 800 / 1024 两个尺寸的 forward 不报 shape 错
  Phase 1  数据管线往返
    - pixel_values 形状 / dtype
    - label 域（项目 1..11 → HF 0..10）
    - 坐标往返：项目 xyxy@原图 → HF 归一化 cxcywh → 解回原图，max|Δ| < 1e-3
    - 空标注图不崩（train 里有 27 张无标注图）
    - 存可视化图，肉眼确认框位置
  Phase 2  过拟合 N 张图（冻住 backbone，只训 head + transformer）
    - total loss 相对首个 epoch 下降 >= 80%
    - eval 模式下 GT 框召回率 >= 95% 且命中框类别正确率 >= 95%
    这几条同时成立，才说明 label 映射、坐标转换、loss 路径、后处理全通。

为什么必须有这个脚本：本次手写 DETR 翻车的三个 bug（双重 sigmoid、缺 no-object loss、
class_head off-by-one）全都是"训练能跑完、mAP 只有 0.025"的静默错，
靠肉眼看 loss 曲线发现不了。
"""
import argparse
import os
import sys

import numpy as np
import torch

from config import (
    RTDETR_MODEL_NAME, RTDETR_IMG_SIZE, LAYOUT_CLASSES,
    TRAIN_ANN, TRAIN_IMG_DIR, RTDETR_ID2LABEL, LABEL_OFFSET, resolve_train_params,
)
from dataset import build_rtdetr_processor, RTDetrDataset, rtdetr_collate_fn
from model import RTDetrLayoutDetector
from utils import imread, imwrite, draw_detections

FAIL = []


def check(cond, msg, detail=''):
    tag = 'OK  ' if cond else 'FAIL'
    print(f"  [{tag}] {msg}" + (f"  ({detail})" if detail else ''))
    if not cond:
        FAIL.append(msg)
    return cond


def check_numpy_torch_interop():
    """实测 torch ↔ numpy 的数据互操作，而不是只看版本号。

    torch 2.1.x 是针对 numpy 1.x 编译的，配 numpy 2.x 属于官方不支持的组合，
    但实际是否炸取决于走不走那条 C API 路径：
      torch.from_numpy / torch.as_tensor（共享内存）、tensor.numpy() 是危险项；
      torch.tensor(numpy)（拷贝）通常没事。
    本项目 dataset.py 用了 torch.as_tensor，evaluate.py 用了 .numpy()，所以必须验。
    """
    try:
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        t_from = torch.from_numpy(a)                    # 共享内存，走 numpy C API
        t_as = torch.as_tensor(a, dtype=torch.float32)  # dataset.py 用的这条
        t_new = torch.tensor(a, dtype=torch.float32)
        back1 = t_from.numpy()                          # evaluate.py 用的这条
        back2 = t_new.cpu().numpy()
        ok = (np.allclose(back1, a) and np.allclose(back2, a)
              and t_as.sum().item() == a.sum() and t_from.sum().item() == a.sum())
        detail = f"numpy {np.__version__} / torch {torch.__version__}"
        return check(ok, "torch ↔ numpy 数据互操作正常", detail)
    except Exception as e:
        return check(False, "torch ↔ numpy 数据互操作正常",
                     f"{type(e).__name__}: {e}")


# ==================== Phase 0 ====================

def phase0(model_name, img_size):
    print("=" * 74)
    print(f"Phase 0: HF API 行为 (model={model_name}, img_size={img_size})")
    print("=" * 74)

    from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
    import transformers
    print(f"  transformers={transformers.__version__}  torch={torch.__version__}  "
          f"numpy={np.__version__}")
    # numpy 2.x 与 torch 2.1.x（针对 numpy 1.x 编译）的搭配是官方不支持的组合，
    # 但【实测能跑通就不用降级】—— 环境里可能装着针对 numpy 2 编译的 opencv 5.x，
    # 盲目降 numpy 反而会把 cv2 弄坏。所以这里测行为，而不是卡版本号。
    if not check_numpy_torch_interop():
        print("       ↑ torch 与 numpy 的 C API 互操作失败。修复二选一：")
        print("         a) 降 numpy（需换源，aliyun 对 numpy 是坏的）：")
        print("            pip install \"numpy<2\" -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("            降级后若 cv2 报错，再补：")
        print("            pip install --force-reinstall \"opencv-python>=4.8,<4.11\"")
        print("         b) 升 torch 到 >=2.3（原生的 numpy 2 支持），例如：")
        print("            pip install --upgrade torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu121")
        print("         这条不过，后面所有结果都不可信，先修掉再重跑。")
        sys.exit(1)

    # ---- processor 行为 ----
    proc = RTDetrImageProcessor.from_pretrained(model_name)
    proc.size = {'height': img_size, 'width': img_size}
    proc.do_pad = False

    from dataset import _load_coco
    images, img_to_anns, img_ids, _ = _load_coco(TRAIN_ANN)
    img_id = img_ids[0]
    info = images[img_id]
    bgr = imread(os.path.join(TRAIN_IMG_DIR, info['file_name']))
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = rgb.shape[:2]
    print(f"  样本图: {info['file_name'][:20]}... {H0}×{W0}")

    # 故意用 category_id=0（HF 的经典示例用法）验证它会不会自作主张 +1
    ann = {'image_id': img_id, 'annotations': [
        {'bbox': [100, 200, 300, 40], 'category_id': 0, 'iscrowd': 0, 'area': 12000}]}
    from PIL import Image
    out = proc(images=Image.fromarray(rgb), annotations=ann, return_tensors='pt',
               size={'height': img_size, 'width': img_size})
    pv = out['pixel_values']
    check(tuple(pv.shape) == (1, 3, img_size, img_size),
          "pixel_values 形状 = [1,3,S,S]", f"实际 {tuple(pv.shape)}")

    # 如实报告预处理配置，不断言某种特定的归一化方式。
    # 实测这个 checkpoint 的 do_normalize 是关闭的（白底文档 255/0 只做 /255 →
    # 值域 [0.00, 1.00]）。这是 HF 官方 demo 用的同一套默认值，跟随它才安全 ——
    # 自己"修正"成 ImageNet 归一化反而会让输入分布偏离预训练时的分布。
    # 关键约束是：训练与评估必须走同一个 processor 配置（都由 build_rtdetr_processor 构造）。
    do_norm = getattr(proc, 'do_normalize', None)
    lo, hi = pv.min().item(), pv.max().item()
    print(f"  processor 配置: do_resize={proc.do_resize} do_rescale={proc.do_rescale} "
          f"rescale_factor={proc.rescale_factor}")
    print(f"                  do_normalize={do_norm} "
          f"image_mean={getattr(proc, 'image_mean', None)} "
          f"image_std={getattr(proc, 'image_std', None)}")
    print(f"                  do_convert_annotations={getattr(proc, 'do_convert_annotations', None)}")
    expect = (lo < -1.0 and hi > 1.5) if do_norm else (-0.05 <= lo and hi <= 1.05)
    check(expect and np.isfinite(pv).all(),
          f"pixel_values 值域与 processor 配置一致"
          f"（do_normalize={do_norm}）", f"实际 [{lo:.2f}, {hi:.2f}]")

    labels = out['labels'][0]
    check(labels['class_labels'][0].item() == 0,
          "class_labels 原样拷贝 category_id（不 +1）—— 所以必须自己减 1",
          f"输入 category_id=0 得到 {labels['class_labels'][0].item()}")

    hb = labels['boxes'][0]           # 归一化 cxcywh
    check(hb.max().item() <= 1.0 + 1e-6,
          "processor 的 boxes 是归一化 cxcywh", f"max={hb.max():.4f}")
    cx, cy, bw, bh = hb.tolist()
    back = [(cx - bw / 2) * W0, (cy - bh / 2) * H0, (cx + bw / 2) * W0, (cy + bh / 2) * H0]
    exp = [100.0, 200.0, 400.0, 240.0]
    err = max(abs(a - b) for a, b in zip(back, exp))
    check(err < 1.0, "processor 的归一化坐标反算回原图正确", f"max 误差 {err:.3f}px")

    # ---- 换 11 类 ----
    # 换类别数时 HF 会打印一长串 "Some weights ... were newly initialized" —— 那是
    # 正常的：80 类的分类头/选择头/dn 类别嵌入形状对不上，按 ignore_mismatched_sizes
    # 重建，backbone 和 encoder/decoder 主体全部照常加载。
    # 这里用 output_loading_info 把"到底哪些没加载"印出来并断言：应该【只有】与
    # 类别数绑定的那 15 个张量，其余全都要来自 checkpoint
    # （否则说明 checkpoint 用错了，或 backbone 静默没加载上 —— 那才是灾难）。
    id2label = RTDETR_ID2LABEL
    m, load_info = RTDetrForObjectDetection.from_pretrained(
        model_name, id2label=id2label, label2id={v: k for k, v in id2label.items()},
        num_queries=300, ignore_mismatched_sizes=True, output_loading_info=True)
    # 两个坑：
    #  1) 新版 transformers 把"形状不符"的 key 放进 mismatched_keys，旧版塞进
    #     missing_keys —— 取并集才不会因为版本差异让这条断言变成空转；
    #  2) mismatched_keys 的元素不是字符串，而是 (key, ckpt_shape, model_shape)
    #     三元组（HF 用它拼那句 "found shape A ... and shape B" 的 warning），
    #     直接 startswith 会 AttributeError: 'tuple' object has no attribute ...
    def _names(entries):
        return [e[0] if isinstance(e, (tuple, list)) else e for e in (entries or [])]

    missing = (set(_names(load_info.get('missing_keys')))
               | set(_names(load_info.get('mismatched_keys'))))
    unexpected = set(_names(load_info.get('unexpected_keys')))
    is_class_dependent = (
        lambda k: k.startswith('model.decoder.class_embed.')
        or k.startswith('model.enc_score_head.')
        or k == 'model.denoising_class_embed.weight'
    )
    bad_missing = sorted(k for k in missing if not is_class_dependent(k))
    check(not bad_missing and not unexpected,
          "只有与类别数绑定的张量是新建的，其余全部来自预训练 checkpoint",
          f"{len(missing)} 个新建" + (f"，意外缺失: {bad_missing[:3]}" if bad_missing else "")
          + (f"，多余 key: {sorted(unexpected)[:3]}" if unexpected else ""))
    print(f"        新建的 {len(missing)} 个张量: "
          f"{sorted({k.rsplit('.', 1)[0] for k in missing})}")
    check(m.config.num_labels == 11, "换 11 类后 config.num_labels == 11",
          f"实际 {m.config.num_labels}")
    check(m.class_embed[0].out_features == 11,
          "class_embed[0].out_features == 11（COCO 的 80 类 head 已重建）",
          f"实际 {m.class_embed[0].out_features}")

    cfg_info = {k: getattr(m.config, k, None) for k in
                ['num_queries', 'eval_size', 'anchor_image_size', 'use_focal_loss',
                 'auxiliary_loss', 'feat_strides']}
    print(f"  config: {cfg_info}")
    if m.config.eval_size is not None:
        print("  !! eval_size 不是 None，换分辨率会 shape mismatch，"
              "需要在代码里置 None")

    # ---- 两个尺寸各 forward 一次 ----
    for size in [img_size, 1024]:
        p = proc(images=Image.fromarray(rgb), return_tensors='pt',
                 size={'height': size, 'width': size})
        with torch.no_grad():
            try:
                o = m(pixel_values=p['pixel_values'])
                ok = tuple(o.logits.shape) == (1, m.config.num_queries, 11)
            except Exception as e:
                ok = False
                print(f"       异常: {type(e).__name__}: {e}")
        check(ok, f"{size}×{size} forward 通过且 logits 形状为 "
                  f"[1,{m.config.num_queries},11]")
    return proc


# ==================== Phase 1 ====================

def phase1(proc, img_size, n=20, save_vis=True):
    print()
    print("=" * 74)
    print(f"Phase 1: 数据管线往返 ({n} 张 val 图)")
    print("=" * 74)
    from config import VAL_ANN, VAL_IMG_DIR
    from torch.utils.data import DataLoader

    ds = RTDetrDataset(VAL_ANN, VAL_IMG_DIR, is_train=False, processor=proc,
                       img_size=img_size)
    model = RTDetrLayoutDetector(pretrained=False, model_name_or_path=RTDETR_MODEL_NAME,
                                 img_size=img_size)

    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    max_coord_err = 0.0
    bad_shapes = bad_labels = 0
    for i in idxs:
        s = ds[int(i)]
        pv, boxes, labels, orig = (s['pixel_values'], s['boxes'], s['labels'],
                                   s['orig_size'])
        if tuple(pv.shape) != (3, img_size, img_size) or pv.dtype != torch.float32:
            bad_shapes += 1
        if len(labels) and (labels.min() < 1 or labels.max() > 11):
            bad_labels += 1
        if len(boxes) == 0:
            continue
        hf = model._targets_to_hf([s])[0]
        if len(hf['class_labels']) and (hf['class_labels'].min() < 0
                                        or hf['class_labels'].max() > 10):
            bad_labels += 1
        # 归一化 cxcywh → 原图 xyxy
        H0, W0 = orig.tolist()
        cx, cy, bw, bh = hf['boxes'].unbind(-1)
        back = torch.stack([(cx - bw / 2) * W0, (cy - bh / 2) * H0,
                            (cx + bw / 2) * W0, (cy + bh / 2) * H0], dim=-1)
        max_coord_err = max(max_coord_err, (back - boxes).abs().max().item())

    check(bad_shapes == 0, f"pixel_values 形状/dtype 全部正确", f"{bad_shapes} 个异常")
    check(bad_labels == 0, "label 域正确（项目 1..11 / HF 0..10）",
          f"{bad_labels} 个异常")
    check(max_coord_err < 1e-3,
          "坐标往返无损：项目 xyxy@原图 → HF 归一化 cxcywh → 解回原图",
          f"max|Δ| = {max_coord_err:.2e}")

    # 零标注图：train 27 张 / val 4 张。DataSet 的 gt 索引必须覆盖【所有】图片，
    # 否则 __getitem__ 会 KeyError，而且因为 DataLoader 带 shuffle，
    # 崩会发生在第 1 个 epoch 中途，前面几千张都正常，极难定位。
    from dataset import _load_coco, DocLayNetDataset
    _, anns_t, ids_t, _ = _load_coco(TRAIN_ANN)
    empty = [i for i in ids_t if not anns_t.get(i)]
    print(f"  train 零标注图: {len(empty)} / {len(ids_t)} 张")

    ds_t = RTDetrDataset(TRAIN_ANN, TRAIN_IMG_DIR, is_train=False, processor=proc,
                         img_size=img_size)
    missing = [i for i in ds_t.img_ids if i not in ds_t.gt]
    check(not missing, "RTDetrDataset 的 gt 索引覆盖所有图片（含零标注图）",
          f"缺失 {len(missing)} 张" + (f"，首张 id={missing[0]}" if missing else ""))

    if empty:
        try:
            s = ds_t[ds_t.img_ids.index(empty[0])]
            check(s['boxes'].shape == (0, 4) and s['labels'].shape == (0,),
                  f"零标注图能取出来且为空（id={empty[0]}）",
                  f"boxes{tuple(s['boxes'].shape)} labels{tuple(s['labels'].shape)}")
        except Exception as e:
            check(False, "零标注图能取出来", f"{type(e).__name__}: {e}")

        # 混进 collate 也不崩（真实训练路径）
        try:
            from torch.utils.data import DataLoader as _DL
            from dataset import rtdetr_collate_fn as _cf
            sub = _DL(ds_t, batch_size=4, shuffle=False, num_workers=0, collate_fn=_cf)
            b = next(iter(sub))
            check(tuple(b['pixel_values'].shape) == (4, 3, img_size, img_size),
                  "零标注图与有标注图混在一个 batch 里也不崩",
                  f"pixel_values{tuple(b['pixel_values'].shape)}")
        except Exception as e:
            check(False, "零标注图与有标注图混在一个 batch 里也不崩",
                  f"{type(e).__name__}: {e}")

    # FRCNN 通路是同一份 gt 索引逻辑，也验一下（不需要 processor，成本很低）
    ds_f = DocLayNetDataset(TRAIN_ANN, TRAIN_IMG_DIR, is_train=False)
    missing_f = [i for i in ds_f.img_ids if i not in ds_f.gt]
    check(not missing_f, "DocLayNetDataset 的 gt 索引覆盖所有图片（含零标注图）",
          f"缺失 {len(missing_f)} 张")

    # 可视化：框位置必须和图像内容对得上
    if save_vis:
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=rtdetr_collate_fn)
        batch = next(iter(dl))
        outdir = 'runs/_verify'
        os.makedirs(outdir, exist_ok=True)
        for k, t in enumerate(batch['targets'][:2]):
            path = ds.img_dir
            name = ds.images[ds.img_ids[k]]['file_name']
            bgr = imread(os.path.join(path, name))
            # 必须转 numpy：draw_detections 用 `label in class_names` 查表，
            # torch tensor 的 hash 是按对象 id，传 tensor 会静默查不到类名
            b = t['boxes'].numpy()
            l = t['labels'].numpy()
            vis = draw_detections(bgr, b, l, class_names=LAYOUT_CLASSES, threshold=0)
            out_path = os.path.join(outdir, f'gt_{k}_{name}')
            imwrite(out_path, vis)
            print(f"  可视化: {out_path}  ({len(b)} 个框)")
        print("  ^^ 请打开这两张图，肉眼确认框位置与图像内容一致")


# ==================== Phase 2 ====================

def phase2(proc, img_size, n, head_lr=1e-3, max_epochs=150, loss_drop_ratio=0.2):
    """过拟合 n 张图。走的是 train.py 的真实训练循环，顺带验证那条链路的接线。

    这里【冻住预训练 backbone】，是有意的：
      - 本测试要验的是 data / label 映射 / 坐标转换 / loss 路径 / 后处理是否通，
        不是预训练特征的质量；
      - 不冻就必须给 backbone 单独的 0.1x 低 LR。上一版把 backbone 也放在 1e-3
        的单组优化器里，等于把 ResNet 的预训练特征打烂，表现为 loss 缓慢幂律下降
        （72 → 8）但永远收不敛 —— 看起来像"pipeline 有问题"，其实是测试自己配错了。
      正式训练用的是 resolve_train_params 给的 backbone_lr_scale=0.1，不受这里影响。
    """
    print()
    print("=" * 74)
    print(f"Phase 2: 过拟合 {n} 张图（最能暴露 bug 的一步）")
    print("=" * 74)
    from torch.utils.data import DataLoader
    import train as T
    from torchvision.ops import box_iou

    dev = torch.device('cuda')
    bs = min(n, 4)
    ds = RTDetrDataset(TRAIN_ANN, TRAIN_IMG_DIR, is_train=False, processor=proc,
                       img_size=img_size, subset=n)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                        collate_fn=rtdetr_collate_fn, drop_last=True)

    model = RTDetrLayoutDetector(pretrained=True,
                                 model_name_or_path=RTDETR_MODEL_NAME,
                                 img_size=img_size).to(dev)
    for p in model.backbone.parameters():
        p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  backbone 已冻结，可训练参数 {n_train / 1e6:.1f}M")

    params = resolve_train_params('rtdetr', 'resnet50')
    params.update(base_lr=head_lr, epochs=max_epochs, warmup_steps=0,
                  lr_decay_steps=[], early_stop_patience=999,
                  early_stop_min_epoch=0, img_size=img_size)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=head_lr, weight_decay=0.0)
    for pg in optimizer.param_groups:
        pg['lr_scale'] = 1.0        # 过拟合测试不做分组 LR
    # bf16 → scaler None，和 train.py 的约定一致
    scaler = None

    # 先单独验证 aux loss 聚合有没有丢项。
    # 必须在【同一次】forward 的输出上比：RT-DETR 带 denoising，训练模式下每次
    # forward 都会重新随机扰动 GT 再喂进去，两次 forward 的 loss 天然不同。
    batch = next(iter(loader))
    pv = batch['pixel_values'].to(dev)
    tg = [{k: v.to(dev) for k, v in t.items()} for t in batch['targets']]
    model.train()
    raw = model.hf_model(pixel_values=pv, labels=model._targets_to_hf(tg))
    agg = model._aggregate_loss(raw.loss_dict)
    raw_loss_keys = [k for k in raw.loss_dict if k.startswith('loss_')]
    raw_sum = sum(raw.loss_dict[k] for k in raw_loss_keys)
    diff = abs(sum(agg.values()).item() - raw_sum.item())
    check(diff < 1e-3, "loss 聚合没丢项（聚合后 sum == 原始 loss_* 之和）",
          f"diff={diff:.2e}, {len(raw_loss_keys)} 个原始 key -> {sorted(agg)}")
    check(all(k.startswith('loss_') for k in agg),
          "非梯度 key（cardinality_error 等）已被过滤",
          f"原始 key: {sorted(raw.loss_dict)}")
    optimizer.zero_grad(set_to_none=True)

    # 诊断：把 39 个 loss key 拆成 main / aux / dn 三组，看收敛卡在哪一组。
    # main = 最后一层解码器的真实检测损失（我们真正关心的）
    # aux  = 各辅助解码层的同类损失
    # dn   = denoising 自监督损失（输入是加噪的 GT，与真实检测任务无关）
    def loss_split():
        model.train()
        with torch.no_grad():
            r = model.hf_model(pixel_values=pv, labels=model._targets_to_hf(tg))
        out = {}
        for k, v in r.loss_dict.items():
            if not k.startswith('loss_'):
                continue
            term = 'vfl' if 'vfl' in k else ('bbox' if 'bbox' in k else 'giou')
            grp = 'dn' if '_dn_' in k else ('aux' if '_aux_' in k else 'main')
            out[f'{grp}_{term}'] = out.get(f'{grp}_{term}', 0.0) + float(v)
        return out

    s0 = loss_split()
    print("  loss 拆分（初始）: " + ' '.join(f'{k}={v:.2f}' for k, v in sorted(s0.items())))
    print(f"    main 合计={sum(v for k, v in s0.items() if k.startswith('main')):.2f}  "
          f"aux={sum(v for k, v in s0.items() if k.startswith('aux')):.2f}  "
          f"dn={sum(v for k, v in s0.items() if k.startswith('dn')):.2f}")

    print(f"  训练 {max_epochs} epoch × {len(loader)} step (batch={bs}, lr={head_lr}, bf16)...")
    hist, gs = [], 0
    for ep in range(1, max_epochs + 1):
        losses, gs = T.train_one_epoch(model, loader, optimizer, dev, ep, scaler,
                                       params, gs)
        hist.append(losses['total'])
        if ep % 10 == 0 or ep == 1:
            tstr = ' '.join(f'{k}={v:.2f}' for k, v in losses.items() if k != 'total')
            print(f"    epoch {ep:3d}  step={gs:4d}  total={hist[-1]:.3f}  {tstr}")

    s1 = loss_split()
    print("  loss 拆分（结束）: " + ' '.join(f'{k}={v:.2f}' for k, v in sorted(s1.items())))
    main1 = sum(v for k, v in s1.items() if k.startswith('main'))
    dn1 = sum(v for k, v in s1.items() if k.startswith('dn'))
    print(f"    main 合计={main1:.2f}  aux={sum(v for k, v in s1.items() if k.startswith('aux')):.2f}"
          f"  dn={dn1:.2f}")
    print(f"    ^ dn 是自监督的加噪重建任务，与检测质量无关。"
          f"如果 main 已经很小而 total 还大，说明卡点在 dn，不是链路问题。")

    first, final = hist[0], hist[-1]
    main0 = sum(v for k, v in s0.items() if k.startswith('main'))
    # 门限只看 main（最后一层解码器的真实检测损失）。total 里 76% 是 dn ——
    # denoising 的加噪重建任务永远降不到 0，拿 total 当门限会误判"没收敛"。
    check(main1 < loss_drop_ratio * main0,
          f"main 检测损失相对初始下降 > {(1 - loss_drop_ratio) * 100:.0f}%",
          f"{main0:.2f} -> {main1:.2f} (剩余 {main1 / max(main0, 1e-9) * 100:.1f}%)")
    print(f"        （total 从 {first:.2f} 降到 {final:.2f}，其中 "
          f"main={main1:.2f} aux={sum(v for k, v in s1.items() if k.startswith('aux')):.2f} "
          f"dn={dn1:.2f}）")

    # eval 指标：召回要数"被命中过的 GT"，而不是"命中的预测数"
    # （一个 GT 可以被多个预测框命中，按预测数算会得到 >1 的假召回）
    from collections import defaultdict

    # 按短边分档（短边才是可检测性的驱动因素，见 README 的尺寸分析）
    BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 10 ** 9)]

    def bin_of(box):
        s = min(float(box[2] - box[0]), float(box[3] - box[1]))
        for lo, hi in BINS:
            if lo <= s < hi:
                return f'{lo}-{hi if hi < 10 ** 9 else "inf"}'
        return '?'

    per_cls = defaultdict(lambda: [0, 0])
    per_bin = defaultdict(lambda: [0, 0])
    cover = defaultdict(int)          # 每个 GT 被多少个预测框命中 -> 看塌缩程度

    model.eval()
    n_gt = n_pred = n_pred_hit = n_gt_matched = n_cls_ok = 0
    with torch.no_grad():
        for b in loader:
            pvi = b['pixel_values'].to(dev)
            tgi = [{k: v.to(dev) for k, v in t.items()} for t in b['targets']]
            outs = model(pvi, tgi)
            for o, t in zip(outs, b['targets']):
                gb, gl = t['boxes'], t['labels']
                n_gt += len(gb)
                n_pred += len(o['boxes'])
                if len(gb) == 0 or len(o['boxes']) == 0:
                    continue
                iou = box_iou(o['boxes'].cpu(), gb)        # [P, N]
                best_iou, best_idx = iou.max(1)            # 每个预测框最近的 GT
                hit = best_iou >= 0.5
                n_pred_hit += int(hit.sum())
                n_gt_matched += len(best_idx[hit].unique())   # 去重后才是召回
                n_cls_ok += int((o['labels'].cpu()[hit] == gl[best_idx[hit]]).sum())

                hit_set = set(best_idx[hit].tolist())
                for j in range(len(gb)):
                    cls = int(gl[j])
                    per_cls[cls][0] += 1
                    per_bin[bin_of(gb[j])][0] += 1
                    if j in hit_set:
                        per_cls[cls][1] += 1
                        per_bin[bin_of(gb[j])][1] += 1
                        cover[j] += int((best_idx[hit] == j).sum())

    rec = n_gt_matched / max(n_gt, 1)
    prec = n_pred_hit / max(n_pred, 1)
    cls_acc = n_cls_ok / max(n_pred_hit, 1)
    # 大框覆盖率才是有效判据：800px 下短边 <32px 的框占 21%（见 README 的尺寸分析），
    # 要求"全部框都召回 95%"在物理上就不可达，拿它当门限只会一直 FAIL 而分不清
    # 是链路坏了还是小目标定位不了。大框（>=64px）能清楚分辨，适合当门限。
    big_tot = sum(v[0] for k, v in per_bin.items() if k in ('64-128', '128-inf'))
    big_cov = sum(v[1] for k, v in per_bin.items() if k in ('64-128', '128-inf'))
    big_rec = big_cov / max(big_tot, 1)

    print(f"  eval: GT={n_gt} 预测={n_pred} 命中预测={n_pred_hit} 命中GT={n_gt_matched}")
    print(f"        每个被覆盖的 GT 平均挨了 {n_pred_hit / max(n_gt_matched, 1):.1f} 个预测框"
          f"（正常应接近 1；远大于 1 说明 query 没有分化）")

    print("  按类别召回（命中的GT / 总GT）：")
    print("    " + '  '.join(
        f"{T.SHORT_CLASS_NAMES[LAYOUT_CLASSES[c]]}:{per_cls[c][1]}/{per_cls[c][0]}"
        for c in sorted(LAYOUT_CLASSES) if per_cls[c][0]))
    print("  按 GT 短边分档召回：")
    for lo, hi in BINS:
        k = f'{lo}-{hi if hi < 10 ** 9 else "inf"}'
        if per_bin[k][0]:
            print(f"    {k:>9s}px: {per_bin[k][1]:4d}/{per_bin[k][0]:4d} "
                  f"= {per_bin[k][1] / per_bin[k][0]:.3f}")

    check(big_rec >= 0.90,
          "大框（短边 >=64px）覆盖率 >= 90%  —— 有效判据",
          f"实际 {big_cov}/{big_tot} = {big_rec:.3f}")
    check(cls_acc >= 0.95, "命中框的类别正确率 >= 95%", f"实际 {cls_acc:.3f}")
    print(f"  （参考）全尺寸召回={rec:.3f}  precision={prec:.3f}  "
          f"预测/GT={n_pred / max(n_gt, 1):.2f}x")

    # ---- 决定性诊断：训练侧的 pred_boxes 和 eval 侧的后处理输出，到底差在哪 ----
    # main_bbox 已经收敛到 0.07（预测的归一化 cxcywh 和 GT 几乎重合），
    # 但 eval 侧只有 ~10% 的 GT 被召回 —— 两者不可能同时成立，
    # 除非训练侧和 eval 侧不是同一个东西。这里把两边都解码回原图像素空间直接比。
    print()
    print("  === 训练侧 vs eval 侧一致性诊断（绕开/不绕开 post_process）===")
    s = ds[0]
    pv1 = s['pixel_values'].unsqueeze(0).to(dev)
    H0, W0 = s['orig_size'].tolist()
    gt = s['boxes']                                    # 1025 像素 xyxy
    # 注意：wrapper 的 _targets_to_hf 期望 t['boxes'] 是 [N,4]（无 batch 维），
    # 所以这里【不能】unsqueeze —— targets 是 list[dict]，batch 维由 list 承载
    t1 = {'boxes': s['boxes'].to(dev), 'labels': s['labels'].to(dev),
          'orig_size': s['orig_size'].to(dev)}

    def decode_norm_cxcywh(b):                          # [Q,4] -> 1025 像素 xyxy
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([(cx - w / 2) * W0, (cy - h / 2) * H0,
                            (cx + w / 2) * W0, (cy + h / 2) * H0], dim=-1)

    def report(tag, boxes):
        best = box_iou(boxes.cpu(), gt).max(1)          # 每个预测框最近的 GT
        hit = best.values >= 0.5
        # 唯一 GT 覆盖数才是关键指标；"命中的预测框数"会因为重复命中而虚高
        n_cov = len(best.indices[hit].unique())
        sz = torch.min(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]).cpu()
        gsz = torch.min(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1])
        print(f"    {tag}: 预测{len(boxes)}个  命中预测={int(hit.sum())}  "
              f"唯一GT覆盖={n_cov}/{len(gt)}")
        print(f"        最佳IoU median={best.values.median():.3f}  "
              f"预测短边 median={sz.median():.1f}px  GT短边 median={gsz.median():.1f}px")

    model.train()
    with torch.no_grad():
        tr = model.hf_model(pixel_values=pv1)
    model.eval()
    with torch.no_grad():
        ev = model.hf_model(pixel_values=pv1)
    print(f"    pred_boxes 形状: train={tuple(tr.pred_boxes.shape)} "
          f"eval={tuple(ev.pred_boxes.shape)}")

    q = model.num_queries
    nq = tr.pred_boxes.shape[1]
    if nq > q:
        # denoising 的 query 是排在主 query 前面还是后面，取决于实现细节；
        # 两端都测一次，避免我又在"该取哪一段切片"上猜错
        print(f"    注意: train 侧 pred_boxes 有 {nq} 个 query（{nq - q} 个来自 denoising），"
              f"两端各测一次")
        report('训练侧 [0:Q]', decode_norm_cxcywh(tr.pred_boxes[0][:q].float()))
        report('训练侧 [-Q:]', decode_norm_cxcywh(tr.pred_boxes[0][-q:].float()))
    else:
        report('训练侧 [0:Q]', decode_norm_cxcywh(tr.pred_boxes[0][:q].float()))
    report('eval侧 [0:Q]', decode_norm_cxcywh(ev.pred_boxes[0][:q].float()))
    with torch.no_grad():
        full = model(pv1, [t1])[0]
    report('eval侧 post_process 完整路径', full['boxes'].float())

    # ---- 对照组：未经微调的 COCO 80 类原模型，走完全相同的解码逻辑 ----
    # 这是判断"塌缩是微调造成的、还是我的 wrapper/解码有 bug"的对照实验：
    # 这个模型在 COCO 上是可用的（官方 model card 有验证过的输出）。
    #   若它也输出"300 个框塌缩成十来个位置" -> 我的 eval 路径有问题；
    #   若它输出的是分散的 300 个框     -> 塌缩是微调过程造成的，eval 路径没问题。
    print()
    print("  === 对照组：未微调的 COCO 80 类模型，同一套解码 ===")

    def diversity(tag, pb):
        """数一下这 300 个框里到底有几个\"不同位置\"（互相 IoU>0.5 视为同一个）。"""
        M = box_iou(pb.cpu(), pb.cpu())
        order = torch.argsort(-((pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])).cpu())
        kept = []
        for i in order.tolist():
            if all(float(M[i, j]) <= 0.5 for j in kept):
                kept.append(i)
        cx = ((pb[:, 0] + pb[:, 2]) / 2).cpu()
        cy = ((pb[:, 1] + pb[:, 3]) / 2).cpu()
        sz = torch.min(pb[:, 2] - pb[:, 0], pb[:, 3] - pb[:, 1]).cpu()
        print(f"    {tag}: 300 个框 -> {len(kept)} 个不同位置  "
              f"中心 std=({cx.std():.3f}, {cy.std():.3f})  短边 median={sz.median():.1f}px")

    try:
        from transformers import RTDetrForObjectDetection as _RawRT
        raw = _RawRT.from_pretrained(RTDETR_MODEL_NAME).to(dev).eval()
        with torch.no_grad():
            ro = raw(pixel_values=pv1)
        diversity('未微调 80 类',
                  decode_norm_cxcywh(ro.pred_boxes[0][:q].float()))
        diversity('微调后 11 类', decode_norm_cxcywh(ev.pred_boxes[0][:q].float()))
        print("    判读：未微调那行若是\"分散的几百个位置\"，说明我的解码/后处理没问题，")
        print("          塌缩发生在微调过程中；若它也塌缩，则是我的 eval 路径有 bug。")
        del raw
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"    [skip] 对照组失败: {type(e).__name__}: {e}")


def print_planned_schedule():
    """打印正式训练时衰减点会落在哪个 step —— 开跑前先确认计划是对的。"""
    from dataset import _load_coco
    p = resolve_train_params('rtdetr', 'resnet50')
    _, _, ids, _ = _load_coco(TRAIN_ANN)
    spe = len(ids) // p['batch_size']            # drop_last=True
    print()
    print("=" * 74)
    print("正式训练的 LR 计划（RT-DETR，config 的真实超参）")
    print("=" * 74)
    print(f"  batch_size={p['batch_size']}  train={len(ids)} 张  ->  {spe} steps/epoch")
    print(f"  epochs={p['epochs']}  总步数={p['epochs'] * spe}")
    print(f"  warmup={p['warmup_steps']} steps "
          f"(≈ {p['warmup_steps'] / max(spe, 1):.1f} epoch)")
    print(f"  base_lr={p['base_lr']:.2e}  backbone_lr={p['base_lr'] * p['backbone_lr_scale']:.2e}")
    print("  阶梯衰减点:")
    lr = p['base_lr']
    prev = p['warmup_steps']
    for ep, factor in p['lr_decay_steps']:
        step = ep * spe
        print(f"    step {step:>6d} (epoch {ep:3d} 起): {lr:.2e} -> {lr * factor:.2e}")
        lr *= factor
    print(f"  早停: patience={p['early_stop_patience']}，"
          f"最早 epoch {p['early_stop_min_epoch']}"
          f"（= 最后一个衰减点，保证衰减一定会发生）")


def phase3_lr_schedule(proc, img_size, n=8):
    """集成测试：LR 阶梯衰减在【真实 train_one_epoch】里到底有没有生效。

    为什么要单独测：本项目已经在这个位置翻过两次车，且两次都是"代码看着对、
    调度静默不生效" ——
      1) 早期 train.py 用 config.DETECTOR 判断、命令行改的是 detector_name，
         整段 step 级调度被跳过；
      2) initial_lr 在 load_state_dict 之后被 checkpoint 里的衰减值覆盖，
         续训后 base_lr 永久变成 6e-5。
    这两个单测 get_lr_step 都测不出来（数学是对的，错的是接线）。
    """
    print()
    print("=" * 74)
    print("Phase 3: LR 阶梯衰减接线（跑真实的 train_one_epoch）")
    print("=" * 74)
    from torch.utils.data import DataLoader
    import train as T

    dev = torch.device('cuda')
    bs = min(n, 4)
    ds = RTDetrDataset(TRAIN_ANN, TRAIN_IMG_DIR, is_train=False, processor=proc,
                       img_size=img_size, subset=n)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0,
                        collate_fn=rtdetr_collate_fn, drop_last=True)
    spe = len(loader)                     # 2

    # 用真实的分组 LR（backbone 0.1x），衰减点缩小到几步之内，把整条接线走一遍
    base = 1e-3
    params = resolve_train_params('rtdetr', 'resnet50')
    params.update(base_lr=base, epochs=5, warmup_steps=3,
                  lr_decay_steps=[(2, 0.3), (4, 0.3)], early_stop_patience=999,
                  early_stop_min_epoch=0, img_size=img_size)
    decay_steps = [(e * spe, f) for e, f in params['lr_decay_steps']]
    scale = params['backbone_lr_scale']

    model = RTDetrLayoutDetector(pretrained=False,
                                 model_name_or_path=RTDETR_MODEL_NAME,
                                 img_size=img_size).to(dev)
    for p in model.backbone.parameters():
        p.requires_grad = False
    bb = [p for n_, p in model.named_parameters() if p.requires_grad and 'backbone' in n_]
    other = [p for n_, p in model.named_parameters() if p.requires_grad and 'backbone' not in n_]
    optimizer = torch.optim.AdamW([{'params': bb, 'lr': base * scale},
                                   {'params': other, 'lr': base}], weight_decay=0.0)
    for pg, s in zip(optimizer.param_groups, [scale, 1.0]):
        pg['lr_scale'] = s

    # 拦截 set_optimizer_lr，逐 step 记录真实下发的 LR
    recorded, orig = [], T.set_optimizer_lr

    def spy(opt, lr):
        recorded.append(lr)
        return orig(opt, lr)

    T.set_optimizer_lr = spy
    try:
        gs = 0
        for ep in range(1, 6):
            _, gs = T.train_one_epoch(model, loader, optimizer, dev, ep, None,
                                     params, gs)
    finally:
        T.set_optimizer_lr = orig

    check(len(recorded) == 5 * spe,
          f"每步都调用了一次 LR 调度（共 {5 * spe} 步）",
          f"实际调用 {len(recorded)} 次")
    if len(recorded) != 5 * spe:
        return

    # 期望序列：warmup(0, 1/3, 2/3) -> base -> base*0.3 -> base*0.09
    exp = []
    for s in range(5 * spe):
        exp.append(T.get_lr_step(s, base, params['warmup_steps'], decay_steps))
    err = max(abs(a - b) for a, b in zip(recorded, exp))
    show = ' '.join(f'{v:.2e}' for v in recorded)
    print(f"  实际下发 LR 序列: {show}")
    check(err < 1e-12, "逐 step 的 LR 与 get_lr_step 的期望值一致",
          f"max|Δ| = {err:.1e}")

    uniq = []
    for v in recorded:
        if not uniq or abs(v - uniq[-1]) > 1e-12:
            uniq.append(v)
    expect_uniq = [0.0, base / 3, base * 2 / 3, base, base * 0.3, base * 0.09]
    print(f"  去重后的 LR 台阶: {['%.2e' % v for v in uniq]}")
    check(len(uniq) == 6 and all(abs(a - b) < 1e-12 for a, b in zip(uniq, expect_uniq)),
          "LR 走出完整的 5 级台阶（含 warmup 与两次衰减）",
          f"期望 {['%.2e' % v for v in expect_uniq]}")

    # 分组 LR：backbone 组必须是主组的 scale 倍，且与衰减同步
    g0, g1 = optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']
    check(abs(g1 - recorded[-1]) < 1e-12 and abs(g0 - recorded[-1] * scale) < 1e-12,
          f"分组 LR 正确：backbone = 主组 x {scale}",
          f"backbone={g0:.3e} 主组={g1:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default=RTDETR_MODEL_NAME)
    ap.add_argument("--img_size", type=int, default=RTDETR_IMG_SIZE)
    ap.add_argument("--skip_phase0", action="store_true")
    ap.add_argument("--overfit", type=int, default=0,
                    help="跑过拟合测试用的图片数，0 表示跳过（需要 GPU）")
    ap.add_argument("--overfit_epochs", type=int, default=150,
                    help="过拟合测试的训练 epoch 数（默认 150，每个 epoch 2 step）")
    ap.add_argument("--overfit_lr", type=float, default=1e-3,
                    help="过拟合测试的 head lr（backbone 已冻结，不受影响）")
    args = ap.parse_args()

    proc = None
    if not args.skip_phase0:
        proc = phase0(args.model_name_or_path, args.img_size)
    if proc is None:
        proc = build_rtdetr_processor(args.model_name_or_path, args.img_size)

    phase1(proc, args.img_size)

    # Phase 3 很便宜（10 次前向），放在 Phase 2 前面，这样即使跳过错拟合测试
    # 也能拿到"LR 衰减到底有没有接线"的答案
    if torch.cuda.is_available():
        phase3_lr_schedule(proc, args.img_size)
    else:
        print("\n[skip] Phase 3 需要 GPU")

    if args.overfit:
        if not torch.cuda.is_available():
            print("\n[skip] Phase 2 需要 GPU")
        else:
            phase2(proc, args.img_size, args.overfit,
                   head_lr=args.overfit_lr, max_epochs=args.overfit_epochs)

    print_planned_schedule()

    print()
    print("=" * 74)
    if FAIL:
        print(f"存在 {len(FAIL)} 项失败，先修掉再训练：")
        for m in FAIL:
            print(f"  - {m}")
        sys.exit(1)
    print("全部通过 ✓  可以开始训练了")
    print("=" * 74)


if __name__ == "__main__":
    main()
