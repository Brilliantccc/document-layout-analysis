"""
文档版面分析推理脚本
单张/批量推理 + 可视化

detector 从 checkpoint 的元信息读取，不需要手动指定。
RT-DETR 走 RTDetrImageProcessor（必须喂 uint8 0-255，不是除过 255 的 float），
再用 orig_sizes 把框映射回原图分辨率。
"""
import os
import sys
import json
import argparse
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import LAYOUT_CLASSES, RTDETR_IMG_SIZE, RTDETR_NUM_QUERIES
from model import LayoutDetector, RTDetrLayoutDetector
from utils import imread, imwrite, draw_detections


class LayoutAnalyzer:
    """文档版面分析推理器"""

    def __init__(self, weights_path, threshold=0.5, device='cuda',
                 backbone=None, detector=None, img_size=None):
        self.threshold = threshold
        self.device = device

        if os.path.isdir(weights_path):
            # ModelScope 上发布的是原生 transformers 目录
            # （config.json + model.safetensors + preprocessor_config.json），
            # 直接用 from_pretrained 加载，不需要 .pth
            self._init_from_hf_dir(weights_path, device, img_size)
        else:
            self._init_from_pth(weights_path, device, backbone, detector, img_size)

        self.model.eval()
        self.class_names = LAYOUT_CLASSES
        print(f"模型加载完成: {weights_path}")
        print(f"detector={self.detector} 阈值={threshold} 设备={device}")

    def _init_from_hf_dir(self, d, device, img_size):
        """加载 ModelScope 发布的原生 HF 目录（目前只支持 RT-DETR）"""
        import transformers
        # transformers 5.x 改了 RT-DETR 的内部键名，加载 4.49 存的权重会大量 missing
        # 且不报错（静默随机初始化）—— 直接卡住版本，避免悄无声息地出垃圾结果
        if int(transformers.__version__.split('.')[0]) >= 5:
            raise SystemExit(
                f"加载原生 HF 权重需要 transformers 4.x（当前 {transformers.__version__}）。\n"
                "  5.x 改了 RT-DETR 的内部键名，会静默加载失败。\n"
                "  请执行: pip install 'transformers>=4.48,<4.50'")
        from transformers import RTDetrConfig
        cfg = RTDetrConfig.from_pretrained(d)
        # 输入尺寸记在 preprocessor_config.json 里（训练时的设置），不要用默认值
        if img_size is None:
            pc = os.path.join(d, 'preprocessor_config.json')
            if os.path.exists(pc):
                with open(pc, encoding='utf-8') as f:
                    sz = json.load(f).get('size') or {}
                img_size = sz.get('height')
        self.detector = 'rtdetr'
        self.img_size = img_size or RTDETR_IMG_SIZE
        # pretrained=True + 本地目录 → 走 from_pretrained，直接读 model.safetensors
        self.model = RTDetrLayoutDetector(pretrained=True, num_queries=cfg.num_queries,
                                          model_name_or_path=d,
                                          img_size=self.img_size).to(device)
        self.processor = self.model.processor

    def _init_from_pth(self, weights_path, device, backbone, detector, img_size):
        """加载训练时保存的 .pth checkpoint"""
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        run_dir = os.path.dirname(weights_path)
        # 优先用 checkpoint 里记录的 detector，避免命令行忘了传
        self.detector = detector or checkpoint.get('detector') or 'frcnn'
        backbone = backbone or checkpoint.get('backbone') or 'resnet50'
        self.img_size = img_size or checkpoint.get('img_size') or RTDETR_IMG_SIZE

        if self.detector == 'rtdetr':
            num_queries = checkpoint.get('num_queries') or RTDETR_NUM_QUERIES
            # 训练时已把 config + processor 存进 run 目录 → 不联网
            src = run_dir if os.path.exists(os.path.join(run_dir, 'config.json')) \
                else checkpoint.get('model_name_or_path')
            self.model = RTDetrLayoutDetector(pretrained=False, num_queries=num_queries,
                                              model_name_or_path=src,
                                              img_size=self.img_size).to(device)
            self.processor = self.model.processor
        else:
            self.model = LayoutDetector(backbone=backbone, detector='frcnn',
                                        pretrained=False).to(device)
            self.processor = None

        self.model.load_state_dict(checkpoint['model_state_dict'])

    @torch.no_grad()
    def _predict(self, img_bgr):
        """返回 {'boxes' xyxy 原始分辨率像素, 'scores', 'labels' 1-based}"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]

        if self.detector == 'rtdetr':
            from PIL import Image
            # processor 要的是 uint8 0-255（它内部才做 /255 + ImageNet 归一化）
            out = self.processor(images=Image.fromarray(np.ascontiguousarray(img_rgb)),
                                 return_tensors='pt',
                                 size={'height': self.img_size, 'width': self.img_size})
            pixel_values = out['pixel_values'].to(self.device)
            orig_sizes = torch.tensor([[H0, W0]])
            return self.model(pixel_values, orig_sizes=orig_sizes)[0]

        tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return self.model([tensor.to(self.device)])[0]

    @torch.no_grad()
    def analyze(self, image_path):
        img = imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取: {image_path}")

        output = self._predict(img)
        mask = output['scores'] >= self.threshold
        labels = output['labels'][mask].cpu()   # 1-based
        return {
            'boxes': output['boxes'][mask].cpu().numpy().tolist(),
            'scores': output['scores'][mask].cpu().numpy().tolist(),
            'classes': labels.numpy().tolist(),
            'class_names': [self.class_names.get(c, str(c)) for c in labels.numpy()],
        }

    @torch.no_grad()
    def analyze_and_visualize(self, image_path, output_path=None):
        img = imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取: {image_path}")

        output = self._predict(img)
        mask = output['scores'] >= self.threshold
        boxes = output['boxes'][mask].cpu().numpy()
        scores = output['scores'][mask].cpu().numpy()
        classes = output['labels'][mask].cpu().numpy()   # 1-based

        result_img = draw_detections(
            img, boxes, classes, scores,
            class_names=self.class_names, threshold=0
        )

        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_result{ext}"

        imwrite(output_path, result_img)
        print(f"结果保存: {output_path}")
        return {'boxes': boxes.tolist(), 'scores': scores.tolist(),
                'classes': classes.tolist()}

    def analyze_batch(self, image_dir, output_dir=None):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_paths = []
        for ext in exts:
            image_paths.extend(glob(os.path.join(image_dir, ext)))
        image_paths = sorted(set(image_paths))

        if not image_paths:
            print(f"未找到图片: {image_dir}")
            return []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"找到 {len(image_paths)} 张图片")
        results = []
        for img_path in tqdm(image_paths, desc='推理'):
            try:
                if output_dir:
                    out_name = os.path.splitext(os.path.basename(img_path))[0] + "_result.jpg"
                    out_path = os.path.join(output_dir, out_name)
                    r = self.analyze_and_visualize(img_path, out_path)
                else:
                    r = self.analyze(img_path)
                results.append(r)
            except Exception as e:
                print(f"  错误: {img_path} - {e}")

        print(f"完成: {len(results)}/{len(image_paths)}")
        return results


def main(args):
    if not os.path.exists(args.weights):
        print(f"错误: 权重不存在: {args.weights}")
        sys.exit(1)

    analyzer = LayoutAnalyzer(args.weights, args.threshold, args.device,
                              args.backbone, args.detector, args.img_size)

    if os.path.isfile(args.input):
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)   # 否则写结果时会 FileNotFoundError
        result = analyzer.analyze_and_visualize(
            args.input,
            os.path.join(args.output_dir, "result.jpg") if args.output_dir else None
        )
        print(f"检测到 {len(result['boxes'])} 个版面元素")
        for i, (cls, score) in enumerate(zip(
            [LAYOUT_CLASSES.get(c, str(c)) for c in result['classes']],
            result['scores']
        )):
            print(f"  {i+1}. {cls} ({score:.3f})")

    elif os.path.isdir(args.input):
        analyzer.analyze_batch(args.input, args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default=None, choices=["resnet50", "swin_t"])
    parser.add_argument("--detector", type=str, default=None, choices=["rtdetr", "frcnn"],
                        help="默认从 checkpoint 的元信息读取")
    parser.add_argument("--img_size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
