# ModelScope 模型卡

这两个文件是 ModelScope 上两个模型仓库的 README 原文，**改完直接上传即可**（内容会被原样显示在模型页面上）。

| 文件 | 对应仓库 |
|------|---------|
| `rtdetr-r50vd-doclaynet-layout.md` | [Brilliantccc/rtdetr-r50vd-doclaynet-layout](https://www.modelscope.cn/models/Brilliantccc/rtdetr-r50vd-doclaynet-layout) |
| `faster-rcnn-r50fpn-doclaynet-layout.md` | [Brilliantccc/faster-rcnn-r50fpn-doclaynet-layout](https://www.modelscope.cn/models/Brilliantccc/faster-rcnn-r50fpn-doclaynet-layout) |

文件名与仓库名一一对应，便于对照。

## 改完怎么上传

```python
from modelscope.hub.api import HubApi

api = HubApi()
api.upload_folder(
    repo_id="Brilliantccc/rtdetr-r50vd-doclaynet-layout",
    folder_path="<含本文件的目录>",      # 需先把 .md 重命名为 README.md
)
```

`upload_folder` 按内容哈希跳过未变的文件，所以只有改动的 README 会被提交，权重不会重传（实测 1 秒左右）。

两个模型仓库的权重**只存在于 ModelScope**，本地不需要保留副本 —— 需要时用
`modelscope download --model Brilliantccc/rtdetr-r50vd-doclaynet-layout` 拉取，或按
`docs/项目计划.md` 里的说明从训练 checkpoint 重新导出。

## 权重是什么形态

| 仓库 | 格式 | 大小 |
|------|------|------|
| rtdetr-r50vd-doclaynet-layout | 原生 transformers 目录（`config.json` + `model.safetensors` + `preprocessor_config.json`） | 171.6 MB |
| faster-rcnn-r50fpn-doclaynet-layout | `.pth`（本项目 `LayoutDetector` 的 state_dict，已去掉优化器状态） | 166.0 MB |

> ⚠ RT-DETR 那份**必须用 transformers 4.48~4.49 加载**。5.x 重构了 RT-DETR 的内部键名，
> 加载会大量 missing 且不报错（静默随机初始化）。`inference.py` / `evaluate.py` 里已有
> 版本检查会直接拦住。

## 改仓库显示名（页面标题）

SDK 没有暴露这个接口（`update_repo_settings` 只支持 studio/skill），要直接调底层端点：

```python
from modelscope.hub.api import HubApi

api = HubApi()
api._api.openapi._request(
    "PATCH",
    "/models/Brilliantccc/rtdetr-r50vd-doclaynet-layout/settings",
    json_body={"display_name": "RT-DETR-R50vd Document Layout Analysis (DocLayNet, 11 classes)"},
)
```

注意字段名是 `display_name`，不是创建仓库时用的 `ChineseName`（两个端点接受的字段集不同）。
同一个端点也可以传 `license`、`description`。
