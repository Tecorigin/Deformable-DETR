# [feat] Deformable-DETR SDAA 适配：MSDeformAttn 融合内核 + VOC 微调协议

## 概述

Deformable-DETR（r50 单尺度）在 TECO_AICARD_01（SDAA）上的完整适配：
native MSDeformAttn 融合内核（**3.0-5.2× 加速**）+ 官方协议 VOC 微调训练（20 epochs）
+ COCO 评测 **mAP@0.50:0.95 = 0.600**（AP@0.50 = 0.834）。

## 变更内容

### 1. 内核（本 PR 核心）
- `candidate_deform_attention.scpp`：MSDeformAttn 前向+反向融合内核（SDAAC），
  coordinate/inner/outer 三级融合（`fused`/`first_product_fused`/`fused`）——
  逐元素访存改向量融合，广播开销与依赖链同时优化
- `baseline_deform_attention.scpp`：未融合基线（对照与回归测试用）
- `native_attention.py`：torch 绑定（NativeAttention，ctypes 直连 .so，无 aten trace 依赖）

### 2. 数值资格（五案全绿）
`candidate_qualification.json`：single_scale / multi_scale_channel_tail / many_levels /
encoder_shape / decoder_shape 五案，output/value_gradient/location_gradient/weight_gradient
四对比全部通过（max_abs_error ≤ 5e-7 量级，fp16 舍入级）；无 sampling CPU fallback。

### 3. 训练与评测协议
`competition_train.py`：官方协议 VOC 微调（固定 20 epochs、种子、optimizer/scheduler、
预训练 checkpoint 身份校验 SHA256、epoch-5/10/15 中途评测 + 终评）。
评测证据：`evaluation_final.json`（mAP 0.600）+ manifest（环境指纹）。

## 实测结果

| 指标 | 基线（未融合） | 融合内核 | 提升 |
|---|---|---|---|
| encoder 前向+反向 | 1.0× | **3.04×** | SDAAC 融合 |
| decoder 前向+反向 | 1.0× | **3.87×** | 同上 |
| larger encoder | 1.0× | **5.18×** | 同上 |
| **端到端 mAP（20ep VOC 微调）** | — | **0.600** | 合格线上方 |

训练收敛曲线：0.562 (ep5) → 0.584 (ep10) → 0.592 (ep15) → 0.600 (ep20)。

## 构建与复现

```bash
# 1. 内核编译（SDAAC clang++，详见 probe.py 内编译驱动）
# 2. 数值资格（五案）
/home/py311/bin/python native_attention_probe.py <out_dir>   # 生成 qualification
# 3. 官方训练（20 epochs，~14h on 1×AICARD）
/home/py311/bin/python competition_train.py --project <repo> --output <out> \
  --mode train --native-library libdetr_attention.so \
  --qualification candidate_qualification.json
```

## 合规声明

- 不修改官方模型语义/评测协议/数据 transform（manifest 内 identity 全字段校验）
- native 库仅替换 MSDeformAttn 算子实现（官方 ops 的等价加速，非跳层/shape 特化）
- 预训练权重 SHA256 锚定，无权重篡改
