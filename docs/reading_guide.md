# ECGTwin 对抗样本生成 Pipeline 代码阅读路线图

## 建议阅读顺序

```
① ecgtwin_utils.py          ← 先看接口，理解整体能力
② vae_model.py              ← 理解 latent 空间
③ DiT_ECGTwin.py            ← 理解噪声预测器
④ efficientnet_victim.py    ← 理解可微分推理链
⑤ adv_generate.py           ← 核心：boundary-guided 生成（最重要）
⑥ finetune.py               ← 训练闭环
⑦ run_online_training.py    ← 端到端 pipeline
```

---

## 第一层：基础组件（理解"积木"）

### 1. VAE 编解码器 — 理解潜空间
- **文件**: `model/ECGTwin/module/vae_model.py` (227 行)
- **核心**: ECG (B,1024,12) <-> Latent (B,4,128) 的压缩/重建
- **重点**:
  - `VAE_Encoder.forward()` (line 98): 编码 + 重参数化技巧
  - `VAE_Decoder.forward()` (line 178): 解码重建
  - 关键细节: `x *= 0.18215` 缩放因子，后面对抗生成时必须 out-of-place 处理

### 2. 条件嵌入 — 理解扩散模型的"输入"
- **文件**: `model/ECGTwin/module/Embedder.py`
  - `TimestepEmbedder` (line 55): 正弦位置编码，标量时间步 -> 256维向量
  - `RoPEEmbedder` (line 34): 旋转位置编码
  - `PatientInfoEmbedder` (line 95): 患者信息 (hr, age, sex) 嵌入
- **文件**: `model/ECGTwin/utils/data_utils.py`
  - 文本嵌入 (Nomic v1.5)、患者信息归一化

---

## 第二层：扩散模型核心（理解"生成器"）

### 3. DiT 噪声预测器 — 核心架构
- **文件**: `model/ECGTwin/module/DiT_ECGTwin.py` (253 行)
- **架构**: 7层 DiT Block, hidden_dim=256, 8 heads
- **重点**:
  - `DiTBlock_ECGTwin` (line 15): 自注意力 + 交叉注意力 + adaLN 调制
  - `DiT_ECGTwin.forward()` (line 128): 信号 + 时间步 + 文本条件 + 个体基向量 -> 预测噪声
- **数据流**:
  ```
  Input:
    x: (B, 4, 128) latent signal
    t: (B,) timestep
    text_embed: (B, L, 768) 文本嵌入
    p: (B, 3) 患者信息
    base_vector: (B, 256) 个体基向量
  Output:
    predicted_noise: (B, 4, 128)
  ```

### 4. IBExtractor — 个体特征提取
- **文件**: `model/ECGTwin/module/IBExtractor.py` (line 56)
- **功能**: 从参考 ECG 的 latent 中提取 (B,256) 的个体基向量
- **架构**: 3层 encoder (自注意力 + 交叉注意力)
- **目的**: 保持生成 ECG 的个体特征一致性

### 5. DDPM 采样过程 — 正向/逆向扩散
- **逆向去噪**: `model/ECGTwin/utils/inference_utils.py` (line 19) `ddpm_generation()`
- **训练加噪**: `model/ECGTwin/utils/training_utils.py` (line 65)
- **配置**: 1000步, beta in [0.00085, 0.012], 线性 schedule

---

## 第三层：高层封装（理解"接口"）

### 6. ECGTwinWrapper — 一站式封装（最重要的文件之一）
- **文件**: `util/ecgtwin_utils.py` (~445 行)
- **重点方法**:
  - `__init__()` (line 41): 加载 noise_predictor + scheduler + IBExtractor + VAE
  - `prepare_conditions()` (line 203): 组装所有生成条件
  - `ddpm_sample()` (line 295): 标准无梯度采样
  - `ddpm_sample_with_grad()` (line 342): **可微分采样 -- 对抗生成的关键!**
  - `decode_latent()` / `encode_ecg()`: VAE 编解码

---

## 第四层：对抗 Pipeline（理解"攻击"）

### 7. EfficientNet Victim — 可微分受害者模型
- **文件**: `adversarial/efficientnet_victim.py` (278 行)
- **核心链路** `_latent_to_logits()` (line 142):
  ```
  latent -> VAE decode -> lead reorder -> resample -> JIT EfficientNet -> logits
  ```
- **关键技巧**: `_decode_latent_differentiable()` (line 226)
  - 手动迭代 decoder 模块，避免 in-place `/= 0.18215` 阻断梯度
- `set_adapter()` (line 94): 注入 adapter，使"受害者"变为"当前最新模型"

### 8. Boundary-Guided AdvDiff 生成 — 整个项目的核心创新
- **文件**: `adversarial/adv_generate.py` (629 行, **必读**)
- **核心思想**: 不是让模型"分错类"，而是生成模型最不确定的样本 (prob ~ 0.5)
- **关键代码**:
  - `boundary_loss()` (line 121): `(logit**2).mean()` -- 推向决策边界
  - `BoundaryAdvDiffGenerator.generate_batch()` (line 240): 主采样循环
    ```
    x_T ~ N(0,1)
    for t in reversed(timesteps):
        x_{t-1} = DDPM_step(x_t, t)            <- ECGTwin 去噪
        if t in last 20%:
            grad = -grad_latent boundary_loss   <- EfficientNet 梯度引导
            x_{t-1} += scale * ||x_{t-1}|| * normalize(grad)
    accept if target_prob in [0.3, 0.7]
    ```
  - `_compute_boundary_gradient()` (line 163): logit 空间梯度，避免 sigmoid 饱和
  - Noise Sampling Guidance (NSG): 更新初始噪声 x_T 重试

### 9. 标签映射 — 连接 PTBXL 和 EfficientNet
- **文件**: `adversarial/label_mapping.py`
- `SCP_TO_EFFICIENTNET` (line 53): 130+ SCP 代码 -> 77 类索引
- `PROMPT_TO_PATTERN_INDICES` (line 147): 生成 prompt -> 目标索引 (如 "MI_inferior" -> [14,36])

---

## 第五层：训练和评估（理解"闭环"）

### 10. Adapter 架构 — 轻量可训练模块
- **文件**: `adversarial/efficientnet_adapter.py` (150 行)
- **架构**: LayerNorm(77) -> Linear(77->128) -> GELU -> Dropout -> Linear(128->77) -> + frozen_logits
- 仅 ~20K 参数, fc2 零初始化确保安全启动

### 11. 微调训练
- **文件**: `adversarial/finetune.py` (line 180 `train_adapter()`)
- WeightedRandomSampler: 生成样本 2x 权重
- BCEWithLogitsLoss + pos_weight 处理类不平衡

### 12. 在线训练脚本 — 端到端 pipeline
- **文件**: `scripts/run_online_training.py`
- `SampleBuffer` (line 93): FIFO 对抗样本缓冲区
- 主循环: 每 epoch 生成 30 个新样本 -> 加入 buffer -> 训练 1 epoch

---

## 整体数据流

```
Reference ECG (B, 1024, 12)
       |
       v
  VAE Encoder -> Latent (B, 4, 128)
       |
       +---> IBExtractor -> base_vector (B, 256)    [个体特征]
       |
       v
  DDPM Reverse Diffusion (1000 steps)
  + Boundary Gradient Guidance (last 20%)            [对抗引导]
       |
       v
  Generated Latent (B, 4, 128)
       |
       v
  VAE Decoder -> ECG (B, 1024, 12)
       |
       v
  Lead Reorder + Resample -> (B, 12, 2500)
       |
       v
  EfficientNet + Adapter -> Logits (B, 77) -> Probs
       |
       v
  Accept if target_prob in [0.3, 0.7]
       |
       v
  FIFO Buffer -> Combined Training -> Adapter Update
```

---

## 关键设计决策

1. **Boundary vs Misclassification**: 生成 prob~0.5 的"难样本"而非"错分样本"
2. **Logit-space loss**: `logit**2` 避免 sigmoid 饱和区梯度消失
3. **Latent-relative step sizing**: 梯度步长按 latent L2 范数归一化
4. **Out-of-place VAE decode**: 手动迭代避免 in-place 操作阻断梯度
5. **Frozen backbone + residual adapter**: 20K 参数高效微调
6. **Weighted sampling**: 生成样本 2x 权重强调难样本
