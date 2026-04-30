# 去 IBE 的 ECGTwin Diffusion 增强器推荐计划

> 面向毕设：用魔改 ECGTwin 生成 ECG 数据，增强 PTB-XL 上训练的 EfficientNetV2 super5 多标签疾病识别模型，并在 PhysioNet/CinC 2021 多中心数据集上测试。
>
> 日期：2026-04-30
> 硬件：RTX 4090D + 15 核 CPU + 80GB RAM

---

## 1. 总结结论

当前确认后的推荐主线是 **方案 B**：

```text
PTB-XL super5 真实样本
    -> VAE latent
    -> 去 IBE 的 class-conditional latent diffusion
    -> 保留 ECGTwin 原 nomic prompt cross-attention 作为辅助条件
    -> 生成 super5 条件 ECG
    -> 训练 real vs real+synth EfficientNet1DV2
    -> PTB-XL fold10 + PhysioNet 2021 七中心测试
```

不建议把 MIMIC-only 作为主线。MIMIC-IV-ECG 数据量大，适合做预训练，但它的 super5 标签需要从报告文本映射，工程量和标签噪声都会增加。毕设时间紧时，主线应优先保证闭环、可解释、可复现实验。

关于 `nomic` 文本编码器的最终执行口径：

```text
本轮采用方案 B：保留 nomic prompt embedding，但只作为辅助语义条件。
super5 multi-hot 仍然是主条件，不能被自然语言 prompt 替代。
```

原因是本文目标不是自由文本到 ECG，而是 super5 疾病类别可控增强。直接使用 `y_super5` multi-hot 类别向量更短、更稳、更容易和 EfficientNetV2 下游任务闭环。`nomic` 的价值是兼容 ECGTwin 原始 text-to-ECG 设计、方便论文承接“文本注入/提示词生成”叙事，同时为系统演示保留 prompt 接口。

因此保留原 DiT 的 `text_projector + CrossAttention` 路径：

```text
super5 multi-hot label -> 主条件，进 AdaLN modulation
nomic prompt embedding -> 辅助语义条件，走 ECGTwin 原 cross-attention
center token           -> 可选中心风格条件，进 AdaLN modulation
```

这样 prompt 写错或 nomic 控制较弱时，`y_super5` 仍然能兜住类别语义。

Center token 暂时不阻塞方案 B 主线。后续如果加，仍不建议做成真正依赖 `nomic` tokenizer 的 textual inversion。更推荐做成 **Textual-Inversion-style center token**：冻结 VAE 和 diffusion 主干，只学习一个或一组可学习的中心风格向量，例如 `<center_nin>`、`<center_extra>`。采样接口可以像 prompt 一样选择 center token，但内部是 modulation-space token，不经过 `nomic` 文本编码器。

一句话论文故事：

> 本文在 ECGTwin 的 VAE-LDM 框架上移除面向患者个性化的 IBE 模块，引入 super5 类别条件作为主控制信号，同时保留 nomic 文本提示词交叉注意力作为辅助语义条件，并结合 CFG/DDIM 采样，将 ECGTwin 改造为面向诊断超类的数据增强器；随后用生成样本增强 PTB-XL 训练的 EfficientNetV2 super5 多标签分类器，并在 PhysioNet/CinC 2021 多中心数据集上验证跨中心泛化能力。少量目标中心 ECG 学习 center token 可作为后续扩展，用于目标医院风格适配。

---

## 2. 架构选择

### 2.1 删除的模块

删除或在新 fork 中绕过以下路径：

- `IBExtractor`
- `ibe_model.pth`
- `base_vector`
- patient-level reference ECG conditioning
- 原 ECGTwin 的 text prompt 作为主控制源

原因：IBE 是 patient identity / personalization 分支，不是 super5 疾病类别增强的核心。继续保留 IBE 会让论文故事变成“患者个性化 + 类别增强”的混合任务，答辩时更难解释。

### 2.2 保留的模块

保留：

- ECGTwin VAE encoder/decoder
- latent diffusion / DiT backbone
- DDPM 训练目标
- DDIM 推理加速
- 已有 `nomic` 文本注入实现作为历史实现/可选对照，不作为主线依赖

VAE 建议冻结，不重训。创新点放在 diffusion 条件控制和下游增强效果。

### 2.3 新增的模块

新增 super5 class conditioning：

```python
class_embedder = nn.Linear(5, hidden_size)
c = timestep_embed(t) + class_embedder(y_super5)
```

其中 `y_super5` 是 PTB-XL 的 5 维 multi-hot 标签：

```text
CD / HYP / MI / NORM / STTC
```

训练时做 classifier-free guidance drop：

```python
if rand < 0.1:
    y_super5 = zeros_like(y_super5)
```

推理时用 CFG：

```python
eps = (1 + scale) * eps_cond - scale * eps_uncond
```

默认：

- `drop_prob = 0.1`
- `guidance_scale = 3.0`
- `DDIM steps = 50`

### 2.4 Center Token：可选但推荐的第二阶段

你的 center token 想法可以做，但要收敛到一个轻量版本：

```text
冻结 VAE
冻结 no-IBE class-conditional diffusion
只训练目标中心的 center token
```

推荐实现不是“把 `<center_x>` 字符串送进 nomic”，而是在 DiT 条件调制里增加一个可学习向量：

```python
class_embed = class_embedder(y_super5)
center_embed = center_token(center_id)        # 或 per-block center token
c = timestep_embed(t) + class_embed + center_embed
```

更强但仍轻量的版本是 per-block token：

```python
c_block_i = timestep_embed(t) + class_embed + center_token_i(center_id)
```

这和 textual inversion 的相似点是：

- 主生成器冻结。
- 只用少量目标域样本学习一个新 token。
- 推理时通过选择 token 控制生成风格。

和真正文本反演的区别是：

- token 不进入 `nomic` 文本编码器。
- token 直接进入 DiT modulation / AdaLN 条件通道。
- 论文里建议表述为“借鉴文本反演思想的中心风格 token”，不要写成“本文训练了 nomic 文本反演词向量”。

训练目标仍可用 diffusion MSE：

```text
目标中心少量 ECG -> VAE latent z
super5 label y
随机 t, eps
z_t = add_noise(z, eps, t)
冻结 diffusion(z_t, t, y, center_token) -> eps_pred
只更新 center_token，使 MSE(eps_pred, eps) 下降
```

目标中心样本量建议：

```text
K = 200-500 条 / center
```

生成样本量不要一开始无限放大。推荐先从：

```text
synth = 3x-10x K
```

开始，并用 fold9 / target-center validation gate 过滤。

### 2.5 如果坚持加入 nomic prompt

可以加入，推荐作为 B 档方案：

```text
A 档：class-only no-IBE diffusion              # 最稳，必须先跑通
B 档：class + nomic prompt no-IBE diffusion    # 保留 ECGTwin text path
C 档：class + nomic prompt + center token      # 最完整，但只在 A/B 跑通后做
```

不要让 B/C 档反过来阻塞 A 档。毕业设计最终结果可以用 A 档兜底，B/C 档作为增强实验或系统展示。

#### 2.5.1 代码复用方式

ECGTwin 原 repo 中 `DiT_ECGTwin` 已经有成熟 text path：

```python
self.text_projector = nn.Linear(text_embed_dim + pat_info_length, hidden_size)
...
p = p.unsqueeze(1).repeat(1, text_embed.shape[1], 1)
c2 = torch.concat([p, text_embed], dim=-1)
c2 = self.text_projector(c2)
c2 = self.c_pos_embed(c2)
x = block(x, c, c2, text_embed_mask)
```

因此最小改造不是重写 cross-attention，而是 fork 一个 no-IBE 版本：

```python
# 原版
c = timestep_embed(t) + ib_projector(base_vector)
c2 = text_projector(concat(pat_info, text_embed))

# no-IBE + class/text/center
c = timestep_embed(t) + class_embedder(y_super5)
if center_token is not None:
    c = c + center_embedder(center_id)
c2 = text_projector(concat(pat_info, text_embed))
```

也就是说：

- 保留 `x_embedder`、`t_embedder`、`text_projector`、`c_pos_embed`、`blocks`、`final_layer`。
- 删除 `IBExtractor`、`ibe_model.pth`、`base_vector` 输入。
- `ib_projector(base_vector)` 的位置改成 `class_embedder(y_super5) + optional center_token`。
- `text_embed` 继续用 `(B, L, 768)`，`text_embed_mask` 继续用 `(B, L)`，复用原 collate 的 padding 逻辑。

#### 2.5.2 prompt 设计

固定 5 类 prompt 表，不在训练中自由发挥：

| super5 | prompt |
|---|---|
| NORM | `normal ecg|sinus rhythm` |
| MI | `myocardial infarction|pathological q wave` |
| STTC | `st-t change|st segment abnormality|t wave abnormality` |
| CD | `conduction disturbance|bundle branch block` |
| HYP | `ventricular hypertrophy|left ventricular hypertrophy` |

多标签样本直接拼接：

```text
MI + STTC -> myocardial infarction|pathological q wave|st-t change|st segment abnormality|t wave abnormality
```

推理时用户看到的是类似 prompt 的接口：

```text
"myocardial infarction|st-t change|<center_nin>"
```

但内部建议拆成三个结构化条件：

```text
y_super5 = [0,0,1,0,1]
text_embed = nomic(prompt_without_center_token)
center_token = "nin"
```

`<center_nin>` 不交给 nomic 编码。

#### 2.5.3 embedding 计算策略

不要在训练循环里在线跑 nomic。正确做法：

```text
构建 dataset 时预计算 text_embed
保存到 .pt
训练 diffusion 时只读 text_embed tensor
```

原因：

- 原 ECGTwin 数据就是 `paired_Mimic_vae_multi_nomic.pt` 这种预编码格式。
- 在线跑文本编码器会拖慢 DataLoader，并引入额外环境依赖。
- super5 prompt 只有固定组合，缓存成本很低。

Classifier-free guidance 的 text dropout 建议用一个 learnable/null prompt，而不是全 mask：

```text
text_embed = null_text_embed, text_embed_mask = ones
```

不要把 `text_embed_mask` 全置 0，因为原 `CrossAttention` 会把所有 key 都 mask 成 `-inf`，softmax 后容易产生 NaN。

---

## 3. 数据路线

### 3.1 主线数据：PTB-XL

PTB-XL 是主线训练数据，理由：

- 官方提供 5 个 diagnostic superclasses，正好对应 super5。
- 已有官方 fold：fold 1-8 train，fold 9 val，fold 10 test。
- 下游 EfficientNetV2 super5 模型已经基于 PTB-XL pipeline。
- 生成器和分类器同源，增强实验更容易讲清。

训练数据：

```text
PTB-XL fold 1-8 latent + super5 multi-hot label
```

验证数据：

```text
PTB-XL fold 9 latent + super5 multi-hot label
```

最终测试：

```text
PTB-XL fold 10
PhysioNet 2021 七中心，必须排除 ptb-xl shard
```

### 3.2 MIMIC 的定位

MIMIC 不作为最小主线。

可选用法：

```text
MIMIC diffusion pretrain -> PTB-XL fine-tune
```

但只有在 PTB-XL 主线已经跑通、还有时间时再做。

不建议：

```text
MIMIC-only diffusion -> 直接增强 PTB-XL classifier
```

原因：

- MIMIC 标签来自报告文本，super5 映射噪声更大。
- MIMIC 是单中心 BIDMC 域，和 PTB-XL/PN2021 的标签协议不完全一致。
- 毕设时间紧时，MIMIC 会把工程重点从“生成增强有效性”拖到“标签清洗和域适配”。

### 3.3 目标中心 few-shot 数据的定位

center token 的目标数据建议来自 PhysioNet 2021 的目标中心，而不是 MIMIC。

推荐选择：

```text
PN2021 某个目标中心 K=200-500 条 ECG
```

用途：

```text
只训练 center token，不训练 VAE，不微调 diffusion 主干
```

这条路线更适合讲“跨中心风格适配”：

- PTB-XL 负责疾病类别语义。
- no-IBE diffusion 负责 ECG 生成能力。
- 少量 PN2021 目标中心 ECG 负责医院/设备/采集风格。
- 最终看 target-center 和 PN2021 七中心指标。

注意：如果目标中心样本参与 center token 训练，最终评估必须把该中心拆成 enroll split 和 eval split：

```text
target center K=200-500 enroll records -> train center token
target center remaining records         -> evaluate classifier
```

不能把用于 center token 训练的 ECG 再当作该中心最终测试样本。

### 3.4 nomic prompt 数据的定位

如果启用 B/C 档 prompt path，PTB-XL dataset 每条样本需要同时包含：

```python
{
    "data": latent,                  # (4, 128)
    "label": {
        "super5": multi_hot,         # (5,)
        "text": prompt_str,
        "text_embed": nomic_embed,   # (L, 768)
        "text_embed_mask": mask,     # collate 后产生
        "hr": hr_or_default,
        "age": age_or_default,
        "sex": sex_or_default,
    }
}
```

如果 PTB-XL 某些样本没有可靠 `hr/age/sex`，可以用默认值保持和 ECGTwin repo 接口兼容：

```text
hr=70, age=60, sex=F
```

这三个 patient info 在新版方法中不是主控制变量，只是为了复用 `text_projector(text_embed + pat_info)` 的输入形状。论文里不要把它们作为个体化条件 claim。

prompt 来源优先级：

```text
super5 label -> 固定 prompt 表 -> nomic embedding
```

不要直接使用原始报告自由文本作为主线 prompt，否则 PTB-XL / PN2021 / MIMIC 的文本风格差异会重新引入标签噪声。

---

## 4. 实施步骤

### Step 1：建立最小 ECGTwin fork

建议新增目录：

```text
methods/ecgtwin_class_super5/
```

不要直接破坏原始 `model/ECGTwin/`。原模型保留作对照和复用 checkpoint。

最小文件：

```text
methods/ecgtwin_class_super5/
├── model.py              # 去 IBE 的 DiT class conditional diffusion
├── prompt_table.py       # optional: super5 -> prompt -> nomic cache
├── dataset.py            # PTB-XL latent + super5 label + optional text_embed
├── train.py              # diffusion training
├── sample.py             # DDIM + CFG sampling
├── center_token.py       # optional: textual-inversion-style center token
├── train_center_token.py # optional: freeze diffusion, learn target-center token
├── eval_synth.py         # synth sanity + victim score
└── README.md
```

### Step 2：训练去 IBE diffusion

训练目标仍是标准 noise prediction MSE：

```text
latent z
noise eps
timestep t
z_t = add_noise(z, eps, t)
model(z_t, t, y_super5) -> eps_pred
loss = MSE(eps_pred, eps)
```

如果启用 nomic prompt：

```text
model(z_t, t, y_super5, text_embed, text_mask, pat_info) -> eps_pred
```

但训练损失不变，prompt 只是额外条件。

默认训练配置：

```yaml
epochs: 100
batch_size: 512
lr: 1e-4
weight_decay: 1e-4
amp: true
num_workers: 12
pin_memory: true
persistent_workers: true
compile: true after smoke test
ema_decay: 0.999
cfg_drop_prob: 0.1
text_drop_prob: 0.1          # only if prompt path is enabled
condition_mode: class_text   # class_only | class_text | class_text_center
log_every: 50
sample_every_epoch: true
save_every_epoch: true
```

推荐训练顺序：

```text
先跑 class_only
再从 class_only 初始化 class_text，打开 text_projector/cross-attn
最后冻结主干或小 lr 训练 center token
```

如果时间极紧，不要从零训练 class_text。优先复用 class_only 权重，再加 text path 微调 20-50 epoch。

#### Step 2.1：训练日志和论文材料落盘

每次 diffusion 训练必须保存以下文件，方便后续写毕设论文和画实验图：

```text
/root/autodl-tmp/ecgtwin_class_super5/<run_name>/
├── config.yaml 或 run_config.json
├── train.log
├── metrics.jsonl          # step/epoch/loss/lr/grad_norm/val_loss
├── loss_curve.csv
├── loss_curve.png
├── checkpoints/
│   ├── latest.pt
│   └── best.pt
├── samples/
│   ├── epoch_*.npz
│   └── final_synth.npz
└── figures/
    └── synthetic_12lead_*.png
```

下游 EfficientNetV2 增强训练也需要保存：

```text
training_log.json
train_result.json
eval_result.json
baseline_vs_synth_metrics.csv
pn2021_per_center_delta.csv
```

这些文件对应论文里最常用的材料：

- loss 曲线：证明 diffusion 训练收敛。
- 12 导联合成图：展示生成质量。
- fold9/fold10 表：证明合成样本没有破坏源域。
- PN2021 per-center/per-class 表：支撑跨中心泛化分析。

### Step 3：可选训练 Center Token

这一步放在主 diffusion 训练完成之后。不要在一开始就做，避免主线被拖慢。

输入：

```text
目标中心 ECG K=200-500
VAE latent z
super5 multi-hot y
```

冻结：

```text
VAE encoder/decoder
no-IBE class-conditional diffusion
```

只训练：

```text
center_token 参数
```

建议配置：

```yaml
total_steps: 1000-5000
batch_size: 8-32
lr: 1e-3 to 5e-3
amp: true
token_type: per_block
num_blocks: match DiT depth
init: zeros
regularization: optional isolation loss against PTB-XL
```

本地已有 `CenterTokenPerBlock` 方向的经验：v1 单 token 容易接近 no-op；v2 per-block token 在 K=200 下有更明显的信号级变化和小幅下游收益。因此如果时间允许，直接做 per-block，不再从单 token 开始。

### Step 4：生成 synthetic pool

每类先生成：

```text
N = 1000 条 / class
共 5000 条 synthetic ECG
```

如果时间更紧，先用：

```text
N = 300 条 / class
共 1500 条 synthetic ECG
```

生成流程：

```text
random latent noise
    -> DDIM 50 steps + CFG
    -> optional nomic prompt embedding
    -> optional center token: <center_x>
    -> VAE decode
    -> ECGTwin lead order 转 PTB-XL lead order
    -> 1024 点重采样到 1000 点
    -> 保存 npz
```

生成集分两类保存：

```text
class_synth/                      # 只有 super5 类别条件
class_text_synth/                 # super5 + nomic prompt
center_synth/<center_name>/       # super5 + nomic prompt + center token
```

### Step 5：训练 EfficientNetV2 增强模型

复用现有：

```text
scripts/triple_labels/train_ptbxl.py --scheme super5
```

新增或扩展一个 synthetic dataset loader，使训练数据变成：

```text
real PTB-XL fold 1-8 + synthetic ECG
```

先扫三个比例：

```text
real : synth = 1 : 0.25
real : synth = 1 : 0.5
real : synth = 1 : 1.0
```

只用 fold 9 选择最佳比例，不能用 fold10 或 PN2021 反调。

center token 分支额外对比：

```text
Baseline:       real PTB-XL
ClassSynth:     real PTB-XL + class_synth
CenterSynth:    real PTB-XL + center_synth/<target_center>
```

这里的关键不是保证七中心平均全部上涨，而是看：

```text
target center 是否提升
同源中心是否提升
PTB-XL fold10 是否不受伤
```

### Step 6：最终测试

对比两组：

```text
Baseline: EfficientNetV2 trained on real PTB-XL
Ours:     EfficientNetV2 trained on real PTB-XL + synthetic ECG
```

报告：

- PTB-XL fold10 macro AUROC / macro AUPRC
- PN2021 七中心 macro AUROC / macro AUPRC
- per-class AUROC / AUPRC
- per-center delta
- synthetic ECG 12-lead 可视化
- CFG scale 可控性图
- center token style ablation：无 token vs 目标 center token

---

## 5. 4090D 时间预算

### 5.1 推荐主线：PTB-XL only

在 VAE latent 已经缓存的前提下：

| 阶段 | 估计时间 |
|---|---:|
| 代码改造 + smoke test | 0.5-1 天 |
| diffusion 100 epoch | 30-90 分钟 |
| optional nomic prompt cache + class_text 微调 | 0.5-2 小时 |
| optional center token K=200-500 | 10-45 分钟 / center |
| synthetic pool 生成 | 20-60 分钟 |
| real+synth classifier 训练 3 个比例 | 30-90 分钟 |
| PTB-XL + PN2021 eval | 30-90 分钟 |
| 图表和结果整理 | 0.5-1 天 |

现实日历：

```text
3-5 天可以跑出最小可答辩闭环
```

如果加入 1-3 个目标中心的 center token：

```text
额外增加 0.5-1.5 天现实日历时间
```

这部分 GPU 训练本身不长，主要时间会花在目标中心拆分、生成样本管理、表格对比和画图上。

如果加入 nomic prompt path：

```text
额外增加 0.5-1 天现实日历时间
```

主要风险不是算力，而是 prompt 表、缓存格式、unconditional/null prompt、以及 text path 消融表需要整理。

### 5.2 可选路线：MIMIC pretrain + PTB-XL fine-tune

如果 MIMIC latent 已经缓存：

| 阶段 | 估计时间 |
|---|---:|
| MIMIC diffusion pretrain 30 epoch | 2-5 小时 |
| PTB-XL fine-tune | 30-90 分钟 |
| 后续增强和评测 | 2-5 小时 |

如果 MIMIC 需要从 raw WFDB 重新预处理和 VAE encode：

```text
总时间可能变成 1-2 天，且更容易卡在 IO / 标签清洗 / cache 上
```

### 5.3 不建议路线：从头训练 VAE

不建议从头训练 VAE。

原因：

- 时间至少按天计算。
- 调参风险高。
- 毕设创新点不在 VAE 重建，而在 class-conditional diffusion augmentation。

### 5.4 推荐工期压缩版

#### 8 小时冲刺版

8 小时可以做出 **方案 B 的最小可验证闭环**，但不应期待完成完整论文级实验矩阵。推荐目标是：

```text
能证明 no-IBE + class_text diffusion 代码跑通，
能生成一小批 synthetic ECG，
能完成 1 个 real+synth 比例的 EfficientNetV2 训练和 PN2021 AUROC/AUPRC 评估。
```

8 小时内建议严格按下面优先级执行：

| 时间段 | 任务 | 产物 |
|---|---|---|
| 0-1h | 新建 `methods/ecgtwin_class_super5/`，复制/改造 DiT no-IBE + dataset skeleton | forward smoke test |
| 1-2h | 构建 PTB-XL latent single-sample dataset + fixed prompt/nomic cache | 小样本 dataloader |
| 2-3h | 训练脚本 AMP/EMA/logging 跑通，batch=512 或自动降 batch | `metrics.jsonl` / loss |
| 3-4.5h | 短训 diffusion：先 10-30 epoch 或固定 step budget | `latest.pt` / loss curve |
| 4.5-5.5h | DDIM 50 或更少步数生成小池：每类 100-300 条 | synth `.npz` + 12-lead 图 |
| 5.5-7h | EfficientNetV2 跑 1 个比例，例如 real:synth=1:0.25 | classifier ckpt |
| 7-8h | PN2021 7-center eval + 汇总 AUROC/AUPRC | `eval_result.json` / delta 表 |

8 小时内建议暂时不做：

- MIMIC pretrain。
- center token。
- 多 synthetic ratio 全扫描。
- 大规模每类 1000+ 采样。
- 复杂医学数字特征门控。

如果 8 小时冲刺里 class_text 出现不稳定或 NaN，立刻 fallback：

```text
保留代码中的 prompt cache，
训练时关掉 cross-attn 或使用 null_text_embed，
先得到 class_only 结果作为答辩兜底。
```

这个 fallback 不会破坏论文故事，因为可以表述为：

> 结构化 super5 条件是主控制信号，文本提示词是兼容 ECGTwin 原框架的辅助条件；当 prompt path 质量不足时，系统可退化为更稳定的类别条件生成器。

如果毕业设计时间很紧，按这个顺序推进：

```text
Day 1: no-IBE class diffusion smoke test + 训练 PTB-XL
Day 2: 可选加入 nomic prompt cache + class_text 微调；同时生成 class_synth
Day 3: 训练 1-3 个 real:synth 比例 + PTB-XL fold10 / PN2021 评测
Day 4: 选 1 个目标中心做 K=200 center token
Day 5: center_synth 下游对比 + 图表 + 写论文实验小节
```

MIMIC pretrain 放到所有核心结果跑通之后。不要让 MIMIC 阻塞主线。

---

## 6. “额外训练 diffusion 增强 EfficientNetV2”成功概率评估

这里的“成功”要拆开看，不能只问一个总概率。

### 6.1 跑通完整系统闭环

估计概率：

```text
80%-90%
```

原因：

- VAE checkpoint 已有。
- EfficientNetV2 super5 pipeline 已有。
- PTB-XL fold 和 super5 标签 pipeline 已有。
- 需要新增的主要是去 IBE DiT 训练和 synthetic loader。

只要不把 MIMIC、Streamlit、TensorRT、复杂消融都塞进主线，闭环概率很高。

### 6.2 diffusion 生成“看起来像 ECG”的样本

估计概率：

```text
70%-85%
```

原因：

- latent diffusion 比 raw waveform diffusion 更稳。
- 冻结 ECGTwin VAE 可以避免从头学习 waveform reconstruction。
- PTB-XL 数据量约 2 万条，训练 5 类粗粒度生成器够用。

主要风险：

- 生成波形幅值或 lead consistency 不稳定。
- 少数类如 HYP/CD 的临床形态不够强。
- CFG scale 太高导致 diversity 降低。

### 6.3 synthetic 数据让 PTB-XL fold10 指标提升

估计概率：

```text
35%-50%
```

这里不能保证一定提升。原因是现有 EfficientNetV2 super5 baseline 已经较强，PTB-XL fold10 in-domain macro AUROC 约 0.90 以上，提升空间有限。

更现实的目标不是大幅提升，而是：

```text
fold10 不明显下降，同时弱类或 AUPRC 有小幅提升
```

可接受结果：

- macro AUROC 持平或 +0.2pp 到 +1.0pp
- HYP/CD/STTC 某些类 AUPRC 提升
- fold9 选择的 synthetic ratio 在 fold10 不崩

### 6.4 synthetic 数据让 PN2021 跨中心指标提升

估计概率：

```text
平均 macro 提升：25%-40%
至少若干中心/若干类提升：60%-75%
```

这比 PTB-XL fold10 更有论文价值，但也更不稳定。

原因：

- PN2021 是多中心外部测试，确实存在 domain gap。
- 生成样本可能增加形态多样性，从而帮助跨中心泛化。
- 但如果 synthetic 数据只复制 PTB-XL 域，可能对 PN2021 平均指标帮助有限。

推荐论文汇报方式：

```text
主表报告 overall macro
重点分析 per-center / per-class delta
```

不要只押注“PN2021 七中心平均一定提升”。更稳的说法是：

> 生成增强对跨中心泛化的收益具有类别和中心依赖性，因此本文引入 fold9 validation gate 和 per-class trust gate，避免低质量合成样本污染训练。

### 6.5 center token 提升目标中心风格匹配

估计概率：

```text
center token 让合成样本更像目标中心：55%-70%
center token 带来目标中心/同源中心小幅分类收益：35%-55%
七中心平均显著提升：20%-35%
```

这个模块有论文价值，但不能过度承诺。更稳的目标是：

```text
目标中心或同源中心出现小幅提升
PTB-XL fold10 不下降
能通过无 token vs center token 的可视化和 style classifier 说明风格确实变化
```

本地已有的 CenterToken v2 记录支持这个判断：per-block token 比单 token 更有效，曾在部分中心/类别上观察到小幅下游收益，但平均 AUROC 提升幅度不大。因此它适合作为“跨中心风格适配增强”的第二创新点，不适合作为唯一主线。

### 6.6 nomic prompt 提升类别可控性

估计概率：

```text
prompt path 让系统展示更像 ECGTwin 原始 text-to-ECG：70%-85%
prompt path 明显提升 super5 class control：25%-45%
prompt path 明显提升下游 EfficientNetV2：15%-35%
```

原因：

- 原 repo 的 text cross-attention 已经实现，工程风险比重写一套低。
- 但 super5 是固定 5 类任务，结构化标签比自然语言 prompt 更可靠。
- prompt 同义词可能引入 MI/STTC/CD 等类别之间的语义纠缠。

因此 B 档实验的成功标准应是：

```text
class_text 不低于 class_only
prompt 修改能产生可观察变化
至少不破坏 fold9 / fold10 / PN2021 主指标
```

### 6.7 最终可答辩成功概率

综合估计：

```text
75%-85%
```

这个概率指：

```text
能完成一个可展示、可解释、可写进论文的 ECG 生成增强系统
```

不是指：

```text
一定在所有指标上击败 baseline
```

如果把“必须 PN2021 七中心平均 macro AUROC 明显提升”作为唯一成功标准，概率会降到：

```text
25%-40%
```

所以毕设验收标准应设计为多证据链：

1. 架构改造完成：去 IBE，class-conditional diffusion。
2. 生成质量成立：可视化 + 基本临床统计 + victim score。
3. 下游增强实验完整：real vs real+synth。
4. 外部测试完整：PN2021 七中心。
5. 可选 nomic prompt：class_only vs class_text 消融。
6. 可选 center token：few-shot 目标中心 enrollment + style ablation。
7. 即使平均提升有限，也能通过 per-class/per-center 分析解释收益和限制。

---

## 7. 风险和 fallback

### 风险 1：synthetic 伤害分类器

处理：

- 用 fold9 选择 synthetic ratio。
- 加 per-class trust gate。
- 只保留 fold9 上不降的类。
- synthetic loss weight 不要超过 real。

### 风险 2：class control 不够强

处理：

- 提高 CFG scale：`3 -> 5`。
- 增加 class embedding 容量。
- 加一个轻量 auxiliary classifier consistency loss，但这放在第二阶段，不放进最小主线。

### 风险 3：HYP/CD 生成不可信

处理：

- 先不强行 claim 所有类都高质量。
- 报告 per-class 结果。
- 如果 HYP/CD 失败，则 trust gate 只使用 NORM/MI/STTC 或 fold9 通过的类。

### 风险 4：MIMIC 训练拖慢

处理：

- MIMIC 只作为 optional pretrain。
- 主线先完成 PTB-XL only。
- 论文里把 MIMIC 写成 future work 或附录实验。

### 风险 5：center token 过拟合目标中心 few-shot 样本

处理：

- 目标中心拆 enroll/eval，enroll 的 K=200-500 不进最终 eval。
- 生成样本先控制在 `3x-10x K`，不要一次性扩到几万条。
- 对比无 token / center token，确认不是简单引入噪声。
- 加 isolation loss 或 PTB-XL anchor，避免 token 破坏疾病类别语义。
- 如果 target center 提升但七中心平均不升，论文就写成“目标中心适配收益”，不要写成“普适跨中心提升”。

### 风险 6：nomic 文本路线和 no-IBE 主线叙事冲突

处理：

- 论文中把 `nomic` 写成前期 ECGTwin 文本条件实现和对照背景。
- 最终方法明确使用结构化 super5 标签作为主条件。
- center token 写成“借鉴 textual inversion 的可学习条件向量”，不写成“训练 nomic 新词”。
- 如果答辩被问为什么不用文本：回答 super5 增强任务的标签空间固定，结构化类别条件更可控，减少自由文本歧义和额外编码器依赖。

如果你硬要加入 nomic，答辩说法改成：

> 本文保留 ECGTwin 的文本交叉注意力路径，用 nomic 对疾病类别提示词进行编码，作为结构化 super5 标签之外的辅助语义条件。由于医学分类增强任务要求类别标签严格可控，最终模型以 super5 multi-hot 作为主条件，以文本提示词作为补充条件，从而兼顾 ECGTwin 原始框架兼容性和下游分类任务稳定性。

### 风险 7：prompt path 引入 NaN 或训练不稳定

处理：

- `text_embed_mask` 不能全 0；unconditional 分支用 `null_text_embed + mask=1`。
- prompt embedding 预计算并保存，不在训练循环里跑 nomic。
- 固定 prompt 表，避免每次采样 prompt 改写导致不可复现。
- 如果 class_text 伤害 fold9，最终主实验退回 class_only，prompt 只做系统 demo。

---

## 8. 最小验收标准

必须交付：

- 去 IBE 的 ECGTwin class-conditional diffusion 训练脚本。
- PTB-XL fold 1-8 训练、fold 9 验证、fold 10 测试。
- synthetic ECG 生成脚本。
- EfficientNetV2 baseline vs real+synth 对比。
- PN2021 七中心评测，排除 `ptb-xl` shard。
- 至少 5 类各 3 条 synthetic 12-lead ECG 图。
- 一张 baseline vs ours 的 AUROC/AUPRC 表。

推荐加做：

- class_only vs class_text 的小消融。
- 1 个目标中心的 K=200 center token enrollment。
- 无 token vs center token 的 synthetic 可视化。
- target center / same-family center 的 per-center delta 表。

可以不做：

- MIMIC pretrain。
- TensorRT。
- Streamlit。
- 大规模消融。
- 从头训练 VAE。
- 在线训练 `nomic` 编码器。
- 真正依赖 `nomic` tokenizer 学 `<center_x>` 新词的文本反演。

---

## 9. 推荐最终论文表述

建议题目或章节标题：

```text
基于去个性化 ECGTwin 与中心风格 Token 的心电生成增强方法
```

核心创新点：

1. 将 ECGTwin 从 patient-level personalization 改造成 super5 class-conditional ECG augmenter。
2. 移除 IBE，简化为 VAE + latent diffusion + class modulation。
3. 保留 ECGTwin 的 nomic 文本交叉注意力路径作为可选辅助语义条件，降低从原 repo 改造的工程风险。
4. 借鉴 textual inversion 思想，冻结生成主干并用少量目标中心 ECG 学习 center token。
5. 使用 CFG/DDIM 提升类别可控性和采样速度。
6. 用 PTB-XL 训练、PhysioNet 2021 多中心测试验证增强对泛化的影响。

推荐主结论写法：

> 实验表明，去 IBE 的类别条件 ECGTwin 可以生成具有可视 ECG 形态和类别相关性的合成样本。保留 nomic 文本交叉注意力路径后，模型能够接受疾病提示词作为辅助条件，但下游增强效果仍主要依赖结构化 super5 标签和合成样本筛选。将生成样本用于 EfficientNetV2 super5 训练后，模型在部分疾病类别和外部中心上取得提升；进一步引入 few-shot center token 后，合成样本可向目标中心风格偏移，并在部分目标中心或类别上带来额外收益。同时，整体平均指标对合成样本质量、混合比例和中心选择敏感，说明医学时序数据生成增强需要类别级与中心级质量筛选。

这个表述稳健，不会把 diffusion augmentation 过度包装成一定提升所有指标。

## 10. 和已有毕设材料的衔接

`final_round` 里的开题报告、中期报告和指导记录已经出现了以下关键词：

```text
VAE / LDM / DiT / nomic 文本注入 / 五分类病理任务 / 跨中心域偏移 / DDIM 加速 / 中心风格学习
```

新版计划和这些材料的对应关系：

- VAE、LDM、DiT：保留为生成模型主干。
- nomic 文本注入：可作为 ECGTwin 兼容 text path 保留，但只作为辅助语义条件。
- 五分类病理任务：落到 PTB-XL diagnostic super5 multi-label。
- 跨中心域偏移：落到 PN2021 七中心外部测试。
- DDIM 加速：落到 50-step sampling。
- 中心风格学习：落到 few-shot center token，而不是 IBE patient personalization。

最终答辩时可以这样解释路线收敛：

> 前期系统复现了 ECGTwin 的文本条件生成流程。为了服务 PTB-XL super5 分类增强任务，最终模型移除患者个性化 IBE 模块，并以结构化疾病类别标签作为主控制条件；同时保留 nomic 文本嵌入交叉注意力作为辅助语义条件，以兼容 ECGTwin 原始提示词生成接口。针对中期报告中发现的跨中心域偏移问题，进一步借鉴文本反演思想，用少量目标中心 ECG 学习轻量中心风格 token，从而在不重训生成主干的情况下实现目标医院风格的合成数据增强。

## 11. 参考 ECGTwin repo 的最低风险落地方案

不要直接改 `model/ECGTwin/` 原文件。新增 fork 目录，复制/改造必要模块：

```text
methods/ecgtwin_class_super5/
├── model.py              # 从 DiT_ECGTwin.py fork，保留 text path，删除 IBE path
├── trainer.py            # 从 ECGTwinTrainer.py + training_utils.py fork
├── dataset.py            # 兼容 ECGTwin 的 text_embed/pat_info collate
├── prompt_table.py       # super5 prompt table + nomic cache
├── sample.py             # DDIM/CFG sampling
└── center_token.py       # optional per-block center token
```

从原 repo 复用：

- `DiTBlock_ECGTwin`
- `CrossAttention`
- `TimestepEmbedder`
- `RoPEEmbedder`
- `PositionalEmbedder`
- `process_pat_info`
- `_pad_text_embed`
- `DDPMScheduler` 配置
- VAE encoder/decoder checkpoint

明确删除或绕过：

- `IBExtractor`
- `ibe_path`
- `base_vector`
- paired reference ECG 训练逻辑

新的 forward 推荐签名：

```python
def forward(
    self,
    x,
    t,
    y_super5,
    text_embed=None,
    text_embed_mask=None,
    pat_info=None,
    center_token=None,
):
    ...
```

实现原则：

```text
class_only 模式：text_embed 使用 null_text_embed，或直接关闭 cross-attn 分支
class_text 模式：复用 ECGTwin 原 text_projector + CrossAttention
class_text_center 模式：在 c 上额外加 center token
```

优先顺序：

```text
1. 跑通 class_only
2. 打开 text path，但只用固定 prompt cache
3. 加 center token
4. 再考虑 MIMIC pretrain
```

这个顺序最符合“参考原 repo 降低风险”：尽量复用 ECGTwin 已有 text cross-attention 和 VAE/DiT 结构，但把最不适合本毕设任务的 IBE 个性化分支拿掉。

## 12. MIMIC 上的训练补全方案

### 12.1 总体判断

MIMIC 建议作为 **no-IBE diffusion 的预训练数据**，不建议替代 PTB-XL 主线。

推荐路线：

```text
MIMIC-IV-ECG single-sample latent pretrain
    -> PTB-XL fold 1-8 fine-tune
    -> PTB-XL / PN2021 生成增强评估
```

不推荐路线：

```text
MIMIC-only generator
    -> 直接生成样本增强 PTB-XL classifier
```

原因：

- MIMIC 样本量大，适合让 DiT 学 ECG latent 分布和报告文本里的粗语义。
- 但 MIMIC super5 标签来自报告正则映射，不如 PTB-XL 官方 diagnostic superclass 干净。
- MIMIC 是 BIDMC 单中心域，直接拿 MIMIC-only generator 增强 PTB-XL，可能把生成分布拉向 MIMIC 域。
- 毕设最终要讲的是“PTB-XL 训练的 EfficientNetV2 在 PN2021 外部中心上的泛化”，因此最终 generator 必须回到 PTB-XL fine-tune 后再采样。

一句话实验定位：

> MIMIC 负责提供大规模 ECG 形态预训练，PTB-XL 负责校准 super5 标签语义，PN2021 负责检验外部中心泛化。

### 12.2 本地可用 MIMIC 数据

当前本地已有 ECGTwin latent cache：

| 文件 | 大小 | 建议用途 |
|---|---:|---|
| `/root/autodl-tmp/ECGTwin_Data/Mimic_vae.pt` | 约 1.8GB | 首选，744,372 条 single-sample latent |
| `/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt` | 约 9.6GB | 原 ECGTwin paired 训练，no-IBE 主线不推荐直接使用 |
| `/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_mix_nomic.pt` | 约 14GB | 更重的 paired/text 版本，不放进毕设主线 |

`Mimic_vae.pt` 每条样本已经包含：

```python
{
    "data": latent,  # (4, 128)
    "label": {
        "subject_id": ...,
        "ecg_time": ...,
        "text": report_text,
        "hr": ...,
        "age": ...,
        "sex": ...,
    }
}
```

因此 MIMIC 阶段不需要重新跑 WFDB 读取和 VAE encode。除非要重新做原始波形质量控制，否则不要回到 raw ECG。

### 12.3 为什么不用作者的 full paired MIMIC 训练

作者 ECGTwin 的 MIMIC 训练核心是：

```text
same patient ECG pair: (reference ECG, target ECG)
reference ECG -> IBE -> base_vector
target ECG latent + noise + text -> diffusion predicts noise
```

我们已经删除 IBE，`reference ECG -> base_vector` 这条路径不存在。继续使用 full paired 数据会带来三个问题：

- 计算量被 pair 数放大：`paired_Mimic_vae_multi_nomic.pt` 约 640 万对，而 single-sample 只有 74 万条。
- 同一患者多次 ECG 会被组合成大量 pair，导致患者样本数多的人被过度采样。
- ref ECG 在 no-IBE 架构里没有明确用途，使用 paired 数据只会增加 I/O 和训练时间。

因此 MIMIC 预训练采用 single-sample 版本：

```text
z = 当前 ECG latent
y = 当前报告映射得到的 super5 label
prompt = y 对应的固定 super5 prompt
train diffusion: z_t, t, y, prompt -> noise
```

这比“硬套作者 paired 训练”更符合去 IBE 后的模型定义，也更容易在答辩里解释。

### 12.4 MIMIC super5 标签策略

标签从报告文本映射：

```python
from scripts.triple_labels.label_schemes import mimic_report_to_super5
```

基础映射后必须增加一个 PTB-XL 语义保护：

```text
如果 CD/HYP/MI/STTC 任一异常类为 1，则强制 NORM=0
```

原因是 MIMIC 报告经常同时写 `sinus rhythm` 和异常诊断。如果不做 NORM guard，NORM 会接近“窦性心律”而不是 PTB-XL 的“无诊断异常”，这会破坏和 PTB-XL super5 的语义一致性。

本地粗略统计结果：

```text
MIMIC latent 总数: 744,372
报告可映射到至少一个 super5: 737,554, 约 99.1%

应用 NORM guard 后：
CD   178,311, 约 24.0%
HYP   86,608, 约 11.6%
MI   167,472, 约 22.5%
NORM 257,351, 约 34.6%
STTC 298,721, 约 40.1%
multi-label 195,334, 约 26.2%
```

这组分布可以支撑 MIMIC 预训练，但不能直接替代 PTB-XL 官方标签。因此 MIMIC 阶段只用于预训练，最终类别语义用 PTB-XL fine-tune 校准。

### 12.5 MIMIC prompt / nomic 使用方式

MIMIC 阶段不建议使用原始报告全文作为主 prompt。推荐和 PTB-XL 保持一致：

```text
report text -> mimic_report_to_super5 -> fixed super5 prompt -> nomic embedding
```

理由：

- 原始 MIMIC report 文本噪声更大，存在否定、可能性、历史诊断和格式差异。
- 本文下游任务是 super5 增强，不是自由文本报告到 ECG。
- 固定 prompt 表能让 MIMIC pretrain 和 PTB-XL fine-tune 使用同一语义接口。

实现上不要给每条 MIMIC 样本保存一份重复的 text embedding。super5 只有 5 类，multi-label 组合最多 31 种，应该缓存 label-combination prompt：

```text
/root/autodl-tmp/ecgtwin_class_super5/prompt_cache/super5_combo_nomic.pt
```

训练时根据 `y_super5` 查表得到：

```python
text_embed, text_embed_mask
```

如果短时间内不想处理 nomic 环境，MIMIC 预训练可以先跑 class-only：

```text
MIMIC class-only pretrain -> PTB-XL class_text fine-tune
```

这不会破坏论文故事。可以解释为 MIMIC 只学习 ECG latent 先验，PTB-XL 阶段再打开文本辅助条件。

### 12.6 数据拆分

MIMIC 预训练只需要 train/val，不作为最终测试集。

必须按 `subject_id` 做 patient-level split：

```text
train subjects: 95%
val subjects:   5%
```

不要随机按 ECG record split，否则同一患者不同时间 ECG 会同时出现在 train/val，验证 loss 会偏乐观。

建议新增一个轻量 metadata cache：

```text
/root/autodl-tmp/ecgtwin_class_super5/mimic_super5_meta.csv
```

字段：

```text
idx, subject_id, ecg_time, split, y_CD, y_HYP, y_MI, y_NORM, y_STTC, prompt_key, hr, age, sex
```

这个 meta 只存索引和标签，不复制 latent tensor。训练 dataset 仍从：

```text
/root/autodl-tmp/ECGTwin_Data/Mimic_vae.pt
```

读取 latent，避免额外占用磁盘。

### 12.7 推荐训练阶段

#### Stage M0：MIMIC smoke / pilot

目标是验证 dataset、标签、text mask、loss 曲线和显存。

```yaml
dataset: MIMIC single-sample latent
subset: 20k-100k records
batch_size: 512
max_steps: 2000-5000
lr: 1e-4
amp: true
class_drop_prob: 0.1
text_drop_prob: 0.1
condition_mode: class_only 或 class_text
```

验收：

```text
loss 正常下降
无 NaN
能采样 5 类各 3-5 条 ECG
12-lead 可视化无严重爆幅/平线
```

#### Stage M1：MIMIC full single-sample pretrain

这是推荐的正式 MIMIC 训练，不使用 paired ref-target。

优先复用作者 DiT 设置：

```yaml
hidden_size: 256
depth: 7
num_heads: 8
ddpm_train_steps: 1000
beta_start: 0.00085
beta_end: 0.0120
lr: 1e-4
batch_size: 512
epochs_author_equivalent: 30
```

实际执行不一定需要完整 30 epoch。MIMIC 有 744k 条，batch=512 时：

```text
1 epoch 约 1454 steps
30 epoch 约 43,620 steps
```

毕业设计时间紧时，推荐先用 step budget：

```text
max_steps = 15,000-25,000
```

如果 loss 和采样质量还在明显改善，再延长到：

```text
max_steps = 40,000-45,000
```

输出目录建议：

```text
/root/autodl-tmp/ecgtwin_class_super5/mimic_scheme_b_pretrain_s25k/
```

#### Stage P：PTB-XL fine-tune

MIMIC pretrain 之后必须回 PTB-XL fine-tune：

```text
init = MIMIC pretrain best.pt
train = PTB-XL fold 1-8
val = PTB-XL fold 9
```

推荐配置：

```yaml
batch_size: 512
lr: 5e-5 to 1e-4
max_steps: 1000-3000
epochs: 30-100
val_every: 200
condition_mode: 与最终采样一致
```

如果 MIMIC 阶段是 class-only，PTB-XL 阶段可以升级为 class_text：

```text
MIMIC class_only -> PTB-XL class_text fine-tune
```

新增的 text path 权重来自 ECGTwin 原始 checkpoint 或随机初始化均可，但需要用 PTB-XL fold9 验证是否比 class_only 稳定。

### 12.8 4090D 时间估计

前提：

```text
使用已有 latent cache
不重新 VAE encode
不使用 paired_Mimic 全量 pair
batch_size 约 512
AMP 开启
num_workers 8-12
```

估计时间：

| 阶段 | 估计时间 |
|---|---:|
| MIMIC meta 构建 + 标签统计 | 5-20 分钟 |
| Stage M0 pilot 2k-5k steps | 20-60 分钟 |
| Stage M1 15k-25k steps | 1.5-3.5 小时 |
| Stage M1 author-equivalent 30 epoch / 43.6k steps | 3-6 小时 |
| PTB-XL fine-tune 1k-3k steps | 0.5-1.5 小时 |
| 采样 1500-5000 条 + VAE decode | 20-80 分钟 |
| EfficientNetV2 real+synth 1-3 个比例 | 1-3 小时 |
| PN2021 七中心评测 | 30-90 分钟 |

现实日历建议：

```text
只做 MIMIC pilot + PTB-XL fine-tune: 0.5 天
做 MIMIC 25k steps + 下游一个比例: 1 天
做 MIMIC 43k steps + 多比例 + 表格: 1-2 天
```

如果直接用作者 full paired MIMIC：

```text
640 万 pair, batch=512, 30 epoch 约 37.5 万 steps
预计 16-28 小时甚至更长
```

这条不适合当前毕设时间表。

### 12.9 MIMIC 方案的实验矩阵

最小矩阵：

| ID | Generator 训练 | Synthetic | Classifier 训练 | 用途 |
|---|---|---|---|---|
| B0 | 无 | 无 | real PTB-XL | baseline |
| B1 | PTB-XL only | class_text synth | real+synth | 主线结果 |
| B2 | MIMIC pretrain -> PTB-XL fine-tune | class_text synth | real+synth | MIMIC 是否有帮助 |

如果时间允许再加：

| ID | Generator 训练 | Synthetic | 用途 |
|---|---|---|---|
| B3 | MIMIC pretrain -> PTB-XL fine-tune -> center token | center_synth | 目标中心适配 |
| B4 | MIMIC-only pretrain | MIMIC-only synth | 负面对照或附录，不作为主结果 |

主表只放 B0/B1/B2。B3 放扩展实验，B4 只有在结果有解释价值时再写。

评价口径：

```text
PTB-XL fold10: 看源域是否受伤
PN2021 avg: 看外部平均泛化
PN2021 per-center: 看是否有中心依赖收益
PN2021 per-class: 看 HYP/CD/MI/STTC 哪些类受益
```

MIMIC 预训练的成功标准不要写成“必须全面提升”。更稳的验收标准：

```text
B2 相比 B1:
1. PTB-XL fold10 AUROC/AUPRC 不明显下降；
2. PN2021 平均或若干中心/类别出现提升；
3. synthetic ECG 可视质量不差于 B1；
4. loss 曲线和采样图能证明大规模预训练有效收敛。
```

如果 B2 没有超过 B1，论文仍可写为：

> 大规模 MIMIC 预训练提升了生成模型对 ECG latent 分布的拟合能力，但由于 MIMIC 报告标签噪声和单中心域差异，下游增强收益需要依赖 PTB-XL fine-tune 与合成样本筛选；最终主方法采用 PTB-XL 校准后的生成器。

### 12.10 需要新增的代码入口

建议在 `methods/ecgtwin_class_super5/` 下新增：

```text
prepare_mimic_super5.py   # 构建 mimic_super5_meta.csv，统计标签分布，做 patient split
dataset_mimic.py          # MIMICLatentSuper5Dataset，可复用 collate_super5_latents
```

也可以直接扩展现有：

```text
methods/ecgtwin_class_super5/dataset.py
methods/ecgtwin_class_super5/train.py
```

推荐 CLI 形态：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -m methods.ecgtwin_class_super5.prepare_mimic_super5 \
  --latent_path /root/autodl-tmp/ECGTwin_Data/Mimic_vae.pt \
  --output_csv /root/autodl-tmp/ecgtwin_class_super5/mimic_super5_meta.csv \
  --val_subject_frac 0.05 \
  --norm_guard true
```

```bash
/root/miniforge3/envs/ECGTwin/bin/python -m methods.ecgtwin_class_super5.train \
  --dataset mimic \
  --latent_path /root/autodl-tmp/ECGTwin_Data/Mimic_vae.pt \
  --mimic_meta /root/autodl-tmp/ecgtwin_class_super5/mimic_super5_meta.csv \
  --output_dir /root/autodl-tmp/ecgtwin_class_super5/mimic_scheme_b_pretrain_s25k \
  --batch_size 512 \
  --num_workers 8 \
  --max_steps 25000 \
  --val_every 500 \
  --val_batches 8 \
  --amp true \
  --class_drop_prob 0.1 \
  --text_drop_prob 0.1
```

```bash
/root/miniforge3/envs/ECGTwin/bin/python -m methods.ecgtwin_class_super5.train \
  --dataset ptbxl \
  --init_checkpoint /root/autodl-tmp/ecgtwin_class_super5/mimic_scheme_b_pretrain_s25k/checkpoints/best.pt \
  --output_dir /root/autodl-tmp/ecgtwin_class_super5/ptbxl_after_mimic_pretrain \
  --batch_size 512 \
  --num_workers 8 \
  --max_steps 3000 \
  --val_every 200 \
  --val_batches 8 \
  --amp true
```

注意：这些 CLI 是目标形态，当前代码如果还没有 `--dataset mimic` 和 `--init_checkpoint`，需要先补实现再执行。

### 12.11 最推荐的执行版本

如果现在就要推进 MIMIC，我建议只做这个版本：

```text
1. 从 Mimic_vae.pt 构建 patient-level split 和 super5 meta。
2. 使用 NORM guard 后的 y_super5。
3. prompt 使用固定 super5 prompt 表，不用原始报告全文。
4. 训练 no-IBE class_text diffusion 25k steps。
5. 用 MIMIC best.pt 初始化 PTB-XL fine-tune 3k steps。
6. 只从 PTB-XL fine-tuned checkpoint 采样。
7. 训练 EfficientNetV2 real+synth=1:0.25 和 1:0.5 两个比例。
8. 报告 PTB-XL fold10 + PN2021 七中心 AUROC/AUPRC。
```

这条路线的优势：

- 充分利用 MIMIC 的大数据量。
- 不被作者 paired/IBE 逻辑拖慢。
- 不把 noisy MIMIC label 当成最终语义标准。
- 和现有 PTB-XL/PN2021 毕设故事完全兼容。
- 即使 MIMIC 增益有限，也能作为“预训练是否改善生成增强”的完整消融实验。

## 13. 生成 ECG 质量验证协议

### 13.1 ECGTwin 作者的验证方式

ECGTwin 作者不是只靠可视化判断生成 ECG 是否可信，而是采用了多层验证。

论文和 repo 中可以归纳为：

| 层级 | 作者指标 / 方法 | 目的 |
|---|---|---|
| Signal level | FID、improved Precision、Recall、F1 | 判断生成 ECG 的整体特征分布是否接近真实 ECG |
| Feature / physiology level | HR-MAE | 判断生成 ECG 的心率是否符合目标 cardiac condition |
| Diagnostic / semantic level | ECG-text CLIP Score | 判断生成 ECG 是否和目标 clinical text report 语义一致 |
| Personal consistency | base vector t-SNE、similarity score、silhouette coefficient | 判断生成 ECG 是否保留同一患者的个体特征 |
| Downstream utility | ECG auto-diagnosis Acc / Macro-F1 | 判断合成样本是否真的提升诊断模型 |
| Qualitative / interpretability | 12 导联图、attention map、prompt-to-prompt editing case | 判断指定 prompt 是否触发对应 ECG 形态，并提供解释性 |

本地 ECGTwin repo 里能直接看到这些实现痕迹：

```text
model/ECGTwin/test_scripts/evaluation.py   # CLIP Score, FID, Precision/Recall/F1
model/ECGTwin/test_scripts/hr_test.py      # XQRS 检测 HR, 计算 HR-MAE
model/ECGTwin/trainer/CLIPTrainer.py       # ECG-text CLIP 训练/评估
model/ECGTwin/module/DiT_ECGTwin.py        # need_weights, swap_report, add_report, prompt-to-prompt editing
model/ECGTwin/utils/inference_utils.py     # 保存 reference/generated ECG 图和 features.json
```

作者的重点是 **patient digital twin**：

```text
reference ECG + reference condition + target condition -> personalized ECG digital twin
```

因此他们必须验证：

- 生成信号像真实 ECG。
- 生成结果符合目标文本、心率、年龄、性别等 target cardiac condition。
- 生成结果保留 reference patient 的个体特征。
- 生成样本能帮助个体化诊断模型。

我们的任务不同。我们删除 IBE 后不再 claim patient-level digital twin，而是 claim：

```text
super5 class-conditional ECG augmentation
```

所以作者的 personal consistency 指标不能原样作为主指标，但 signal / physiology / semantic / downstream 四类验证仍然应该保留，并改造成适合 super5 增强任务的版本。

### 13.2 本文采用的质量验证证据链

推荐把生成质量验证写成五层：

```text
Level 0: 基础数值和生理 sanity
Level 1: super5 class / prompt 一致性
Level 2: real-vs-synth 特征分布相似性
Level 3: 数字 ECG 医学阈值抽查
Level 4: 下游增强有效性
```

这比“只看 PN2021 是否提升”更稳，因为下游指标可能受 synthetic ratio、类别不平衡、训练随机性影响。即使最终平均 AUROC 提升有限，也能用多证据链证明生成器和增强方法是完整、可解释、可复现的。

### 13.3 Level 0：基础数值和生理 sanity

每次采样后先跑基础 sanity。目标是排除明显坏样本，而不是做医学诊断。

使用现有工具：

```python
from util.ecg_viz import sanity_check
```

检查项：

- NaN / Inf。
- flatline leads。
- saturated leads。
- 每导联 DC offset。
- 每导联 peak-to-peak amplitude。
- Einthoven 定律残差：`II ≈ I + III`。
- aVR 关系残差：`aVR ≈ -(I + II) / 2`。
- lead II 粗略 HR 估计。
- HR 是否在 `[30, 200] bpm` 生理范围。

建议输出：

```text
synth_quality_summary.json
per_sample_quality.csv
per_class_quality.csv
figures/gallery_safe/*.png
```

建议最低通过线：

```text
NaN/Inf: 0
flatline: < 1% samples
saturation: < 1% samples
HR detected: >= 80% samples
HR in [30,200]: >= 90% detected samples
median |DC offset|: < 0.5 mV
median Einthoven residual: < 0.25
median aVR residual: < 0.35
```

这些阈值用于工程筛查，不要写成临床诊断标准。如果某个阈值不满足，先检查：

- lead order 是否正确。
- ECGTwin lead order 是否已经转 PTB-XL lead order。
- VAE decode 输出是否是 mV 尺度。
- 重采样是否从 1024 到 1000 后仍保持 `(N,1000,12)`。

### 13.4 Level 1：super5 class / prompt 一致性

作者用 ECG-text CLIP Score 检查“生成 ECG 是否符合 clinical report”。我们的任务是 super5 分类增强，因此更直接的语义一致性验证是：

```text
固定 EfficientNet1DV2 super5 baseline
synthetic ECG -> victim classifier
检查目标类别概率和 top-k 命中
```

建议使用已有 baseline：

```text
/root/autodl-tmp/triple_labels/super5/best_model.pt
```

每条 synthetic 记录保存：

```text
target_super5
prompt
victim_prob_CD
victim_prob_HYP
victim_prob_MI
victim_prob_NORM
victim_prob_STTC
target_prob
top1
top2
passes_victim_gate
```

建议初始 gate：

```text
target_prob >= 该类在真实 fold9 阳性样本上的 P25
或 target 类进入 top2
```

NORM 单独处理：

```text
NORM target:
    NORM probability high
    CD/HYP/MI/STTC abnormal probabilities不能整体偏高
```

注意：victim classifier 只能作为一致性筛查，不能作为唯一质量证明。因为它可能偏向自己的训练分布，也可能对伪影过度自信。论文里建议写成：

> 本文使用冻结的下游分类器作为类别一致性过滤器，而非作为合成 ECG 临床正确性的唯一证据。

### 13.5 Level 2：real-vs-synth 特征分布相似性

借鉴作者的 signal-level evaluation，建议增加 feature-space 分布指标：

```text
FID / rFID
Precision
Recall
F1
t-SNE / UMAP 可视化
```

特征提取器可以二选一：

```text
方案 A：EfficientNet1DV2 penultimate feature
方案 B：ECGTwin repo 的 ECG-text CLIP signal feature
```

考虑毕业时间，优先方案 A。原因：

- EfficientNet1DV2 是我们下游任务实际使用的 backbone。
- 不需要重新训练 ECGTwin CLIP。
- 更贴近“synthetic 是否落在分类器可用的数据流形上”。

对比集合：

```text
real_train = PTB-XL fold 1-8
real_val   = PTB-XL fold 9
synth      = generated pool
```

建议报告：

```text
FID(real_val, synth)
rFID = FID(real_val, synth) / FID(real_train_splitA, real_train_splitB)
precision/recall/F1(real_val manifold, synth manifold)
```

解释口径：

- FID 低：synthetic 总体接近真实 ECG 特征分布。
- precision 高：synthetic 多数落在真实 ECG 流形附近，伪样本较少。
- recall 高：synthetic 覆盖真实样本多样性较好。
- precision 高但 recall 低：样本较保守，适合小比例增强。
- recall 高但 precision 低：多样但可能噪声大，需要 gate。

### 13.6 Level 3：数字 ECG 医学阈值抽查

本项目已有数字 ECG 特征工具：

```text
util/ecg_digital_features.py
docs/ecg_digital_thresholds.md
scripts/ecgtwin_gen/digital_gt_validate.py
docs/ecgtwin_super5_digital_gt_validation.md
```

可抽查的数字特征：

- HR。
- RR irregularity。
- P wave。
- PR interval。
- QRS duration。
- ST level。
- T polarity / inversion。
- HYP voltage criteria。
- CD wide-QRS / LBBB / RBBB / AVB criteria。

这层不建议作为所有 synthetic 的硬过滤器。原因：

- 100Hz/102.4Hz 信号对 PR、QRS onset、Q wave、J point 估计比较粗。
- ECGTwin VAE decode 可能存在 amplitude attenuation，HYP/CD 这类依赖绝对电压或细粒度形态的类容易被数字阈值判负。
- super5 是粗粒度 multi-label，不等于某个单一 ECG 数字诊断规则。

建议用法：

```text
每个最终 generator checkpoint:
    每类抽 20-50 条
    跑 digital feature extractor
    输出 per-class pass / warning report
```

报告策略：

- NORM：看 HR、P wave、QRS、ST/T 是否基本合理。
- MI：看 ST elevation 或 pathological Q 是否有支持证据。
- STTC：看 ST depression / T inversion 是否有支持证据。
- HYP：谨慎报告 absolute voltage，可能受 VAE 幅值压缩影响。
- CD：谨慎报告 QRS width / rsR' / PR，失败时写成限制。

已有数字验证经验显示：

```text
NORM / MI / STTC 相对更容易通过数字检查
HYP / CD 可能因为电压尺度或精细传导形态失败
```

因此最终 synthetic augmentation 可以采用 per-class trust gate：

```text
如果 HYP/CD 质量门不过，先只使用 NORM/MI/STTC synthetic 做主增强实验；
HYP/CD 作为失败分析或 future work。
```

### 13.7 Level 4：下游增强有效性

这是最终证据，但不是唯一证据。

主对比：

```text
Baseline: real PTB-XL only
Ours:     real PTB-XL + gated synthetic ECG
```

必须报告：

```text
PTB-XL fold10 macro AUROC / macro AUPRC
PN2021 7-center macro AUROC / macro AUPRC
per-class AUROC / AUPRC
per-center AUROC / AUPRC
per-center delta vs baseline
```

选择 synthetic ratio 的规则：

```text
只能用 PTB-XL fold9 选择比例和 gate；
不能用 fold10 或 PN2021 反调参数。
```

推荐比率：

```text
real:synth = 1:0.1
real:synth = 1:0.25
real:synth = 1:0.5
```

第一轮 `1:0.25` 已经跑过但未超过 baseline，因此下一轮不要盲目加大 synthetic。更合理的下一步是：

```text
先做质量 gate，再跑 1:0.1 和 1:0.25
优先比较 all-class synth vs trusted-class synth
```

### 13.8 推荐新增 eval_synth.py

建议新增：

```text
methods/ecgtwin_class_super5/eval_synth.py
```

输入：

```text
--synth_npz /path/to/synth_waveforms.npz
--baseline_model /root/autodl-tmp/triple_labels/super5/best_model.pt
--real_feature_ref /root/autodl-tmp/ptbxl/...
--output_dir /root/autodl-tmp/ecgtwin_class_super5/<run>/quality_eval
```

输出：

```text
quality_eval/
├── synth_quality_summary.json
├── per_sample_quality.csv
├── per_class_quality.csv
├── victim_scores.csv
├── feature_distribution.json
├── digital_criteria_report.md
└── figures/
    ├── class_gallery_*.png
    ├── victim_prob_hist_*.png
    ├── feature_tsne.png
    └── hr_distribution.png
```

最小实现顺序：

1. 读取 `synth_waveforms.npz`。
2. 跑 `util.ecg_viz.sanity_check`。
3. 跑 EfficientNet1DV2 victim classifier。
4. 按类汇总 target probability / top-k / warning rate。
5. 输出 `trusted_indices.npy` 或 `trusted_synth_waveforms.npz`。
6. 后续再加 feature FID / precision / recall。

这样下游训练可以从：

```text
raw synth pool
```

升级为：

```text
quality-gated synth pool
```

论文中可以写成：

> 为降低低质量合成 ECG 对分类器训练的负面影响，本文在合成样本进入下游训练前引入质量门控，包括基础信号合理性、类别一致性和特征分布筛查。

### 13.9 论文中推荐的验证表格

建议最终至少准备 4 张表：

**表 1：生成质量 sanity summary**

| generator | n | NaN/Inf | HR detected | HR in range | median p2p | median Einthoven residual | warning rate |
|---|---:|---:|---:|---:|---:|---:|---:|

**表 2：super5 class consistency**

| class | n | target prob mean | target prob P25 | top1 hit | top2 hit | trusted rate |
|---|---:|---:|---:|---:|---:|---:|

**表 3：feature distribution**

| generator | FID ↓ | rFID ↓ | precision ↑ | recall ↑ | F1 ↑ |
|---|---:|---:|---:|---:|---:|

**表 4：downstream augmentation**

| train data | PTB-XL AUROC | PTB-XL AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|

如果时间只够最小闭环，表 1、表 2、表 4 必做；表 3 可以放入扩展实验。

### 13.10 答辩口径

不要说：

```text
生成 ECG 已经被证明完全符合临床诊断标准。
```

建议说：

> 生成 ECG 的质量通过多层证据验证：首先排除数值异常和明显生理不合理样本；其次使用冻结分类器检查 super5 类别一致性；再通过特征分布指标比较 synthetic 和真实 PTB-XL 的流形接近程度；最后把质量筛选后的 synthetic 数据用于 EfficientNetV2 训练，并在 PTB-XL fold10 和 PN2021 七中心上评估外部泛化。对于 HYP/CD 等依赖精细电压或传导形态的类别，本文采用 per-class trust gate，避免低可信合成样本污染训练。

这条表述比单纯 claim “prompt 生成了对应疾病 ECG”更稳，也能解释为什么医学生成增强不一定在所有中心、所有类别上平均提升。
