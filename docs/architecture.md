# ECGTwin Inference 数据流详解

## 一、输入数据格式

生成 ECG 需要准备一个 `.pt` 文件（参考 `data/prepared_input/normal_1.pt`），包含：

```python
{
    "data":  tensor (4, 128),        # 参考ECG经VAE编码后的 latent
    "label": {
        "text":       "Sinus rhythm|Normal ECG",   # 参考ECG的诊断描述（用 | 分隔多个诊断）
        "text_embed": tensor (2, 768),              # 诊断文本经Nomic编码后的向量（2段诊断→2个向量）
        "hr":         91.23,                        # 心率 (beats/min)
        "age":        55,                           # 年龄
        "sex":        "F",                          # 性别 "F" 或 "M"
        "subject_id": 10000117,                     # 患者ID（生成时不使用）
        "ecg_time":   "2181-03-04 17:14:00"         # 记录时间（生成时不使用）
    }
}
```

### 原始 ECG 怎么变成这个格式

```
原始 12 导联 ECG 信号 (1024, 12) @ 500Hz
              │
              ▼
     ╔═════════════════╗
     ║   VAE Encoder   ║    提前离线跑好，存成 .pt
     ╚════════╤════════╝
              │
              ▼
      ref_latent (4, 128)     ← 这就是 "data" 字段

原始诊断报告文本 "Sinus rhythm|Normal ECG"
              │
              ▼
     ╔══════════════════╗
     ║ Nomic Text Model ║   提前离线跑好，存进 label
     ╚════════╤═════════╝
              │
              ▼
      text_embed (2, 768)     ← 这就是 label["text_embed"] 字段

患者信息从数据库直接读取     ← hr, age, sex 存进 label
```

> **注意**：你不需要自己准备原始 ECG 波形。输入文件里存的是提前用 VAE 编码好的 latent + 元数据。


---

## 二、完整 Inference 数据流

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                    .pt 输入文件（一个患者的参考数据）                       │
 │                                                                          │
 │   ref_latent (4,128)    text_embed (2,768)    hr=91  age=55  sex="F"    │
 └───────┬─────────────────────┬──────────────────────┬────────────────────┘
         │                     │                      │
         │                     │                      ▼
         │                     │            ┌──────────────────┐
         │                     │            │ process_pat_info │
         │                     │            │ hr → 91/200=0.46 │
         │                     │            │ age → 55/100=0.55│
         │                     │            │ sex → "F" → 0    │
         │                     │            └────────┬─────────┘
         │                     │                     │
         │                     │           pat_info_ref (B, 3)
         │                     │             [0.46, 0.55, 0]
         │                     │                     │
         ▼                     ▼                     │
 ┌──────────────┐    ┌─────────────────┐             │
 │  transpose   │    │  text_embed_ref │             │
 │ (4,128)      │    │    (B, 2, 768)  │             │
 │  → (128,4)   │    └────────┬────────┘             │
 └──────┬───────┘             │                      │
        │                     │                      │
        │          ┌──────────┴──────────┐           │
        │          │ concat pat_info 到   │           │
        │          │ text_embed 最后一维   │           │
        │          │ (B,2,768)+(B,2,3)   │           │
        │          │  → (B, 2, 771)      │           │
        │          │  → Linear → (B,2,256)│          │
        │          └──────────┬──────────┘           │
        │                     │                      │
        │                condition c (B, 2, 256)     │
        │                     │                      │
        ▼                     ▼                      │
 ╔══════════════════════════════════════════╗         │
 ║           IBExtractor（3层）              ║         │
 ║                                          ║         │
 ║  ref_latent(128,4) → Linear → (128,256) ║         │
 ║                  → RoPE位置编码           ║         │
 ║                          │               ║         │
 ║           ┌──────────────┼───────┐       ║         │
 ║           │    每层 IBEncoderLayer│       ║         │
 ║           │                      │       ║         │
 ║           │  x ─→ Self-Attn ─→ x'       ║         │
 ║           │  x'+ Cross-Attn(x',c)→ x''  ║         │
 ║           │  x''─→ FFN ─→ x_out         ║         │
 ║           │                      │       ║         │
 ║           └──────── ×3层 ────────┘       ║         │
 ║                      │                   ║         │
 ║              (B, 128, 256)               ║         │
 ║                      │                   ║         │
 ║               mean pooling               ║         │
 ║                      │                   ║         │
 ║              (B, 256) ← base_vector      ║         │
 ╚══════════════════════╤═══════════════════╝         │
                        │                             │
                        │   "这个人的心脏长什么样"       │
                        │                             │
 ═══════════════════════╪═════════════════════════════╪══════════
   以上：从参考ECG提取个人特征                         │
   以下：根据目标条件生成新ECG                         │
 ═══════════════════════╪═════════════════════════════╪══════════
                        │                             │
                        │         ┌───────────────────┘
                        │         │
                        │         │     目标配置 (来自 yaml inference_setting)
                        │         │     text: "sinus rhythm|normal ecg."
                        │         │     hr: 70, age: 70, sex: "F"
                        │         │              │
                        │         │              ├───────────────────┐
                        │         │              │                   │
                        │         │              ▼                   ▼
                        │         │   ┌───────────────────┐  ┌──────────────┐
                        │         │   │ Tokenizer + Nomic │  │process_pat_  │
                        │         │   │                   │  │    info      │
                        │         │   │ "sinus rhythm"    │  │ hr→70/200   │
                        │         │   │    → [101,..,102] │  │ age→70/100  │
                        │         │   │    → Nomic        │  │ sex→0       │
                        │         │   │    → mean pool    │  └──────┬───────┘
                        │         │   │    → (1, 768)     │         │
                        │         │   │ "normal ecg."     │  pat_info_tar
                        │         │   │    → (1, 768)     │    (B, 3)
                        │         │   │                   │         │
                        │         │   │ stack → (B,2,768) │         │
                        │         │   └─────────┬─────────┘         │
                        │         │             │                   │
                        │         │    text_embed_tar (B,2,768)     │
                        │         │      "想生成什么病"               │
                        │         │             │                   │
                        ▼         │             ▼                   ▼
                                  │
  x_T ~ N(0,I)                   │
   (B, 4, 128)                   │
    纯随机噪声                    │
        │                         │
        ▼                         │
 ╔══════════════════════════════════════════════════════════════════╗
 ║                DiT_ECGTwin 去噪循环 (1000步)                      ║
 ║                                                                  ║
 ║  for t = 999, 998, ..., 0:                                      ║
 ║                                                                  ║
 ║    predicted_noise = noise_predictor(                            ║
 ║        x_t,              ← 当前带噪 latent (B, 4, 128)          ║
 ║        t,                ← 当前时间步                             ║
 ║        text_embed_tar,   ← 目标诊断文本向量 (B, 2, 768)          ║
 ║        text_embed_mask,  ← None（无mask）                        ║
 ║        pat_info_tar,     ← 目标患者体征 (B, 3)                   ║
 ║        base_vector       ← 参考ECG个人特征 (B, 256)              ║
 ║    )                                                             ║
 ║                                                                  ║
 ║    x_{t-1} = DDPMScheduler.step(predicted_noise, t, x_t)       ║
 ║                                                                  ║
 ╚══════════════════════════════╤═══════════════════════════════════╝
                                │
                                ▼
                         x_0 (B, 4, 128)
                          干净的 latent
                                │
                                ▼
                       ╔════════════════╗
                       ║  VAE Decoder   ║
                       ║ (4,128)→(1024,12)║
                       ╚═══════╤════════╝
                               │
                               ▼
                    生成 ECG (B, 1024, 12)
                      12导联  500Hz  ~2秒
                               │
                        ┌──────┴──────┐
                        │             │
                        ▼             ▼
                  保存 .pt       ecg_plot 画图
                  (latent)       保存 .png
```


---

## 三、ref（参考）vs tar（目标）的分工

```
ref（参考ECG）提供：                    tar（目标配置）控制：
├── ref_latent → IBExtractor           ├── text → Nomic → DiT cross-attn
├── text_embed → IBExtractor           ├── hr → pat_info_tar → DiT
├── pat_info   → IBExtractor           ├── age → pat_info_tar → DiT
│                                      └── sex → pat_info_tar → DiT
└──→ base_vector (256维)
     注入 DiT: c = timestep_embed + base_vector

含义：                                  含义：
"参考这个人的心脏形态特征"                "但是生成这种疾病/这个年龄/这个心率的ECG"
```

> **核心设计思想**：base_vector 锁定"谁的心脏"，text_embed_tar + pat_info_tar 控制"生成什么样的心电图"。
> 这样可以做到：拿一个健康人的参考ECG，生成出"如果这个人得了心梗，心电图会长什么样"。

---

## 四、关键尺寸速查表

| 数据 | Shape | 含义 |
|------|-------|------|
| 原始 ECG | `(1024, 12)` | 12导联 × 1024采样点 @ 500Hz ≈ 2.048秒 |
| ref_latent | `(4, 128)` | VAE 压缩后：4通道 × 128时间步 |
| pat_info | `(B, 3)` | [hr_norm, age_norm, sex_binary] |
| text_embed | `(B, L, 768)` | L = 诊断描述段数（用 \| 分隔），每段 768 维 |
| base_vector | `(B, 256)` | IBExtractor 输出的个人特征向量 |
| x_T (噪声) | `(B, 4, 128)` | 与 ref_latent 同 shape 的高斯噪声 |
| x_0 (去噪后) | `(B, 4, 128)` | 去噪完成的 latent |
| 生成 ECG | `(B, 1024, 12)` | 最终输出的 ECG 信号 |

---

## 五、导联顺序

```
ECGTwin 导联顺序:
  索引:  0    1     2     3     4     5    6    7    8    9   10   11
  导联:  I    II   III   aVR   aVF   aVL   V1   V2   V3   V4   V5   V6
```
