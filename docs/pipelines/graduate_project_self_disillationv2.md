# ECGTwin 合成样本自蒸馏训练方案

## 0. 一句话结论

在“**ECGTwin 生成大量合成样本 + 仅 2000 条 PTB-XL 真实样本**”的低资源场景下，推荐优先采用：

```text
自蒸馏 + Teacher 软标签 + 合成样本质量筛选
```

而不是一开始就使用在线 FGSM/PGD 对抗训练。

核心理由是：

```text
ECGTwin 生成样本数量多，但标签和波形质量可能不完全可靠；
自蒸馏可以降低合成样本标签噪声，让 EfficientNetV2-1D 更稳定地利用合成数据；
在线对抗训练主要提升鲁棒性，不一定提升 clean AUROC / AUPRC，甚至可能降低干净测试性能。
```

---

## 1. 方案目标

本文方案用于训练：

```text
EfficientNetV2-1D Superclass 5 分类模型
```

分类类别：

```text
NORM / MI / STTC / HYP / CD
```

训练数据设定：

```text
真实数据：2000 条 PTB-XL 训练样本
合成数据：ECGTwin 按 superclass 条件生成的大量 ECG
测试数据：真实 PTB-XL fold10 或固定真实测试集
```

目标指标：

```text
Macro-AUROC
Macro-AUPRC
每类 AUROC
每类 AUPRC
每类 Recall / F1
```

重点关注：

```text
Macro-AUPRC
MI AUPRC
STTC AUPRC
HYP AUPRC
CD AUPRC
```

原因：PTB-XL superclass 存在类别不均衡，AUPRC 比 Accuracy 更能反映少数类提升。

---

## 2. 总体流程

完整流程如下：

```text
Step 1: 从 PTB-XL train fold 中分层抽取 2000 条真实 ECG
        ↓
Step 2: 用这 2000 条真实 ECG 训练 Teacher EfficientNetV2-1D
        ↓
Step 3: ECGTwin 按 NORM / MI / STTC / HYP / CD 生成合成 ECG
        ↓
Step 4: Teacher 对每条合成 ECG 做 multi-crop 预测
        ↓
Step 5: 根据 Teacher 置信度筛选高质量样本、困难样本、可疑样本
        ↓
Step 6: 为保留的合成样本构造 soft label
        ↓
Step 7: 使用真实样本 hard label + 合成样本 soft label 训练 Student
        ↓
Step 8: 在真实 PTB-XL 测试集上评估 AUROC / AUPRC
```

一句话概括：

```text
ECGTwin 负责扩大数据规模；
Teacher 负责判断合成样本质量；
Soft label 负责降低标签噪声；
Student 负责最终提升分类性能。
```

---

## 3. 为什么不能直接把 ECGTwin 生成样本硬标签加入训练？

假设 ECGTwin 生成一条 MI 样本：

```text
condition = MI
hard label = MI
```

如果直接用硬标签训练，会有风险：

| 合成样本情况 | 直接硬标签训练的风险 |
|---|---|
| 确实像 MI | 有帮助 |
| 同时像 MI 和 STTC | 硬标成 MI 太绝对 |
| 波形有生成伪影 | 模型可能学习 ECGTwin 伪影 |
| 实际更像 NORM 或 STTC | 错标签污染训练 |
| 合成样本数量远超真实样本 | 真实数据分布被合成分布淹没 |

因此不推荐：

```text
2000 real + 20000 synthetic hard label 直接训练
```

更推荐：

```text
ECGTwin 生成候选样本
Teacher 判断可信度
Student 学习筛选后的 soft label 样本
```

---

## 4. Teacher 如何训练？

Teacher 是一个先训练好的 EfficientNetV2-1D 分类器，用于给合成样本打分。

### 4.1 主实验推荐：Real-2000 Teacher

```text
Teacher = 仅使用同一批 2000 条真实 PTB-XL 样本训练得到的模型
```

优点：

```text
不使用额外真实标签；
低资源实验设定公平；
答辩时不容易被质疑“偷看了全量 PTB-XL”。
```

缺点：

```text
Teacher 能力有限，筛选合成样本时可能不够稳定。
```

### 4.2 更稳推荐：3-seed Ensemble Teacher

训练三个不同随机种子的 Teacher：

```text
Teacher_1: seed = 1
Teacher_2: seed = 2
Teacher_3: seed = 3
```

预测时取平均：

```text
p_teacher = mean(p_1, p_2, p_3)
```

优点：

```text
预测更稳定；
降低单个 Teacher 偏差；
更适合作为合成样本质量筛选器。
```

这是本文最推荐的主实验 Teacher 设置。

### 4.3 不建议作为主实验：Full PTB-XL Teacher

也可以用全量 PTB-XL 训练强 Teacher：

```text
Teacher = 全量 PTB-XL train fold 训练得到的模型
```

但这个设置不适合作为主实验，因为它引入了额外真实标签信息。

可以作为：

```text
Oracle Teacher
Upper-bound Teacher
```

用于补充消融，不作为主要结论来源。

---

## 5. ECGTwin 合成样本生成

按 superclass 条件生成：

```text
NORM
MI
STTC
HYP
CD
```

建议先做合成比例消融：

| 真实样本数 | 合成样本数 | 合成比例 |
|---:|---:|---:|
| 2000 | 500 | 25% |
| 2000 | 1000 | 50% |
| 2000 | 2000 | 100% |
| 2000 | 4000 | 200% |

初始推荐：

```text
2000 real + 2000 synthetic
```

不建议一开始生成极大量样本，例如：

```text
2000 real + 20000 synthetic
```

原因是合成样本可能压过真实分布。

### 5.1 合成样本进入分类器前的格式要求

ECGTwin 解码后通常是：

```text
(B, 1024, 12)
ECGTwin / MIMIC lead order
raw mV scale
```

进入 EfficientNetV2-1D 分类器前，应转成：

```text
(B, 1000, 12)
PTB-XL classifier lead order
classifier scale
```

至少检查：

```text
shape 是否为 (N, 1000, 12)
lead order 是否为 I, II, III, aVR, aVL, aVF, V1-V6
是否存在 NaN / Inf
是否存在全零导联
每导联幅值是否异常
```

---

## 6. Teacher 如何给合成样本打分？

每条合成 ECG 是 10 秒：

```text
x_synth.shape = (1000, 12)
```

EfficientNetV2-1D 实际输入是 2.5 秒 crop：

```text
model input = (12, 250)
```

为了避免只看中间 2.5 秒导致判断不稳定，建议使用 multi-crop Teacher prediction。

### 6.1 Multi-crop 预测

从 10 秒 ECG 中取四个 2.5 秒片段：

```text
crop1: 0.0s - 2.5s
crop2: 2.5s - 5.0s
crop3: 5.0s - 7.5s
crop4: 7.5s - 10.0s
```

Teacher 对四个片段分别预测：

```text
p_1 = Teacher(crop1)
p_2 = Teacher(crop2)
p_3 = Teacher(crop3)
p_4 = Teacher(crop4)
```

最终概率：

```text
p_teacher = mean(p_1, p_2, p_3, p_4)
```

如果使用 3 个 Teacher ensemble：

```text
p_teacher = mean(Teacher_1 crops, Teacher_2 crops, Teacher_3 crops)
```

### 6.2 输出形式

PTB-XL superclass 更自然是 multi-label 任务，因此推荐用 sigmoid：

```text
p_teacher = sigmoid(logits_teacher)
```

输出五个概率：

```text
p_teacher = [p_NORM, p_MI, p_STTC, p_HYP, p_CD]
```

---

## 7. 合成样本筛选规则

假设一条合成 ECG 的生成条件是：

```text
target_class = MI
```

Teacher 输出：

```text
NORM: 0.05
MI:   0.72
STTC: 0.25
HYP:  0.02
CD:   0.06
```

由于 `p_MI = 0.72`，Teacher 也认为它像 MI，因此可以保留。

### 7.1 高质量样本

条件：

```text
p_teacher[target_class] >= 0.6
且 target_class in top-2
```

处理：

```text
保留
sample_weight = 1.0
```

### 7.2 困难样本

条件：

```text
0.35 <= p_teacher[target_class] < 0.6
且 target_class in top-2
```

例子：

```text
MI:   0.45
STTC: 0.40
```

这类样本可能靠近决策边界，有训练价值。

处理：

```text
保留
sample_weight = 0.3 ~ 0.5
```

初始推荐：

```text
sample_weight = 0.5
```

### 7.3 可疑样本

条件：

```text
p_teacher[target_class] < 0.35
或 target_class 不在 top-2
```

处理：

```text
丢弃
```

### 7.4 NORM 类特殊规则

NORM 正常类要更严格。

如果 ECGTwin 生成条件是 NORM，推荐保留条件：

```text
p_NORM >= 0.7
max(p_MI, p_STTC, p_HYP, p_CD) <= 0.3
```

否则丢弃。

原因：

```text
正常样本不应同时表现出明显异常类高置信度。
```

---

## 8. 更严谨的阈值：按类自适应阈值

固定阈值 `0.6 / 0.35` 简单，但每类难度不同。

更严谨做法是使用真实验证集 fold9 来估计每类 Teacher 置信度分布。

例如对真实 MI 阳性样本，Teacher 输出 `p_MI` 的分布为：

```text
0.92, 0.85, 0.78, 0.71, 0.62, 0.55, ...
```

可以定义：

```text
tau_high_MI = MI 阳性样本 p_MI 的 50% 分位数
tau_low_MI  = MI 阳性样本 p_MI 的 20% 分位数
```

筛选规则：

```text
p_MI >= tau_high_MI → 高质量样本
tau_low_MI <= p_MI < tau_high_MI → 困难样本
p_MI < tau_low_MI → 丢弃
```

每类独立计算：

```text
tau_high_NORM, tau_low_NORM
tau_high_MI,   tau_low_MI
tau_high_STTC, tau_low_STTC
tau_high_HYP,  tau_low_HYP
tau_high_CD,   tau_low_CD
```

优点：

```text
对 HYP/CD 等较难类别更公平；
避免用同一阈值误伤难分类类别；
实验解释更严谨。
```

---

## 9. Soft label 构造

对于保留的合成样本，不直接使用硬标签，而是构造 soft label。

### 9.1 条件硬标签

假设 ECGTwin 条件是 MI：

```text
y_condition = [0, 1, 0, 0, 0]
```

### 9.2 Teacher soft label

Teacher 输出：

```text
p_teacher = [0.05, 0.72, 0.25, 0.02, 0.06]
```

### 9.3 混合 soft label

定义：

```text
y_soft = γ * y_condition + (1 - γ) * p_teacher
```

推荐：

```text
γ = 0.3
```

即：

```text
30% 相信 ECGTwin 条件标签
70% 相信 Teacher 判断
```

例子：

```text
y_soft
= 0.3 * [0, 1, 0, 0, 0]
+ 0.7 * [0.05, 0.72, 0.25, 0.02, 0.06]
= [0.035, 0.804, 0.175, 0.014, 0.042]
```

这个标签表达：

```text
该合成样本主要是 MI，但与 STTC 也存在一定相似性。
```

这比硬标签 `[0,1,0,0,0]` 更适合 ECG 类别间存在重叠的情况。

---

## 10. Student 训练

Student 是最终模型：

```text
EfficientNetV2-1D Superclass 5 classifier
```

训练数据分两类：

```text
真实 PTB-XL: hard label
ECGTwin synthetic: soft label
```

### 10.1 真实样本损失

真实样本：

```text
x_real, y_real
```

损失：

```text
L_real = BCEWithLogitsLoss(Student(x_real), y_real)
```

### 10.2 合成样本损失

合成样本：

```text
x_synth, y_soft, sample_weight
```

损失：

```text
L_synth = weighted_BCEWithLogitsLoss(Student(x_synth), y_soft, sample_weight)
```

### 10.3 总损失

```text
L_total = L_real + β * L_synth
```

推荐初始值：

```text
β = 0.5
```

含义：

```text
合成样本参与训练，但权重低于真实样本。
```

如果合成样本质量较差，可调低：

```text
β = 0.3
```

如果合成样本质量较好，可尝试：

```text
β = 0.7
```

---

## 11. Batch 采样比例

推荐固定真实/合成比例。

### 方案 A：1:1

```text
batch_size = 64
real = 32
synth = 32
```

适合初始实验。

### 方案 B：2:1

```text
batch_size = 64
real = 42 或 48
synth = 22 或 16
```

如果合成样本导致性能下降，使用 2:1 更稳。

初始推荐：

```text
real:synth = 1:1
β = 0.5
```

若性能下降：

```text
real:synth = 2:1
β = 0.3
```

---

## 12. PyTorch 伪代码

### 12.1 Teacher multi-crop prediction

```python
import torch

@torch.no_grad()
def teacher_predict_multicrop(teacher, x_1000_12, device):
    """
    x_1000_12: Tensor, shape = (1000, 12)
    return: Tensor, shape = (5,)
    """
    teacher.eval()

    crop_len = 250
    starts = [0, 250, 500, 750]

    probs = []
    for s in starts:
        crop = x_1000_12[s:s + crop_len, :]      # (250, 12)
        crop = crop.T.unsqueeze(0).to(device)    # (1, 12, 250)

        logits = teacher(crop)
        prob = torch.sigmoid(logits).squeeze(0)  # (5,)
        probs.append(prob.cpu())

    return torch.stack(probs, dim=0).mean(dim=0)
```

### 12.2 Ensemble Teacher prediction

```python
@torch.no_grad()
def ensemble_teacher_predict(teachers, x_1000_12, device):
    all_probs = []
    for teacher in teachers:
        p = teacher_predict_multicrop(teacher, x_1000_12, device)
        all_probs.append(p)
    return torch.stack(all_probs, dim=0).mean(dim=0)
```

### 12.3 合成样本筛选

```python
def filter_synth_sample(p_teacher, target_idx):
    """
    p_teacher: Tensor, shape = (5,)
    target_idx: int
    return: status, sample_weight
    """
    p_target = float(p_teacher[target_idx])
    top2 = torch.topk(p_teacher, k=2).indices.tolist()

    if p_target >= 0.6 and target_idx in top2:
        return "high", 1.0

    if p_target >= 0.35 and target_idx in top2:
        return "hard", 0.5

    return "reject", 0.0
```

### 12.4 NORM 特殊筛选

```python
def filter_norm_sample(p_teacher, norm_idx=0):
    p_norm = float(p_teacher[norm_idx])
    abnormal_indices = [i for i in range(len(p_teacher)) if i != norm_idx]
    max_abnormal = float(p_teacher[abnormal_indices].max())

    if p_norm >= 0.7 and max_abnormal <= 0.3:
        return "high", 1.0

    return "reject", 0.0
```

### 12.5 Soft label 构造

```python
def build_soft_label(p_teacher, target_idx, num_classes=5, gamma=0.3):
    y_cond = torch.zeros(num_classes)
    y_cond[target_idx] = 1.0

    p_teacher = p_teacher.float()

    y_soft = gamma * y_cond + (1.0 - gamma) * p_teacher
    y_soft = torch.clamp(y_soft, 0.0, 1.0)

    return y_soft
```

### 12.6 Weighted BCE loss

```python
import torch.nn.functional as F


def weighted_bce_with_logits(logits, targets, sample_weights=None):
    """
    logits: Tensor, shape = (B, 5)
    targets: Tensor, shape = (B, 5), soft labels allowed
    sample_weights: Tensor, shape = (B,)
    """
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )  # (B, 5)

    loss = loss.mean(dim=1)  # (B,)

    if sample_weights is not None:
        loss = loss * sample_weights

    return loss.mean()
```

### 12.7 Student train step

```python
def train_step(student, real_batch, synth_batch, optimizer, beta=0.5):
    student.train()

    x_real, y_real = real_batch
    x_synth, y_synth_soft, synth_weights = synth_batch

    logits_real = student(x_real)
    logits_synth = student(x_synth)

    loss_real = F.binary_cross_entropy_with_logits(
        logits_real,
        y_real.float()
    )

    loss_synth = weighted_bce_with_logits(
        logits_synth,
        y_synth_soft.float(),
        synth_weights.float()
    )

    loss = loss_real + beta * loss_synth

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "loss_real": float(loss_real.item()),
        "loss_synth": float(loss_synth.item()),
    }
```

---

## 13. 可选增强：EMA Teacher

可以维护一个 EMA Teacher：

```text
Teacher = Student 参数的滑动平均
```

更新公式：

```text
θ_teacher = m * θ_teacher + (1 - m) * θ_student
```

推荐：

```text
m = 0.99 或 0.999
```

代码：

```python
@torch.no_grad()
def update_ema_teacher(student, teacher, momentum=0.99):
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
```

EMA Teacher 的优点：

```text
预测更平滑；
训练更稳定；
可以配合 consistency loss。
```

---

## 14. 可选增强：Consistency Loss

对同一条 ECG 构造两个版本：

```text
weak augmentation: 轻微增强或 center crop
strong augmentation: random crop + noise + baseline shift + lead mask
```

Teacher 看 weak：

```text
p_teacher = Teacher(x_weak)
```

Student 看 strong：

```text
p_student = Student(x_strong)
```

一致性损失：

```text
L_consistency = MSE(sigmoid(Student(x_strong)), sigmoid(Teacher(x_weak)))
```

总损失：

```text
L_total = L_real + β L_synth + α L_consistency
```

推荐：

```text
α = 0.1
```

这比在线 PGD/FGSM 对抗训练更稳，更适合毕设落地。

---

## 15. 推荐实验设计

至少做以下实验：

| 实验编号 | 方法 | Real-2000 | ECGTwin synth | 筛选 | Soft label | Consistency | 目的 |
|---|---|---:|---:|---:|---:|---:|---|
| E1 | Real-2000 Baseline | 是 | 否 | 否 | 否 | 否 | 低资源基线 |
| E2 | Hard Synth | 是 | 是 | 否 | 否 | 否 | 验证直接合成增强 |
| E3 | Filtered Hard Synth | 是 | 是 | 是 | 否 | 否 | 验证筛选是否有效 |
| E4 | Soft Synth Distill | 是 | 是 | 是 | 是 | 否 | 主方法 |
| E5 | Soft Synth + Consistency | 是 | 是 | 是 | 是 | 是 | 增强版 |
| E6 | Full Real Upper Bound | 全量 | 否 | 否 | 否 | 否 | 性能上限参考 |

理想结果：

```text
E4 > E3 > E2 > E1
```

如果出现：

```text
E2 < E1，但 E4 > E1
```

反而说明你的方法更有价值：

```text
直接加入合成样本会带来噪声；
Teacher-guided soft label 能稳定利用合成样本。
```

---

## 16. 推荐超参数

| 参数 | 推荐初始值 | 说明 |
|---|---:|---|
| real samples | 2000 | 低资源设定 |
| synth samples | 2000 | 初始 1:1 |
| Teacher | 3-seed ensemble | 更稳定 |
| crop | 4-crop | 用于 Teacher 打分 |
| high threshold | 0.6 | 高质量样本阈值 |
| hard threshold | 0.35 | 困难样本阈值 |
| hard sample weight | 0.5 | 困难样本降权 |
| gamma | 0.3 | 条件标签与 Teacher soft label 混合比例 |
| beta | 0.5 | 合成样本损失权重 |
| batch ratio | 1:1 | real:synth |
| EMA momentum | 0.99 | 可选 |
| consistency alpha | 0.1 | 可选 |

---

## 17. 常见问题与调整策略

### 17.1 合成样本加入后性能下降

可能原因：

```text
合成样本质量差；
合成样本太多；
beta 太大；
筛选阈值太低。
```

调整：

```text
合成样本数 2000 → 1000
beta 0.5 → 0.3
high threshold 0.6 → 0.7
只保留 high，不保留 hard
real:synth 1:1 → 2:1
```

### 17.2 AUROC 提升但 AUPRC 不提升

可能原因：

```text
模型排序能力提升，但少数类 precision/recall 仍弱；
类别不平衡仍然存在。
```

调整：

```text
增加 MI/HYP/CD 合成样本；
使用 class-balanced sampling；
使用 pos_weight；
按类别设置不同筛选阈值。
```

### 17.3 Teacher 太弱

调整：

```text
使用 3-seed ensemble Teacher；
训练 Teacher 更久；
使用 EMA Teacher；
只把 Full-PTBXL Teacher 作为 upper-bound 消融。
```

### 17.4 Teacher 太保守，筛掉太多样本

调整：

```text
降低 hard threshold；
保留 target in top-2 的困难样本；
降低困难样本权重，而不是全部丢弃。
```

---

## 18. 论文方法描述模板

可以直接写入论文：

> 为提升低资源条件下 ECG 异常检测模型对合成样本的利用效率，本文提出基于自蒸馏的 ECGTwin 合成样本软标签增强训练策略。首先，仅使用 2000 条真实 PTB-XL 训练样本训练 Teacher 分类器。随后，利用 ECGTwin 按 NORM、MI、STTC、HYP、CD 五个 superclass 条件生成合成 ECG 样本。考虑到生成样本可能存在标签不确定性和生成伪影，本文不直接采用生成条件作为硬标签，而是使用 Teacher 对合成样本进行多片段预测，并根据目标类别置信度筛选高质量样本与困难样本。对于保留的合成样本，本文将 ECGTwin 条件标签与 Teacher 输出概率进行加权融合，构造 soft label。最终，Student 分类器在真实样本硬标签监督和合成样本软标签蒸馏约束下联合训练，从而降低低质量合成样本对模型分类边界的干扰，提高少样本场景下 Superclass 五分类的 AUROC 和 AUPRC 表现。

---

## 19. 答辩解释模板

答辩时可以这样说：

> ECGTwin 可以生成大量指定 superclass 的 ECG 样本，但生成样本并不一定全部可靠。如果直接用生成条件作为硬标签训练分类器，可能会引入标签噪声和生成伪影。因此，本文引入自蒸馏思想，先用少量真实 PTB-XL 样本训练 Teacher 模型，再让 Teacher 对 ECGTwin 合成样本进行质量筛选和软标签标注。Student 模型最终同时学习真实样本的硬标签和合成样本的软标签。这样既利用了 ECGTwin 的数据扩充能力，又减少了低质量合成样本对分类器的负面影响。

---

## 20. 最终推荐版本

最小可跑版本：

```text
Teacher:
  Real-2000 EfficientNetV2-1D best checkpoint
  或 3-seed ensemble Teacher

Synthetic:
  每类生成 400 条，总共 2000 条

Teacher inference:
  4-crop 平均预测

Filtering:
  target prob >= 0.6 → high, weight = 1.0
  0.35 <= target prob < 0.6 且 target in top-2 → hard, weight = 0.5
  其他丢弃

Soft label:
  y_soft = 0.3 * y_condition + 0.7 * p_teacher

Student:
  Real hard label + Synth soft label
  BCEWithLogitsLoss
  beta = 0.5
  real:synth batch ratio = 1:1

Evaluation:
  PTB-XL fold10 real test only
  Macro-AUROC / Macro-AUPRC / per-class AUPRC
```

如果该版本有效，再做增强：

```text
EMA Teacher
Consistency Loss
DANN real/synthetic alignment
PN2021-C robustness evaluation
```

---

## 21. 与在线对抗训练的关系

该方案不等价于在线 FGSM/PGD 对抗训练。

区别：

| 方法 | 主要作用 | 是否主动制造最坏扰动 | 对 clean AUROC/AUPRC 的稳定性 |
|---|---|---:|---:|
| 自蒸馏 + soft label | 稳定利用合成样本，降低标签噪声 | 否 | 通常更稳 |
| 在线 FGSM/PGD | 提升抗扰动鲁棒性 | 是 | 不一定提升，可能下降 |

本文建议：

```text
主方案：自蒸馏 + ECGTwin 软标签筛选
可选增强：corruption consistency
最后再考虑：在线对抗训练
```

最终结论：

> 如果目标是在 2000 条 PTB-XL 真实样本 + 大量 ECGTwin 合成样本场景下提升 EfficientNetV2-1D Superclass 5 的 AUROC / AUPRC，自蒸馏软标签方案比在线对抗训练更适合作为主方案。

---

## 22. 2026-05-03 首次落地实验结果

本方案已经按当前毕业设计主线完成一次可复现实验。

与本文原始设想的区别：

```text
原始设想:
  Teacher 使用 4-crop，对应 2.5s 输入模型。

本次执行:
  当前 EfficientNet1DV2 主线已经改为 10s 输入，所以 teacher 使用 full-10s
  预测，不使用 4-crop。
```

实际执行配置：

```text
Teacher:
  3-seed real2000 teacher ensemble

Synthetic candidate pool:
  20000 ECGTwin synthetic ECG

Filtering:
  每类最多保留 400 条
  总共保留 2000 条
  high threshold = 0.6
  hard threshold = 0.35
  NORM threshold = p_NORM >= 0.7 and max abnormal <= 0.3

Soft label:
  y_soft = 0.3 * y_condition + 0.7 * p_teacher_ensemble

Student:
  scratch EfficientNet1DV2
  real2000 hard BCE + synth2000 soft BCE
  beta = 0.5
  real:synth = 1:1
```

Artifacts:

```text
Filtered synth:
  /root/autodl-tmp/graduate_project/self_distill_v2_filtered_real2000_ens3_seed42/synth_v2_filtered_top2000_gamma03.npz

Student run:
  /root/autodl-tmp/graduate_project/self_distill_v2_e4_real2000_ens3_filtered2000_gamma03_scratch_seed42
```

结果：

| model | custom seed42 AUROC | custom seed42 AUPRC | official fold10 AUROC | official fold10 AUPRC | PN2021 avg AUROC | PN2021 avg AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| best real2000 teacher seed44 | 0.8506 | 0.6469 | 0.8417 | 0.6482 | 0.7594 | 0.4339 |
| v2 E4 scratch student | 0.8500 | 0.6371 | 0.8405 | 0.6380 | 0.7616 | 0.4546 |

结论：

```text
v2 E4 没有超过最强 real2000 teacher 的 PTB-XL AUPRC。
v2 E4 超过了所有 real2000 teachers 的 PN2021 avg AUPRC。
因此当前证据支持：
  v2 soft-label filtered synthetic samples 更像是 cross-center regularizer；
  它对外部中心泛化有帮助，但还没有证明能提高 PTB-XL in-domain 上限。
```
