# Thesis Reference Index

ECGTwin Adversarial Cross-Center Augmentation 方向的论文 + 代码 reference 总汇。
所有 arxiv PDF 已本地化，4 个核心实现 repo 已 clone。

**Generated**: 2026-04-25
**PDFs**: 32 篇 / 7 主题 / ~285 MB
**Repos**: 4 个 / `/root/autodl-tmp/external_repos/` / ~234 MB

---

## 优先级图例
- **P0** = 必读，毕设故事核心 / 强 prior art
- **P1** = 推荐，方法论参考
- **P2** = 选读，工具书 / 综述

---

## Topic 1 — Adversarial Diffusion (`01_adv_diffusion/`)

| 优先级 | Paper | arxiv | 与本方案关系 |
|---|---|---|---|
| **P0** | **AdvDiff** (ECCV 2024) | [2307.12499](https://arxiv.org/abs/2307.12499) | PGD-on-VAE-latent 的扩散域父工作；**核查后发现"最后 20% 引导"是误传，实际全步引导**，但 Sec 4.5 自己说早期 t 信号弱——直接用作我们方案的 motivation |
| **P0** | **Madry PGD** (ICLR 2018) | [1706.06083](https://arxiv.org/abs/1706.06083) | 经典对抗训练 minimax 公式，Latent-PGD 的 first-order 解 |
| **P0** | **NatADiff** (arXiv 2025) | [2505.20934](https://arxiv.org/abs/2505.20934) | time-travel + 双向 boundary loss，对抗扩散最新升级；如果做 Mode B 这是直接模板 |
| P1 | Better Diffusion AT (Wang ICML 2023) | [2302.04638](https://arxiv.org/abs/2302.04638) | "扩散合成数据真的提升对抗鲁棒性"的图像域 SOTA 证据 |
| P1 | DiffPure (NeurIPS 2022) | [2205.07460](https://arxiv.org/abs/2205.07460) | 反向视角：理解 SDEdit/AdvDiff/DiffPure 三角关系 |
| P2 | COUP (ECAI 2024) | [2408.05900](https://arxiv.org/abs/2408.05900) | 反向用法：当 PGD 输出的"流形质量闸"过滤工具 |

---

## Topic 2 — Few-shot Diffusion DA (`02_fewshot_diffusion/`)

| 优先级 | Paper | arxiv | 与本方案关系 |
|---|---|---|---|
| **P0** | **Textual Inversion** (Gal ICLR 2023) | [2208.01618](https://arxiv.org/abs/2208.01618) | **CenterToken 的方法论祖宗**——只学 token embedding 不动权重 |
| **P0** | **LoRA** (Hu ICLR 2022) | [2106.09685](https://arxiv.org/abs/2106.09685) | 如果 CenterToken 容量不够，下一步必然是 LoRA on DiT |
| **P0** | **DIPSY** (BMVC 2025) | [2509.22635](https://arxiv.org/abs/2509.22635) | 双 IP-Adapter (positive + negative refs) 训练-free，与 K-shot 目标中心场景 1:1 mirror |
| P1 | DreamBooth (CVPR 2023) | [2208.12242](https://arxiv.org/abs/2208.12242) | CenterToken 的对照面 (full-tune vs token-only) |
| P1 | Custom Diffusion (CVPR 2023) | [2212.04488](https://arxiv.org/abs/2212.04488) | 中间档：只 fine-tune cross-attn 的 K/V，多 center 合并训练蓝本 |
| P1 | IP-Adapter (Ye 2023) | [2308.06721](https://arxiv.org/abs/2308.06721) | "参考 ECG 直接做 image prompt"路径的标准范式 |

---

## Topic 3 — Diffusion Augmentation (`03_diffusion_aug/`)

| 优先级 | Paper | arxiv | 与本方案关系 |
|---|---|---|---|
| **P0** | **SDEdit** (Meng ICLR 2022) | [2108.01073](https://arxiv.org/abs/2108.01073) | "去噪深度"调度的源头；ECG cross-center 翻译时 t 选择的理论基础 |
| P1 | Diff-Mix (CVPR 2024) | [2403.19600](https://arxiv.org/abs/2403.19600) | super5 类间 boundary-aware augmentation 的直接模板 |
| P2 | DiffuseMix (CVPR 2024) | [2405.14881](https://arxiv.org/abs/2405.14881) | "label-preserving" 形式化讨论 |
| P2 | Diffusion Aug Review (arXiv 2024) | [2407.04103](https://arxiv.org/abs/2407.04103) | 综述章前 1 小时翻 |

---

## Topic 4 — Domain Adaptation (`04_domain_adaptation/`)

| 优先级 | Paper | arxiv | 与本方案关系 |
|---|---|---|---|
| P1 | GDA (CVPR 2024) | [2404.00095](https://arxiv.org/abs/2404.00095) | 测试时把跨中心样本扔回源域 manifold；K=0 极端少样本互补 |
| P1 | SDA (CVPR 2025) | [2406.04295](https://arxiv.org/abs/2406.04295) | "ECGTwin 合成域当共同锚点"的清晰范式；解释 vanilla null-style 为啥只在 cpsc_2018_extra 涨 |

---

## Topic 5 — ECG Synthesis (`05_ecg_synthesis/`)

| 优先级 | Paper | arxiv | 与本方案关系 |
|---|---|---|---|
| **P0** | **SSSD-ECG** (CBM 2023) | [2301.08227](https://arxiv.org/abs/2301.08227) | ECGTwin 可比 baseline，做合成质量 head-to-head |
| **P0** | **PTB-XL Benchmarks** (Strodthoff JBHI 2020) | [2004.13701](https://arxiv.org/abs/2004.13701) | super5/sub23 标签层级的源头；评测协议 ground truth |
| P1 | DiffECG (arXiv 2023/2024) | [2306.01875](https://arxiv.org/abs/2306.01875) | 同框架支持 generation/imputation/forecasting |
| P2 | SSSD-ECG-nle (arXiv 2024) | [2407.11108](https://arxiv.org/abs/2407.11108) | label embedding 升级版 |

---

## Topic 6 — Robustness Theory (`06_robustness_theory/`)

| 优先级 | Paper | arxiv | 与本方案关系 |
|---|---|---|---|
| **P0** | **Tsipras et al.** (ICLR 2019) | [1805.12152](https://arxiv.org/abs/1805.12152) | "Robustness May Be at Odds with Accuracy"——解释对抗增强可能轻伤 in-domain，写 Discussion 必引 |
| **P0** | **Han ECG Attack** (Nat Med 2020) | [1905.05163](https://arxiv.org/abs/1905.05163) | ECG adversarial 领域奠基；标准 PGD 在 ECG 上方波 artifact 问题 |
| P1 | Schmidt et al. (NeurIPS 2018) | [1804.11285](https://arxiv.org/abs/1804.11285) | "robust learning 需要更多数据"——你用合成补样本量的理论 motivation |

---

## Topic 7 — On-Manifold AT (CORE LINEAGE) (`07_onmanifold_AT/`)

> **本方案的直接 lineage**——VAE-latent PGD 落在这条线上。这一格的引用是论文叙事的支柱。

| 优先级 | Paper | arxiv | 引用 | 与本方案关系 |
|---|---|---|---|---|
| **P0🌟** | **Stutz et al. — Disentangling Robustness and Generalization** (CVPR 2019) | [1812.00740](https://arxiv.org/abs/1812.00740) | **414** | **canonical on-manifold AT 论文**。结论：**on-manifold AT 提升 generalization，off-manifold AT 提升 robustness——两者正交**。这恰好就是你 `pgd_cross_center_ablation.md` 里看到的：8/15 cells 跨中心 +0.3pp = on-manifold AT 提升 generalization 的 ECG 实证。**毕设 framing 锚点** |
| **P0🌟** | **Wong et al. — Learning Perturbation Sets** (ICLR 2020) | [2007.08714](https://arxiv.org/abs/2007.08714) | ~270 | **最接近本方案的方法论模板**。CVAE latent 球 = 真实扰动集，PGD on latent 给可证明 robustness |
| **P0🌟** | **Pediatric ECG On-Manifold AT** (arXiv 2025) | [2509.19564](https://arxiv.org/abs/2509.19564) | 新 | **唯一发表的 ECG on-manifold AT 论文**——必读 + head-to-head 对比 |
| P1 | DMAT — Dual Manifold AT (NeurIPS 2020) | [2009.02470](https://arxiv.org/abs/2009.02470) | ~150 | 重要 caveat：**纯 latent-only PGD 防不住信号域 Lp 攻击**——在 Limitations 段提一句 |
| P1 | Zhao — Generating Natural Adversarial Examples (ICLR 2018) | [1710.11342](https://arxiv.org/abs/1710.11342) | ~900 | GAN-latent 对抗的奠基论 |
| P1 | Song — Constructing Unrestricted Adversarial Examples (NeurIPS 2018) | [1805.07894](https://arxiv.org/abs/1805.07894) | ~600 | latent-space 攻击能跨过 robust models |
| P2 | Sheshadri — Latent Adversarial Training (NeurIPS 2024) | [2407.15549](https://arxiv.org/abs/2407.15549) | ~95 | "LAT" 名字源头（LLM residual stream，非 VAE） |
| P2 | Khan — VAE Robustness via Local Geometry (AISTATS 2023) | [2208.03923](https://arxiv.org/abs/2208.03923) | ~30 | VAE latent **anisotropic** 的理论：不是每个维度都"语义保持" |
| P2 | Shukla — Latent Space Attack (CVPR-W 2023) | [2304.04386](https://arxiv.org/abs/2304.04386) | ~15 | autoencoder-latent attack pipeline 现代轻量参考 |

---

## Reference Repos (`/root/autodl-tmp/external_repos/`)

| Repo | Path | 用途 |
|---|---|---|
| **AdvDiff** | `external_repos/AdvDiff/` | 反向扩散 classifier guidance 的代码模板；Mode B 升级路径参考 |
| **Stutz_DisentanglingRobustness** | `external_repos/Stutz_DisentanglingRobustness/` | on-manifold AT 实现参考；和本方案 lineage 完全一致 |
| **IP-Adapter** | `external_repos/IP-Adapter/` | 如果 β' 路线 (DIPSY 风格)，参考 1D 化适配 |
| **HF_peft** | `external_repos/HF_peft/` | LoRA 标准实现，挂 ECGTwin DiT 用 |

---

## 推荐阅读路径 (2 周)

### Week 1 — 基础 + 方案 lineage
1. **Stutz 2019** — 你的 framing 锚点 (Section 3-4 + Conclusion，30 min)
2. **Wong 2020 perturbation sets** — 方法论模板
3. **Pediatric ECG On-Manifold 2025** — 1:1 prior art
4. **AdvDiff 2024** — Section 4.5 重点读（早期 t 信号弱的实证）
5. **Madry PGD** — 跳读 2-3
6. **Han ECG attack 2020** — 跳读 main result

### Week 2 — 方案细化 + 替代路线
7. **NatADiff 2025** — 如果决定升级 Mode B
8. **DIPSY 2025** — 如果决定 β' 替代路线
9. **DMAT 2020** — Limitations 段
10. **Tsipras 2019** — Discussion 段
11. **SSSD-ECG** — 合成质量 head-to-head 对比
12. **Strodthoff 2020** — 评测协议引用

### 工具书（写 method 章时翻）
- LoRA / Custom Diffusion / DreamBooth / IP-Adapter / SDEdit

---

## 一键 grep

```bash
# 找某主题所有 PDF
ls /root/ECG_adv_Gen/docs/papers/07_onmanifold_AT/

# 在所有 PDF 文本里搜关键词（需 pdftotext）
grep -rli "on-manifold" /root/ECG_adv_Gen/docs/papers/

# 看某 repo 结构
tree /root/autodl-tmp/external_repos/AdvDiff -L 2
```
