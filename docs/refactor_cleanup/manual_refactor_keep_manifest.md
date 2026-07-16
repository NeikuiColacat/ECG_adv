# 手动重构保留清单与旧代码清理闸门

更新日期：2026-07-16

## 目的

这份文档记录本轮从零手动重构所建立或明确修改的代码、配置和测试。
后续清理老旧代码时，以这里的逐文件清单作为保护白名单，并通过下方的
删除闸门确认旧文件已经被替代。

这不是立即删除授权，也不是“未列出的文件都可以直接删除”。任何破坏性
清理仍需先生成候选列表、逐项验证替代关系，并由用户确认后执行。

## 状态定义

| 状态 | 含义 | 清理规则 |
|---|---|---|
| `KEEP-MANUAL` | 本轮手动重构的新主线文件 | 默认禁止删除；改名或合并后必须同步更新本清单 |
| `TRANSITION` | 搬迁过来的旧实现，尚未完成重新设计 | 不算新主线；替代功能完成并验证后优先删除 |
| `SUPPORT` | 测试或验证辅助文件，不是产品主线 | 可以随被保护实现一起重写，但不能先于对应契约删除 |
| `CALIBRATION-ONLY` | 一次性校准或证据回放入口 | 运行记录保留完整配置后，可以进入删除候选 |
| `LEGACY-BRIDGE` | 仍被旧体系调用的现有文件 | 不受手工主线保护；完成调用迁移后删除 |
| `LOCAL-ONLY` | 本机路径、软链接或运行环境配置 | 不进入公共代码主线；不能删除其指向的外部数据或模型 |
| `GENERATED` | 缓存、字节码、实验输出 | 不进入 Git，可在确认没有运行占用后清理 |

## A. 手动重构保护白名单

### A1. 数据预处理配置

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `configs/data/README.md` | `KEEP-MANUAL` | 数据配置目录说明和缓存契约 | 新文档完整接管并更新所有引用 |
| `configs/data/PTBXL.yaml` | `KEEP-MANUAL` | PTB-XL 路径、采样率、导联、缓存配置 | 新配置通过相同数据契约测试 |
| `configs/data/PN2021.yaml` | `KEEP-MANUAL` | PN2021 路径、中心、采样率、导联和缓存配置 | 新配置通过相同数据契约测试 |
| `configs/data/PN2021_super5_v7.yaml` | `KEEP-MANUAL` | PN2021 到 Super5 v7 的配置化映射 | 映射版本/hash 和标签测试完全等价 |
| `configs/data/splits.yaml` | `KEEP-MANUAL` | PTB-XL 官方折与 PN2021 四逻辑中心 K500 切分契约；逻辑 `cpsc_2018` 合并 CPSC 2018/Extra | 新配置保留 source cache、患者隔离、合并中心、seed、K500 和 ref-exclusion 身份 |
| `configs/data/data_load.yaml` | `KEEP-MANUAL` | 统一 DataLoader batch/worker/pin-memory/mmap/shuffle 与模型输入变换默认参数；允许受管实验显式覆盖 | 新配置保留严格 schema、multi-worker mmap 安全门、评估不 shuffle、解析值和配置 SHA256 身份 |
| `configs/random_seed.yaml` | `KEEP-MANUAL` | 手动重构代码统一使用的项目基础随机种子 | 新配置保留基础 seed、配置 SHA256 和所有白名单随机入口 |

### A2. 数据预处理代码

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `data_preprocess/PTBXL_preprocess.py` | `KEEP-MANUAL` | PTB-XL 读取、非有限值质量控制、10 秒截取、100/500 Hz NPY 缓存和唯一 ID | 新入口完成同一缓存契约并通过数据验证 |
| `data_preprocess/PN2021_preprocess.py` | `KEEP-MANUAL` | PN2021 多中心读取、非有限值质量控制、Super5 映射、10 秒截取、100/500 Hz NPY 缓存和唯一 ID | 新入口完成同一缓存契约并通过数据验证 |
| `data_preprocess/augmentations_cache.py` | `KEEP-MANUAL` | 从白名单 PN2021 缓存生成五中心、depth2+3、100/500 Hz PN2021-C 连续缓存 | 新入口保留源缓存身份、20组合、随机种子和双采样率契约 |
| `data_preprocess/load_cache.py` | `KEEP-MANUAL` | 统一校验并只读访问 PTB-XL、PN2021、PN2021-C 缓存；按实时可用内存自动选择 RAM/mmap，并支持单条/批量 hash 到 index 的严格解析 | 新数据层保留 manifest/hash、布局、中心、view、内存安全门和原始 mV 契约 |
| `data_preprocess/split_cache.py` | `KEEP-MANUAL` | 依赖统一 cache loader 生成 ID-only PTB-XL 官方折和 PN2021 K500/ref-excluded 切分；合并 CPSC 2018/Extra 为一个逻辑中心 | 新切分层保留 source manifest、映射、seed、候选池、split hash、患者隔离和零 K500 泄漏契约 |
| `data_preprocess/data_runtime.py` | `KEEP-MANUAL` | 校验 split/cache 身份并通过统一 `get_dataloader` 提供 PTB-XL 三划分、目标中心 K500、ref-excluded PN2021/PN2021-C、运行时增强后清理/全样本 z-score、worker-safe Dataset 和确定性 DataLoader | 新运行时层保留原始 mV 到模型输入的顺序、显式 PN2021-C view、CPSC 合并中心、100/500 Hz 布局和 seed 身份 |
| `data_preprocess/prepare_ptbxl_for_ecgtwin.py` | `TRANSITION` | 旧 ECGTwin VAE/Nomic 搬迁脚本；1000→1024 已改为线性插值 | 所需 VAE/Nomic 功能进入正式模块，或确认主线不再需要 |

说明：`data_preprocess/prepare_ptbxl_for_ecgtwin.py` 最初从旧路径原样搬迁，
现已先将 1000→1024 的 FFT 重采样替换为线性插值，但其余 VAE/Nomic
职责仍未完成模块化，因此继续标记为 `TRANSITION`。

当前手工预处理插值策略统一为 `linear + align_corners=True`：PN2021
native-rate 到 100 Hz、100 Hz 到 500 Hz，以及迁移脚本中的 PTB-XL
1000 点到 ECGTwin 1024 点均使用该策略。PN2021 新缓存身份为
`pn2021_100hz_linear_v2`，禁止把旧算法缓存改写元数据后复用。

### A3. 五算子及 PN2021-C 配置

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `util/augmentations/__init__.py` | `KEEP-MANUAL` | 五算子公共导出接口 | 新包提供相同公共接口并完成调用方迁移 |
| `util/augmentations/operators.py` | `KEEP-MANUAL` | fairseq-signals 对齐的五算子函数实现 | 单一新实现通过官方对齐、随机性和形状测试 |
| `util/augmentations/torch_operators.py` | `KEEP-MANUAL` | 与 NumPy 接口对齐、可在 CPU/CUDA 设备原生执行的五算子实现 | 新实现同时通过 CPU 方程对齐和显式 CUDA 测试 |
| `util/random_seed.py` | `KEEP-MANUAL` | 全项目读取 seed YAML、记录身份，并为 Python/NumPy/Torch/CUDA/DataLoader 构造 namespace 隔离随机流 | 新实现保留基础/有效 seed、namespace、配置 SHA256、派生算法和 backend 内复现测试 |
| `util/config_bundle.py` | `KEEP-MANUAL` | 将 YAML 间引用限制在当前选中的 configs-shaped 配置束内，使完整配置副本可移植且不回落主配置 | 新实现保留显式 config root、禁止相对路径逃逸和副本优先解析测试 |
| `configs/augmentation/operators.yaml` | `KEEP-MANUAL` | paper-anchored S5 参数、单位和 depth2+3 组合契约 | 新配置保留 profile 名、参数来源和可复现 hash |
| `configs/augmentation/cache.yaml` | `KEEP-MANUAL` | 五中心 PN2021-C depth2+3 双采样率缓存输入、输出和断点续建契约 | 新配置保留源缓存身份、组合顺序、采样率和机械盘输出身份 |

### A4. 模型定义与构建接口

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `models/__init__.py` | `KEEP-MANUAL` | 新模型包唯一公共导出面 | 新包完成所有调用迁移并提供相同公开接口 |
| `models/contracts.py` | `KEEP-MANUAL` | 锁定 EfficientNet `(B,12,1000)`、ECGFounder `(B,12,5000)` 与 Super5 五原始 logits 契约 | 新实现保留采样率、布局、类别顺序及输入输出严格校验 |
| `models/checkpoints.py` | `KEEP-MANUAL` | 严格提取 raw/state_dict/model_state_dict、去除并行前缀、记录 SHA256，并仅为显式官方 ECGFounder checkpoint 开放 trusted 读取 | 新实现保留安全默认、严格键校验和 checkpoint 身份记录 |
| `models/efficientnet1d.py` | `KEEP-MANUAL` | 独立重写历史 EfficientNet1DV2 S-V2 五分类定义及 checkpoint 构建接口 | 新实现与571个state key、旧checkpoint和100 Hz前向逐元素兼容 |
| `models/ecgfounder.py` | `KEEP-MANUAL` | 独立重写官方12导联 ECGFounder Net1D、预训练backbone加载、Super5 head及 full/head trainable scope | 新实现与509个官方state key/shape、官方checkpoint和500 Hz前向兼容 |
| `models/factory.py` | `KEEP-MANUAL` | `build_model`、模型名规范化、可用模型和输入spec统一查询 | 新模型工厂完整接管两个backbone且不引入旧训练器依赖 |
| `models/vae.py` | `KEEP-MANUAL` | 独立重写 ECGTwin VAE encoder/decoder、严格读取双 state-dict checkpoint，并返回冻结的可微解码组件 | 新实现保留 `(B,1024,12)` raw mV、scaled `(B,4,128)` latent、导联交换、0.18215 scale、240个 state key 和真实 checkpoint 前向契约 |

### A5. 在线 VAE-LHAT 与三链 AugMix 核心

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `configs/train/vae.yaml` | `KEEP-MANUAL` | ECGTwin VAE checkpoint、输入/latent、冻结方式和分类器桥接契约 | 新配置保留 checkpoint 双组件、raw mV、1024点、lead reorder、latent scale 和严格键数量 |
| `configs/train/lhat.yaml` | `KEEP-MANUAL` | repaired M20 exact-label/non-self latent hull、标准化、L2投影、解码域和诊断参数 | 新配置保留 train-only standardizer、`include_anchor=false`、显式 epsilon、攻击几何和全部诊断 |
| `configs/train/augmix.yaml` | `KEEP-MANUAL` | 两条独立 depth2/3 paper-anchored S5 腐蚀链加一条纯 VAE-LHAT 波形链的三链混合与 loss 默认值，并明确该 profile 不是官方 preset | 新配置保留 profile 身份、chain3 不再腐蚀、pre-zscore、100/500 Hz、透明 Dirichlet/Beta 权重和 JSD 契约 |
| `core/__init__.py` | `KEEP-MANUAL` | 在线攻击与三链样本生成唯一公共导出面 | 新核心包接管相同接口并完成训练调用迁移 |
| `core/lhat.py` | `KEEP-MANUAL` | train-only latent standardizer、exact-label non-self 候选选择、临时 hull-logit 优化、可微 VAE 解码及攻击诊断 | 新实现保留仅优化 batch-local hull 权重、不污染模型梯度、100/500 Hz桥接及全部机制诊断 |
| `core/augmix.py` | `KEEP-MANUAL` | 批量采样20种 depth2+3 composition、生成两条腐蚀链、直通 chain3、三链混合、全样本 z-score 和多标签 JSD | 新实现保留 GPU batch、隔离 RNG、非原地输入、chain3 无附加腐蚀和可复算混合公式 |

## B. 当前不属于手工主线的文件

### B1. 非核心支持与待退出文件

这些文件不属于 A 区的手工重构保护白名单。它们当前仍有验证、兼容或证据
作用，因此只是允许后续退出，不是本次直接删除清单。

| 文件 | 状态 | 当前作用 | 何时可以删除或还原 |
|---|---|---|---|
| `configs/experiments/pn2021c_effnet_paper_anchored_s5_depth23_composite.yaml` | `CALIBRATION-ONLY` | 复现本次四中心 direct-K500 depth2+3 校准 | managed run 已保存 resolved config，且确认不再重跑该校准 |
| `methods/augmix/ecg_ops.py` | `LEGACY-BRIDGE` | 旧 PN2021-C 调用入口以及临时 profile 兼容 | 调用方全部迁移到 `util/augmentations/`，历史回放测试通过 |
| `ecg_adv_gen/data/synthetic_npz.py` | `LEGACY-BRIDGE` | 旧分类器输入布局兼容 | 新数据接口统一输出布局并迁移所有训练入口 |
| `ecg_adv_gen/training/effnet_super5.py` | `LEGACY-BRIDGE` | 旧 EfficientNet Dataset 布局兼容 | 新训练 Dataset 完成并通过 smoke test |
| `util/tests/test_augmentations.py` | `SUPPORT` | 验证新五算子参考对齐 | 对应契约迁入最终测试目录，或算子实现被整体替代 |
| `util/tests/test_pn2021_corruptions.py` | `SUPPORT` | 验证旧 PN2021-C bridge 和自定义 profile | bridge 删除后，将仍有价值的测试迁入最终接口测试 |
| `util/tests/test_data_contracts.py` | `SUPPORT` | 验证旧/新数据布局兼容 | 新数据接口拥有独立契约测试后移除本轮追加部分 |
| `util/tests/test_labels_super5.py` | `SUPPORT` | 验证 Super5 v7 标签映射 | 映射测试迁入最终数据预处理测试后移除本轮追加部分 |
| `util/tests/test_preprocess_nonfinite.py` | `SUPPORT` | 验证 PTB-XL/PN2021 非有限值插值修复、边界填补与严重异常剔除契约 | 质量控制逻辑迁入最终数据层时同步迁移这些测试 |
| `util/tests/test_augmentations_cache.py` | `SUPPORT` | 验证 PN2021-C 白名单依赖、20组合、确定性随机流、双采样率及断点缓存契约 | 缓存入口迁移时同步迁移这些测试 |
| `util/tests/test_load_cache.py` | `SUPPORT` | 验证三类缓存身份、自动 RAM/mmap 决策、中心/hash/view 访问和布局转换 | 统一数据访问层迁移时同步迁移这些契约测试 |
| `util/tests/test_split_cache.py` | `SUPPORT` | 验证 PTB-XL 官方患者隔离折、Super5 全零排除、CPSC 2018/Extra 逻辑合并、确定性 K500 与 ref-exclusion | 切分层迁移时同步迁移这些契约测试 |
| `util/tests/test_data_runtime.py` | `SUPPORT` | 验证 split/cache 身份绑定、CPSC 合并中心、PN2021-C ref-excluded/view 闸门、运行时变换顺序、DataLoader 随机种子以及 data_load YAML/显式覆盖优先级 | 运行时数据层迁移时同步迁移这些契约测试 |
| `util/tests/test_models.py` | `SUPPORT` | 验证两个模型的输入输出、冻结范围、统一factory、外部参考state键及 EfficientNet逐元素前向对齐 | 模型包迁移时同步迁移官方/历史兼容契约测试 |
| `util/tests/test_vae_lhat_augmix.py` | `SUPPORT` | 验证真实 VAE checkpoint、latent standardizer/候选、LHAT 几何诊断、三链 AugMix 公式、确定性和白名单 import 闭包 | 在线训练核心迁移时同步迁移这些契约测试 |
| `util/tests/test_random_seed.py` | `SUPPORT` | 验证全局 seed YAML 身份、namespace 隔离、Python/NumPy/Torch 复现和 process 初始化 | 随机基础设施迁移时同步迁移这些契约测试 |
| `util/tests/test_config_bundle.py` | `SUPPORT` | 验证完整 configs 副本内的 seed/operator 引用、路径逃逸闸门和核心 YAML 加载 | 配置系统迁移时同步迁移这些契约测试 |

### B2. 本机运行支持

| 路径 | 状态 | 说明 |
|---|---|---|
| `configs/local/linbinhao_manual_refactor.yaml` | `LOCAL-ONLY` | 指向当前重构仓库和 `/home/linbinhao/ECG_adv_data`；由 `configs/local/.gitignore` 排除 |
| `model/DeepECG`、`model/ECGTwin`、`model/advdiff`、`model/ecg_ptbxl_benchmarking`、`model/ecgfounder` | `LOCAL-ONLY` | 当前主机的外部模型句柄；允许重建软链接，禁止删除链接目标 |

### B3. 生成物

以下内容不纳入代码保护白名单：

- `**/__pycache__/`
- `*.pyc`
- 数据缓存：`*.npy`、`*.npz`、`*.pt`、`*.pth`、`*.ckpt`
- `/home/linbinhao/ECG_adv_data/runs/` 下的运行输出
- `/dev/shm` 下的临时日志和中间诊断

实验结果可以按证据策略归档或清理，但不能因代码清理而误删外部数据集、
选定 checkpoint、最终指标或运行复现记录。

## C. 已识别的旧实现清理候选

这些路径只是候选，当前禁止直接删除。

| 候选 | 预期替代 | 当前阻塞条件 |
|---|---|---|
| `data/prepare_ptbxl_for_ecgtwin.py` | `data_preprocess/prepare_ptbxl_for_ecgtwin.py` 或未来正式 ECGTwin 预处理模块 | 当前工作树已表现为搬迁；新位置仍是旧实现，尚未完成模块化 |
| `methods/augmix/ecg_ops.py` | `util/augmentations/` 的最终单一实现 | PN2021-C 和训练代码仍有直接 import；自定义 profile 兼容逻辑仍在此处 |
| `methods/augmix/severity.py` | `configs/augmentation/operators.yaml` 加统一 profile loader | 旧 standard S5 和历史实验仍需回放；必须先迁移调用方 |
| `configs/experiments/pn2021c_effnet_paper_anchored_s5_depth23_composite.yaml` | managed run 保存的 resolved config 与 `configs/augmentation/operators.yaml` | 确认不再需要从源码树重新运行该校准 |
| `configs/corruption_profiles/` 中被替代的历史 profile | `configs/augmentation/operators.yaml` | 需要逐文件检查 active config、证据记录和历史回放引用，禁止整目录删除 |
| 旧数据预处理入口和旧实验 YAML | `data_preprocess/` 与受管实验 YAML | 需要从 `configs/active_scripts.yaml`、import closure 和运行记录生成精确候选清单 |

## D. 删除闸门

一个旧文件只有同时满足以下条件，才能从“候选”变成“允许删除”：

1. 在本清单中写明它的精确替代文件和职责映射。
2. 使用 `rg` 确认没有活动代码、YAML、测试或文档仍引用旧路径。
3. 新实现通过对应单元测试和数据契约测试。
4. 相关 `scripts/run_experiment.py --dry-run` 通过协议审计。
5. 涉及数据或模型输入时，至少完成一次小规模 smoke test。
6. 涉及论文结果时，确认旧文件不是唯一的回放或证据来源。
7. 生成待删除的精确路径清单；禁止使用目录级通配删除。
8. 用户明确确认该批删除后，再使用非交互式 `git rm` 或逐文件删除。
9. 删除后重新运行测试、workspace audit 和 `git diff --check`。

建议的最低验证命令：

```bash
git diff --check
pytest -q util/tests/test_augmentations.py \
  util/tests/test_torch_augmentations.py \
  util/tests/test_data_contracts.py \
  util/tests/test_labels_super5.py \
  util/tests/test_pn2021_corruptions.py
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
```

涉及 GPU、长推理或缓存构建的验证，仍需先检查共享服务器资源并显式选择 GPU。

## E. 清理批次记录

每次实际删除前后，在这里追加记录；不要改写历史批次。

| 日期 | 批次 | 删除路径 | 替代路径 | 验证结果 | 用户确认 |
|---|---|---|---|---|---|
| 待执行 | 无 | 无 | 无 | 无 | 无 |

## F. 当前下一步

- [x] 为两个数据预处理脚本补齐非有限值质量控制的独立最小测试。
- [x] 实现 PTB-XL、PN2021、PN2021-C 的统一只读缓存加载与访问接口。
- [x] 实现 PTB-XL 官方折及 PN2021 四逻辑中心 K500/ref-excluded 的 ID-only 切分层，并将 CPSC 2018/Extra 合并为一个逻辑中心。
- [x] 实现从受管 split/cache 到模型 batch 的统一运行时数据层，包括显式 corruption view、增强后全样本 z-score、worker-safe Dataset 和确定性 DataLoader。
- [x] 建立独立 `models/` 包，完成 EfficientNet1DV2、ECGFounder、严格checkpoint身份和统一 `build_model` 接口，并通过真实checkpoint CPU smoke test。
- [x] 独立重写 ECGTwin VAE encoder/decoder，并实现 repaired VAE-LHAT 在线攻击、两腐蚀链加一纯 VAE 链的三链 AugMix 及其受管训练 YAML。
- [ ] 决定 ECGTwin VAE/Nomic 预处理是否保留，并重写或删除 `TRANSITION` 脚本。
- [ ] 将五算子收敛成单一实现，迁移所有活动 import。
- [ ] 将 profile loader 与 `configs/augmentation/operators.yaml` 接入训练和评估共同入口。
- [x] PN2021-C 缓存构建完成后，将随机种子能力提升为全项目公共基础设施：
  已将 `configs/radom_seed.yaml` 更名为 `configs/random_seed.yaml`，并将
  `util/augmentations/random_seed.py` 迁移为 `util/random_seed.py`，统一提供
  YAML 加载、身份化 seed 派生、Python/NumPy/Torch/CUDA 初始化和独立 RNG
  工厂；当前白名单内的模型工厂、切分、DataLoader worker、数据增强和
  对抗攻击已迁移到新入口。公平对比的各方法臂必须共享
  comparison-group/replicate seed，
  每个运行记录基础 seed、有效 seed、namespace 和 seed YAML SHA256；迁移后
  删除了增强子包中的旧实现，不保留旧拼写兼容入口。
- [x] 增加 configs-shaped 配置束解析：复制整个 `configs/` 后，内部 YAML
  引用优先解析副本自身；数据运行时和模型工厂支持 `config_root`，各独立
  预处理/缓存/LHAT/AugMix 入口支持显式副本 YAML 路径。
- [ ] 新训练器/统一实验 launcher 建立时，只接受一个 experiment YAML 加
  `config_root`，启动前解析并保存依赖 YAML 闭包、SHA256、seed identity、
  Git 身份和精确命令；禁止重新引入另一套隐式默认配置搜索规则。
- [ ] 生成第一批逐文件旧代码候选，但暂不删除。
- [ ] 用户审阅候选清单后再执行破坏性清理。
