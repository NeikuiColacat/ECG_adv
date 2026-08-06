# 手动重构保留清单与旧代码清理闸门

更新日期：2026-08-06

## 目的

这份文档记录本轮从零手动重构所建立或明确修改的代码、配置和测试。
后续清理老旧代码时，以这里的逐文件清单作为保护白名单，并通过下方的
删除闸门确认旧文件已经被替代。

这不是立即删除授权，也不是“未列出的文件都可以直接删除”。任何破坏性
清理仍需先生成候选列表、逐项验证替代关系，并由用户确认后执行。

当前新增或改造的主线代码只能依赖本清单 A 区文件或其明确产物。实现时优先
原地复用这些文件；没有用户确认不得为了兼容旧协议另建 Python 模块、版本化
YAML 或隐式回退入口。

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

### A0. 项目入口与边界

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `AGENTS.md` | `KEEP-MANUAL` | 共享服务器安全、白名单依赖边界、主线数据/方法/证据契约和验证入口 | 新代理入口完整接管前 100 行安全规则、白名单原则和论文证据边界 |
| `README.md` | `KEEP-MANUAL` | 人类可读的手工重构入口、目录导航、dry-run 和开发证据边界 | 新项目首页完整接管当前 launcher、数据契约和验证命令 |
| `.codex/skills/README.md` | `KEEP-MANUAL` | 说明随仓库迁移的公开项目技能及当前用户级安装方式 | 新技能索引完整接管同一公开/私有边界和安装入口 |
| `.codex/skills/artifact-git-guard/SKILL.md` | `KEEP-MANUAL` | 提交与 push 前阻止权重、缓存、外部模型链接、秘密和运行产物进入 Git | 新 Git 防护流程覆盖相同风险并完成实际提交验证 |
| `.codex/skills/data-prep-validator/SKILL.md` | `KEEP-MANUAL` | 审核 v7 Super5、100 Hz canonical、VAE 桥、K500 与 ref-exclusion 数据契约 | 新数据验证技能接管相同身份、布局和泄漏闸门 |
| `.codex/skills/ecg-adv-gen/SKILL.md` | `KEEP-MANUAL` | 当前 clean-room 主线总路由；显式把旧 ECG_adv_Gen 内容降为历史来源 | 新总技能完整指向当前 AGENTS、keep manifest、active index 和 evidence registry |
| `.codex/skills/ecg-agent-retrospective/SKILL.md` | `KEEP-MANUAL` | 将重复经验路由到 AGENTS、manifest、registry、基线文档或小技能 | 新回顾入口保留最小更新、证据和不读取全局私有会话的约束 |
| `.codex/skills/ecg-agent-retrospective/references/handoff-template.md` | `KEEP-MANUAL` | 当前 clean worktree 的紧凑交接模板 | 新模板保留目标、状态、证据、未完成项和下一命令字段 |
| `.codex/skills/ecg-agent-retrospective/references/session-hygiene.md` | `KEEP-MANUAL` | 项目会话卫生与禁止自动删除/归档边界 | 新规则接管相同人工确认和隐私边界 |
| `.codex/skills/ecg-agent-retrospective/references/update-rules.md` | `KEEP-MANUAL` | 判断经验应进入 AGENTS、技能还是证据文件的最小规则 | 新规则保留可复用性、证据和上下文成本闸门 |
| `.codex/skills/ecg-vae-online-at/SKILL.md` | `KEEP-MANUAL` | VAE-LHAT 攻击、回缩、诊断、ASR 和论文安全选择的专项流程；当前 YAML 优先于历史设置 | 新专项技能接管当前 M20/λ0.6/ε12/10步契约和历史降级规则 |
| `.codex/skills/ecg-vae-online-at/references/literature_and_repos.md` | `KEEP-MANUAL` | VAE latent adversarial training、AugMix 及相关实现的文献与仓库线索 | 新参考文件保留来源区分和可追溯链接 |
| `.codex/skills/model-eval/SKILL.md` | `KEEP-MANUAL` | 统一两骨干 PN2021/PN2021-C、drop-all-zero、逐中心/逐类和 matched baseline 评估口径 | 新评估技能接管全部 metric identity 与 ref-exclusion 闸门 |
| `.codex/skills/reproducibility-check/SKILL.md` | `KEEP-MANUAL` | 审核配置闭包、Git/seed/checkpoint/命令/指标的可回放身份 | 新复现技能接管相同 run-record 和证据闭环 |
| `.codex/skills/shared-gpu-server-discipline/SKILL.md` | `KEEP-MANUAL` | GPU、CPU、内存、IO、进程、端口和输出目录的共享服务器前置检查 | 新资源纪律入口完整接管共享服务器安全规则 |
| `configs/README.md` | `KEEP-MANUAL` | configs-shaped 配置束、单一 launcher 和相对引用规则 | 新配置文档完整接管 bundle root、闭包、输出与覆盖规则 |
| `docs/refactor_cleanup/manual_refactor_keep_manifest.md` | `KEEP-MANUAL` | 手工重构保护边界、退出候选与破坏性删除闸门 | 新清单逐文件接管全部 KEEP/TRANSITION/证据和用户确认记录 |

仓库技能只保存公开、可复用的项目程序与必要参考。全局
`/home/linbinhao/.codex/memories`、Codex 会话、个人配置、凭据和本机私有状态
不属于学术仓库，不复制到本清单。

### A1. 数据预处理配置

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `configs/data/README.md` | `KEEP-MANUAL` | 数据配置目录说明和缓存契约 | 新文档完整接管并更新所有引用 |
| `configs/data/PTBXL.yaml` | `KEEP-MANUAL` | PTB-XL 路径、采样率、导联、缓存配置 | 新配置通过相同数据契约测试 |
| `configs/data/PN2021.yaml` | `KEEP-MANUAL` | PN2021 路径、中心、采样率、导联和缓存配置 | 新配置通过相同数据契约测试 |
| `configs/data/PN2021_super5_v7.yaml` | `KEEP-MANUAL` | PN2021 到 Super5 v7 的配置化映射 | 映射版本/hash 和标签测试完全等价 |
| `configs/data/splits.yaml` | `KEEP-MANUAL` | PTB-XL 官方折与 PN2021 四逻辑中心固定 K500；在不改变父 K500 身份的前提下派生 400/100 内部调参划分；逻辑 `cpsc_2018` 合并 CPSC 2018/Extra | 新配置保留 source cache、患者隔离、合并中心、父 K500 hash、独立 tuning seed、400/100 互斥并集和 ref-exclusion 身份 |
| `configs/data/data_load.yaml` | `KEEP-MANUAL` | 唯一数据运行配置；统一控制 RAM/mmap、K500 selection-resident、worker、pin-memory 和 prefetch | 新配置保留 split/cache/hash/顺序/RNG 身份，并保证只 gather 已验证 selection、连续 CPU 存储、源 mmap 及时关闭、worker 安全门和 mmap 数值等价 |
| `configs/data/k500_handoff.yaml` | `KEEP-MANUAL` | 学术方法对比的数据接口交接契约；锁定所需配置 SHA、四中心 K500/K400/K100/ref-excluded hash 身份和只读边界，不复制数据或 split 数组 | 新交接契约保留 source/split manifest、逐中心 hash-set、映射、类序、seed 来源、配置闭包及 heldout 只读语义 |
| `configs/random_seed.yaml` | `KEEP-MANUAL` | 手动重构代码统一使用的项目基础随机种子 | 新配置保留基础 seed、配置 SHA256 和所有白名单随机入口 |

### A2. 数据预处理代码

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `data_preprocess/PTBXL_preprocess.py` | `KEEP-MANUAL` | PTB-XL 读取、非有限值质量控制、10 秒截取、100/500 Hz NPY 缓存和唯一 ID | 新入口完成同一缓存契约并通过数据验证 |
| `data_preprocess/pn2021_metadata.py` | `KEEP-MANUAL` | 独立解析 PN2021 `.hea` 中的 SNOMED、年龄和性别，不导入旧 `ecg_adv_gen.data` | 新实现保持表头解析语义并通过独立 golden contract 测试 |
| `data_preprocess/PN2021_preprocess.py` | `KEEP-MANUAL` | PN2021 多中心读取、非有限值质量控制、Super5 映射、10 秒截取、100/500 Hz NPY 缓存和唯一 ID | 新入口完成同一缓存契约并通过数据验证 |
| `data_preprocess/augmentations_cache.py` | `KEEP-MANUAL` | 从白名单 PN2021 缓存生成五中心、depth2+3、100/500 Hz PN2021-C 连续缓存 | 新入口保留源缓存身份、20组合、随机种子和双采样率契约 |
| `data_preprocess/load_cache.py` | `KEEP-MANUAL` | 统一校验并只读访问 PTB-XL、PN2021、PN2021-C 缓存；按实时可用内存自动选择 RAM/mmap，并支持单条/批量 hash 到 index 的严格解析 | 新数据层保留 manifest/hash、布局、中心、view、内存安全门和原始 mV 契约 |
| `data_preprocess/split_cache.py` | `KEEP-MANUAL` | 依赖统一 cache loader 生成 ID-only PTB-XL 官方折、PN2021 固定 K500、确定性多标签/物理来源近似分层 400/100 及 ref-excluded 切分；singleton positive 保留在 train，CPSC/Extra validation 按父 K500 来源比例配额 | 新切分层保留 source manifest、映射、父 K500 seed/身份、独立 tuning seed、候选池、split hash、400/100 互斥并集、来源/类别计数、患者隔离和零 K500 泄漏契约 |
| `data_preprocess/data_runtime.py` | `KEEP-MANUAL` | 校验 split/cache 身份并统一提供各划分 DataLoader；支持批量 mmap/prefetch、只 gather 已验证 K500 到连续 CPU tensor 后关闭源 mmap，以及跨 clean+20 views 复用 mmap/selection 的 `SequentialEvaluationDataSession` | 新运行时层保留父 K500=400+100、cache index/hash/record/source-center/sampler 顺序、原始 mV 到模型输入的顺序、显式 PN2021-C view、CPSC 合并中心、100/500 Hz 布局、seed 身份、resident/mmap 数值等价和 loader/session 所有权契约 |
当前手工预处理插值策略统一为 `linear + align_corners=True`：PN2021
native-rate 到 100 Hz，以及 100 Hz 到 500 Hz 均使用该策略。PN2021 新缓存身份为
`pn2021_100hz_linear_v2`，禁止把旧算法缓存改写元数据后复用。

### A3. 五算子及 PN2021-C 配置

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `util/augmentations/__init__.py` | `KEEP-MANUAL` | 五算子公共导出接口 | 新包提供相同公共接口并完成调用方迁移 |
| `util/augmentations/profile.py` | `KEEP-MANUAL` | 离线 NumPy 缓存与在线 Torch AugMix 共用的五算子 profile/schema/seed/config-bundle 解析和 SHA256 身份 | 新实现保留单一参数源、severity/profile/canonical-order 校验、配置副本解析和 seed 身份 |
| `util/augmentations/operators.py` | `KEEP-MANUAL` | fairseq-signals 对齐的五算子函数实现 | 单一新实现通过官方对齐、随机性和形状测试 |
| `util/augmentations/torch_operators.py` | `KEEP-MANUAL` | 与 NumPy 公共接口对齐、可在 CPU/CUDA 原生批量执行的五算子，并为已统一校验的共享 kernel 提供无重复 finite 同步内部入口 | 新实现保留公共严格校验、CPU 方程对齐、显式 CUDA 测试和仅限白名单调用的 prevalidated fast path |
| `util/random_seed.py` | `KEEP-MANUAL` | 全项目读取 seed YAML、记录身份，并为 Python/NumPy/Torch/CUDA/DataLoader 构造 namespace 隔离随机流 | 新实现保留基础/有效 seed、namespace、配置 SHA256、派生算法和 backend 内复现测试 |
| `util/config_bundle.py` | `KEEP-MANUAL` | 将 YAML 间引用限制在当前选中的 configs-shaped 配置束内，使完整配置副本可移植且不回落主配置；提供递归 YAML 配置闭包供 selection/refit 留证复核 | 新实现保留显式 config root、禁止相对路径逃逸、副本优先解析和闭包测试 |
| `configs/augmentation/operators.yaml` | `KEEP-MANUAL` | paper-anchored S5 参数、单位和 depth2+3 组合契约 | 新配置保留 profile 名、参数来源和可复现 hash |
| `configs/augmentation/cache.yaml` | `KEEP-MANUAL` | 五中心 PN2021-C depth2+3 双采样率缓存输入、输出和断点续建契约 | 新配置保留源缓存身份、组合顺序、采样率和机械盘输出身份 |

### A4. 模型定义与构建接口

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `models/__init__.py` | `KEEP-MANUAL` | 新模型包唯一公共导出面 | 新包完成所有调用迁移并提供相同公开接口 |
| `models/contracts.py` | `KEEP-MANUAL` | 锁定 EfficientNet `(B,12,1000)`、ECGFounder `(B,12,5000)` 与 Super5 五原始 logits 契约 | 新实现保留采样率、布局、类别顺序及输入输出严格校验 |
| `models/input_adapter.py` | `KEEP-MANUAL` | 统一 canonical raw 100 Hz BTC 到两模型输入域的设备驻留适配：sanitize、ECGFounder 线性升采样、全局 z-score、BCT | 新实现保留 EffNet identity、ECGFounder `1000→5000` linear `align_corners=True`、先升采样后归一化、有限 float32 输出及非原地语义 |
| `models/checkpoints.py` | `KEEP-MANUAL` | 严格提取 raw/state_dict/model_state_dict、去除并行前缀、记录 SHA256，并仅为显式官方 ECGFounder checkpoint 开放 trusted 读取 | 新实现保留安全默认、严格键校验和 checkpoint 身份记录 |
| `models/efficientnet1d.py` | `KEEP-MANUAL` | 独立重写历史 EfficientNet1DV2 S-V2 五分类定义及 checkpoint 构建接口，并显式暴露冻结分类头前的 `forward_features` 供两链 AugMix-SimCLR 使用 | 新实现与571个state key、旧checkpoint和100 Hz前向逐元素兼容，且 feature API 不改变原分类前向 |
| `models/ecgfounder.py` | `KEEP-MANUAL` | 独立重写官方12导联 ECGFounder Net1D、预训练backbone加载、Super5 head及 full/head trainable scope；复用既有 `forward_features` 接入同一 Stage-1 | 新实现与509个官方state key/shape、官方checkpoint和500 Hz前向兼容，且两个骨干共享同一表征学习接口 |
| `models/factory.py` | `KEEP-MANUAL` | `build_model`、模型名规范化、可用模型和输入spec统一查询 | 新模型工厂完整接管两个backbone且不引入旧训练器依赖 |
| `models/vae.py` | `KEEP-MANUAL` | 独立重写 ECGTwin VAE encoder/decoder、严格读取双 state-dict checkpoint，并返回冻结的可微解码组件 | 新实现保留 `(B,1024,12)` raw mV、scaled `(B,4,128)` latent、导联交换、0.18215 scale、240个 state key 和真实 checkpoint 前向契约 |

### A5. 两链 AugMix-SimCLR 与收缩式 VAE-LHAT 主线

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `configs/train/vae.yaml` | `KEEP-MANUAL` | ECGTwin VAE checkpoint、输入/latent、冻结方式和分类器桥接契约 | 新配置保留 checkpoint 双组件、raw mV、1024点、lead reorder、latent scale 和严格键数量 |
| `configs/train/lhat.yaml` | `KEEP-MANUAL` | M20 exact-label/non-self latent hull、标准化、λ0.6、ε12、10步 BCE 困难搜索，以及 `[0.25,0.5,0.75,1]` 预翻转最大损失收缩网格和端点残差修正 | 新配置保留 train-only standardizer、`include_anchor=false`、100 Hz 信息瓶颈、clean-correct margin 50% 保留、无 heldout feedback 和 raw/contract 双层诊断 |
| `configs/train/augmix.yaml` | `KEEP-MANUAL` | Stage-1 两条独立 depth2/3 腐蚀链在 500 Hz 算子域执行并回到 canonical100，以 Dirichlet(0.5,0.5) 混链、Beta(0.5,0.5) 混 clean；同文件继续承载非主线 latent-threechain 配置 | 新配置保留顺序隔离 RNG、pre-zscore raw mV、两链公式/温度0.5、非原地输入和配置束内单一算子来源 |
| `configs/train/PN2021.yaml` | `KEEP-MANUAL` | 冻结的前瞻复现配方：EffNet Stage-1 1024步 + E23/T30，ECGFounder Stage-1 256步 + E30/T30；统一完整 K500、resident loader、最后 checkpoint、TensorBoard 和无 heldout 选模 | 新配置保留同源 checkpoint、完整 K500、两阶段超参、matched-base 单步预算、ref-exclusion、v7 映射、drop-all-zero 主口径和100 Hz bottleneck |
| `configs/train/PN2021_fixed20.yaml` | `KEEP-MANUAL` | family-balanced Direct fixed20 唯一配置；定义400/100调参与完整K500 refit、EffNet 30轮和ECGFounder 20轮搜索上限及运行参数 | 新配置保留固定20组合、clean/corruption 各50% loss、每 base batch 一次 optimizer step、source registry、resident loader 和不物化训练腐蚀缓存 |
| `configs/train/methods/a0_clean_v1.yaml` | `KEEP-MANUAL` | 锁定 clean K500 监督适配的 typed graph、objective 和零 VAE 资源契约 | 新 profile 保留 A0 身份、单 clean BCE、完整 base exposure 和一个 outer optimizer step |
| `configs/train/methods/a3c_depth23_v1.yaml` | `KEEP-MANUAL` | 锁定在线 depth2/3 canonical 腐蚀基线的 typed graph、双 BCE 和 500 Hz 算子域 | 新 profile 保留 A3c 身份、每 origin 单一腐蚀 view、A0 相同 base budget 且不物化 20 倍数据集 |
| `configs/train/methods/direct_depth23_fixed20.yaml` | `KEEP-MANUAL` | family-balanced Direct 基线：每条记录一次 clean、10个 depth2 和10个 depth3 canonical composition，所有损失累计后单次更新 | 新 profile 保留显式0..19顺序、`0.5 clean + 0.5 mean(corruption)`、500 Hz Torch 在线生成、每 base batch 一次 optimizer step 且不物化训练腐蚀缓存 |
| `configs/train/methods/direct_depth23_fixed20_raw_aux.yaml` | `KEEP-MANUAL` | matched 22-exposure 原始 clean 辅助分支控制：`0.45 clean + 0.45 fixed20 + 0.10 raw auxiliary`，匹配额外 forward、BN 和优化器步预算 | 新 profile 保留与 Direct 公共 corruption RNG 身份/namespace、单次 outer step、raw auxiliary 独立 forward 及禁止 heldout feedback |
| `configs/train/methods/direct_depth23_fixed20_vae_reconstruction_aux.yaml` | `KEEP-MANUAL` | matched 22-exposure VAE reconstruction-only 控制；使用 deterministic posterior mean 拆分 raw→reconstruction 位移 | 新 profile 保留同预算/同随机 fixed20、冻结 encoder/decoder、1000↔1024/导联桥、质量拒绝 clean-only 和无 latent-hull 搜索 |
| `configs/train/methods/direct_depth23_fixed20_lhat_aux.yaml` | `KEEP-MANUAL` | matched 22-exposure LHAT hard-endpoint 候选；在相同 normalized auxiliary 路径中隔离 reconstruction→hard 位移 | 新 profile 保留 exact-label train-only pool、公共 fixed20 随机轨迹、0.10 auxiliary 总权重、accepted-mask 诊断、单次 outer step 和无 heldout feedback |
| `configs/train/methods/augmix_simclr_lhat.yaml` | `KEEP-MANUAL` | 唯一主线方法身份：Stage-1 两链 AugMix-SimCLR + 等权 source-logit anchor；Stage-2 clean/四轮换腐蚀 family balance + contracted VAE-LHAT BCE | 新 profile 保留 K500-only、无 replay/VICReg/JSD/PCGrad、五轮覆盖20组合、一次 outer step、辅助权重2、BN/RNG snapshot-restore、坏波形 clean-only 和无 heldout feedback |
| `configs/train/methods/exp_lhat_replay_pool_v1.yaml` | `KEEP-MANUAL` | LHAT replay-pool 组合方案的可审计、暂不可执行 profile | 实现 replay 生命周期、预算和 stale-view 契约前必须保持 `contracts.executable=false` |
| `configs/train/methods/exp_lhat_as_sixth_branch_v1.yaml` | `KEEP-MANUAL` | 把 LHAT 作为第六分支的可审计、暂不可执行 profile | 完成六分支采样和公平 exposure 定义前必须保持 `contracts.executable=false` |
| `configs/train/methods/exp_paired_augmix_latent_bridge_v1.yaml` | `KEEP-MANUAL` | 为保持白名单稳定文件名而原位覆盖为锁定的可执行 latent-threechain candidate：同一 clean 生成两个独立三腐蚀链 latent 混合 view，以 clean BCE + 每 view `0.75` BCE + `3×` Bernoulli JSD 训练；ECGFounder 由 K500 内部 frozen validation400 选 E19 | 新 profile 保留双 view 独立可复放 RNG、encoder/decoder 无 latent-pool、每 view 三条 depth2/3 链、质量交集、BN 权重 `0.5/0.25/0.25`、无 heldout feedback；不得把它表述成 LHAT/adversarial search、原 AugMix 精确复现或把 LR 收益算成方法收益 |
| `configs/train/methods/exp_augmix_guided_latent_simplex_v1.yaml` | `KEEP-MANUAL` | AugMix 难度引导 latent simplex 搜索的可审计、暂不可执行 profile | 完成 guidance 泄漏边界、攻击预算和标签安全契约前必须保持 `contracts.executable=false` |
| `core/__init__.py` | `KEEP-MANUAL` | 通用监督训练、在线方法训练和 typed method graph 的唯一公共导出面 | 新核心包接管相同接口并完成训练调用迁移；禁止重新导出固定方法臂枚举 |
| `core/corruption.py` | `KEEP-MANUAL` | 共享 GPU canonical 腐蚀内核；把两条链合并成 `2B`，固定在 500 Hz 调用五算子并返回 100 Hz raw-mV 波形及逐样本 provenance | 新实现保留线性 `100→500→100`、五算子各一次批量调用、depth2/3 组合掩码、无逐组合 CPU 分支和有限值诊断 |
| `core/lhat.py` | `KEEP-MANUAL` | train-only latent standardizer、exact-label non-self M20 hull 优化、100 Hz canonical 可微解码与两模型攻击桥；攻击后一次批量解码收缩路径、端点残差修正并选择最高 BCE 的标签边界保护点 | 新实现保留只优化 batch-local hull 权重、不污染分类器/decoder 梯度、ε12/10步 raw 搜索、clean-correct margin 保护、finite/std/20mV gate、clean fallback、ASR/接受率/t 诊断和紧凑 D2H |
| `core/augmix.py` | `KEEP-MANUAL` | 实现冻结 Stage-1 两链强视图；两条链按锁定 RNG 顺序独立生成后做逐样本 Dirichlet 与 Beta 混合，并继续承载审计中的 latent-threechain 生成函数 | 新实现保留 raw100 输入、500 Hz 算子域、canonical100输出、显式 generator、非原地语义、可复算权重和不把两条链合并成会改变 RNG 身份的 `2B` 调用 |
| `core/latent_pool.py` | `KEEP-MANUAL` | 从 raw 100 Hz K500 建立 frozen VAE latent pool；schema v2 预计算 stable exact-label neighbor table，缓存 eligible hash tuple/set，并把邻居表 SHA256 纳入身份 | 新实现保留 encoder/cache/label/latent/eligibility/standardizer 身份、候选 distinct/non-self、与原 stable search 逐元素一致及不可用样本显式清单 |
| `core/methods/__init__.py` | `KEEP-MANUAL` | typed method graph 的稳定公共导出面 | 新包保留 compiler/executor/contracts/runtime 的显式 API，不暴露动态 import |
| `core/methods/contracts.py` | `KEEP-MANUAL` | 定义 canonical raw-mV waveform、latent、pair、candidate、bundle、objective 和 provenance 类型契约 | 新实现保留 `(B,1000,12)`/`(B,4,128)`、Super5、valid-mask、origin 对齐和有限值校验 |
| `core/methods/registry.py` | `KEEP-MANUAL` | 严格加载方法 YAML，以显式 node 白名单编译 DAG、类型边、objective、资源需求和 profile 身份；注册 `vae_lhat_attack_then_contract_view` 且禁止 YAML 动态 import | 新编译器保留 schema/环/类型/引用检查、节点端口及资源名/type/必需键的代码白名单、VAE资源需求和未知资源 fail-closed |
| `core/methods/executor.py` | `KEEP-MANUAL` | 按编译拓扑执行 typed nodes，并从 comparison identity、namespace、node、stream 及 execution identity 派生 Torch RNG stream | 新 executor 保留旧 profile 默认隔离、显式 matched profiles 公共随机数、不同 step 分流、资源闸门、audit-only 拒绝、RNG 诊断和 named output bundle |
| `core/methods/runtime.py` | `KEEP-MANUAL` | 将 typed graph 接到 canonical corruption、raw LHAT 和 attack-then-contract；只为候选充足且通过收缩/QC 的记录暴露 VAE view，其余 clean-only | 新 runtime 保留 K500 hash/标签绑定、exact-label pool、500 Hz 算子域、contract accepted mask、拒绝原因、raw/contract 诊断、显式 RNG provenance 和无 agent_workspace import |
| `core/methods/nodes/__init__.py` | `KEEP-MANUAL` | 汇总全部代码所有的显式 node callable | 新节点集合保留静态导出且不得自动发现任意模块 |
| `core/methods/nodes/selectors.py` | `KEEP-MANUAL` | clean source、严格配对和候选选择 node 适配器 | 新实现保留 origin/label 对齐与显式 runner adapter 边界 |
| `core/methods/nodes/codecs.py` | `KEEP-MANUAL` | VAE encode/decode 的显式 node 适配器，不复制 ECGTwin 实现 | 新实现保留 caller-owned codec 和 typed 输入输出边界 |
| `core/methods/nodes/waveform_ops.py` | `KEEP-MANUAL` | canonical waveform corruption node 适配器 | 新实现必须继续委托单一共享腐蚀内核，不复制五算子公式 |
| `core/methods/nodes/latent_ops.py` | `KEEP-MANUAL` | LHAT 与 paired latent bridge node 适配器 | 新实现保留 runner-owned 资源及暂未实现节点的 audit-only 隔离 |
| `core/methods/nodes/mixers.py` | `KEEP-MANUAL` | AugMix/view mixing node 适配器 | 新实现保留显式 adapter 调用，不能在 registry 中隐藏训练 objective |
| `core/methods/nodes/buffers.py` | `KEEP-MANUAL` | replay read/write node 适配器 | replay profile 可执行前必须定义 run-scoped 生命周期、容量、采样和证据身份 |
| `core/online_trainer.py` | `KEEP-MANUAL` | 在原 typed trainer 内执行两阶段主线：冻结分类头的两链 AugMix-SimCLR、source/stage2 logit anchor、五轮 rotating4、VAE 辅助直接相加及一次 outer update；保存 Stage-1 checkpoint、两阶段 TensorBoard 和资源身份 | 新实现保留 pre-zscore raw mV、K500 teacher cache、两个骨干 feature API、单步 family balance、辅助 BN/RNG snapshot-restore、optimizer/view计数、最终 checkpoint、无 heldout 选模和完整配置/seed闭包 |
| `core/train_PN2021.py` | `KEEP-MANUAL` | 通过 `data_runtime` 建立 raw 100 Hz K500 或 train400 loader，并按 method requirements 分别路由 latent pool、runtime frozen encoder 和 decoder | 新适配层保留四逻辑中心、CPSC/Extra 合并、partition身份、resident强制workers=0/persistent=false、无pool VAE方法资源隔离及方法不进入DataLoader seed |

### A6. 通用监督训练入口

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `configs/train/PTBXL.yaml` | `KEEP-MANUAL` | 通用训练器的 PTB-XL source profile；记录 folds 1-8/9/10、AdamW、BCE、AMP、fold9 macro-AUPRC 选模、TensorBoard 观察配置，以及 EffNet/ECGFounder boot 默认值和独立输出目录 | 新配置保留相同数据身份、模型输入、选模边界、模型初始化、配置束引用和 run-scoped TensorBoard 路径 |
| `configs/train/PN2021_direct_tune.yaml` | `KEEP-MANUAL` | family-balanced Direct 内部 train400/validation100、冻结 clean+20 views 的 packed-resident 执行、逐 epoch raw-logit 留证、四中心 pooled clean/robust AUPRC、clean -1pp floor、最早 tie 及 full-K500 refit 契约 | 新配置保留 source best.pt SHA、raw100→linear model-domain→zscore-after-resample、validation 禁止训练/latent pool、view-major clean+20 顺序、逐 composition pooled400 后求均值、0.5/0.5 score、400独立ECG预算和scheduler horizon |
| `configs/train/PN2021_matched_raw_aux_tune.yaml` | `KEEP-MANUAL` | raw auxiliary 因果控制的 train400/frozen-validation100 配置；复用 Direct H30、packed validation、v7映射和 source registry | 新配置保留相同数据/优化器/随机身份、仅 method profile 与 normalized auxiliary 契约变化、无 heldout 选模及 refit 前置 pooled selection |
| `configs/train/PN2021_matched_vae_reconstruction_aux_tune.yaml` | `KEEP-MANUAL` | VAE reconstruction-only 因果控制的 train400/frozen-validation100 配置 | 新配置保留 raw 控制相同预算、冻结 deterministic codec、packed validation、source SHA 和无 latent-hull 搜索 |
| `configs/train/PN2021_matched_lhat_aux_tune.yaml` | `KEEP-MANUAL` | LHAT hard-endpoint 因果候选的 train400/frozen-validation100 配置 | 新配置保留前两臂相同 fixed20 随机轨迹/优化器预算、train-only exact-label latent pool、完整攻击诊断和 pooled selection 前禁止 refit |
| `core/supervised_trainer.py` | `KEEP-MANUAL` | 接受调用方构建的 Torch model 与 DataLoader，完成多标签监督训练；统一 raw-logit AUROC/average-precision；支持 H2D 后的可选模型输入适配、逐 epoch raw prediction 留证、选模、严格 checkpoint 和 TensorBoard | 新训练器保留默认无适配路径的行为，以及 model/DataLoader/seed/config/checkpoint/override/logging/input-adapter 身份、单中心显式 undefined policy、单调 optimizer step，并通过 PTB-XL source、PN2021 pooled tuning 与 last-epoch 微调契约测试 |
| `core/train_PTBXL.py` | `KEEP-MANUAL` | 根据任意模型 `ModelSpec` 自动选择 100/500 Hz，使用 `data_runtime` 建立 folds 1-8/9；仅在 YAML 显式最终测试时才构建 fold10 loader，再委托通用 trainer并关闭自有 loader | 新 PTB-XL 适配器保留 split/cache/采样率/归一化/loader 参数身份、未授权时不构造fold10，并继续只依赖白名单数据层和通用训练器 |
| `core/pn2021_tuning.py` | `KEEP-MANUAL` | 通用 train400/frozen-validation100 适配层：动态校验 executable method/protocol，把 clean+20 个冻结 PN2021-C view 打包成 view-major resident bank，并按 requirements 路由可选 train400 latent pool、runtime encoder 与 decoder；统一校验 refit 的 full-FT、pos-weight、seed group/replicate/文件 SHA | 新适配器保留 Direct 默认身份不漂移、source SHA、full-FT、packed bank 的21-view/hash/label/order及 source-loader release、pool 方法另用独立 shuffle=false train400 loader、train400/validation100 哈希互斥、validation只读、禁止 replay/heldout feedback及完整缓存/composition/hash证据 |
| `boot_scripts/__init__.py` | `KEEP-MANUAL` | PTB-XL 薄启动脚本包边界 | 新启动包完整接管两个模型入口且不承载训练业务逻辑 |
| `boot_scripts/train_ptbxl_effnet.py` | `KEEP-MANUAL` | 从 PTBXL YAML 构建 EfficientNet1DV2、合并显式 CLI 覆盖并调用 `train_ptbxl`；支持无副作用 dry-run | 新入口保留 100 Hz 模型契约、配置束、随机种子和输出身份 |
| `boot_scripts/train_ptbxl_ecgfounder.py` | `KEEP-MANUAL` | 从 PTBXL YAML 加载官方 ECGFounder backbone、full/head scope，合并显式 CLI 覆盖并调用 `train_ptbxl`；支持无副作用 dry-run | 新入口保留 500 Hz bottleneck、官方 checkpoint、训练范围、配置束、随机种子和输出身份 |
| `boot_scripts/train_pn2021.py` | `KEEP-MANUAL` | 从 PN2021 外层协议、显式 method profile/source checkpoint/model/center 启动匹配 K500 训练；按 latent-pool/encoder/decoder requirements 精确加载 VAE 组件，支持无副作用 dry-run | 新入口保留源 checkpoint 强制输入、完整 pooled-selection rule、selection 时 scientific online-config 快照、同组 seed/full-FT/pos-weight、方法 profile 隔离、A0/A3c 零 VAE、主线严格 VAE 加载和全部显式覆盖记录 |
| `boot_scripts/tune_pn2021_direct.py` | `KEEP-MANUAL` | 保留稳定文件名的通用 train400 + frozen validation100 启动入口；Direct 保持原 action/seed 身份，其他 executable 方法按 pool/runtime-encoder/decoder requirements 精确加载一次 VAE 并复用同一适配层 | 新入口保留 full-FT、model/center 唯一输出、两模型独立超参、source SHA、dry-run不读权重、主线 train400-only pool、Direct历史身份和 pooled-selection pending 身份 |
| `boot_scripts/select_pn2021_direct.py` | `KEEP-MANUAL` | 保留稳定文件名的通用只读 pooled-selection 入口；发现四中心逐epoch clean+20 prediction artifacts，Direct 保持原 action，其他方法使用通用 action且不加载波形、模型或GPU | 新入口保留四中心精确覆盖、clean拼400、每composition拼400、0.5/0.5 score、clean floor、方法身份不误标、输出不覆盖和无副作用dry-run |
| `boot_scripts/refit_pn2021_direct.py` | `KEEP-MANUAL` | 消费已完成受管 family-balanced selection，在完整K500上按全局E* refit fixed20 Direct；严格校验selection/source/配置闭包/seed SHA并支持无副作用dry-run | 新入口保留protocol/method/四中心身份、完整selection rule、method及online/tuning配置SHA、E*与原scheduler horizon传播、full-FT/pos-weight不漂移、锁定source checkpoint、无fixed-cycle兼容参数和refit contract |

### A7. 训练观察与 ECG 可视化

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `.gitignore` | `KEEP-MANUAL` | 排除 TensorBoard events、run-scoped ECG probe 和其他本地产生的大型训练观察物 | 新规则继续阻止事件、图片、缓存和运行输出误入 Git，且不屏蔽受管源码/配置 |
| `util/tensorboard_logging.py` | `KEEP-MANUAL` | 异步记录 objective/攻击/吞吐；支持 `epoch_policy=final_only`、`tensorboard_image=false`，manifest 在 flush/close 批量原子写，同时保留 PNG/NPY probe | 新监控层保留禁用零副作用、run-scoped 路径、有限数值校验、单调 step、任意方法 view、raw mV/采样率/导联身份及不消费训练 RNG 的契约 |
| `util/visualize_ecg.py` | `KEEP-MANUAL` | 移植 ECGTwin 作者 `ecg_plot.plot()` 可视化路径，将显式 time-channel/channel-time 波形转换为 lead-time ECG 纸格图并支持 PNG 输出 | 新绘图层保留作者调用方式、实际采样率、显式 ECGTwin/项目导联顺序、非原地输入和 figure 关闭契约 |

### A8. 正式评估、统一启动与运行留证

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `configs/active_scripts.yaml` | `KEEP-MANUAL` | 区分历史 trusted mainline 与当前前瞻主线，并登记白名单 launcher/config/trainer、CPU 验证状态和待完成的 GPU 重复 | 新索引必须保留 trusted/development 分层，禁止仅因代码迁移把开发候选升级为论文证据 |
| `configs/active_evidence_registry.yaml` | `KEEP-MANUAL` | 记录精简主线的 K500-only 数据边界、冻结拓扑、开发选择审计、白名单实现路径和论文晋级条件 | 新注册表保留 heldout-tuned/single-seed 边界、无 outside-K500 model access、v7/drop-all-zero 契约及多随机重复要求 |
| `configs/eval/PN2021.yaml` | `KEEP-MANUAL` | PN2021正式评估唯一配置；单进程顺序session复用clean/PN2021-C mmap和selection，覆盖四中心clean+20 views | 新配置保留v7映射、K500 ref-exclusion、view顺序、raw100输入、四种canonical metric view及session/loader所有权 |
| `util/evaluation/__init__.py` | `KEEP-MANUAL` | 手工重构正式评估公共导出面 | 新包接管相同 metrics/PN2021 API 且调用方完成迁移 |
| `util/evaluation/metrics.py` | `KEEP-MANUAL` | 直接从 raw logits 计算 Super5 AUROC 与 sklearn Average Precision；显式 strict/skip-undefined、per-class/macro、kept/drop、中心/view 等权聚合 | 新实现禁止 sigmoid 饱和改变排序，保留 AP 非梯形 PR-AUC 定义、classes-used/正负计数、metric view、record/hash-label、depth2/3/23 和 equal-weight 语义 |
| `util/evaluation/direct_baseline_selection.py` | `KEEP-MANUAL` | 保留稳定文件名的通用四中心 pooled selector；动态校验 protocol/method，clean拼400，每composition拼400后算20个macro-AUPRC均值，以0.5/0.5 score和clean -1pp floor选择共享E* | 新实现保留Direct历史 artifact type、其他方法通用 artifact type、strict五类、exact tie选最早、禁止flatten 8000 views、中断grid/身份漂移/不同预算混用、完整输入SHA和heldout未参与证据 |
| `util/evaluation/pn2021.py` | `KEEP-MANUAL` | 固定评估raw100 clean+20-view PN2021-C；统一复用一个sequential session，不保留旧worker/runtime profile回退 | 新实现保留checkpoint SHA、禁止PTB-XL shard、拒绝500 Hz cache、clean/corrupted同记录、显式view、record/hash-label顺序、loader/session关闭及完整输入链证据 |
| `boot_scripts/evaluate_pn2021.py` | `KEEP-MANUAL` | 从评估 YAML 和显式 checkpoint 构建模型并调用 canonical100 v2 正式 PN2021 评估；支持只读 dry-run | 新入口保留严格 checkpoint 加载、模型spec、配置束、输入适配链、运行覆盖和无隐式选模 |
| `util/run_record.py` | `KEEP-MANUAL` | 记录命令、配置闭包、Git/dirty SHA、环境、typed method/数据/选择证据，识别 Direct per-center pending 与 Direct/latent 通用 pooled E* 产物，并为所有运行文件建立 SHA256 索引；refit 可复用其 managed member、配置闭包及按SHA解析的不可变配置快照校验 | 新实现保留禁止 worktree 输出、原子 JSON、method ID/scientific arm、显式 last/validation/pooled selection、配置快照、完整索引及篡改检测 |
| `boot_scripts/run_experiment.py` | `KEEP-MANUAL` | 单一 experiment YAML + config-root 的白名单 launcher，委托 PTB-XL、PN2021训练、Direct tuning/pooled selection/full-K500 refit 及评估入口；dry-run零副作用 | 新入口保留受限entrypoint/flags、配置闭包、外部唯一run dir、精确argv和RunRecorder生命周期 |
| `configs/baselines/ptbxl_source_v1.yaml` | `KEEP-MANUAL` | 锁定 EfficientNet1DV2/ECGFounder 的 PTB-XL source `best.pt`、SHA256、fold9 选模及 fold10 指标 | 新注册表完整接管相同 checkpoint 身份与禁止使用 `last.pt` 的下游契约 |
| `docs/baselines/ptbxl_source_v1.md` | `KEEP-MANUAL` | PTB-XL source baseline v1 的人类可读锁定日志、复现实验命令、原始证据路径及 10-epoch 否决结论 | 新决策日志完整保留相同配置、权重、SHA256、指标和下游使用规则 |
| `configs/baselines/pn2021_direct_v1.yaml` | `KEEP-MANUAL` | family-balanced Direct+fixed20 锁定注册表；原地保留same-backbone clean-floor、pooled选择、full-K500 证据，并登记 ECGFounder matched-LR/latent-threechain、EfficientNet L36/L37 及 ECGFounder L34/L35 两随机种子机制证据和 source-floor 否决 | 新注册表完整接管clean-floor、选择身份、配置闭包、全部 checkpoint/eval 证据、LR/VAE归因拆分、heldout-tuned状态及禁止将 source-floor 失败的开发锁冒充论文最终结果的契约 |
| `docs/baselines/pn2021_direct_v1.md` | `KEEP-MANUAL` | family-balanced Direct+fixed20 人类可读锁定日志，并记录 matched LR `3e-5`、latent-threechain、L37 及 ECGFounder L35 两随机种子 VAE-LHAT 机制结果；文件名仅为白名单稳定性原地保留 | 新决策日志完整接管协议、超参、收敛结论、权重身份、四中心指标、归因边界、source-floor限制和TensorBoard入口 |
| `agent_workspace/performance_summary_20260727/aligned_targetonly_augmix_lhat_lock_20260804.json` | `KEEP-MANUAL` | 固定 URL 配套的机器可读精简主线开发锁；记录两骨干公共 AugMix→attack-then-contract VAE-LHAT 契约、候选/权重/eval SHA、drop-all-zero 指标、严格无 VAE 消融和单 seed 边界 | 后续注册表接管时必须保留 lock/supersedes 身份、K500-only/ref-exclusion/v7 映射、公共 M20/λ0.6/ε12/10步/margin0.5/BCE 契约、骨干特定优化差异及 heldout-tuned 禁止论文终局声明 |
| `agent_workspace/performance_summary_20260727/aligned_targetonly_augmix_lhat_report_20260804.html` | `KEEP-MANUAL` | 在 `127.0.0.1:9092` 固定入口展示上述精简主线流程、绝对指标、Direct/source 对比、严格 VAE 消融及攻击—回缩诊断 | 新正式报告接管前保持 URL 和机器 lock 对齐；禁止把单 seed 开发指标、raw ASR 或完整方法相对 Direct 的收益误写成 VAE 单独贡献 |
| `configs/experiments/manual_refactor_ptbxl_effnet.yaml` | `KEEP-MANUAL` | 统一 launcher 的 PTB-XL EfficientNet source 示例 | 新示例或注册表完整替代并可 dry-run |
| `configs/experiments/manual_refactor_ptbxl_ecgfounder.yaml` | `KEEP-MANUAL` | 统一 launcher 的 PTB-XL ECGFounder full-FT source 示例 | 新示例或注册表完整替代并可 dry-run |
| `configs/experiments/manual_refactor_ptbxl_ecgfounder_e10_probe.yaml` | `KEEP-MANUAL` | ECGFounder 锁定 source 配方仅延长到 10 epoch 的单变量敏感性实验 | 结论进入 source baseline 注册表或明确判定无需保留该敏感性证据 |
| `configs/experiments/manual_refactor_pn2021_effnet_direct_ningbo.yaml` | `KEEP-MANUAL` | 统一 launcher 的 Ningbo A0 clean-K500 示例 | 新示例或注册表保留显式源 checkpoint/model/method-config/center |
| `configs/experiments/manual_refactor_pn2021_effnet_a3c_ningbo.yaml` | `KEEP-MANUAL` | 统一 launcher 的 Ningbo A3c 在线 depth2/3 腐蚀示例 | 新示例或注册表保留显式源 checkpoint/model/method-config/center 和配置闭包 |
| `configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_ningbo.yaml` | `KEEP-MANUAL` | 统一 launcher 的 Ningbo 前瞻复现示例：两链 AugMix-SimCLR → rotating4 + contracted VAE-LHAT | 新示例或注册表保留显式源 checkpoint/model/method-config/center、VAE 资源、两阶段超参和配置闭包 |
| `configs/experiments/manual_refactor_pn2021_eval_effnet.yaml` | `KEEP-MANUAL` | 统一 launcher 的固定 EfficientNet checkpoint 正式评估示例 | 新示例或注册表保留固定 checkpoint 和 eval YAML |
| `configs/experiments/manual_refactor_pn2021_effnet_direct_tune_ningbo.yaml` | `KEEP-MANUAL` | EfficientNet Ningbo train400 + frozen clean/20-view validation100 受管入口 | 新入口保留模型、中心、family-balanced profile 和唯一输出身份 |
| `configs/experiments/manual_refactor_pn2021_effnet_matched_raw_aux_e2_ningbo.yaml` | `KEEP-MANUAL` | EfficientNet Ningbo E2 raw auxiliary 实现/预算控制受管入口 | 新入口保留相同 PTB-XL source、H30 horizon、K400/K100、公共 RNG 身份、唯一外部输出目录和无副作用 dry-run |
| `configs/experiments/manual_refactor_pn2021_effnet_matched_vae_reconstruction_aux_e2_ningbo.yaml` | `KEEP-MANUAL` | EfficientNet Ningbo E2 deterministic VAE reconstruction-only 受管入口 | 同上，并保留冻结 encoder/decoder 资源闭包和 reconstruction-only 科学身份 |
| `configs/experiments/manual_refactor_pn2021_effnet_matched_lhat_aux_e2_ningbo.yaml` | `KEEP-MANUAL` | EfficientNet Ningbo E2 LHAT hard-endpoint 受管入口 | 同上，并保留 train-only latent pool、LHAT配置闭包和 accepted-mask/trace 诊断身份 |
| `configs/experiments/manual_refactor_pn2021_effnet_direct_tune_chapman_shaoxing.yaml` | `KEEP-MANUAL` | EfficientNet Chapman-Shaoxing family-balanced tuning 入口 | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_direct_tune_cpsc_2018.yaml` | `KEEP-MANUAL` | EfficientNet CPSC+Extra family-balanced tuning 入口 | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_direct_tune_georgia.yaml` | `KEEP-MANUAL` | EfficientNet Georgia family-balanced tuning 入口 | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_direct_select.yaml` | `KEEP-MANUAL` | 汇总 EfficientNet 四中心 clean+20 validation 并按0.5/0.5与clean floor选择全局E* | 新入口保留四个受管run路径、冻结view身份和独立selection run |
| `configs/experiments/manual_refactor_pn2021_effnet_fixed20_refit_ningbo.yaml` | `KEEP-MANUAL` | 按全局E*在Ningbo完整K500 refit family-balanced EfficientNet | 新入口保留受管selection、原scheduler horizon、锁定source、在线Torch腐蚀和唯一输出身份 |
| `configs/experiments/manual_refactor_pn2021_effnet_fixed20_refit_chapman_shaoxing.yaml` | `KEEP-MANUAL` | 按全局E*在Chapman-Shaoxing完整K500 refit family-balanced EfficientNet | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_fixed20_refit_cpsc_2018.yaml` | `KEEP-MANUAL` | 按全局E*在CPSC+Extra完整K500 refit family-balanced EfficientNet | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_fixed20_refit_georgia.yaml` | `KEEP-MANUAL` | 按全局E*在Georgia完整K500 refit family-balanced EfficientNet | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_direct_tune_ningbo.yaml` | `KEEP-MANUAL` | ECGFounder Ningbo train400 + frozen clean/20-view validation100 受管入口 | 新入口保留模型、中心、family-balanced profile 和唯一输出身份 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_direct_tune_chapman_shaoxing.yaml` | `KEEP-MANUAL` | ECGFounder Chapman-Shaoxing family-balanced tuning 入口 | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_direct_tune_cpsc_2018.yaml` | `KEEP-MANUAL` | ECGFounder CPSC+Extra family-balanced tuning 入口 | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_direct_tune_georgia.yaml` | `KEEP-MANUAL` | ECGFounder Georgia family-balanced tuning 入口 | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_direct_select.yaml` | `KEEP-MANUAL` | 汇总 ECGFounder 四中心 clean+20 validation 并按0.5/0.5与clean floor选择全局E* | 新入口保留四个受管run路径、冻结view身份和独立selection run |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_refit_ningbo.yaml` | `KEEP-MANUAL` | 按全局E*在Ningbo完整K500 refit family-balanced ECGFounder | 新入口保留受管selection、原scheduler horizon、锁定source、在线Torch腐蚀和唯一输出身份 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_refit_chapman_shaoxing.yaml` | `KEEP-MANUAL` | 按全局E*在Chapman-Shaoxing完整K500 refit family-balanced ECGFounder | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_refit_cpsc_2018.yaml` | `KEEP-MANUAL` | 按全局E*在CPSC+Extra完整K500 refit family-balanced ECGFounder | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_refit_georgia.yaml` | `KEEP-MANUAL` | 按全局E*在Georgia完整K500 refit family-balanced ECGFounder | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_clean_eval_ningbo.yaml` | `KEEP-MANUAL` | 固定 EfficientNet clean refit checkpoint，通过 v2 canonical100 协议仅评估 ref-excluded Ningbo clean 与 PN2021-C 20 views | 新入口保留显式 target center、checkpoint 路径、共享输入适配链、kept/drop 与 depth2/3/23 指标和唯一受管输出 |
| `configs/experiments/manual_refactor_pn2021_effnet_clean_eval_chapman_shaoxing.yaml` | `KEEP-MANUAL` | 固定 EfficientNet clean refit checkpoint，仅评估 ref-excluded Chapman-Shaoxing clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_clean_eval_cpsc_2018.yaml` | `KEEP-MANUAL` | 固定 EfficientNet clean refit checkpoint，仅评估 ref-excluded CPSC+Extra clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_clean_eval_georgia.yaml` | `KEEP-MANUAL` | 固定 EfficientNet clean refit checkpoint，仅评估 ref-excluded Georgia clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_fixed20_eval_ningbo.yaml` | `KEEP-MANUAL` | 固定 EfficientNet fixed20 refit checkpoint，仅评估 ref-excluded Ningbo clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_fixed20_eval_chapman_shaoxing.yaml` | `KEEP-MANUAL` | 固定 EfficientNet fixed20 refit checkpoint，仅评估 ref-excluded Chapman-Shaoxing clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_fixed20_eval_cpsc_2018.yaml` | `KEEP-MANUAL` | 固定 EfficientNet fixed20 refit checkpoint，仅评估 ref-excluded CPSC+Extra clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_fixed20_eval_georgia.yaml` | `KEEP-MANUAL` | 固定 EfficientNet fixed20 refit checkpoint，仅评估 ref-excluded Georgia clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_clean_eval_ningbo.yaml` | `KEEP-MANUAL` | 固定 ECGFounder clean refit checkpoint，通过 v2 canonical100→linear500 协议仅评估 ref-excluded Ningbo clean 与 PN2021-C 20 views | 新入口保留显式 target center、checkpoint 路径、共享输入适配链、kept/drop 与 depth2/3/23 指标和唯一受管输出 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_clean_eval_chapman_shaoxing.yaml` | `KEEP-MANUAL` | 固定 ECGFounder clean refit checkpoint，仅评估 ref-excluded Chapman-Shaoxing clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_clean_eval_cpsc_2018.yaml` | `KEEP-MANUAL` | 固定 ECGFounder clean refit checkpoint，仅评估 ref-excluded CPSC+Extra clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_clean_eval_georgia.yaml` | `KEEP-MANUAL` | 固定 ECGFounder clean refit checkpoint，仅评估 ref-excluded Georgia clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_eval_ningbo.yaml` | `KEEP-MANUAL` | 固定 ECGFounder fixed20 refit checkpoint，仅评估 ref-excluded Ningbo clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_eval_chapman_shaoxing.yaml` | `KEEP-MANUAL` | 固定 ECGFounder fixed20 refit checkpoint，仅评估 ref-excluded Chapman-Shaoxing clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_eval_cpsc_2018.yaml` | `KEEP-MANUAL` | 固定 ECGFounder fixed20 refit checkpoint，仅评估 ref-excluded CPSC+Extra clean 与 PN2021-C 20 views | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_eval_georgia.yaml` | `KEEP-MANUAL` | 固定 ECGFounder fixed20 refit checkpoint，仅评估 ref-excluded Georgia clean 与 PN2021-C 20 views | 同上 |

### A9. 最终基础契约测试

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `util/tests/__init__.py` | `KEEP-MANUAL` | 保留测试包边界 | 测试布局整体迁移时同步迁移 |
| `util/tests/test_augmentations.py` | `KEEP-MANUAL` | 使用固定上游 revision 与 seeded golden summary 验证 NumPy 五算子，不再运行时导入旧 `methods` | 新算子实现接管同一 revision、公式、随机和非原地契约 |
| `util/tests/test_torch_augmentations.py` | `KEEP-MANUAL` | 验证 Torch 五算子的接口、CPU 数值对齐、批量与设备契约 | 新设备算子接管同一接口与批量数值契约 |
| `util/tests/test_pn2021_corruptions.py` | `KEEP-MANUAL` | 直接验证白名单 profile、20 个 depth2/3 组合、确定性复合、修正后的 Baseline Shift 与 RLM 生存导联 | 新 PN2021-C 内核接管同一 profile、组合及随机身份 |
| `util/tests/test_data_contracts.py` | `KEEP-MANUAL` | 直接验证白名单缓存、模型输入、100/500 Hz 线性插值、Super5/导联顺序及主线 YAML 闭包 | 新数据/模型边界接管相同布局、插值顺序和 config closure |
| `util/tests/test_labels_super5.py` | `KEEP-MANUAL` | 直接验证白名单 PN2021 Super5 v7 映射、hash、NORM 抑制和 PTB-XL diagnostic class 转换 | 新标签层接管相同 mapping identity 和逐代码 golden policy |

## B. 当前不属于手工主线的文件

### B1. 非核心支持与待退出文件

这些文件不属于 A 区的手工重构保护白名单。它们当前仍有验证、兼容或证据
作用，因此只是允许后续退出，不是本次直接删除清单。

| 文件 | 状态 | 当前作用 | 何时可以删除或还原 |
|---|---|---|---|
| `methods/augmix/ecg_ops.py` | `LEGACY-BRIDGE` | 旧 PN2021-C 调用入口以及临时 profile 兼容 | 调用方全部迁移到 `util/augmentations/`，历史回放测试通过 |
| `util/tests/test_pn2021_metadata.py` | `SUPPORT` | 验证独立 PN2021 表头解析语义及预处理入口不再导入旧数据层 | 表头解析器迁移时同步保留相同 golden contract 与 import 闭包检查 |
| `util/tests/test_preprocess_nonfinite.py` | `SUPPORT` | 验证 PTB-XL/PN2021 非有限值插值修复、边界填补与严重异常剔除契约 | 质量控制逻辑迁入最终数据层时同步迁移这些测试 |
| `util/tests/test_augmentations_cache.py` | `SUPPORT` | 验证 PN2021-C 白名单依赖、20组合、确定性随机流、双采样率及断点缓存契约 | 缓存入口迁移时同步迁移这些测试 |
| `util/tests/test_load_cache.py` | `SUPPORT` | 验证三类缓存身份、自动 RAM/mmap 决策、中心/hash/view 访问和布局转换 | 统一数据访问层迁移时同步迁移这些契约测试 |
| `util/tests/test_split_cache.py` | `SUPPORT` | 验证 PTB-XL 患者隔离、PN2021 确定性父 K500、独立 400/100、singleton train、CPSC 来源配额、互斥并集与 ref-exclusion | 切分层迁移时同步迁移这些契约测试 |
| `util/tests/test_data_runtime.py` | `SUPPORT` | 验证split/cache、K500/400/100、CPSC合并、唯一data-load配置、resident/mmap数值/hash/sampler顺序、PN2021-C validation显式view及session所有权 | 运行时数据层迁移时同步迁移这些契约测试 |
| `util/tests/test_models.py` | `SUPPORT` | 验证两个模型的输入输出、冻结范围、统一factory、外部参考state键及 EfficientNet逐元素前向对齐 | 模型包迁移时同步迁移官方/历史兼容契约测试 |
| `util/tests/test_model_input_adapter.py` | `SUPPORT` | 验证 canonical raw100 的 sanitize、EffNet identity、ECGFounder 设备内 linear 1000→5000、先升采样后 global z-score、非原地及严格输入契约 | 模型输入桥迁移时同步保留两模型数值参考和操作顺序回归测试 |
| `util/tests/test_canonical_corruption.py` | `SUPPORT` | 验证在线 GPU 腐蚀内核固定 `100→500→100`、20种 depth2/3 掩码、两链批量调用结构、确定性、非原地输入和逐样本 finite provenance | 腐蚀内核迁移时同步保留域、组合、批量执行和设备驻留契约 |
| `util/tests/test_vae_lhat_augmix.py` | `SUPPORT` | 验证真实 VAE、LHAT M候选几何/2B probe、attack-then-contract 标签边界保护与端点残差，以及 Stage-1 两链 Dirichlet/Beta 公式、确定性和白名单 import 闭包 | 在线训练核心迁移时同步迁移这些契约测试 |
| `util/tests/test_random_seed.py` | `SUPPORT` | 验证全局 seed YAML 身份、namespace 隔离、Python/NumPy/Torch 复现和 process 初始化 | 随机基础设施迁移时同步迁移这些契约测试 |
| `util/tests/test_config_bundle.py` | `SUPPORT` | 验证完整 configs 副本内的 seed/operator 引用、路径逃逸闸门和核心 YAML 加载 | 配置系统迁移时同步迁移这些契约测试 |
| `util/tests/test_supervised_trainer.py` | `SUPPORT` | 验证配置束、调用方 DataLoader、参数更新、验证选模、raw prediction artifact、单中心缺类留证、scheduler horizon、测试落盘及 last-epoch 微调 | 通用监督训练入口迁移时同步迁移这些最小契约测试 |
| `util/tests/test_evaluation_metrics.py` | `SUPPORT` | 验证 raw-logit 排序不被 sigmoid 饱和破坏、Average Precision 定义及 strict/skip-undefined | 指标层迁移时同步保留定义与极端 logits 回归样例 |
| `util/tests/test_direct_baseline_selection.py` | `SUPPORT` | 验证clean pooled400、20个逐composition pooled400、robust均值、0.5/0.5 score、clean floor、严格五类、身份/grid漂移拒绝及exact tie | Direct baseline选模层迁移时同步保留全部泄漏、不可flatten和可比性闸门 |
| `util/tests/test_pn2021_tuning.py` | `SUPPORT` | 验证train400、clean validation100、20个冻结PN2021-C view、packed bank 逐 view 等价、动态 executable method、可选 runtime encoder/decoder 与主线 train400 latent pool、source registry及hash/label/order一致 | tuning适配层迁移时同步保留Direct身份、packed raw/logit/metric等价、资源精确路由、pool不消费训练shuffle及只读validation边界 |
| `util/tests/test_pn2021_direct_boot_scripts.py` | `SUPPORT` | 验证tune/select/refit零副作用dry-run、Direct稳定身份、latent VAE requirements/dry-run不读权重、family-balanced selection及E*/scheduler传播 | 启动层迁移时同步保留CLI、配置束和资源加载契约 |
| `util/tests/test_train_ptbxl.py` | `SUPPORT` | 验证 PTB-XL 100/500 Hz 自动适配、官方三折 loader、参数下传和 loader 所有权 | PTB-XL 数据适配入口迁移时同步迁移这些最小契约测试 |
| `util/tests/test_ptbxl_boot_scripts.py` | `SUPPORT` | 验证 EffNet/ECGFounder dry-run 的模型采样率、默认 full-FT profile 和显式 CLI 覆盖 | 两个 PTB-XL boot 入口迁移时同步迁移这些最小契约测试 |
| `util/tests/test_tensorboard_logging.py` | `SUPPORT` | 验证 final-only probe、manifest 延迟 flush、PNG/NPY 保留、可关闭 event image、TensorBoard 标量和 disabled 零副作用 | 观察层迁移时同步迁移相同最小契约测试 |
| `util/tests/test_augmentation_profile.py` | `SUPPORT` | 验证离线缓存与在线 AugMix 共用同一算子参数、seed/config SHA 及 copied config bundle | profile loader 迁移时同步保留共享单一来源契约 |
| `util/tests/test_latent_pool.py` | `SUPPORT` | 验证 deterministic-mean latent、M20 eligibility、预计算 neighbor table 与 stable reference 逐元素一致、table SHA 稳定及 hash/index 访问 | latent pool 迁移时同步保留候选和身份契约 |
| `util/tests/test_method_graph.py` | `SUPPORT` | 验证 A0/A3c、主线 attack-contract、可执行 latent-threechain 及 audit-only profiles 的静态编译、typed execution、资源白名单、RNG 身份和动态实现键禁令 | 方法图迁移时同步保留白名单、类型、执行性、资源闭包及 matched comparison RNG 身份契约 |
| `util/tests/test_online_trainer.py` | `SUPPORT` | 验证主线 Stage-1 两链 SimCLR 冻结头、五轮 rotating4 全覆盖、完整 Stage-1→Stage-2 CPU 闭环、单 optimizer-step、family/BN/RNG balance、计数及 final-only checkpoint | 在线trainer迁移时同步保留两阶段数值、预算、teacher/checkpoint身份和 last-checkpoint 日志语义 |
| `util/tests/test_train_pn2021.py` | `SUPPORT` | 验证raw100 K500/train400、canonical resident参数、worker安全门、partition进入seed身份，以及 latent-pool/encoder/decoder requirements 的精确资源路由 | PN2021适配层迁移时同步保留数据、seed和方法资源边界 |
| `util/tests/test_pn2021_evaluation.py` | `SUPPORT` | 验证clean+20 views共用canonical session、关闭所有权、ECGFounder linear500+z-score数值链、K500 identity及拒绝500 Hz cache | 正式评估层迁移时同步保留canonical bottleneck、共享adapter、metric view和证据契约 |
| `util/tests/test_manual_run_experiment.py` | `SUPPORT` | 验证统一launcher dry-run、entrypoint/flag白名单、removed flag拒绝、配置闭包、family-balanced selection run record、文件索引和篡改检测 | launcher/run-record迁移时同步保留零副作用、方法身份和完整性契约 |

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
| `methods/augmix/ecg_ops.py` | `util/augmentations/operators.py`、`util/augmentations/torch_operators.py` | 白名单调用与 golden 测试已全部迁移；剩余引用均位于待退出旧树，需与旧调用方同批删除 |
| `methods/augmix/severity.py` | `configs/augmentation/operators.yaml`、`util/augmentations/profile.py` | 白名单已无引用；剩余引用均位于待退出旧树，需与旧调用方同批删除 |
| `configs/corruption_profiles/` 中被替代的历史 profile | `configs/augmentation/operators.yaml` | 需要逐文件检查 active config、证据记录和历史回放引用，禁止整目录删除 |
| 旧数据预处理入口和旧实验 YAML | `data_preprocess/` 与受管实验 YAML | 需要从 `configs/active_scripts.yaml`、import closure 和运行记录生成精确候选清单 |

## D. 删除闸门

一个旧文件只有同时满足以下条件，才能从“候选”变成“允许删除”：

1. 在本清单中写明它的精确替代文件和职责映射。
2. 使用 `rg` 确认没有活动代码、YAML、测试或文档仍引用旧路径。
3. 新实现通过对应单元测试和数据契约测试。
4. 相关 `boot_scripts/run_experiment.py --dry-run` 通过协议审计。
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
/home/linbinhao/micromamba/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_ningbo.yaml \
  --dry-run
```

涉及 GPU、长推理或缓存构建的验证，仍需先检查共享服务器资源并显式选择 GPU。

## E. 清理批次记录

每次实际删除前后，在这里追加记录；不要改写历史批次。

| 日期 | 批次 | 删除路径 | 替代路径 | 验证结果 | 用户确认 |
|---|---|---|---|---|---|
| 2026-08-06 | 批次 1（已执行） | `ecg_adv_gen/data/synthetic_npz.py`；`ecg_adv_gen/training/effnet_super5.py`；`data_preprocess/prepare_ptbxl_for_ecgtwin.py`；`configs/experiments/pn2021c_effnet_paper_anchored_s5_depth23_composite.yaml` | `data_preprocess/load_cache.py` + `data_preprocess/data_runtime.py`；`core/supervised_trainer.py`；`models/vae.py` + `core/train_PN2021.py`；`configs/augmentation/operators.yaml` + `configs/eval/PN2021.yaml` | 删除后：四路径活动引用=0；白名单 Python issue=0；44 个实验 YAML 闭包 issue=0；5 个活动阶段 dry-run 通过；148 passed/16 skipped；`git diff --check` 通过 | 用户于 2026-08-06 明确回复“帮我删除” |
| 2026-08-06 | 批次 2（干净 worktree 初始裁剪） | 仅在 `/home/linbinhao/ECG_manual_refactor_clean` 删除 A 区以外的 449 个 Git 跟踪项；精确路径由本批次提交相对父提交 `6a662c9` 的删除 diff 定义；排序后 NUL 分隔清单 SHA256 为 `f70452fed8bef77d4b171fa77086f1a727b1cb75746e058b0d1fd15a38079bac` | A0–A9 共 154 个 `KEEP-MANUAL` 文件；源 worktree `/home/linbinhao/ECG_manual_refactor` 保持历史现场 | 索引恰好 154/154，无额外或缺失路径；65 个 Python 文件本地 import issue=0；44 个实验 YAML、70 文件闭包 issue=0；73 passed/10 skipped；5 个活动阶段使用独立 `/home/linbinhao` dry-run 目标全部通过；`git diff --check` 通过 | 用户于 2026-08-06 明确回复“好的接受你的建议开始迁移” |
| 2026-08-06 | 批次 3（公开代理记忆迁移） | 从迁移边界提交恢复 `.codex/skills/` 13 个项目文件；修正当前 clean worktree 路径、已删除入口和主线优先级 | `AGENTS.md` + keep manifest + active scripts/evidence + 八个 repo-tracked skills；全局 memory/session 不迁移 | skill frontmatter、引用路径、白名单精确集合、最低 CPU 测试和 Git 产物检查通过 | 用户于 2026-08-06 明确要求相关 AGENTS 与 skill 项目记忆一并迁移 |

`methods/augmix/ecg_ops.py` 与 `methods/augmix/severity.py` 暂不放进候选批次 1：
白名单已经切断依赖，但旧 `ecg_adv_gen/` 与 `methods/` 树内部仍相互引用。
它们应在后续“整棵旧运行树”批次中一起删除，避免暂时留下明显断裂的旧树。

## F. 当前下一步

- [x] 将 PN2021 `.hea` 的 SNOMED/人口学解析迁入独立白名单模块，移除
  `PN2021_preprocess.py` 对旧 `ecg_adv_gen.data` 包的运行时依赖，并锁定原有
  解析语义。
- [x] 为两个数据预处理脚本补齐非有限值质量控制的独立最小测试。
- [x] 实现 PTB-XL、PN2021、PN2021-C 的统一只读缓存加载与访问接口。
- [x] 实现 PTB-XL 官方折及 PN2021 四逻辑中心 K500/ref-excluded 的 ID-only 切分层，并将 CPSC 2018/Extra 合并为一个逻辑中心。
- [x] 实现从受管 split/cache 到模型 batch 的统一运行时数据层，包括显式 corruption view、增强后全样本 z-score、worker-safe Dataset 和确定性 DataLoader。
- [x] 建立独立 `models/` 包，完成 EfficientNet1DV2、ECGFounder、严格checkpoint身份和统一 `build_model` 接口，并通过真实checkpoint CPU smoke test。
- [x] 独立重写 ECGTwin VAE encoder/decoder，并在同一白名单模块中实现 exact-label VAE-LHAT、攻击后标签边界收缩和 Stage-1 两链 AugMix-SimCLR 受管 YAML。
- [x] 新增 `configs/train/PTBXL.yaml` 和 `core/supervised_trainer.py`；调用方通过统一 `data_runtime` 构建任意 PTB-XL/PN2021 DataLoader，再由通用训练器接受 model + DataLoader，支持验证集 macro-AUPRC 选模、最终测试与无验证集 last-epoch 微调。
- [x] 新增 `core/train_PTBXL.py` 作为白名单 PTB-XL 数据适配层，并新增 EffNet/ECGFounder 两个 `boot_scripts/` 薄入口；模型采样率自动绑定 100/500 Hz，YAML 默认值和显式覆盖均可审计。
- [x] 接入 run-scoped TensorBoard 观察层并移植 ECGTwin 作者 `ecg_plot` 绘图路径；监督 trainer 记录标量，在线训练接收 executor 提供的固定 hash raw-mV named views，event、PNG、NPY 和 manifest 均写入实验输出目录且不改变训练 RNG。
- [x] 当前主线不保留旧 ECGTwin VAE/Nomic 预处理入口；该 `TRANSITION`
  脚本已在批次 1 删除。
- [x] 将五算子收敛到 `util/augmentations/` 与单一 profile loader，并迁移
  全部白名单活动 import；旧树内部 bridge 留待整树批次删除。
- [x] 将共享 profile loader 与 `configs/augmentation/operators.yaml` 接入离线
  PN2021-C 缓存和在线 Torch AugMix；两条路径读取相同参数、seed/config SHA，
  不再维护重复 YAML 解析逻辑。
- [x] PN2021-C 缓存构建完成后，将随机种子能力提升为全项目公共基础设施：
  已将 `configs/radom_seed.yaml` 更名为 `configs/random_seed.yaml`，并将
  `util/augmentations/random_seed.py` 迁移为 `util/random_seed.py`，统一提供
  YAML 加载、身份化 seed 派生、Python/NumPy/Torch/CUDA 初始化和独立 RNG
  工厂；当前白名单内的模型工厂、切分、DataLoader worker、数据增强和
  对抗攻击已迁移到新入口。公平对比的各 method profile 必须共享
  comparison-group/replicate seed，
  每个运行记录基础 seed、有效 seed、namespace 和 seed YAML SHA256；迁移后
  删除了增强子包中的旧实现，不保留旧拼写兼容入口。
- [x] 增加 configs-shaped 配置束解析：复制整个 `configs/` 后，内部 YAML
  引用优先解析副本自身；数据运行时和模型工厂支持 `config_root`，各独立
  预处理/缓存/LHAT/AugMix 入口支持显式副本 YAML 路径。
- [x] 新训练器/统一实验 launcher 只接受一个 experiment YAML 加
  `config_root`，启动前解析并保存依赖 YAML 闭包、SHA256、seed identity、
  Git 身份和精确命令；禁止重新引入另一套隐式默认配置搜索规则。
- [x] 建立完整 K500 latent pool 和 A0/LHAT/主线匹配在线训练层；候选不足
  M20 的 anchor 不丢弃、不重复填充，只贡献 clean loss并落盘清单。
- [x] 将主线重构为受限 typed method graph：Stage-1 在 trainer 内使用
  K500 两链 AugMix-SimCLR，Stage-2 graph 只声明 clean、rotating4 corruption
  与 attack-then-contract LHAT view；梯度累计后每个 base batch 只更新一次。
  audit-only 组合 profile 在节点与预算契约完成前继续 fail-closed。
- [x] 将 `agent_workspace` 中锁定的精简开发 recipe 迁入白名单运行面：
  `augmix_simclr_lhat.yaml`、`PN2021.yaml`、统一 launcher 示例、Stage-1
  checkpoint/TensorBoard、VAE contract 诊断和配置/seed/resource 身份均已接管；
  仍标记为 prospective replication，未因代码迁移自动升级为论文最终证据。
- [x] 在 `util/evaluation/` 建立固定 checkpoint 的四中心 clean/PN2021-C
  正式评估层，输出 kept/drop、per-class/macro、depth2/3/23、per-center和
  四中心等权指标及完整 split/cache/checkpoint 身份。
- [x] 在固定父 K500 内物化确定性 train400/validation100 v2 split；Direct
  tuning 逐 epoch 保存 raw logits/targets/hash，允许单中心 rare class 缺失，
  再将四中心 400 条拼接后按严格五类 sklearn Average Precision 选择单一
  全局 epoch；EffNet/ECGFounder 均已提供受管 tuning/selection dry-run 入口。
- [x] 完成两个模型的四中心 family-balanced Direct tuning、pooled E* 选择、
  完整 K500 refit 与 ref-excluded clean/PN2021-C 评估；EffNet 锁定 E23/T30，
  ECGFounder 原始 Direct 锁定 E20/T20。
- [x] 完成 ECGFounder latent-threechain 候选的 K500 内部调参、E19/T20 full-K500
  refit、四中心正式评估及 LR `3e-5` matched Direct 归因控制；配置、选择证据、
  权重 SHA、TensorBoard 与方法增益边界均写入现有白名单注册表和决策日志。
- [x] 在保留源 worktree 的前提下生成 A 区以外 449 个 Git 跟踪项的精确
  候选集合，并记录排序清单 SHA256。
- [x] 用户确认迁移后，仅在新干净 worktree 执行裁剪；源 worktree 的历史
  文件、未跟踪实验和外部模型目标均未删除。
