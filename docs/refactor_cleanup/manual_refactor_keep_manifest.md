# 手动重构保留清单与旧代码清理闸门

更新日期：2026-08-12

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
| `pytest.ini` | `KEEP-MANUAL` | 将公共 CPU 回归测试发现范围锁定在 `util/tests/test_*.py`，并严格拒绝未知配置和 marker | 新测试配置完整接管同一发现边界和严格模式 |
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
| `environments/README.md` | `KEEP-MANUAL` | Miniforge ECGTwin 重建、验证、切换和旧环境退出契约 | 新环境文档接管 conda/pip 分层、轮子哈希、验证闸门和外部产物边界 |
| `environments/cli-tools-miniforge.yml` | `KEEP-MANUAL` | 锁定纯 conda-forge 的 tmux 3.7b 与 Git 2.51.0 CLI 环境 | 3.6a 兼容阶段通过旧 server 查询；3.7b 通过隔离 tmux 配置和仓库 Git smoke test，最终切换需重启 server |
| `environments/ecgtwin-miniforge.yml` | `KEEP-MANUAL` | 锁定纯 conda-forge Python 3.11.5 基础环境 | 新基础环境规范保持 Python、pip、setuptools、wheel 和 Git 兼容版本 |
| `environments/ecgtwin-pip-lock.txt` | `KEEP-MANUAL` | 锁定 clean-room 主线所需的非 PyTorch pip 依赖闭包 | 新锁文件通过 pip check、CPU 契约和模型 checkpoint smoke test |
| `environments/pytorch-cu118-wheel-manifest.sha256` | `KEEP-MANUAL` | 记录外部 PyTorch 2.1.1 CUDA 11.8 三轮子的内容哈希 | 新轮子清单保持版本组合、文件名和 SHA256 可验证 |
| `docs/refactor_cleanup/manual_refactor_keep_manifest.md` | `KEEP-MANUAL` | 手工重构保护边界、退出候选与破坏性删除闸门 | 新清单逐文件接管全部 KEEP/TRANSITION/证据和用户确认记录 |
| `docs/refactor_cleanup/pipeline_reproduction_log_20260806.md` | `KEEP-MANUAL` | 记录干净工作树全链路复现的目标、问题、修复、证据和最终判定 | 本轮复现结论已经迁入新的可追溯运行记录或交接文档 |

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
| `configs/data/data_load.yaml` | `KEEP-MANUAL` | 绑定 tracked content ledger 并仅声明共享 mmap engine 顺序/预取策略；batch、worker、resident、validation 与 model-input 由三种 finite LoaderPlan 及其训练/评估 YAML 拥有 | 新配置保留 ledger descriptor、mmap 顺序/有界预取与配置 SHA；不得重新开放 generic loader 默认或隐式 model-input transform |
| `configs/data/data_content_ledger_v1.jsonl` | `KEEP-MANUAL` | 121 个活动派生 cache/split 文件的 canonical relocatable SHA256 内容账本；不含绝对数据根 | 新证据保留 exact root/path/size/SHA、122 行 canonical JSONL、两遍 full-content 身份与 derived-only scope，不冒充 raw WFDB/preprocessing correctness seal |
| `configs/data/k500_handoff.yaml` | `KEEP-MANUAL` | 学术方法对比的数据接口交接契约；锁定配置 SHA、内容账本、四中心 K500/K400/K100/ref-excluded hash 身份和只读边界 | 新交接契约保留 source/split manifest、内容账本、逐中心 hash-set、映射、类序、seed 来源、配置闭包及 heldout 只读语义 |
| `configs/random_seed.yaml` | `KEEP-MANUAL` | 手动重构代码统一使用的项目基础随机种子 | 新配置保留基础 seed、配置 SHA256 和所有白名单随机入口 |

### A2. 数据预处理代码

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `data_preprocess/PTBXL_preprocess.py` | `KEEP-MANUAL` | PTB-XL 读取、非有限值质量控制、10 秒截取、100/500 Hz NPY 缓存和唯一 ID | 新入口完成同一缓存契约并通过数据验证 |
| `data_preprocess/pn2021_metadata.py` | `KEEP-MANUAL` | 独立解析 PN2021 `.hea` 中的 SNOMED、年龄和性别，不导入旧 `ecg_adv_gen.data` | 新实现保持表头解析语义并通过独立 golden contract 测试 |
| `data_preprocess/PN2021_preprocess.py` | `KEEP-MANUAL` | PN2021 多中心读取、非有限值质量控制、Super5 映射、10 秒截取、100/500 Hz NPY 缓存和唯一 ID | 新入口完成同一缓存契约并通过数据验证 |
| `data_preprocess/augmentations_cache.py` | `KEEP-MANUAL` | 从白名单 PN2021 缓存和唯一 `AugmentationProfile` 生成五中心、depth2+3、100/500 Hz PN2021-C 连续缓存 | 新入口保留源缓存身份、20组合、随机种子、双采样率契约和单一严格 profile loader，不恢复 dict fallback 或 snapshot wrapper |
| `data_preprocess/load_cache.py` | `KEEP-MANUAL` | 统一校验并以只读 mmap 访问 PTB-XL、PN2021、PN2021-C 缓存；支持批量 hash 到 index 的严格解析 | 新数据层保留 manifest/hash、布局、中心、view、只读 mmap 和原始 mV 契约；不得恢复 auto/RAM cache 模式 |
| `data_preprocess/split_cache.py` | `KEEP-MANUAL` | 依赖统一 cache loader 生成 ID-only PTB-XL 官方折、PN2021 固定 K500、确定性多标签/物理来源近似分层 400/100 及 ref-excluded 切分；singleton positive 保留在 train，CPSC/Extra validation 按父 K500 来源比例配额 | 新切分层保留 source manifest、映射、父 K500 seed/身份、独立 tuning seed、候选池、split hash、400/100 互斥并集、来源/类别计数、患者隔离和零 K500 泄漏契约 |
| `data_preprocess/data_ledger.py` | `KEEP-MANUAL` | 从活动配置解析四个逻辑数据根，生成/校验可迁移且确定排序的 JSONL 内容账本；quick 只核精确成员集合/大小，full 额外流式 SHA256，并拒绝 symlink、特殊文件及 metadata 可见的扫描期变化 | 新证据层保留 canonical JSONL、路径迁移等价、原子 no-clobber、逐文件与整轮 metadata 稳定性、quick/full 明确分级及真实数据根配置来源；正式封存仍要求 quiescent roots + 独立第二次 full verify，不宣称 filesystem snapshot isolation |
| `data_preprocess/data_runtime.py` | `KEEP-MANUAL` | 校验 split/cache 身份并以 `PTBXLLoaderPlan`、`PN2021K500LoaderPlan`、`PN2021EvaluationLoaderPlan` 提供有限 raw100 DataLoader 面；支持批量 mmap/prefetch、只 gather 已验证 K500 到连续 CPU tensor 后关闭源 mmap，以及跨 clean+20 views 复用 mmap/selection 的 `SequentialEvaluationDataSession` | 新运行时层保留父 K500=400+100、cache index/hash/record/source-center/sampler 顺序、canonical raw mV `(B,1000,12)`、显式 PN2021-C view、CPSC 合并中心、seed 身份、resident/mmap 数值等价和 plan/loader/session 所有权；不得重开 public free-form loader |
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
| `util/config_bundle.py` | `KEEP-MANUAL` | 将 YAML 间引用限制在当前选中的 configs-shaped 配置束内，使完整配置副本可移植且不回落主配置；提供受管训练/评估的递归 YAML 配置闭包 | 新实现保留显式 config root、禁止相对路径逃逸、副本优先解析和闭包测试 |
| `configs/augmentation/operators.yaml` | `KEEP-MANUAL` | paper-anchored S5 参数、单位和 depth2+3 组合契约 | 新配置保留 profile 名、参数来源和可复现 hash |
| `configs/augmentation/cache.yaml` | `KEEP-MANUAL` | 五中心 PN2021-C depth2+3 双采样率缓存输入、输出和断点续建契约 | 新配置保留源缓存身份、组合顺序、采样率和机械盘输出身份 |

### A4. 模型定义与构建接口

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `models/__init__.py` | `KEEP-MANUAL` | 受管模型包的五函数公共导出面：`available_models`、`build_model`、`get_model_spec`、`build_ecgtwin_vae`、`load_vae_config` | 新包完成受管调用迁移并保留这五个构建/查询函数；模型类、spec、checkpoint、input adapter 与 VAE 细节仅从 owner module 导入，不恢复 package-level re-export |
| `models/contracts.py` | `KEEP-MANUAL` | 锁定 EfficientNet `(B,12,1000)`、ECGFounder `(B,12,5000)` 与 Super5 五原始 logits 契约 | 新实现保留采样率、布局、类别顺序及输入输出严格校验 |
| `models/input_adapter.py` | `KEEP-MANUAL` | 统一 canonical raw 100 Hz BTC 到两模型输入域的设备驻留适配：sanitize、ECGFounder 线性升采样、全局 z-score、BCT | 新实现保留 EffNet identity、ECGFounder `1000→5000` linear `align_corners=True`、先升采样后归一化、有限 float32 输出及非原地语义 |
| `models/checkpoints.py` | `KEEP-MANUAL` | 严格提取 raw/state_dict/model_state_dict、去除并行前缀、记录 SHA256，并仅为显式官方 ECGFounder checkpoint 开放 trusted 读取 | 新实现保留安全默认、严格键校验和 checkpoint 身份记录 |
| `models/efficientnet1d.py` | `KEEP-MANUAL` | 独立重写历史 EfficientNet1DV2 S-V2 五分类定义及 checkpoint 构建接口，并显式暴露冻结分类头前的 `forward_features` 供两链 AugMix-SimCLR 使用 | 新实现与571个state key、旧checkpoint和100 Hz前向逐元素兼容，且 feature API 不改变原分类前向 |
| `models/ecgfounder.py` | `KEEP-MANUAL` | 独立重写官方12导联 ECGFounder Net1D、预训练backbone加载、Super5 head及 full/head trainable scope；复用既有 `forward_features` 接入同一 Stage-1 | 新实现与509个官方state key/shape、官方checkpoint和500 Hz前向兼容，且两个骨干共享同一表征学习接口 |
| `models/factory.py` | `KEEP-MANUAL` | 以一个私有 canonical registry 统一拥有两骨干 spec/builder，并仅公开 `available_models`、`build_model`、`get_model_spec` | 新模型工厂完整接管两个 canonical backbone 名与隔离初始化 seed，拒绝旧 alias/normalize 旁路且不公开可变 registry、不引入训练器依赖 |
| `models/vae.py` | `KEEP-MANUAL` | 独立重写 ECGTwin VAE encoder/decoder、严格读取双 state-dict checkpoint；encoder 仅返回确定性 `(scaled_mean, mean, log_variance)` 三元组，decoder helper 仅返回 canonical raw100 | 新实现保留 `(B,1024,12)` raw mV、scaled `(B,4,128)` posterior mean、logvar、导联交换、0.18215 scale、固定 `(B,1000,12)` 解码、240个 state key 和真实 checkpoint 前向契约；不恢复 encoder 随机采样或 decoder 5000点分支 |

### A5. 两链 AugMix-SimCLR 与收缩式 VAE-LHAT 主线

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `configs/train/vae.yaml` | `KEEP-MANUAL` | ECGTwin VAE checkpoint、输入/latent、冻结方式和分类器桥接契约 | 新配置保留 checkpoint 双组件、raw mV、1024点、lead reorder、latent scale 和严格键数量 |
| `configs/train/lhat.yaml` | `KEEP-MANUAL` | M20 exact-label/non-self latent hull、标准化、λ0.6、ε12、10步 BCE 困难搜索，以及 `[0.25,0.5,0.75,1]` 预翻转最大损失收缩网格和端点残差修正 | 新配置保留 train-only standardizer、`include_anchor=false`、100 Hz 信息瓶颈、clean-correct margin 50% 保留、无 heldout feedback 和 raw/contract 双层诊断 |
| `configs/train/augmix.yaml` | `KEEP-MANUAL` | 仅配置 Stage-1 两条独立 depth2/3 腐蚀链：在 500 Hz 算子域执行并回到 canonical100，以 Dirichlet(0.5,0.5) 混链、Beta(0.5,0.5) 混 clean | 新配置保留顺序隔离 RNG、pre-zscore raw mV、两链公式/温度0.5、非原地输入和配置束内单一算子来源 |
| `configs/train/PN2021.yaml` | `KEEP-MANUAL` | 四个文件选择器及唯一 code-owned matched-no-VAE slot 共用的 PN2021 训练控制：EffNet Stage-1 1024步 + E23/T30，ECGFounder Stage-1 256步 + E30/T30；统一完整 K500、resident loader、最后 checkpoint、epoch history 和无 heldout 选模 | 新配置保留同源 checkpoint、完整 K500、两阶段超参、matched-base 单步预算、ref-exclusion、v7 映射、drop-all-zero 主口径和100 Hz bottleneck；不重复 recipe 身份或科学常量 |
| `configs/train/methods/a0_clean_v1.yaml` | `KEEP-MANUAL` | schema-v2 A0 clean RecipeSpec 选择器；仅声明 recipe 身份且资源为空 | 新 selector 保留 A0 身份；单 clean BCE、完整 base exposure 和单次 outer step 必须由代码所有 |
| `configs/train/methods/a3c_depth23_v1.yaml` | `KEEP-MANUAL` | schema-v2 A3c random-depth23 RecipeSpec 选择器；仅引用 operator profile 与隔离 RNG | 新 selector 保留 A3c 身份和外部资源闭包；每 origin 单一腐蚀 view、双 BCE、500 Hz 算子域和 matched-base 预算必须由代码所有 |
| `configs/train/methods/direct_depth23_fixed20.yaml` | `KEEP-MANUAL` | schema-v2 Direct fixed20 RecipeSpec 选择器；仅引用 operator profile 与隔离 RNG，不承诺 tuning/refit 受管重放 | 新 selector 保留 Direct 身份和外部资源闭包；显式0..19、`0.5 clean + 0.5 mean(corruption)`、在线生成和单次 optimizer step 必须由代码所有；历史性能只引用冻结证据 |
| `configs/train/methods/augmix_simclr_lhat.yaml` | `KEEP-MANUAL` | schema-v2 唯一主线 RecipeSpec 选择器；固定 `contracted_lhat`，仅引用 operator/AugMix/VAE/LHAT 与两条隔离 RNG | 新 selector 保留主线身份和完整外部资源闭包；K500-only、两阶段拓扑、五轮覆盖、family balance、辅助权重2、BN/RNG恢复和无 heldout feedback 必须由代码所有 |
| `core/__init__.py` | `KEEP-MANUAL` | 仅作为显式空 package boundary；所有生产消费者直接从 owner module 导入 | 新核心包保留空 `__all__`，不恢复训练、方法、数据或配置的 eager re-export |
| `core/corruption.py` | `KEEP-MANUAL` | 共享 GPU canonical 腐蚀内核；把两条链合并成 `2B`，固定在 500 Hz 调用五算子并返回 100 Hz raw-mV 波形及逐样本 provenance | 新实现保留线性 `100→500→100`、五算子各一次批量调用、depth2/3 组合掩码、无逐组合 CPU 分支和有限值诊断 |
| `core/lhat.py` | `KEEP-MANUAL` | train-only latent standardizer、exact-label non-self M20 hull 优化、固定 100 Hz canonical 可微解码后再走两模型攻击桥；攻击后一次批量解码收缩路径、端点残差修正并选择最高 BCE 的标签边界保护点 | 新实现保留只优化 batch-local hull 权重、不污染分类器/decoder 梯度、ε12/10步 raw 搜索、clean-correct margin 保护、finite/std/20mV gate、clean fallback、ASR/接受率/t 诊断和紧凑 D2H；不得把 backbone 目标长度重新下沉到 VAE decoder helper |
| `core/augmix.py` | `KEEP-MANUAL` | 仅实现冻结 Stage-1 两链强视图；两条链按锁定 RNG 顺序独立生成后做逐样本 Dirichlet 与 Beta 混合 | 新实现保留 raw100 输入、500 Hz 算子域、canonical100输出、显式 generator、非原地语义、可复算权重和不把两条链合并成会改变 RNG 身份的 `2B` 调用 |
| `core/latent_pool.py` | `KEEP-MANUAL` | 从 raw 100 Hz K500 经 compact deterministic encoder tuple 建立 frozen VAE posterior-mean latent pool；schema v2 预计算 stable exact-label neighbor table，缓存 eligible hash tuple/set，并把邻居表 SHA256 纳入身份 | 新实现保留 encoder/cache/label/latent/eligibility/standardizer 身份、`(scaled_mean, mean, logvar)` shape/finite 校验、候选 distinct/non-self、与原 stable search 逐元素一致及不可用样本显式清单 |
| `core/methods/__init__.py` | `KEEP-MANUAL` | 仅作为显式空 recipe package boundary；contracts、registry 与 runtime 各自拥有其 API | 新包保留空 `__all__`与 owner-module direct import，不恢复 barrel re-export、DAG、动态 import 或插件式组合 |
| `core/methods/contracts.py` | `KEEP-MANUAL` | 定义 canonical raw-mV waveform、Super5 target、valid-mask、objective 和 code-owned resource requirement 的最小契约 | 新实现保留 `(B,1000,12)`、Super5、有限值、batch/sample-id 对齐与不可变输出检查 |
| `core/methods/registry.py` | `KEEP-MANUAL` | 严格加载四个 schema-v2 文件选择器并保留唯一 code-owned matched-no-VAE slot，以代码白名单绑定 kind/variant/科学合同/objective/requirements/output/RNG 身份并生成路径无关 spec SHA | 新 loader 保留根键/recipe键/资源键精确检查、未知值与动态 import fail-closed、外部配置引用闭包和完整科学身份哈希 |
| `core/methods/runtime.py` | `KEEP-MANUAL` | 按有限 recipe 直接分派 canonical corruption 与 attack-then-contract LHAT；只为候选充足且通过收缩/QC 的记录暴露辅助 view，其余 clean-only | 新 runtime 保留 K500 hash/标签绑定、exact-label pool、500 Hz 算子域、legacy RNG payload、contract accepted mask、组合拒绝原因、raw/contract 诊断、显式随机 trace 和无 agent_workspace import |
| `core/online_trainer.py` | `KEEP-MANUAL` | 执行四个文件选择器与唯一 code-owned matched-no-VAE slot；主线为冻结分类头的两链 AugMix-SimCLR、source/stage2 logit anchor、五轮 rotating4、VAE 辅助直接相加及一次 outer update；Stage-1 消费 runtime 已解析的同一 AugMixConfig | 新实现保留 pre-zscore raw mV、K500 teacher cache、两个骨干 feature API、legacy RNG、单步 family balance、辅助 BN/RNG snapshot-restore、optimizer/view计数、最终 checkpoint、无 heldout 选模和完整配置/seed闭包 |
| `core/train_PN2021.py` | `KEEP-MANUAL` | 通过 `data_runtime` 建立 raw 100 Hz 完整 K500 loader，校验 source checkpoint 锁，并按 RecipeSpec requirements 用 frozen encoder 构建 latent pool 后释放、仅向在线 runtime 路由 pool/decoder | 新适配层保留四逻辑中心、CPSC/Extra 合并、K500 partition 身份、source path/SHA 锁、resident workers 约束、无pool recipe资源隔离及 recipe 不进入 DataLoader seed |

### A6. 通用监督训练入口

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `configs/train/PTBXL.yaml` | `KEEP-MANUAL` | 通用训练器的 prospective PTB-XL source profile；统一声明 raw physical-mV 100 Hz BTC loader、两骨干 code-owned 输入适配、folds 1-8/9/10、AdamW、BCE、AMP、fold9 macro-AUPRC 选模及独立输出目录 | 新配置保留相同数据身份、raw100 bottleneck、EffNet canonical100、ECGFounder device linear `1000→5000 align_corners=True` 后归一化、选模边界、模型初始化、配置束引用和 run-scoped JSON/checkpoint 路径；不得把 live config 冒充历史 checkpoint bitwise replay |
| `core/supervised_trainer.py` | `KEEP-MANUAL` | 接受调用方构建的 Torch model 与 DataLoader，完成多标签监督训练；统一 raw-logit AUROC/average-precision，支持 H2D 后的可选模型输入适配、普通 validation 选模、严格 checkpoint 和 JSON 历史 | 新训练器保留默认无适配路径的行为，以及 model/DataLoader/seed/config/checkpoint/override/input-adapter 身份、单调 optimizer step、PTB-XL validation 选模与 last-epoch 微调契约；不恢复零消费者 raw prediction 导出层 |
| `core/train_PTBXL.py` | `KEEP-MANUAL` | 唯一拥有两骨干 PTB-XL YAML profile→model kwargs/dry-run JSON/训练委托，并从同一配置构造 raw100 `PTBXLLoaderPlan`、folds 1-8/9/可选10 与 code-owned model input adapter | 新 PTB-XL 适配器保留 split/cache/raw100/finite-plan 身份、三份 direct boot JSON、EffNet/ECGFounder 精确构建/适配顺序、Founder 唯一 `epochs=10` override、未授权时不构造 fold10，并继续只依赖白名单数据层和通用训练器 |
| `boot_scripts/__init__.py` | `KEEP-MANUAL` | PTB-XL 薄启动脚本包边界 | 新启动包完整接管两个模型入口且不承载训练业务逻辑 |
| `boot_scripts/train_ptbxl_effnet.py` | `KEEP-MANUAL` | 保留原脚本路径与 config/config-root/output/dry-run parser，向 `core.train_PTBXL.run_ptbxl_boot` 传递固定 EfficientNet selector | 新入口保留原 CLI/JSON、raw100→EffNet、配置束、随机种子和输出身份；不得承载 profile/model/training 业务或恢复 checkpoint/device/DataLoader 超参旁路 |
| `boot_scripts/train_ptbxl_ecgfounder.py` | `KEEP-MANUAL` | 保留原脚本路径与 config/config-root/output/dry-run/唯一 `--epochs=10` parser，向共享 owner 传递固定 ECGFounder selector | 新入口保留原 CLI/JSON、raw100→device linear5000、官方 checkpoint/full scope、配置束、随机种子和输出身份；不得承载重复业务或恢复任意 runtime override |
| `boot_scripts/train_pn2021.py` | `KEEP-MANUAL` | 只接受四个业务 selector（source checkpoint、model、center、method config）启动完整 K500 训练；runtime/training/resource 细节由 PN2021 YAML 与 code-owned RecipeSpec requirements 唯一拥有，支持无副作用 dry-run | 新入口保留源 checkpoint 强制输入、同组 seed/full-FT、recipe 隔离、A0/A3c 零 VAE、主线严格 VAE 加载和 YAML-owned 运行身份；不接受 CLI 超参旁路、selection/refit 或任意组合分支 |

### A7. 训练观察与 ECG 可视化

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `.gitignore` | `KEEP-MANUAL` | 排除 TensorBoard events、run-scoped ECG probe 和其他本地产生的大型训练观察物 | 新规则继续阻止事件、图片、缓存和运行输出误入 Git，且不屏蔽受管源码/配置 |

### A8. 正式评估、统一启动与运行留证

| 文件 | 状态 | 当前职责 | 删除或合并前必须满足 |
|---|---|---|---|
| `configs/active_scripts.yaml` | `KEEP-MANUAL` | 区分当前 finite RecipeSpec CPU/CUDA-diagnostic-verified 运行面与 r2 legacy typed-graph metric evidence，并登记 schema-v1 train result、schema-v3 checkpoint/evaluation、三种互斥评估 subject、新输出根、诊断 smoke 及仍待完成的正式 GPU 重复 | 新索引必须保留 trusted/development/legacy-runtime 分层、artifact lineage 与 closure 边界，禁止把单中心单轮 smoke 或 r2 冒充完整 RecipeSpec replay、性能证据或论文证据 |
| `configs/active_evidence_registry.yaml` | `KEEP-MANUAL` | 记录精简主线的 K500-only 数据边界、有限 recipe/spec SHA、开发选择审计、r2 legacy 身份缺口和论文晋级条件 | 新注册表保留 heldout-tuned/single-seed、无 outside-K500 model access、v7/drop-all-zero、旧 file/compiled SHA、Stage-1 AugMix snapshot 缺口及多随机重复要求 |
| `configs/eval/PN2021.yaml` | `KEEP-MANUAL` | PN2021正式评估唯一配置；单进程顺序session复用clean/PN2021-C mmap和selection，覆盖四中心clean+20 views | 新配置保留v7映射、K500 ref-exclusion、view顺序、raw100输入、四种canonical metric view及session/loader所有权 |
| `configs/eval/PN2021_matrix.yaml` | `KEEP-MANUAL` | 前瞻四中心 diagonal 聚合配置；按骨干锁 canonical 顺序、method/replicate/source/seed、公共数据身份、四中心 K500 身份、eval artifact locks 与配置 SHA | 新配置保留 config-owned exact cohort、equal-view→equal-center 和唯一结果名；不得重新引入数据加载、模型构建或 checkpoint 反序列化配置 |
| `util/pn2021_artifact_contract.py` | `KEEP-MANUAL` | 纯 stdlib 的 PN2021 lineage/train/evaluation/checkpoint/source/legacy artifact validator；唯一拥有 PN2021 artifact JSON mapping loader、path/SHA reference builder、SHA 与 owner-relative resolver，并复哈 checkpoint bytes | 新实现保留 exact schema、三模式隔离、heldout-free selection、K500/ref-exclusion、config-owned cohort 与无 Torch 导入；不得以宽松 truthy 字段或声明 SHA 代替字节验证 |
| `util/evaluation/__init__.py` | `KEEP-MANUAL` | 空的显式导入边界，避免 artifact-only matrix import 触发 PN2021 data/model/Torch 运行时 | 新包保持依赖透明；调用方从具体模块导入，不恢复 eager package re-export |
| `util/evaluation/metrics.py` | `KEEP-MANUAL` | 直接从 raw logits 计算 Super5 AUROC 与 sklearn Average Precision；显式 strict/skip-undefined、per-class/macro、kept/drop、中心/view 等权聚合，并复算校验所有派生 aggregate | 新实现禁止 sigmoid 饱和改变排序，保留 AP 非梯形 PR-AUC 定义、classes-used/正负计数、metric view、record/hash-label、depth2/3/23、equal-weight、canonical 四中心顺序和 derived-field exact compare |
| `util/evaluation/pn2021.py` | `KEEP-MANUAL` | 固定评估raw100 clean+20-view PN2021-C；schema-v3 result 绑定 subject，复用 artifact contract 的映射/中心/JSON/SHA/reference 身份与 cache loader 的 `EXPECTED_LEADS`，并统一复用一个sequential session | 新实现保留subject/model/中心一致性、checkpoint SHA、禁止PTB-XL shard、拒绝500 Hz cache、clean/corrupted同记录、显式view、record/hash-label顺序、loader/session关闭及完整输入链证据；不恢复本地 artifact/科学常量副本或旧worker/runtime profile回退 |
| `util/evaluation/matrix.py` | `KEEP-MANUAL` | 从四个 schema-v3 prospective 单中心结果重算 diagonal clean/20-view 与 canonical 四中心等权汇总；复用公共 JSON loader 与 path/SHA reference builder | 新实现保留输入顺序/唯一性、config-owned cohort、中心特异 K500/ref-exclusion、成员与实际 checkpoint 字节 SHA，以及成员 aggregate 复算校验；不得导入 Torch、重复 artifact utility 或反序列化权重 |
| `boot_scripts/evaluate_pn2021.py` | `KEEP-MANUAL` | 从评估 YAML 选择且仅选择一种 subject：前瞻 train-result+method-config 单中心、legacy A0/Direct 显式 checkpoint 单中心或 source-registry canonical4；构建模型并调用 canonical100 v2 正式评估，支持只读 dry-run | 新入口保留 schema/方法/模型/中心/checkpoint/lineage fail-closed、输入适配链和无隐式选模；前瞻 schema-v3 checkpoint 必须与 schema-v1 train result 同 lineage，legacy 仅接受已锁 A0/Direct schema-v2，source 仅接受注册表锁定 schema-v1 checkpoint |
| `boot_scripts/aggregate_pn2021.py` | `KEEP-MANUAL` | 四个前瞻单中心 evaluation JSON 的 CPU/JSON diagonal 聚合薄入口；受管 launcher dry-run 不执行 delegate | 新入口保留 exactly-four `--result`、按 `--model` 选择 config-owned cohort、不读取 ECG/不构建模型或 GPU、只流式复哈 checkpoint 而不反序列化，以及唯一外部输出 |
| `util/run_record.py` | `KEEP-MANUAL` | 记录命令、私有不可变配置闭包快照、data-ledger 双快照、Git/dirty SHA和环境；复用并 re-export 公共 SHA，复用 owner-relative artifact resolver 与统一 validator 校验 PN2021 train/evaluation/matrix | 新实现保留禁止 worktree 输出、原子 JSON、heldout-free last/eval、三模式 subject、四成员 matrix/config-owned cohort、派生指标复算、配置/ledger 快照与复哈、完整索引及实际 artifact 字节篡改检测；不恢复零调用 public `snapshot_yaml_files` 或 recorder 私有 resolver 副本 |
| `boot_scripts/run_experiment.py` | `KEEP-MANUAL` | 单一 experiment YAML + config-root 的白名单 launcher；五个 code-owned entrypoint 锁定脚本/result，dry-run零副作用；实际数据入口在建 run dir 前 quick 校验 ledger 并从完整 closure 快照启动 | 新入口保留五入口受限 flags、配置闭包、数据账本、外部唯一run dir、精确argv、预期结果身份和RunRecorder生命周期；外部结果 artifact 不得混入 YAML closure |
| `configs/baselines/ptbxl_source_v1.yaml` | `KEEP-MANUAL` | 锁定 EfficientNet1DV2/ECGFounder 的 PTB-XL source `best.pt`、SHA256、fold9 选模及 fold10 指标 | 新注册表完整接管相同 checkpoint 身份与禁止使用 `last.pt` 的下游契约 |
| `docs/baselines/ptbxl_source_v1.md` | `KEEP-MANUAL` | PTB-XL source baseline v1 的人类可读锁定日志、历史 config/checkpoint/metrics 与 live prospective finite-loader runtime 分界、原始证据路径及 10-epoch 否决结论 | 新决策日志完整保留历史配置、权重、SHA256、指标和下游使用规则，并禁止把 live runtime 迁移冒充 bitwise/metric replay |
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
| `configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_chapman_shaoxing.yaml` | `KEEP-MANUAL` | EfficientNet Chapman-Shaoxing 前瞻主线复现 | 保留同一方法闭包、锁定 source checkpoint、中心身份和唯一输出 |
| `configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_cpsc_2018.yaml` | `KEEP-MANUAL` | EfficientNet CPSC 2018+Extra 前瞻主线复现 | 同上，并保留合并逻辑中心身份 |
| `configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_georgia.yaml` | `KEEP-MANUAL` | EfficientNet Georgia 前瞻主线复现 | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_ningbo.yaml` | `KEEP-MANUAL` | ECGFounder Ningbo 前瞻主线复现 | 保留同一方法闭包、锁定 source checkpoint、中心身份和骨干特定常规优化参数 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_chapman_shaoxing.yaml` | `KEEP-MANUAL` | ECGFounder Chapman-Shaoxing 前瞻主线复现 | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_cpsc_2018.yaml` | `KEEP-MANUAL` | ECGFounder CPSC 2018+Extra 前瞻主线复现 | 同上，并保留合并逻辑中心身份 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_georgia.yaml` | `KEEP-MANUAL` | ECGFounder Georgia 前瞻主线复现 | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_eval_ningbo.yaml` | `KEEP-MANUAL` | 由 train result + RecipeSpec 绑定本次 EfficientNet Ningbo 主线 lineage 的 ref-excluded Clean/PN2021-C 评估 | 保留前瞻单中心 subject、train-result path/SHA、method-config 闭包、drop-all-zero 和 20-view 契约 |
| `configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_eval_chapman_shaoxing.yaml` | `KEEP-MANUAL` | 由 train result + RecipeSpec 绑定本次 EfficientNet Chapman-Shaoxing 主线 lineage 的评估 | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_eval_cpsc_2018.yaml` | `KEEP-MANUAL` | 由 train result + RecipeSpec 绑定本次 EfficientNet CPSC 2018+Extra 主线 lineage 的评估 | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_eval_georgia.yaml` | `KEEP-MANUAL` | 由 train result + RecipeSpec 绑定本次 EfficientNet Georgia 主线 lineage 的评估 | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_eval_ningbo.yaml` | `KEEP-MANUAL` | 由 train result + RecipeSpec 绑定本次 ECGFounder Ningbo 主线 lineage 的 ref-excluded Clean/PN2021-C 评估 | 保留前瞻单中心 subject、train-result path/SHA、method-config 闭包、drop-all-zero 和 20-view 契约 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_eval_chapman_shaoxing.yaml` | `KEEP-MANUAL` | 由 train result + RecipeSpec 绑定本次 ECGFounder Chapman-Shaoxing 主线 lineage 的评估 | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_eval_cpsc_2018.yaml` | `KEEP-MANUAL` | 由 train result + RecipeSpec 绑定本次 ECGFounder CPSC 2018+Extra 主线 lineage 的评估 | 同上 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_eval_georgia.yaml` | `KEEP-MANUAL` | 由 train result + RecipeSpec 绑定本次 ECGFounder Georgia 主线 lineage 的评估 | 同上 |
| `configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_matrix.yaml` | `KEEP-MANUAL` | EfficientNet 前瞻四中心 diagonal evaluation-result 聚合 | 保留 canonical 顺序、四个唯一外部结果路径、最小 matrix 配置闭包和唯一输出 |
| `configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_matrix.yaml` | `KEEP-MANUAL` | ECGFounder 前瞻四中心 diagonal evaluation-result 聚合 | 同上，并禁止跨骨干或跨 replicate 混合 |
| `configs/experiments/manual_refactor_pn2021_eval_effnet.yaml` | `KEEP-MANUAL` | 统一 launcher 的 PTB-XL EfficientNet source-registry canonical4 正式评估示例 | 新示例或注册表保留 source registry path/SHA、canonical 四中心精确顺序和 eval YAML 闭包 |
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
| `util/tests/test_data_contracts.py` | `KEEP-MANUAL` | 直接验证白名单缓存、tracked data-ledger/hash DAG、模型输入、100/500 Hz 线性插值、Super5/导联顺序及主线 YAML 闭包 | 新数据/模型边界接管相同内容账本、布局、插值顺序和 config closure |
| `util/tests/test_labels_super5.py` | `KEEP-MANUAL` | 直接验证白名单 PN2021 Super5 v7 映射、hash、NORM 抑制和 PTB-XL diagnostic class 转换 | 新标签层接管相同 mapping identity 和逐代码 golden policy |
| `util/tests/test_data_runtime.py` | `KEEP-MANUAL` | 验证唯一 data-load mmap 配置、拒绝 auto/RAM、raw100-only finite-plan public surface、split/cache 身份、K500 resident/mmap 等价、有界预取和跨 view session 所有权；并仅以临时小 fixture 锁 canonical ledger、迁移等价、missing/extra/size/same-size drift、symlink/no-clobber/TOCTOU 边界 | 新运行时/账本层接管相同 raw BTC、finite plan、split/ref-exclusion、seed、resident/mmap/session、有界预取，以及 quick inventory-only/full content 分级和扫描稳定性契约 |
| `util/tests/test_train_ptbxl.py` | `KEEP-MANUAL` | 验证两骨干 raw100 finite plan/fold/loader/input adapter、两层 thin boot CLI、五函数 model package surface、canonical-only factory 与共享 owner 的 dry-run/execute 委托 | 新 PTB-XL 适配层接管相同 raw BTC、`1000→5000 align_corners=True`、先插值后 z-score、配置单一来源、关闭语义、三份 direct JSON 与无 runtime/model alias override 契约 |
| `util/tests/test_method_graph.py` | `KEEP-MANUAL` | 验证四个文件选择器与唯一 matched-no-VAE slot 的精确 schema/resource closure、code-owned objective/requirements/scientific contract、路径无关 spec SHA、RNG golden、退役 latent selector 拒绝和动态 import 禁令 | 新 recipe 层接管相同有限集合、运行行为、资源身份、随机兼容与静态白名单契约 |
| `util/tests/test_online_trainer.py` | `KEEP-MANUAL` | 验证在线配置/参数闭包、两骨干 source checkpoint path/SHA 锁、五轮 rotating4、family-balanced BatchNorm、compact VAE encoder/decode signature，以及 exact lineage | 新 trainer 接管相同 source 身份、预算、调度、数据读取、BatchNorm、heldout-free last、lineage 和 VAE runtime 边界契约 |
| `util/tests/test_manual_run_experiment.py` | `KEEP-MANUAL` | 验证五入口 launcher 配置闭包、ledger quick gate/快照防篡改、40 YAML result 注册、dry-run/collision、run record/file index 和参数禁令 | 新 launcher 接管相同零副作用、数据/配置 TOCTOU 闭口、输出防覆盖、白名单参数与完整性契约，并保持配置引用与外部结果 artifact 隔离 |
| `util/tests/test_evaluation_metrics.py` | `KEEP-MANUAL` | 验证 raw-logit AUROC/AP、undefined-class 口径、drop-all-zero 过滤、equal-view→equal-center 和 canonical 四中心精确顺序 | 新指标层接管相同分数输入、宏平均分母、过滤、聚合与语义降级契约 |
| `util/tests/test_pn2021_evaluation.py` | `KEEP-MANUAL` | 验证三种互斥 subject、legacy/source bypass 拒绝、schema-v3 result、corruption 锁、canonical4、ECGFounder canonical100、真实 evaluator→recorder seal、派生指标篡改、checkpoint 字节复哈、config-owned cohort、共享 artifact helpers/re-export 与 artifact-only import 隔离 | 新评测与聚合层接管相同 subject lineage、checkpoint/schema、corruption、中心聚合、ref-exclusion、CPSC 合并、模型输入和无 Torch matrix import 契约 |

## B. 当前不属于手工主线的文件

### B1. 已退出当前树的历史支持参考

以下路径来自清理前父提交 `6a662c9`，当前工作树中并不存在；表内 `SUPPORT`
仅表示可按需从 Git 历史查阅的契约 oracle，不表示测试仍被 pytest 收集，也不
构成恢复 legacy wrapper 的授权。需要恢复某项能力时，应针对 A 区 clean API
重写最小测试，并把真实存在的测试移入 A9，而不是整批还原历史测试树。

| 历史路径 | 状态 | 历史作用 | 若按 clean API 重写必须保留 |
|---|---|---|---|
| `methods/augmix/ecg_ops.py` | `LEGACY-BRIDGE` | 旧 PN2021-C 调用入口以及临时 profile 兼容 | 调用方全部迁移到 `util/augmentations/`，历史回放测试通过 |
| `util/tests/test_pn2021_metadata.py` | `SUPPORT` | 验证独立 PN2021 表头解析语义及预处理入口不再导入旧数据层 | 表头解析器迁移时同步保留相同 golden contract 与 import 闭包检查 |
| `util/tests/test_preprocess_nonfinite.py` | `SUPPORT` | 验证 PTB-XL/PN2021 非有限值插值修复、边界填补与严重异常剔除契约 | 质量控制逻辑迁入最终数据层时同步迁移这些测试 |
| `util/tests/test_augmentations_cache.py` | `SUPPORT` | 验证 PN2021-C 白名单依赖、20组合、确定性随机流、双采样率及断点缓存契约 | 缓存入口迁移时同步迁移这些测试 |
| `util/tests/test_load_cache.py` | `SUPPORT` | 历史上验证三类缓存身份、已退出的自动 RAM/mmap 决策及 free-form 便利访问 | 仅按当前 mmap-only clean API 重写仍属活动契约的最小测试，不恢复 auto/RAM、单 hash、中心或 view-list 兼容面 |
| `util/tests/test_split_cache.py` | `SUPPORT` | 验证 PTB-XL 患者隔离、PN2021 确定性父 K500、独立 400/100、singleton train、CPSC 来源配额、互斥并集与 ref-exclusion | 切分层迁移时同步迁移这些契约测试 |
| `util/tests/test_models.py` | `SUPPORT` | 验证两个模型的输入输出、冻结范围、统一factory、外部参考state键及 EfficientNet逐元素前向对齐 | 模型包迁移时同步迁移官方/历史兼容契约测试 |
| `util/tests/test_model_input_adapter.py` | `SUPPORT` | 验证 canonical raw100 的 sanitize、EffNet identity、ECGFounder 设备内 linear 1000→5000、先升采样后 global z-score、非原地及严格输入契约 | 模型输入桥迁移时同步保留两模型数值参考和操作顺序回归测试 |
| `util/tests/test_canonical_corruption.py` | `SUPPORT` | 验证在线 GPU 腐蚀内核固定 `100→500→100`、20种 depth2/3 掩码、两链批量调用结构、确定性、非原地输入和逐样本 finite provenance | 腐蚀内核迁移时同步保留域、组合、批量执行和设备驻留契约 |
| `util/tests/test_vae_lhat_augmix.py` | `SUPPORT` | 验证真实 VAE、LHAT M候选几何/2B probe、attack-then-contract 标签边界保护与端点残差，以及 Stage-1 两链 Dirichlet/Beta 公式、确定性和白名单 import 闭包 | 在线训练核心迁移时同步迁移这些契约测试 |
| `util/tests/test_random_seed.py` | `SUPPORT` | 验证全局 seed YAML 身份、namespace 隔离、Python/NumPy/Torch 复现和 process 初始化 | 随机基础设施迁移时同步迁移这些契约测试 |
| `util/tests/test_config_bundle.py` | `SUPPORT` | 验证完整 configs 副本内的 seed/operator 引用、路径逃逸闸门和核心 YAML 加载 | 配置系统迁移时同步迁移这些契约测试 |
| `util/tests/test_supervised_trainer.py` | `SUPPORT` | 验证配置束、调用方 DataLoader、参数更新、普通 validation 选模、scheduler horizon、测试落盘及 last-epoch 微调 | 通用监督训练入口迁移时同步迁移这些最小契约测试；不恢复已退出的 raw prediction artifact 分支 |
| `util/tests/test_ptbxl_boot_scripts.py` | `SUPPORT` | 验证 EffNet/ECGFounder dry-run 的模型采样率、默认 full-FT profile 和显式 CLI 覆盖 | 两个 PTB-XL boot 入口迁移时同步迁移这些最小契约测试 |
| `util/tests/test_augmentation_profile.py` | `SUPPORT` | 验证离线缓存与在线 AugMix 共用同一算子参数、seed/config SHA 及 copied config bundle | profile loader 迁移时同步保留共享单一来源契约 |
| `util/tests/test_latent_pool.py` | `SUPPORT` | 验证 deterministic-mean latent、M20 eligibility、预计算 neighbor table 与 stable reference 逐元素一致、table SHA 稳定及 hash/index 访问 | latent pool 迁移时同步保留候选和身份契约 |
| `util/tests/test_train_pn2021.py` | `SUPPORT` | 验证raw100 K500/train400、canonical resident参数、worker安全门、partition进入seed身份，以及 pool-only encoder 与 runtime decoder requirements 的精确资源路由 | PN2021适配层迁移时同步保留数据、seed和方法资源边界 |

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
/home/linbinhao/miniforge3/envs/ECGTwin/bin/python \
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
| 2026-08-08 | 批次 4（ECGTwin Miniforge 环境迁移） | `/home/linbinhao/micromamba/envs/ECGTwin`；旧 Micromamba 根下的索引、压缩包和未使用 package cache | `/home/linbinhao/miniforge3/envs/ECGTwin`；保留 `/home/linbinhao/micromamba/envs/cli-tools`、Micromamba 本体和 shell 初始化 | 新旧三类 checkpoint smoke 摘要一致；GPU 2 smoke 通过；删除后 73 passed/10 skipped、`pip check`、受管 dry-run、9092 报告服务和 `cli-tools` Git 2.51.0 均通过；旧前缀不存在；Micromamba 根约 454 MiB | 用户于 2026-08-08 明确回复“好的接受，按第一种开始帮我执行” |
| 2026-08-08 | 批次 5（CLI 与默认 tmux clean switch） | `/home/linbinhao/micromamba/envs/cli-tools`；旧默认 tmux 3.6a server；旧 Micromamba 未使用 cache | `/home/linbinhao/miniforge3/envs/cli-tools`，tmux 3.7b + Git 2.51.0；重建 session 0/1 | 3.7b 隔离配置、Vim copy-mode、Git 和 shell 通过；最终 server PID 2087767 使用 Miniforge；Mihomo 7890/9090 正常；FRP 登录及 remote_ssh proxy 成功；旧前缀进程=0 且环境已删除；9091/9092 独立报告 server 保留 | 用户于 2026-08-08 明确回复“备用 SSH 已开，不依赖当前 FRP，允许重启 tmux、Mihomo 和 FRP” |
| 2026-08-11 | 批次 6（paper-kernel v2 Batch A） | 3 个 audit-only method profile 与 3 套未运行 matched E2 experiment/train/method 配置链，共 12 个 YAML、914 LOC；排序后 NUL 分隔路径集合 SHA256 为 `e4c5ed1c81baecf9e54573e2b0869ae14877093c5f7a8cffe3bb86a812c1f427` | 保留 A0、A3c、Direct fixed20、latent-threechain 与当前 AugMix-SimCLR + attack-then-contract LHAT 主线；历史由冻结 tag `manual-clean-development-pre-kernel-v2-20260811` 恢复 | active/evidence 引用=0、外部 snapshot=0；56 个 experiment YAML dry-run 通过；138 passed/10 skipped；删除路径引用=0；`git diff --check` 通过 | 用户于 2026-08-11 明确回复“确认删除” |
| 2026-08-12 | 批次 7（paper-kernel v2 CP1） | `util/tensorboard_logging.py`、`util/visualize_ecg.py`；同步移除 trainer/selector 调用、method probe 字段、两个零调用 raw-probe API 及 3 个训练配置的观察块；含证据身份测试后净减 1,411 physical LOC | 保留 checkpoint、JSON history/diagnostics/train result、selection/run manifest；历史配置 SHA 继续绑定 Git/run snapshot，仓库外既有 event/PNG/NPY/run 产物原样保留 | supervised/online 固定 CPU fixture 在 observer 开/关及删除后 state tensor、history、step 逐项相同；138 passed/10 skipped；56/56 experiment dry-run；删除符号引用=0；`git diff --check` 通过 | 用户已在接受的 CP1 重构范围中明确授权删除 |
| 2026-08-12 | 批次 8（paper-kernel v2 CP2） | Direct tuning/selection/refit 的 5 个 Python、2 个 train YAML 与 18 个 experiment YAML，共 25 路径、精确 gross deletion 3,305 physical LOC；排序后 NUL 分隔路径集合 SHA256 为 `4b0838976f6e24f4052859876d9c497b0bd4740392348421d5157e5f52885126` | Direct 仅保留历史数学原语、冻结注册表/决策日志与 8 个 fixed20 固定 checkpoint 评估 YAML；source checkpoint 最小锁移入 `core/train_PN2021.py`，受管面收口为四入口及 train/eval 两类结果 | 相对 CP1 提交 `d39eda1` 净减 3,474 physical LOC（1,054 insertions / 4,528 deletions）；140 passed/10 skipped；38/38 experiment dry-run；活动代码/配置的删除入口引用为 0，冻结历史账本路径字符串保留；训练 state/step/history 除有意移除的空 `validation: null` 字段外等价；recorder 的 PN/PTB/eval 结果类型 fail-closed；`git diff --check` 通过 | 用户已在接受的 CP2 重构范围中明确授权删除 |
| 2026-08-12 | 批次 9（paper-kernel v2 CP3） | `core/methods/executor.py` 与 `core/methods/nodes/` 下 7 个文件，共 8 路径、精确 gross deletion 524 physical LOC；排序后 NUL 分隔路径集合 SHA256 为 `d8666d17b0d0ce1e0c242ab195bd056b4444e7724dc854e9b0e2a965534653f1` | 五个 schema-v2 selector + code-owned finite RecipeSpec/直接 runtime；保留 CLI `--method-config`，主线 16 个 experiment 改用独立 `manual_refactor_paper_kernel_v2_recipe_v1_r1` 根 | 相对 `f9c6a78` 净减 1,940 physical LOC（1,598 insertions / 3,538 deletions）；153 passed/10 skipped；38/38 experiment closure/dry-run；旧 DAG 活动引用=0；`git diff --check` 通过。当前 tracked physical 37,770（runtime Python 25,418；test Python 3,504；config YAML 4,157；含 environments 的全部 tracked YAML 4,174），Phase 1 累计净减 6,825 行 | 用户已在接受的 CP3 重构范围中明确授权删除 |
| 2026-08-12 | 批次 10（paper-kernel v2 CP4 artifact lineage） | 无运行文件删除；收紧 PN2021 train-result/checkpoint/evaluation 身份链与评估入口 | schema-v1 `pn2021_train_result` + exact-lineage schema-v3 final checkpoint/history；schema-v3 evaluation subject 三模式：前瞻 train-result+method-config 单中心、legacy A0/Direct 显式 checkpoint 单中心、source-registry canonical4；`--train-result` 不进 YAML closure但锁 resolved path+SHA | 相对 smoke 登记提交 `82e568e` 净增 1,586 physical LOC（1,746 insertions / 160 deletions）；189 passed/10 skipped；38/38 experiment dry-run 且零 data/model/GPU side effect；16 个历史 A0/Direct eval YAML 字节不变；错中心、错方法、错 seed/checkpoint、空 view/aggregate 与污染 legacy protocol 均 fail-closed；`git diff --check` 通过。当前 tracked physical 39,489（runtime Python 26,275；test Python 4,237；config YAML 4,245） | 用户已在接受的 CP4 重构范围中授权继续执行；本批次不改变历史 smoke SHA 或 claim boundary |

| 2026-08-12 | 批次 11（paper-kernel v2 diagonal matrix） | 无运行文件删除；新增纯 JSON 四中心 diagonal 聚合面 | `aggregate_pn2021` + 最小 matrix config + 两骨干 experiment；只接受四个 prospective 单中心 schema-v3 result，并从成员 clean/per-view 重算等权指标 | 相对 `bca67ea` 净增 726 physical LOC（737 insertions / 11 deletions）；203 passed/10 skipped；40/40 experiment dry-run 且零 data/model/GPU 加载；独立 contract review APPROVE；wrong order/duplicate/model/recipe/replicate/source/eval-config/split drift 均 fail-closed；`git diff --check` 通过。当前 tracked physical 40,215（runtime Python 26,633；test Python 4,505；config YAML 4,321；含 environments 的全部 tracked YAML 4,338） | 用户已在当前 CP4 后续执行范围中授权继续；未运行的矩阵配置不构成性能或论文证据 |
| 2026-08-12 | 批次 12（paper-kernel v2 portable artifact contract） | 无运行文件删除；将 PN2021 lineage/train/evaluation/checkpoint seal 收敛到纯 stdlib 公共合同，并移除 recorder/evaluator/matrix 重复浅校验 | `util/pn2021_artifact_contract.py` + config-owned matrix cohort；保留三 subject 隔离、实际 checkpoint 字节复哈、K500/ref-exclusion、canonical metric aliases/20 corruption signatures 与派生 aggregate 重算 | 相对 `35e2d9c` 净增 72 physical LOC（1,090 insertions / 1,018 deletions）；生产 Python净增 70 行、测试 Python净减 85 行、配置 YAML净增 80 行；199 passed/10 skipped；40/40 experiment dry-run 且零 data/model/GPU side effect；独立恶意 probe 对 legacy/source 降级、coherent identity/metric/composition 篡改、checkpoint byte drift 与整体错 cohort 全部 fail-closed；matrix import 不加载 Torch/PN2021/model checkpoint；`git diff --check` 通过。当前 tracked physical 40,287（runtime Python 26,703；test Python 4,420；config YAML 4,401；含 environments 的全部 tracked YAML 4,418） | 用户已在当前 CP4 后续执行范围中授权继续；本批次不新增性能或论文证据 |
| 2026-08-12 | 批次 13a（data content ledger tool） | 无运行文件删除；新增确定性数据内容账本生成/校验工具及临时小 fixture 契约 | `data_preprocess/data_ledger.py`；保留活动四根的 code-owned 解析、canonical JSONL、quick inventory-only/full content 分级、原子 no-clobber 与 metadata 可见扫描变化拒绝 | 相对 `220ad5b` 净增 595 physical LOC（597 insertions / 2 deletions；生产 Python +394、测试 Python +179）；206 passed/10 skipped；实根只读 inventory=121 files/404,751,718,887 B；独立 tiny probes APPROVE；`git diff --check` 通过。当前 tracked physical 40,882（runtime Python 27,097；test Python 4,599；config YAML 4,408；含 environments 的全部 tracked YAML 4,425）。尚未生成或跟踪真实 ledger，尚未接入 launcher/run recorder，`data_cache_payload_bytes_not_content_locked` 限制继续成立 | 用户已授权继续下一步；本子批次不新增性能、数据内容或论文证据 |
| 2026-08-12 | 批次 13b（managed data content seal） | 无运行文件删除；将 121 成员 ledger 接入 data-load descriptor、受管 launcher 与 run recorder | `configs/data/data_content_ledger_v1.jsonl` + pre-run quick inventory/size gate + immutable YAML/ledger snapshots；保留 offline full seal 与 runtime quick 的证据分级 | 相对 `6722754` 净增 780 physical LOC（817 insertions / 37 deletions；生产 Python +263、测试 Python +301、tracked ledger +122）；两遍独立 full-content pass 一致锁定 404,751,718,887 bytes / SHA256 `d3bf1f18…af665`；225 passed/10 skipped；40/40 dry-run；独立 evidence/integration review APPROVE；`git diff --check` 通过。当前 tracked physical 41,662（runtime Python 27,360；test Python 4,900；config YAML 4,473；全部 tracked YAML 4,490） | 用户已授权继续下一步；本批只封存 prospective managed data bytes，不新增性能或论文证据 |
| 2026-08-12 | 批次 13c（runtime characterization checkpoint） | 无生产路径删除；新增 tiny mmap/resident/session 实现行为特征测试 | 提交 `5cfdebd`；为后续 finite LoaderPlan 收缩锁定 batch gather、seeded order、mmap close 与 sequential session 生命周期 | 相对 `1167c2b` 净增 358 physical LOC（360 insertions / 2 deletions）；21 个 data-runtime 定向契约通过；`git diff --check` 通过。当前 tracked physical 42,020（runtime Python 27,360；test Python 5,258；config YAML 4,473；全部 tracked YAML 4,490） | 用户要求分多次 commit 并继续下一步；本 checkpoint 不新增性能证据 |
| 2026-08-12 | 批次 13d（managed argv schema checkpoint） | 无运行文件删除；收紧 5 个 code-owned entrypoint 的参数签名与 execution-time YAML 绑定 | 提交 `588a2cc`；保留唯一 e10 灵敏度参数、三种 eval subject 与 four-result matrix 签名 | 相对 `5cfdebd` 净增 297 physical LOC（307 insertions / 10 deletions）；246 passed/10 skipped；40/40 dry-run；参数注入、同形 retarget、YAML drift 与 mutable-list TOCTOU 均在 ledger/run-dir/delegate 前拒绝；`git diff --check` 通过。当前 tracked physical 42,317（runtime Python 27,436；test Python 5,479；config YAML 4,473；全部 tracked YAML 4,490） | 用户要求分多次 commit 并继续下一步；outer dry-run 仅审计闭包/命令，不执行 delegate |
| 2026-08-12 | 批次 14（finite RuntimeLoaderPlan） | 退出 public free-form loader/config API 及 PTB/PN/eval boot 的 66 个 runtime/training override 声明；断开 active runtime 对 1,158 行 `split_cache.py` builder 的 import | `PTBXLLoaderPlan`、`PN2021K500LoaderPlan`、`PN2021EvaluationLoaderPlan` + private `_LoaderRequest` engine；统一 raw physical-mV 100 Hz BTC，ECGFounder H2D 后 linear `1000→5000` 再 z-score | 相对 `588a2cc` 净增 617 physical LOC（2,273 insertions / 1,656 deletions），但生产 Python 净删 594 行（其中 `data_runtime.py` 净删 163）；测试 Python 净增 1,129 行；320 passed/10 skipped；40/40 outer dry-run 且零 data/model/GPU；直接 boot dry-run 在 model/data/GPU 前构造并校验 finite plan；独立 science/closure review APPROVE；旧 generic 生产引用=0；`split_cache.py` blob 保持 `d90bc78c…` 未改；`git diff --check` 通过。当前 tracked physical 42,934（runtime Python 26,842；test Python 6,608；config YAML 4,523；全部 tracked YAML 4,540） | 用户已授权继续工作并要求按逻辑分批 commit；本批只改 prospective runtime，不重写历史 checkpoint/metric 证据；outer launcher dry-run 仅审计 YAML 闭包/命令，不执行 delegate |
| 2026-08-12 | 批次 15（raw100 runtime slim） | 删除零生产调用的 runtime transform/sanitize/z-score/layout API、selection-overlap helper、3 个隐式 config default、冗余 plan `config_root` 与 22 个低层 public export | 模型输入变换只由 `models/input_adapter.py` 拥有；data runtime 只输出 raw100 BTC，并仅公开三种 finite plan、`RuntimeDataLoader` 与 sequential evaluation session | 相对 `6756bc8` 净减 253 physical LOC（55 insertions / 308 deletions）；生产 Python净删 217 行、测试 Python净删 37 行；321 passed/10 skipped；40/40 outer dry-run 且零 data/model/GPU；独立 science/surface review APPROVE；旧符号引用=0；`git diff --check` 通过。当前 tracked physical 42,728（runtime Python 26,625；test Python 6,607；config YAML 4,534；全部 tracked YAML 4,551） | 用户已授权继续并要求分批 commit；本批不改变数据、split、训练/评测配置或历史 evidence，只移除重复 model-input 责任 |
| 2026-08-12 | 批次 16（evaluation test dedupe） | 删除测试内平行实现的 fake evaluation plan/session；测试节点与参数化集合不变 | evaluator 测试仅替换 private loader builder，真实走 `PN2021EvaluationLoaderPlan`、`SequentialEvaluationDataSession` 与 request；runtime tiny mmap 测试走 clean + 两个 corrupted loader 的完整 open/iterate/close 链 | 相对 `e4bc834` 净减 96 physical LOC（71 insertions / 167 deletions），测试 Python净删 97 行；321 passed/10 skipped；独立 coverage/reviewer APPROVE；四中心 84 loader、单中心 21 loader、两 cache/selection 复用及 mmap 最终关闭继续锁定；`git diff --check` 通过。当前 tracked physical 42,632（runtime Python 26,625；test Python 6,510；config YAML 4,534；全部 tracked YAML 4,551） | 用户已授权继续并要求分批 commit；本批只收敛测试 fixture，不改变生产、配置、数据或 evidence |
| 2026-08-12 | 批次 17（post-CP4 CUDA diagnostic smoke registration） | 无运行文件删除；登记当前 finite raw-100-Hz、content-ledger 与 RecipeSpec CUDA 迁移诊断 | 保留旧 `dfd00ec` smoke、`68c120f` 失败现场和历史性能证据原样；新增 clean `929c05e` 下 Direct EfficientNet、mainline EfficientNet、mainline ECGFounder 三个 Ningbo K500 单轮产物的 summary/file-index/final-checkpoint SHA | 相对 `929c05e` 净增 82 physical LOC（93 insertions / 11 deletions）；322 passed/10 skipped；40/40 outer dry-run；69 个外部 run 文件、三份 file index/checkpoint、配置与 ledger 快照独立复哈通过；科学审计锁定 Direct 21 views、mainline 6 views、Stage-1 两步和三个 LHAT 诊断作用域；`git diff --check` 通过。当前 tracked physical 42,714（runtime Python 26,625；test Python 6,553；config YAML 4,571；全部 tracked YAML 4,588）；claim 仅为 `migration_smoke_only_not_performance_evidence` | 用户报告 GPU 已空并授权继续；本批不登记吞吐或指标，不构成完整预算复现、性能或论文证据 |
| 2026-08-13 | 批次 18（online observer pipeline slim） | 删除 `_SampledStepTimer`、observer exclusion、sampled objective host-copy、step/epoch `diagnostics.jsonl` 镜像、performance 聚合及对应 PN2021 配置；不删除历史运行产物 | `training_history.json` 唯一保留 epoch-level objective/count/exposure、LHAT 三作用域、ASR/acceptance/BCE gain、distribution/rate、stochastic trace、quality/ineligible 证据；checkpoint/result/method-resources 输出不变 | 相对 `01b5a9c` 净减 533 physical LOC（25 insertions / 558 deletions）；生产 Python净删 525 行，测试 Python净增 2 行，config YAML净删 11 行；322 passed/10 skipped；40/40 outer dry-run；旧 timer/step/performance 生产引用=0；live matrix training-config SHA 已级联为 `3ed5b267…7acc0`；`git diff --check` 通过。当前 tracked physical 42,181（runtime Python 26,100；test Python 6,555；config YAML 4,560；全部 tracked YAML 4,577） | 用户在 2026-08-13 的 12 小时精简 goal 中明确授权本批；不改 loss、RNG、optimizer、BN、recipe、model、data 或历史 evidence |
| 2026-08-13 | 批次 19（latent-threechain executable island retirement） | 删除 `configs/train/methods/exp_paired_augmix_latent_bridge_v1.yaml` 及 code-owned latent-threechain kind/objective/resource/runtime/BN/exposure 链；从 `core/augmix.py` 删除零活动调用的 waveform three-chain、latent three-chain 与 Bernoulli-JSD 岛 | 保留四个文件选择器和唯一 matched-no-VAE slot；主线仍用 frozen VAE encoder 构建 K500 latent pool，以 decoder 执行 attack-then-contract LHAT，Stage-1 仅保留冻结两链 AugMix-SimCLR | 相对 `1cf965d` 净减 983 physical LOC（62 insertions / 1,045 deletions）；生产 Python净删 889 行、测试 Python净删 33 行、config YAML净删 61 行；317 passed/10 skipped；40/40 outer dry-run；退役符号生产引用=0；Stage-1 13 项逐张量 SHA256 与删除前完全一致；主线 selector/spec SHA 仍为 `eab7f244…2036` / `de3d6948…7602`，live AugMix SHA 级联为 `51e7ebb6…b00f`；`git diff --check` 通过。当前 tracked physical 41,198（runtime Python 25,211；test Python 6,522；config YAML 4,499；全部 tracked YAML 4,516） | 用户在 2026-08-13 的 12 小时精简 goal 中明确授权本批 exact deletion map；不改 loss、RNG、optimizer、BN、主线 recipe、model、data 或冻结历史 evidence |
| 2026-08-13 | 批次 20A（artifact utility ownership） | 无运行文件删除；删除 evaluator/matrix/recorder 的重复 JSON loader、SHA、path/SHA builder、artifact resolver、PN2021 映射/中心/导联常量及零调用 public `snapshot_yaml_files` wrapper | `util/pn2021_artifact_contract.py` 唯一拥有纯 stdlib JSON/SHA/reference/resolver 与 PN2021 artifact 身份；evaluator 保留 `PN2021_*` aliases 并从 cache owner 复用 `EXPECTED_LEADS`；recorder 保留 `sha256_file` re-export 与私有 config snapshot 生命周期 | 相对 `8b02f63` 净减 50 physical LOC（79 insertions / 129 deletions）；生产 Python净删 68 行、测试 Python净增 17 行、文档净增 1 行；focused 115 passed；shared helper/re-export、relative owner resolution、matrix artifact-only import 与 `snapshot_yaml_files` 退出均受现有节点约束；`git diff --check` 通过。当前 tracked physical 41,148（runtime Python 25,143；test Python 6,539；config YAML 4,499；全部 tracked YAML 4,516） | 用户在 2026-08-13 的 12 小时精简 goal 中授权 Batch20 A；本批不改 protocol、数据、模型、训练、指标、配置或 evidence，不新增测试节点 |
| 2026-08-13 | 批次 21A（compact deterministic VAE boundary） | 无运行文件删除；删除 VAE encoder 的 noise/sample/generator 随机分支和 decoder helper 的 target-points/5000点分支，同步 latent-pool 与 LHAT 三处调用 | encoder 固定返回 deterministic scaled posterior mean、mean、logvar；decoder 固定输出 PTB-XL 顺序 canonical raw `(B,1000,12)`，ECGFounder 5000点适配继续唯一属于后续 model attack/input bridge | 相对 `ff81eff` 净减 33 physical LOC（15 insertions / 48 deletions）；生产 Python净删 37 行、测试 Python净增 3 行、文档净增 1 行；现有 28-node focused 全绿；compact signatures 与无旧调用引用受现有节点约束，独立真实 checkpoint 对照证明 deterministic mean/canonical1000 输出逐位一致；`git diff --check` 通过。当前 tracked physical 41,115（runtime Python 25,106；test Python 6,542；config YAML 4,499；全部 tracked YAML 4,516） | 用户在 2026-08-13 的 12 小时精简 goal 中授权 Batch21 A tuple 低风险方案；本批不改 YAML、registry、其他 model、训练 protocol、指标或 evidence，不新增测试节点 |
| 2026-08-13 | 批次 21B（model surface and PTB-XL boot ownership） | 无运行文件删除；package re-export 由 26 项收口为五函数，factory 删除四 alias/normalize/三个公开 registry，两个 PTB-XL boot 删除重复 profile/model/plan/JSON/train 业务 | `models.factory` 私有合并 registry；`core/train_PTBXL.py` 唯一拥有两骨干有限 boot 业务；原两脚本仅保留路径、parser、固定 selector，Founder 仍仅允许 `--epochs=10` | 相对 `4a175b8` 净减 114 physical LOC（116 insertions / 230 deletions）；生产 Python净删 122 行、测试 Python净增 7 行、文档净增 1 行；focused 58 passed；三份 direct boot JSON（EffNet、Founder默认、Founder E10）改前改后逐字节一致；package surface/alias 退出与 single-owner monkeypatch 受现有节点约束；`py_compile` 与 `git diff --check` 通过。当前 tracked physical 41,001（runtime Python 24,984；test Python 6,549；config YAML 4,499；全部 tracked YAML 4,516） | 用户在 2026-08-13 的 12 小时精简 goal 中授权 Batch21 B；本批不改 YAML、checkpoint/state、模型架构、训练科学合同、指标、registry 或 evidence，不新增测试节点 |
| 2026-08-13 | 批次 22（core package boundary） | 无底层实现文件删除；`core/__init__.py` 与 `core/methods/__init__.py` 收为 docstring + 空 `__all__`，删除全部 eager barrel re-export | 三个生产消费者与相关测试直接导入 `core.methods.contracts` / `registry` / `runtime` owner；保留底层训练、recipe 与 runtime 实现不变 | 相对 `3969074` 净减 160 physical LOC（22 insertions / 182 deletions）；生产 Python净删 161 行、测试 Python 行数不变、文档净增 1 行；focused 93 passed；两个 package 空导出面、owner-module direct imports 与旧 barrel 符号退出受现有节点约束；`py_compile` 与 `git diff --check` 通过。当前 tracked physical 40,841（runtime Python 24,823；test Python 6,549；config YAML 4,499；全部 tracked YAML 4,516） | 用户在 2026-08-13 的 12 小时精简 goal 中授权 Batch22 barrel audit 方案；本批不改 YAML、算法、底层实现或 evidence，不增加测试节点 |
| 2026-08-13 | 批次 23A（mmap-only data boundary） | 无数据或配置删除；删除 cache auto/RAM 决策与实时内存分支、三个零调用 cache 便利方法、augmentation operator snapshot wrapper 和 dict fallback，并合并评估 cache-mode 校验 | `load_cache` 与三种 finite plan 仅接受只读 mmap；K500 selection-resident gather、批量 hash、显式 corruption view、batch mmap/prefetch 与 sequential evaluation session 保持；PN2021-C cache builder 仅接受唯一严格 `AugmentationProfile` | 相对 `3c1fd5d` 净减 122 physical LOC（70 insertions / 192 deletions）；生产 Python净删 157 行、测试 Python净增 34 行、文档净增 1 行；focused 174 passed；非 mmap 在 manifest/shared-cache 前拒绝、临时 NPY 只读 memmap、旧符号退出受现有节点与引用扫描约束；`py_compile` 与 `git diff --check` 通过。当前 tracked physical 40,719（runtime Python 24,666；test Python 6,583；config YAML 4,499；全部 tracked YAML 4,516） | 用户在 2026-08-13 的 12 小时精简 goal 中授权继续 low-risk finite runtime 收敛；本批不改 batch-read、layout、resident、split/cache 字节、YAML、算法或 evidence |
| 2026-08-13 | 批次 24（locked evaluation input contract） | 无运行文件删除；将 `_validate_input_pipeline` 的逐层 mapping/key/literal 重复校验收敛为单一 code-owned `_LOCKED_INPUT_PIPELINE` exact dict 比较 | 保留 canonical raw100、两骨干 model domain、sanitize、global z-score、channel-time float32，以及 align-corners、epsilon、variance-correction 三组 exact type gates；`_resolved_input_pipeline`、plan/result、metrics、subject 与 YAML 不变 | 相对 `491bea3` 净减 81 physical LOC（52 insertions / 133 deletions）；生产 Python净删 101 行、测试 Python净增 19 行、文档净增 1 行；focused 115 passed；747-case old/new mutation 差分 acceptance/error-class mismatch=0，protected evaluator AST 与 config/metrics/artifact contract 保持不变；`py_compile` 与 `git diff --check` 通过。当前 tracked physical 40,638（runtime Python 24,565；test Python 6,602；config YAML 4,499；全部 tracked YAML 4,516） | 用户在 2026-08-13 的 12 小时精简 goal 中授权继续 low-risk evaluation validator 收敛；本批不改 plan/result/YAML/metric/subject、数据、模型、算法或 evidence |

`methods/augmix/ecg_ops.py` 与 `methods/augmix/severity.py` 暂不放进候选批次 1：
白名单已经切断依赖，但旧 `ecg_adv_gen/` 与 `methods/` 树内部仍相互引用。
它们应在后续“整棵旧运行树”批次中一起删除，避免暂时留下明显断裂的旧树。

## F. 当前下一步

- [x] 生成并独立复核四个活动派生 cache/split 根的 tracked content ledger；
  数据型受管执行在建 run dir 前做 exact inventory/size quick gate，并从不可变
  config/ledger 快照启动。该 seal 不覆盖 raw WFDB 或 preprocessing correctness。

- [x] 将 PN2021 `.hea` 的 SNOMED/人口学解析迁入独立白名单模块，移除
  `PN2021_preprocess.py` 对旧 `ecg_adv_gen.data` 包的运行时依赖，并锁定原有
  解析语义。
- [x] 为两个数据预处理脚本补齐非有限值质量控制的独立最小测试。
- [x] 实现 PTB-XL、PN2021、PN2021-C 的统一只读缓存加载与访问接口。
- [x] 实现 PTB-XL 官方折及 PN2021 四逻辑中心 K500/ref-excluded 的 ID-only 切分层，并将 CPSC 2018/Extra 合并为一个逻辑中心。
- [x] 实现从受管 split/cache 到 raw100 model-adapter boundary 的统一运行时数据层；以三种有限 LoaderPlan 接管 PTB-XL/K500/evaluation，保留显式 corruption view、worker-safe Dataset、确定性 DataLoader 和明确 loader/session ownership。
- [x] 建立独立 `models/` 包，完成 EfficientNet1DV2、ECGFounder、严格checkpoint身份和统一 `build_model` 接口，并通过真实checkpoint CPU smoke test。
- [x] 独立重写 ECGTwin VAE encoder/decoder，并在同一白名单模块中实现 exact-label VAE-LHAT、攻击后标签边界收缩和 Stage-1 两链 AugMix-SimCLR 受管 YAML。
- [x] 新增 `configs/train/PTBXL.yaml` 和 `core/supervised_trainer.py`；调用方仅通过三种 finite LoaderPlan 打开受管 PTB-XL/PN2021 DataLoader，再由通用训练器接受 model + DataLoader，支持验证集 macro-AUPRC 选模、最终测试与无验证集 last-epoch 微调。
- [x] 新增 `core/train_PTBXL.py` 作为白名单 PTB-XL 数据适配层，并新增 EffNet/ECGFounder 两个 `boot_scripts/` 薄入口；两骨干统一读取 raw physical-mV 100 Hz BTC，EffNet 保持1000点，ECGFounder在输入设备上 `linear 1000→5000 align_corners=True` 后全局 z-score，boot CLI 收敛为有限 launcher-owned 面。
- [x] run-scoped TensorBoard/`ecg_plot` 观察层曾用于开发审计，已在 paper-kernel v2 CP1 退役；仓库外历史 event/PNG/NPY/manifest 原样保留，当前运行面只写 checkpoint、`training_history.json`、`train_result.json` 与 `method_resources.json`。
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
  对抗攻击已迁移到新入口。公平对比的各 finite recipe 必须共享
  comparison-group/replicate seed，
  每个运行记录基础 seed、有效 seed、namespace 和 seed YAML SHA256；迁移后
  删除了增强子包中的旧实现，不保留旧拼写兼容入口。
- [x] 增加 configs-shaped 配置束解析：复制整个 `configs/` 后，内部 YAML
  引用优先解析副本自身；finite loader plan 接收已解析的显式 YAML 路径，模型工厂支持 `config_root`，各独立
  预处理/缓存/LHAT/AugMix 入口支持显式副本 YAML 路径。
- [x] 新训练器/统一实验 launcher 只接受一个 experiment YAML 加
  `config_root`，启动前解析并保存依赖 YAML 闭包、SHA256、seed identity、
  Git 身份和精确命令；禁止重新引入另一套隐式默认配置搜索规则。
- [x] 建立完整 K500 latent pool 和 A0/LHAT/主线匹配在线训练层；候选不足
  M20 的 anchor 不丢弃、不重复填充，只贡献 clean loss并落盘清单。
- [x] 将主线收敛为四个文件选择器与唯一 code-owned matched-no-VAE slot：YAML 只保留固定身份与
  外部资源引用，Stage-1 在 trainer 内使用 K500 两链 AugMix-SimCLR，Stage-2
  直接调度 clean、rotating4 corruption 与 attack-then-contract LHAT view；
  梯度累计后每个 base batch 只更新一次，不开放任意 DAG/node/auxiliary 组合。
- [x] 将 `agent_workspace` 中锁定的精简开发 recipe 迁入白名单运行面：
  `augmix_simclr_lhat.yaml`、`PN2021.yaml`、统一 launcher 示例、Stage-1
  checkpoint/epoch-level training history、VAE contract 诊断和配置/seed/resource 身份均已接管；
  仍标记为 prospective replication，未因代码迁移自动升级为论文最终证据。
- [x] 在 `util/evaluation/` 建立 PN2021 clean/PN2021-C 正式评估层：前瞻和
  legacy adapted subject 仅评估其单一目标中心，source-registry subject 才评估
  canonical4；schema-v3 result 输出 kept/drop、per-class/macro、depth2/3/23、
  per-center 和等权指标，并锁定 split/cache/checkpoint/subject lineage。
- [x] 已将 Direct train400/validation100、四中心 pooled E* 与 full-K500 refit
  结果冻结为历史证据；EffNet 为 E23/T30，ECGFounder 原始 Direct 为 E20/T20。
  CP2 已退出其 tuning/selection/refit 受管重放代码，`PN2021.yaml` 不表示 Direct
  的精确 replay，保留的 fixed20 YAML 仅评估冻结 checkpoint。
- [x] ECGFounder latent-threechain 的 E19/T20、四中心正式评估及 LR `3e-5`
  matched Direct 归因控制已冻结进白名单注册表和决策日志；权重 SHA 与方法增益
  边界继续作为历史证据，不由当前四入口 launcher 重新选择或 refit。
- [x] 在保留源 worktree 的前提下生成 A 区以外 449 个 Git 跟踪项的精确
  候选集合，并记录排序清单 SHA256。
- [x] 用户确认迁移后，仅在新干净 worktree 执行裁剪；源 worktree 的历史
  文件、未跟踪实验和外部模型目标均未删除。
