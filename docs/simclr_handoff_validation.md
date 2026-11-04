# Traditional SimCLR 交接验证记录

日期：2026-11-04。交付分支 `direction/traditional-simclr`，主入口见
[README](../README.md)。本记录区分代码验证、历史结果与尚未恢复的实验。

## 本次修复与方法身份

在线 PN2021-C 评估先前将“除波形外的 ledger 成员总数等于十”作为条件，
新增的 `RETIREMENT_20261017.md` 导致三套 K500 新账本被误拒绝。
现在要求十个指定路径全部存在并验证每个文件的大小与完整 SHA256；
额外 ledger sidecar 单列为未验证，不读取或认证已退役的波形数组。
缺失、换名、内容篡改、符号链接、错误账本 SHA 继续拒绝。
原 `verify_ledger` 的严格 inventory/full 检查未放宽。

主方法仍是完整两阶段 `augmix_simclr_lhat`，selector 文件 SHA256
`f08d8555b1591021c85c19219d5284f2a0321bf0d5829955fac8b796e73b35d6`，
recipe spec SHA256
`6245e89f572b49bd24f5d688bda7d3f8b16fa4972051353afa311d007cd9f675`。
已有 74 个配方的冻结 characterization 测试通过；独立 joint 配方单独验证，
没有替换默认方法。

## 已完成的检查

- 元数据回归：7 passed，覆盖 sidecar 在旧/新 ledger 中的情况、已退役波形、
  必需路径缺失/换名、同大小篡改、符号链接、错误 ledger SHA。
- 真实原账本与 draw0/1/2 新账本：四套均对十个必要元数据进行完整哈希验证，
  新账本明确记录归档说明为未验证 sidecar。
- 两个骨干的四个交接 YAML：通过真实 CLI `--dry-run`，无数据/模型/GPU 加载、
  无输出目录创建；训练与评估 `--train-result` 精确配对，均使用锁定主配方。
- GPU 0 固定随机流检查：1 passed；GPU 在线评估的 Founder 128 / EfficientNet
  512 batch 得到完全一致的固定 32-record 腐蚀流，未改全局 CUDA RNG。
- 完整 CPU suite：**644 passed、15 skipped、9 failed**，172.04 秒。
  九项失败与此前对未修改 baseline 的结果逐项一致，没有新增失败。
- README/导航本地链接、配置闭包、`git diff --check` 在提交前另行检查。

CPU XML 与四份 dry-run 记录位于外置目录：
`/home/linbinhao/ECG_adv_data/reports/traditional_simclr_handoff_20261104_r0/`。
失败集合对照基准：
`/home/linbinhao/ECG_adv_data/runs/simclr_joint_novae_20261104_r0/baseline_existing_failures.xml`。

## 真实 GPU 评估

此前失败的同一份 no-VAE、EfficientNet、Ningbo、draw0 checkpoint 已通过
独立 managed run 完成 clean + 20 个 severity-5 depth2/3 视图评估：

`/home/linbinhao/ECG_adv_data/runs/traditional_simclr_fixcheck_20261104_r0/`。

父 manifest 为 complete、exit 0；run file index 无差异；评估记录包含十项
元数据完整验证、sidecar 未验证声明、在线 GPU backend、固定共同 cohort。
这是修复入口的完整执行验证，不是四中心/三抽样实验的完成声明。

完整 VAE-LHAT 主方法的已训练 checkpoint 另在
`/home/linbinhao/ECG_adv_data/runs/traditional_simclr_fixcheck_full_20261104_r0/`
完成同样的 clean + 20 视图验证，父 manifest complete、exit 0、file index 无差异。
两个结果均通过独立的聚合重算检查，主方法仍为原锁定 recipe spec。

| 修复验证 | `evaluation/evaluation_result.json` SHA256 |
| --- | --- |
| 两阶段 no-VAE | `7300869ccd4b3e4ee961b16edb8568e371e3fc6468a6133b23afc0aabe01ddce` |
| 两阶段完整版 | `cc5a0fb9c4cc91ef2b74bbb85db7f120f3090ef2e841f12454bdb9075adb398d` |

两次均在独立目录串行使用当时空闲的物理 GPU 0，未重新训练。
原目录的 0/48 评估计数保持原失败历史，以上两份成功评估属于独立修复验证。

## 基线失败与环境限制

以下九项失败保留，未修改无关方向或放宽安全检查：

| 测试 | 原因 |
| --- | --- |
| `test_dual_jsd::test_visual_configs_and_finite_grid` | 历史 run 已归档到 `/data`，旧路径边界拒绝 |
| `test_ecg_image_llm::test_image_llm_config_and_dry_run_contract` | 外置模型归档路径不在旧 `/home` 边界 |
| `test_ecg_image_llm::test_two_gpu_placement_is_only_a_full_precision_llama_smoke` | 同上 |
| `test_pn2021_corruptions::test_cache_staging_persists_only_the_canonical_100hz_waveform` | 旧 cache-build fixture 使用真实磁盘余量，低于其 reserve；未创建缓存 |
| `test_pulse_benchmark::test_managed_pair_config_closure[admission]` | 历史训练结果归档到 `/data`，旧路径边界拒绝 |
| `test_pulse_benchmark::test_managed_pair_config_closure[full]` | 同上 |
| `test_pulse_finetune::test_resume_config_preserves_the_uninterrupted_training_protocol` | 历史 resume checkpoint 的归档路径越过旧边界 |
| `test_pulse_pixel_mixing::test_fixed_workflow_has_three_admissions_eight_trains_and_one_evaluation` | 同上 |
| `test_pulse_training_queue::test_queue_closure_and_exact_paired_jobs` | 历史 PULSE artifact 归档路径越过旧边界 |

本机 `python -m pip check` 另外报告：
`imagecorruptions 1.1.2 requires opencv-python, which is not installed.`
该额外包不在交接 pip lock 中，传统路径源码没有导入它。本次保持环境不变；
新机器仍需独立完成依赖检查、CPU 检查与获授权 GPU 上的准入验证。

## 原 K500 实验的交接边界

原目录 `runs/k500_sampling_20261104_r0` 中，full/no-VAE 两份
EfficientNet/Ningbo/draw0 训练已完成；首次评估失败、父队列退出，
原队列计数为 **2/48 train、0/48 eval**。上述修复验证使用独立新目录，
未修改原失败状态、结果或 source snapshots，也没有生成最终 sampling report。

保留的工程准入绑定旧源码。完整恢复需新的明确恢复配置及新源码准入，
并核验旧训练 checkpoint 的复用范围；不能改写旧 admission/hash 来放行。
不重启原队列、不覆盖失败目录、不把短验证当成全部实验完成。
