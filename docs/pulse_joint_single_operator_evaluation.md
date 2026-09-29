# PULSE 单算子波形–图像联合评估

本轮替代未完成的 `joint_v2` 全量任务；旧输出保留在原目录，不与新结果混用。

每条记录固定 24 个条件：

- 1 个 clean；
- 5 个单波形扰动：`powerline_noise`、`emg_noise`、`baseline_wander`、`baseline_shift`、`random_leads_masking`；
- 3 个单图像采集扰动：`perspective`、`low_resolution`、`illumination`；
- 15 个 5×3 的单波形–单图像联合条件。

不再使用波形 depth-2/depth-3 链。四个模型臂仍为 original、clean、single、three，
每中心使用固定非 K500 cohort 的 128 条记录。该评估仍是 PN2021 development
validation，不是独立外部测试；图片在线生成，预测与契约文件写入
`/home/linbinhao/ECG_adv_data/runs/pulse_joint_singleop_20260918`。
