# PULSE full LoRA evaluation, September 28

This run evaluates the completed rank-32 single-chain and three-chain models on
all four PN2021 centers, excluding each center's 500 LoRA adaptation records.
It reuses the eight completed 200-step training results in the pinned
`pulse_full_lora32_pipeline_20260927_recovery4gpu` archive; it does not retrain.
The archive's integrity receipt is verified, while its earlier overall run is
marked failed/interrupted because full evaluation did not finish. Only its eight
complete training results are reused; this evaluation gets a new output identity.

The fixed evaluation contains clean plus five GPU image corruptions at severity
5, with original, single-chain, and three-chain predictions on the same records.
There are 39,879 records total: Ningbo 18,727, Chapman/Shaoxing 5,322,
CPSC 2018 including Extra 7,619, and Georgia 8,211. The result is development
evidence, not an independent test claim.

The workflow first performs a four-center engineering smoke and token-equivalence
check. Because the inference adapter changed after the archived run, this repeats
only the four-step smoke training/resume checks and four-record evaluation checks;
it does not repeat the eight full 200-step trainings. It then schedules 512-record blocks
across up to four devices, with inference batch size 1. A worker is started only
on a device that the scheduler observes idle with no compute process for at
least 60 seconds. The coordinator can wait safely while other users occupy GPUs.

Every fully committed record batch contains all six views and all three model
arms. The coordinator reads these committed files and refreshes
`state/live_metrics.json` every 30 seconds. It reports completed records by
center, per-center macro-F1 for clean and image conditions, and the three-chain
minus single-chain difference in percentage points. Interim values are
provisional and use only records already completed. After the blocks finish,
the workflow recomputes center metrics from the merged predictions and requires
them to match the live count-based metrics before marking the live file complete.
Before GPU execution, the live metric calculation was also compared with the
four existing K256 result files: all 1,024 records, six views, three arms, and
per-condition macro-F1, exact match, Hamming loss, parse-failure rate, and
truncation rate matched exactly (maximum difference 0). This checks the metric
calculation, not a new model run; the fresh GPU smoke remains pending.

Run data stages in task-scoped `/dev/shm` and archives to the authorized HDD
path from the config. If RAM or disk headroom drops below the configured reserve,
the scheduler waits instead of starting another worker.

Launch configuration: `configs/experiments/pulse_full_lora32_live_eval_20260928.yaml`.
The run state is under
`/dev/shm/linbinhao-pulse-hybrid/full_lora32_live_eval_20260928_r0/run/state/`;
the durable archive is `/data/linbinhao/ecg_llm_runs/pulse_full_lora32_live_eval_20260928_r0/`.
