| method | test_macro_auroc | test_macro_auprc | epochs_trained | run_dir | note |
| --- | ---: | ---: | ---: | --- | --- |
| real2000 + prompt-token synth | 0.8477 | 0.6419 | 50 | /root/autodl-tmp/graduate_project/method_b_real2000_plus_synth3364_mv4_seed42 | Direct real+synth training from random init. |
| real2000 + actual-report prompt synth | 0.8412 | 0.6239 | 50 | /root/autodl-tmp/graduate_project/method_b_real2000_plus_synth3313_actual_report_mv4_step2000_seed42 | Actual-report prompt variant. |
| real2000 + class-fallback new generation | 0.8432 | 0.6065 | 50 | /root/autodl-tmp/graduate_project/method_b_real2000_plus_synth3391_classfallback_newgen_sameclass_cap20_seed42 | Class-fallback prompt variant. |
| no-token synthetic-only pretrain | 0.6023 | 0.3851 | 11 | /root/autodl-tmp/graduate_project/ablation_vanilla_no_token_synthonly_pretrain_n20000_seed42 | Synthetic-only classifier is weak without real fine-tuning. |
| center-token hard pretrain -> real fine-tune | 0.8735 | 0.7013 | 17 | /root/autodl-tmp/graduate_project/self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc | Center-token synthetic pretraining followed by real2000 fine-tune; thesis main method. |
