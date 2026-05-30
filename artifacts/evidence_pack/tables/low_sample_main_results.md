| method | test_macro_auroc | test_macro_auprc | delta_auroc_vs_real2000 | delta_auprc_vs_real2000 | epochs_trained | run_dir | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| real2000 baseline | 0.8433 | 0.6234 | 0.0000 | 0.0000 | 50 | /root/autodl-tmp/graduate_project/method_a_real2000_seed42 | Paper main baseline run paired with the original downstream comparison. |
| no-token hard pretrain -> real fine-tune | 0.8669 | 0.6828 | 0.0236 | 0.0594 | 25 | /root/autodl-tmp/graduate_project/self_distill_v2_e24_v46_no_token_hardlabel_r10_realfine_lr1e4_seed42_auroc | Strict no-token hard-label pretrain followed by real2000 fine-tune. |
| center-token hard pretrain -> real fine-tune | 0.8735 | 0.7013 | 0.0302 | 0.0779 | 17 | /root/autodl-tmp/graduate_project/self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc | Best thesis low-sample recipe. |
