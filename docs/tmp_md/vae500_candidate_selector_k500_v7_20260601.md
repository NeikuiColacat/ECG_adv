# VAE500 K500 Classwise Selector Results

Date: 2026-06-01

| center | direct | plain | old | v7scope | selector | selector - direct | choices |
|---|---:|---:|---:|---:|---:|---:|---|
| ningbo | 0.8798 / 0.5023 | 0.8774 / 0.4986 | 0.8790 / 0.4964 | 0.8784 / 0.4955 | 0.8802 / 0.5014 | +0.04pp / -0.09pp | CD:old, HYP:plain, MI:direct, NORM:direct, STTC:old |
| chapman_shaoxing | 0.8725 / 0.4448 | 0.8728 / 0.4459 | 0.8749 / 0.4507 | 0.8739 / 0.4511 | 0.8751 / 0.4494 | +0.26pp / +0.46pp | CD:plain, HYP:old, MI:direct, NORM:v7scope, STTC:v7scope |
| cpsc_2018 | 0.8533 / 0.6011 | 0.8482 / 0.5932 | 0.8671 / 0.6221 | 0.8568 / 0.6017 | 0.8655 / 0.6178 | +1.22pp / +1.67pp | CD:v7scope, HYP:direct, MI:direct, NORM:v7scope, STTC:old |
| georgia | 0.8347 / 0.6180 | 0.8327 / 0.6135 | 0.8360 / 0.6192 | 0.8358 / 0.6191 | 0.8366 / 0.6209 | +0.19pp / +0.29pp | CD:old, HYP:old, MI:direct, NORM:direct, STTC:plain |

| method | mean AUROC / AUPRC | delta vs direct |
|---|---:|---:|
| direct | 0.8601 / 0.5416 | +0.00pp / +0.00pp |
| plain | 0.8578 / 0.5378 | -0.23pp / -0.38pp |
| old | 0.8643 / 0.5471 | +0.42pp / +0.55pp |
| v7scope | 0.8612 / 0.5419 | +0.12pp / +0.03pp |
| selector | 0.8643 / 0.5474 | +0.43pp / +0.58pp |
