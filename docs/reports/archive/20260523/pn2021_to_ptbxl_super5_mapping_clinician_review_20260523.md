# PN2021 到 PTB-XL Super5 标签映射策略医生校对版

- 生成日期：2026-05-23
- 代码来源：`scripts/triple_labels/label_schemes.py`
- 当前映射版本：`v5_super5_strict_voltage_pacing_suppress_20260522`
- 当前映射 hash：`1141e0a9f94b`
- Super5 类别顺序：`CD, HYP, MI, NORM, STTC`
- PN2021 统计范围：本地 7 个外部评估中心 `chapman_shaoxing, cpsc_2018, cpsc_2018_extra, georgia, ningbo, ptb, st_petersburg_incart`，总记录数 `66416`。统计不使用 `ptb-xl/ptbxl` 泄漏 shard。

## 1. 这份映射在做什么

PTB-XL 的 5 类 Super5 来自 PTB-XL 官方 `scp_statements.csv` 的 `diagnostic_class`，包括 `CD, HYP, MI, NORM, STTC`。PN2021 使用 SNOMED-CT 诊断码，官方没有提供 `PN2021 SNOMED -> PTB-XL Super5` 的 crosswalk。因此这里是我们为跨中心评估定义的保守语义映射，需要医生校对。

PN2021 原生标签体系不是 5 类，而是每条 ECG header 里的多标签 SNOMED-CT code list。我们当前本地官方标签表包含 `30` 个 scored labels 和 `103` 个 unscored labels，共 `133` 个唯一 SNOMED 标签；本报告把这 133 个官方标签全部列入 direct / NORM candidate / suppress-only / unmapped 四类处理桶。

映射规则分成三类：

- **直接阳性**：明确属于 `CD/HYP/MI/STTC` 的 SNOMED 码，直接置对应类别为 1。
- **NORM candidate**：只有 `sinus rhythm` 可以作为 NORM 候选；若同条 ECG 还有任何异常阳性或 suppress-only 标签，则 `NORM=0`。
- **suppress-only**：说明 ECG 不应被视为正常，但也不强行归入 `CD/HYP/MI/STTC`，例如 AF/AFL、早搏、轴偏、pacing、high voltage 等。

## 2. 请医生重点校对的问题

1. `pacing rhythm / atrial pacing pattern / ventricular pacing pattern`：当前只 suppress NORM，不作为 CD 阳性。请确认是否应在主分析中归为 CD，或只做敏感性分析。
2. `left ventricular high voltage / right atrial high voltage`：当前只 suppress NORM，不作为 HYP 阳性。请确认 high voltage 是否足以作为 HYP 标签。
3. `Q wave abnormal`：当前只 suppress NORM，不作为 MI 或 STTC 阳性。请确认是否应映射为 MI-sensitive 或 STTC-sensitive 消融。
4. `early repolarization`：当前只 suppress NORM，不作为 STTC 阳性。请确认是否合理。
5. rhythm-only / ectopy / tachyarrhythmia 标签是否都应保持 outside Super5，而不是映射进 CD/STTC。

## 3. 五类映射总览

| Super5 | 当前映射标签数 | 映射原则 | 本地 7-center 原始标签出现次数合计 |
| --- | --- | --- | --- |
| CD | 20 | 明确传导系统异常：AV block、BBB、fascicular block、IVCD、WPW/pre-excitation。 | 11684 |
| HYP | 7 | 明确 hypertrophy/enlargement：室/房肥厚或心腔增大。 | 3388 |
| MI | 7 | 明确 myocardial infarction/infarction。ischemia 不归 MI。 | 2225 |
| NORM | 1 | 只接受 explicit sinus rhythm 作为候选；有异常或 suppress-only 时取消 NORM。 | 10879 |
| STTC | 12 | ST/T/QT 异常和 myocardial ischemia。 | 30956 |

## 4.1 `CD` 对应的 PN2021 标签

直接阳性映射：出现该 SNOMED 码时，对应 Super5 类置 1。

| SNOMED code | PN2021 标签名 | 缩写 | 来源 | 官方表 Total | 本地 7-center 出现次数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 270492004 | 1st degree av block | IAVB | scored | 3534 | 2737 |  |
| 195042002 | 2nd degree av block | IIAVB | unscored | 124 | 110 |  |
| 26749005 | Wolff-Parkinson-White syndrome | WPW-alt | legacy/internal | 0 | 0 | 代码中保留的 legacy/alias；当前官方表或本地数据中未出现。 |
| 233917008 | av block | AVB | unscored | 323 | 323 |  |
| 6374002 | bundle branch block | BBB | scored | 522 | 522 |  |
| 27885002 | complete heart block | CHB | unscored | 127 | 111 |  |
| 733534002 | complete left bundle branch block | CLBBB | scored | 213 | 213 | We score 733534002 and 164909002 as the same diagnosis |
| 713427006 | complete right bundle branch block | CRBBB | scored | 1779 | 1237 | We score 713427006 and 59118001 as the same diagnosis. |
| 251120003 | incomplete left bundle branch block | ILBBB | unscored | 211 | 134 |  |
| 713426002 | incomplete right bundle branch block | IRBBB | scored | 1857 | 739 |  |
| 445118002 | left anterior fascicular block | LAnFB | scored | 2186 | 560 |  |
| 164909002 | left bundle branch block | LBBB | scored | 1281 | 745 | We score 733534002 and 164909002 as the same diagnosis |
| 445211001 | left posterior fascicular block | LPFB | unscored | 207 | 30 |  |
| 426183003 | mobitz type II atrioventricular block | IIAVBII | unscored | 7 | 7 |  |
| 54016002 | mobitz type i wenckebach atrioventricular block | MoI | unscored | 34 | 34 |  |
| 698252002 | nonspecific intraventricular conduction disorder | NSIVCB | scored | 1768 | 979 |  |
| 164947007 | prolonged pr interval | LPR | scored | 392 | 52 |  |
| 59118001 | right bundle branch block | RBBB | scored | 3051 | 3051 | We score 713427006 and 59118001 as the same diagnosis. |
| 195060002 | ventricular pre excitation | VPEx | unscored | 20 | 20 |  |
| 74390002 | wolff parkinson white pattern | WPW | unscored | 160 | 80 |  |

## 4.2 `HYP` 对应的 PN2021 标签

直接阳性映射：出现该 SNOMED 码时，对应 Super5 类置 1。

| SNOMED code | PN2021 标签名 | 缩写 | 来源 | 官方表 Total | 本地 7-center 出现次数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 195126007 | atrial hypertrophy | AH | unscored | 62 | 62 |  |
| 67741000119109 | left atrial enlargement | LAE | unscored | 1299 | 872 |  |
| 446813000 | left atrial hypertrophy | LAH | unscored | 48 | 48 |  |
| 164873001 | left ventricular hypertrophy | LVH | unscored | 4406 | 2047 |  |
| 446358003 | right atrial hypertrophy | RAH | unscored | 153 | 54 |  |
| 89792004 | right ventricular hypertrophy | RVH | unscored | 342 | 216 |  |
| 266249003 | ventricular hypertrophy | VH | unscored | 119 | 89 |  |

## 4.3 `MI` 对应的 PN2021 标签

直接阳性映射：出现该 SNOMED 码时，对应 Super5 类置 1。

| SNOMED code | PN2021 标签名 | 缩写 | 来源 | 官方表 Total | 本地 7-center 出现次数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 401303003 | acute anterior myocardial infarction | AAntMI | legacy/internal | 0 | 0 | 代码中保留的 legacy/alias；当前官方表或本地数据中未出现。 |
| 57054005 | acute myocardial infarction | AMI | unscored | 55 | 55 |  |
| 54329005 | anterior myocardial infarction | AnMI | unscored | 473 | 119 |  |
| 233843008 | inferior myocardial infarction | InfMI | legacy/internal | 0 | 0 | 代码中保留的 legacy/alias；当前官方表或本地数据中未出现。 |
| 164865005 | myocardial infarction | MI | unscored | 6144 | 883 |  |
| 164867002 | old myocardial infarction | OldMI | unscored | 1168 | 1168 |  |
| 22298006 | subacute myocardial infarction | SubMI | legacy/internal | 0 | 0 | 代码中保留的 legacy/alias；当前官方表或本地数据中未出现。 |

## 4.4 `NORM` 对应的 PN2021 标签

NORM candidate：只有在同条记录没有直接异常阳性、也没有 suppress-only 标签时，才最终置 `NORM=1`。

| SNOMED code | PN2021 标签名 | 缩写 | 来源 | 官方表 Total | 本地 7-center 出现次数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 426783006 | sinus rhythm | NSR | scored | 28971 | 10879 |  |

## 4.5 `STTC` 对应的 PN2021 标签

直接阳性映射：出现该 SNOMED 码时，对应 Super5 类置 1。

| SNOMED code | PN2021 标签名 | 缩写 | 来源 | 官方表 Total | 本地 7-center 出现次数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 426434006 | anterior ischemia | AnMIs | unscored | 325 | 281 |  |
| 425419005 | inferior ischaemia | IIs | unscored | 670 | 451 |  |
| 425623009 | lateral ischaemia | LIs | unscored | 1045 | 903 |  |
| 164861001 | myocardial ischemia | MIs | unscored | 2559 | 384 |  |
| 428750005 | nonspecific st t abnormality | NSSTTA | unscored | 4712 | 4331 |  |
| 111975006 | prolonged qt interval | LQT | scored | 1907 | 1789 |  |
| 55930002 | s t changes | STC | unscored | 5009 | 4239 |  |
| 429622005 | st depression | STD | unscored | 3645 | 2636 |  |
| 164931005 | st elevation | STE | unscored | 628 | 600 |  |
| 164930006 | st interval abnormal | STIAb | unscored | 2276 | 2276 |  |
| 164934002 | t wave abnormal | TAb | scored | 11716 | 9371 |  |
| 59931005 | t wave inversion | TInv | scored | 3989 | 3695 |  |

## 5. suppress-only：不能直接匹配 Super5，但用于取消 NORM

这些标签不置 `CD/HYP/MI/STTC=1`，也不置 `NORM=1`。如果一条 ECG 只有这些标签，最终 5 类标签可能是全 0；这不代表正常，而是代表“PN2021 标签不属于 PTB-XL Super5 主分析阳性类”。

| SNOMED code | PN2021 标签名 | 缩写 | 来源 | 官方表 Total | 本地 7-center 出现次数 | 当前处理原因 |
| --- | --- | --- | --- | --- | --- | --- |
| 164889003 | atrial fibrillation | AF | scored | 5255 | 3741 | 房颤/房扑类节律异常，不属于 PTB-XL Super5 主类。 |
| 164890007 | atrial flutter | AFL | scored | 8374 | 8301 | 房颤/房扑类节律异常，不属于 PTB-XL Super5 主类。 |
| 426627000 | bradycardia | Brady | scored | 295 | 295 | 节律/心率变异，不等同 PTB-XL NORM。 |
| 39732003 | left axis deviation | LAD | scored | 7631 | 2485 | 轴偏，不能稳定归入 Super5。 |
| 251146004 | low qrs voltages | LQRSV | scored | 1599 | 1417 | 形态/电压异常，取消 NORM 但不强制映射。 |
| 10370003 | pacing rhythm | PR | scored | 1481 | 1185 | pacing/device rhythm：改变形态，但不等同 PTB-XL CD 诊断子类。 |
| 365413008 | poor R wave Progression | PRWP | scored | 638 | 638 | 形态/电压异常，取消 NORM 但不强制映射。 |
| 284470004 | premature atrial contraction | PAC | scored | 3041 | 2643 | 早搏/异位搏动，不能稳定归入 Super5。 |
| 427172004 | premature ventricular contractions | PVC | scored | 1279 | 1279 | 早搏/异位搏动，不能稳定归入 Super5。 |
| 164917005 | qwave abnormal | QAb | scored | 2076 | 1528 | 边界形态标签：主分析保守处理，建议敏感性分析。 |
| 47665007 | right axis deviation | RAD | scored | 1280 | 937 | 轴偏，不能稳定归入 Super5。 |
| 427393009 | sinus arrhythmia | SA | scored | 3790 | 3018 | 节律/心率变异，不等同 PTB-XL NORM。 |
| 426177001 | sinus bradycardia | SB | scored | 18918 | 18281 | 节律/心率变异，不等同 PTB-XL NORM。 |
| 427084000 | sinus tachycardia | STach | scored | 9657 | 8831 | 节律/心率变异，不等同 PTB-XL NORM。 |
| 63593006 | supraventricular premature beats | SVPB | scored | 224 | 67 | 早搏/异位搏动，不能稳定归入 Super5。 |
| 17338001 | ventricular premature beats | VPB | scored | 659 | 659 | 早搏/异位搏动，不能稳定归入 Super5。 |
| 164951009 | abnormal QRS | abQRS | unscored | 3389 | 0 | 形态/电压异常，取消 NORM 但不强制映射。 |
| 233892002 | accelerated atrial escape rhythm | AAR | unscored | 16 | 16 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 61277005 | accelerated idioventricular rhythm | AIVR | unscored | 14 | 14 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 426664006 | accelerated junctional rhythm | AJR | unscored | 31 | 31 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 251173003 | atrial bigeminy | AB | unscored | 6 | 6 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 251187003 | atrial escape beat | AED | unscored | 17 | 17 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 195080001 | atrial fibrillation and flutter | AFAFL | unscored | 41 | 41 | 房颤/房扑类节律异常，不属于 PTB-XL Super5 主类。 |
| 251268003 | atrial pacing pattern | AP | unscored | 52 | 52 | pacing/device rhythm：改变形态，但不等同 PTB-XL CD 诊断子类。 |
| 713422000 | atrial tachycardia | ATach | unscored | 340 | 340 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 251166008 | atrioventricular  node reentrant tachycardia | AVNRT | unscored | 16 | 16 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 50799005 | atrioventricular dissociation | AVD | unscored | 59 | 59 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 29320008 | atrioventricular junctional rhythm | AVJR | unscored | 145 | 145 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 233897008 | atrioventricular reentrant tachycardia | AVRT | unscored | 26 | 26 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 251170000 | blocked premature atrial contraction | BPAC | unscored | 67 | 67 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 74615001 | brady tachy syndrome | BTS | unscored | 2 | 2 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 698247007 | cardiac dysrhythmia | CD | unscored | 16 | 16 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 426749004 | chronic atrial fibrillation | CAF | unscored | 1 | 1 | 房颤/房扑类节律异常，不属于 PTB-XL Super5 主类。 |
| 251198002 | clockwise rotation | CR | unscored | 76 | 76 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 251199005 | countercolockwise rotation | CCR | unscored | 162 | 162 | Super5 外标签，取消 NORM 但不作为主分析阳性。 |
| 428417006 | early repolarization | ERe | unscored | 506 | 506 | 边界形态标签：主分析保守处理，建议敏感性分析。 |
| 55827005 | left ventricular high voltage | LVHV | unscored | 5401 | 5401 | high voltage：电压表现，证据弱于 hypertrophy/enlargement。 |
| 67751000119106 | right atrial  high voltage | RAHV | unscored | 36 | 36 | high voltage：电压表现，证据弱于 hypertrophy/enlargement。 |
| 251266004 | ventricular pacing pattern | VPP | unscored | 46 | 46 | pacing/device rhythm：改变形态，但不等同 PTB-XL CD 诊断子类。 |

## 6. 完全未匹配到 Super5 的 PN2021 标签

下表是 PN2021 官方 scored/unscored 标签中，当前既不作为 Super5 直接阳性、也不是 NORM candidate、也不是 suppress-only 的标签。它们在本项目 Super5 主分析中不产生任何 5 类阳性；若一条记录只含这些标签，最终会变成 all-zero。医生校对时可重点判断其中是否有应纳入 `CD/HYP/MI/STTC` 或 suppress-only 的项目。

注意：unmapped 与 suppress-only 不同。unmapped 标签本身不会取消 `sinus rhythm -> NORM` 的候选关系；如果某条 ECG 同时含有 `sinus rhythm` 和某个 unmapped 标签，且没有其他 direct/suppress-only 异常，当前实现可能仍得到 `NORM=1`。因此医生校对时也需要判断这些 unmapped 标签里哪些至少应该改为 suppress-only。

| SNOMED code | PN2021 标签名 | 缩写 | 来源 | 官方表 Total | 本地 7-center 出现次数 |
| --- | --- | --- | --- | --- | --- |
| 413444003 | acute myocardial ischemia | AMIs | unscored | 2 | 2 |
| 106068003 | atrial rhythm | ARH | unscored | 215 | 215 |
| 418818005 | brugada | BRU | unscored | 5 | 5 |
| 413844008 | chronic myocardial ischemia | CMI | unscored | 161 | 161 |
| 61721007 | clockwise or counterclockwise vectorcardiographic loop | CVCL/CCVCL | unscored | 653 | 653 |
| 204384007 | congenital incomplete atrioventricular heart block | CIAHB | unscored | 2 | 2 |
| 53741008 | coronary heart disease | CHD | unscored | 37 | 37 |
| 77867006 | decreased qt interval | SQT | unscored | 3 | 3 |
| 82226007 | diffuse intraventricular block | DIB | unscored | 1 | 1 |
| 164942001 | fqrs wave | FQRS | unscored | 3 | 3 |
| 13640000 | fusion beats | FB | unscored | 123 | 123 |
| 84114007 | heart failure | HF | unscored | 7 | 7 |
| 368009 | heart valve disorder | HVD | unscored | 6 | 6 |
| 251259000 | high t-voltage | HTV | unscored | 1 | 1 |
| 49260003 | idioventricular rhythm | IR | unscored | 2 | 2 |
| 251200008 | indeterminate cardiac axis | ICA | unscored | 156 | 0 |
| 704997005 | inferior ST segment depression | ISTD | unscored | 1 | 1 |
| 426995002 | junctional escape | JE | unscored | 84 | 84 |
| 251164006 | junctional premature complex | JPC | unscored | 13 | 13 |
| 426648003 | junctional tachycardia | JTach | unscored | 30 | 30 |
| 253352002 | left atrial abnormality | LAA | unscored | 72 | 72 |
| 370365005 | left ventricular strain | LVS | unscored | 1 | 1 |
| 164912004 | p wave change | PWC | unscored | 142 | 142 |
| 251182009 | paired ventricular premature complexes | VPVC | unscored | 23 | 23 |
| 282825002 | paroxysmal atrial fibrillation | PAF | unscored | 2 | 2 |
| 67198005 | paroxysmal supraventricular tachycardia | PSVT | unscored | 27 | 3 |
| 425856008 | paroxysmal ventricular tachycardia | PVT | unscored | 124 | 124 |
| 251205003 | prolonged P wave | PPW | unscored | 106 | 106 |
| 164921003 | r wave abnormal | RAb | unscored | 11 | 11 |
| 314208002 | rapid atrial fibrillation | RAF | unscored | 2 | 2 |
| 253339007 | right atrial abnormality | RAAb | unscored | 14 | 14 |
| 49578007 | shortened pr interval | SPRI | unscored | 28 | 28 |
| 65778007 | sinoatrial block | SAB | unscored | 14 | 14 |
| 5609005 | sinus arrest | SARR | unscored | 33 | 33 |
| 17366009 | sinus atrium to atrial wandering rhythm | SAAWR | unscored | 7 | 7 |
| 60423000 | sinus node dysfunction | SND | unscored | 2 | 2 |
| 251168009 | supraventricular bigeminy | SVB | unscored | 1 | 1 |
| 426761007 | supraventricular tachycardia | SVT | unscored | 787 | 760 |
| 251139008 | suspect arm ecg leads reversed | ALR | unscored | 12 | 12 |
| 251223006 | tall p wave | TPW | unscored | 215 | 215 |
| 266257000 | transient ischemic attack | TIA | unscored | 7 | 7 |
| 164937009 | u wave abnormal | UAb | unscored | 137 | 137 |
| 11157007 | ventricular bigeminy | VBig | unscored | 101 | 19 |
| 164884008 | ventricular ectopics | VEB | unscored | 1944 | 790 |
| 75532003 | ventricular escape beat | VEsB | unscored | 60 | 60 |
| 81898007 | ventricular escape rhythm | VEsR | unscored | 98 | 98 |
| 164896001 | ventricular fibrillation | VF | unscored | 97 | 97 |
| 111288001 | ventricular flutter | VFL | unscored | 8 | 8 |
| 164895002 | ventricular tachycardia | VTach | unscored | 12 | 12 |
| 251180001 | ventricular trigeminy | VTrig | unscored | 37 | 17 |
| 195101003 | wandering atrial pacemaker | WAP | unscored | 9 | 9 |

## 7. 当前映射造成的 all-zero 情况

all-zero 表示该 ECG 没有被映射到 `CD/HYP/MI/NORM/STTC` 的任何一类，不等价于正常。它通常来自 suppress-only 或 outside-Super5 标签。

| center | N | CD | HYP | MI | NORM | STTC | all-zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chapman_shaoxing | 10247 | 1226 | 21 | 40 | 1371 | 2951 | 5146 (50.2%) |
| cpsc_2018 | 6877 | 2797 | 0 | 0 | 918 | 1087 | 2131 (31.0%) |
| cpsc_2018_extra | 3453 | 378 | 225 | 1515 | 0 | 2212 | 89 (2.6%) |
| georgia | 10344 | 2238 | 2145 | 7 | 1752 | 5011 | 1668 (16.1%) |
| ningbo | 34905 | 3894 | 749 | 165 | 4606 | 10407 | 17697 (50.7%) |
| ptb | 516 | 22 | 13 | 368 | 80 | 0 | 36 (7.0%) |
| st_petersburg_incart | 74 | 10 | 10 | 15 | 0 | 11 | 33 (44.6%) |
| total | 66416 | 10565 | 3163 | 2110 | 8727 | 21679 | 26800 (40.4%) |

## 8. 建议医生反馈格式

请医生优先按下面三类给意见：

1. **应直接映射**：某个 SNOMED 应该明确归入 `CD/HYP/MI/STTC/NORM`。
2. **应 suppress-only**：某个 SNOMED 不属于 5 类阳性，但也不应当作正常。
3. **应完全排除/不确定**：某个 SNOMED 不适合进入 Super5 主分析，或需要单独敏感性分析。

建议特别回复以下争议项：pacing/device rhythm、high voltage、Q wave abnormal、early repolarization、AF/AFL/PAC/PVC、SVT/VT 类节律异常是否应该保持 outside Super5。

## 9. 代码位置

- 当前映射：`scripts/triple_labels/label_schemes.py`
- 映射函数：`snomed_list_to_super5`
- 直接阳性表：`SNOMED_TO_SUPER5_POSITIVE`
- NORM 候选：`NORM_POSITIVE_SNOMEDS`
- suppress-only：`NORM_SUPPRESS_SNOMEDS`
