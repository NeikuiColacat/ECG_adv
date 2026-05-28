# PN2021 到 PTB-XL Super5标签体系

## 标签映射策略医生审核版

含五类直接映射表 + 未直接归类标签补充判断表

## 一、审核需求说明

本项目需要把 PN2021 心电数据集中的 137 个 SNOMED-CT 诊断标签，尽可能与 PTB-XL 的 Super5 五类标签体系对齐。Super5 五类为 CD、HYP、MI、NORM、STTC。

当前策略中已有 47 个标签 被直接映射到五类；另有 90 个标签 尚未直接归入任何一类。本版将这些未直接归类标签单独列出，方便医生判断是否应补充映射。

请医生重点判断：每个标签应直接归入 CD/HYP/MI/NORM/STTC，还是保持五类之外、仅用于取消正常标签，或完全放弃不参与标签映射。

本版新增“中文译名/简短医学语义说明”列，帮助医生快速理解英文 PN2021 标签的中文临床含义。

# 二、Super5 五类标签的医学语义

| **类别** | **英文/中文含义** | **医学语义** | **本策略映射原则** |
| --- | --- | --- | --- |
| CD | Conduction Disturbance<br>传导障碍 | 房室传导阻滞、束支传导阻滞、分支阻滞、室内传导延迟、预激综合征等传导系统异常。 | 看到明确传导阻滞/传导延迟/预激类诊断时，归入 CD。 |
| HYP | Hypertrophy<br>肥厚/心腔扩大 | 心房或心室肥厚、心腔扩大等结构性负荷改变在心电图上的诊断标签。 | 看到明确 hypertrophy / enlargement 标签时，归入 HYP；单纯 high voltage 需医生判断。 |
| MI | Myocardial Infarction<br>心肌梗死 | 急性、亚急性、陈旧性或部位性心肌梗死相关标签，强调 infarction/MI 语义。 | 明确 infarction / myocardial infarction 归入 MI；单纯 ischemia 不归 MI。 |
| NORM | Normal ECG candidate<br>正常候选 | 正常人心电图候选标签。本策略中 PN2021 只使用 sinus rhythm 作为 NORM 候选。 | 窦性心律不等于完全正常；仅当同条 ECG 无其他异常标签时才最终判为 NORM。 |
| STTC | ST/T Change<br>ST-T/QT/缺血相关改变 | ST 段改变、T 波异常、QT 间期异常，以及心肌缺血相关心电改变。 | ST depression/elevation、ST-T abnormality、T wave abnormal/inversion、QT prolongation、ischemia 归入 STTC。 |

# 三、五类直接映射表

以下表格展示当前策略中直接映射到 Super5 五类的 PN2021 标签。每个表均保留医生意见栏，便于逐项修改。

## 3.1 被映射为 CD（传导障碍）的 PN2021 标签

映射原则：明确房室传导、束支/分支传导、室内传导或预激相关异常，归入 CD。

| **SNOMED code** | **PN2021 标签名** | **中文译名/简短医学语义说明** | **缩写** | **医学语义/拟映射理由** | **医生意见** |
| --- | --- | --- | --- | --- | --- |
| 270492004 | 1st degree av block | 一度房室传导阻滞：PR 间期延长，属房室传导延迟。 | IAVB | 房室传导延迟或阻滞，属于传导障碍。 | 均赞成RGQ |
| 195042002 | 2nd degree av block | 二度房室传导阻滞：部分 P 波不能下传，属房室传导异常。 | IIAVB | 房室传导延迟或阻滞，属于传导障碍。 |  |
| 26749005 | Wolff-Parkinson-White syndrome | Wolff-Parkinson-White 综合征：旁路预激相关综合征。 | WPW-alt | 预激综合征/旁路传导相关异常，属于传导障碍。 |  |
| 233917008 | av block | 房室传导阻滞：房室结或希浦系统传导受阻。 | AVB | 房室传导延迟或阻滞，属于传导障碍。 |  |
| 6374002 | bundle branch block | 束支传导阻滞：左右束支传导延迟或中断。 | BBB | 束支、分支或室内传导异常，属于传导障碍。 |  |
| 27885002 | complete heart block | 完全性房室传导阻滞：房室传导完全阻断。 | CHB | 房室传导延迟或阻滞，属于传导障碍。 |  |
| 733534002 | complete left bundle branch block | 完全性左束支传导阻滞：左束支传导阻断，QRS 增宽。 | CLBBB | 束支、分支或室内传导异常，属于传导障碍。<br>同义/近义标签：733534002 与 164909002 按同一诊断处理。 |  |
| 713427006 | complete right bundle branch block | 完全性右束支传导阻滞：右束支传导阻断，QRS 增宽。 | CRBBB | 束支、分支或室内传导异常，属于传导障碍。<br>同义/近义标签：713427006 与 59118001 按同一诊断处理。 |  |
| 251120003 | incomplete left bundle branch block | 不完全性左束支传导阻滞：左束支传导延迟但未完全阻断。 | ILBBB | 束支、分支或室内传导异常，属于传导障碍。 |  |
| 713426002 | incomplete right bundle branch block | 不完全性右束支传导阻滞：右束支传导延迟但未完全阻断。 | IRBBB | 束支、分支或室内传导异常，属于传导障碍。 |  |
| 445118002 | left anterior fascicular block | 左前分支传导阻滞：左束支前分支传导异常。 | LAnFB | 束支、分支或室内传导异常，属于传导障碍。 |  |
| 164909002 | left bundle branch block | 左束支传导阻滞：左束支传导异常，常伴 QRS 增宽。 | LBBB | 束支、分支或室内传导异常，属于传导障碍。<br>同义/近义标签：733534002 与 164909002 按同一诊断处理。 |  |
| 445211001 | left posterior fascicular block | 左后分支传导阻滞：左束支后分支传导异常。 | LPFB | 束支、分支或室内传导异常，属于传导障碍。 |  |
| 426183003 | mobitz type II atrioventricular block | Mobitz II 型房室传导阻滞：PR 多固定，间歇性 QRS 脱漏。 | IIAVBII | 房室传导延迟或阻滞，属于传导障碍。 |  |
| 54016002 | mobitz type i wenckebach atrioventricular block | Mobitz I 型/文氏房室传导阻滞：PR 逐渐延长后脱漏。 | MoI | 房室传导延迟或阻滞，属于传导障碍。 |  |
| 698252002 | nonspecific intraventricular conduction disorder | 非特异性室内传导障碍：QRS 传导延迟但不典型束支阻滞。 | NSIVCB | 束支、分支或室内传导异常，属于传导障碍。 |  |
| 164947007 | prolonged pr interval | PR 间期延长：提示房室传导延迟。 | LPR | 房室传导延迟或阻滞，属于传导障碍。 |  |
| 59118001 | right bundle branch block | 右束支传导阻滞：右束支传导异常，常伴 QRS 增宽。 | RBBB | 束支、分支或室内传导异常，属于传导障碍。<br>同义/近义标签：713427006 与 59118001 按同一诊断处理。 |  |
| 195060002 | ventricular pre excitation | 心室预激：旁路提前激动心室，可见短 PR/δ 波。 | VPEx | 预激综合征/旁路传导相关异常，属于传导障碍。 |  |
| 74390002 | wolff parkinson white pattern | WPW 图形/预激图形：短 PR、δ 波等旁路预激表现。 | WPW | 预激综合征/旁路传导相关异常，属于传导障碍。 | 赞成RGQ |

## 3.2 被映射为 HYP（肥厚/心腔扩大）的 PN2021 标签

映射原则：明确心房/心室肥厚或心腔扩大相关诊断，归入 HYP。

| **SNOMED code** | **PN2021 标签名** | **中文译名/简短医学语义说明** | **缩写** | **医学语义/拟映射理由** | **医生意见** |
| --- | --- | --- | --- | --- | --- |
| 195126007 | atrial hypertrophy | 心房肥厚：心房负荷增大相关心电表现。 | AH | 心房肥厚或扩大相关标签，拟归入 HYP。 | 均赞成RGQ |
| 67741000119109 | left atrial enlargement | 左心房扩大：提示左房容量或压力负荷增大。 | LAE | 心房肥厚或扩大相关标签，拟归入 HYP。 |  |
| 446813000 | left atrial hypertrophy | 左心房肥厚：左房负荷增大相关心电表现。 | LAH | 心房肥厚或扩大相关标签，拟归入 HYP。 |  |
| 164873001 | left ventricular hypertrophy | 左心室肥厚：左室压力/容量负荷增大相关心电表现。 | LVH | 心室肥厚相关标签，拟归入 HYP。 |  |
| 446358003 | right atrial hypertrophy | 右心房肥厚：右房负荷增大相关心电表现。 | RAH | 心房肥厚或扩大相关标签，拟归入 HYP。 |  |
| 89792004 | right ventricular hypertrophy | 右心室肥厚：右室压力/容量负荷增大相关心电表现。 | RVH | 心室肥厚相关标签，拟归入 HYP。 |  |
| 266249003 | ventricular hypertrophy | 心室肥厚：心室负荷增大相关心电诊断。 | VH | 心室肥厚相关标签，拟归入 HYP。 |  |

## 3.3 被映射为 MI（心肌梗死）的 PN2021 标签

映射原则：明确 myocardial infarction / infarction 相关诊断，归入 MI。

| **SNOMED code** | **PN2021 标签名** | **中文译名/简短医学语义说明** | **缩写** | **医学语义/拟映射理由** | **医生意见** |
| --- | --- | --- | --- | --- | --- |
| 401303003 | acute anterior myocardial infarction | 急性前壁心肌梗死：前壁急性梗死相关诊断。 | AAntMI | 明确心肌梗死/梗死后改变相关标签，拟归入 MI；单纯 ischemia 不按 MI 处理。 | 均赞成RGQ |
| 57054005 | acute myocardial infarction | 急性心肌梗死：急性冠脉闭塞/心肌坏死相关诊断。 | AMI | 明确心肌梗死/梗死后改变相关标签，拟归入 MI；单纯 ischemia 不按 MI 处理。 |  |
| 54329005 | anterior myocardial infarction | 前壁心肌梗死：前壁梗死相关诊断。 | AnMI | 明确心肌梗死/梗死后改变相关标签，拟归入 MI；单纯 ischemia 不按 MI 处理。 |  |
| 233843008 | inferior myocardial infarction | 下壁心肌梗死：下壁梗死相关诊断。 | InfMI | 明确心肌梗死/梗死后改变相关标签，拟归入 MI；单纯 ischemia 不按 MI 处理。 |  |
| 164865005 | myocardial infarction | 心肌梗死：心肌坏死或梗死后改变相关诊断。 | MI | 明确心肌梗死/梗死后改变相关标签，拟归入 MI；单纯 ischemia 不按 MI 处理。 |  |
| 164867002 | old myocardial infarction | 陈旧性心肌梗死：既往梗死或梗死后改变。 | OldMI | 明确心肌梗死/梗死后改变相关标签，拟归入 MI；单纯 ischemia 不按 MI 处理。 |  |
| 22298006 | subacute myocardial infarction | 亚急性心肌梗死：非急性期但仍属梗死相关诊断。 | SubMI | 明确心肌梗死/梗死后改变相关标签，拟归入 MI；单纯 ischemia 不按 MI 处理。 |  |

## 3.4 被映射为 NORM（正常候选）的 PN2021 标签

映射原则：只将 sinus rhythm 作为 NORM 候选；若同条 ECG 有其他异常标签，则不判为 NORM。

| **SNOMED code** | **PN2021 标签名** | **中文译名/简短医学语义说明** | **缩写** | **医学语义/拟映射理由** | **医生意见** |
| --- | --- | --- | --- | --- | --- |
| 426783006 | sinus rhythm | 窦性心律：窦房结主导心律；仅为正常候选，不等同完全正常。 | NSR | 窦性心律仅作为正常候选；需同条 ECG 无其他异常标签时才判为 NORM。 | 赞成RGQ |

## 3.5 被映射为 STTC（ST-T/QT/缺血相关改变）的 PN2021 标签

映射原则：ST 段、T 波、QT 间期异常及心肌缺血相关标签，归入 STTC。

| **SNOMED code** | **PN2021 标签名** | **中文译名/简短医学语义说明** | **缩写** | **医学语义/拟映射理由** | **医生意见** |
| --- | --- | --- | --- | --- | --- |
| 426434006 | anterior ischemia | 前壁缺血：前壁心肌缺血相关心电改变。 | AnMIs | 心肌缺血相关电图改变，拟归入 STTC，而不是 MI。 | 均赞成RGQ |
| 425419005 | inferior ischaemia | 下壁缺血：下壁心肌缺血相关心电改变。 | IIs | 心肌缺血相关电图改变，拟归入 STTC，而不是 MI。 |  |
| 425623009 | lateral ischaemia | 侧壁缺血：侧壁心肌缺血相关心电改变。 | LIs | 心肌缺血相关电图改变，拟归入 STTC，而不是 MI。 |  |
| 164861001 | myocardial ischemia | 心肌缺血：心肌供血不足相关心电或诊断语义。 | MIs | 心肌缺血相关电图改变，拟归入 STTC，而不是 MI。 |  |
| 428750005 | nonspecific st t abnormality | 非特异性 ST-T 异常：ST 段/T 波改变，特异性有限。 | NSSTTA | ST 段改变/压低/抬高，属于 ST-T 改变，拟归入 STTC。 |  |
| 111975006 | prolonged qt interval | QT 间期延长：心室复极时程延长。 | LQT | QT 间期异常，属于复极/时程异常，拟归入 STTC。 |  |
| 55930002 | s t changes | ST 段改变：ST 段抬高、压低或形态异常的笼统标签。 | STC | ST 段、T 波或复极异常相关标签，拟归入 STTC。 |  |
| 429622005 | st depression | ST 段压低：常见缺血或复极异常表现。 | STD | ST 段改变/压低/抬高，属于 ST-T 改变，拟归入 STTC。 |  |
| 164931005 | st elevation | ST 段抬高：可见于急性损伤、早复极等多种情况。 | STE | ST 段改变/压低/抬高，属于 ST-T 改变，拟归入 STTC。 |  |
| 164930006 | st interval abnormal | ST 间期异常：ST 段/间期形态或位置异常。 | STIAb | ST 段改变/压低/抬高，属于 ST-T 改变，拟归入 STTC。 |  |
| 164934002 | t wave abnormal | T 波异常：心室复极异常的笼统表现。 | TAb | T 波异常或倒置，属于 ST-T 改变，拟归入 STTC。 |  |
| 59931005 | t wave inversion | T 波倒置：复极异常或缺血相关表现之一。 | TInv | T 波异常或倒置，属于 ST-T 改变，拟归入 STTC。 |  |

# 四、未直接归入 Super5 五类的标签清单

> **本节用途**
>
> 下面这些标签当前没有被直接置为 CD/HYP/MI/NORM/STTC 的阳性标签。
> 请医生逐项判断：是否应补充归入 CD/HYP/MI/NORM/STTC；如果不适合归类，请判断是否至少应作为异常状态标签用于取消 NORM，或保持完全未归类。

## 4.1 当前仅用于取消 NORM、但未直接归入五类的标签

当前处理：这些标签不归类到 CD/HYP/MI/STTC=1；如果某条 ECG 出现这些标签，会判定为异常状态，不归入 NORM。<mark>请医生审核并判断其中是否有标签应补充映射到五类。</mark>

| **SNOMED code** | **PN2021 标签名** | **中文译名/简短医学语义说明** | **缩写** | **当前处理原因** | **审核意见** |
| --- | --- | --- | --- | --- | --- |
| 164889003 | atrial fibrillation | 心房颤动：房性快速不规则节律，RR 间期绝对不齐。 | AF | 房颤/房扑类节律异常，不属于 PTB-XL Super5 主类。 | 保持外部，取消 NORM |
| 164890007 | atrial flutter | 心房扑动：心房快速折返节律，可见扑动波。 | AFL | 房颤/房扑类节律异常，不属于 PTB-XL Super5 主类。 | 保持外部，取消 NORM |
| 426627000 | bradycardia | 心动过缓：心率低于正常范围的心率/节律表现。 | Brady | 节律/心率变异，不等同 PTB-XL NORM。 | 保持外部，取消 NORM |
| 39732003 | left axis deviation | 左电轴偏移：额面 QRS 电轴偏左。 | LAD | 轴偏，不能稳定归入 Super5。 | 保持外部，取消 NORM |
| 251146004 | low qrs voltages | QRS 低电压：QRS 波群振幅降低。 | LQRSV | 形态/电压异常，取消 NORM 但不强制映射。 | 保持外部，取消 NORM |
| 10370003 | pacing rhythm | 起搏心律：起搏器驱动或参与的心律，形态受起搏影响。 | PR | pacing/device rhythm：改变形态，但不等同 PTB-XL CD 诊断子类。 | 保持外部，取消 NORM |
| 365413008 | poor R wave Progression | R 波递增不良：胸前导联 R 波递增不足。 | PRWP | 形态/电压异常，取消 NORM 但不强制映射。 | 保持外部，取消 NORM |
| 284470004 | premature atrial contraction | 房性早搏：提前出现的房性激动。 | PAC | 早搏/异位搏动，不能稳定归入 Super5。 | 保持外部，取消 NORM |
| 427172004 | premature ventricular contractions | 室性早搏：提前出现的室性激动，QRS 常宽大畸形。 | PVC | 早搏/异位搏动，不能稳定归入 Super5。 | 保持外部，取消 NORM |
| 164917005 | qwave abnormal | Q 波异常：异常 Q 波或病理性 Q 波可疑表现。 | QAb | 边界形态标签：主分析保守处理，建议敏感性分析。 | 保持外部，取消 NORM |
| 47665007 | right axis deviation | 右电轴偏移：额面 QRS 电轴偏右。 | RAD | 轴偏，不能稳定归入 Super5。 | 保持外部，取消 NORM |
| 427393009 | sinus arrhythmia | 窦性心律不齐：窦性节律下 RR 间期周期性变异。 | SA | 节律/心率变异，不等同 PTB-XL NORM。 | 保持外部，取消 NORM |
| 426177001 | sinus bradycardia | 窦性心动过缓：窦性节律且心率偏慢。 | SB | 节律/心率变异，不等同 PTB-XL NORM。 | 保持外部，取消 NORM |
| 427084000 | sinus tachycardia | 窦性心动过速：窦性节律且心率偏快。 | STach | 节律/心率变异，不等同 PTB-XL NORM。 | 保持外部，取消 NORM |
| 63593006 | supraventricular premature beats | 室上性早搏：希氏束以上来源的提前激动。 | SVPB | 早搏/异位搏动，不能稳定归入 Super5。 | 保持外部，取消 NORM |
| 17338001 | ventricular premature beats | 室性早搏：心室来源的提前激动。 | VPB | 早搏/异位搏动，不能稳定归入 Super5。 | 保持外部，取消 NORM |
| 164951009 | abnormal QRS | QRS 波群异常：QRS 时限、形态或振幅异常的笼统描述。 | abQRS | 形态/电压异常，取消 NORM 但不强制映射。 | 保持外部，取消 NORM |
| 233892002 | accelerated atrial escape rhythm | 加速性房性逸搏心律：房性逸搏频率高于通常逸搏范围。 | AAR | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 保持外部，取消 NORM |
| 61277005 | accelerated idioventricular rhythm | 加速性室性自主心律：室性自主节律，频率低于典型室速。 | AIVR | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 保持外部，取消 NORM |
| 426664006 | accelerated junctional rhythm | 加速性交界性心律：房室交界区起搏点加速。 | AJR | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 保持外部，取消 NORM |
| 251173003 | atrial bigeminy | 房性二联律：房早与窦性搏动交替或规律出现。 | AB | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 251187003 | atrial escape beat | 房性逸搏：窦性冲动延迟后的房性补充搏动。 | AED | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 195080001 | atrial fibrillation and flutter | 房颤/房扑：心房快速节律异常的合并标签。 | AFAFL | 房颤/房扑类节律异常，不属于 PTB-XL Super5 主类。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 251268003 | atrial pacing pattern | 心房起搏图形：心房起搏刺激引发的心电图表现。 | AP | pacing/device rhythm：改变形态，但不等同 PTB-XL CD 诊断子类。 | 保持外部，取消 NORM |
| 713422000 | atrial tachycardia | 房性心动过速：起源于心房的快速心律。 | ATach | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 251166008 | atrioventricular node reentrant tachycardia | 房室结折返性心动过速：AVNRT，室上性折返性心动过速。 | AVNRT | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 50799005 | atrioventricular dissociation | 房室分离：心房与心室激动互不相关。 | AVD | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 严重的窦房结/全局节律异常（保持外部，取消 NORM） |
| 29320008 | atrioventricular junctional rhythm | 房室交界性心律：起源于房室交界区的节律。 | AVJR | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 233897008 | atrioventricular reentrant tachycardia | 房室折返性心动过速：旁路参与的折返性室上速。 | AVRT | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 251170000 | blocked premature atrial contraction | 未下传房性早搏：房早未传至心室，可造成长间歇。 | BPAC | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 74615001 | brady tachy syndrome | 快慢综合征：心动过缓与快速心律失常交替。 | BTS | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 严重的窦房结/全局节律异常（保持外部，取消 NORM） |
| 698247007 | cardiac dysrhythmia | 心律失常：笼统的心脏节律异常诊断。 | CD | Super5 外标签，取消 NORM 但不作为主分析阳性。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 426749004 | chronic atrial fibrillation | 慢性房颤：持续或长期存在的心房颤动。 | CAF | 房颤/房扑类节律异常，不属于 PTB-XL Super5 主类。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 251198002 | clockwise rotation | 顺钟向转位：胸前导联过渡区后移的转位表现。 | CR | Super5 外标签，取消 NORM 但不作为主分析阳性。 | <span style="color:#00B050">完全放弃，不参与标签映射</span> |
| 251199005 | countercolockwise rotation | 逆钟向转位：胸前导联过渡区前移的转位表现。 | CCR | Super5 外标签，取消 NORM 但不作为主分析阳性。 | <span style="color:#00B050">完全放弃，不参与标签映射</span> |
| 428417006 | early repolarization | 早复极：J 点/ST 段抬高样复极表现，多为边界或良性模式。 | ERe | 边界形态标签：主分析保守处理，建议敏感性分析。 | <span style="color:#00B050">完全放弃，不参与标签映射</span> |
| 55827005 | left ventricular high voltage | 左室高电压：QRS 电压增高，可能提示左室肥厚但不等同。 | LVHV | high voltage：电压表现，证据弱于 hypertrophy/enlargement。 | <span style="color:#EE0000">映射至 HYP</span> |
| 67751000119106 | right atrial high voltage | 右房高电压：P 波/房性电压增高，可能提示右房负荷。 | RAHV | high voltage：电压表现，证据弱于 hypertrophy/enlargement。 | <span style="color:#EE0000">映射至 HYP</span> |
| 251266004 | ventricular pacing pattern | 心室起搏图形：心室起搏导致宽 QRS 及继发 ST-T 改变。 | VPP | pacing/device rhythm：改变形态，但不等同 PTB-XL CD 诊断子类。 | 保持外部，取消 NORM |

## 4.2 当前完全未纳入五类、也未用于取消 NORM 的标签

当前处理：这些标签既不是五类阳性，也不会主动取消 NORM。<mark>请医生审核判断其中是否有需要补充到五类标签体系中。</mark>

| **SNOMED code** | **PN2021 标签名** | **中文译名/简短医学语义说明** | **缩写** | **初步判断提示** | **审核意见** |
| --- | --- | --- | --- | --- | --- |
| 413444003 | acute myocardial ischemia | 急性心肌缺血：急性缺血相关心电/临床语义，常与 ST-T 改变相关。 | AMIs | 可能涉及 ST-T/QT/缺血或复极异常；请医生判断是否归入 STTC 或仅作为排除正常。 | <span style="color:#EE0000">归入 STTC，未发展为梗死的缺血应放在 STTC 范畴</span> |
| 106068003 | atrial rhythm | 房性心律：窦房结外心房起搏点主导心律。 | ARH | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 418818005 | brugada | Brugada 图形/综合征：右胸导联特征性 ST 抬高及传导异常相关。 | BRU | 可能涉及传导系统或特定心电综合征；请判断是否归入 CD 或保持外部标签。 | <span style="color:#EE0000">归入 CD，临床常将其视为特殊的心脏电生理传导传导系统疾病</span> |
| 413844008 | chronic myocardial ischemia | 慢性心肌缺血：慢性缺血相关语义，常需结合 ST-T 改变判断。 | CMI | 可能涉及 ST-T/QT/缺血或复极异常；请医生判断是否归入 STTC 或仅作为排除正常。 | <span style="color:#EE0000">归入 STTC，明确的缺血相关心电改变</span> |
| 61721007 | clockwise or counterclockwise vectorcardiographic loop | 心电向量环顺/逆钟向转位：电轴或转位形态描述。 | CVCL/CCVCL | 形态、轴向、导联或技术性标签；请判断是否仅排除正常或保持未归类。 | <span style="color:#00B050">完全放弃、不参与标签映射</span> |
| 204384007 | congenital incomplete atrioventricular heart block | 先天性不完全性房室传导阻滞：先天性房室传导部分阻滞。 | CIAHB | 可能涉及传导系统或特定心电综合征；请判断是否归入 CD 或保持外部标签。 | <span style="color:#EE0000">归入 CD，属于明确的房室传导阻滞范畴</span> |
| 53741008 | coronary heart disease | 冠状动脉粥样硬化性心脏病：临床疾病/病史诊断，非特异 ECG 模式。 | CHD | 更像临床病史/疾病诊断，不一定是 ECG 五类模式；请判断是否保持未归类。 | <span style="color:#00B050">完全放弃、不参与标签映射</span> |
| 77867006 | decreased qt interval | QT 间期缩短：心室复极时程缩短。 | SQT | 可能涉及 ST-T/QT/缺血或复极异常；请医生判断是否归入 STTC 或仅作为排除正常。 | <span style="color:#EE0000">归入 STTC，属于 QT 间期/复极时程异常</span> |
| 82226007 | diffuse intraventricular block | 弥漫性室内传导阻滞：广泛室内传导延迟。 | DIB | 可能涉及传导系统或特定心电综合征；请判断是否归入 CD 或保持外部标签。 | <span style="color:#EE0000">归入 CD，属于明确的室内传导延迟</span> |
| 164942001 | fqrs wave | 碎裂 QRS 波：QRS 切迹/碎裂，提示传导不均一或瘢痕可能。 | FQRS | 形态、轴向、导联或技术性标签；请判断是否仅排除正常或保持未归类。 | 边缘波形与形态描述性标签（保持外部，取消 NORM） |
| 13640000 | fusion beats | 融合波：室上性与室性激动同时作用形成的 QRS。 | FB | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 84114007 | heart failure | 心力衰竭：临床综合征，非单一 ECG 形态诊断。 | HF | 更像临床病史/疾病诊断，不一定是 ECG 五类模式；请判断是否保持未归类。 | <span style="color:#00B050">完全放弃、不参与标签映射</span> |
| 368009 | heart valve disorder | 心脏瓣膜病：结构性/临床诊断，ECG 特异性有限。 | HVD | 更像临床病史/疾病诊断，不一定是 ECG 五类模式；请判断是否保持未归类。 | <span style="color:#00B050">完全放弃、不参与标签映射</span> |
| 251259000 | high t-voltage | T 波高电压/高尖 T 波：T 波振幅增高，属复极表现。 | HTV | 可能涉及 ST-T/QT/缺血或复极异常；请医生判断是否归入 STTC 或仅作为排除正常。 | <span style="color:#EE0000">归入 STTC，高尖 T 波是典型的复极异常表现</span> |
| 49260003 | idioventricular rhythm | 室性自主心律：心室起搏点主导的逸搏或自主节律。 | IR | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 251200008 | indeterminate cardiac axis | 电轴不确定：QRS 电轴难以明确判定。 | ICA | 形态、轴向、导联或技术性标签；请判断是否仅排除正常或保持未归类。 | 边缘波形与形态描述性标签（保持外部，取消 NORM） |
| 704997005 | inferior ST segment depression | 下壁导联 ST 段压低：II、III、aVF 等导联 ST 压低。 | ISTD | 可能涉及 ST-T/QT/缺血或复极异常；请医生判断是否归入 STTC 或仅作为排除正常。 | <span style="color:#EE0000">归入 STTC，明确的 ST 段改变</span> |
| 426995002 | junctional escape | 交界性逸搏：房室交界区补充搏动。 | JE | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 251164006 | junctional premature complex | 交界性早搏：房室交界区提前激动。 | JPC | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 426648003 | junctional tachycardia | 交界性心动过速：房室交界区快速节律。 | JTach | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 253352002 | left atrial abnormality | 左房异常：P 波形态提示左房负荷或异常。 | LAA | 可能提示心房异常/负荷改变；请判断是否归入 HYP、仅取消 NORM 或保持未归类。 | <span style="color:#EE0000">归入 HYP。在心电图上，左房异常（如 P 波时限延长、双峰）是左心房肥大或扩大的直接心电表征</span> |
| 370365005 | left ventricular strain | 左室劳损/应变：常见于左室肥厚相关继发 ST-T 改变。 | LVS | 可能涉及 ST-T/QT/缺血或复极异常；请医生判断是否归入 STTC 或仅作为排除正常。 | <span style="color:#EE0000">归入 STTC。左室劳损在 ECG 上表现为特征性的 ST 段压低和 T 波倒置，虽然它常继发于左室肥厚（HYP），但其直接波形改变属于 ST-T 异常</span> |
| 164912004 | p wave change | P 波改变：P 波形态、时限或电压异常。 | PWC | 可能提示心房异常/负荷改变；请判断是否归入 HYP、仅取消 NORM 或保持未归类。 | <span style="color:#EE0000">映射至 HYP</span> |
| 251182009 | paired ventricular premature complexes | 成对室早：连续两个室性早搏。 | VPVC | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 282825002 | paroxysmal atrial fibrillation | 阵发性房颤：间歇性发作的心房颤动。 | PAF | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 67198005 | paroxysmal supraventricular tachycardia | 阵发性室上性心动过速：突发突止的室上性快速节律。 | PSVT | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 425856008 | paroxysmal ventricular tachycardia | 阵发性室性心动过速：突发性室性快速节律。 | PVT | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 251205003 | prolonged P wave | P 波增宽/延长：提示心房内传导延迟或房房阻滞可能。 | PPW | 可能提示心房异常/负荷改变；请判断是否归入 HYP、仅取消 NORM 或保持未归类。 | 边缘波形与形态描述性标签（保持外部，取消 NORM） |
| 164921003 | r wave abnormal | R 波异常：R 波振幅、形态或进展异常。 | RAb | 形态、轴向、导联或技术性标签；请判断是否仅排除正常或保持未归类。 | 边缘波形与形态描述性标签（保持外部，取消 NORM） |
| 314208002 | rapid atrial fibrillation | 快速房颤：房颤伴快速心室率。 | RAF | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 253339007 | right atrial abnormality | 右房异常：P 波电压或形态提示右房负荷。 | RAAb | 可能提示心房异常/负荷改变；请判断是否归入 HYP、仅取消 NORM 或保持未归类。 | <span style="color:#EE0000">归入 HYP，右房异常（如 P 波高尖）直接对应右心房肥大或扩大（肺型 P 波）</span> |
| 49578007 | shortened pr interval | PR 间期缩短：房室传导提前或预激可能。 | SPRI | 可能涉及传导系统或特定心电综合征；请判断是否归入 CD 或保持外部标签。 | <span style="color:#EE0000">归入 CD，PR 间期缩短通常提示存在房室旁路（如预激综合征相关异常）或房室结内传导加速，与原策略中将预激图形归入 CD 的逻辑一致</span> |
| 65778007 | sinoatrial block | 窦房传导阻滞：窦房结冲动传出受阻。 | SAB | 可能涉及传导系统或特定心电综合征；请判断是否归入 CD 或保持外部标签。 | <span style="color:#EE0000">归入 CD，由于窦房结周围组织传导障碍导致，属于明确的传导系统异常</span> |
| 5609005 | sinus arrest | 窦性停搏：窦房结暂停发放冲动。 | SARR | 当前未直接映射到 Super5；请医生判断是否可归类、仅取消 NORM，或保持未归类。 | 严重的窦房结/全局节律异常（保持外部，取消 NORM） |
| 17366009 | sinus atrium to atrial wandering rhythm | 窦房结至心房游走性心律：起搏点在窦房结/心房内游走。 | SAAWR | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 60423000 | sinus node dysfunction | 窦房结功能不全：窦房结起搏或传导功能异常。 | SND | 当前未直接映射到 Super5；请医生判断是否可归类、仅取消 NORM，或保持未归类。 | 严重的窦房结/全局节律异常（保持外部，取消 NORM） |
| 251168009 | supraventricular bigeminy | 室上性二联律：室上性早搏呈二联律。 | SVB | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 426761007 | supraventricular tachycardia | 室上性心动过速：起源于希氏束以上的快速心律。 | SVT | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 251139008 | suspect arm ecg leads reversed | 疑似上肢导联反接：左右上肢导联位置可能接反。 | ALR | 形态、轴向、导联或技术性标签；请判断是否仅排除正常或保持未归类。 | <span style="color:#00B050">完全放弃、不参与标签映射</span> |
| 251223006 | tall p wave | <mark>P 波高尖：常提示右房负荷或肺型 P 波。</mark> | <mark>TPW</mark> | <mark>可能提示心房异常/负荷改变；请判断是否归入 HYP、仅取消 NORM 或保持未归类。</mark> | <span style="color:#EE0000">归入 HYP，右房异常（P 波高尖）直接对应右心房肥大或扩大（肺型 P 波）RGQ/</span><mark style="background-color:#40E0D0"><span style="color:#EE0000">SJR修改后</span></mark> |
| 266257000 | transient ischemic attack | 短暂性脑缺血发作：神经系统临床诊断，非 ECG 标签。 | TIA | 更像临床病史/疾病诊断，不一定是 ECG 五类模式；请判断是否保持未归类。 | <span style="color:#00B050">完全放弃、不参与标签映射</span> |
| 164937009 | u wave abnormal | U 波异常：U 波形态或振幅异常，属复极相关表现。 | UAb | 可能涉及 ST-T/QT/缺血或复极异常；请医生判断是否归入 STTC 或仅作为排除正常。 | <span style="color:#EE0000">归入 STTC，U 波异常（如倒置或显著增高）属于心室复极晚期的异常改变，纳入 STTC 范畴最为合理</span> |
| 11157007 | ventricular bigeminy | 室性二联律：室早规律性每隔一搏出现。 | VBig | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 164884008 | ventricular ectopics | 室性异位搏动：室性早搏或室性异位激动总称。 | VEB | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 75532003 | ventricular escape beat | 室性逸搏：心室补充搏动。 | VEsB | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 81898007 | ventricular escape rhythm | 室性逸搏心律：连续室性逸搏主导节律。 | VEsR | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 164896001 | ventricular fibrillation | 心室颤动：快速紊乱的室性电活动，属严重心律失常。 | VF | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 111288001 | ventricular flutter | 心室扑动：快速相对规则的室性扑动样电活动。 | VFL | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 164895002 | ventricular tachycardia | 室性心动过速：连续室性搏动导致的快速心律。 | VTach | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
| 251180001 | ventricular trigeminy | 室性三联律：每两个正常搏动后出现室早等三联节律。 | VTrig | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 早搏 / 规律性联律 / 逸搏（保持外部，取消 NORM） |
| 195101003 | wandering atrial pacemaker | 游走性房性起搏点：心房内起搏点位置变化，P 波形态多变。 | WAP | 主要为节律/异位搏动类标签；请判断是否保持 Super5 外、仅取消 NORM，或另设节律类分析。 | 异位节律 / 阵发性快速心律失常（保持外部，取消 NORM） |
