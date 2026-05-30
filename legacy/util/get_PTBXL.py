import pickle
import sys
from pathlib import Path

sys.path.append("/root/ECG_adv_Gen/model/ecg_ptbxl_benchmarking/code/")

import torch
from experiments.scp_experiment import SCP_Experiment
from torch.utils.data import DataLoader, TensorDataset



# 使用作者的类获取数据


def get_ecg_dataset():
    """
    索引 0: CD
    索引 1: HYP
    索引 2: MI
    索引 3: NORM
    索引 4: STTC
    """

    exp = SCP_Experiment(
        experiment_name="exp1.1.1",
        task="superdiagnostic",
        datafolder="/root/ECG_adv_Gen/model/ecg_ptbxl_benchmarking/data/ptbxl/",  # 绝对路径
        outputfolder="/root/ECG_adv_Gen/model/ecg_ptbxl_benchmarking/output/",  # 绝对路径
        models=[],
        sampling_frequency=100,
    )
    exp.prepare()

    # 转换为 PyTorch Dataset 和 DataLoader
    def create_dataloader(X, y, batch_size=1, shuffle=False):
        # numpy (N, time, channels) -> torch (N, channels, time)
        X_tensor = torch.from_numpy(X).float().transpose(1, 2)
        y_tensor = torch.from_numpy(y).float()

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )
        return loader

    def get_norm_scaler():
        scaler_path = (
            f"{exp.outputfolder}{exp.experiment_name}/data/standard_scaler.pkl"
        )
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return scaler

    # 创建三个 DataLoader
    train_loader = create_dataloader(exp.X_train, exp.y_train, shuffle=False)
    val_loader = create_dataloader(exp.X_val, exp.y_val, shuffle=False)
    test_loader = create_dataloader(exp.X_test, exp.y_test, shuffle=False)

    scaler = get_norm_scaler()
    mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    std = torch.tensor(scaler.scale_, dtype=torch.float32)

    # mlb_path = f'{exp.outputfolder}{exp.experiment_name}/data/mlb.pkl'
    # with open(mlb_path, 'rb') as f:
    #     mlb = pickle.load(f)

    # print("疾病类别对应关系:")
    # for idx, class_name in enumerate(mlb.classes_):
    #     print(f"  索引 {idx}: {class_name}")

    return train_loader, val_loader, test_loader, mean, std


def get_resnet1d_wang(
    num_classes=5,
    input_channels=12,
    weight_path="/root/ECG_adv_Gen/model/ecg_ptbxl_benchmarking/output/exp1.1.1/models/fastai_resnet1d_wang/models/fastai_resnet1d_wang.pth",
    device="cuda",
):
    from models.resnet1d import resnet1d_wang

    # 创建 resnet1d_wang 模型，参数与 fastai_model 中的默认值相同
    model  = resnet1d_wang(
        num_classes=num_classes,
        input_channels=input_channels,
        inplanes=128,  # 作者hardcode的默认值
        kernel_size=[5, 3],  # wang特定：kernel_size会被设置为[5,3]
        ps_head=0.5,  # 分类头dropout率
        lin_ftrs_head=[128],  # 分类头隐藏层维度
        # 以下是 resnet1d_wang 内部自动设置的参数
        # kernel_size_stem=7,
        # stride_stem=1,
        # pooling_stem=False,
    )

    if weight_path is not None:
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")

        # 加载权重文件
        checkpoint = torch.load(str(weight_path), map_location=device)

        # 处理权重格式
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # 加载到模型
        model.load_state_dict(state_dict, strict=True)
        print(f"✓ 权重已加载: {weight_path}")
    else:
        assert False

    model = model.to(device)
    model.eval()

    return model
