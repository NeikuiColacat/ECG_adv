from util.advdiff_attack import AdvDiffAttacker

attacker = AdvDiffAttacker(device='cuda:0')

attacker.attack(
    target_label=2,  # MI
    batch_size=8,
    num_inference_steps=100,
    adversarial_guidance_scale=10.0,   # 新的 scale（归一化后）
    noise_sampling_guidance_scale=2.0,
    num_noise_sampling_steps=5,
)