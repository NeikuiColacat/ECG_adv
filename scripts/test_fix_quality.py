#!/usr/bin/env python
"""
Quick test: verify the 3 bug fixes produce clean adversarial ECGs.

Generates ~10 samples with adversarial guidance and compares:
1. Latent L2 norm (should be ~4-5, not 10+)
2. ECG amplitude range (should be ±2mV, not ±20mV)
3. Visual quality via ecg_plot
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("/root/ECG_adv_Gen/outputs/mi_experiment/fix_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PTBXL_500MI = "/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_500MI.pt"
DEVICE = "cuda"


def main():
    from adversarial.adv_generate import (
        BoundaryAdvDiffGenerator,
        DEFAULT_HYPERPARAMS,
        prepare_conditions_for_prompt,
    )
    from adversarial.label_mapping import PROMPT_TO_PATTERN_INDICES
    from adversarial.efficientnet_victim import EfficientNetVictim
    from util.ecgtwin_utils import ECGTwinWrapper

    print("=" * 60)
    print("Bug Fix Verification Test")
    print("=" * 60)

    # Load PTBXL encoded data
    print("\nLoading PTBXL encoded data...")
    ptbxl_items = torch.load(PTBXL_500MI, map_location="cpu")
    mi_items = [it for it in ptbxl_items if it["label"].get("diagnostic_class") == "MI"]
    print(f"  {len(mi_items)} MI reference samples")

    # Load models
    print("Loading ECGTwin...")
    ecgtwin = ECGTwinWrapper(device=DEVICE, load_encoder=False, load_text_model=True)

    print("Loading EfficientNet...")
    victim = EfficientNetVictim(device=DEVICE, ecgtwin_wrapper=ecgtwin)

    # Test config
    prompt_key = "MIMIC_MI_inferior"
    text_prompt = "inferior myocardial infarction|abnormal ecg"
    target_indices = PROMPT_TO_PATTERN_INDICES[prompt_key]
    batch_size = 4

    print(f"\nPrompt: {prompt_key}")
    print(f"Target indices: {target_indices}")

    # --- Pure ECGTwin (no adversarial) ---
    print("\n--- Pure ECGTwin (no adversarial) ---")
    hp_pure = dict(DEFAULT_HYPERPARAMS)
    hp_pure["num_inference_steps"] = 50
    hp_pure["adversarial_guidance_scale"] = 0.0
    hp_pure["noise_sampling_guidance_scale"] = 0.0
    hp_pure["num_noise_sampling_steps"] = 1
    hp_pure["acceptance_range"] = [0.0, 1.0]  # accept all

    gen_pure = BoundaryAdvDiffGenerator(
        ecgtwin_wrapper=ecgtwin,
        victim=victim,
        device=DEVICE,
        hyperparams=hp_pure,
    )

    conditions = prepare_conditions_for_prompt(
        ecgtwin=ecgtwin,
        text_prompt=text_prompt,
        reference_items=mi_items,
        batch_size=batch_size,
        device=DEVICE,
    )

    result_pure = gen_pure.generate_batch(conditions, target_indices, batch_size=batch_size)
    pure_ecg = None
    pure_latent_norm = None
    if result_pure.get("latents") is not None and result_pure["latents"].shape[0] > 0:
        pure_latent_norm = result_pure["latents"].norm(p=2, dim=(1, 2)).mean().item()
        pure_ecg = result_pure["ecg"]
        print(f"  Accepted: {pure_ecg.shape[0]} samples")
        print(f"  Latent L2 norm: {pure_latent_norm:.3f}")
        print(f"  ECG amplitude: min={pure_ecg.min():.3f}, max={pure_ecg.max():.3f}, "
              f"std={pure_ecg.std():.4f}")
        print(f"  Target probs: {result_pure['probs'][:, target_indices].mean(dim=0).tolist()}")
    else:
        print("  No samples generated")

    # --- Adversarial (with fixes) ---
    print("\n--- Adversarial (with bug fixes) ---")
    hp_adv = dict(DEFAULT_HYPERPARAMS)
    hp_adv["num_inference_steps"] = 50
    hp_adv["adversarial_guidance_scale"] = 0.03   # 3% of latent norm per step
    hp_adv["noise_sampling_guidance_scale"] = 0.01  # 1% of x_T norm
    hp_adv["num_noise_sampling_steps"] = 3
    hp_adv["guidance_start_fraction"] = 0.8
    hp_adv["acceptance_range"] = [0.3, 0.7]

    gen_adv = BoundaryAdvDiffGenerator(
        ecgtwin_wrapper=ecgtwin,
        victim=victim,
        device=DEVICE,
        hyperparams=hp_adv,
    )

    conditions = prepare_conditions_for_prompt(
        ecgtwin=ecgtwin,
        text_prompt=text_prompt,
        reference_items=mi_items,
        batch_size=batch_size,
        device=DEVICE,
    )

    result_adv = gen_adv.generate_batch(conditions, target_indices, batch_size=batch_size)
    adv_ecg = None
    adv_latent_norm = None
    if result_adv.get("latents") is not None and result_adv["latents"].shape[0] > 0:
        adv_latent_norm = result_adv["latents"].norm(p=2, dim=(1, 2)).mean().item()
        adv_ecg = result_adv["ecg"]
        print(f"  Accepted: {adv_ecg.shape[0]} samples")
        print(f"  Latent L2 norm: {adv_latent_norm:.3f}")
        print(f"  ECG amplitude: min={adv_ecg.min():.3f}, max={adv_ecg.max():.3f}, "
              f"std={adv_ecg.std():.4f}")
        print(f"  Target probs: {result_adv['probs'][:, target_indices].mean(dim=0).tolist()}")
    else:
        print("  No samples accepted (try widening acceptance_range)")

    # --- Diagnostic: measure raw gradient magnitude ---
    print("\n--- Gradient Magnitude Diagnostic ---")
    conditions = prepare_conditions_for_prompt(
        ecgtwin=ecgtwin,
        text_prompt=text_prompt,
        reference_items=mi_items,
        batch_size=batch_size,
        device=DEVICE,
    )
    # Generate a single clean latent
    ecgtwin.scheduler.set_timesteps(50)
    x_t = torch.randn(batch_size, 4, 128, device=DEVICE)
    timesteps = ecgtwin.scheduler.timesteps
    for t in timesteps:
        t_batch = t * torch.ones(batch_size, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            noise_pred = ecgtwin.noise_predictor(
                x_t, t_batch,
                conditions["text_embed"], conditions["text_embed_mask"],
                conditions["pat_info"], conditions["base_vector"],
            )
        output = ecgtwin.scheduler.step(model_output=noise_pred, timestep=t, sample=x_t)
        x_t = output["prev_sample"]
    x_0 = x_t

    latent_norm = x_0.norm(p=2, dim=(1, 2)).mean().item()
    print(f"  Clean latent L2 norm: {latent_norm:.3f}")

    # Compute raw gradient and show new step size calculation
    grad = gen_adv._compute_boundary_gradient(x_0, target_indices)
    grad_norm = grad.norm(p=2, dim=(1, 2)).mean().item()
    # New step: guidance_scale * latent_norm (normalized grad direction)
    step_size = hp_adv["adversarial_guidance_scale"] * latent_norm
    print(f"  Raw gradient L2 norm: {grad_norm:.6f}")
    print(f"  Step size (s * latent_norm): {step_size:.4f}")
    print(f"  Step/latent ratio: {hp_adv['adversarial_guidance_scale'] * 100:.1f}% per step")
    n_guidance_steps = int(hp_adv["num_inference_steps"] * (1 - hp_adv["guidance_start_fraction"]))
    print(f"  Over {n_guidance_steps} guidance steps: {hp_adv['adversarial_guidance_scale'] * n_guidance_steps * 100:.1f}%")

    # --- Plot ---
    print("\n--- Plotting ECGs ---")
    try:
        import ecg_plot

        lead_index = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # ecg_plot.save_as_png expects (directory, filename_without_ext)
        # It does: plt.savefig(path + file_name + '.png')
        save_dir = str(OUTPUT_DIR) + "/"

        if pure_ecg is not None and pure_ecg.shape[0] > 0:
            ecg_np = pure_ecg[0].numpy()
            ecg_plot.plot(ecg_np, sample_rate=250, lead_index=lead_index, columns=1, row_height=4)
            ecg_plot.save_as_png(file_name="pure_ecgtwin", path=save_dir)
            print(f"  Saved → {OUTPUT_DIR / 'pure_ecgtwin.png'}")

        if adv_ecg is not None and adv_ecg.shape[0] > 0:
            for i in range(min(3, adv_ecg.shape[0])):
                ecg_np = adv_ecg[i].numpy()
                ecg_plot.plot(ecg_np, sample_rate=250, lead_index=lead_index, columns=1, row_height=4)
                ecg_plot.save_as_png(file_name=f"adv_fixed_{i}", path=save_dir)
                print(f"  Saved → {OUTPUT_DIR / f'adv_fixed_{i}.png'}")

    except Exception as e:
        print(f"  ecg_plot error: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if pure_latent_norm is not None:
        print(f"  Pure latent L2:  {pure_latent_norm:.3f}")
    if adv_latent_norm is not None:
        print(f"  Adv latent L2:   {adv_latent_norm:.3f}")
        if pure_latent_norm is not None:
            ratio = adv_latent_norm / pure_latent_norm
            print(f"  Norm ratio:      {ratio:.2f}x (was 2.58x before fix, should be ~1.0-1.3x)")
    if adv_ecg is not None and adv_ecg.shape[0] > 0:
        amp = adv_ecg.abs().max().item()
        print(f"  Adv ECG max amp: {amp:.3f} mV (normal: ~1-2 mV)")
    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
