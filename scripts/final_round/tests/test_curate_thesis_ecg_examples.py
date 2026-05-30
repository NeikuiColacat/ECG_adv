import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.final_round import curate_thesis_ecg_examples as curate


def test_write_thesis_composite_png_stacks_class_figures(tmp_path):
    pngs = []
    for idx, class_name in enumerate(["NORM", "MI", "STTC"]):
        path = tmp_path / f"thesis_{class_name}_12lead.png"
        Image.new("RGB", (120 + idx * 10, 80), color=(240 - idx, 240, 240)).save(path)
        pngs.append(path)

    out_path = tmp_path / "thesis_synthetic_12lead.png"
    curate.write_thesis_composite_png(pngs, out_path, width_px=180, gap_px=12)

    assert out_path.exists()
    image = Image.open(out_path)
    assert image.width == 180
    assert image.height > 80 * len(pngs)


def test_resolve_default_synth_npz_prefers_repo_sample(tmp_path):
    repo_npz = tmp_path / "repo_sample.npz"
    data_npz = tmp_path / "data_sample.npz"
    repo_npz.write_bytes(b"repo")
    data_npz.write_bytes(b"data")

    assert curate.resolve_default_synth_npz(repo_npz, data_npz) == repo_npz
