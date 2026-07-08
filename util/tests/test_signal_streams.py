import numpy as np

from ecg_adv_gen.training.signal_streams import (
    MemorySignalDataset,
    build_weighted_signal_stream_loader_from_datasets,
)


def test_adv_sample_weights_scale_only_adv_stream_sampler_weights():
    source = MemorySignalDataset(np.zeros((2, 12, 8), dtype=np.float32), np.zeros((2, 5), dtype=np.float32))
    target = MemorySignalDataset(np.zeros((1, 12, 8), dtype=np.float32), np.zeros((1, 5), dtype=np.float32))

    loader = build_weighted_signal_stream_loader_from_datasets(
        source_dataset=source,
        target_dataset=target,
        source_weight=1.0,
        target_real_weight=2.0,
        adv_weight=10.0,
        batch_size=2,
        num_workers=0,
        num_classes=5,
        adv_signals=np.zeros((2, 12, 8), dtype=np.float32),
        adv_labels=np.zeros((2, 5), dtype=np.float32),
        adv_sample_weights=np.asarray([0.1, 1.0], dtype=np.float32),
    )

    assert np.allclose(loader.sampler.weights.numpy(), [1.0, 1.0, 2.0, 1.0, 10.0])
