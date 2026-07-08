import torch
import numpy as np

from ecg_adv_gen.training.online_buffer import QualityAwareBuffer


def test_quality_score_evicts_separately_from_sampling_weight():
    buf = QualityAwareBuffer(max_size=2)
    x = torch.zeros(12, 8)
    y = torch.zeros(5)

    buf.add_one(x, y, score=0.90, sample_weight=0.10)
    buf.add_one(x + 1, y, score=0.40, sample_weight=1.00)
    buf.add_one(x + 2, y, score=0.80, sample_weight=1.00)

    assert buf.score_list == [0.90, 0.80]
    assert np.allclose(buf.get_sampling_weights(), [0.09, 0.80])
