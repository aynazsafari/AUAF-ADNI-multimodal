from __future__ import annotations

import numpy as np
import pytest
import torch

from auaf_model_factory import build_model


def test_all_model_variants_handle_both_and_single_modalities():
    torch.manual_seed(2026)
    dimensions = {"clinical_cognitive": 7, "fluid_biomarker": 3}
    inputs = {
        "clinical_cognitive": torch.randn(6, 7),
        "fluid_biomarker": torch.randn(6, 3),
    }
    masks = torch.tensor([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    for variant in ["EarlyFusion", "LateFusion", "AttentionFusion", "AUAF"]:
        model = build_model(variant, dimensions, n_classes=3)
        for training in [False, True]:
            model.train(training)
            with torch.set_grad_enabled(training):
                logits = model(inputs, masks)
            assert tuple(logits.shape) == (6, 3)
            assert torch.isfinite(logits).all()

        model.eval()
        with pytest.raises(ValueError):
            model(inputs, torch.zeros_like(masks))

        if variant == "AUAF":
            result = model.mc_forward(inputs, masks, n_passes=4)
            for key in ["mean_prob", "std_prob", "entropy", "mean_gates"]:
                assert np.isfinite(result[key]).all()
            assert model.training is False
