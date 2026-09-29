"""CNN architecture and training-augmentation checks (no real training)."""

from __future__ import annotations

import numpy as np

from sonic.models.cnn import _batches, build, spec_augment


def test_architecture_shapes_and_size():
    model = build(n_classes=9)
    assert model.input_shape == (None, 64, 63, 1)
    assert model.output_shape == (None, 9)
    assert 1_000_000 < model.count_params() < 1_500_000  # ~1.2 M parameters, as planned
    out = model.predict(np.zeros((2, 64, 63, 1), np.float32), verbose=0)
    assert np.allclose(out.sum(axis=1), 1, atol=1e-5)


def test_spec_augment_masks_stripes_without_changing_shape():
    x = np.ones((4, 64, 63, 1), np.float32)
    out = spec_augment(x, np.random.default_rng(0), masks=2, max_width=8)
    assert out.shape == x.shape
    assert (out == 0).any() and (out == 1).any()
    assert (x == 1).all()  # the input is not modified


def test_mixup_batches_have_valid_soft_labels():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((130, 64, 63, 1)).astype(np.float32)
    y = rng.integers(0, 3, 130)
    batches = list(_batches(X, y, 3, mixup=0.2, rng=rng))
    assert sum(len(xb) for xb, _ in batches) == 130
    for _, yb in batches:
        assert np.allclose(yb.sum(axis=1), 1)
        assert (yb >= 0).all()
