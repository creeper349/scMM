import numpy as np
import pytest

from scMM.util.normalize import normalize


def test_total_normalization_preserves_zero_rows() -> None:
    data = np.array([[1.0, 3.0], [0.0, 0.0]])

    result = normalize(data, "total", {"scale": 100.0})

    np.testing.assert_allclose(result, [[25.0, 75.0], [0.0, 0.0]])


def test_max_normalization_does_not_divide_by_zero() -> None:
    data = np.array([[0.0, 0.0], [2.0, 4.0]])

    result = normalize(data, "max")

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, [[0.0, 0.0], [0.5, 1.0]])


def test_pqn_rejects_reference_with_wrong_shape() -> None:
    with pytest.raises(ValueError, match="one value per feature"):
        normalize(np.ones((2, 3)), "pqn", {"reference": [1.0, 2.0]})


def test_normalize_validates_method_and_options() -> None:
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize(np.ones((2, 2)), "missing")
    with pytest.raises(TypeError, match="mapping"):
        normalize(np.ones((2, 2)), norm_kwargs=[])  # type: ignore[arg-type]
