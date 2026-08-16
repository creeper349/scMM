import numpy as np
import pytest

from scMM.util.denoise import peak_recon, r1_decomposition


def test_rank_one_decomposition_handles_zero_matrix() -> None:
    a, b = r1_decomposition(np.zeros((3, 2)))

    np.testing.assert_array_equal(a, np.zeros((3, 1)))
    np.testing.assert_array_equal(b, np.zeros((2, 1)))


def test_rank_one_decomposition_reconstructs_rank_one_input() -> None:
    matrix = np.array([[2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])

    a, b = r1_decomposition(matrix)

    np.testing.assert_allclose(a @ b.T, matrix, rtol=1e-5)


def test_peak_reconstruction_validates_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        peak_recon(np.ones((3, 2)), np.ones((2, 3)))
