"""Tests for IC analysis module — synthetic data with known correlations."""

import pytest

from src.research.ic_analysis import compute_ic, _compute_brier


# ── compute_ic ───────────────────────────────────────────────────────────────


class TestComputeIC:
    def test_perfect_positive_correlation(self):
        """Monotonically increasing confidence and returns → IC ≈ 1.0."""
        confidences = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        returns = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        result = compute_ic(confidences, returns)
        assert result.ic == pytest.approx(1.0, abs=0.01)
        assert result.p_value < 0.05
        assert result.n == 7

    def test_perfect_negative_correlation(self):
        """Inversely ordered → IC ≈ -1.0."""
        confidences = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        returns = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        result = compute_ic(confidences, returns)
        assert result.ic == pytest.approx(-1.0, abs=0.01)
        assert result.p_value < 0.05

    def test_no_correlation(self):
        """Unordered data → IC near 0."""
        # Deliberately constructed to have near-zero rank correlation
        confidences = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        returns = [4.0, 6.0, 2.0, 7.0, 3.0, 5.0, 1.0]
        result = compute_ic(confidences, returns)
        assert abs(result.ic) < 0.5  # not strongly correlated

    def test_too_few_observations(self):
        """< 5 observations returns IC=0, p=1."""
        result = compute_ic([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        assert result.ic == 0.0
        assert result.p_value == 1.0
        assert result.n == 3

    def test_empty_input(self):
        result = compute_ic([], [])
        assert result.ic == 0.0
        assert result.p_value == 1.0
        assert result.n == 0

    def test_zero_variance_confidences(self):
        """All same confidence → IC=0 (no ranking possible)."""
        result = compute_ic([50.0] * 10, list(range(10)))
        assert result.ic == 0.0
        assert result.p_value == 1.0

    def test_zero_variance_returns(self):
        """All same returns → IC=0."""
        result = compute_ic(list(range(10)), [3.0] * 10)
        assert result.ic == 0.0
        assert result.p_value == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_ic([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_exactly_five_observations(self):
        """Boundary: exactly 5 observations should compute normally."""
        confidences = [10.0, 20.0, 30.0, 40.0, 50.0]
        returns = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_ic(confidences, returns)
        assert result.ic == pytest.approx(1.0, abs=0.01)
        assert result.n == 5


# ── _compute_brier ───────────────────────────────────────────────────────────


class TestComputeBrier:
    def test_perfect_calibration(self):
        """100% confidence, all hits → Brier = 0."""
        brier = _compute_brier([100.0, 100.0], [True, True])
        assert brier == pytest.approx(0.0, abs=0.001)

    def test_worst_calibration(self):
        """100% confidence, all misses → Brier = 1."""
        brier = _compute_brier([100.0, 100.0], [False, False])
        assert brier == pytest.approx(1.0, abs=0.001)

    def test_empty_input(self):
        assert _compute_brier([], []) == 1.0

    def test_fifty_fifty(self):
        """50% confidence, mixed outcomes → Brier = 0.25."""
        brier = _compute_brier([50.0, 50.0], [True, False])
        assert brier == pytest.approx(0.25, abs=0.001)
