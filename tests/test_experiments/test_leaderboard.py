"""Tests for leaderboard composite scoring."""

import pytest

from src.experiments.leaderboard import (
    TrackScorecard,
    _compute_composite_scores,
    _compute_baseline_deltas,
)


class TestCompositeScores:
    def _make_scorecards(self):
        return [
            TrackScorecard(
                name="track_a", track_id=1, status="active", generation=1,
                parent_track=None, config={},
                resolved_picks=10, has_sufficient_data=True,
                sharpe_ratio=2.0, profit_factor=1.5, win_rate=0.6,
                calmar_ratio=3.0, avg_return_pct=1.5,
            ),
            TrackScorecard(
                name="track_b", track_id=2, status="active", generation=1,
                parent_track=None, config={},
                resolved_picks=10, has_sufficient_data=True,
                sharpe_ratio=1.0, profit_factor=1.0, win_rate=0.4,
                calmar_ratio=1.0, avg_return_pct=0.5,
            ),
            TrackScorecard(
                name="track_c", track_id=3, status="active", generation=1,
                parent_track=None, config={},
                resolved_picks=3, has_sufficient_data=False,  # insufficient data
                sharpe_ratio=5.0, profit_factor=3.0, win_rate=0.9,
                calmar_ratio=10.0, avg_return_pct=5.0,
            ),
        ]

    def test_eligible_tracks_scored(self):
        scs = self._make_scorecards()
        _compute_composite_scores(scs)
        assert scs[0].composite_score > 0  # track_a (best)
        assert scs[1].composite_score >= 0  # track_b

    def test_insufficient_data_not_scored(self):
        scs = self._make_scorecards()
        _compute_composite_scores(scs)
        assert scs[2].composite_score == 0  # track_c has insufficient data

    def test_best_track_highest_score(self):
        scs = self._make_scorecards()
        _compute_composite_scores(scs)
        assert scs[0].composite_score > scs[1].composite_score

    def test_no_eligible_tracks(self):
        scs = [
            TrackScorecard(
                name="t", track_id=1, status="active", generation=1,
                parent_track=None, config={},
                has_sufficient_data=False,
            ),
        ]
        _compute_composite_scores(scs)
        assert scs[0].composite_score == 0

    def test_single_eligible_track(self):
        scs = [
            TrackScorecard(
                name="t", track_id=1, status="active", generation=1,
                parent_track=None, config={},
                resolved_picks=10, has_sufficient_data=True,
                sharpe_ratio=1.5, profit_factor=1.2, win_rate=0.55,
                calmar_ratio=2.0, avg_return_pct=1.0,
            ),
        ]
        _compute_composite_scores(scs)
        # Single track → all normalized to 0.5
        assert scs[0].composite_score == pytest.approx(0.5, abs=0.01)


class TestBaselineDeltas:
    def test_delta_vs_baseline(self):
        scs = [
            TrackScorecard(
                name="_baseline", track_id=1, status="active", generation=1,
                parent_track=None, config={},
                sharpe_ratio=1.0, win_rate=0.5, avg_return_pct=0.5,
            ),
            TrackScorecard(
                name="track_a", track_id=2, status="active", generation=1,
                parent_track=None, config={},
                sharpe_ratio=2.0, win_rate=0.6, avg_return_pct=1.5,
            ),
        ]
        _compute_baseline_deltas(scs)
        assert scs[1].delta_sharpe == pytest.approx(1.0, abs=0.01)
        assert scs[1].delta_win_rate == pytest.approx(0.1, abs=0.01)
        assert scs[1].delta_avg_return == pytest.approx(1.0, abs=0.01)

    def test_baseline_self_delta_zero(self):
        scs = [
            TrackScorecard(
                name="_baseline", track_id=1, status="active", generation=1,
                parent_track=None, config={},
                sharpe_ratio=1.0, win_rate=0.5, avg_return_pct=0.5,
            ),
        ]
        _compute_baseline_deltas(scs)
        assert scs[0].delta_sharpe == 0
        assert scs[0].delta_win_rate == 0
