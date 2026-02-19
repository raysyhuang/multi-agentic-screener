from src.config import get_settings
from src.engines.credibility import compute_convergence_multiplier


def test_convergence_multiplier_includes_single_engine_setting():
    settings = get_settings()
    old_1 = settings.convergence_1_engine_multiplier
    old_2 = settings.convergence_2_engine_multiplier
    old_3 = settings.convergence_3_engine_multiplier
    old_4 = settings.convergence_4_engine_multiplier
    try:
        settings.convergence_1_engine_multiplier = 0.85
        settings.convergence_2_engine_multiplier = 1.3
        settings.convergence_3_engine_multiplier = 1.0
        settings.convergence_4_engine_multiplier = 0.95

        assert compute_convergence_multiplier(1) == 0.85
        assert compute_convergence_multiplier(2) == 1.3
        assert compute_convergence_multiplier(3) == 1.0
        assert compute_convergence_multiplier(4) == 0.95
        assert compute_convergence_multiplier(8) == 0.95
    finally:
        settings.convergence_1_engine_multiplier = old_1
        settings.convergence_2_engine_multiplier = old_2
        settings.convergence_3_engine_multiplier = old_3
        settings.convergence_4_engine_multiplier = old_4
