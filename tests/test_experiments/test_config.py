"""Tests for shadow track config loading and validation."""

from textwrap import dedent

from src.experiments.config import (
    FLAT_OVERRIDE_KEYS,
    load_tracks_from_yaml,
    validate_overrides,
)


class TestValidateOverrides:
    def test_valid_flat_keys(self):
        overrides = {
            "convergence_2_engine_multiplier": 1.5,
            "guardian_bear_sizing": 0.5,
        }
        assert validate_overrides(overrides) == []

    def test_valid_nested_regime_multipliers(self):
        overrides = {
            "regime_multipliers": {"bull": {"breakout": 1.5}},
        }
        assert validate_overrides(overrides) == []

    def test_unknown_key_returns_error(self):
        overrides = {"nonexistent_param": 42}
        errors = validate_overrides(overrides)
        assert len(errors) == 1
        assert "nonexistent_param" in errors[0]

    def test_empty_overrides_valid(self):
        assert validate_overrides({}) == []


class TestLoadTracksFromYaml:
    def test_load_from_file(self, tmp_path):
        yaml_file = tmp_path / "tracks.yaml"
        yaml_file.write_text(dedent("""\
            tracks:
              - name: test_track
                description: "Test"
                overrides:
                  convergence_2_engine_multiplier: 1.5
              - name: another_track
                description: "Another"
                overrides:
                  guardian_bear_sizing: 0.4
        """))

        tracks = load_tracks_from_yaml(yaml_file)
        assert len(tracks) == 2
        assert tracks[0].name == "test_track"
        assert tracks[0].overrides["convergence_2_engine_multiplier"] == 1.5
        assert tracks[1].name == "another_track"

    def test_skip_invalid_overrides(self, tmp_path):
        yaml_file = tmp_path / "tracks.yaml"
        yaml_file.write_text(dedent("""\
            tracks:
              - name: valid_track
                overrides:
                  guardian_bear_sizing: 0.5
              - name: invalid_track
                overrides:
                  fake_key: 99
        """))

        tracks = load_tracks_from_yaml(yaml_file)
        assert len(tracks) == 1
        assert tracks[0].name == "valid_track"

    def test_missing_file_returns_empty(self, tmp_path):
        tracks = load_tracks_from_yaml(tmp_path / "nonexistent.yaml")
        assert tracks == []

    def test_skip_entry_without_name(self, tmp_path):
        yaml_file = tmp_path / "tracks.yaml"
        yaml_file.write_text(dedent("""\
            tracks:
              - description: "Missing name"
                overrides:
                  guardian_bear_sizing: 0.5
        """))

        tracks = load_tracks_from_yaml(yaml_file)
        assert tracks == []

    def test_empty_yaml(self, tmp_path):
        yaml_file = tmp_path / "tracks.yaml"
        yaml_file.write_text("")
        tracks = load_tracks_from_yaml(yaml_file)
        assert tracks == []


class TestFlatOverrideKeysComplete:
    """Ensure all documented override keys are in the validation set."""

    def test_guardian_keys_present(self):
        guardian_keys = {k for k in FLAT_OVERRIDE_KEYS if k.startswith("guardian_")}
        assert "guardian_bear_sizing" in guardian_keys
        assert "guardian_max_drawdown_pct" in guardian_keys

    def test_convergence_keys_present(self):
        conv_keys = {k for k in FLAT_OVERRIDE_KEYS if k.startswith("convergence_")}
        assert len(conv_keys) >= 4  # 1/2/3/4 engine + sector
