"""Tests for crucible.core.redact — secret redaction."""
from __future__ import annotations

from crucible.core.redact import redact_secrets, _is_secret_key, _redact_string


# ---------------------------------------------------------------------------
# _is_secret_key
# ---------------------------------------------------------------------------


class TestIsSecretKey:
    def test_known_keys(self):
        assert _is_secret_key("RUNPOD_API_KEY")
        assert _is_secret_key("WANDB_API_KEY")
        assert _is_secret_key("ANTHROPIC_API_KEY")
        assert _is_secret_key("HF_TOKEN")
        assert _is_secret_key("HUGGING_FACE_HUB_TOKEN")

    def test_pattern_suffix_key(self):
        assert _is_secret_key("MY_CUSTOM_KEY")
        assert _is_secret_key("DB_SECRET")
        assert _is_secret_key("AUTH_TOKEN")
        assert _is_secret_key("MY_PASSWORD")
        assert _is_secret_key("SERVICE_CREDENTIAL")

    def test_non_secret_keys(self):
        assert not _is_secret_key("MODEL_FAMILY")
        assert not _is_secret_key("LR")
        assert not _is_secret_key("ITERATIONS")
        assert not _is_secret_key("gpu_type")
        assert not _is_secret_key("name")

    def test_case_insensitive_suffix(self):
        assert _is_secret_key("my_api_key")
        assert _is_secret_key("MY_API_KEY")
        assert _is_secret_key("Auth_Token")


# ---------------------------------------------------------------------------
# _redact_string
# ---------------------------------------------------------------------------


class TestRedactString:
    def test_anthropic_key(self):
        s = "my key is sk-abcdefghijklmnopqrstuvwxyz here"
        result = _redact_string(s)
        assert "<REDACTED>" in result
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result

    def test_wandb_key(self):
        s = "token: wandb_v1_abcdefghijklmnopqrstuv"
        result = _redact_string(s)
        assert "<REDACTED>" in result
        assert "wandb_v1_" not in result

    def test_runpod_key(self):
        s = "rpd_abcdefghijklmnopqrstuv123"
        result = _redact_string(s)
        assert result == "<REDACTED>"

    def test_huggingface_token(self):
        s = "hf_abcdefghijklmnopqrstuv123"
        result = _redact_string(s)
        assert result == "<REDACTED>"

    def test_non_secret_string_preserved(self):
        s = "hello world, lr=0.001"
        assert _redact_string(s) == s

    def test_short_prefix_not_matched(self):
        """Short strings that start with sk- but aren't long enough shouldn't match."""
        s = "sk-short"
        assert _redact_string(s) == s

    def test_anthropic_key_with_dashes(self):
        """Real sk-ant- keys contain internal dashes; pattern must capture them."""
        s = "API key sk-ant-api03-aBcD_eFgH-iJkLmNoPqRsTuVwXyZ123 done"
        result = _redact_string(s)
        assert "sk-ant-api03" not in result
        assert "<REDACTED>" in result

    def test_github_classic_pat(self):
        for prefix in ("ghp", "gho", "ghu", "ghs", "ghr"):
            s = f"token {prefix}_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 leaked"
            result = _redact_string(s)
            assert f"{prefix}_AbCd" not in result
            assert "<REDACTED>" in result

    def test_github_fine_grained_pat(self):
        s = "github_pat_11ABC123_aBcDeFgHiJkLmNoPqRsTuV here"
        result = _redact_string(s)
        assert "github_pat_11ABC" not in result
        assert "<REDACTED>" in result

    def test_aws_access_key(self):
        s = "creds AKIAIOSFODNN7EXAMPLE here, also ASIAQWERTY1234567890"
        result = _redact_string(s)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "ASIAQWERTY1234567890" not in result
        assert result.count("<REDACTED>") == 2

    def test_bearer_token(self):
        s = "Authorization: Bearer abc123def456ghi789jkl012mno345"
        result = _redact_string(s)
        assert "abc123def456ghi789jkl" not in result
        assert "<REDACTED>" in result

    def test_env_assignment_scrubbing(self):
        """Lines like `WANDB_API_KEY=value` should keep the key visible but scrub the value."""
        s = "WANDB_API_KEY=somevalue123 trained successfully"
        result = _redact_string(s)
        assert "WANDB_API_KEY" in result
        assert "somevalue123" not in result
        assert "<REDACTED>" in result

    def test_env_assignment_with_export(self):
        s = "export HF_TOKEN=abcd1234 && python train.py"
        result = _redact_string(s)
        assert "HF_TOKEN" in result
        assert "abcd1234" not in result

    def test_env_assignment_colon_form(self):
        """YAML-style `KEY: value` also scrubs."""
        s = "  ANTHROPIC_API_KEY: sk-ant-api03-foo\n"
        result = _redact_string(s)
        assert "ANTHROPIC_API_KEY" in result
        # Both the env-rule AND the sk-ant pattern would catch this; either is fine.
        assert "<REDACTED>" in result
        assert "sk-ant-api03-foo" not in result

    def test_env_assignment_skips_non_secret_keys(self):
        """`step=100` should NOT match because `step` lacks TOKEN/KEY/etc."""
        s = "step=100 lr=0.001 wd=0.1"
        assert _redact_string(s) == s


# ---------------------------------------------------------------------------
# redact_secrets (deep structure)
# ---------------------------------------------------------------------------


class TestRedactSecrets:
    def test_flat_dict_with_secret_key(self):
        data = {"WANDB_API_KEY": "wk_123456", "name": "experiment"}
        result = redact_secrets(data)
        assert result["WANDB_API_KEY"] == "<REDACTED>"
        assert result["name"] == "experiment"

    def test_nested_dict(self):
        data = {"outer": {"ANTHROPIC_API_KEY": "sk-1234567890abcdefghijklmn", "x": 1}}
        result = redact_secrets(data)
        assert result["outer"]["ANTHROPIC_API_KEY"] == "<REDACTED>"
        assert result["outer"]["x"] == 1

    def test_list_redaction(self):
        data = [{"HF_TOKEN": "hf_abc"}, {"name": "exp"}]
        result = redact_secrets(data)
        assert result[0]["HF_TOKEN"] == "<REDACTED>"
        assert result[1]["name"] == "exp"

    def test_string_value_with_embedded_key(self):
        data = {"config": "use key sk-abcdefghijklmnopqrstuv99 for auth"}
        result = redact_secrets(data)
        assert "sk-abcdefghijklmnopqrstuv99" not in result["config"]
        assert "<REDACTED>" in result["config"]

    def test_non_secret_values_preserved(self):
        data = {"LR": 0.001, "steps": 1000, "name": "baseline", "tags": ["a", "b"]}
        result = redact_secrets(data)
        assert result == data

    def test_integer_preserved(self):
        assert redact_secrets(42) == 42

    def test_float_preserved(self):
        assert redact_secrets(3.14) == 3.14

    def test_none_preserved(self):
        assert redact_secrets(None) is None

    def test_bool_preserved(self):
        assert redact_secrets(True) is True

    def test_deeply_nested(self):
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "MY_SECRET": "should_be_redacted",
                        "safe": "visible",
                    }
                }
            }
        }
        result = redact_secrets(data)
        assert result["level1"]["level2"]["level3"]["MY_SECRET"] == "<REDACTED>"
        assert result["level1"]["level2"]["level3"]["safe"] == "visible"

    def test_list_of_strings_with_secrets(self):
        data = ["safe string", "has rpd_abcdefghijklmnopqrstuv in it", "also safe"]
        result = redact_secrets(data)
        assert result[0] == "safe string"
        assert "rpd_" not in result[1]
        assert "<REDACTED>" in result[1]
        assert result[2] == "also safe"

    def test_does_not_mutate_original(self):
        data = {"WANDB_API_KEY": "original_value", "nested": {"x": 1}}
        _ = redact_secrets(data)
        assert data["WANDB_API_KEY"] == "original_value"
        assert data["nested"]["x"] == 1
