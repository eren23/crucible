"""Tests for crucible.core.config."""
from pathlib import Path
import tempfile

from crucible.core.config import load_config, generate_default_config


def test_generate_default_config():
    text = generate_default_config()
    assert "name:" in text
    assert "provider:" in text
    assert "training:" in text
    assert "presets:" in text


def test_load_config_default():
    config = load_config(path=None)
    assert config.name == "crucible-project"
    assert config.version == "0.1.0"


def test_load_config_from_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "crucible.yaml"
        yaml_path.write_text(
            "name: test-project\\nversion: '0.2.0'\\nprovider:\\n  type: ssh\\n",
            encoding="utf-8",
        )
        config = load_config(yaml_path)
        assert config.name == "test-project"
        assert config.version == "0.2.0"
        assert config.provider.type == "ssh"
