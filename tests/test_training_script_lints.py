"""Tests for the training-script reliability lints."""
from __future__ import annotations

from crucible.mcp.tools import _training_script_lints


def test_clean_script_no_warnings():
    text = """
    for step in range(100):
        loss = train_step()
        print(f"step:{step}/100 train_loss:{loss}")
    torch.save(model.state_dict(), "ckpt.pt")
    """
    assert _training_script_lints(text) == []


def test_loads_without_saves_warns():
    text = """
    model = AutoModel.from_pretrained("bert-base-uncased")
    output = model(inputs)
    """
    warnings = _training_script_lints(text)
    assert any("loads a pretrained model" in w for w in warnings)


def test_load_with_save_no_warning():
    text = """
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.save_pretrained("./ckpt")
    """
    warnings = _training_script_lints(text)
    assert not any("loads a pretrained model" in w for w in warnings)


def test_train_loop_without_loss_emit_warns():
    text = """
    model.train()
    for epoch in range(3):
        for batch in loader:
            output = model(batch)
            output.loss.backward()
            optimizer.step()
    """
    warnings = _training_script_lints(text)
    assert any("train_loss:" in w for w in warnings)


def test_train_loop_with_loss_emit_no_warning():
    text = """
    for step in range(1000):
        loss = compute_loss()
        if step % 10 == 0:
            print(f"step:{step}/1000 train_loss:{loss:.4f}")
    """
    warnings = _training_script_lints(text)
    assert not any("train_loss:" in w for w in warnings)


def test_lora_without_save_warns():
    text = """
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=8)
    model = get_peft_model(base, config)
    for step in range(100):
        print(f"step:{step}/100 train_loss:{loss}")
    """
    warnings = _training_script_lints(text)
    assert any("LoRA" in w or "PEFT" in w for w in warnings)


def test_lora_with_save_no_warning():
    text = """
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=8)
    model = get_peft_model(base, config)
    for step in range(100):
        print(f"step:{step}/100 train_loss:{loss}")
    model.save_pretrained("./adapter")
    """
    warnings = _training_script_lints(text)
    assert not any("LoRA" in w or "PEFT" in w for w in warnings)


def test_empty_script_no_warnings():
    assert _training_script_lints("") == []


def test_script_lints_run_in_enqueue_experiment(tmp_path, monkeypatch):
    """Lints from the project's training script flow through enqueue_experiment.warnings."""
    from crucible.mcp.tools import _lint_default_training_script

    project = tmp_path / "proj"
    project.mkdir()
    script = project / "train.py"
    script.write_text(
        "model = AutoModel.from_pretrained('x')\n"
        "for step in range(10):\n"
        "    print(f'step:{step}/10 train_loss:{0.1}')\n"
    )

    class FakeTrainingCfg:
        script = "train.py"

    class FakeConfig:
        project_root = project
        training = [FakeTrainingCfg()]

    out = _lint_default_training_script(FakeConfig())
    # Loads from_pretrained but no save_pretrained → first lint should fire.
    assert any("loads a pretrained model" in w for w in out)


def test_script_path_missing_returns_empty():
    from crucible.mcp.tools import _lint_default_training_script

    class FakeTrainingCfg:
        script = "/nonexistent/train.py"

    class FakeConfig:
        project_root = None
        training = [FakeTrainingCfg()]

    assert _lint_default_training_script(FakeConfig()) == []


def test_multi_backend_lints_all_entries(tmp_path):
    """Project specs may declare training: [mlx_cfg, torch_cfg].
    Earlier code only lint-checked training[0], silently passing pathologies
    in any other backend's script. This test guards against that regression.
    """
    from crucible.mcp.tools import _lint_default_training_script

    project = tmp_path / "proj"
    project.mkdir()
    # MLX script: clean (proper save + train_loss emit)
    (project / "train_mlx.py").write_text(
        "model = AutoModel.from_pretrained('x')\n"
        "model.save_pretrained('./ckpt')\n"
        "for step in range(10):\n"
        "    print(f'step:{step}/10 train_loss:{0.1}')\n"
    )
    # Torch script: dirty (loads but never saves)
    (project / "train_torch.py").write_text(
        "model = AutoModel.from_pretrained('x')\n"
        "for step in range(10):\n"
        "    print(f'step:{step}/10 train_loss:{0.1}')\n"
    )

    class MLXCfg:
        backend = "mlx"
        script = "train_mlx.py"

    class TorchCfg:
        backend = "torch"
        script = "train_torch.py"

    class FakeConfig:
        project_root = project
        training = [MLXCfg(), TorchCfg()]

    out = _lint_default_training_script(FakeConfig())
    # The dirty torch script should produce a warning even though it's not training[0].
    assert any("loads a pretrained model" in w for w in out)
    # When more than one backend exists, warnings should be backend-prefixed.
    assert any(w.startswith("[backend=torch]") for w in out)
    # The clean MLX script should not contribute a warning.
    assert not any("[backend=mlx]" in w for w in out)


def test_single_backend_keeps_unprefixed_warning(tmp_path):
    """With only one training entry, warnings stay unprefixed (no backend tag)."""
    from crucible.mcp.tools import _lint_default_training_script

    project = tmp_path / "proj"
    project.mkdir()
    (project / "t.py").write_text(
        "model = AutoModel.from_pretrained('x')\n"  # no save → triggers lint
    )

    class Cfg:
        backend = "torch"
        script = "t.py"

    class FakeConfig:
        project_root = project
        training = [Cfg()]

    out = _lint_default_training_script(FakeConfig())
    assert out, "expected at least one warning"
    assert not any(w.startswith("[backend=") for w in out)
