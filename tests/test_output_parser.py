"""Tests for crucible.runner.output_parser."""
from __future__ import annotations

import signal

import pytest

from crucible.runner.output_parser import (
    OutputParser,
    parse_output,
    steps_seen,
    classify_failure,
    tail,
    FINAL_RE,
    STEP_RE,
    TRAIN_LOSS_RE,
    VAL_RE,
    MODEL_BYTES_RE,
    STOPPING_RE,
    WARMUP_RE,
    TRAIN_TIME_RE,
)


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------

class TestRegexPatterns:
    def test_final_re(self):
        line = "final_roundtrip val_loss:1.4567 val_bpb:1.2345"
        m = FINAL_RE.search(line)
        assert m is not None
        assert float(m.group(1)) == 1.4567
        assert float(m.group(2)) == 1.2345

    def test_step_re(self):
        line = "step:100/5000 train_loss:2.345"
        m = STEP_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 100
        assert int(m.group(2)) == 5000

    def test_train_loss_re(self):
        line = "step:50/1000 train_loss:3.1415"
        m = TRAIN_LOSS_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 50
        assert int(m.group(2)) == 1000
        assert float(m.group(3)) == pytest.approx(3.1415)

    def test_val_re(self):
        line = "step:200/5000 val_loss:1.5678 val_bpb:1.3456"
        m = VAL_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 200
        assert int(m.group(2)) == 5000
        assert float(m.group(3)) == pytest.approx(1.5678)
        assert float(m.group(4)) == pytest.approx(1.3456)

    def test_model_bytes_re_serialized_format(self):
        line = "Serialized model int8+zlib: 45000 bytes"
        m = MODEL_BYTES_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 45000

    def test_model_bytes_re_tag_format(self):
        line = "serialized_model_int8_zlib: 50000 bytes"
        m = MODEL_BYTES_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 50000

    def test_stopping_re(self):
        line = "stopping_early at step:500/1000"
        m = STOPPING_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 500

    def test_warmup_re(self):
        line = "warmup_step:5/10"
        m = WARMUP_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 5
        assert int(m.group(2)) == 10

    def test_train_time_re(self):
        line = "train_time:12345ms"
        m = TRAIN_TIME_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 12345


# ---------------------------------------------------------------------------
# OutputParser
# ---------------------------------------------------------------------------

class TestOutputParser:
    def _full_output(self) -> str:
        return (
            "warmup_step:1/5\n"
            "warmup_step:5/5\n"
            "step:10/100 train_loss:3.500\n"
            "step:50/100 train_loss:2.100\n"
            "step:50/100 val_loss:1.800 val_bpb:1.500\n"
            "step:100/100 train_loss:1.200\n"
            "Serialized model int8+zlib: 42000 bytes\n"
            "final_roundtrip val_loss:1.4000 val_bpb:1.2000\n"
            "train_time:60000ms\n"
        )

    def test_parse_completed(self):
        text = self._full_output()
        result = parse_output(text)
        assert result is not None
        assert result["status"] == "completed"
        assert result["result"]["val_loss"] == pytest.approx(1.4)
        assert result["result"]["val_bpb"] == pytest.approx(1.2)
        assert result["model_bytes"] == 42000
        assert result["result"]["train_time_ms"] == 60000
        assert result["result"]["steps_completed"] == 100

    def test_parse_partial_recoverable(self):
        text = "step:30/100 train_loss:2.500\nstep:50/100 train_loss:2.100\n"
        result = parse_output(text)
        assert result is not None
        assert result["status"] == "partial_recoverable"
        assert result["result"]["train_loss_fallback"] == pytest.approx(2.1)
        assert result["result"]["steps_completed"] == 50
        assert result["model_bytes"] is None

    def test_parse_returns_none_for_empty(self):
        assert parse_output("") is None
        assert parse_output("random text\nno patterns here\n") is None

    def test_steps_seen(self):
        text = "step:10/100 train_loss:3.0\nstep:50/100 train_loss:2.0\n"
        assert steps_seen(text) == 50

    def test_steps_seen_with_stopping(self):
        text = (
            "step:10/100 train_loss:3.0\n"
            "stopping_early at step:75/100\n"
        )
        assert steps_seen(text) == 75

    def test_parse_with_ttt_lora(self):
        text = (
            "final_roundtrip val_loss:1.4000 val_bpb:1.2000\n"
            "final_roundtrip_ttt_lora val_loss:1.3000 val_bpb:1.1000\n"
            "Serialized model int8+zlib: 42000 bytes\n"
        )
        result = parse_output(text)
        assert result is not None
        assert result["status"] == "completed"
        assert result["result"]["ttt_val_loss"] == pytest.approx(1.3)
        assert result["result"]["ttt_val_bpb"] == pytest.approx(1.1)


class TestOutputParserParseLine:
    def test_warmup_line(self):
        parser = OutputParser()
        result = parser.parse_line("warmup_step:3/10")
        assert result is not None
        assert result["type"] == "warmup"
        assert result["step"] == 3
        assert result["total"] == 10

    def test_train_loss_line(self):
        parser = OutputParser()
        result = parser.parse_line("step:50/1000 train_loss:2.345")
        assert result is not None
        assert result["type"] == "train_loss"
        assert result["step"] == 50
        assert result["total_steps"] == 1000
        assert result["train_loss"] == pytest.approx(2.345)

    def test_val_line(self):
        parser = OutputParser()
        result = parser.parse_line("step:200/5000 val_loss:1.5 val_bpb:1.3")
        assert result is not None
        assert result["type"] == "val"
        assert result["step"] == 200
        assert result["val_loss"] == pytest.approx(1.5)
        assert result["val_bpb"] == pytest.approx(1.3)

    def test_serializing_line(self):
        parser = OutputParser()
        result = parser.parse_line("Serialized model int8+zlib: 50000 bytes")
        assert result is not None
        assert result["type"] == "serializing"

    def test_final_line(self):
        parser = OutputParser()
        result = parser.parse_line("final_roundtrip val_loss:1.4 val_bpb:1.2")
        # Note: parse_line only detects "_roundtrip" final lines
        # The actual line detection is for saving state; parse_line might
        # match train_loss or val first; check if not None or None
        # parse_line checks warmup -> train_loss -> val -> serializing -> final
        # Since the line has val_loss and val_bpb, it matches val_re first
        # Actually let's check: this line has "val_loss:" and "val_bpb:" but
        # also starts with "final_" not "step:"
        # The val_re requires "step:\d+/\d+", so it won't match here.
        # The "_roundtrip" check will catch it.
        assert result is not None
        assert result["type"] == "final"

    def test_empty_line(self):
        parser = OutputParser()
        assert parser.parse_line("") is None
        assert parser.parse_line("   ") is None

    def test_unknown_line(self):
        parser = OutputParser()
        assert parser.parse_line("some random text") is None


class TestOutputParserFromConfig:
    def test_default_from_config(self):
        parser = OutputParser.from_config(None)
        assert parser.final_re == FINAL_RE

    def test_custom_pattern(self):
        parser = OutputParser.from_config({"final": r"DONE val_loss:(\d+\.\d+) val_bpb:(\d+\.\d+)"})
        text = "DONE val_loss:1.5 val_bpb:1.3"
        m = parser.final_re.search(text)
        assert m is not None


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------

class TestClassifyFailure:
    def test_timeout(self):
        status, cls = classify_failure(1, "some text", timed_out=True)
        assert status == "timeout"
        assert cls == "timeout"

    def test_completed(self):
        status, cls = classify_failure(0, "all good", timed_out=False)
        assert status == "completed"
        assert cls is None

    def test_none_returncode(self):
        status, cls = classify_failure(None, "", timed_out=False)
        assert status == "failed"
        assert cls == "unknown_exit"

    def test_oom_by_text(self):
        status, cls = classify_failure(1, "CUDA out of memory", timed_out=False)
        assert status == "failed"
        assert cls == "oom_suspected"

    def test_oom_general(self):
        status, cls = classify_failure(1, "out of memory error", timed_out=False)
        assert status == "failed"
        assert cls == "oom_suspected"

    def test_sigkill(self):
        status, cls = classify_failure(-signal.SIGKILL, "", timed_out=False)
        assert status == "killed"
        assert cls == "sigkill"

    def test_sigkill_with_oom(self):
        status, cls = classify_failure(-signal.SIGKILL, "OOM killer", timed_out=False)
        assert status == "killed"
        assert cls == "oom_suspected"

    def test_sigterm(self):
        status, cls = classify_failure(-signal.SIGTERM, "", timed_out=False)
        assert status == "killed"
        assert cls == "sigterm"

    def test_traceback_error(self):
        status, cls = classify_failure(1, "Traceback (most recent call last)", timed_out=False)
        assert status == "failed"
        assert cls == "runtime_error"

    def test_config_validation_error(self):
        status, cls = classify_failure(1, "ValueError: invalid config", timed_out=False)
        assert status == "failed"
        assert cls == "config_validation"

    def test_nan_divergence(self):
        status, cls = classify_failure(1, "fatal: train_loss is nan", timed_out=False)
        assert status == "failed"
        assert cls == "nan_divergence"

    def test_nonzero_exit_generic(self):
        status, cls = classify_failure(1, "something went wrong", timed_out=False)
        assert status == "failed"
        assert cls == "nonzero_exit"


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------

class TestTail:
    def test_short_text(self):
        assert tail("hello") == "hello"

    def test_long_text(self):
        text = "x" * 5000
        result = tail(text, limit=100)
        assert len(result) == 100

    def test_custom_limit(self):
        text = "abcdefgh"
        assert tail(text, limit=3) == "fgh"
