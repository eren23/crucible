"""Tests for the generic metric:name=value regex in OutputParser."""
from __future__ import annotations

import pytest

from crucible.runner.output_parser import (
    GENERIC_METRIC_RE,
    OutputParser,
    parse_output,
)


# ---------------------------------------------------------------------------
# Raw regex tests
# ---------------------------------------------------------------------------


class TestGenericMetricRegex:
    def test_accuracy_match(self):
        m = GENERIC_METRIC_RE.search("metric:accuracy=0.95")
        assert m is not None
        assert m.group(1) == "accuracy"
        assert float(m.group(2)) == pytest.approx(0.95)

    def test_fid_match(self):
        m = GENERIC_METRIC_RE.search("metric:fid=23.4")
        assert m is not None
        assert m.group(1) == "fid"
        assert float(m.group(2)) == pytest.approx(23.4)

    def test_negative_value(self):
        m = GENERIC_METRIC_RE.search("metric:delta=-0.5")
        assert m is not None
        assert m.group(1) == "delta"
        assert float(m.group(2)) == pytest.approx(-0.5)

    def test_scientific_notation(self):
        m = GENERIC_METRIC_RE.search("metric:lr=3e-4")
        assert m is not None
        assert m.group(1) == "lr"
        assert float(m.group(2)) == pytest.approx(3e-4)

    def test_scientific_notation_uppercase(self):
        m = GENERIC_METRIC_RE.search("metric:lr=3E-4")
        assert m is not None
        assert float(m.group(2)) == pytest.approx(3e-4)

    def test_integer_value(self):
        m = GENERIC_METRIC_RE.search("metric:epoch=10")
        assert m is not None
        assert float(m.group(2)) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# parse_line for generic metrics
# ---------------------------------------------------------------------------


class TestParseLineGenericMetric:
    def test_parse_line_generic_metric(self):
        parser = OutputParser()
        result = parser.parse_line("metric:accuracy=0.95")
        assert result is not None
        assert result["type"] == "generic_metric"
        assert result["name"] == "accuracy"
        assert result["value"] == pytest.approx(0.95)

    def test_parse_line_generic_does_not_mask_train_loss(self):
        parser = OutputParser()
        # train_loss line should still match as train_loss type, not generic
        result = parser.parse_line("step:10/100 train_loss:1.23")
        assert result is not None
        assert result["type"] == "train_loss"


# ---------------------------------------------------------------------------
# parse() aggregate with generic metrics
# ---------------------------------------------------------------------------


class TestParseAggregateGenericMetrics:
    def test_generic_metric_in_completed_output(self):
        text = (
            "step:100/100 train_loss:1.200\n"
            "metric:accuracy=0.95\n"
            "metric:fid=23.4\n"
            "Serialized model int8+zlib: 42000 bytes\n"
            "final_roundtrip val_loss:1.4000 val_bpb:1.2000\n"
        )
        result = parse_output(text)
        assert result is not None
        assert result["status"] == "completed"
        # generic metrics should be in result dict
        assert result["result"]["accuracy"] == pytest.approx(0.95)
        assert result["result"]["fid"] == pytest.approx(23.4)

    def test_lm_keys_take_precedence_over_generic(self):
        """If both a LM-specific key and a generic metric share a name,
        the LM-specific value wins."""
        text = (
            "step:100/100 train_loss:1.200\n"
            "metric:val_loss=99.9\n"
            "final_roundtrip val_loss:1.4000 val_bpb:1.2000\n"
        )
        result = parse_output(text)
        assert result is not None
        # val_loss should come from the final line, not from the generic metric
        assert result["result"]["val_loss"] == pytest.approx(1.4)

    def test_generic_metrics_in_partial_output(self):
        text = (
            "step:30/100 train_loss:2.500\n"
            "metric:custom_score=42.0\n"
        )
        result = parse_output(text)
        assert result is not None
        assert result["status"] == "partial_recoverable"
        assert result["result"]["custom_score"] == pytest.approx(42.0)

    def test_generic_only_output(self):
        """When only generic metrics are present (no final or step lines),
        they should still be collected."""
        text = "metric:bleu=0.35\nmetric:rouge=0.72\n"
        result = parse_output(text)
        assert result is not None
        assert result["status"] == "partial_recoverable"
        assert result["result"]["bleu"] == pytest.approx(0.35)
        assert result["result"]["rouge"] == pytest.approx(0.72)

    def test_negative_generic_value(self):
        text = "metric:delta=-0.5\n"
        result = parse_output(text)
        assert result is not None
        assert result["result"]["delta"] == pytest.approx(-0.5)

    def test_scientific_notation_generic(self):
        text = "metric:lr=3e-4\n"
        result = parse_output(text)
        assert result is not None
        assert result["result"]["lr"] == pytest.approx(3e-4)

    def test_alongside_train_loss(self):
        """Generic metrics alongside standard step/train_loss output."""
        text = (
            "step:10/100 train_loss:1.23\n"
            "metric:accuracy=0.95\n"
            "step:20/100 train_loss:1.10\n"
        )
        result = parse_output(text)
        assert result is not None
        assert result["status"] == "partial_recoverable"
        assert result["result"]["accuracy"] == pytest.approx(0.95)
        # train_loss_fallback should be from the last step
        assert result["result"]["train_loss_fallback"] == pytest.approx(1.10)
