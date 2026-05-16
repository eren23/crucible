"""Built-in evaluator plugins.

Each module self-registers its evaluator class via
``register_evaluator()`` at module import time, mirroring the
``crucible/data_sources/`` pattern.
"""
from crucible.evaluators.lm_eval_harness import LMEvalHarnessEvaluator

__all__ = ["LMEvalHarnessEvaluator"]
