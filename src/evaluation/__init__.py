"""
Evaluation utilities for training pipelines.

This package currently provides:
- comprehensive_evaluation: run standard metrics/plots for a single run
- generate_loso_summary_plots: visualize LOSO cross‑validation results
"""

from .evaluator import comprehensive_evaluation, generate_loso_summary_plots

__all__ = ["comprehensive_evaluation", "generate_loso_summary_plots"]

