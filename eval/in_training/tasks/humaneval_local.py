"""Scoring functions of the stock lm-eval humaneval task, from the installed lm-eval.

Importing them loads the evaluate "code_eval" metric, which must be in the offline cache.
"""
from lm_eval.tasks.humaneval.utils import build_predictions, pass_at_k

# itr_humaneval.yaml names these functions with !function, so they are the interface of this
# module. Listing them here also tells linters that the imports are used.
__all__ = ["build_predictions", "pass_at_k"]
