"""The import surface of the in-training evaluation (the package src/orbit/in_training_eval).

train.py imports the three trainer hooks from here, and orbit.export imports ship_chat_export for
its --chat-export option. The module is also the command line of the evaluation setup, run from the
repository root with PYTHONPATH=src in the trainer environment:

    python -m orbit.train_eval plan --suite S --world-size 64
    python -m orbit.train_eval prefetch --suite S --eval-python P --cache-dir C --tokenizer T [--harness H]

eval/in_training/README.md describes the configuration keys, the suites and the setup.
"""
from __future__ import annotations

import sys

from .in_training_eval.cli import main
from .in_training_eval.evaluator import InTrainingEval, build_in_training_eval
from .in_training_eval.export import ship_chat_export
from .in_training_eval.options import add_eval_arguments, validate_eval_arguments

__all__ = [
    "InTrainingEval",
    "add_eval_arguments",
    "build_in_training_eval",
    "main",
    "ship_chat_export",
    "validate_eval_arguments",
]


if __name__ == "__main__":
    sys.exit(main())
