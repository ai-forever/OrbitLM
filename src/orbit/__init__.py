"""OrbitLM: the trainer of the looped (Ouro-architecture) language model and its HF export.

The modules, in the order a training run uses them:

    experiment_config  loads a YAML config (configs/train/) and expands ${VAR} paths
    train              the trainer and its entry point (python -m orbit.train --config ...)
    data               MDS streams mixed by one Mosaic StreamingDataset, cut into (x, y) pairs
    configuration_ouro the model config; modelling_ouro the model, with document masking
    losses             the exit-weighted LM loss and the exit entropy term
    fsdp_utils         FSDP wrapping and distributed state dicts
    checkpointing      rank checkpoints, meta.json and latest.txt
    train_eval         the in-training evaluation (owned by eval/in_training)
    full_state         gathers the full weights for the evaluation's HF export
    export             trainer checkpoint -> HF folder (python -m orbit.export ...)

README.md in this folder describes the stages, the config keys and the export.
"""
from __future__ import annotations

from .train import main

__all__ = ["main"]
