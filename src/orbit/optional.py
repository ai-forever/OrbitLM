"""Optional dependencies of the trainer. Each name is None when its package is missing.

ClearML and python-dotenv are needed only with clearml_enabled; TensorBoard logging needs
torch.utils.tensorboard; mosaicml-streaming is needed to read the training data.
"""
from __future__ import annotations

try:
    from clearml import Task  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Task = None

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from streaming import Stream, StreamingDataset
    from streaming.base.util import clean_stale_shared_memory
except Exception as exc:  # pragma: no cover - optional dependency
    StreamingDataset = None
    Stream = None
    clean_stale_shared_memory = None
    STREAMING_IMPORT_ERROR = exc
else:
    STREAMING_IMPORT_ERROR = None
