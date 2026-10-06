"""Model loading from the Hugging Face cache.

Libraries check the Hub for newer files every time a model loads, which costs a
network round trip (and a warning) on every CLI run. We try the local cache first and
only go online when the model isn't downloaded yet, so VidSense also works offline
after the first run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def load_local_first(loader: Callable[..., T], *args, **kwargs) -> T:
    try:
        return loader(*args, local_files_only=True, **kwargs)
    except Exception:  # not cached yet; the loaders raise OSError, ValueError or hub errors
        return loader(*args, **kwargs)


def quiet_transformers() -> None:
    """Hide the per-load "Loading weights" progress bars from transformers."""
    from transformers.utils import logging

    logging.disable_progress_bar()
