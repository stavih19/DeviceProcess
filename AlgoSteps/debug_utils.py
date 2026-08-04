from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from io import StringIO
from typing import ContextManager, TextIO


def debug_print_context(enabled: bool) -> ContextManager[TextIO | None]:
    """Suppress stage stdout unless its wrapper enables debug printing."""
    if enabled:
        return nullcontext()
    return redirect_stdout(StringIO())
