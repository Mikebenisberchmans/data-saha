"""
Minimal logging setup with a redaction helper.

This does not attempt to be a full secret-scanning log pipeline. It exists
so that any code that logs a data source or credential-adjacent object has
an obvious, easy-to-use helper that guarantees the secret field is dropped
rather than relying on every call site remembering to do it by hand.
"""

from __future__ import annotations

import logging
from typing import Any

_REDACTED = "***REDACTED***"
_SENSITIVE_KEYS = {"pat", "token", "secret", "password", "credential", "authorization"}


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def redact(value: Any) -> Any:
    """Recursively redact dict keys that look sensitive. Used before logging
    or before including any object in an error message. Does not mutate the
    input."""
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _looks_sensitive(k) else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def _looks_sensitive(key: str) -> bool:
    key_lower = str(key).lower()
    return any(sensitive in key_lower for sensitive in _SENSITIVE_KEYS)
