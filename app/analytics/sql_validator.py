"""
SQL safety validation layer (product spec section 17).

This is a REAL validation layer, not something left to prompting the model
to "please only write SELECT statements." It runs on every SQL string
before it's handed to an MCP tool, regardless of what the model intended.

Design: default-deny. Rather than only blocklisting dangerous keywords, the
validator first strips comments (so a keyword can't be hidden inside
`/* ... */` or after `--`), then requires the statement to *start* with
SELECT or WITH (a CTE), and separately scans the entire de-commented string
for disallowed keywords/patterns anywhere in it — not just as the leading
word. That combination catches the specific bypass shapes called out in
the spec:

- multiple statements ("SELECT 1; DROP TABLE users;")
- comments used to hide a second statement or a keyword
- stored procedure calls (EXEC / EXECUTE / CALL)
- unsafe/exfiltration functions (LOAD_FILE, INTO OUTFILE, xp_cmdshell,
  pg_read_file, COPY ... TO/FROM, etc.)

SQL_SAFETY_MODE (settings.sql_safety_mode) has two modes:
- "strict" (default): all checks below apply.
- "permissive": DML/DDL keywords, stored-procedure calls, and multiple
  statements are STILL always rejected — those are never safe to allow in
  an analytics tool-calling context. Permissive mode only relaxes the
  supplementary "unsafe function" blocklist (logging instead of blocking),
  for environments that need a function this list doesn't know about.
  There is no mode that allows write operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Keywords that indicate a write/DDL/administrative operation. Checked
# as whole words (via regex word boundaries), case-insensitive, anywhere
# in the de-commented SQL — not just at the start — so a keyword hidden
# after a semicolon or inside what looks like a sub-clause is still caught.
_DISALLOWED_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE",
    "GRANT", "REVOKE", "MERGE", "REPLACE", "EXEC", "EXECUTE", "CALL",
    "ATTACH", "DETACH", "VACUUM", "REINDEX", "PRAGMA",
]

# Statement leading keywords that are allowed. Anything else at the start
# of the (comment-stripped, whitespace-trimmed) SQL is rejected outright.
_ALLOWED_LEADING_KEYWORDS = ["SELECT", "WITH"]

# Supplementary blocklist for data-exfiltration / OS-interaction patterns
# that aren't simple single keywords. Relaxed (logged, not blocked) in
# "permissive" mode; always enforced in "strict" mode.
_UNSAFE_PATTERNS = [
    r"\bLOAD_FILE\s*\(",
    r"\bINTO\s+OUTFILE\b",
    r"\bINTO\s+DUMPFILE\b",
    r"\bXP_CMDSHELL\b",
    r"\bPG_READ_FILE\s*\(",
    r"\bPG_WRITE_FILE\s*\(",
    r"\bLO_IMPORT\s*\(",
    r"\bLO_EXPORT\s*\(",
    r"\bCOPY\b.*\b(TO|FROM)\b",
    r"\bOPENROWSET\s*\(",
    r"\bDBMS_LOB\b",
]

_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


@dataclass
class ValidationResult:
    is_safe: bool
    reason: str | None = None


def _strip_comments(sql: str) -> str:
    sql = _BLOCK_COMMENT_RE.sub(" ", sql)
    sql = _LINE_COMMENT_RE.sub(" ", sql)
    return sql


def _has_multiple_statements(sql: str) -> bool:
    """After stripping a single optional trailing semicolon, any remaining
    semicolon means more than one statement is present."""
    trimmed = sql.strip()
    if trimmed.endswith(";"):
        trimmed = trimmed[:-1]
    return ";" in trimmed


def _find_disallowed_keyword(sql: str) -> str | None:
    for keyword in _DISALLOWED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", sql, re.IGNORECASE):
            return keyword
    return None


def _find_unsafe_pattern(sql: str) -> str | None:
    for pattern in _UNSAFE_PATTERNS:
        if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
            return pattern
    return None


def _leading_keyword_allowed(sql: str) -> bool:
    stripped = sql.strip()
    if not stripped:
        return False
    first_word = re.match(r"[A-Za-z]+", stripped)
    if not first_word:
        return False
    return first_word.group(0).upper() in _ALLOWED_LEADING_KEYWORDS


def validate_sql(sql: str, mode: str | None = None) -> ValidationResult:
    """Validates one SQL string for read-only, single-statement analytics
    use. `mode` overrides settings.sql_safety_mode for testing; normally
    left as None to read from configuration."""
    settings = get_settings()
    effective_mode = mode or settings.sql_safety_mode

    if not sql or not sql.strip():
        return ValidationResult(is_safe=False, reason="Empty SQL statement.")

    cleaned = _strip_comments(sql)

    if not _leading_keyword_allowed(cleaned):
        return ValidationResult(
            is_safe=False,
            reason=(
                "Only SELECT (or WITH ... SELECT) statements are allowed "
                "for analytics queries."
            ),
        )

    if _has_multiple_statements(cleaned):
        return ValidationResult(
            is_safe=False,
            reason="Multiple SQL statements are not allowed in one call.",
        )

    disallowed = _find_disallowed_keyword(cleaned)
    if disallowed:
        return ValidationResult(
            is_safe=False,
            reason=f"Disallowed operation detected: {disallowed}.",
        )

    unsafe = _find_unsafe_pattern(cleaned)
    if unsafe:
        if effective_mode == "strict":
            return ValidationResult(
                is_safe=False,
                reason="Query uses a disallowed function or file/OS access pattern.",
            )
        logger.warning(
            "SQL matched unsafe-pattern blocklist but was allowed "
            "(SQL_SAFETY_MODE=permissive): pattern=%s",
            unsafe,
        )

    return ValidationResult(is_safe=True)