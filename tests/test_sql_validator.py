from __future__ import annotations

import pytest

from app.analytics.sql_validator import validate_sql


# --- allowed ---------------------------------------------------------------


def test_simple_select_is_allowed():
    result = validate_sql("SELECT * FROM sales")
    assert result.is_safe


def test_select_with_where_and_order_is_allowed():
    result = validate_sql(
        "SELECT region, revenue FROM sales WHERE year = 2025 ORDER BY revenue DESC"
    )
    assert result.is_safe


def test_cte_with_select_is_allowed():
    result = validate_sql(
        "WITH recent AS (SELECT * FROM sales WHERE year = 2025) "
        "SELECT * FROM recent"
    )
    assert result.is_safe


def test_leading_whitespace_and_trailing_semicolon_is_allowed():
    result = validate_sql("   SELECT * FROM sales;   ")
    assert result.is_safe


def test_harmless_trailing_comment_is_allowed():
    result = validate_sql("SELECT * FROM sales -- get all sales rows")
    assert result.is_safe


# --- empty / non-query ------------------------------------------------------


def test_empty_sql_rejected():
    assert not validate_sql("").is_safe
    assert not validate_sql("   ").is_safe


def test_non_select_leading_statement_rejected():
    result = validate_sql("SHOW TABLES")
    assert not result.is_safe


# --- disallowed DML/DDL keywords -------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO sales VALUES (1, 2)",
        "UPDATE sales SET revenue = 0",
        "DELETE FROM sales",
        "DROP TABLE sales",
        "ALTER TABLE sales ADD COLUMN x INT",
        "TRUNCATE TABLE sales",
        "CREATE TABLE evil (x INT)",
        "GRANT ALL ON sales TO public",
        "REVOKE ALL ON sales FROM public",
        "MERGE INTO sales USING x ON (1=1)",
        "EXEC sp_helpdb",
        "EXECUTE sp_helpdb",
        "CALL some_procedure()",
    ],
)
def test_disallowed_keywords_rejected(sql):
    result = validate_sql(sql)
    assert not result.is_safe
    assert result.reason is not None


# --- multiple statements -----------------------------------------------------


def test_multiple_statements_rejected():
    result = validate_sql("SELECT * FROM sales; DROP TABLE sales;")
    assert not result.is_safe


def test_multiple_select_statements_still_rejected():
    result = validate_sql("SELECT 1; SELECT 2;")
    assert not result.is_safe


# --- comment-based obfuscation bypass attempts ------------------------------


def test_dangerous_statement_hidden_after_block_comment_rejected():
    result = validate_sql("SELECT 1; /* looks harmless */ DROP TABLE sales;")
    assert not result.is_safe


def test_dangerous_statement_hidden_after_line_comment_rejected():
    result = validate_sql("SELECT 1 -- comment\n; DROP TABLE sales;")
    assert not result.is_safe


def test_keyword_split_across_block_comment_is_still_caught():
    # A naive validator that only checks the string before stripping
    # comments could miss "DROP" reconstructed as "DR/**/OP"; stripping
    # comments first turns this into "DR OP" which doesn't match \bDROP\b
    # either way — but the important guarantee is the whole statement is
    # still rejected because it isn't a bare SELECT/WITH statement once
    # the comment is gone (it becomes invalid SQL shape entirely).
    result = validate_sql("DR/**/OP TABLE sales")
    assert not result.is_safe


# --- unsafe function / exfiltration patterns --------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT LOAD_FILE('/etc/passwd')",
        "SELECT * FROM sales INTO OUTFILE '/tmp/dump.csv'",
        "SELECT xp_cmdshell('dir')",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM OPENROWSET('SQLNCLI', 'evil')",
    ],
)
def test_unsafe_patterns_rejected_in_strict_mode(sql):
    result = validate_sql(sql, mode="strict")
    assert not result.is_safe


def test_unsafe_pattern_allowed_but_logged_in_permissive_mode(caplog):
    result = validate_sql("SELECT LOAD_FILE('/etc/passwd')", mode="permissive")
    assert result.is_safe
    assert any("permissive" in r.message.lower() for r in caplog.records)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "SELECT 1; DROP TABLE sales;",
        "EXEC sp_helpdb",
    ],
)
def test_dml_ddl_and_multi_statement_still_rejected_in_permissive_mode(sql):
    """Permissive mode only relaxes the supplementary unsafe-function
    blocklist — it must never allow writes, admin ops, or multiple
    statements."""
    result = validate_sql(sql, mode="permissive")
    assert not result.is_safe