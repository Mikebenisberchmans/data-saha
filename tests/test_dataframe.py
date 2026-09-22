from __future__ import annotations

import json

from app.analytics.dataframe import combine_frames, summarize_frame, to_dataframe
from app.mcp.models import ToolCallResult


def _result(source_id: str, tool_name: str, content, is_error: bool = False):
    if not isinstance(content, str):
        content = json.dumps(content)
    return ToolCallResult(
        source_id=source_id, tool_name=tool_name, content=content, is_error=is_error
    )


# --- to_dataframe: shape parsing --------------------------------------------


def test_columns_rows_shape_parses():
    result = _result(
        "src-a", "execute_sql",
        {"columns": ["region", "revenue"], "rows": [["US", 1000], ["EU", 800]]},
    )
    frame = to_dataframe(result)
    assert frame.df is not None
    assert list(frame.df.columns) == ["region", "revenue"]
    assert len(frame.df) == 2
    assert frame.df["revenue"].sum() == 1800


def test_list_of_dicts_shape_parses():
    result = _result(
        "src-a", "query",
        [{"region": "US", "revenue": 1000}, {"region": "EU", "revenue": 800}],
    )
    frame = to_dataframe(result)
    assert frame.df is not None
    assert set(frame.df.columns) == {"region", "revenue"}
    assert len(frame.df) == 2


def test_single_dict_treated_as_one_row():
    result = _result("src-a", "get_summary", {"total_revenue": 5000, "year": 2025})
    frame = to_dataframe(result)
    assert frame.df is not None
    assert len(frame.df) == 1
    assert frame.df.iloc[0]["total_revenue"] == 5000


def test_non_json_text_falls_back_to_none_df():
    result = _result("src-a", "get_notes", "just a plain text note, not JSON")
    frame = to_dataframe(result)
    assert frame.df is None
    assert frame.raw_content == "just a plain text note, not JSON"


def test_malformed_json_falls_back_gracefully():
    result = ToolCallResult(
        source_id="src-a", tool_name="x", content="{not valid json", is_error=False
    )
    frame = to_dataframe(result)
    assert frame.df is None


def test_error_result_never_parsed_as_data():
    result = _result("src-a", "execute_sql", "connection refused", is_error=True)
    frame = to_dataframe(result)
    assert frame.df is None


def test_empty_content_handled():
    result = ToolCallResult(source_id="src-a", tool_name="x", content="", is_error=False)
    frame = to_dataframe(result)
    assert frame.df is None


def test_scalar_json_does_not_crash():
    result = _result("src-a", "count_rows", 42)
    frame = to_dataframe(result)
    # A bare number isn't a dict/list-of-dicts/columns-rows shape -> no df,
    # but must not raise.
    assert frame.df is None
    assert frame.raw_content == "42"


# --- summarize_frame ---------------------------------------------------------


def test_summarize_frame_includes_columns_and_row_count():
    result = _result(
        "src-a", "execute_sql",
        {"columns": ["region", "revenue"], "rows": [["US", 1000], ["EU", 800]]},
    )
    frame = to_dataframe(result)
    summary = summarize_frame(frame)
    assert "2 row(s)" in summary
    assert "region" in summary and "revenue" in summary


def test_summarize_frame_includes_numeric_stats():
    result = _result(
        "src-a", "execute_sql",
        {"columns": ["region", "revenue"], "rows": [["US", 1000], ["EU", 800]]},
    )
    frame = to_dataframe(result)
    summary = summarize_frame(frame)
    assert "Numeric column stats" in summary
    assert "1800" in summary  # sum of revenue


def test_summarize_frame_falls_back_to_raw_text_when_not_tabular():
    result = _result("src-a", "get_notes", "plain text response")
    frame = to_dataframe(result)
    summary = summarize_frame(frame)
    assert summary == "plain text response"


def test_summarize_frame_truncates_large_results():
    rows = [["region", i] for i in range(50)]
    result = _result("src-a", "execute_sql", {"columns": ["region", "n"], "rows": rows})
    frame = to_dataframe(result)
    summary = summarize_frame(frame, max_rows=5)
    assert "more row(s) not shown" in summary


# --- combine_frames: cross-source analysis -----------------------------------


def test_combine_frames_returns_none_for_single_source():
    result = _result(
        "sales-src", "execute_sql",
        {"columns": ["region", "revenue"], "rows": [["US", 1000]]},
    )
    frame = to_dataframe(result)
    assert combine_frames([frame]) is None


def test_combine_frames_returns_none_when_only_one_source_has_data():
    tabular = to_dataframe(
        _result("sales-src", "execute_sql", {"columns": ["revenue"], "rows": [[1000]]})
    )
    non_tabular = to_dataframe(_result("finance-src", "get_notes", "no data available"))
    assert combine_frames([tabular, non_tabular]) is None


def test_combine_frames_computes_totals_across_two_sources():
    sales_frame = to_dataframe(
        _result(
            "sales-src", "execute_sql",
            {"columns": ["region", "revenue"], "rows": [["US", 1000], ["EU", 500]]},
        )
    )
    finance_frame = to_dataframe(
        _result(
            "finance-src", "execute_sql",
            {"columns": ["quarter", "expenses"], "rows": [["Q4", 300]]},
        )
    )

    combined = combine_frames([sales_frame, finance_frame])
    assert combined is not None
    assert "sales-src" in combined
    assert "finance-src" in combined
    assert "1500" in combined  # sales total (1000 + 500)
    assert "300" in combined  # finance total


def test_combine_frames_handles_source_with_no_numeric_column():
    sales_frame = to_dataframe(
        _result(
            "sales-src", "execute_sql",
            {"columns": ["region", "revenue"], "rows": [["US", 1000]]},
        )
    )
    names_frame = to_dataframe(
        _result(
            "other-src", "list_names",
            {"columns": ["name"], "rows": [["Alice"], ["Bob"]]},
        )
    )
    combined = combine_frames([sales_frame, names_frame])
    assert combined is not None
    assert "no numeric column to total" in combined