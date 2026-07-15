"""
Offline tests for the deterministic parts of csv_qa_agent.py — the sandboxed
code execution and the schema description. These don't call the Claude API,
so they run for free and don't need ANTHROPIC_API_KEY set.

Run with:
    pytest tests/test_agent.py -v
"""

import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csv_qa_agent import run_pandas_code, describe_dataframe


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Amount": [10.0, 200.0, 5.5, 999.0],
        "Class": [0, 1, 0, 1],
    })


def test_valid_code_returns_correct_result(sample_df):
    result = run_pandas_code(sample_df, "print(df['Amount'].mean())")
    assert result == str(sample_df["Amount"].mean())


def test_valid_code_with_groupby(sample_df):
    result = run_pandas_code(sample_df, "print(df.groupby('Class')['Amount'].mean().to_dict())")
    assert "0:" in result.replace(" ", "") or "0.0" in result


def test_missing_print_gives_helpful_message(sample_df):
    result = run_pandas_code(sample_df, "df['Amount'].mean()")
    assert "no printed output" in result.lower()


def test_referencing_missing_column_returns_error_not_crash(sample_df):
    result = run_pandas_code(sample_df, "print(df['DoesNotExist'].mean())")
    assert "Error while executing code" in result
    assert "KeyError" in result


def test_syntax_error_returns_error_not_crash(sample_df):
    result = run_pandas_code(sample_df, "print(df[")
    assert "Error while executing code" in result


def test_import_is_blocked(sample_df):
    result = run_pandas_code(sample_df, "import os; print(os.listdir('.'))")
    assert "Error while executing code" in result
    assert "not found" in result or "ImportError" in result


def test_open_is_blocked(sample_df):
    result = run_pandas_code(sample_df, "print(open('/etc/passwd').read())")
    assert "Error while executing code" in result
    assert "NameError" in result


def test_exec_is_blocked(sample_df):
    # Guard against escaping the restricted builtins via nested exec/eval.
    result = run_pandas_code(sample_df, "exec('import os'); print('escaped')")
    assert "Error while executing code" in result


def test_describe_dataframe_includes_shape_and_columns(sample_df):
    schema = describe_dataframe(sample_df)
    assert "4 rows x 2 columns" in schema
    assert "Amount" in schema
    assert "Class" in schema


def test_describe_dataframe_includes_sample_rows(sample_df):
    schema = describe_dataframe(sample_df)
    assert "First 3 rows" in schema
