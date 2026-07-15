"""
csv_qa_agent.py

A minimal, from-scratch AI agent that answers natural-language questions about
any CSV file by writing and executing pandas code — no agent framework, just a
direct loop against the Claude API.

How it works:
    1. Load the CSV into a pandas dataframe and describe its schema.
    2. Send the user's question, plus the schema, to Claude with one tool
       available: `run_pandas_code`.
    3. Claude calls the tool with a short pandas snippet. We execute it in a
       restricted namespace (only `df` and `pd` are in scope — no imports, no
       file/network/system access) and capture whatever it print()s, or the
       error if it fails.
    4. The result (or error) is sent back to Claude as a tool result. Claude
       either retries with corrected code, or — once it has what it needs —
       responds with a final plain-English answer and no further tool calls.

Usage:
    python csv_qa_agent.py path/to/data.csv
    python csv_qa_agent.py path/to/data.csv --question "How many rows are there?"
    python csv_qa_agent.py path/to/data.csv --quiet     # hide the code/result trace
"""

import argparse
import contextlib
import io
import os
import sys
import traceback

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
MAX_TOOL_ROUNDS = 5

# Only these names are reachable from code the agent writes. No `__import__`,
# no `open`, no `exec`/`eval`, no access to `os`/`sys` — deliberately narrow.
SAFE_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max, "sorted": sorted,
    "round": round, "range": range, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "str": str, "int": int, "float": float,
    "bool": bool, "abs": abs, "enumerate": enumerate, "zip": zip,
    "print": print,
}

TOOL_DEFINITION = {
    "name": "run_pandas_code",
    "description": (
        "Execute Python/pandas code against the loaded dataframe `df`. "
        "The dataframe is already in scope as `df`, and pandas is available as `pd`. "
        "Your code MUST call print() on whatever value answers the question — "
        "only what is printed is returned to you. "
        "No imports and no file/network/system access are available; only `df` and `pd`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Must print() the result.",
            }
        },
        "required": ["code"],
    },
}


def describe_dataframe(df: pd.DataFrame) -> str:
    """Build a schema description to ground the agent in what `df` actually contains."""
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    return (
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        f"Columns and dtypes:\n{info_str}\n\n"
        f"First 3 rows:\n{df.head(3).to_string()}\n"
    )


def run_pandas_code(df: pd.DataFrame, code: str) -> str:
    """Execute agent-provided code in a restricted namespace; return printed output or error text."""
    local_ns = {"df": df, "pd": pd}
    global_ns = {"__builtins__": SAFE_BUILTINS}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, global_ns, local_ns)
        output = stdout.getvalue().strip()
        if not output:
            return "(Code ran with no printed output. Remember to print() the result.)"
        return output
    except Exception:
        return f"Error while executing code:\n{traceback.format_exc(limit=2)}"


def ask(client: Anthropic, df: pd.DataFrame, schema: str, question: str, verbose: bool = True) -> str:
    system_prompt = (
        "You are a data analysis agent. You answer questions about a pandas "
        "dataframe called `df` by writing short pandas code snippets and running them "
        "with the run_pandas_code tool, then explaining the result in plain English.\n\n"
        "Dataframe description:\n" + schema + "\n\n"
        "Rules:\n"
        "- Always use the tool to compute answers; never guess a number.\n"
        "- Keep each code snippet short and focused on one question.\n"
        "- If the tool returns an error, read it, fix your code, and try again.\n"
        "- Once you have the answer, respond with a concise plain-English sentence "
        "(no more tool calls) that directly answers the question, including the "
        "relevant number(s)."
    )

    messages = [{"role": "user", "content": question}]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            tools=[TOOL_DEFINITION],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_blocks).strip()

        tool_results = []
        for call in tool_calls:
            code = call.input.get("code", "")
            if verbose:
                print(f"    [running code]\n{code}\n")
            result = run_pandas_code(df, code)
            if verbose:
                print(f"    [result] {result}\n")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    return "I couldn't reach a confident answer within the allowed number of steps — try rephrasing the question."


def main():
    parser = argparse.ArgumentParser(description="Ask natural-language questions about a CSV file.")
    parser.add_argument("csv_path", help="Path to the CSV file to load.")
    parser.add_argument("--question", "-q", help="Ask a single question and exit (non-interactive).")
    parser.add_argument("--quiet", action="store_true", help="Hide the intermediate code/result trace.")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY in your environment or a .env file before running this.")

    client = Anthropic(api_key=api_key)

    print(f"Loading {args.csv_path} ...")
    df = pd.read_csv(args.csv_path)
    schema = describe_dataframe(df)
    print(schema)

    if args.question:
        answer = ask(client, df, schema, args.question, verbose=not args.quiet)
        print(f"\nA: {answer}")
        return

    print("CSV Q&A agent ready. Ask a question about the data, or type 'exit' to quit.\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        answer = ask(client, df, schema, question, verbose=not args.quiet)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
