# CSV Q&A Agent

A small AI agent that answers natural-language questions about any CSV file by
writing and running pandas code — built from scratch with a direct loop against
the Claude API, no agent framework in between.

## Why this is an "agent" and not just an LLM call

A single LLM call can't reliably compute things like averages, correlations,
or group-bys — it can only guess. This project gives the model one tool,
`run_pandas_code`, and lets it decide when and how to use it:

1. The question, plus a description of the dataframe's shape/columns/dtypes,
   is sent to Claude.
2. Claude responds by calling `run_pandas_code` with a short pandas snippet.
3. The snippet is executed in a restricted namespace (only `df` and `pandas`
   are in scope — no imports, no file/network/system access) and whatever it
   `print()`s, or the error if it fails, is sent back to Claude as the tool
   result.
4. Claude either fixes its code and retries, or — once it has a real answer —
   responds in plain English with no further tool calls.

That loop (call → execute → observe → retry or answer) is the entire "agent"
here. It's deliberately minimal so the logic is easy to read end to end in
`csv_qa_agent.py`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# then edit .env and add your ANTHROPIC_API_KEY
```

## Usage

Interactive mode:

```bash
python csv_qa_agent.py path/to/data.csv
```

```
Loading path/to/data.csv ...
Shape: 284807 rows x 31 columns
...

CSV Q&A agent ready. Ask a question about the data, or type 'exit' to quit.

Q: What percentage of transactions are fraudulent?
    [running code]
    fraud_pct = (df['Class'].sum() / len(df)) * 100
    print(f"{fraud_pct:.3f}%")

    [result] 0.173%

A: About 0.173% of transactions in the dataset are fraudulent (492 out of 284,807).
```

Single question, non-interactive (handy for scripting or a demo GIF):

```bash
python csv_qa_agent.py path/to/data.csv -q "Which column correlates most with Class?"
python csv_qa_agent.py path/to/data.csv -q "How many duplicate rows are there?" --quiet
```

## Example questions this handles well

This was built and tested against the (anonymized) Kaggle credit card fraud
dataset — `Time`, `Amount`, `Class`, and PCA-anonymized features `V1`-`V28` —
but works against any tabular CSV. Examples that exercise different kinds of
generated pandas code:

- How many rows are there, and how many are flagged as fraud?
- What's the average transaction amount for fraud vs. non-fraud rows?
- Which of the V1-V28 columns correlates most strongly with `Class`?
- What time span does this dataset cover?
- Show the 10 largest transactions and how many were fraud.
- Are there any duplicate rows?

## Testing

There are two layers worth testing separately.

**1. Offline, no API key needed.** The sandboxed code execution and schema
description (`run_pandas_code`, `describe_dataframe`) are pure Python with no
network calls, so they're covered by `tests/test_agent.py`:

```bash
pip install pytest
pytest tests/test_agent.py -v
```

This checks that valid code returns the right answer, missing `print()` gives
a helpful nudge instead of silently failing, real errors (bad column names,
syntax errors) come back as readable messages instead of crashing the
process, and that `import`, `open()`, and nested `exec()` are all blocked.

**2. Live smoke test, needs `ANTHROPIC_API_KEY`.** The actual agent loop (does
Claude write correct pandas code, and recover when it doesn't) can only be
checked by running it for real. Point it at your dataset and run through a
handful of the questions from the list below, checking the answers against
values you already know or can compute yourself in the notebook — for
example, `df['Class'].value_counts()` should match whatever the agent reports
for "how many transactions are fraudulent":

```bash
python csv_qa_agent.py path/to/creditcard.csv -q "How many transactions are fraudulent?"
python csv_qa_agent.py path/to/creditcard.csv -q "What's the average amount for fraud vs non-fraud transactions?"
python csv_qa_agent.py path/to/creditcard.csv -q "Which of the V1-V28 columns correlates most strongly with Class?"
```

Worth deliberately trying a question or two that should fail gracefully, too
— e.g. asking about a column that doesn't exist — to confirm the retry logic
kicks in instead of the whole thing erroring out.

## Safety notes

- Code the agent writes runs with a whitelisted set of builtins only (`len`,
  `sum`, `print`, etc.) — no `import`, `open`, `exec`/`eval`, or access to
  `os`/`sys`.
- The dataframe is loaded read-only into the process; nothing the agent does
  can write back to the original CSV.
- The loop caps at 5 tool-call rounds per question to avoid runaway retries.
- Don't point this at untrusted CSVs from the internet without additional
  hardening — pandas parsing itself is generally safe, but this project hasn't
  been audited for that threat model.

## Possible extensions

- Swap the CLI loop for a small FastAPI wrapper to make it a real service.
- Add a second tool for generating a matplotlib chart alongside the answer.
- Cache the schema description so it isn't rebuilt on every question.
- Extend `tests/test_agent.py` with regression tests against a fixed sample
  CSV and known expected answers, once the live agent's behavior stabilizes.
