# Cursor-Agent QA

The experiment uses cursor-agent version: **cursor-agent-2025.09.04-fc40cd1**

## Setup

1. Set the `CURSOR_API_KEY` environment variable:
```bash
export CURSOR_API_KEY="your-api-key-here"
```

2. (Optional) Set the `CURSOR_AGENT_PATH` environment variable if cursor-agent is not in PATH:
```bash
export CURSOR_AGENT_PATH="/path/to/cursor-agent"
```

## Usage

```bash
python batch_script.py \
  --repo-base-dir /path/to/repositories \
  --question-dir /path/to/questions \
  --output-dir /path/to/output \
  --model auto \
  --max-concurrency 4 \
  --output-format json \
  --repo-filter sympy \
  --max-tool-calls 50 \
  --force
```

See [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md) for detailed usage instructions.