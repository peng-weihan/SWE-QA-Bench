# Running batch_script.py Instructions

## Dependencies Installation

```bash
pip install pydantic python-dotenv rich tqdm
```

## Configuration Methods

### Method 1: Using Configuration File (Recommended)

Create a `config.json` file:

```json
{
  "repo_base_dir": "/path/to/repositories",
  "question_dir": "/path/to/questions",
  "output_dir": "/path/to/output",
  "model": "auto",
  "max_concurrency": 4,
  "output_format": "json",
  "cursor_agent_path": "/path/to/cursor-agent"
}
```

Then run:
```bash
python batch_script.py
```

### Method 2: Using Command Line Arguments

```bash
python batch_script.py \
  --repo-base-dir /path/to/repositories \
  --question-dir /path/to/questions \
  --output-dir /path/to/output \
  --model auto \
  --max-concurrency 4 \
  --output-format json
```

## Parameter Description

- `--config`: Configuration file path (default: `config.json`)
- `--repo-base-dir`: Root directory path for repositories
- `--question-dir`: Directory containing question files (.jsonl format)
- `--output-dir`: Output results directory
- `--model`: Model to use (default: `auto`)
- `--max-concurrency`: Maximum concurrency (default: 4)
- `--repo-filter`: Process only specific repositories (e.g., `--repo-filter conan`)
- `--output-format`: Output format, options: `text`, `json`, `stream-json` (default: `json`)
- `--cursor-agent-path`: Full path to cursor-agent executable

## Question File Format

Question files should be in `.jsonl` format, with one JSON object per line:

```json
{"question": "Question content", "ground_truth": "Reference answer"}
{"question": "Another question", "ground_truth": "Another answer"}
```

File name format: `{repo_name}.jsonl` (e.g., `conan.jsonl`)

## Output Format

Output files are `{repo_name}.jsonl`, with each line containing a result:
- `question`: Question
- `answer`: Answer
- `trajectory`: Complete trajectory
- `latency`: Latency (seconds)
- `input_tokens`: Number of input tokens
- `output_tokens`: Number of output tokens
- `total_tokens`: Total number of tokens

## Examples

```bash
# Process all repositories
python batch_script.py --config config.json

# Process only specific repository
python batch_script.py --repo-filter conan --question-dir ./questions --output-dir ./results

# Custom concurrency
python batch_script.py --max-concurrency 8 --question-dir ./questions --output-dir ./results
```

## Notes

- The script automatically skips already processed questions (based on output files)
- Ensure cursor-agent is installed and in PATH, or specify via `--cursor-agent-path`
- API key must be set via `CURSOR_API_KEY` environment variable before running the script

