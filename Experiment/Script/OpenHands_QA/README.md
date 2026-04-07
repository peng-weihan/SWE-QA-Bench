# OpenHands QA - Code Repository Question Answering System

An automated question answering system for code repositories using OpenHands SDK. This tool processes questions about codebases by creating AI agents that explore repository structures and generate comprehensive answers.

## Features

- **Automated Code Exploration**: Uses OpenHands agents with default tools (terminal, file operations) to explore repository structures and find relevant files
- **Tool-based Search**: Leverages CLI mode tools for efficient file searching, code reading, and repository navigation
- **Batch Processing**: Processes multiple repositories and questions sequentially
- **Resume Capability**: Automatically skips already answered questions for interrupted runs
- **Token Tracking**: Detailed tracking of prompt tokens (input) and completion tokens (output)
- **Fallback Mechanism**: If max iterations are reached without an answer, generates a summary based on conversation history
- **Comprehensive Statistics**: Tracks time costs and token usage for each question and repository

## Requirements

- Python 3.7+
- OpenHands SDK
- OpenAI Python client

### Installation

```bash
pip install openhands openai
```

## Configuration

Before running, configure the following settings in `main.py`:

### 1. LLM Configuration

Edit the `LLM_CONFIG` dictionary:

```python
LLM_CONFIG = {
    "model": "your-model-name",           # LLM model name
    "api_key": "your-api-key",            # API key for LLM
    "base_url": "your-api-base-url",      # Base URL for API endpoint
    "usage_id": "agent"                   # Usage identifier
}
```

### 2. Repository Configuration

Edit the `REPOS_CONFIG` list to specify repositories to process:

```python
REPOS_CONFIG = [
    {
        "name": "repository-name",
        "workspace": "/path/to/repository/workspace",
        "input_file": "/path/to/questions/repository-name.jsonl"
    },
    # Add more repositories as needed
]
```

### 3. Output Directory

Set the output directory for results:

```python
OUTPUT_DIR = "/path/to/output/directory"
```

### 4. Max Iterations

Configure the maximum number of iterations per question:

```python
MAX_ITERATION_PER_RUN = 10  # Default: 10
```

### 5. Tools Configuration

The script uses OpenHands default agent tools configured with CLI mode. The agent is created as follows:

```python
from openhands.tools.preset.default import get_default_agent

agent = get_default_agent(llm=llm, cli_mode=True)
```

**Important Notes about Tools:**

- **Default Tools**: The `get_default_agent()` function provides a preset collection of tools that enable the agent to:
  - **Terminal/Shell Access**: Execute shell commands to search files, navigate directories, and run code
  - **File Operations**: Read, write, and search files in the repository
  - **Code Analysis**: Analyze code structure and dependencies
  - **Repository Exploration**: Explore the codebase structure systematically

- **CLI Mode**: The `cli_mode=True` parameter enables command-line interface capabilities, allowing the agent to:
  - Use terminal tools for file searching (e.g., `grep`, `find`, `ls`)
  - Execute commands to understand the codebase structure
  - Navigate through directories efficiently

- **Tool Usage in Workflow**: 
  - The agent is explicitly instructed to use terminal tools for file searching
  - Tools are used iteratively (up to `MAX_ITERATION_PER_RUN` times) to explore and gather information
  - Each tool execution generates an `ObservationEvent` that is captured in the conversation history

- **Customization**: If you need to customize the tools, you can modify the agent creation in the `process_single_question()` function. However, the default tools are typically sufficient for code repository exploration tasks.

**Tool Execution Flow:**
1. Agent receives a question
2. Agent uses terminal tools to search for relevant files
3. Agent reads and analyzes relevant code files
4. Agent synthesizes information to generate an answer
5. All tool executions are logged as `ActionEvent` and `ObservationEvent` in the conversation state

## Input Format

Questions should be provided in JSONL format (one JSON object per line). Each line should contain:

```json
{"question": "Your question about the codebase"}
```

Example `questions.jsonl`:
```jsonl
{"question": "How does authentication work in this project?"}
{"question": "What is the main entry point of the application?"}
{"question": "How are database connections managed?"}
```

## Output Format

Answers are saved to JSONL files (one JSON object per line) with the following structure:

```json
{
    "question": "Original question",
    "answer": "Generated answer",
    "timestamp": "2024-01-01T12:00:00",
    "time_cost": 45.67,
    "token_cost": 12345,
    "prompt_tokens": 8000,
    "completion_tokens": 4345
}
```

Output files are named as `{repository_name}_answers.jsonl` and saved in the `OUTPUT_DIR` directory.

## Usage

### Basic Usage

```bash
python main.py
```

The script will:
1. Process each repository in `REPOS_CONFIG` in order
2. Load questions from the specified input file
3. Skip questions that have already been answered (resume capability)
4. For each question:
   - Create an agent to explore the codebase
   - Search for relevant files and code
   - Generate an answer
   - Save results immediately (append mode)
5. Print statistics after processing each repository and overall statistics

### Workflow

1. **Question Loading**: Questions are loaded from the input JSONL file
2. **Answer Checking**: Already answered questions are skipped
3. **Agent Creation**: An independent agent is created for each question with default tools enabled (CLI mode)
4. **Code Exploration**: The agent explores the repository structure using tools:
   - Terminal tools for file searching (`grep`, `find`, etc.)
   - File operation tools for reading and analyzing code
   - Navigation tools for exploring directory structures
5. **Tool Execution Tracking**: All tool actions and observations are logged in the conversation state
6. **Answer Generation**: The agent generates an answer based on findings from tool executions
7. **Fallback**: If max iterations are reached without an answer, a summary is generated from conversation history (including all tool execution logs)
8. **Statistics Collection**: Time and token usage are tracked for each question

## How It Works

1. **Initial Exploration**: The agent is instructed to first explore the codebase structure
2. **Tool-based File Search**: Uses terminal tools (via `cli_mode=True`) to search for files related to the question:
   - Executes shell commands (e.g., `grep`, `find`, `ls`) to locate relevant files
   - Reads and analyzes code files using file operation tools
   - Navigates through directory structures systematically
3. **Iterative Process**: The agent can perform up to `MAX_ITERATION_PER_RUN` tool actions:
   - Each action (e.g., file search, code reading) is logged as an `ActionEvent`
   - Tool execution results are captured as `ObservationEvent`
   - All events are stored in the conversation state for later analysis
4. **Answer Extraction**: Extracts the final answer from agent messages after tool exploration
5. **History-based Fallback**: If no direct answer is found after max iterations, uses the full conversation history (including all tool executions) to generate a summary answer via LLM

## Statistics

The script provides comprehensive statistics:

### Per Repository
- Success/Failed/Total counts
- Average and total time costs
- Average and total token costs
- Breakdown of prompt tokens (input) and completion tokens (output)

### Global
- Aggregate statistics across all repositories
- Overall time and token usage

## Error Handling

- Invalid JSON lines in input files are skipped with a warning
- Missing input files cause the repository to be skipped
- Failed questions are recorded with error messages
- Conversation cleanup errors are ignored to prevent crashes

## Notes

- **Incremental Processing**: Results are appended to output files, allowing safe interruption and resumption
- **Sequential Processing**: Questions are processed one at a time within each repository
- **Token Tracking**: Both aggregated (token_cost) and separated (prompt_tokens, completion_tokens) metrics are provided
- **Debug Output**: Extensive debug logging is available for troubleshooting

## Example Output

```
============================================================
Starting to process repository 1/3: reflex
============================================================
Workspace: /path/to/repos/reflex
Question file: /path/to/questions/reflex.jsonl
Loading questions from /path/to/questions/reflex.jsonl...
Loaded 50 questions in total
All questions are unanswered, will process all 50 questions

[reflex][1/50] Completed: How does authentication work in this...
[reflex][2/50] Completed: What is the main entry point...

Repository reflex processing completed:
  Success: 48, Failed: 2, Total: 50
  Average time_cost: 45.23 seconds
  Total time_cost: 2171.04 seconds
  Average token_cost: 12345 tokens
  Total token_cost: 592560 tokens
  Average prompt_tokens: 8000, completion_tokens: 4345
  Total prompt_tokens: 384000, completion_tokens: 208560
  Results saved to: /path/to/answer/OpenHands/reflex_answers.jsonl
```

## Troubleshooting

- **Import Errors**: Ensure OpenHands SDK is properly installed
- **Path Issues**: Verify all paths in configuration are correct and accessible
- **API Errors**: Check LLM configuration (API key, base URL, model name)
- **Empty Answers**: Check if max iterations are too low or if questions require more exploration

