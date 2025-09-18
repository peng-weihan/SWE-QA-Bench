# SWE-QA Agent

A large language model-based software engineering question-answering agent that can analyze code repositories and answer related questions.

## Features

- **Intelligent Code Analysis**: Uses RAG (Retrieval-Augmented Generation) technology to search for relevant code snippets
- **Multi-tool Support**: Integrates file reading, directory traversal, code search, and other tools
- **Batch Processing**: Supports both single Q&A and batch processing modes
- **Fault Tolerance**: Built-in token limit retry and error handling mechanisms
- **Visual Logging**: Beautiful console output using the Rich library

## Core Components

### Main Files

- `main.py`: Main program entry point, supports single Q&A and batch processing
- `agent.py`: Core Agent implementation, based on LangGraph state graph workflow
- `config.py`: Configuration management, including API keys and model parameters
- `history.py`: Conversation history management

### Tool Modules

- `tools/repo_rag.py`: RAG search tool using Voyage AI for semantic search
- `tools/repo_read.py`: File reading tool supporting tree, ls, cat, grep commands

### Prompt Templates

- `prompts/react_prompt.txt`: ReAct-style prompt template guiding the Agent to efficiently analyze code

## Installation

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv sync
```

## Environment Configuration

Create a `.env` file and configure the following environment variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0

# Voyage AI Configuration (for RAG search)
VOYAGE_API_KEY=your_voyage_api_key
VOYAGE_MODEL=voyage-large-2

# Repository Paths
REPOS=requests,flask,sqlfluff,pytest,sphinx,xarray,pylint,matplotlib,scikit-learn,astropy,django,sympy
```

## Usage

### Single Q&A Mode

```bash
python main.py single "Your question here" /path/to/repository
```

### Batch Processing Mode

```bash
python main.py batch
```

### Default Batch Processing

Running `main.py` directly will process all configured repositories:

```bash
python main.py
```

## Input Format

Batch processing input files should be in JSONL format, with one question per line:

```json
{"question": "How does the authentication work in this codebase?"}
{"question": "What is the main entry point of the application?"}
```

**Note**: Batch processing automatically uses the following file structure:
- Input files: `{PROJECT_ROOT}/datasets/questions/{repo}.jsonl`
- Output files: `{PROJECT_ROOT}/datasets/answers/swe_qa_agent/{model}/{repo}.jsonl`
- Repository paths: `{PROJECT_ROOT}/datasets/repos/{repo}`

## Output Format

Output contains the following fields:

```json
{
  "question": "User question",
  "answer": "Generated answer",
  "status": "✅ Success",
  "history_size_used": 5,
  "retry_attempts": 0,
  "error": null
}
```

## Workflow

1. **Initialization**: Load configuration and tools
2. **Question Analysis**: Understand user questions
3. **Code Search**: Use RAG or direct file operations to search for relevant code
4. **Information Gathering**: Read and analyze relevant files
5. **Answer Generation**: Generate final answers based on collected information
6. **Result Output**: Return structured answers and metadata

## Configuration Parameters

- `MAX_ITERATIONS`: Maximum number of iterations (default: 5)
- `HISTORY_WINDOW`: Conversation history window size (default: 5)
- `SEARCH_RESULTS_LIMIT`: Search results limit (default: 20)
- `GREP_WINDOW_SIZE`: Grep context window size (default: 100)
- `PARALLEL_WORKERS`: Number of parallel workers for batch processing (default: 1)

## Technology Stack

- **LangChain**: Large language model integration framework
- **LangGraph**: State graph workflow management
- **Rich**: Terminal beautification output
- **Voyage AI**: Semantic search embedding model
- **FAISS**: Vector similarity search

## Notes

- Ensure sufficient API quota
- Large repositories may require longer processing time
- Recommend testing configuration in a test environment first
- Batch processing uses thread pool for parallel execution
- Results are written to output files immediately after each query completion
- The tool automatically creates necessary output directories
