# LLM Direct: Direct Q&A Method

LLM Direct is a direct question-answering method in the SWE-QA project that answers questions by calling large language models (LLM) directly, without additional tool calls or code analysis.

## Features

- **Direct Q&A**: Based on LLM's direct question-answering capabilities, no tool calls required
- **Batch Processing**: Supports parallel processing of multiple repositories for improved efficiency
- **Real-time Saving**: Saves results immediately after processing each question to prevent data loss
- **Multi-threading**: Supports question-level parallel processing to maximize processing speed
- **Error Handling**: Comprehensive exception handling mechanism to ensure stable program operation

## Architecture Design

### Overall Architecture
```
Question Files → LLM Direct → Parallel Processing → Real-time Saving → Answer Files
```

### Core Components

1. **Question Loader**: Loads question data from JSONL files
2. **LLM Client**: Uses OpenAI-compatible API to call large language models
3. **Parallel Processor**: Multi-threaded question processing for improved efficiency
4. **Real-time Saver**: Saves results immediately after processing each question

## Configuration

### Environment Variables Setup

1. **Copy the example configuration file:**
   ```bash
   cp config.env.example .env
   ```

2. **Edit the `.env` file with your actual values:**
   ```bash
   # API Configuration
   OPENAI_BASE_URL=https://xxx.xxx.xxx
   OPENAI_API_KEY=your_actual_api_key_here
   
   # Model Configuration
   MODEL=gpt-4o
   TEMPERATURE=0

   # Parallel Processing Configuration
   REPO_MAX_WORKERS=1
   QUESTION_MAX_WORKERS=1
   
   # Repository Configuration (comma-separated list)
   REPOS=requests,flask,sqlfluff,pytest,sphinx,xarray,pylint,matplotlib,scikit-learn,astropy,django,sympy
   ```

### Configuration Items

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | API key for OpenAI service | - | Yes |
| `OPENAI_BASE_URL` | Base URL for API requests | `https://api.openai.com/v1` | No |
| `MODEL` | Model name to use | `gpt-4o` | No |
| `TEMPERATURE` | Generation temperature parameter | `0` | No |
| `REPO_MAX_WORKERS` | Repository-level parallelism | `1` | No |
| `QUESTION_MAX_WORKERS` | Question-level parallelism | `1` | No |
| `REPOS` | Comma-separated list of repositories to process | `requests,flask,sqlfluff,pytest,sphinx,xarray,pylint,matplotlib,scikit-learn,astropy,django,sympy` | No |

### Path Configuration

The system automatically uses the following default paths:
- **Questions Directory**: `{PROJECT_ROOT}/datasets/questions`
- **Answers Directory**: `{PROJECT_ROOT}/datasets/answers/direct`

### Supported Repositories

The system supports processing multiple repositories. You can configure which repositories to process using the `REPOS` environment variable. By default, it processes the following 12 open-source repositories:

- requests
- flask
- sqlfluff
- pytest
- sphinx
- xarray
- pylint
- matplotlib
- scikit-learn
- astropy
- django
- sympy

To process only specific repositories, set the `REPOS` variable in your `.env` file:
```bash
# Process only specific repositories
REPOS=requests,flask,pytest
```

## Usage

### Running the Program

```bash
# 1. Navigate to the llm_direct directory
cd {PROJECT_ROOT}/methods/llm_direct

# 2. Ensure .env file exists and is configured
cp env.example .env
# Edit .env file with your actual API key

# 4. Run the program
python main.py
```

### Input Format

Question files should be in JSONL format, with one JSON object per line:

```jsonl
{"question": "What is the main functionality of this project?"}
{"question": "How to install this library?"}
{"question": "What are the parameters of this function?"}
```

### Output Format

Answer files are in JSONL format, with each line containing the original question and generated answer:

```jsonl
{"question": "What is the main functionality of this project?", "answer": "This is a library for..."}
{"question": "How to install this library?", "answer": "You can install it via pip..."}
```

## Core Functions

### 1. Question Processing Flow

```python
def process_single_question(question_data, repo_name):
    """Process a single question and return results"""
    question = question_data['question']
    direct_answer = get_llm_answer(question, repo_name)
    question_data['answer'] = direct_answer
    return question_data
```

### 2. LLM Q&A

```python
def get_llm_answer(question: str, repo_name: str):
    """Get direct answer from LLM"""
    system_messages = [
        "You are a direct answer generator. Provide ONLY the direct answer to the question. Do not include explanations, citations, references, or any additional content. Give the most concise and direct response possible."
    ]
    # ... API call logic
```

### 3. Parallel Processing

- **Repository-level Parallelism**: Process multiple repositories simultaneously (default: 1)
- **Question-level Parallelism**: Process multiple questions in parallel within each repository (default: 32)
- **Real-time Saving**: Save immediately after processing each question to prevent data loss

## Performance Optimization

### Parallel Processing Strategy

1. **Repository-level Parallelism**: Control the number of repositories processed simultaneously to avoid resource competition
2. **Question-level Parallelism**: Process questions in parallel within a single repository to maximize throughput
3. **Real-time Saving**: Avoid memory accumulation and release resources promptly

### Error Handling

- Return error information when API calls fail
- Log error information when file read/write exceptions occur
- Handle network timeout and other exception scenarios

## Project Structure

```
llm_direct/
├── __init__.py          # Package initialization file
├── main.py             # Main program entry point
└── README.md           # Project documentation
```

## Dependencies

- Python 3.7+
- openai
- tqdm
- python-dotenv
- concurrent.futures

### Installation

```bash
pip install openai tqdm python-dotenv
```

## Notes

1. **Working Directory**: Must run the program from the `llm_direct` directory
2. **Environment Setup**: Copy `config.env.example` to `.env` and configure your API key
3. **API Configuration**: Ensure OpenAI API key is valid and has sufficient quota
4. **Path Configuration**: Ensure input and output paths exist and have read/write permissions
5. **Parallel Control**: Adjust parallel parameters based on system resources
6. **Error Monitoring**: Pay attention to error information in console output

## FAQ

### 1. API Call Failure
- Check if OpenAI API key is correctly set in `.env` file
- Confirm network connection is normal
- Verify OpenAI API quota is sufficient

### 2. File Path Error
- Ensure input files exist
- Check output directory permissions
- Verify path configuration is correct

### 3. Parallel Processing Exception
- Reduce parallel count
- Check system resource usage
- Monitor memory and CPU usage

## License

[MIT License](LICENSE)

## Contributing

Pull Requests and Issues are welcome!
