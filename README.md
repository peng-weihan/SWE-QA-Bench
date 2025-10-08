# SWE-QA

**SWE-QA: Can Language Models Answer Repository-level Code Questions?**
This repository contains code and data for the SWE-QA benchmark, which evaluates language models' ability to answer repository-level code questions across 12 popular Python projects including Django, Flask, Requests, and more.

## 📝 Prompts

The detailed prompt templates used in the paper are in the `supplementary.pdf` file

## 📊 Dataset

The benchmark dataset is available on Hugging Face:
- **Dataset**: [SWE-QA-Benchmark](https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark)

## 📖 Paper

For more details about the methodology and results, please refer to the paper:
- **Paper**: "SWE-QA: Can Language Models Answer Repository-level Code Questions?"

## 📁 Repository Structure

```
SWE-QA-Bench/
├── SWE-QA-Bench/                    # Main package directory
│   ├── datasets/              # Dataset files and repositories
│   │   ├── questions/         # Question datasets (JSONL format)
│   │   │   ├── astropy.jsonl  # Project-specific datasets
│   │   │   ├── django.jsonl
│   │   │   ...
│   │   ├── answers/           # Answer datasets
│   │   ├── faiss/             # FAISS index files
│   │   └── repos/             # Repository data
│   ├── issue_analyzer/        # GitHub issue analysis
│   │   ├── get_question_from_issue.py
│   │   └── pull_issues.py
│   ├── methods/               # Evaluation methods
│   │   ├── llm_direct/        # Direct LLM evaluation
│   │   ├── rag_function_chunk/ # RAG with function chunking
│   │   ├── rag_sliding_window/ # RAG with sliding window
│   │   ├── code_formatting.py
│   │   └── data_models.py
│   ├── score/                 # Scoring utilities
│   │   └── llm-score.py       # LLM-as-a-judge evaluation
│   ├── models/                # Data models
│   │   └── data_models.py
│   └── utils/                 # Utility functions
├── docs/                      # Documentation of each part
│   └── README.md
├── LICENSE                    # License file
├── supplementary.pdf          # Supplementary file (prompts)
├── clone_repos.sh             # Script to clone repositories at specific commits
├── repos.txt                  # List of repository URLs and commit hashes
├── requirements.txt           # Python dependencies required to run the project
└── README.md                  # This file
```


## 🚀 Environment Setup

### Prerequisites

- Python 3.12
- pip or conda for package management
- OpenAI API access (required for all evaluation methods)
- Voyage AI API access (required for RAG-based methods)

### Installation
**Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
**SWE Repository Prerequisites:**
   ```bash
   # Use the provided script to clone all repositories at specific commits
   ./clone_repos.sh
   ```

## ⚡ Quick Start

### 1. Direct LLM Evaluation

Before executing, you need to configure the environment variables by filling the `.env` file in the `SWE-QA-Bench/methods/llm_direct` directory:
```bash
OPENAI_BASE_URL=your_openai_base_url
OPENAI_API_KEY=your_api_key
MODEL=your_model_name
```

Evaluate language models directly on repository-level questions:
```bash
cd SWE-QA-Bench/methods/llm_direct
python main.py
```

This method will:
- Load questions from the dataset
- Send questions directly to the LLM
- Generate answers without additional context
- Save results to `datasets/answers/direct/`

### 2. RAG with Function Chunking
Before executing, you need to configure the environment variables by filling the `.env` file in the `SWE-QA-Bench/methods/rag_function_chunk` directory:
```bash
# Voyage AI Configuration
VOYAGE_API_KEY=
VOYAGE_MODEL=  # voyage-code-3 recommended

# OpenAI Configuration
OPENAI_BASE_URL=
OPENAI_API_KEY=
MODEL=
```

Use RAG with function-level code chunking:

```bash
cd SWE-QA-Bench/methods/rag_function_chunk
python main.py
```

This method will:
- Parse code into function-level chunks
- Build vector embeddings for code chunks
- Retrieve relevant code context for each question
- Generate answers using retrieved context

### 3. RAG with Sliding Window

Before executing, you need to configure the environment variables by filling the `.env` file in the `SWE-QA-Bench/methods/rag_sliding_window` directory:
```bash
# Voyage AI Configuration
VOYAGE_API_KEY=
VOYAGE_MODEL=   # voyage-code-3 recommended

# OpenAI Configuration
OPENAI_URL=
OPENAI_KEY=
MODEL=
```

Use RAG with sliding window text chunking:

```bash
cd SWE-QA-Bench/methods/rag_sliding_window
python main.py
```

This method will:
- Split code into overlapping text windows
- Create embeddings for text chunks
- Retrieve relevant chunks for each question
- Generate contextual answers

### 4. Evaluation and Scoring
Before executing, you need to configure the environment variables by filling the `.env` file in the `SWE-QA-Bench/score` directory:
```bash
OPENAI_BASE_URL=your_openai_base_url
OPENAI_API_KEY=your_api_key
MODEL=your_model_name

METHOD= # choose from [direct, func_chunk, sliding_window]
```

Evaluate generated answers using LLM-as-a-judge:
```bash
cd SWE-QA-Bench/score
python llm-score.py
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.