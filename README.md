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
└── README.md                  # This file

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.