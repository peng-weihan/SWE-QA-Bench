# RAG Function Chunk Parallel Processing Module

A parallel question-answering processing module based on function chunk RAG technology for processing question-answer pairs from multiple code repositories.

## Key Features

- 🔄 **Parallel Processing**: Multi-threaded concurrent processing
- 🧠 **RAG Technology**: Function chunk RAG with Voyage AI embeddings
- 📊 **Statistics**: Processing results and answer quality metrics
- 💾 **FAISS Indexing**: Efficient code similarity search

## File Structure
```
rag_function_chunk/
├── main.py                 # Main entry point
├── func_chunk_rag.py       # RAG implementation
├── env.example            # Configuration template
└── README.md              # This file
```

## Quick Start

### 1. Installation
```bash
pip install openai voyageai faiss-cpu numpy tqdm python-dotenv
```

### 2. Configuration
Copy `env.example` to `.env` and configure:

```bash
# Required API Keys
VOYAGE_API_KEY=your_voyage_api_key
OPENAI_API_KEY=your_openai_api_key

# Model Configuration
MODEL=DeepSeek-V3
TEMPERATURE=0
MAX_WORKERS=16

# Repository List (comma-separated)
REPOS=requests,flask,pytest,sphinx,xarray,pylint,matplotlib

# Paths
BASE_INPUT_PATH=/path/to/input/questions
BASE_OUTPUT_PATH=/path/to/output/answers
EMBEDDINGS_BASE_PATH=/path/to/embeddings/storage
```

### 3. Usage
```bash
cd {PROJECT_ROOT}/methods/rag_function_chunk
python main.py
```

## Data Formats

### Input (JSONL)
```jsonl
{"question": "What are Requests' built-in authentication handlers?", "answer": null, "relative_code_list": null, "ground_truth": "Requests' built-in authentication handlers include...", "score": null}
```

### Output (JSONL)
```jsonl
{"question": "What are Requests' built-in authentication handlers?", "answer": "Requests' built-in authentication handlers include HTTPBasicAuth, HTTPDigestAuth, and HTTPProxyAuth.", "thought": "The question asks about Requests' built-in authentication handlers...", "ground_truth": "Requests' built-in authentication handlers include..."}
```

## Core Components

### FuncChunkRAG Class
- **Code Embedding**: Voyage AI `voyage-code-3` model
- **Vector Search**: FAISS IVFFlat index for similarity search
- **Answer Generation**: LLM with structured JSON output

### Processing Engine
- **Concurrent Processing**: Configurable thread pool
- **Thread Safety**: Lock-protected file operations
- **Error Handling**: Automatic retry and graceful recovery

## Configuration Parameters

| Variable | Description | Default |
|----------|-------------|---------|
| `VOYAGE_API_KEY` | Voyage AI API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `MODEL` | LLM model name | `DeepSeek-V3` |
| `MAX_WORKERS` | Concurrent threads | `16` |
| `REPOS` | Repository list | Required |
| `BASE_INPUT_PATH` | Input directory | Required |
| `BASE_OUTPUT_PATH` | Output directory | Required |
| `EMBEDDINGS_BASE_PATH` | Embeddings storage | Required |

## API Reference

### FuncChunkRAG
```python
from methods.rag_function_chunk import FuncChunkRAG
from methods import QAPair

# Initialize
rag = FuncChunkRAG(save_path="embeddings.json")

# Process single question
qa_pair = QAPair(question="Your question", answer=None, ...)
result = rag.process_qa_pair(qa_pair)

# Find relevant code
relevant_code = rag.find_relevant_code("query", top_k=10)
```

### Main Functions
- `load_data_from_jsonl(path)`: Load QAPair data from JSONL
- `run_questions_concurrently(...)`: Process questions in parallel
- `append_data_to_jsonl(path, data)`: Thread-safe file writing

## Troubleshooting

### Common Issues
- **JSON Parsing Error**: Automatically handles markdown formatting
- **File Path Error**: Auto-creates required directories
- **API Limits**: Adjust `MAX_WORKERS` parameter

### Performance Tips
- Use SSD storage for FAISS index
- Monitor memory usage with large datasets
- Adjust batch size based on API limits

## License
Follows project root directory license agreement.