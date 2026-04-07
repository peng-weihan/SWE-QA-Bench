# RAG Sliding Window Parallel Processing Module

A parallel question-answering processing module based on sliding window RAG technology for processing question-answer pairs from multiple code repositories using FAISS vector search and Voyage AI embeddings.

## Key Features

- 🔄 **Parallel Processing**: Multi-threaded concurrent processing with configurable worker threads
- 🧠 **Sliding Window RAG**: Advanced RAG implementation with sliding window approach
- 📊 **Statistics**: Comprehensive processing results and answer quality metrics
- 💾 **FAISS Indexing**: Efficient vector similarity search with automatic fallback strategies
- 🌊 **Voyage AI Integration**: High-quality code embeddings with batch processing optimization

## File Structure

```
rag_sliding_window/
├── main.py                    # Main entry point for parallel processing
├── sliding_windows_rag.py     # Core RAG implementation with sliding window approach
├── env.example               # Configuration template
├── __init__.py               # Package initialization
└── README.md                 # This file
```

## Core Components

### 1. RAGSlidingWindowsCodeQA Class

The main RAG implementation featuring:
- **Voyage AI Embeddings**: High-quality code embeddings with automatic batch size optimization
- **FAISS Vector Search**: Efficient similarity search with IVFFlat and Flat indexing strategies
- **Sliding Window Approach**: Advanced code chunking for better context understanding
- **Error Handling**: Robust error handling with automatic retry mechanisms

### 2. VoyageEmbeddingModel Wrapper

- **Batch Processing**: Optimized batch processing with automatic size adjustment
- **Error Recovery**: Automatic fallback to smaller batch sizes on API errors
- **Truncation Support**: Built-in text truncation for API compatibility

### 3. Parallel Processing Engine

- **Thread-Safe Operations**: File writing with thread locks to prevent data corruption
- **Concurrent Execution**: Configurable worker thread pool for maximum efficiency
- **Progress Tracking**: Real-time progress monitoring with detailed statistics

## Quick Start

### 1. Installation

```bash
pip install openai voyageai faiss-cpu numpy tqdm python-dotenv
```

### 2. Configuration

Copy `env.example` to `.env` and configure:

```bash
# Voyage AI Configuration
VOYAGE_API_KEY=your_voyage_api_key
VOYAGE_MODEL=voyage-code-3

# OpenAI Configuration
OPENAI_URL=https://your-openai-proxy.com/v1
OPENAI_KEY=your_openai_api_key

# Model Configuration
MODEL=DeepSeek-V3
TEMPERATURE=0

# Concurrency Configuration
MAX_WORKERS=16
MAX_LINES=100

# Repository Configuration (comma-separated list)
REPOS=requests,flask,sqlfluff,pytest,sphinx,xarray,pylint,matplotlib,scikit-learn,astropy,django,sympy
```

### 3. Usage

```bash
# Run parallel processing for all configured repositories
python main.py

# The script will automatically:
# 1. Load question-answer pairs from input JSONL files
# 2. Process questions using sliding window RAG
# 3. Generate answers with thought process and ground truth
# 4. Save results to output JSONL files
# 5. Generate comprehensive statistics
```

## Processing Workflow

1. **Data Loading**: Load QAPair objects from JSONL input files
2. **RAG Initialization**: Initialize RAG model with FAISS index and Voyage embeddings
3. **Parallel Processing**: Process questions concurrently using ThreadPoolExecutor
4. **Answer Generation**: Generate structured answers with JSON format validation
5. **Result Storage**: Save results with thread-safe file operations
6. **Statistics**: Generate comprehensive processing statistics

## Output Format

Each processed question generates a structured result:

```json
{
  "question": "Original question text",
  "answer": "Generated answer",
  "thought": "Reasoning process",
  "ground_truth": "Ground truth answer"
}
```

## Statistics Generated

The module provides detailed statistics including:
- **Processing Count**: Successfully processed items
- **Error Count**: Failed processing attempts
- **Word Count Metrics**: Min, max, average, and median word counts
- **Processing Time**: Total time per repository
- **Success Rate**: Percentage of successful processing

## Advanced Features

### FAISS Index Management

- **Automatic Index Creation**: Creates FAISS indices with optimal parameters
- **Index Persistence**: Saves and loads indices for reuse
- **Memory Optimization**: Uses IVFFlat for large datasets, Flat for small datasets
- **Index Statistics**: Provides detailed index information and health metrics

### Embedding Optimization

- **Batch Size Adaptation**: Automatically adjusts batch sizes based on API limits
- **Error Recovery**: Implements retry mechanisms with exponential backoff
- **Memory Management**: Efficient memory usage for large-scale processing

### Error Handling

- **Graceful Degradation**: Continues processing even when individual items fail
- **Detailed Logging**: Comprehensive error logging with context information
- **Resource Cleanup**: Proper cleanup on interruption or errors

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_WORKERS` | Number of concurrent threads | 16 |
| `VOYAGE_MODEL` | Voyage AI embedding model | voyage-code-3 |
| `MODEL` | LLM model for answer generation | gpt-4o |
| `TEMPERATURE` | LLM temperature setting | 0 |
| `REPOS` | Comma-separated repository list | Multiple repos |

## Performance Optimization

- **Concurrent Processing**: Utilizes all available CPU cores
- **Batch Embeddings**: Optimized batch processing for API efficiency
- **FAISS Indexing**: Fast vector similarity search
- **Memory Management**: Efficient memory usage patterns
- **File I/O Optimization**: Thread-safe file operations

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce `MAX_WORKERS` if hitting rate limits
2. **Memory Issues**: Monitor memory usage with large datasets
3. **FAISS Index Errors**: Ensure sufficient disk space for index files
4. **JSON Parsing Errors**: Check LLM output format consistency

### Debug Mode

Enable detailed logging by modifying the print statements in the code for more verbose output.

## Dependencies

- `openai`: OpenAI API client
- `voyageai`: Voyage AI embeddings
- `faiss-cpu`: FAISS vector search
- `numpy`: Numerical operations
- `tqdm`: Progress bars
- `python-dotenv`: Environment variable management

## License

This module is part of the SWE-QA project for automated question-answering on software engineering repositories.
