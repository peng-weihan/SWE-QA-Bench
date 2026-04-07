# SWE-Agent Lite: Lightweight QA Agent

This is a lightweight QA system extracted from SWE-agent that **does not require Docker**, directly using the local file system to execute commands.

## Key Design Philosophy

**This implementation preserves the original SWE-agent's core logic for tools and agent architecture.** We only modified the **task type interface (entry and exit points)** to adapt it for QA scenarios without requiring Docker. The core agent logic, tool handling system, and execution flow remain consistent with the original SWE-agent design.

### What's Preserved

- ✅ **Agent Core Logic**: The main agent loop, step execution, and decision-making logic
- ✅ **Tool System**: Tool parsing, execution, and filtering mechanisms
- ✅ **Model Interface**: Model configuration and forwarding logic
- ✅ **Execution Flow**: The iterative exploration and response generation workflow

### What's Changed

- 🔄 **Task Interface**: Modified entry (problem statement) and exit (answer extraction) to support QA tasks
- 🔄 **Execution Environment**: Replaced Docker with lightweight local file system execution
- 🔄 **Task Focus**: Adapted for question answering instead of code fixing

## Features

- ✅ **No Docker dependency**: Directly uses local file system
- ✅ **Preserves SWE-agent core**: Maintains original tools and agent logic
- ✅ **Lightweight**: No Docker overhead for QA scenarios
- ✅ **Easy to use**: Simple command-line interface

## Installation

```bash
cd swe-agent-lite
pip install litellm pydantic
```

## Usage

### Basic Usage

```bash
python main.py \
    --question "What is the main functionality of this project?" \
    --repo /path/to/repo \
    --model gpt-4o \
    --api-key your-api-key
```

### Using Custom API

```bash
python main.py \
    --question "What is the main functionality of this project?" \
    --repo /path/to/repo \
    --model gpt-4o \
    --api-key your-api-key \
    --api-base https://api.openai.com/v1
```

### Full Parameters

```bash
python main.py \
    --question "Question" \
    --repo /path/to/repo \
    --model gpt-4o \
    --api-key your-api-key \
    --api-base https://api.openai.com/v1 \
    --max-steps 50
```

## Architecture

- `local_env.py`: Local execution environment (replaces Docker)
- `qa_agent.py`: QA Agent main class
- `agent_models.py`: Model interface (based on LiteLLM)
- `agent_tools.py`: Tool handling system
- `problem_statement.py`: Problem statement processing
- `main.py`: Main program entry point

## Differences from SWE-agent

1. **No Docker**: Replaced Docker container execution with lightweight local file system, eliminating Docker dependency for QA scenarios
2. **Task Type Adaptation**: Only modified the **entry point** (problem statement interface) and **exit point** (answer extraction) to support QA tasks, while preserving all core agent and tool logic
3. **Simplified Infrastructure**: Removed complex components like deployment, hooks, reviewer, etc., that are specific to code fixing workflows
4. **QA-focused**: Adapted specifically for question answering tasks while maintaining the same core exploration and tool usage patterns

## Limitations

- Does not support complex tool bundles
- Does not support multi-step code modifications
- Focused on QA tasks, not suitable for code fixing
