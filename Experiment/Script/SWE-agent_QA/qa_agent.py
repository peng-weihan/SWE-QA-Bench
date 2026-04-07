"""Lightweight QA Agent, based on SWE-agent's core framework"""
import logging
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field

from local_env import LocalEnv
from agent_models import ModelConfig, get_model
from agent_tools import ToolHandler, ToolConfig
from problem_statement import TextProblemStatement, ProblemStatement

logger = logging.getLogger(__name__)


class QAAgentConfig(BaseModel):
    """QA Agent configuration"""
    model: ModelConfig = Field(description="Model configuration")
    tools: ToolConfig = Field(default_factory=ToolConfig, description="Tool configuration")
    max_steps: int = 10
    """Maximum execution steps (iteration limit)"""
    max_observation_length: int = 100_000
    """Maximum observation length"""


class QAAgent:
    """Lightweight QA Agent"""
    
    def __init__(
        self,
        config: QAAgentConfig,
        env: LocalEnv,
        problem_statement: ProblemStatement,
    ):
        self.config = config
        self.env = env
        self.problem_statement = problem_statement
        self.model = get_model(config.model)
        self.tool_handler = ToolHandler(config.tools, env)
        self.history = []
        self.step_count = 0
        self.stats = {
            "total_latency": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
        
    def run(self) -> dict[str, Any]:
        """Run QA task"""
        logger.info(f"Starting QA task: {self.problem_statement.id}")
        
        # Initialize environment variables
        if self.config.tools.env_variables:
            self.env.set_env_variables(self.config.tools.env_variables)
        
        # Build initial prompt
        system_prompt = self._build_system_prompt()
        instance_prompt = self._build_instance_prompt()
        
        # Add to history
        self.history.append({"role": "system", "content": system_prompt})
        self.history.append({"role": "user", "content": instance_prompt})
        
        # Main loop
        while self.step_count < self.config.max_steps:
            self.step_count += 1
            logger.info(f"Step {self.step_count}/{self.config.max_steps}")
            
            # Warn when approaching limit
            if self.step_count >= self.config.max_steps - 2:
                logger.warning(f"Approaching iteration limit ({self.config.max_steps}), {self.config.max_steps - self.step_count} steps remaining")
            
            # Call model
            response, step_stats = self.model.forward(self.history)
            logger.debug(f"Model response: {response[:200]}...")
            
            # Accumulate statistics
            self.stats["total_latency"] += step_stats["latency"]
            self.stats["total_input_tokens"] += step_stats["input_tokens"]
            self.stats["total_output_tokens"] += step_stats["output_tokens"]
            
            # Parse tool calls
            action = self.tool_handler.parse_action(response)
            
            if action is None:
                # Check if contains explicit end signal
                if any(keyword in response.lower() for keyword in ["final answer", "answer:"]):
                    logger.info("Found final answer signal, treating as final answer")
                    self.history.append({"role": "assistant", "content": response})
                    break
                # If no tool call, prompt model to use tools
                logger.warning("No action parsed from response, prompting to use tools")
                self.history.append({"role": "assistant", "content": response})
                self.history.append({
                    "role": "user",
                    "content": "Please use the available tools to explore the repository. Use JSON format: {\"name\": \"tool_name\", \"arguments\": {...}}. Remember: Your final answer must be in English."
                })
                continue
            
            # Execute tool
            observation = self.tool_handler.execute_action(action, self.env)
            
            # Truncate observation
            if len(observation) > self.config.max_observation_length:
                observation = observation[:self.config.max_observation_length] + "\n<response clipped>"
            
            # Add to history
            self.history.append({"role": "assistant", "content": response})
            self.history.append({
                "role": "user",
                "content": f"Observation:\n{observation}"
            })
        
        # Check if limit reached
        if self.step_count >= self.config.max_steps:
            logger.warning(f"Reached iteration limit ({self.config.max_steps}), stopping execution")
            # Add prompt to history
            self.history.append({
                "role": "user",
                "content": f"You have reached the maximum number of steps ({self.config.max_steps}). Please provide your final answer based on the information you have gathered. IMPORTANT: Your final answer MUST be written in English."
            })
            # Last call to model to get answer
            response, step_stats = self.model.forward(self.history)
            self.stats["total_latency"] += step_stats["latency"]
            self.stats["total_input_tokens"] += step_stats["input_tokens"]
            self.stats["total_output_tokens"] += step_stats["output_tokens"]
            self.history.append({"role": "assistant", "content": response})
        
        # Extract final answer
        final_answer = self._extract_answer()
        
        return {
            "answer": final_answer,
            "steps": self.step_count,
            "history": self.history,
            "latency": self.stats["total_latency"],
            "input_tokens": self.stats["total_input_tokens"],
            "output_tokens": self.stats["total_output_tokens"],
        }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        tools_desc = """
Available tools:
1. bash: Execute bash commands. Format: {"name": "bash", "arguments": {"command": "your command"}}
   Example: {"name": "bash", "arguments": {"command": "ls -la"}}
2. read_file: Read a file. Format: {"name": "read_file", "arguments": {"path": "file/path"}}
3. grep: Search for patterns in files. Format: {"name": "grep", "arguments": {"pattern": "pattern", "path": "directory"}}
4. ls: List directory contents. Format: {"name": "ls", "arguments": {"path": "directory"}}

When you want to use a tool, respond with a JSON object in this format:
{"name": "tool_name", "arguments": {...}}

After each tool execution, you will receive the output. Continue exploring until you have enough information to answer the question.

IMPORTANT: You must provide your final answer in English. All responses, explanations, and conclusions should be written in English.
"""
        return f"""You are a helpful assistant that can interact with a code repository to answer questions about it.
You have access to the repository's files and can use various tools to explore and understand the codebase.

{tools_desc}"""
    
    def _build_instance_prompt(self) -> str:
        """Build instance prompt"""
        question = self.problem_statement.get_problem_statement()
        repo_path = self.env.repo_path
        
        return f"""I've uploaded a code repository in the directory {repo_path}. Please answer the following question about this repository:

<question>
{question}
</question>

Your task is to thoroughly explore the repository and provide a comprehensive answer to the question.
Follow these steps to answer the question:
1. First, explore the repository structure to understand what the codebase is about
2. Search for relevant code, files, and documentation related to the question
3. Read and analyze the relevant code sections
4. Provide a clear, detailed answer based on your findings
5. If applicable, include code examples or references to specific files/functions

Your thinking should be thorough and comprehensive. Take your time to explore the codebase properly.

IMPORTANT: You MUST provide your final answer in English. All your responses, explanations, code analysis, and conclusions must be written in English, regardless of the language of the question or code comments."""
    
    def _extract_answer(self) -> str:
        """Extract final answer from history"""
        # Simple implementation: return last assistant message
        for msg in reversed(self.history):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return "No answer found"

