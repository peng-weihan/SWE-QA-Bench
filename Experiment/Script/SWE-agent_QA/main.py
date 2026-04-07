#!/usr/bin/env python3
"""Lightweight QA Agent main program"""
import argparse
import logging
from pathlib import Path

from qa_agent import QAAgent, QAAgentConfig
from agent_models import ModelConfig
from agent_tools import ToolConfig
from local_env import LocalEnv
from problem_statement import TextProblemStatement

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Lightweight QA Agent")
    parser.add_argument("--question", "-q", required=True, help="Question to answer")
    parser.add_argument("--repo", "-r", required=True, type=Path, help="Repository path")
    parser.add_argument("--model", "-m", default="gpt-4o", help="Model name")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--api-base", help="API base URL")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum execution steps (iteration limit), default 10")
    
    args = parser.parse_args()
    
    # Create model configuration
    model_config = ModelConfig(
        name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    
    # Create tool configuration
    tool_config = ToolConfig()
    
    # Create Agent configuration
    agent_config = QAAgentConfig(
        model=model_config,
        tools=tool_config,
        max_steps=args.max_steps,
    )
    
    # Create environment
    env = LocalEnv(repo_path=args.repo)
    
    # Create problem statement
    problem_statement = TextProblemStatement(text=args.question)
    
    # Create and run Agent
    agent = QAAgent(agent_config, env, problem_statement)
    result = agent.run()
    
    # Output results
    print("\n" + "="*80)
    print("Final Answer:")
    print("="*80)
    print(result["answer"])
    print(f"\nExecuted {result['steps']} steps")


if __name__ == "__main__":
    main()

