"""Simplified tool handling system, based on SWE-agent's tool framework"""
import json
import re
import logging
from pathlib import Path
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field

from local_env import LocalEnv

logger = logging.getLogger(__name__)


class ToolFilterConfig(BaseModel):
    """Tool filter configuration"""
    blocklist: list[str] = Field(default_factory=lambda: [
        "vim", "vi", "emacs", "nano", "python", "python3", "bash", "sh"
    ])


class ToolConfig(BaseModel):
    """Tool configuration"""
    filter: ToolFilterConfig = Field(default_factory=ToolFilterConfig)
    env_variables: Dict[str, Any] = Field(default_factory=lambda: {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
    })
    enable_bash_tool: bool = True


class ToolHandler:
    """Tool handler"""
    
    def __init__(self, config: ToolConfig, env: LocalEnv):
        self.config = config
        self.env = env
    
    def parse_action(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from model response
        
        Supports two formats:
        1. Function calling (JSON)
        2. Simple command format
        """
        # Try to parse complete JSON object (may contain nesting)
        try:
            # First try to parse entire response directly
            action = json.loads(response.strip())
            if isinstance(action, dict) and "name" in action:
                return action
        except:
            pass
        
        # Try to parse JSON object (more lenient matching)
        try:
            # Find JSON object, support multi-line and nested
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"name"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                action = json.loads(json_match.group())
                if isinstance(action, dict) and "name" in action:
                    return action
        except:
            pass
        
        # Try to parse JSON in code blocks
        try:
            json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_block:
                action = json.loads(json_block.group(1))
                if isinstance(action, dict) and "name" in action:
                    return action
        except:
            pass
        
        # Try to parse simple command format
        # Format: bash: "command"
        bash_match = re.search(r'bash:\s*["\']([^"\']+)["\']', response)
        if bash_match:
            return {
                "name": "bash",
                "arguments": {"command": bash_match.group(1)}
            }
        
        # Try to parse grep, cat, ls, etc. commands
        for cmd in ["grep", "cat", "ls", "find", "tree"]:
            pattern = rf'{cmd}:\s*["\']([^"\']+)["\']'
            match = re.search(pattern, response)
            if match:
                return {
                    "name": cmd,
                    "arguments": {"path": match.group(1)}
                }
        
        return None
    
    def execute_action(self, action: Dict[str, Any], env: LocalEnv) -> str:
        """Execute tool call"""
        action_name = action.get("name", "")
        arguments = action.get("arguments", {})
        
        # Check if blocked
        if self._is_blocked(action_name, arguments):
            return f"Error: Command '{action_name}' is blocked by the environment."
        
        try:
            if action_name == "bash":
                command = arguments.get("command", "")
                return env.communicate(command, check="warn")
            
            elif action_name == "read_file" or action_name == "cat":
                path = arguments.get("path", "")
                return env.read_file(path)
            
            elif action_name == "grep":
                pattern = arguments.get("pattern", "")
                path = arguments.get("path", ".")
                command = f"grep -r '{pattern}' {path}"
                return env.communicate(command, check="warn")
            
            elif action_name == "ls":
                path = arguments.get("path", ".")
                return env.communicate(f"ls -la {path}", check="warn")
            
            elif action_name == "find":
                pattern = arguments.get("pattern", "*")
                path = arguments.get("path", ".")
                return env.communicate(f"find {path} -name '{pattern}'", check="warn")
            
            else:
                return f"Error: Unknown action '{action_name}'"
        
        except Exception as e:
            return f"Error executing action: {str(e)}"
    
    def _is_blocked(self, action_name: str, arguments: Dict[str, Any]) -> bool:
        """Check if action is blocked"""
        blocklist = self.config.filter.blocklist
        
        # Check complete command
        if action_name in blocklist:
            return True
        
        # Check blocked items in bash commands
        if action_name == "bash":
            command = arguments.get("command", "")
            for blocked in blocklist:
                if command.startswith(blocked):
                    return True
        
        return False

