"""Lightweight local execution environment, replaces Docker environment"""
from __future__ import annotations

import subprocess
import logging
from pathlib import Path
from typing import Optional, Union, Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LocalEnv:
    """Local execution environment, directly executes commands without Docker"""
    
    def __init__(self, repo_path: Path, working_dir: Optional[Path] = None):
        self.repo_path = Path(repo_path).resolve()
        self.working_dir = working_dir or self.repo_path
        self.env_variables = {}
        
    def communicate(
        self,
        command: str,
        timeout: int = 25,
        check: str = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        """Execute command locally
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            check: "ignore", "warn", "raise"
            error_msg: Error message
            
        Returns:
            Command output
        """
        logger.debug(f"Executing: {command}")
        try:
            env = {**self.env_variables}
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            output = result.stdout + result.stderr
            
            if check != "ignore" and result.returncode != 0:
                logger.error(f"{error_msg}:\n{output}")
                if check == "raise":
                    raise RuntimeError(f"Command {command!r} failed ({result.returncode=}): {error_msg}")
            
            return output
        except subprocess.TimeoutExpired:
            msg = f"Command {command!r} timed out after {timeout}s"
            logger.error(msg)
            if check == "raise":
                raise RuntimeError(msg)
            return f"Command timed out after {timeout}s"
    
    def read_file(self, path: Union[str, Path], encoding: Optional[str] = None, errors: Optional[str] = None) -> str:
        """Read file content
        
        Args:
            path: File path (relative to repo_path or absolute path)
            encoding: Encoding
            errors: Error handling
            
        Returns:
            File content
        """
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.repo_path / file_path
        
        if encoding:
            return file_path.read_text(encoding=encoding, errors=errors or "strict")
        return file_path.read_text(errors=errors or "strict")
    
    def write_file(self, path: Union[str, Path], content: str) -> None:
        """Write file
        
        Args:
            path: File path (relative to repo_path or absolute path)
            content: File content
        """
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.repo_path / file_path
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    
    def set_env_variables(self, env_variables: Dict[str, str]) -> None:
        """Set environment variables"""
        self.env_variables.update(env_variables)
        logger.debug(f"Set environment variables: {list(env_variables.keys())}")
    
    def execute_command(
        self,
        command: str,
        shell: bool = True,
        check: bool = False,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        """Execute command (independent process)"""
        env_vars = {**self.env_variables}
        if env:
            env_vars.update(env)
        
        subprocess.run(
            command,
            shell=shell,
            cwd=cwd or str(self.working_dir),
            env=env_vars,
            check=check,
        )

