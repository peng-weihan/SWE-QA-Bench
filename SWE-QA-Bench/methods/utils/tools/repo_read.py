import os
import subprocess
import re
import traceback
from config import Config

class RepoReadTool:
    """Code repository reading tool"""

    def __init__(self, repo_path: str):
        """
        Initialize tool

        Args:
            repo_path: Absolute path of the repository
        """
        self.repo_path = repo_path
        self.allowed_commands = Config.ALLOWED_COMMANDS

    def _validate_path(self, repo_dir: str) -> str:
        """
        Validate if repo_dir is within repo_path
        """
        full_path = os.path.join(self.repo_path, repo_dir)
        if not os.path.exists(full_path):
            raise ValueError(f"Path {repo_dir} is not in the repository")
        return full_path
        
    def _validate_command(self, read_cmd: str) -> None:
        """
        Validate if command is in whitelist

        Args:
            read_cmd: bash command

        Raises:
            ValueError: If command is not in whitelist
        """
        cmd_parts = read_cmd.strip().split()
        if not cmd_parts:
            raise ValueError("Empty command")

        base_cmd = cmd_parts[0]
        if base_cmd not in self.allowed_commands:
            raise ValueError(
                f"Command '{base_cmd}' is not allowed. Allowed commands: {self.allowed_commands}"
            )

    def _add_grep_window(self, grep_output: str, target_path: str) -> str:
        """
        Add context window to grep output

        Args:
            grep_output: Output from grep command
            target_path: Target file path

        Returns:
            Output with context window
        """
        if not grep_output.strip():
            return grep_output

        lines = grep_output.strip().split("\n")
        result_lines = []

        for line in lines:
            # Parse grep output format "filename:line_number:content"
            match = re.match(r"^([^:]+):(\d+):(.*)", line)
            if not match:
                result_lines.append(line)
                continue

            filename, line_num, content = match.groups()
            line_num = int(line_num)

            # Build complete file path
            if filename.startswith("/"):
                file_path = filename
            else:
                file_path = os.path.join(target_path, filename)

            if not os.path.exists(file_path):
                result_lines.append(line)
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_lines = f.readlines()

                # Calculate window range
                window_half = Config.GREP_WINDOW_SIZE // 2
                start_line = max(0, line_num - 1 - window_half)
                end_line = min(len(file_lines), line_num + window_half)

                result_lines.append(
                    f"\n--- Context window for {filename}:{line_num} ---"
                )
                for i in range(start_line, end_line):
                    prefix = ">>> " if i == line_num - 1 else "    "
                    result_lines.append(f"{prefix}{i + 1:4d}: {file_lines[i].rstrip()}")
                result_lines.append("--- End of context window ---\n")

            except Exception as e:
                traceback.print_exc()
                result_lines.append(f"Error reading context for {filename}: {e}")
                result_lines.append(line)

        return "\n".join(result_lines)

    def execute(self, repo_dir: str, read_cmd: str, enable_window: bool = False) -> str:
        """
        Execute read command

        Args:
            repo_dir: Path relative to working directory
            read_cmd: bash command
            enable_window: Whether to enable grep window functionality

        Returns:
            Command execution result
        """
        try:
            # Validate path and command
            target_path = self._validate_path(repo_dir)
            self._validate_command(read_cmd)
            if "grep" in read_cmd:
                read_cmd.replace("grep", "grep -Hn")  # Make grep output include filename and line number

            # Execute command
            result = subprocess.run(
                read_cmd,
                shell=True,
                cwd=target_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"

            # If it's a grep command and window is enabled, add context
            if enable_window and read_cmd.strip().startswith("grep"):
                output = self._add_grep_window(output, target_path)

            return output

        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            traceback.print_exc()
            return f"Error: {str(e)}"

def repo_read(
    repo_dir: str,
    read_cmd: str,
    enable_window: bool = False,
    repo_path: str | None = None,
) -> str:
    """
    Code repository reading tool function

    Args:
        repo_dir: Path relative to working directory
        read_cmd: bash command
        enable_window: Whether to enable grep window functionality
        repo_path: Repository path, if None use current directory

    Returns:
        Command execution result
    """
    tool = RepoReadTool(repo_path)
    return tool.execute(repo_dir, read_cmd, enable_window)
