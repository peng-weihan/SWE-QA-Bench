from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from subprocess import Popen, PIPE
import rich
import json
import threading
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import subprocess
load_dotenv()

class TestResult(BaseModel):
    question: str
    answer: str
    trajectory: str
    latency: float = 0.0  # Latency in seconds
    input_tokens: int = 0  # Number of input tokens
    output_tokens: int = 0  # Number of output tokens
    total_tokens: int = 0  # Total number of tokens

def run_cmd(cmd: str, output_format: str = "text") -> tuple[str, float, dict]:
    """Execute command and print output in real-time, format: (Thread i): output
    Returns: (output content, latency time, token info dict)"""
    import select
    import time
    
    thread_id = threading.current_thread().ident
    start_time = time.time()
    
    # Debug: Print command (without API key)
    # Mask API keys in debug output
    import re
    debug_cmd = re.sub(r'key_[a-f0-9]+', '***', cmd) if "key_" in cmd else cmd
    rich.print(f"[cyan](Thread {thread_id}) DEBUG: Executing command: {debug_cmd[:300]}...[/cyan]", flush=True)
    
    # Ensure PATH includes common locations for cursor-agent
    env = os.environ.copy()
    local_bin = os.path.expanduser("~/.local/bin")
    if local_bin not in env.get("PATH", ""):
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    output_lines = []
    stderr_lines = []
    last_output_time = time.time()
    timeout_seconds = 120  # Timeout if no new output for 120 seconds
    token_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    try:
        while True:
            # Check if stdout has readable data, non-blocking
            if select.select([process.stdout], [], [], 1.0)[0]:  # 1 second timeout check
                line = process.stdout.readline()
                if line:
                    line = line.rstrip('\n')
                    output_lines.append(line)
                    rich.print(f"[blue](Thread {thread_id})[/blue]: {line}", flush=True)
                    last_output_time = time.time()  # Update last output time
                else:
                    # readline returned empty, process may have ended
                    if process.poll() is not None:
                        break
            
            # Check if stderr has readable data (may contain token info)
            if select.select([process.stderr], [], [], 0.1)[0]:
                line = process.stderr.readline()
                if line:
                    line = line.rstrip('\n')
                    stderr_lines.append(line)
                    rich.print(f"[red](Thread {thread_id}) STDERR:[/red] {line}", flush=True)
            
            # Check if process has ended
            if process.poll() is not None:
                # Read remaining output
                remaining = process.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        if line:
                            output_lines.append(line)
                            rich.print(f"[blue](Thread {thread_id})[/blue]: {line}", flush=True)
                # Read remaining stderr
                remaining_stderr = process.stderr.read()
                if remaining_stderr:
                    for line in remaining_stderr.splitlines():
                        if line:
                            stderr_lines.append(line)
                            rich.print(f"[red](Thread {thread_id}) STDERR:[/red] {line}", flush=True)
                break
            # Check if final answer is found
            if "<end_of_answer>" in ''.join(output_lines):
                process.kill()
                process.wait()
                break
            # Check if timeout (120 seconds without new output)
            if time.time() - last_output_time > timeout_seconds:
                rich.print(f"[red](Thread {thread_id}) No output for {timeout_seconds}s - killing process[/red]", flush=True)
                process.kill()
                process.wait()  # Wait for process cleanup
                break
        
        # Wait for process to complete and ensure we read all output
        process.wait()
        return_code = process.returncode
        
        # Final read of any remaining output (in case process ended quickly)
        if not output_lines:
            remaining = process.stdout.read()
            if remaining:
                for line in remaining.splitlines():
                    if line:
                        output_lines.append(line)
                        rich.print(f"[blue](Thread {thread_id})[/blue]: {line}", flush=True)
        
        if not stderr_lines:
            remaining_stderr = process.stderr.read()
            if remaining_stderr:
                for line in remaining_stderr.splitlines():
                    if line:
                        stderr_lines.append(line)
                        rich.print(f"[red](Thread {thread_id}) STDERR:[/red] {line}", flush=True)
        
        if return_code != 0:
            rich.print(f"[yellow](Thread {thread_id}) Process exited with code {return_code}[/yellow]", flush=True)
            if stderr_lines:
                rich.print(f"[red](Thread {thread_id}) All STDERR content:[/red]", flush=True)
                for err_line in stderr_lines:
                    rich.print(f"[red](Thread {thread_id})   {err_line}[/red]", flush=True)
            else:
                rich.print(f"[red](Thread {thread_id}) No stderr captured, but process failed![/red]", flush=True)
                rich.print(f"[red](Thread {thread_id}) Command was: {debug_cmd[:200]}...[/red]", flush=True)
        
        # Try to parse token info from stderr or output
        all_output = '\n'.join(output_lines + stderr_lines)
        extracted_text = '\n'.join(output_lines)  # Used to extract text content
        full_trajectory = '\n'.join(output_lines)  # Save complete trajectory (including all output)
        
        # Debug: Print summary if output is empty or suspicious
        if not output_lines:
            rich.print(f"[yellow](Thread {thread_id}) DEBUG: No output lines captured[/yellow]", flush=True)
            if stderr_lines:
                rich.print(f"[yellow](Thread {thread_id}) DEBUG: STDERR content: {stderr_lines[:5]}[/yellow]", flush=True)
        
        if output_format in ["json", "stream-json"]:
            # If JSON format, try to parse JSON
            json_text_parts = []
            json_objects = []  # Save all JSON objects for building complete trajectory
            for line in output_lines:
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        # Save complete JSON object
                        json_objects.append(data)
                        
                        # Extract text content - handle cursor-agent's actual JSON format
                        if "result" in data:
                            json_text_parts.append(str(data["result"]))
                        elif "content" in data:
                            json_text_parts.append(str(data["content"]))
                        elif "text" in data:
                            json_text_parts.append(str(data["text"]))
                        elif "message" in data:
                            msg = data["message"]
                            if isinstance(msg, dict) and "content" in msg:
                                json_text_parts.append(str(msg["content"]))
                            elif isinstance(msg, str):
                                json_text_parts.append(msg)
                        elif "response" in data:
                            json_text_parts.append(str(data["response"]))
                        # Check if there is tool call information
                        elif "type" in data:
                            # If tool call type, save detailed information
                            if data.get("type") in ["tool_call", "function_call", "action"]:
                                json_text_parts.append(f"[Tool Call: {data.get('name', 'unknown')}] {json.dumps(data, indent=2)}")
                            elif data.get("type") == "result" and "result" in data:
                                json_text_parts.append(str(data["result"]))
                        
                        # Check if there is token info in JSON - check all possible fields
                        if "usage" in data:
                            usage = data["usage"]
                            token_info["input_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                            token_info["output_tokens"] = usage.get("completion_tokens", usage.get("output_tokens", 0))
                            token_info["total_tokens"] = usage.get("total_tokens", 0)
                        elif "tokens" in data:
                            tokens = data["tokens"]
                            if isinstance(tokens, dict):
                                token_info["input_tokens"] = tokens.get("input", tokens.get("prompt", tokens.get("prompt_tokens", 0)))
                                token_info["output_tokens"] = tokens.get("output", tokens.get("completion", tokens.get("completion_tokens", 0)))
                                token_info["total_tokens"] = tokens.get("total", tokens.get("total_tokens", 0))
                        elif "prompt_tokens" in data or "completion_tokens" in data:
                            token_info["input_tokens"] = data.get("prompt_tokens", data.get("input_tokens", 0))
                            token_info["output_tokens"] = data.get("completion_tokens", data.get("output_tokens", 0))
                            token_info["total_tokens"] = data.get("total_tokens", 0)
                        # Check nested fields
                        elif "metadata" in data and isinstance(data["metadata"], dict):
                            metadata = data["metadata"]
                            if "usage" in metadata:
                                usage = metadata["usage"]
                                token_info["input_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                                token_info["output_tokens"] = usage.get("completion_tokens", usage.get("output_tokens", 0))
                                token_info["total_tokens"] = usage.get("total_tokens", 0)
                        elif "stats" in data and isinstance(data["stats"], dict):
                            stats = data["stats"]
                            token_info["input_tokens"] = stats.get("prompt_tokens", stats.get("input_tokens", 0))
                            token_info["output_tokens"] = stats.get("completion_tokens", stats.get("output_tokens", 0))
                            token_info["total_tokens"] = stats.get("total_tokens", 0)
                except json.JSONDecodeError:
                    # If not JSON, keep original text
                    json_text_parts.append(line)
            
            # If text was extracted from JSON, use extracted text
            if json_text_parts:
                extracted_text = '\n'.join(json_text_parts)
            
            # Build complete trajectory: save formatted version of all JSON objects
            if json_objects:
                full_trajectory = json.dumps(json_objects, indent=2, ensure_ascii=False)
            else:
                full_trajectory = '\n'.join(output_lines)
        
        # If JSON parsing failed or no token info found, try to parse from text
        if token_info["total_tokens"] == 0:
            token_info = parse_token_info(all_output)
            # If still not found, try to parse from stderr (some tools may output token info to stderr)
            if token_info["total_tokens"] == 0 and stderr_lines:
                stderr_output = '\n'.join(stderr_lines)
                stderr_token_info = parse_token_info(stderr_output)
                if stderr_token_info["total_tokens"] > 0:
                    token_info = stderr_token_info
            
    except Exception as e:
        rich.print(f"[red](Thread {thread_id}) ERROR: {str(e)}[/red]", flush=True)
        try:
            process.kill()
        except:
            pass
        latency = time.time() - start_time
        return (f"Error: {str(e)}", latency, token_info)
    
    # Calculate latency
    latency = time.time() - start_time
    
    # Debug: Print summary before returning
    rich.print(f"[cyan](Thread {thread_id}) DEBUG Summary: latency={latency:.2f}s, output_lines={len(output_lines)}, stderr_lines={len(stderr_lines)}, tokens={token_info}[/cyan]", flush=True)
    
    # Return (extracted text content for extracting answer, complete trajectory, latency, token info)
    return (extracted_text, full_trajectory, latency, token_info)

def parse_token_info(trajectory: str) -> dict:
    """Parse token info from output
    Try to match common token statistics formats"""
    import re
    token_info = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    # Try to match various possible token formats
    patterns = [
        (r'total[_\s]*tokens?[:\s]+(\d+)', 'total_tokens'),
        (r'input[_\s]*tokens?[:\s]+(\d+)', 'input_tokens'),
        (r'output[_\s]*tokens?[:\s]+(\d+)', 'output_tokens'),
        (r'prompt[_\s]*tokens?[:\s]+(\d+)', 'input_tokens'),
        (r'completion[_\s]*tokens?[:\s]+(\d+)', 'output_tokens'),
        (r'"total_tokens":\s*(\d+)', 'total_tokens'),
        (r'"prompt_tokens":\s*(\d+)', 'input_tokens'),
        (r'"completion_tokens":\s*(\d+)', 'output_tokens'),
    ]
    
    for pattern, key in patterns:
        matches = re.findall(pattern, trajectory, re.IGNORECASE)
        if matches:
            try:
                token_info[key] = int(matches[-1])  # Take the last match
            except ValueError:
                pass
    
    # If total_tokens not found, try to calculate
    if token_info["total_tokens"] == 0 and (token_info["input_tokens"] > 0 or token_info["output_tokens"] > 0):
        token_info["total_tokens"] = token_info["input_tokens"] + token_info["output_tokens"]
    
    return token_info

def check_cursor_index(repo_path: str) -> tuple[bool, str]:
    """Check if Cursor has indexed the repository
    Returns: (is_indexed, status_message)"""
    cursor_projects_dir = os.path.expanduser("~/.cursor/projects")
    
    if not os.path.exists(cursor_projects_dir):
        return False, "Cursor projects directory not found (~/.cursor/projects)"
    
    # Normalize repo path to absolute path
    abs_repo_path = os.path.abspath(repo_path) if repo_path else None
    if not abs_repo_path or not os.path.exists(abs_repo_path):
        return False, f"Repository path does not exist: {repo_path}"
    
    # Cursor project directories are named based on workspace paths
    # They encode the path information in the directory name
    # We'll check by matching path components in directory names
    try:
        project_dirs = [d for d in os.listdir(cursor_projects_dir) 
                       if os.path.isdir(os.path.join(cursor_projects_dir, d))]
        
        if not project_dirs:
            return False, "No Cursor projects found"
        
        # Extract key path components for matching
        repo_name = os.path.basename(abs_repo_path.rstrip('/'))
        repo_path_parts = [p for p in abs_repo_path.split(os.sep) if p]
        # Use last 2-3 path components for matching (more reliable)
        match_parts = repo_path_parts[-2:] if len(repo_path_parts) >= 2 else repo_path_parts
        
        # Check each project directory
        best_match = None
        best_match_score = 0
        
        for project_dir in project_dirs:
            project_path = os.path.join(cursor_projects_dir, project_dir)
            repo_json_path = os.path.join(project_path, "repo.json")
            
            if not os.path.exists(repo_json_path):
                continue
            
            # Calculate match score based on path components
            score = 0
            project_dir_lower = project_dir.lower()
            
            # Check if repo name matches
            if repo_name.lower() in project_dir_lower:
                score += 2
            
            # Check if path parts match
            for part in match_parts:
                if part.lower() in project_dir_lower:
                    score += 1
            
            if score > best_match_score:
                best_match_score = score
                best_match = project_dir
        
        # If we found a good match (score >= 2), consider it indexed
        if best_match_score >= 2:
            try:
                # Verify repo.json exists and has content
                repo_json_path = os.path.join(cursor_projects_dir, best_match, "repo.json")
                with open(repo_json_path, 'r') as f:
                    repo_data = json.load(f)
                    if repo_data:
                        return True, f"Index found (project: {best_match})"
            except Exception as e:
                pass
        
        # If we found some projects but no good match
        return False, f"No matching index found (checked {len(project_dirs)} projects)"
            
    except Exception as e:
        return False, f"Error checking index: {str(e)}"

def load_processed_questions(output_file: str) -> set[str]:
    """Load set of processed questions from output file"""
    processed_questions = set()
    if os.path.exists(output_file):
        if os.path.isdir(output_file):
            rich.print(f"[red]Error: Output path {output_file} is a directory, not a file. Please remove it first.[/red]")
            return processed_questions
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "question" in data:
                                processed_questions.add(data["question"])
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            rich.print(f"[yellow]Warning: Failed to load processed questions from {output_file}: {e}[/yellow]")
    return processed_questions

def generate_trajactory(questions: List[str], repo_path: str, model: str = "auto", max_concurrency: int = os.cpu_count(), output_file: str = None, output_format: str = "json", cursor_agent_path: str = None, max_tool_calls: int = 30, use_force: bool = False) -> List[TestResult]:
    results = []
    # Get API key from environment variable or config, with fallback to empty string
    cursor_api_key = os.getenv("CURSOR_API_KEY", "")
    if not cursor_api_key:
        raise ValueError("CURSOR_API_KEY environment variable is required. Please set it before running the script.")
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    
    # Check if output_file is a directory
    if output_file and os.path.exists(output_file) and os.path.isdir(output_file):
        rich.print(f"[red]Error: Output path {output_file} is a directory, not a file. Cannot write results.[/red]")
        return results
    
    # Determine cursor-agent path: prioritize configured full path, then environment variable, then common paths
    # Check environment variable first
    env_cursor_agent_path = os.getenv("CURSOR_AGENT_PATH")
    
    if cursor_agent_path and os.path.exists(cursor_agent_path):
        cursor_agent_cmd = cursor_agent_path
    elif env_cursor_agent_path and os.path.exists(env_cursor_agent_path):
        cursor_agent_cmd = env_cursor_agent_path
        rich.print(f"[green]Using cursor-agent from CURSOR_AGENT_PATH: {cursor_agent_cmd}[/green]", flush=True)
    else:
        # Try common installation paths
        common_paths = [
            os.path.expanduser("~/.local/bin/cursor-agent"),
            os.path.expanduser("~/.local/share/cursor-agent/versions/*/cursor-agent"),
        ]
        
        cursor_agent_cmd = None
        for path_pattern in common_paths:
            if '*' in path_pattern:
                # Handle glob patterns
                import glob
                matches = glob.glob(path_pattern)
                if matches:
                    # Use the most recent version
                    cursor_agent_cmd = sorted(matches)[-1]
                    break
            elif os.path.exists(path_pattern):
                cursor_agent_cmd = path_pattern
                break
        
        if cursor_agent_cmd:
            rich.print(f"[green]Using cursor-agent at: {cursor_agent_cmd}[/green]", flush=True)
        else:
            # Final fallback: assume cursor-agent is in PATH
            cursor_agent_cmd = "cursor-agent"
            rich.print(f"[yellow]Warning: cursor-agent not found in common paths, using 'cursor-agent' from PATH[/yellow]", flush=True)
    
    # Create list of (command, question) pairs
    cmd_question_pairs = []
    prompt_template = """
You are an expert code analyst. Your task is reading and searching through code repositories and answer questions.

Task Details:  
1. Thoroughly explore and analyze the code, documentation, configuration files, tests, examples, and all relevant knowledge inside this repository related to the following question.
2. Use multiple search strategies: search by keywords, browse directory structures, read relevant files, check documentation, examine test cases, and trace code execution paths.
3. Do not use or assume external knowledge unless explicitly required. Base your answer strictly on repository content.
4. Think step by step: first understand the context, then search for relevant information, analyze findings, and synthesize a comprehensive answer.
5. You can make up to {max_tool_calls} tool calls to thoroughly explore the repository. Use them wisely to gather comprehensive information.
6. Explore different parts of the repository: main code, tests, documentation, examples, configuration files, etc.
7. IMPORTANT: You must respond in English only. Do not use Chinese or any other language.

When you have thoroughly explored the repository and gathered sufficient information to provide a comprehensive answer, provide your final answer wrapped in:

<start_of_answer>
Your answer here
<end_of_answer>


Then stop further analysis, finish the task.
Note: Provide the final answer concisely and directly, without code snippets, extra explanations or commentary. Always respond in English.

Now the code repo at {repo_path}. Thoroughly explore it and answer my question: {question}
    """
    
    # Ensure repo_path is absolute for --workspace flag
    abs_repo_path = os.path.abspath(repo_path) if repo_path and os.path.exists(repo_path) else repo_path
    
    for question in questions:
        prompt = prompt_template.format(repo_path=repo_path, question=question, max_tool_calls=max_tool_calls)
        # Use shlex.quote to properly escape the prompt for shell
        import shlex
        prompt_quoted = shlex.quote(prompt)
        # Build command with workspace and optional force flag
        workspace_flag = f"--workspace {shlex.quote(abs_repo_path)}" if abs_repo_path and os.path.exists(abs_repo_path) else ""
        force_flag = "--force" if use_force else ""
        # -p is --print flag, prompt should be passed as positional argument
        cursor_cmd = f"""{cursor_agent_cmd} --api-key={cursor_api_key} --print --model {shlex.quote(model)} --output-format {output_format} {workspace_flag} {force_flag} {prompt_quoted}""".strip()
        # Clean up multiple spaces
        cursor_cmd = ' '.join(cursor_cmd.split())
        cmd_question_pairs.append((cursor_cmd, question))
        # Debug: Print command info (without API key)
        debug_cmd = cursor_cmd.replace(cursor_api_key, "***") if cursor_api_key else cursor_cmd
        rich.print(f"[cyan]DEBUG: Command for question '{question[:50]}...': {debug_cmd[:200]}...[/cyan]", flush=True)
    
    # Use as_completed to avoid blocking
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # Submit all tasks
        future_to_question = {
            executor.submit(run_cmd, cmd, output_format): question 
            for cmd, question in cmd_question_pairs
        }
        
        # Process completed tasks
        for future in tqdm(as_completed(future_to_question), total=len(questions), desc="Processing questions"):
            question = future_to_question[future]
            try:
                extracted_text, full_trajectory, latency, token_info = future.result()
                total_latency += latency
                
                # Use token info parsed from command output
                total_input_tokens += token_info["input_tokens"]
                total_output_tokens += token_info["output_tokens"]
                total_tokens += token_info["total_tokens"]
                
                # Extract final answer (from extracted text)
                result = None
                if "<start_of_answer>" in extracted_text and "<end_of_answer>" in extracted_text:
                    final_answer = extracted_text.split("<start_of_answer>")[1].split("<end_of_answer>")[0].strip()
                    result = TestResult(
                        question=question,
                        answer=final_answer,
                        trajectory=full_trajectory,  # Save complete trajectory
                        latency=latency,
                        input_tokens=token_info["input_tokens"],
                        output_tokens=token_info["output_tokens"],
                        total_tokens=token_info["total_tokens"],
                    )
                # HINT: Sometimes there may be errors with multiple `/`, but such tags are intentionally not in XML format to bypass cursor-agent's mysterious rate limiting and review
                elif "<start_of_answer>" in extracted_text and "</end_of_answer>" in extracted_text:
                    final_answer = extracted_text.split("<start_of_answer>")[1].split("</end_of_answer>")[0].strip()
                    result = TestResult(
                        question=question,
                        answer=final_answer,
                        trajectory=full_trajectory,  # Save complete trajectory
                        latency=latency,
                        input_tokens=token_info["input_tokens"],
                        output_tokens=token_info["output_tokens"],
                        total_tokens=token_info["total_tokens"],
                    )
                elif "<start_of_answer>" in extracted_text:
                    final_answer = extracted_text.split("<start_of_answer>")[1].strip()
                    result = TestResult(
                        question=question,
                        answer=final_answer,
                        trajectory=full_trajectory,  # Save complete trajectory
                        latency=latency,
                        input_tokens=token_info["input_tokens"],
                        output_tokens=token_info["output_tokens"],
                        total_tokens=token_info["total_tokens"],
                    )
                else:
                    # Debug: Print information when no answer found
                    rich.print(f"[yellow]DEBUG: No answer found for question: {question[:100]}...[/yellow]", flush=True)
                    rich.print(f"[yellow]DEBUG: Extracted text length: {len(extracted_text)}, Trajectory length: {len(full_trajectory)}[/yellow]", flush=True)
                    if extracted_text:
                        rich.print(f"[yellow]DEBUG: First 500 chars of extracted text: {extracted_text[:500]}[/yellow]", flush=True)
                    else:
                        rich.print(f"[yellow]DEBUG: Extracted text is empty![/yellow]", flush=True)
                    if token_info["total_tokens"] == 0:
                        rich.print(f"[yellow]DEBUG: No token info found (all zeros)[/yellow]", flush=True)
                    result = TestResult(
                        question=question,
                        answer="No answer found",
                        trajectory=full_trajectory,  # Save complete trajectory
                        latency=latency,
                        input_tokens=token_info["input_tokens"],
                        output_tokens=token_info["output_tokens"],
                        total_tokens=token_info["total_tokens"],
                    )
                
                results.append(result)
                # Write to file immediately
                if output_file:
                    with open(output_file, "a") as f:
                        f.write(result.model_dump_json())
                        f.write("\n")
            except Exception as e:
                result = TestResult(
                    question=question,
                    answer=f"Error: {str(e)}",
                    trajectory="",
                    latency=0.0,
                )
                results.append(result)
                # Write to file immediately
                if output_file:
                    with open(output_file, "a") as f:
                        f.write(result.model_dump_json())
                        f.write("\n")
    
    # Print statistics
    avg_latency = total_latency / len(questions) if len(questions) > 0 else 0
    rich.print(f"\n[bold green]Summary:[/bold green] {len(results)}/{len(questions)} questions answered")
    rich.print(f"[bold cyan]Latency:[/bold cyan] Total: {total_latency:.2f}s, Average: {avg_latency:.2f}s")
    if total_tokens > 0:
        rich.print(f"[bold cyan]Token Usage:[/bold cyan] Input: {total_input_tokens:,}, Output: {total_output_tokens:,}, Total: {total_tokens:,}")
    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file (JSON format)")
    parser.add_argument("--repo-base-dir", type=str, default=None, help="Path to the repository")
    parser.add_argument("--question-dir", type=str, default=None, help="Directory containing question files")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store output files")
    parser.add_argument("--model", type=str, default=None, help="auto")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Maximum number of concurrent processes")
    parser.add_argument("--repo-filter", type=str, default=None, help="Only process specific repo (e.g., 'conan')")
    parser.add_argument("--output-format", type=str, default="json", choices=["text", "json", "stream-json"], help="Output format for cursor-agent (text, json, or stream-json)")
    parser.add_argument("--cursor-agent-path", type=str, default=None, help="Full path to cursor-agent executable (locks version)")
    parser.add_argument("--max-tool-calls", type=int, default=30, help="Maximum number of tool calls allowed for exploring repository (default: 30)")
    parser.add_argument("--force", action="store_true", help="Force allow commands unless explicitly denied (enables more thorough exploration)")
    args = parser.parse_args()
    
    # If config is a relative path, search in script directory
    if not os.path.isabs(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config
    
    # Read parameters from config file
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        rich.print(f"[green]Loaded config from {config_path}[/green]")
    else:
        rich.print(f"[yellow]Config file {config_path} not found, using defaults[/yellow]")
    
    # Command line arguments take priority over config file
    repo_base_dir = args.repo_base_dir or config.get("repo_base_dir", "repos")
    question_dir = args.question_dir or config.get("question_dir", "/path/to/questions")
    output_dir = args.output_dir or config.get("output_dir", "cursor-questions/results")
    model = args.model or config.get("model", "auto")
    max_concurrency = args.max_concurrency or config.get("max_concurrency", 4)
    repo_filter = args.repo_filter or config.get("repo_filter", None)
    output_format = args.output_format or config.get("output_format", "json")
    cursor_agent_path = args.cursor_agent_path or config.get("cursor_agent_path", None)
    max_tool_calls = args.max_tool_calls if args.max_tool_calls is not None else config.get("max_tool_calls", 30)
    use_force = args.force or config.get("force", False)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    repos: list[str] = []
    questions: list[list[str]] = []
    ground_truths: list[list[str]] = []
    for question_file in os.listdir(question_dir):
        question_file_path = os.path.join(question_dir, question_file)
        # Skip directories
        if os.path.isdir(question_file_path):
            continue
        # Only process .jsonl files
        if not question_file.endswith(".jsonl"):
            continue
        # If repository filter specified, only process matching files
        if repo_filter and not question_file.startswith(f"{repo_filter}.jsonl"):
            continue
        with open(question_file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]  # Filter out empty lines
            question_list = []
            ground_truth_list = []
            invalid_count = 0
            for line in lines:
                try:
                    data = json.loads(line)
                    if "question" not in data:
                        invalid_count += 1
                        continue
                    question_list.append(data["question"])
                    ground_truth_list.append(data.get("ground_truth", ""))
                except json.JSONDecodeError as e:
                    invalid_count += 1
                    continue
            if invalid_count > 0:
                rich.print(f"[yellow]Warning: Skipped {invalid_count} invalid line(s) in {question_file}[/yellow]")
            if len(question_list) == 0:
                rich.print(f"[yellow]Warning: No valid questions found in {question_file}, skipping...[/yellow]")
                continue
            questions.append(question_list)
            ground_truths.append(ground_truth_list)
            repos.append(
                 question_file.split(".")[0]
            )
    # Overall statistics
    all_total_latency = 0.0
    all_total_input_tokens = 0
    all_total_output_tokens = 0
    all_total_tokens = 0
    all_total_questions = 0
    
    for repo, question_list, ground_truth_list in tqdm(zip(repos, questions, ground_truths), desc="Processing repos"):
        # Build complete repository path
        if os.path.isdir(repo_base_dir):
            # repo_base_dir is base directory, combine with repo name
            repo_path = os.path.join(repo_base_dir, repo)
            # If combined path doesn't exist, try using repo_base_dir directly (might be full path)
            if not os.path.exists(repo_path):
                repo_path = repo_base_dir if os.path.exists(repo_base_dir) else repo
        else:
            # repo_base_dir is not a directory, might be full path or doesn't exist
            repo_path = repo_base_dir if os.path.exists(repo_base_dir) else repo
        
        rich.print(f"[cyan]Processing repo: {repo_path}[/cyan]")
        
        # Check Cursor index status
        is_indexed, index_status = check_cursor_index(repo_path)
        if is_indexed:
            rich.print(f"[green]✓ Cursor index found: {index_status}[/green]")
        else:
            rich.print(f"[yellow]⚠ Cursor index not found: {index_status}[/yellow]")
            rich.print(f"[yellow]  Tip: Open this repository in Cursor editor to build index for better results[/yellow]")
        
        output_file_path = os.path.join(output_dir, f"{repo}.jsonl")
        
        # Check if output path is a directory (should be a file)
        if os.path.exists(output_file_path) and os.path.isdir(output_file_path):
            rich.print(f"[red]Error: Output path {output_file_path} is a directory, not a file. Please remove it first.[/red]")
            rich.print(f"[yellow]You can remove it with: rm -rf {output_file_path}[/yellow]")
            continue
        
        # Load processed questions
        processed_questions = load_processed_questions(output_file_path)
        if processed_questions:
            rich.print(f"[yellow]Found {len(processed_questions)} already processed questions, filtering...[/yellow]")
        
        # Filter out processed questions while maintaining correspondence between questions and ground_truth
        filtered_questions = []
        filtered_ground_truths = []
        skipped_count = 0
        for q, gt in zip(question_list, ground_truth_list):
            if q not in processed_questions:
                filtered_questions.append(q)
                filtered_ground_truths.append(gt)
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            rich.print(f"[green]Skipped {skipped_count} already processed questions, {len(filtered_questions)} remaining[/green]")
        
        if len(filtered_questions) == 0:
            rich.print(f"[yellow]All questions for {repo} have been processed, skipping...[/yellow]")
            all_total_questions += len(question_list)
            continue
        
        test_results = generate_trajactory(filtered_questions, repo_path, model=model, max_concurrency=max_concurrency, output_file=output_file_path, output_format=output_format, cursor_agent_path=cursor_agent_path, max_tool_calls=max_tool_calls, use_force=use_force)
        
        # Accumulate statistics (including newly processed and skipped)
        for r in test_results:
            all_total_latency += r.latency
            all_total_input_tokens += r.input_tokens
            all_total_output_tokens += r.output_tokens
            all_total_tokens += r.total_tokens
        all_total_questions += len(question_list)  # Count all questions, including skipped
        
        rich.print(f"Results saved to {output_file_path}")
    
    # Print overall statistics
    if all_total_questions > 0:
        avg_latency = all_total_latency / all_total_questions
        rich.print(f"\n[bold yellow]=== Overall Statistics ===[/bold yellow]")
        rich.print(f"[bold]Total Questions:[/bold] {all_total_questions}")
        rich.print(f"[bold]Total Latency:[/bold] {all_total_latency:.2f}s")
        rich.print(f"[bold]Average Latency:[/bold] {avg_latency:.2f}s per question")
        if all_total_tokens > 0:
            rich.print(f"[bold]Total Token Usage:[/bold] Input: {all_total_input_tokens:,}, Output: {all_total_output_tokens:,}, Total: {all_total_tokens:,}")
    