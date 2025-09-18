import os
import re
import traceback
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage
import rich
from typing_extensions import TypedDict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import box
import time
import json
from langchain_core.prompts import ChatPromptTemplate

from config import Config
from history import ConversationHistory
from tools.repo_rag import repo_search_rag
from tools.repo_read import repo_read


class AgentState(TypedDict):
    """Agent state definition"""

    trajectory: List[Dict[str, Any]]
    question: str
    repo_path: str
    current_step: int
    final_answer: str
    tool_calls: List[Dict[str, Any]]
    history_manager: ConversationHistory

class SWEQAAgent:
    """SWE Code Repository QA Agent"""

    def __init__(self, repo_path: str):
        """Initialize Agent"""
        Config.validate()

        # Initialize Rich Console
        self.console = Console()
        self.repo_path = repo_path
        # Define tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "repo_read",
                    "description": "Read files and directories in the repository using bash commands like tree, ls, cat, grep",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_dir": {
                                "type": "string",
                                "description": "A relative path from {repo_path}, e.g. 'src/' or 'docs/', the command will be executed under the path {repo_path}/<repo_dir>",
                            },
                            "read_cmd": {
                                "type": "string",
                                "description": "Bash command to execute: 'tree' (list directory structure), 'ls' (list files), 'cat filename' (read file), 'grep pattern' (search pattern)",
                            },
                            "enable_window": {
                                "type": "boolean",
                                "description": "For grep commands, whether to include 100 lines of context around matches",
                                "default": False,
                            },
                        },
                        "required": ["repo_dir", "read_cmd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "repo_search_rag",
                    "description": "Search through the repository using RAG to find relevant code sections",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant code sections",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        # Initialize LLM with tools
        self.llm = ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
        ).bind_tools(self.tools, parallel_tool_calls=False)

        # self.llm = ChatAnthropic(
        #     api_key=Config.ANTHROPIC_API_KEY,
        #     base_url=Config.ANTHROPIC_BASE_URL,
        #     model=Config.ANTHROPIC_MODEL,
        #     temperature=Config.ANTHROPIC_TEMPERATURE,
        # ).bind_tools(self.tools, parallel_tool_calls=False)

        # Load prompt template
        self.prompt_template = self._load_prompt_template()

        # Create state graph
        self.graph = self._create_graph()
        self.token_usages = []

    def _log_step_start(self, step: int):
        """Log step start"""
        panel = Panel(
            f"🚀 Step {step + 1} Starting",
            title="[bold blue]Agent Step[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def _log_llm_call(self, step: int, prompt_length: int):
        """Log LLM call"""
        table = Table(title=f"🤖 LLM Call - Step {step + 1}", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model", Config.OPENAI_MODEL)
        table.add_row("Prompt Length", f"{prompt_length:,} chars")
        table.add_row("Temperature", "0.5")

        self.console.print(table)

    def _log_llm_response(self, response_content: str, token_usage: dict = None):
        """Log LLM response"""
        # Token usage table
        if token_usage:
            usage_table = Table(title="📊 Token Usage", box=box.SIMPLE)
            usage_table.add_column("Type", style="cyan")
            usage_table.add_column("Count", style="green")

            usage_table.add_row(
                "Input Tokens", str(token_usage.get("prompt_tokens", "N/A"))
            )
            usage_table.add_row(
                "Output Tokens", str(token_usage.get("completion_tokens", "N/A"))
            )
            usage_table.add_row(
                "Total Tokens", str(token_usage.get("total_tokens", "N/A"))
            )

            self.console.print(usage_table)
            self.token_usages.append(token_usage)

        # Response content preview
        preview_content = (
            response_content[:500] + "..."
            if len(response_content) > 500
            else response_content
        )
        response_panel = Panel(
            Syntax(
                preview_content, "markdown", theme="github-dark", line_numbers=False
            ),
            title="[bold green]🤖 LLM Response[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )
        self.console.print(response_panel)

    def _log_tool_call(self, tool_name: str, args: dict):
        """Log tool call"""
        tool_table = Table(title=f"🔧 Tool Call: {tool_name}", box=box.ROUNDED)
        tool_table.add_column("Parameter", style="cyan")
        tool_table.add_column("Value", style="yellow")

        for key, value in args.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:100] + "..."
            tool_table.add_row(key, str_value)

        self.console.print(tool_table)

    def _log_tool_result(self, tool_name: str, result: str, execution_time: float):
        """Log tool execution result"""
        # Result preview
        preview_result = result[:300] + "..." if len(result) > 300 else result

        result_panel = Panel(
            f"[bold]Execution Time:[/bold] {execution_time:.2f}s\n\n"
            f"[bold]Result Preview:[/bold]\n{preview_result}",
            title=f"[bold cyan]🔧 Tool Result: {tool_name}[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
        self.console.print(result_panel)

    def _log_final_answer(self, answer: str):
        """Log final answer"""
        answer_panel = Panel(
            Markdown(answer),
            title="[bold green]✅ Final Answer[/bold green]",
            border_style="green",
            box=box.DOUBLE,
        )
        self.console.print(answer_panel)

    def _log_force_answer_generation(self, step: int):
        """Log force answer generation"""
        force_panel = Panel(
            f"[bold]Step {step + 1}:[/bold] Maximum iterations reached. Forcing answer generation based on collected information.",
            title="[bold yellow]⚠️ Force Answer Generation[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        )
        self.console.print(force_panel)

    def _log_step_summary(self, step: int, tool_calls_count: int):
        """Log step summary"""
        summary_table = Table(title=f"📋 Step {step + 1} Summary", box=box.SIMPLE)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")

        summary_table.add_row("Tool Calls", str(tool_calls_count))
        summary_table.add_row("Status", "✅ Completed")

        self.console.print(summary_table)
        self.console.print()  # Add empty line separator

# 2. Search strategically: Use repo_search to find relevant code sections
# 2. repo_search: Search through the repository using semantic search
    def _load_prompt_template(self) -> ChatPromptTemplate:
        """Load prompt template"""
        sys_prompt_path = os.path.join(Config.PROMPTS_DIR, "react_prompt.txt")
        try:
            with open(sys_prompt_path, "r", encoding="utf-8") as f:
                sys_prompt = f.read()
        except FileNotFoundError:
            sys_prompt = """You are an expert code analyst. You have access to tools that allow you to read and search through code repositories.

                Your task is to analyze the given code repository and answer the user's question efficiently and accurately. 
                Prioritize speed and efficiency - aim to answer within 5 steps, but you can use more steps if absolutely necessary for complex questions.
                However, avoid excessive iterations beyond what's needed.

                Available tools:
                1. repo_read: Read files and directories using bash commands (tree, ls, cat, grep)
                2. repo_search_rag: Search through the repository using RAG to find relevant code sections

                EFFICIENCY GUIDELINES:
                1. Be decisive: If you can answer with basic repository exploration, do so immediately
                2. Use RAG search first: repo_search_rag is often the fastest way to find relevant information
                3. Limit exploration: Use "tree" or "ls" only if absolutely necessary to understand structure
                4. Read selectively: Only read specific files that are directly relevant to the question
                5. Avoid over-analysis: Stop when you have sufficient information, don't seek perfection
                6. Prefer breadth over depth: Get a good overview rather than deep analysis of every detail

                When you have sufficient information to provide a reasonable answer (even if not perfect), provide your final answer wrapped in:
                <final_answer>
                Your answer here
                </final_answer>

                Note: Provide the final answer concisely and directly, without code snippets, extra explanations or commentary. 

                Now the code repo at {repo_path}. Gather info from it, and answer my question {question}.
                """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_prompt),
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # Specifically for handling message lists
            ]
        )
        return prompt

    def _create_graph(self) -> StateGraph:
        """Create state graph"""

        def agent_node(state: AgentState) -> AgentState:
            """Agent reasoning node"""
            question = state["question"]
            repo_path = state["repo_path"]
            current_step = state.get("current_step", 0)
            history_manager = state["history_manager"]

            # Log step start
            self._log_step_start(current_step)

            # Check if maximum iterations reached
            if current_step >= Config.MAX_ITERATIONS:
                # Force the model to provide an answer based on collected information
                force_answer_prompt = f"""
                    Based on the Chat history and tool results and any other information you have collected so far, you MUST provide a final answer to the question.
                    Do not say you need more information or cannot answer. Use what you have learned to provide the best possible answer.
                    DO NOT CALL ANY TOOLS. ONLY OUTPUT THE FINAL ANSWER IN <final_answer> FORMAT.

                    Question: {question}

                    Repository path: {repo_path}

                    Chat history and tool results:
                    {history_manager.flatten()}

                    IMPORTANT: You must provide a comprehensive answer based on the information you have collected, even if it's not complete. 

                    Please provide your final answer in the format:
                    <final_answer>
                    Your comprehensive answer here based on the collected information
                    </final_answer>

                    Note: Provide the final answer concisely and directly, without code snippets, extra explanations or commentary. 
                    """

                # Log force answer generation
                self._log_force_answer_generation(current_step)
                
                # Call LLM to force answer generation
                try:
                    # Use correct message format to call LLM
                    force_messages = [
                        SystemMessage(content="You are an expert code analyst. You must provide a final answer based on the information you have collected."),
                        HumanMessage(content=force_answer_prompt)
                    ]
                    force_response = self.llm.invoke(force_messages)
                    
                    # Log force response debug information
                    rich.print(f"[yellow]Force Response Content:[/yellow] {force_response.content}")
                    
                    # Extract final answer
                    final_answer_match = re.search(
                        r"<final_answer>(.*?)</final_answer>", force_response.content, re.DOTALL
                    )
                    if final_answer_match:
                        final_answer = final_answer_match.group(1).strip()
                        rich.print(f"[green]Extracted answer from tags:[/green] {final_answer[:100]}...")
                    else:
                        # If no tags found, use entire response as answer
                        final_answer = force_response.content.strip()
                        rich.print(f"[yellow]Using full response as answer:[/yellow] {final_answer[:100]}...")
                    
                    if not final_answer:
                        final_answer = "Unable to generate answer due to insufficient information collected."
                        rich.print("[red]Warning: Generated empty answer, using fallback message[/red]")
                    
                except Exception as e:
                    rich.print(f"[red]Error during force answer generation: {e}[/red]")
                    final_answer = f"Error generating answer: {str(e)}"
                
                state["final_answer"] = final_answer
                self._log_final_answer(final_answer)
                return state

            # Build prompt
            prompt = self.prompt_template.format(
                question=question,
                repo_path=repo_path,
                chat_history=history_manager.flatten(),
            )
            step_messages = []

            # Log LLM call
            self._log_llm_call(current_step, len(prompt))

            # Call LLM
            response = self.llm.invoke(prompt)
            print("full response: ", response)
            step_messages.append(response)
            # Get token usage
            token_usage = None
            if (
                hasattr(response, "response_metadata")
                and "token_usage" in response.response_metadata
            ):
                token_usage = response.response_metadata["token_usage"]

            # Log LLM response
            self._log_llm_response(response.content, token_usage)

            rich.print(f"LLM Response: {response.content}")

            # Check if contains final answer
            final_answer_match = re.search(
                r"<final_answer>(.*?)</final_answer>", response.content, re.DOTALL
            )
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                state["final_answer"] = final_answer
                self._log_final_answer(final_answer)
                return state

            # Check if there are tool calls
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    # Log tool call
                    self._log_tool_call(tool_call["name"], tool_call["args"])

                    # Execute tool
                    tool_start_time = time.time()
                    result = self._execute_tool_with_name(
                        tool_call["name"], tool_call["args"], repo_path
                    )
                    tool_execution_time = time.time() - tool_start_time

                    tool_calls.append(
                        {
                            "tool": tool_call["name"],
                            "args": tool_call["args"],
                            "result": result,
                        }
                    )

                    # Log tool result
                    self._log_tool_result(
                        tool_call["name"], result, tool_execution_time
                    )

                    # Add tool result to messages
                    tool_message = ToolMessage(result, tool_call_id=tool_call["id"])
                    step_messages.append(tool_message)

            # Log step summary
            self._log_step_summary(current_step, len(tool_calls))

            history_manager.add_interaction(step_messages)
            state["trajectory"].append(
                {
                    "step": current_step,
                    "prompt": prompt,
                    "response": response.content,
                    "tool_calls": tool_calls,
                }
            )
            state["current_step"] = current_step + 1
            state["history_manager"] = history_manager
            state["tool_calls"] = state.get("tool_calls", []) + tool_calls
            return state

        def should_continue(state: AgentState) -> str:
            """Decide whether to continue iteration"""
            if state.get("final_answer"):
                return "end"
            if state.get("current_step", 0) > Config.MAX_ITERATIONS:
                return "end"
            return "continue"

        # Create graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"continue": "agent", "end": END}
        )

        return workflow.compile()

    def _execute_tool_with_name(
        self, tool_name: str, args: Dict[str, Any], repo_path: str
        ) -> str:
        """Execute tool call (using tool name and parameters)"""
        try:
            if tool_name == "repo_read":
                return repo_read(
                    repo_dir=args["repo_dir"],
                    read_cmd=args["read_cmd"],
                    enable_window=args.get("enable_window", False),
                    repo_path=repo_path,
                )
            elif tool_name == "repo_search_rag":
                results = repo_search_rag(
                    query=args["query"],
                    repo_name = os.path.basename(repo_path) 
                )
                return f"{results}"
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            traceback.print_exc()
            return f"Tool execution error: {str(e)}"

    def query(self, question: str, repo_path: str) -> Dict[str, Any]:
        """
        Process single query with token limit retry mechanism

        Args:
            question: User question
            repo_path: Code repository path

        Returns:
            Dictionary containing answer and trajectory
        """
        # Log query start
        start_panel = Panel(
            f"[bold]Question:[/bold] {question}\n"
            f"[bold]Repository:[/bold] {repo_path}\n"
            f"[bold]Max Iterations:[/bold] {Config.MAX_ITERATIONS}",
            title="[bold blue]🤔 New Query Started[/bold blue]",
            border_style="blue",
            box=box.DOUBLE,
        )
        self.console.print(start_panel)
        self.console.print()

        # Retry mechanism: start from default history size, gradually reduce
        history_sizes = [Config.HISTORY_WINDOW, 4 ,3 ,2, 1]
        last_error = None

        for attempt, history_size in enumerate(history_sizes):
            try:
                # Prepare initial state
                initial_state = AgentState(
                    question=question,
                    repo_path=repo_path,
                    current_step=0,
                    final_answer="",
                    tool_calls=[],
                    history_manager=ConversationHistory(history_size),
                    trajectory=[],
                )

                # Log retry information
                if attempt > 0:
                    retry_panel = Panel(
                        f"[bold]Retry Attempt {attempt + 1}[/bold]\n"
                        f"[bold]History Size:[/bold] {history_size} (reduced from {Config.HISTORY_WINDOW})",
                        title="[bold yellow]🔄 Retry with Reduced History[/bold yellow]",
                        border_style="yellow",
                        box=box.ROUNDED,
                    )
                    self.console.print(retry_panel)

                # Execute reasoning
                start_time = time.time()
                final_state = self.graph.invoke(initial_state)
                total_time = time.time() - start_time

                # Extract results
                answer = final_state.get("final_answer", "No answer found")
                tool_calls = final_state.get("tool_calls", [])
                current_step = final_state.get("current_step", 0)
                max_iterations = Config.MAX_ITERATIONS

                # Build trajectory
                trajectory = final_state["trajectory"]

                # Determine stop reason
                if answer != "No answer found":
                    if current_step >= max_iterations:
                        stop_reason = f"Answer generated after reaching max iterations ({max_iterations})"
                        status = "✅ Success (Forced)"
                    else:
                        stop_reason = "Found answer"
                        status = "✅ Success"
                else:
                    stop_reason = "Unknown"
                    status = "❓ Unknown"

                # Log query completion
                completion_table = Table(title="🎉 Query Completed", box=box.ROUNDED)
                completion_table.add_column("Metric", style="cyan")
                completion_table.add_column("Value", style="green")

                completion_table.add_row("Total Time", f"{total_time:.2f}s")
                completion_table.add_row("Total Steps", f"{current_step}/{max_iterations}")
                completion_table.add_row("Total Tool Calls", str(len(tool_calls)))
                completion_table.add_row("Stop Reason", stop_reason)
                completion_table.add_row("Status", status)
                completion_table.add_row("History Size Used", str(history_size))
                if attempt > 0:
                    completion_table.add_row("Retry Attempts", str(attempt))
                
                input_tokens = sum(item["prompt_tokens"] for item in self.token_usages)
                output_tokens = sum(item["completion_tokens"] for item in self.token_usages)
                completion_table.add_row("Input tokens", str(input_tokens))
                completion_table.add_row("Output tokens", str(output_tokens))

                self.console.print(completion_table)
                self.console.print()
                
                return {
                    "query": question,
                    "code_base_dir": repo_path,
                    "answer": answer,
                    "status": status,
                    "stop_reason": stop_reason,
                    "steps_completed": current_step,
                    "max_iterations": max_iterations,
                    "trajectory": trajectory,
                    "history_size_used": history_size,
                    "retry_attempts": attempt,
                }

            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's a token-related error
                is_token_error = any(keyword in error_msg for keyword in [
                    "context length", "max tokens", "token limit", "too many tokens",
                    "input too long", "context window", "maximum context length"
                ])
                
                last_error = e
                
                if is_token_error and attempt < len(history_sizes) - 1:
                    # Log token error and prepare retry
                    error_panel = Panel(
                        f"[bold]Token Limit Error:[/bold] {str(e)[:200]}...\n"
                        f"[bold]Retrying with reduced history size:[/bold] {history_sizes[attempt + 1]}",
                        title="[bold red]⚠️ Token Limit Exceeded[/bold red]",
                        border_style="red",
                        box=box.ROUNDED,
                    )
                    self.console.print(error_panel)
                    continue
                else:
                    # If not a token error, or all history sizes have been retried, throw exception
                    if is_token_error:
                        # Log final failure
                        final_error_panel = Panel(
                            f"[bold]Final Token Limit Error:[/bold] {str(e)[:200]}...\n"
                            f"[bold]All retry attempts exhausted.[/bold]",
                            title="[bold red]❌ Query Failed - Token Limit[/bold red]",
                            border_style="red",
                            box=box.DOUBLE,
                        )
                        self.console.print(final_error_panel)
                    else:
                        # Other types of errors
                        other_error_panel = Panel(
                            f"[bold]Error:[/bold] {str(e)[:200]}...",
                            title="[bold red]❌ Query Failed - Other Error[/bold red]",
                            border_style="red",
                            box=box.DOUBLE,
                        )
                        self.console.print(other_error_panel)
                    
                    # Return error result
                    return {
                        "query": question,
                        "code_base_dir": repo_path,
                        "answer": f"Error: {str(e)}",
                        "status": "❌ Failed",
                        "stop_reason": f"Error after {attempt + 1} attempts",
                        "steps_completed": 0,
                        "max_iterations": Config.MAX_ITERATIONS,
                        "trajectory": [],
                        "history_size_used": history_size,
                        "retry_attempts": attempt,
                        "error": str(e),
                    }
