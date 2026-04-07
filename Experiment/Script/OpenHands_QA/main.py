import os
import json
import time
from datetime import datetime
from openai import OpenAI

from openhands.sdk import LLM, Conversation
from openhands.tools.preset.default import get_default_agent
from openhands.sdk.event import MessageEvent, ActionEvent, ObservationEvent

# Configuration
LLM_CONFIG = {
    "model": "",
    "api_key": "",
    "base_url": "",
    "usage_id": "agent"  # Use usage_id to replace deprecated service_id
}

# Repository configuration: process reflex, streamlink, conan in order
REPOS_CONFIG = [
    {
        "name": "reflex",
        "workspace": "/path/to/repos/reflex",
        "input_file": "/path/to/questions/reflex.jsonl"
    },
    {
        "name": "streamlink",
        "workspace": "/path/to/repos/streamlink",
        "input_file": "/path/to/questions/streamlink.jsonl"
    },
    {
        "name": "conan",
        "workspace": "/path/to/repos/conan",
        "input_file": "/path/to/questions/conan.jsonl"
    }
]

OUTPUT_DIR = "/path/to/answer/OpenHands"
MAX_ITERATION_PER_RUN = 10

# Extract repository name from input file path
def get_repo_name_from_path(file_path):
    """Extract repository name from file path"""
    basename = os.path.basename(file_path)
    # Remove .jsonl extension
    if basename.endswith('.jsonl'):
        return basename[:-6]
    return basename

def load_questions_from_jsonl(file_path):
    """Load questions from jsonl file"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                question = data.get('question', '')
                if question:
                    questions.append(data)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
    return questions

def load_answered_questions(output_file):
    """Load answered questions from output file"""
    answered_questions = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        question = data.get('question', '')
                        if question:
                            answered_questions.add(question)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading answered questions: {e}")
    return answered_questions

def get_message_history(state):
    """Extract complete message history from conversation state"""
    messages = []
    events = list(state.events)
    
    print(f"[DEBUG] Total events: {len(events)}")
    
    for idx, event in enumerate(events):
        event_type = type(event).__name__
        
        # Handle MessageEvent
        if isinstance(event, MessageEvent):
            source = getattr(event, 'source', 'unknown')
            
            # Extract message content
            content = ""
            if hasattr(event, 'extended_content') and event.extended_content:
                text_parts = []
                for item in event.extended_content:
                    if hasattr(item, 'text'):
                        text_parts.append(str(item.text))
                    elif isinstance(item, str):
                        text_parts.append(item)
                if text_parts:
                    content = "\n".join(text_parts)
            
            if not content and hasattr(event, 'llm_message') and event.llm_message:
                msg = event.llm_message
                if hasattr(msg, 'content') and msg.content:
                    if isinstance(msg.content, list):
                        text_parts = []
                        for item in msg.content:
                            if hasattr(item, 'text'):
                                text_parts.append(str(item.text))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        if text_parts:
                            content = "\n".join(text_parts)
                    elif isinstance(msg.content, str):
                        content = msg.content
            
            if content:
                messages.append({
                    'role': source,  # 'user', 'agent', 'tool'
                    'content': content,
                    'timestamp': getattr(event, 'timestamp', None),
                    'event_type': event_type
                })
                print(f"[DEBUG Event {idx}] MessageEvent, source={source}, content length={len(content)}")
        
        # Handle ActionEvent (actions executed by agent)
        elif isinstance(event, ActionEvent):
            action_info = ""
            if hasattr(event, 'action') and event.action:
                action = event.action
                action_name = getattr(action, 'name', 'unknown')
                action_info = f"Action: {action_name}"
                
                # Try to get action parameters
                if hasattr(action, 'model_dump'):
                    try:
                        action_dict = action.model_dump()
                        action_info += f"\nParameters: {json.dumps(action_dict, ensure_ascii=False, indent=2)}"
                    except:
                        action_info += f"\nAction object: {str(action)}"
            
            if action_info:
                messages.append({
                    'role': 'agent',
                    'content': action_info,
                    'timestamp': getattr(event, 'timestamp', None),
                    'event_type': event_type
                })
                print(f"[DEBUG Event {idx}] ActionEvent, content length={len(action_info)}")
        
        # Handle ObservationEvent (tool execution results)
        elif isinstance(event, ObservationEvent):
            observation_info = ""
            if hasattr(event, 'observation') and event.observation:
                observation = event.observation
                # Try to get observation result
                if hasattr(observation, 'message'):
                    observation_info = f"Observation: {observation.message}"
                elif hasattr(observation, 'content'):
                    observation_info = f"Observation: {observation.content}"
                elif hasattr(observation, 'model_dump'):
                    try:
                        obs_dict = observation.model_dump()
                        observation_info = f"Observation: {json.dumps(obs_dict, ensure_ascii=False, indent=2)}"
                    except:
                        observation_info = f"Observation: {str(observation)}"
                else:
                    observation_info = f"Observation: {str(observation)}"
            
            if observation_info:
                messages.append({
                    'role': 'tool',
                    'content': observation_info,
                    'timestamp': getattr(event, 'timestamp', None),
                    'event_type': event_type
                })
                print(f"[DEBUG Event {idx}] ObservationEvent, content length={len(observation_info)}")
        else:
            # Other event types, also try to extract information
            print(f"[DEBUG Event {idx}] Other event type: {event_type}")
    
    print(f"[DEBUG] Extracted {len(messages)} messages")
    return messages

def generate_answer_from_history(llm_config, question, message_history):
    """Generate answer using LLM based on message_history"""
    try:
        client = OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"]
        )
        
        # Format message_history as text to avoid tool role issues
        # Merge all messages into a single text instead of using tool role
        conversation_text = []
        
        for msg in message_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if content:
                # Format messages from different roles as text
                if role == 'user':
                    conversation_text.append(f"User: {content}")
                elif role == 'agent':
                    conversation_text.append(f"Assistant: {content}")
                elif role == 'tool':
                    conversation_text.append(f"Tool Output: {content}")
                else:
                    conversation_text.append(f"{role}: {content}")
        
        # Build complete prompt
        full_conversation = "\n\n".join(conversation_text)
        
        system_prompt = """You are a code repository question answering assistant. 
Based on the conversation history provided, synthesize all the information gathered and provide a comprehensive answer to the user's question.
Even if the information is incomplete, provide the best answer you can based on what was discovered during the exploration."""
        
        user_prompt = f"""Original Question: {question}

Conversation History:
{full_conversation}

Based on the conversation history above, please provide a comprehensive answer to the original question."""
        
        formatted_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"[DEBUG] Calling LLM to generate answer, conversation history length: {len(full_conversation)} characters")
        
        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=formatted_messages,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Return answer and token usage (return prompt and completion separately)
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        
        print(f"[DEBUG] LLM generated answer successfully, length: {len(answer)} characters")
        print(f"[DEBUG] Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        
        return answer, (prompt_tokens, completion_tokens)
    except Exception as e:
        print(f"[DEBUG] Error generating answer: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def process_single_question(qa_data, workspace):
    """Process a single question"""
    question = qa_data.get('question', '')
    if not question:
        return None
    
    # Create independent agent and conversation for each task
    llm = LLM(**LLM_CONFIG)
    agent = get_default_agent(llm=llm, cli_mode=True)
    
    # Variable to store answer
    answer_data = {
        "question": question,
        "answer": "",
        "timestamp": datetime.now().isoformat(),
        "time_cost": 0.0,
        "token_cost": 0,  # Total token count (backward compatibility)
        "prompt_tokens": 0,  # input tokens
        "completion_tokens": 0  # output tokens
    }
    
    # Callback function to capture answer
    def on_event(event):
        nonlocal answer_data
        if isinstance(event, MessageEvent):
            if hasattr(event, 'source') and event.source == 'agent':
                if hasattr(event, 'extended_content') and event.extended_content:
                    text_content = []
                    for item in event.extended_content:
                        if hasattr(item, 'text'):
                            text_content.append(item.text)
                        elif isinstance(item, str):
                            text_content.append(item)
                    if text_content:
                        answer_data["answer"] = "\n".join(text_content)
                elif hasattr(event, 'llm_message') and event.llm_message:
                    msg = event.llm_message
                    if hasattr(msg, 'content') and msg.content:
                        if isinstance(msg.content, list):
                            text_content = []
                            for item in msg.content:
                                if hasattr(item, 'text'):
                                    text_content.append(item.text)
                                elif isinstance(item, str):
                                    text_content.append(item)
                            if text_content:
                                answer_data["answer"] = "\n".join(text_content)
                        elif isinstance(msg.content, str):
                            answer_data["answer"] = msg.content
    
    try:
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=MAX_ITERATION_PER_RUN,
            callbacks=[on_event]
        )
        
        # Record start time
        start_time = time.time()
        
        # Add exploration prompt
        enhanced_question = f"""Please first explore the codebase structure to find the relevant files.
Use the terminal tool to search for files related to the question.
Then answer: {question}"""
        
        conversation.send_message(enhanced_question)
        conversation.run()
        
        # Calculate time cost
        end_time = time.time()
        answer_data["time_cost"] = round(end_time - start_time, 2)
        
        # Get final answer and token usage from conversation state
        state = conversation.state
        
        # Get complete message history (after running 10 iterations)
        message_history = get_message_history(state)
        print(f"[DEBUG] Retrieved {len(message_history)} message history items")
        
        # Get token usage (separately count input and output)
        if hasattr(state, 'stats') and state.stats:
            stats = state.stats
            if hasattr(stats, 'usage_to_metrics'):
                total_prompt_tokens = 0
                total_completion_tokens = 0
                for usage_id, metrics in stats.usage_to_metrics.items():
                    if hasattr(metrics, 'accumulated_token_usage') and metrics.accumulated_token_usage:
                        token_usage = metrics.accumulated_token_usage
                        if hasattr(token_usage, 'prompt_tokens') and hasattr(token_usage, 'completion_tokens'):
                            total_prompt_tokens += token_usage.prompt_tokens
                            total_completion_tokens += token_usage.completion_tokens
                    if hasattr(metrics, 'token_usages') and metrics.token_usages:
                        for token_usage in metrics.token_usages:
                            if hasattr(token_usage, 'prompt_tokens') and hasattr(token_usage, 'completion_tokens'):
                                total_prompt_tokens += token_usage.prompt_tokens
                                total_completion_tokens += token_usage.completion_tokens
                answer_data["prompt_tokens"] = total_prompt_tokens
                answer_data["completion_tokens"] = total_completion_tokens
                answer_data["token_cost"] = total_prompt_tokens + total_completion_tokens  # Total token count (backward compatibility)
        
        # If answer not yet obtained, search in events
        if not answer_data["answer"]:
            events = list(state.events)
            for event in reversed(events):
                if isinstance(event, MessageEvent) and hasattr(event, 'source') and event.source == 'agent':
                    if hasattr(event, 'extended_content') and event.extended_content:
                        text_content = []
                        for item in event.extended_content:
                            if hasattr(item, 'text'):
                                text_content.append(item.text)
                            elif isinstance(item, str):
                                text_content.append(item)
                        if text_content:
                            answer_data["answer"] = "\n".join(text_content)
                            break
                    elif hasattr(event, 'llm_message') and event.llm_message:
                        msg = event.llm_message
                        if hasattr(msg, 'content') and msg.content:
                            if isinstance(msg.content, list):
                                text_content = []
                                for item in msg.content:
                                    if hasattr(item, 'text'):
                                        text_content.append(item.text)
                                    elif isinstance(item, str):
                                        text_content.append(item)
                                if text_content:
                                    answer_data["answer"] = "\n".join(text_content)
                                    break
                            elif isinstance(msg.content, str):
                                answer_data["answer"] = msg.content
                                break
        
        # If still no answer, check if MAX_ITERATION_PER_RUN limit is reached
        if not answer_data["answer"]:
            # Count ActionEvent to determine if limit is reached
            events = list(state.events)
            action_count = sum(1 for event in events if isinstance(event, ActionEvent))
            
            print(f"[DEBUG] Answer not found, ActionEvent count: {action_count}, max limit: {MAX_ITERATION_PER_RUN}")
            
            # If limit is reached or exceeded, use message_history to generate answer
            if action_count >= MAX_ITERATION_PER_RUN:
                print(f"[DEBUG] Reached max iteration limit ({MAX_ITERATION_PER_RUN}), using message_history to generate final answer...")
                
                if message_history and len(message_history) > 0:
                    forced_answer, (forced_prompt_tokens, forced_completion_tokens) = generate_answer_from_history(
                        LLM_CONFIG, question, message_history
                    )
                    
                    if forced_answer:
                        answer_data["answer"] = forced_answer
                        # Update token cost (accumulate forced generation tokens)
                        previous_prompt = answer_data.get("prompt_tokens", 0)
                        previous_completion = answer_data.get("completion_tokens", 0)
                        answer_data["prompt_tokens"] = previous_prompt + forced_prompt_tokens
                        answer_data["completion_tokens"] = previous_completion + forced_completion_tokens
                        answer_data["token_cost"] = answer_data["prompt_tokens"] + answer_data["completion_tokens"]
                        print(f"[DEBUG] Used message_history to generate answer, answer length: {len(answer_data['answer'])} characters")
                        print(f"[DEBUG] Token stats: prompt={previous_prompt}+{forced_prompt_tokens}={answer_data['prompt_tokens']}, completion={previous_completion}+{forced_completion_tokens}={answer_data['completion_tokens']}, total={answer_data['token_cost']}")
                    else:
                        print(f"[DEBUG] Warning: Failed to generate answer using message_history")
                        answer_data["answer"] = "Unable to generate answer based on conversation history."
                else:
                    print(f"[DEBUG] Warning: message_history is empty, unable to generate answer")
                    answer_data["answer"] = "No conversation history available to generate answer."
        
        # Explicitly close conversation to avoid cleanup errors
        try:
            conversation.close()
        except Exception:
            pass  # Ignore errors when closing
        
        return answer_data
        
    except Exception as e:
        print(f"Failed to process question: {question[:50]}... Error: {e}")
        answer_data["answer"] = f"Error: {str(e)}"
        # Ensure conversation is closed even on exception
        try:
            if 'conversation' in locals():
                conversation.close()
        except Exception:
            pass
        return answer_data

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Global statistics
    all_processed_count = 0
    all_error_count = 0
    all_time_costs = []
    all_token_costs = []
    all_prompt_tokens_list = []
    all_completion_tokens_list = []
    
    # Process each repository in order
    for repo_idx, repo_config in enumerate(REPOS_CONFIG, 1):
        repo_name = repo_config["name"]
        workspace = repo_config["workspace"]
        input_file = repo_config["input_file"]
        
        print(f"\n{'='*60}")
        print(f"Starting to process repository {repo_idx}/{len(REPOS_CONFIG)}: {repo_name}")
        print(f"{'='*60}")
        print(f"Workspace: {workspace}")
        print(f"Question file: {input_file}")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Warning: Input file does not exist, skipping: {input_file}")
            continue
        
        output_file = os.path.join(OUTPUT_DIR, f"{repo_name}_answers.jsonl")
        
        # Load answered questions
        answered_questions = load_answered_questions(output_file)
        if answered_questions:
            print(f"Found {len(answered_questions)} answered questions")
        
        # Load all questions
        print(f"Loading questions from {input_file}...")
        all_questions = load_questions_from_jsonl(input_file)
        print(f"Loaded {len(all_questions)} questions in total")
        
        # Filter out answered questions
        questions = [
            qa_data for qa_data in all_questions 
            if qa_data.get('question', '') not in answered_questions
        ]
        
        if len(questions) < len(all_questions):
            print(f"After filtering, {len(questions)} unanswered questions remaining")
        else:
            print(f"All questions are unanswered, will process all {len(questions)} questions")
        
        if len(questions) == 0:
            print(f"Repository {repo_name} has no questions to process, skipping")
            continue
        
        # Process sequentially
        processed_count = 0
        error_count = 0
        time_costs = []
        token_costs = []
        prompt_tokens_list = []
        completion_tokens_list = []
        
        for idx, qa_data in enumerate(questions, 1):
            try:
                result = process_single_question(qa_data, workspace)
                if result:
                    # Write to file
                    with open(output_file, 'a', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write('\n')
                    
                    # Collect statistics
                    time_cost = result.get('time_cost', 0)
                    token_cost = result.get('token_cost', 0)
                    prompt_tokens = result.get('prompt_tokens', 0)
                    completion_tokens = result.get('completion_tokens', 0)
                    
                    if time_cost is not None:
                        time_costs.append(time_cost)
                        all_time_costs.append(time_cost)
                    if token_cost is not None:
                        token_costs.append(token_cost)
                        all_token_costs.append(token_cost)
                    if prompt_tokens is not None:
                        prompt_tokens_list.append(prompt_tokens)
                        all_prompt_tokens_list.append(prompt_tokens)
                    if completion_tokens is not None:
                        completion_tokens_list.append(completion_tokens)
                        all_completion_tokens_list.append(completion_tokens)
                    
                    processed_count += 1
                    all_processed_count += 1
                    print(f"[{repo_name}][{idx}/{len(questions)}] Completed: {result['question'][:50]}...")
                else:
                    error_count += 1
                    all_error_count += 1
            except Exception as e:
                error_count += 1
                all_error_count += 1
                print(f"[{repo_name}][{idx}/{len(questions)}] Processing failed: {qa_data.get('question', '')[:50]}... Error: {e}")
        
        # Calculate statistics for current repository
        avg_time_cost = sum(time_costs) / len(time_costs) if time_costs else 0
        avg_token_cost = sum(token_costs) / len(token_costs) if token_costs else 0
        total_time_cost = sum(time_costs)
        total_token_cost = sum(token_costs)
        
        avg_prompt_tokens = sum(prompt_tokens_list) / len(prompt_tokens_list) if prompt_tokens_list else 0
        avg_completion_tokens = sum(completion_tokens_list) / len(completion_tokens_list) if completion_tokens_list else 0
        total_prompt_tokens = sum(prompt_tokens_list)
        total_completion_tokens = sum(completion_tokens_list)
        
        print(f"\nRepository {repo_name} processing completed:")
        print(f"  Success: {processed_count}, Failed: {error_count}, Total: {len(questions)}")
        print(f"  Average time_cost: {avg_time_cost:.2f} seconds")
        print(f"  Total time_cost: {total_time_cost:.2f} seconds")
        print(f"  Average token_cost: {avg_token_cost:.0f} tokens")
        print(f"  Total token_cost: {total_token_cost:.0f} tokens")
        print(f"  Average prompt_tokens: {avg_prompt_tokens:.0f}, completion_tokens: {avg_completion_tokens:.0f}")
        print(f"  Total prompt_tokens: {total_prompt_tokens:.0f}, completion_tokens: {total_completion_tokens:.0f}")
        print(f"  Results saved to: {output_file}")
    
    # Calculate global statistics
    avg_time_cost = sum(all_time_costs) / len(all_time_costs) if all_time_costs else 0
    avg_token_cost = sum(all_token_costs) / len(all_token_costs) if all_token_costs else 0
    total_time_cost = sum(all_time_costs)
    total_token_cost = sum(all_token_costs)
    
    avg_prompt_tokens = sum(all_prompt_tokens_list) / len(all_prompt_tokens_list) if all_prompt_tokens_list else 0
    avg_completion_tokens = sum(all_completion_tokens_list) / len(all_completion_tokens_list) if all_completion_tokens_list else 0
    total_prompt_tokens = sum(all_prompt_tokens_list)
    total_completion_tokens = sum(all_completion_tokens_list)
    
    print(f"\n{'='*60}")
    print(f"All repositories processing completed!")
    print(f"{'='*60}")
    print(f"Success: {all_processed_count}, Failed: {all_error_count}")
    print(f"\nGlobal statistics:")
    print(f"  Average time_cost: {avg_time_cost:.2f} seconds")
    print(f"  Total time_cost: {total_time_cost:.2f} seconds")
    print(f"\nToken statistics (total):")
    print(f"  Average token_cost: {avg_token_cost:.0f} tokens")
    print(f"  Total token_cost: {total_token_cost:.0f} tokens")
    print(f"\nToken statistics (separated):")
    print(f"  Average prompt_tokens (input): {avg_prompt_tokens:.0f} tokens")
    print(f"  Average completion_tokens (output): {avg_completion_tokens:.0f} tokens")
    print(f"  Total prompt_tokens (input): {total_prompt_tokens:.0f} tokens")
    print(f"  Total completion_tokens (output): {total_completion_tokens:.0f} tokens")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()