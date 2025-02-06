"""
Executor Agent for Deep Research system.
Responsible for executing concrete tasks and providing results.
"""

import logging
import os
import json
import time
import sys
from datetime import datetime
from typing import List, Set, Dict, Optional
from dataclasses import dataclass

import openai

from tools import chat_completion
from tool_definitions import function_definitions
from common import TokenUsage, TokenTracker

logger = logging.getLogger(__name__)

# Check if debug mode is enabled via command line argument
DEBUG_MODE = '--debug' in sys.argv

@dataclass
class ExecutorContext:
    """Context information for the Executor agent."""
    created_files: Set[str]
    scratchpad_content: Optional[str] = None
    total_usage: Optional[TokenUsage] = None
    debug: bool = DEBUG_MODE  # Default to command line debug setting

def save_prompt_to_file(messages: List[Dict[str, str]], round_time: str = None, prefix: str = "executor"):
    """Save prompt messages to a file for debugging."""
    if not os.path.exists('prompts'):
        os.makedirs('prompts')
    
    # Generate timestamp at save time if not provided
    if round_time is None:
        round_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    filename = f"prompts/{round_time}_{prefix}_prompt.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for msg in messages:
            f.write(f"Role: {msg['role']}\n")
            f.write("Content:\n")
            f.write(f"{msg['content']}\n")
            f.write("-" * 80 + "\n")
    logger.debug(f"Saved prompt to {filename}")

def save_response_to_file(response: str, tool_calls: List[Dict] = None, round_time: str = None, prefix: str = "executor"):
    """Save response and tool calls to a file for debugging."""
    if not os.path.exists('prompts'):
        os.makedirs('prompts')
    
    # Generate timestamp at save time if not provided
    if round_time is None:
        round_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    filename = f"prompts/{round_time}_{prefix}_response.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== Response ===\n")
        f.write(f"{response}\n")
        if tool_calls:
            f.write("\n=== Tool Calls ===\n")
            for tool_call in tool_calls:
                f.write(f"Tool: {tool_call.get('name', 'unknown')}\n")
                f.write("Arguments:\n")
                f.write(f"{json.dumps(tool_call.get('arguments', {}), indent=2, ensure_ascii=False)}\n")
                f.write("-" * 80 + "\n")
    logger.debug(f"Saved response to {filename}")

def log_usage(usage: Dict[str, int], thinking_time: float, step_name: str, model: str):
    """Log token usage and cost information."""
    cached_tokens = usage.get('cached_prompt_tokens', 0)
    cost = TokenTracker.calculate_cost(
        prompt_tokens=usage['prompt_tokens'],
        completion_tokens=usage['completion_tokens'],
        cached_tokens=cached_tokens,
        model=model
    )
    
    logger.info(f"\n{step_name} Token Usage:")
    logger.info(f"Input tokens: {usage['prompt_tokens']:,}")
    logger.info(f"Output tokens: {usage['completion_tokens']:,}")
    logger.info(f"Cached tokens: {cached_tokens:,}")
    logger.info(f"Total tokens: {usage['total_tokens']:,}")
    logger.info(f"Total cost: ${cost:.6f}")
    logger.info(f"Thinking time: {thinking_time:.2f}s")
    
    # Update the usage dict with the new cost
    usage['total_cost'] = cost

class ExecutorAgent:
    """
    Executor agent that performs concrete tasks based on Planner's instructions.
    Reads from .executorrules for system prompt.
    """
    
    def __init__(self, model: str):
        """Initialize the Executor agent.
        
        Args:
            model: The OpenAI model to use
        """
        self.model = model
        self.system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """Load system prompt from .executorrules file."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_prompt = f"""You are the Executor agent in a multi-agent research system. Today's date is {today}. Take this into consideration when you search for and analyze information."""
        
        if os.path.exists('.executorrules'):
            with open('.executorrules', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.debug("Loaded executor rules")
                return f"{content}\n{today_prompt}"
        else:
            raise FileNotFoundError("Required .executorrules file not found")

    def _load_file_contents(self, context: ExecutorContext) -> Dict[str, str]:
        """Load contents of all created files."""
        file_contents = {}
        for filename in context.created_files:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    logger.debug(f"Loaded file {filename}")
                    file_contents[filename] = content
            except Exception as e:
                logger.error(f"Error reading file {filename}: {e}")
                file_contents[filename] = f"[Error reading file: {str(e)}]"
        return file_contents

    def _build_prompt(self, context: ExecutorContext) -> List[Dict[str, str]]:
        """Build the complete prompt including context and files."""
        logger.debug("Building executor prompt")
        messages = [
            {"role": "user", "content": self.system_prompt},
        ]
        
        # Add file contents and task context
        file_contents = self._load_file_contents(context)
            
        # Build context message
        context_message = "\nRelevant Files:\n"
        
        # Add all files including scratchpad.md
        if file_contents:
            for filename, content in file_contents.items():
                context_message += f"\n--- {filename} ---\n{content}\n"
        
        # Add available files list
        context_message += f"\nAvailable Files: {', '.join(context.created_files)}\n"
        
        messages.append({"role": "user", "content": context_message})
        return messages

    def execute(self, context: ExecutorContext) -> str:
        """Execute task based on instructions."""
        logger.info("=== Starting Executor execution ===")
        
        # Store the context
        self.context = context
        
        messages = self._build_prompt(context)
        
        # Save prompt if debug mode is enabled
        if context.debug:
            save_prompt_to_file(messages)
        
        try:
            while True:  # Loop to handle chained tool calls
                # Start timer
                start_time = time.time()
                
                logger.debug("Calling chat completion")
                response = chat_completion.chat_completion(
                    messages=messages,
                    model=self.model,
                    functions=function_definitions,
                    function_call="auto"
                )
                
                # Calculate thinking time and token usage
                thinking_time = time.time() - start_time
                usage = chat_completion.get_token_usage()
                
                # Log usage statistics
                log_usage(usage, thinking_time, "Step", self.model)
                
                # Update the context's total usage
                if not context.total_usage:
                    context.total_usage = TokenUsage(
                        prompt_tokens=usage['prompt_tokens'],
                        completion_tokens=usage['completion_tokens'],
                        total_tokens=usage['total_tokens'],
                        total_cost=usage['total_cost'],
                        thinking_time=thinking_time,
                        cached_prompt_tokens=usage.get('cached_prompt_tokens', 0)
                    )
                else:
                    # Just use the current round's usage directly from chat_completion
                    context.total_usage = chat_completion.token_tracker.get_total_usage()
                
                message = response.choices[0].message
                logger.debug(f"Received response type: {'content' if message.content else 'tool call'}")
                
                # Save response if debug mode is enabled
                if context.debug:
                    tool_calls = []
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        tool_calls = [{'name': tc.function.name, 'arguments': json.loads(tc.function.arguments)} 
                                    for tc in message.tool_calls]
                    elif hasattr(message, 'function_call') and message.function_call:
                        tool_calls = [{'name': message.function_call.name, 
                                     'arguments': json.loads(message.function_call.arguments)}]
                    save_response_to_file(message.content or "", tool_calls)

                # Check for tool calls (new format) or function call (old format)
                has_tool_call = False
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    logger.info("Processing tool_calls (new format)")
                    if len(message.tool_calls) > 1:
                        raise ValueError(f"Multiple tool calls not supported. Received {len(message.tool_calls)} calls.")
                    tool_call = message.tool_calls[0]  # For now, handle first tool call
                    func_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    logger.info(f"Tool call detected - Function: {func_name}")
                    logger.debug(f"Tool arguments: {json.dumps(arguments, ensure_ascii=False)}")
                    has_tool_call = True
                elif hasattr(message, 'function_call') and message.function_call:
                    logger.info("Processing function_call (old format)")
                    func_name = message.function_call.name
                    arguments = json.loads(message.function_call.arguments)
                    logger.info(f"Function call detected - Function: {func_name}")
                    logger.debug(f"Function arguments: {json.dumps(arguments, ensure_ascii=False)}")
                    has_tool_call = True
                
                if has_tool_call:
                    # Execute the tool
                    logger.info(f"Executing tool: {func_name}")
                    start_time = time.time()
                    tool_result = self._execute_tool(func_name, arguments)
                    tool_execution_time = time.time() - start_time
                    logger.info(f"Tool execution completed in {tool_execution_time:.2f}s")
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": func_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False)
                        }
                    })
                    # Include function result in the next user message instead of using 'function' role
                    result_preview = tool_result[:200] + "..." if tool_result and len(tool_result) > 200 else tool_result
                    logger.debug(f"Tool result preview: {result_preview}")
                    messages.append({
                        "role": "user",
                        "content": f"Function {func_name} returned: {tool_result}"
                    })
                    logger.info("Added tool result to conversation history")
                    
                    # Continue the loop to get model's interpretation of the result
                    continue
                else:
                    # No more tool calls, return the final response
                    logger.info("No tool calls detected, returning final response")
                    return message.content or "Task completed successfully"
            
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            return f"Error during execution: {str(e)}"
            
    def _execute_tool(self, func_name: str, arguments: Dict) -> Optional[str]:
        """Execute a tool with given name and arguments."""
        try:
            result = None
            if func_name == "create_file":
                from tools import create_file
                filename = arguments.get('filename')
                logger.info(f"Creating file: {filename}")
                result = create_file(**arguments)
                # Add the created file to the set
                if filename:
                    self.context.created_files.add(filename)
                logger.info("File creation completed")
            elif func_name == "perform_search":
                from tools import perform_search
                query = arguments.get('query', '')
                logger.info(f"Starting search with query: {query}")
                result = perform_search(**arguments)
                result_lines = len(result.split('\n'))
                result_chars = len(result)
                logger.info(f"Search completed. Response size: {result_chars} chars, {result_lines} lines")
                logger.debug(f"First 200 chars of response: {result[:200]}...")
            elif func_name == "fetch_web_content":
                from tools import fetch_web_content
                urls = arguments.get('urls', [])
                logger.info(f"Starting content fetch from {len(urls)} URLs")
                for i, url in enumerate(urls, 1):
                    logger.info(f"Fetching URL {i}/{len(urls)}: {url}")
                result = fetch_web_content(**arguments)
                result_size = len(result)
                logger.info(f"Content fetch completed. Total response size: {result_size} chars")
            elif func_name == "execute_command":
                command = arguments.get("command")
                explanation = arguments.get("explanation")
                
                if not command:
                    result = "Error: No command provided"
                    logger.error("Command execution failed: no command provided")
                else:
                    logger.info(f"Preparing to execute command: {command}")
                    logger.info(f"Command explanation: {explanation}")
                    
                    # Ask for user confirmation
                    print(f"\nConfirm execution of command: {command}")
                    print(f"Explanation: {explanation}")
                    confirmation = input("[y/N]: ").strip().lower()
                    
                    if confirmation != 'y':
                        result = "Command execution cancelled by user"
                        logger.info("Command execution cancelled by user")
                    else:
                        # Execute command
                        import subprocess
                        try:
                            logger.info("Starting command execution...")
                            cmd_result = subprocess.run(
                                command,
                                shell=True,
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            stdout_size = len(cmd_result.stdout)
                            stderr_size = len(cmd_result.stderr)
                            result = f"stdout:\n{cmd_result.stdout}\nstderr:\n{cmd_result.stderr}"
                            logger.info(f"Command execution completed. stdout: {stdout_size} chars, stderr: {stderr_size} chars")
                        except subprocess.CalledProcessError as e:
                            error_msg = f"Error executing command: stdout={e.stdout}, stderr={e.stderr}"
                            logger.error(error_msg)
                            result = error_msg
                        except Exception as e:
                            error_msg = f"Error executing command: {str(e)}"
                            logger.error(error_msg)
                            result = error_msg
            else:
                error_msg = f"Unknown function: {func_name}"
                logger.error(error_msg)
                result = error_msg
            
            # Log the result size for all tools
            if result:
                result_size = len(result)
                logger.info(f"Tool {func_name} completed with result size: {result_size} chars")
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing tool {func_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg 