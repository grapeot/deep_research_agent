"""
Executor Agent for Deep Research system.
Responsible for executing concrete tasks and providing results.
"""

import logging
import os
import json
import time
import sys
import subprocess
from datetime import datetime
from typing import List, Set, Dict, Optional, Any, Union
from dataclasses import dataclass

from anthropic import Anthropic

# Import tools module instead of individual functions that may not be directly exposed
import tools
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
                f.write(f"{json.dumps(tool_call.get('input', {}), indent=2, ensure_ascii=False)}\n")
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
            model: The model to use (only Claude 3.7 Sonnet is supported)
        """
        # Always use Claude 3.7 Sonnet regardless of input
        self.model = "claude-3-7-sonnet-20250219"
        self.system_prompt = self._load_system_prompt()
        
        # Get API key from environment
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment. Please set it before using Claude.")
            raise ValueError("ANTHROPIC_API_KEY environment variable is required but not set")
        
        # Initialize the Anthropic client
        try:
            self.client = Anthropic(api_key=api_key)
            logger.info("Successfully initialized Anthropic client")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            raise

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

    def _build_prompt(self, context: ExecutorContext) -> List[Dict[str, Any]]:
        """Build the complete prompt including context and files."""
        logger.debug("Building executor prompt")
        
        # Build message for Claude's API
        messages = []
        
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
        
        # Create a user message with system instructions and context
        messages.append({
            "role": "user", 
            "content": [
                {"type": "text", "text": self.system_prompt},
                {"type": "text", "text": context_message}
            ]
        })
        
        return messages

    def _format_tools_for_claude(self, tools: List[Dict]) -> List[Dict]:
        """Format OpenAI-style function tools for Claude's API."""
        claude_tools = []
        
        for tool in tools:
            name = tool["name"]
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
            
            claude_tool = {
                "name": name,
                "description": description,
                "input_schema": parameters
            }
            
            claude_tools.append(claude_tool)
        
        return claude_tools

    def _extract_response_text(self, response):
        """Extract text from Claude response."""
        try:
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
                content = response_dict.get('content', [])
                
                # Look for text content
                text_parts = []
                for content_block in content:
                    if content_block.get('type') == "text":
                        text_parts.append(content_block.get('text', ''))
                
                if text_parts:
                    return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting response text: {e}")
        
        return "No text in response"

    def _get_tool_use(self, response):
        """Extract tool use from Claude response."""
        try:
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
                
                # Look for tool_use blocks in content
                content = response_dict.get('content', [])
                tool_calls = []
                
                for block in content:
                    if block.get('type') == 'tool_use':
                        # Create a tool call object with the expected format
                        tool_call = {
                            'id': block.get('id'),
                            'name': block.get('name'),
                            'input': block.get('input', {})
                        }
                        tool_calls.append(tool_call)
                
                if tool_calls:
                    logger.debug(f"Found {len(tool_calls)} tool calls in response")
                    return tool_calls
                
                # If stop_reason is tool_use but we didn't find tool calls in content,
                # something unexpected happened
                if response_dict.get('stop_reason') == 'tool_use':
                    logger.warning("Response has stop_reason='tool_use' but no tool calls were extracted")
        except Exception as e:
            logger.error(f"Error extracting tool calls: {e}")
        
        return []

    def _process_tool_calls(self, tool_calls, context):
        """Process function calls and return results for each call."""
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_input = tool_call.get("input", {})
            tool_id = tool_call.get("id")
            
            logger.info(f"Processing tool call: {tool_name} with input: {tool_input}")
            
            result = None
            if tool_name == "perform_search":
                query = tool_input.get("query", "")
                max_results = tool_input.get("max_results", 10)
                max_retries = tool_input.get("max_retries", 3)
                result = tools.perform_search(query=query, max_results=max_results, max_retries=max_retries)
            elif tool_name == "fetch_web_content":
                urls = tool_input.get("urls", [])
                max_concurrent = tool_input.get("max_concurrent", 3)
                result = tools.fetch_web_content(urls=urls, max_concurrent=max_concurrent)
            elif tool_name == "create_file":
                filename = tool_input.get("filename", "")
                content = tool_input.get("content", "")
                result = tools.create_file(filename=filename, content=content)
                # Add the created file to the set
                if filename:
                    context.created_files.add(filename)
            elif tool_name == "execute_command":
                command = tool_input.get("command", "")
                explanation = tool_input.get("explanation", "")
                
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
                error_msg = f"Unknown function: {tool_name}"
                logger.error(error_msg)
                result = error_msg
            
            # Convert result to string if it's not already a string
            if not isinstance(result, str):
                result = json.dumps(result)
            
            # Log the length of the result instead of its type
            logger.info(f"Tool call result length: {len(result)} characters")
            logger.debug(f"Tool call result: {result[:200]}..." if len(result) > 200 else result)
            
            tool_results.append({
                "tool_use_id": tool_id,
                "content": result
            })
        
        return tool_results

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
            iteration = 0
            max_iterations = 12  # Prevent infinite loops
            
            while iteration < max_iterations:
                # Start timer
                start_time = time.time()
                
                logger.debug("Calling Claude chat completion")
                
                # Format tools for Claude
                claude_tools = self._format_tools_for_claude(function_definitions)
                
                # Prepare API call parameters
                params = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 4000,
                    "tools": claude_tools,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 2000
                    }
                }
                
                # Make the API call
                try:
                    response = self.client.beta.messages.create(**params)
                except Exception as e:
                    logger.error(f"API call error: {str(e)}")
                    return f"Error during API call: {str(e)}"
                
                # Calculate thinking time and token usage
                thinking_time = time.time() - start_time
                
                # Extract token usage
                completion_tokens = response.usage.output_tokens
                prompt_tokens = response.usage.input_tokens
                total_tokens = prompt_tokens + completion_tokens
                
                # Calculate cost
                cost = TokenTracker.calculate_cost(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=0,  # Claude doesn't have cached tokens
                    model=self.model
                )
                
                # Create usage dictionary
                usage = {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                    'total_cost': cost,
                    'cached_prompt_tokens': 0  # Claude doesn't have cached tokens
                }
                
                # Log usage statistics for this step only
                log_usage(usage, thinking_time, "Step", self.model)
                
                # Store the current step's usage in context
                if not context.total_usage:
                    context.total_usage = TokenUsage(
                        prompt_tokens=usage['prompt_tokens'],
                        completion_tokens=usage['completion_tokens'],
                        total_tokens=usage['total_tokens'],
                        total_cost=usage['total_cost'],
                        thinking_time=thinking_time,
                        cached_prompt_tokens=0
                    )
                else:
                    # Add this step's usage to context's running total
                    context.total_usage.prompt_tokens += usage['prompt_tokens']
                    context.total_usage.completion_tokens += usage['completion_tokens']
                    context.total_usage.total_tokens += usage['total_tokens']
                    context.total_usage.total_cost += usage['total_cost']
                    context.total_usage.thinking_time += thinking_time
                
                # Extract text from response
                text_response = self._extract_response_text(response)
                logger.info(f"Claude Response Content:\n{text_response}")
                
                # Get the complete response content to preserve tool_use blocks
                response_dict = response.model_dump()
                response_content = response_dict.get('content', [])
                
                # Extract tool calls from response
                tool_calls = self._get_tool_use(response)
                
                # Save response if debug mode is enabled
                if context.debug:
                    save_response_to_file(text_response, tool_calls)
                
                # Add Claude's complete response to conversation
                messages.append({
                    "role": "assistant",
                    "content": response_content
                })
                
                # Check if there are tool calls
                if tool_calls:
                    logger.info(f"Claude wants to use {len(tool_calls)} tools")
                    
                    # Process tool calls
                    tool_results = self._process_tool_calls(tool_calls, self.context)
                    
                    # Format tool results as content blocks in a user message
                    tool_result_blocks = []
                    for result in tool_results:
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": result["tool_use_id"],
                            "content": result["content"]
                        })
                    
                    # Add a user message with tool result content blocks
                    messages.append({
                        "role": "user", 
                        "content": tool_result_blocks
                    })
                    
                    # Continue to next iteration
                    iteration += 1
                    continue
                else:
                    # No more tool calls, return the final response
                    logger.info("No tool calls detected, returning final response")
                    
                    # Check for special markers in the response
                    if text_response.strip().startswith("WAIT_USER_CONFIRMATION"):
                        return text_response
                    
                    return text_response or "Task completed successfully"
            
            # If we've reached max iterations without resolution
            return "Exceeded maximum number of tool call iterations without completing the task."
            
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            return f"Error during execution: {str(e)}" 