"""
Planner Agent for Deep Research system.
Responsible for high-level planning and task decomposition.
"""

import logging
import os
import time
import sys
from datetime import datetime
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
import json

import openai

from tools import chat_completion
from tool_definitions import function_definitions
from common import calculate_cost, TokenUsage

logger = logging.getLogger(__name__)

# Check if debug mode is enabled via command line argument
DEBUG_MODE = '--debug' in sys.argv

@dataclass
class PlannerContext:
    """Context information for the Planner agent."""
    conversation_history: List[Dict[str, str]]
    created_files: Set[str]
    user_input: str
    scratchpad_content: Optional[str] = None
    total_usage: Optional[TokenUsage] = None
    debug: bool = DEBUG_MODE  # Default to command line debug setting

def save_prompt_to_file(messages: List[Dict[str, str]], round_time: str = None, step: str = "planning"):
    """Save prompt messages to a file for debugging."""
    if not os.path.exists('prompts'):
        os.makedirs('prompts')
    
    # Generate timestamp at save time if not provided
    if round_time is None:
        round_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    filename = f"prompts/{round_time}_planner_{step}_prompt.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for msg in messages:
            f.write(f"Role: {msg['role']}\n")
            f.write("Content:\n")
            f.write(f"{msg['content']}\n")
            if msg.get('function_call'):
                f.write("Function Call:\n")
                f.write(f"{json.dumps(msg['function_call'], indent=2)}\n")
            f.write("-" * 80 + "\n")
    logger.debug(f"Saved prompt to {filename}")

def save_response_to_file(response: str, tool_calls: List[Dict] = None, round_time: str = None, step: str = "planning"):
    """Save response and tool calls to a file for debugging."""
    if not os.path.exists('prompts'):
        os.makedirs('prompts')
    
    # Generate timestamp at save time if not provided
    if round_time is None:
        round_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    filename = f"prompts/{round_time}_planner_{step}_response.txt"
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
    cost = calculate_cost(
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

class PlannerAgent:
    """
    Planner agent that maintains full context and plans next steps.
    Reads from .plannerrules for system prompt.
    """
    
    def __init__(self, model: str):
        """Initialize the Planner agent.
        
        Args:
            model: The OpenAI model to use
        """
        self.model = model
        self.system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """Load system prompt from .plannerrules file."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_prompt = f"""You are the Planner agent in a multi-agent research system. Today's date is {today}. Take this into consideration when you plan tasks and analyze progress."""
        
        if os.path.exists('.plannerrules'):
            with open('.plannerrules', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.debug("Loaded planner rules")
                return f"{content}\n{today_prompt}"
        else:
            raise FileNotFoundError("Required .plannerrules file not found")

    def _load_file_contents(self, context: PlannerContext) -> Dict[str, str]:
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

    def _build_prompt(self, context: PlannerContext) -> List[Dict[str, str]]:
        """Build the complete prompt including context and files."""
        logger.debug("Building planner prompt")
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # Add file contents
        file_contents = self._load_file_contents(context)
            
        # Build context message
        context_message = "\nCurrent User Request:\n"
        context_message += f"{context.user_input}\n\n"
        
        # Add all files including scratchpad.md
        if file_contents:
            context_message += "Relevant Files:\n"
            for filename, content in file_contents.items():
                context_message += f"\n--- {filename} ---\n{content}\n"
        
        # Add available files list
        context_message += f"\nAvailable Files: {', '.join(context.created_files)}\n"
        
        messages.append({"role": "user", "content": context_message})
        return messages

    def plan(self, context: PlannerContext) -> str:
        """Plan next steps based on current state and user input."""
        logger.info("=== Starting Planner planning ===")
        
        messages = self._build_prompt(context)
        
        # Save prompt if debug mode is enabled
        if context.debug:
            save_prompt_to_file(messages)
        
        try:
            # Initialize total usage for this round
            round_usage = TokenUsage(0, 0, 0, 0.0, 0.0, 0)
            
            # Start timer
            start_time = time.time()
            
            logger.debug("Calling chat completion")
            response = chat_completion.chat_completion(
                messages=messages,
                model=self.model,
                functions=[{
                    "name": "create_file",
                    "description": "Create or update a file with the given content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Name of the file to create"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["filename", "content"]
                    }
                }],
                function_call={"name": "create_file"}
            )
            
            # Calculate thinking time and token usage
            thinking_time = time.time() - start_time
            usage = chat_completion.get_token_usage()
            
            # Log usage statistics
            log_usage(usage, thinking_time, "Step", self.model)
            
            # Update round usage
            round_usage.prompt_tokens = usage['prompt_tokens']
            round_usage.completion_tokens = usage['completion_tokens']
            round_usage.total_tokens = usage['total_tokens']
            round_usage.total_cost = usage['total_cost']
            round_usage.thinking_time = thinking_time
            round_usage.cached_prompt_tokens = usage.get('cached_prompt_tokens', 0)
            
            # Update the context's total usage with this round's usage
            if not context.total_usage:
                context.total_usage = round_usage
            else:
                # Only add the new tokens from this round
                context.total_usage.prompt_tokens = max(context.total_usage.prompt_tokens, round_usage.prompt_tokens)
                context.total_usage.completion_tokens += round_usage.completion_tokens
                context.total_usage.total_tokens = context.total_usage.prompt_tokens + context.total_usage.completion_tokens
                context.total_usage.total_cost += round_usage.total_cost
                context.total_usage.thinking_time += round_usage.thinking_time
                context.total_usage.cached_prompt_tokens = max(context.total_usage.cached_prompt_tokens, round_usage.cached_prompt_tokens)
            
            message = response.choices[0].message
            logger.debug(f"Received response type: {'content' if message.content else 'function call'}")
            
            # Save response if debug mode is enabled
            if context.debug:
                tool_calls = []
                if hasattr(message, 'function_call') and message.function_call:
                    tool_calls = [{'name': message.function_call.name, 
                                 'arguments': json.loads(message.function_call.arguments)}]
                save_response_to_file(message.content or "", tool_calls)

            # Handle the function call to update scratchpad
            if not message.function_call:
                logger.error("Model did not provide a function call despite being forced")
                return "Error: Failed to update progress tracking"
            
            # Parse the function call
            arguments = json.loads(message.function_call.arguments)
            filename = arguments.get("filename")
            content = arguments.get("content")
            
            # Validate that we're updating the scratchpad
            if filename != "scratchpad.md":
                logger.warning(f"Model tried to create/update {filename} instead of scratchpad.md")
                filename = "scratchpad.md"
            
            # Update the scratchpad
            from tools import create_file
            create_file(filename=filename, content=content)
            
            return "Successfully updated scratchpad.md with next steps"
            
        except Exception as e:
            logger.error(f"Error during planning: {e}", exc_info=True)
            return f"Error during planning: {str(e)}" 