#!/usr/bin/env python3
"""
Interactive chat script with integrated tools for web search, content scraping, and package management.
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from typing import Optional, List
from dataclasses import dataclass
import time

import openai

from tool_definitions import function_definitions
import tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None

@dataclass
class ChatResponse:
    content: str
    token_usage: TokenUsage
    cost: float
    thinking_time: float = 0.0

def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate the cost of API usage based on model pricing.
    
    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of completion tokens
        model: Model name to determine pricing
    
    Returns:
        Total cost in USD
    """
    # Pricing per 1M tokens for different models
    MODEL_PRICING = {
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o1": {"input": 15.0, "output": 60.0},
    }
    
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["o3-mini"])
    
    # Convert to millions and calculate
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost

def load_system_prompt(config_file: str = '.deep_research_rules') -> str:
    """
    Load system prompt from configuration file.

    Args:
        config_file: Path to the configuration file containing system prompt

    Returns:
        System prompt string
    """
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    today_prompt = f"""You are an AI staff helping to execute tasks using the tools at your hand. Today's date is {today}. Take this into consideration when you think about history and future."""
    
    try:
        if not os.path.exists(config_file):
            logger.warning(f"Config file {config_file} not found. Using default system prompt.")
            return today_prompt
        
        with open(config_file, 'r', encoding='utf-8') as f:
            custom_prompt = f.read().strip()
            return f"{custom_prompt}\n{today_prompt}"
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        return f"""You are an AI staff helping to execute tasks using the tools at your hand. Today's date is {today}."""

def handle_function_call(message: openai.types.chat.ChatCompletionMessage) -> Optional[str]:
    """
    Execute the function call from the assistant's message and return the result.

    Args:
        message: OpenAI chat completion message containing function call

    Returns:
        Function result string or None if no function call
    """
    function_call = message.function_call
    if function_call:
        func_name = function_call.name
        arguments = json.loads(function_call.arguments)
        
        # Handle terminal command execution
        if func_name == "execute_command":
            command = arguments.get("command")
            explanation = arguments.get("explanation")
            
            if not command:
                return "Error: No command provided"
            
            # Ask for user confirmation
            print(f"\nConfirm execution of command: {command}")
            print(f"Explanation: {explanation}")
            confirmation = input("[y/N]: ").strip().lower()
            
            if confirmation != 'y':
                return "Command execution cancelled by user."
            
            # Execute command
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True
                )
                return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            except subprocess.CalledProcessError as e:
                return f"Error executing command: stdout={e.stdout}, stderr={e.stderr}"
            except Exception as e:
                return f"Error executing command: {str(e)}"
        
        # Handle other functions
        elif func_name == "perform_search":
            return tools.perform_search(**arguments)
        elif func_name == "fetch_web_content":
            return tools.fetch_web_content(**arguments)
        elif func_name == "create_file":
            return tools.create_file(**arguments)
        else:
            return f"Unknown function: {func_name}"
    return None

def chat_loop(model: str, query: str, system_prompt: str) -> None:
    """
    Main chat loop function.

    Args:
        model: OpenAI model to use
        query: User's query
        system_prompt: System prompt for the assistant
    """
    # Start a conversation
    conversation = []
    prompt = system_prompt + "\n\nUser's request:\n" + query
    print(f"\nTask:\n{prompt}")
    
    # Add initial prompt to conversation
    conversation.append({"role": "user", "content": prompt})
    
    # Counter for non-tool responses
    non_tool_responses = 0
    max_retries = 3  # Maximum number of retries for empty responses
    retry_count = 0  # Counter for retries
    
    while True:
        try:
            # Start timer
            start_time = time.time()
            
            logger.info("Getting assistant's response...")
            # Get assistant's response using cached chat completion
            response = tools.chat_completion.chat_completion(
                model=model,
                messages=conversation,
                functions=function_definitions,
                function_call="auto",
                reasoning_effort='high'
            )
            
            # Calculate thinking time
            thinking_time = time.time() - start_time
            
            # Log usage for this step
            usage = tools.chat_completion.get_token_usage()
            logger.info(f"\nStep Token Usage:")
            logger.info(f"Input tokens: {usage['prompt_tokens']}")
            logger.info(f"Output tokens: {usage['completion_tokens']}")
            logger.info(f"Total tokens: {usage['total_tokens']}")
            logger.info(f"Total cost: ${usage['total_cost']:.6f}")
            logger.info(f"Thinking time: {thinking_time:.2f}s")
            
            assistant_message = response.choices[0].message
            logger.info(f"Assistant message: {assistant_message}")
            
            # Check if the message is empty and has no tool calls
            if not assistant_message.content and not getattr(assistant_message, 'tool_calls', None) and not getattr(assistant_message, 'function_call', None):
                logger.error("Received empty message from assistant with no tool calls")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\nReceived empty response. Retrying... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(1)  # Add a small delay between retries
                    continue
                else:
                    print("\nError: Assistant returned empty responses after maximum retries. Please try again later.")
                    break
            
            # Reset retry count on successful response
            retry_count = 0
            
            # Check for tool calls (new format) or function call (old format)
            has_tool_call = False
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                logger.info("Found tool_calls in message")
                tool_call = assistant_message.tool_calls[0]  # For now, handle first tool call
                func_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                has_tool_call = True
            elif hasattr(assistant_message, 'function_call') and assistant_message.function_call:
                logger.info("Found function_call in message")
                func_name = assistant_message.function_call.name
                arguments = json.loads(assistant_message.function_call.arguments)
                has_tool_call = True
            
            # If the assistant wants to use a tool
            if has_tool_call:
                # Parse and display the tool call details
                print(f"\nAssistant: Using tool '{func_name}' with parameters:")
                for key, value in arguments.items():
                    print(f"  - {key}: {value}")
                
                # Execute the tool
                result = handle_function_call(assistant_message)
                print("\nTool output:")
                if result and len(result) > 500:
                    print(f"{result[:500]}...\n[Output truncated, total length: {len(result)} chars]")
                else:
                    print(result)
                
                # Add the function call and result to conversation
                conversation.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": func_name,
                        "arguments": json.dumps(arguments)
                    }
                })
                # Include function result in the next user message
                conversation.append({
                    "role": "user",
                    "content": f"Function {func_name} returned: {result}"
                })
            else:
                # If it's a final response (no more tool calls needed)
                print("\nAssistant's Response:")
                print(assistant_message.content)
                conversation.append({"role": "assistant", "content": assistant_message.content})
                
                # Give user a chance to provide additional input
                print("\nWould you like to add any comments or provide additional input? (Press Enter to skip, 'q' to quit)")
                user_input = input("> ").strip()
                if user_input.lower() == 'q':
                    # Print token usage statistics
                    usage = tools.chat_completion.get_token_usage()
                    print("\nToken Usage Statistics:")
                    print(f"Total prompt tokens: {usage['prompt_tokens']}")
                    print(f"Total completion tokens: {usage['completion_tokens']}")
                    print(f"Total cached tokens: {usage['cached_prompt_tokens']}")
                    print(f"Total tokens: {usage['total_tokens']}")
                    print(f"\nCost Information:")
                    print(f"Total cost: ${usage['total_cost']:.6f}")
                    break
                elif user_input:
                    print("\nReceived your input. Sending request to assistant...")
                    conversation.append({"role": "user", "content": user_input})
                    continue
                else:
                    print("\nNo additional input received. Continuing with the conversation...")
                
                non_tool_responses += 1
                if non_tool_responses == 2:
                    # Print token usage statistics
                    usage = tools.chat_completion.get_token_usage()
                    print("\nToken Usage Statistics:")
                    print(f"Total prompt tokens: {usage['prompt_tokens']}")
                    print(f"Total completion tokens: {usage['completion_tokens']}")
                    print(f"Total cached tokens: {usage['cached_prompt_tokens']}")
                    print(f"Total tokens: {usage['total_tokens']}")
                    print(f"\nCost Information:")
                    print(f"Total cost: ${usage['total_cost']:.6f}")
                    break
                elif non_tool_responses == 1:
                    # Prepare reflection prompt
                    reflection_prompt = "Do you think you have fully addressed the user's request? Please carefully consider if there are any missing aspects, e.g. are the facts backed by evidences? If the task is not completely resolved, please continue using the tools to complete it."
                    
                    # Add scratchpad content if it exists
                    try:
                        if os.path.exists('scratchpad.md'):
                            with open('scratchpad.md', 'r', encoding='utf-8') as f:
                                scratchpad_content = f.read()
                                reflection_prompt += f"\n\nHere is the current scratchpad content for your reference in determining task completion:\n\n{scratchpad_content}"
                                reflection_prompt += "\n\nHere is the user's request:\n\n" + query
                                reflection_prompt += "\n\nPlease first update the scratchpad to reflect the progress of the task. Then think about how to further improve the report, and execute the plan to improve the report."
                    except Exception as e:
                        logger.warning(f"Failed to read scratchpad.md: {e}")
                    
                    conversation.append({"role": "user", "content": reflection_prompt})
                    print("\nAsking assistant to reflect on task completion...")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Error details:", type(e).__name__)
            break

def main() -> None:
    """Main function to parse arguments and start the chat."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with integrated tools for research and analysis."
    )
    parser.add_argument(
        "query",
        help="The research query or task to investigate"
    )
    parser.add_argument(
        "--model",
        default="o3-mini",
        help="OpenAI model to use (default: o3-mini)"
    )
    parser.add_argument(
        "--config",
        default=".deep_research_rules",
        help="Path to configuration file containing system prompt (default: .deep_research_rules)"
    )
    
    args = parser.parse_args()
    
    # Load system prompt
    system_prompt = load_system_prompt(args.config)
    
    # Start chat loop
    chat_loop(args.model, args.query, system_prompt)

if __name__ == "__main__":
    main() 