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
from typing import Optional

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
    # Initialize OpenAI client
    client = openai.OpenAI()
    
    # Start a conversation
    conversation = []
    prompt = system_prompt + "\n\nUser's request:\n" + query
    print(f"\nTask:\n{prompt}")
    
    # Add initial prompt to conversation
    conversation.append({"role": "user", "content": prompt})
    
    # Counter for non-tool responses
    non_tool_responses = 0
    
    while True:
        try:
            # Get assistant's response
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
                functions=function_definitions,
                function_call="auto",
                reasoning_effort='high'
            )
            assistant_message = response.choices[0].message
            
            # If the assistant wants to use a tool
            if assistant_message.function_call:
                # Parse and display the tool call details
                func_name = assistant_message.function_call.name
                arguments = json.loads(assistant_message.function_call.arguments)
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
                        "arguments": assistant_message.function_call.arguments
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
                
                non_tool_responses += 1
                if non_tool_responses == 1:
                    # Prepare reflection prompt
                    reflection_prompt = "Do you think you have fully addressed the user's request? Please carefully consider if there are any missing aspects. If the task is not completely resolved, please continue using the tools to complete it."
                    
                    # Add scratchpad content if it exists
                    try:
                        if os.path.exists('scratchpad.md'):
                            with open('scratchpad.md', 'r', encoding='utf-8') as f:
                                scratchpad_content = f.read()
                                reflection_prompt += f"\n\nHere is the current scratchpad content for your reference in determining task completion:\n\n{scratchpad_content}"
                    except Exception as e:
                        logger.warning(f"Failed to read scratchpad.md: {e}")
                    
                    conversation.append({"role": "user", "content": reflection_prompt})
                    print("\nAsking assistant to reflect on task completion...")
                elif non_tool_responses == 2:
                    # End the conversation after second non-tool response
                    break
                
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