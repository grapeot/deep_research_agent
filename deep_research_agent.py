#!/usr/bin/env python3
"""
Interactive chat script with integrated tools for web search, content scraping, and package management.
Using a multi-agent architecture with Planner and Executor agents.
"""

import argparse
import logging
import os
import sys
import signal
from typing import Set, Optional, Dict, Any
from datetime import datetime

from planner_agent import PlannerAgent, PlannerContext
from executor_agent import ExecutorAgent, ExecutorContext
from common import TokenUsage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deep Research Agent')
    parser.add_argument('query', help='The research query to process')
    parser.add_argument('--model', default='gpt-4-turbo-preview', help='The OpenAI model to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

class AgentCommunication:
    """Handles structured communication between Planner and Executor agents."""
    
    @staticmethod
    def format_planner_instructions(
        task: str,
        deliverables: list,
        constraints: list,
        prerequisites: list
    ) -> str:
        """Format instructions from Planner to Executor."""
        return f"""PLANNER_INSTRUCTIONS:
- Task: {task}
- Required Deliverables:
  {chr(10).join(f'  - {d}' for d in deliverables)}
- Constraints:
  {chr(10).join(f'  - {c}' for c in constraints)}
- Prerequisites:
  {chr(10).join(f'  - {p}' for p in prerequisites)}"""

    @staticmethod
    def format_executor_feedback(
        status: str,
        completion_details: str,
        blockers: list,
        resources_needed: list,
        next_task_ready: bool
    ) -> str:
        """Format feedback from Executor to Planner."""
        return f"""EXECUTOR_FEEDBACK:
- Task Status: {status}
- Completion Details: {completion_details}
- Blockers:
  {chr(10).join(f'  - {b}' for b in blockers)}
- Resources Needed:
  {chr(10).join(f'  - {r}' for r in resources_needed)}
- Next Task Readiness: {'Ready' if next_task_ready else 'Not Ready'}"""

    @staticmethod
    def parse_planner_instructions(instructions: str) -> Dict[str, Any]:
        """Parse structured instructions from Planner."""
        # Basic parsing implementation
        sections = instructions.split('\n')
        result = {
            'task': '',
            'deliverables': [],
            'constraints': [],
            'prerequisites': []
        }
        current_section = None
        
        for line in sections:
            line = line.strip()
            if line.startswith('- Task:'):
                result['task'] = line[7:].strip()
            elif line.startswith('- Required Deliverables:'):
                current_section = 'deliverables'
            elif line.startswith('- Constraints:'):
                current_section = 'constraints'
            elif line.startswith('- Prerequisites:'):
                current_section = 'prerequisites'
            elif line.startswith('  - ') and current_section:
                result[current_section].append(line[4:])
        
        return result

    @staticmethod
    def parse_executor_feedback(feedback: str) -> Dict[str, Any]:
        """Parse structured feedback from Executor."""
        # Basic parsing implementation
        sections = feedback.split('\n')
        result = {
            'status': '',
            'completion_details': '',
            'blockers': [],
            'resources_needed': [],
            'next_task_ready': False
        }
        current_section = None
        
        for line in sections:
            line = line.strip()
            if line.startswith('- Task Status:'):
                result['status'] = line[13:].strip()
            elif line.startswith('- Completion Details:'):
                result['completion_details'] = line[20:].strip()
            elif line.startswith('- Blockers:'):
                current_section = 'blockers'
            elif line.startswith('- Resources Needed:'):
                current_section = 'resources_needed'
            elif line.startswith('- Next Task Readiness:'):
                result['next_task_ready'] = 'Ready' in line
            elif line.startswith('  - ') and current_section:
                result[current_section].append(line[4:])
        
        return result

class ResearchSession:
    """Manages the research session with Planner and Executor agents."""
    
    def __init__(self, model: str, debug: bool = False):
        """Initialize the research session.
        
        Args:
            model: The OpenAI model to use
            debug: Whether to enable debug mode
        """
        self.model = model
        self.debug = debug
        self.planner = PlannerAgent(model=model)
        self.executor = ExecutorAgent(model=model)
        self.created_files: Set[str] = set()
        self.total_usage = TokenUsage(0, 0, 0, 0.0, 0.0, 0)
        self.agent_communication = AgentCommunication()
        
        # Always create a fresh scratchpad
        with open('scratchpad.md', 'w', encoding='utf-8') as f:
            f.write("")  # Empty file
        self.created_files.add('scratchpad.md')
        logger.info("Created empty scratchpad.md")

    def print_total_usage(self) -> None:
        """Print total token usage statistics."""
        if self.total_usage:
            logger.info("\n=== Total Session Usage ===")
            logger.info(f"Total Input Tokens: {self.total_usage.prompt_tokens:,}")
            logger.info(f"Total Output Tokens: {self.total_usage.completion_tokens:,}")
            logger.info(f"Total Cached Tokens: {self.total_usage.cached_prompt_tokens:,}")
            logger.info(f"Total Tokens: {self.total_usage.total_tokens:,}")
            logger.info(f"Total Cost: ${self.total_usage.total_cost:.6f}")
            logger.info(f"Total Thinking Time: {self.total_usage.thinking_time:.2f}s")
    
    def _get_scratchpad_content(self) -> str:
        """Get current content of scratchpad.md."""
        try:
            with open('scratchpad.md', 'r', encoding='utf-8') as f:
                content = f.read()
                logger.debug(f"Read scratchpad content: {content[:200]}...")
                return content
        except Exception as e:
            logger.error(f"Error reading scratchpad: {e}")
            return ""
    
    def _is_user_input_needed(self, response: str) -> bool:
        """Check if the response indicates need for user input."""
        # Simply check for the standardized marker
        return response.strip().startswith("WAIT_USER_CONFIRMATION")
    
    def chat_loop(self, initial_query: str) -> None:
        """Main chat loop for the research session."""
        current_query = initial_query
        conversation_history = []
        task_complete = False
        
        try:
            while not task_complete:
                # Create planner context
                planner_context = PlannerContext(
                    conversation_history=conversation_history,
                    created_files=self.created_files,
                    user_input=current_query,
                    scratchpad_content=self._get_scratchpad_content(),
                    total_usage=self.total_usage,
                    debug=self.debug
                )
                
                # Get next steps from planner
                next_steps = self.planner.plan(planner_context)
                if not next_steps:
                    logger.error("Planner failed to provide next steps")
                    break
                    
                # Update total usage from planner
                self.total_usage = planner_context.total_usage
                
                # Create executor context
                executor_context = ExecutorContext(
                    created_files=self.created_files,
                    scratchpad_content=self._get_scratchpad_content(),
                    total_usage=self.total_usage,
                    debug=self.debug
                )
                
                # Execute the steps
                result = self.executor.execute(executor_context)
                if not result:
                    logger.error("Executor failed to provide results")
                    break
                    
                # Update total usage from executor
                self.total_usage = executor_context.total_usage
                
                # Check if task is complete
                if result.strip().startswith("TASK_COMPLETE"):
                    logger.info("Task completed successfully")
                    task_complete = True
                    break
                elif result.strip().startswith("WAIT_USER_CONFIRMATION"):
                    user_input = input("\nPlease review and provide feedback (or press Enter to continue, 'q' to quit): ")
                    if user_input.lower() == 'q':
                        break
                    if user_input:
                        current_query = user_input
                        continue
                
                # Update conversation history
                conversation_history.append({"role": "assistant", "content": result})
                
                # Check for errors
                if result.startswith("Error"):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
        finally:
            self.print_total_usage()

def main() -> None:
    """Main function to parse arguments and start the research session."""
    parser = argparse.ArgumentParser(
        description="Interactive research system with Planner and Executor agents."
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
        "--debug",
        action="store_true",
        help="Enable debug mode to save prompts"
    )
    
    args = parser.parse_args()
    
    # Start research session
    session = ResearchSession(model=args.model, debug=args.debug)
    session.chat_loop(args.query)

if __name__ == "__main__":
    main() 