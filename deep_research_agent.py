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
from common import TokenUsage, TokenTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,  # Default to INFO level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deep Research Agent')
    parser.add_argument('query', help='The research query to process')
    parser.add_argument('--model', default='gpt-4o', help='The OpenAI model to use for the planner (default: gpt-4o)')
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
    
    def __init__(self, planner_model: str, executor_model: str = None, debug: bool = False):
        """Initialize the research session.
        
        Args:
            planner_model: The OpenAI model to use for Planner agent
            executor_model: Not used, as executor always uses Claude 3.7
            debug: Whether to enable debug mode
        """
        self.planner_model = planner_model
        self.debug = debug
        
        # Set up debug logging if enabled
        if debug:
            logging.getLogger('tools').setLevel(logging.DEBUG)
            logging.getLogger('executor_agent').setLevel(logging.DEBUG)
            logging.getLogger('planner_agent').setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Initialize Planner with OpenAI model
        self.planner = PlannerAgent(model=planner_model)
        
        # Initialize Executor with Claude 3.7 (executor_model param is ignored)
        self.executor = ExecutorAgent(model="claude-3-7-sonnet-20250219")
        
        self.created_files: Set[str] = set()
        self.token_tracker = TokenTracker()
        self.agent_communication = AgentCommunication()
        
        # Initialize scratchpad with required sections
        self._initialize_scratchpad()
        self.created_files.add('scratchpad.md')
        logger.info("Created scratchpad.md with initial sections")

    def _initialize_scratchpad(self) -> None:
        """Initialize scratchpad.md with the required sections."""
        initial_content = """### Background and Motivation
(Planner writes: User/business requirements, macro objectives, why this problem needs to be solved)

### Key Challenges and Analysis
(Planner: Records of technical barriers, resource constraints, potential risks)

### Verifiable Success Criteria
(Planner: List measurable or verifiable goals to be achieved)

### High-level Task Breakdown
(Planner: List subtasks by phase, or break down into modules)

### Current Status / Progress Tracking
(Executor: Update completion status after each subtask. If needed, use bullet points or tables to show Done/In progress/Blocked status)

### Next Steps and Action Items
(Planner: Specific arrangements for the Executor)

### Executor's Feedback or Assistance Requests
(Executor: Write here when encountering blockers, questions, or need for more information during execution)
"""
        with open('scratchpad.md', 'w', encoding='utf-8') as f:
            f.write(initial_content)

    def _update_scratchpad_section(self, section_name: str, content: str, role: str = "Planner") -> None:
        """Update a specific section in the scratchpad.
        
        Args:
            section_name: Name of the section to update (without '###')
            content: New content to append to the section
            role: Role making the update ('Planner' or 'Executor')
        """
        try:
            current_content = self._get_scratchpad_content()
            sections = current_content.split('\n### ')
            
            # Find the target section
            target_section_idx = -1
            for i, section in enumerate(sections):
                if section.startswith(section_name) or section.startswith('### ' + section_name):
                    target_section_idx = i
                    break
            
            if target_section_idx == -1:
                logger.error(f"Section '{section_name}' not found in scratchpad")
                return
                
            # Format the new content with timestamp and role
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            formatted_content = f"\n[{role} @ {timestamp}]\n{content.strip()}\n"
            
            # Append the new content to the section
            if target_section_idx == 0:
                sections[0] = sections[0] + formatted_content
            else:
                sections[target_section_idx] = sections[target_section_idx] + formatted_content
            
            # Reconstruct the document
            updated_content = sections[0]
            for section in sections[1:]:
                updated_content += '\n### ' + section
            
            # Write back to file
            with open('scratchpad.md', 'w', encoding='utf-8') as f:
                f.write(updated_content)
                
            logger.debug(f"Updated section '{section_name}' in scratchpad")
            
        except Exception as e:
            logger.error(f"Error updating scratchpad section: {e}")

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
            # Initialize Background and Motivation with the initial query
            self._update_scratchpad_section(
                "Background and Motivation",
                f"Initial research query: {initial_query}",
                "Planner"
            )
            
            while not task_complete:
                logger.info("=== Control Flow: Starting new planning cycle ===")
                # Create planner context
                planner_context = PlannerContext(
                    conversation_history=conversation_history,
                    created_files=self.created_files,
                    user_input=current_query,
                    scratchpad_content=self._get_scratchpad_content(),
                    total_usage=self.token_tracker.total_usage,
                    debug=self.debug
                )
                
                # Get next steps from planner
                logger.info("=== Control Flow: Requesting next steps from Planner ===")
                next_steps = self.planner.plan(planner_context)
                if not next_steps:
                    logger.error("Planner failed to provide next steps")
                    break
                
                logger.info(f"Planner next steps:\n{next_steps}")
                
                # Update global token tracker with planner's usage
                if planner_context.total_usage:
                    logger.debug("Updating global token tracker with planner usage")
                    self.token_tracker.update_from_token_usage(planner_context.total_usage)
                    # Clear the context usage to avoid double counting
                    planner_context.total_usage = None
                
                # Check if planner indicates task completion
                if next_steps.strip().startswith("TASK_COMPLETE"):
                    logger.info("Planner indicates task is complete")
                    self._update_scratchpad_section(
                        "Current Status / Progress Tracking",
                        "Task completed successfully - Waiting for final user feedback",
                        "Planner"
                    )
                    
                    # Request final user feedback
                    user_input = input("\nTask completed. Please provide any additional feedback or press Enter to finish (or 'q' to quit): ")
                    if user_input.lower() == 'q':
                        break
                    if user_input:
                        current_query = user_input
                        self._update_scratchpad_section(
                            "Current Status / Progress Tracking",
                            f"Received additional user feedback after completion: {user_input} - Continuing task",
                            "Planner"
                        )
                        continue
                    
                    # If no additional feedback, mark as complete and break
                    self._update_scratchpad_section(
                        "Current Status / Progress Tracking",
                        "Task completed and confirmed by user",
                        "Planner"
                    )
                    task_complete = True
                    break
                
                # If not complete, proceed with execution
                logger.info("=== Control Flow: Transferring control to Executor ===")
                
                # Create executor context
                executor_context = ExecutorContext(
                    created_files=self.created_files,
                    scratchpad_content=self._get_scratchpad_content(),
                    total_usage=self.token_tracker.total_usage,
                    debug=self.debug
                )
                
                # Execute the steps
                result = self.executor.execute(executor_context)
                if not result:
                    logger.error("Executor failed to provide results")
                    self._update_scratchpad_section(
                        "Executor's Feedback or Assistance Requests",
                        "Execution failed: No results provided",
                        "Executor"
                    )
                    break
                    
                logger.info(f"Executor result:\n{result}")
                
                # Update global token tracker with executor's usage
                if executor_context.total_usage:
                    logger.debug("Updating global token tracker with executor usage")
                    self.token_tracker.update_from_token_usage(executor_context.total_usage)
                    # Clear the context usage to avoid double counting
                    executor_context.total_usage = None
                
                # Handle user input requests
                if result.strip().startswith("WAIT_USER_CONFIRMATION"):
                    self._update_scratchpad_section(
                        "Current Status / Progress Tracking",
                        "Waiting for user confirmation",
                        "Executor"
                    )
                    user_input = input("\nPlease review and provide feedback (or press Enter to continue, 'q' to quit): ")
                    if user_input.lower() == 'q':
                        break
                    if user_input:
                        current_query = user_input
                        self._update_scratchpad_section(
                            "Current Status / Progress Tracking",
                            f"Received user feedback: {user_input}",
                            "Executor"
                        )
                        continue
                
                # Update conversation history
                conversation_history.append({"role": "assistant", "content": result})
                
                # Check for errors
                if result.startswith("Error"):
                    self._update_scratchpad_section(
                        "Executor's Feedback or Assistance Requests",
                        f"Error encountered: {result}",
                        "Executor"
                    )
                    break
                    
                logger.info("=== Control Flow: Executor completed, returning control to Planner ===")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self._update_scratchpad_section(
                "Current Status / Progress Tracking",
                "Task interrupted by user",
                "Planner"
            )
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            self._update_scratchpad_section(
                "Current Status / Progress Tracking",
                f"Error occurred: {str(e)}",
                "Planner"
            )
        finally:
            self.print_total_usage()

    def print_total_usage(self) -> None:
        """Print total token usage statistics."""
        self.token_tracker.print_total_usage()

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
        "--planner-model",
        default="o1",
        help="OpenAI model to use for Planner agent (default: o1)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to save prompts"
    )
    
    args = parser.parse_args()
    
    # Start research session
    logger.info(f"Starting research session with Planner model: {args.planner_model} and Executor: Claude 3.7")
    session = ResearchSession(
        planner_model=args.planner_model,
        debug=args.debug
    )
    session.chat_loop(args.query)

if __name__ == "__main__":
    main() 