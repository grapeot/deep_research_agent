"""
Common types and utilities shared across the deep research system.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float
    thinking_time: float = 0.0
    cached_prompt_tokens: int = 0

class TokenTracker:
    """Centralized token usage and cost tracking."""
    
    # Model pricing per 1M tokens
    MODEL_PRICING = {
        # OpenAI models
        "o1": {
            "input": 15.0,
            "output": 60.0
        },
        "o3-mini": {
            "input": 1.10,
            "output": 4.40
        },
        "gpt-4o": {
            "input": 2.50,
            "output": 10.00
        },
        # Claude models
        "claude-3-7-sonnet-20250219": {
            "input": 3.00,
            "output": 15.00
        },
        "claude-3-5-sonnet": {
            "input": 1.50,
            "output": 6.00
        },
        "claude-3-haiku": {
            "input": 0.25,
            "output": 1.25
        },
        "claude-3-opus": {
            "input": 15.00,
            "output": 75.00
        },
        "claude-3-sonnet": {
            "input": 3.00,
            "output": 15.00
        }
    }
    
    def __init__(self):
        self.total_usage = TokenUsage(0, 0, 0, 0.0, 0.0, 0)
    
    @classmethod
    def calculate_cost(cls, prompt_tokens: int, completion_tokens: int, cached_tokens: int, model: str) -> float:
        """Calculate the cost of API usage based on model pricing."""
        # Default to o3-mini pricing if model not found
        pricing = cls.MODEL_PRICING.get(model, cls.MODEL_PRICING["o3-mini"])
        
        # For Claude models, we don't have cached token discounts
        if model.startswith("claude"):
            regular_input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
            output_cost = (completion_tokens / 1_000_000) * pricing["output"]
            return regular_input_cost + output_cost
        
        # For OpenAI models with cached token discounts
        regular_input_tokens = prompt_tokens - cached_tokens
        regular_input_cost = (regular_input_tokens / 1_000_000) * pricing["input"]
        
        # Calculate cached tokens cost (half price)
        cached_cost = (cached_tokens / 1_000_000) * (pricing["input"] / 2)
        
        # Calculate output cost
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return regular_input_cost + cached_cost + output_cost
    
    def update_usage(self, usage: Dict[str, int], thinking_time: float, model: str):
        """Update total usage with new API call results."""
        cached_tokens = usage.get('cached_prompt_tokens', 0)
        cost = self.calculate_cost(
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
            cached_tokens=cached_tokens,
            model=model
        )
        
        # Update totals
        self.total_usage.prompt_tokens += usage['prompt_tokens']
        self.total_usage.completion_tokens += usage['completion_tokens']
        self.total_usage.cached_prompt_tokens += cached_tokens
        self.total_usage.total_tokens = self.total_usage.prompt_tokens + self.total_usage.completion_tokens
        self.total_usage.total_cost += cost
        self.total_usage.thinking_time += thinking_time
        
        # Log the usage
        self._log_usage(usage, thinking_time, cost)
        
        return cost

    def update_from_token_usage(self, token_usage: TokenUsage):
        """Update total usage from a TokenUsage object."""
        # Simply add the values
        self.total_usage.prompt_tokens += token_usage.prompt_tokens
        self.total_usage.completion_tokens += token_usage.completion_tokens
        self.total_usage.cached_prompt_tokens += token_usage.cached_prompt_tokens
        self.total_usage.total_tokens = self.total_usage.prompt_tokens + self.total_usage.completion_tokens
        self.total_usage.total_cost += token_usage.total_cost
        self.total_usage.thinking_time += token_usage.thinking_time
    
    def _log_usage(self, usage: Dict[str, int], thinking_time: float, cost: float):
        """Log token usage and cost information."""
        cached_tokens = usage.get('cached_prompt_tokens', 0)
        
        logger.info("\nToken Usage:")
        logger.info(f"Input tokens: {usage['prompt_tokens']:,}")
        logger.info(f"Output tokens: {usage['completion_tokens']:,}")
        logger.info(f"Cached tokens: {cached_tokens:,}")
        logger.info(f"Total tokens: {usage['total_tokens']:,}")
        logger.info(f"Cost: ${cost:.6f}")
        logger.info(f"Thinking time: {thinking_time:.2f}s")
    
    def get_total_usage(self) -> TokenUsage:
        """Get current total token usage statistics."""
        return self.total_usage
    
    def print_total_usage(self):
        """Print total token usage statistics."""
        logger.info("\n=== Total Session Usage ===")
        logger.info(f"Total Input Tokens: {self.total_usage.prompt_tokens:,}")
        logger.info(f"Total Output Tokens: {self.total_usage.completion_tokens:,}")
        logger.info(f"Total Cached Tokens: {self.total_usage.cached_prompt_tokens:,}")
        logger.info(f"Total Tokens: {self.total_usage.total_tokens:,}")
        logger.info(f"Total Cost: ${self.total_usage.total_cost:.6f}")
        logger.info(f"Total Thinking Time: {self.total_usage.thinking_time:.2f}s") 