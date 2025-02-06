"""
Common types and utilities shared across the deep research system.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float
    thinking_time: float = 0.0
    cached_prompt_tokens: int = 0

def calculate_cost(prompt_tokens: int, completion_tokens: int, cached_tokens: int, model: str) -> float:
    """Calculate the cost of API usage based on model pricing.
    
    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of completion tokens
        cached_tokens: Number of cached tokens (charged at half price)
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
    
    # Calculate regular input cost (excluding cached tokens)
    regular_input_tokens = prompt_tokens - cached_tokens
    regular_input_cost = (regular_input_tokens / 1_000_000) * pricing["input"]
    
    # Calculate cached tokens cost (half price)
    cached_cost = (cached_tokens / 1_000_000) * (pricing["input"] / 2)
    
    # Calculate output cost
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    
    return regular_input_cost + cached_cost + output_cost 