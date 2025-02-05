#!/usr/bin/env python3
"""
A simple test script to demonstrate prompt caching with token counting.
Based on: https://cookbook.openai.com/examples/prompt_caching101
"""

import openai
import hashlib
import json
from typing import Dict, List, Optional, Union
import time

def hash_messages(messages: List[Dict[str, str]]) -> str:
    """Create a deterministic hash of the messages."""
    messages_str = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(messages_str.encode()).hexdigest()

class CachedChatCompletion:
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.client = openai.OpenAI()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_prompt_tokens = 0  # Track cached prompt tokens
        self.cache_hits = 0
        self.cache_misses = 0
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "o3-mini",
        temperature: float = 1.0,
        max_completion_tokens: Optional[int] = 1000,
    ) -> Dict:
        """Get chat completion with caching."""
        cache_key = hash_messages(messages)
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            cached_result = self.cache[cache_key]
            # Add cached prompt tokens to total
            self.total_cached_prompt_tokens += cached_result["usage"]["prompt_tokens"]
            return cached_result
        
        self.cache_misses += 1
        
        # Make API call
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        
        # Convert response to dict for caching
        response_dict = {
            "id": response.id,
            "choices": [{"message": {"content": choice.message.content}} for choice in response.choices],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
        
        # Update token counts
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        
        # Cache the response
        self.cache[cache_key] = response_dict
        return response_dict

def main():
    cached_chat = CachedChatCompletion()
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    # Make same request multiple times
    for i in range(3):
        print(f"\nRequest {i+1}:")
        start_time = time.time()
        response = cached_chat.chat_completion(messages)
        end_time = time.time()
        
        print(f"Response: {response['choices'][0]['message']['content']}")
        print(f"Time taken: {end_time - start_time:.2f}s")
    
    # Print statistics
    print("\nCache Statistics:")
    print(f"Cache hits: {cached_chat.cache_hits}")
    print(f"Cache misses: {cached_chat.cache_misses}")
    print("\nToken Usage:")
    print(f"Total prompt tokens (API calls): {cached_chat.total_prompt_tokens}")
    print(f"Total completion tokens: {cached_chat.total_completion_tokens}")
    print(f"Total cached prompt tokens: {cached_chat.total_cached_prompt_tokens}")
    print(f"Total tokens (including cached): {cached_chat.total_prompt_tokens + cached_chat.total_completion_tokens + cached_chat.total_cached_prompt_tokens}")

if __name__ == "__main__":
    main() 