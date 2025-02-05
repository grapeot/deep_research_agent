"""
Tool implementations for interactive chat system.
Contains search, web scraping, file operations, and package management tools.
"""

import asyncio
import logging
import os
import subprocess
import time
import hashlib
import json
from multiprocessing import Pool
from typing import List, Optional, Union, Dict
from urllib.parse import urlparse

from playwright.async_api import async_playwright
import html5lib
from duckduckgo_search import DDGS
import openai

logger = logging.getLogger(__name__)

class CachedChatCompletion:
    """Handles chat completions with caching to optimize token usage and latency."""
    
    # Model pricing per 1M tokens
    MODEL_PRICING = {
        "o1": {
            "input": 15.0,
            "cached_input": 7.50,
            "output": 60.0
        },
        "o3-mini": {
            "input": 1.10,
            "cached_input": 0.55,
            "output": 4.40
        }
    }
    
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.client = openai.OpenAI()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_prompt_tokens = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int, model: str) -> float:
        """Calculate the cost of API usage based on model pricing."""
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["o3-mini"])
        
        # Convert to millions and calculate
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        cached_cost = (cached_tokens / 1_000_000) * pricing["cached_input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + cached_cost + output_cost
    
    def hash_messages(self, data: Dict) -> str:
        """Create a deterministic hash of the messages and function definitions."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "o3-mini",
        temperature: float = 1.0,
        max_completion_tokens: Optional[int] = 1000,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Union[str, Dict]] = None,
        reasoning_effort: str = 'high',
    ) -> openai.types.chat.ChatCompletion:
        """Get chat completion with caching."""
        # Create cache key from all parameters
        cache_data = {
            "messages": messages,
            "functions": functions,
            "function_call": function_call,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort
        }
        cache_key = self.hash_messages(cache_data)
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            cached_result = self.cache[cache_key]
            self.total_cached_prompt_tokens += cached_result.usage.prompt_tokens
            # Calculate cost for cached request
            step_cost = self.calculate_cost(
                prompt_tokens=0,  # No new input tokens
                completion_tokens=0,  # No new completion tokens
                cached_tokens=cached_result.usage.prompt_tokens,
                model=model
            )
            self.total_cost += step_cost
            return cached_result
        
        self.cache_misses += 1
        
        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort
        }
        if functions:
            params["functions"] = functions
        if function_call:
            params["function_call"] = function_call
            
        # Make API call
        response = self.client.chat.completions.create(**params)
        
        # Update token counts and calculate cost
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        step_cost = self.calculate_cost(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cached_tokens=0,  # No cached tokens for new request
            model=model
        )
        self.total_cost += step_cost
        
        # Cache the response
        self.cache[cache_key] = response
        return response
    
    def get_token_usage(self) -> Dict[str, Union[int, float]]:
        """Get current token usage statistics and cost."""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cached_prompt_tokens": self.total_cached_prompt_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens + self.total_cached_prompt_tokens,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_cost": self.total_cost
        }

# Initialize global chat completion instance
chat_completion = CachedChatCompletion()

# Search Engine Implementation
def search_with_retry(query: str, max_results: int = 10, max_retries: int = 3) -> List[dict]:
    """
    Search using DuckDuckGo and return results with URLs and text snippets.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        max_retries: Maximum number of retry attempts

    Returns:
        List of dictionaries containing search results
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Searching for query: {query} (attempt {attempt + 1}/{max_retries})")
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
            if not results:
                logger.info("No results found")
                return []
            
            logger.info(f"Found {len(results)} results")
            return results
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Waiting 1 second before retry...")
                time.sleep(1)
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise

def format_search_results(results: List[dict]) -> str:
    """
    Format search results into a readable string.

    Args:
        results: List of search result dictionaries

    Returns:
        Formatted string containing search results
    """
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\n=== Result {i} ===")
        output.append(f"URL: {result.get('href', 'N/A')}")
        output.append(f"Title: {result.get('title', 'N/A')}")
        output.append(f"Snippet: {result.get('body', 'N/A')}")
    return "\n".join(output)

# Web Scraper Implementation
async def fetch_page(url: str, context) -> Optional[str]:
    """
    Asynchronously fetch a webpage's content using Playwright.

    Args:
        url: URL to fetch
        context: Playwright browser context

    Returns:
        Page content as string if successful, None otherwise
    """
    page = await context.new_page()
    try:
        logger.info(f"Fetching {url}")
        await page.goto(url)
        await page.wait_for_load_state('networkidle')
        content = await page.content()
        logger.info(f"Successfully fetched {url}")
        return content
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None
    finally:
        await page.close()

def parse_html(html_content: Optional[str]) -> str:
    """
    Parse HTML content and extract text with hyperlinks in markdown format.

    Args:
        html_content: HTML content to parse

    Returns:
        Extracted text in markdown format
    """
    if not html_content:
        return ""
    
    try:
        document = html5lib.parse(html_content)
        result = []
        seen_texts = set()
        
        def should_skip_element(elem) -> bool:
            """Check if the element should be skipped during parsing."""
            if elem.tag in ['{http://www.w3.org/1999/xhtml}script', 
                          '{http://www.w3.org/1999/xhtml}style']:
                return True
            if not any(text.strip() for text in elem.itertext()):
                return True
            return False
        
        def process_element(elem, depth: int = 0) -> None:
            """Process an HTML element and its children recursively."""
            if should_skip_element(elem):
                return
            
            if hasattr(elem, 'text') and elem.text:
                text = elem.text.strip()
                if text and text not in seen_texts:
                    if elem.tag == '{http://www.w3.org/1999/xhtml}a':
                        href = None
                        for attr, value in elem.items():
                            if attr.endswith('href'):
                                href = value
                                break
                        if href and not href.startswith(('#', 'javascript:')):
                            link_text = f"[{text}]({href})"
                            result.append("  " * depth + link_text)
                            seen_texts.add(text)
                    else:
                        result.append("  " * depth + text)
                        seen_texts.add(text)
            
            for child in elem:
                process_element(child, depth + 1)
            
            if hasattr(elem, 'tail') and elem.tail:
                tail = elem.tail.strip()
                if tail and tail not in seen_texts:
                    result.append("  " * depth + tail)
                    seen_texts.add(tail)
        
        body = document.find('.//{http://www.w3.org/1999/xhtml}body')
        if body is not None:
            process_element(body)
        else:
            process_element(document)
        
        filtered_result = []
        for line in result:
            if any(pattern in line.lower() for pattern in [
                'var ', 
                'function()', 
                '.js',
                '.css',
                'google-analytics',
                'disqus',
                '{',
                '}'
            ]):
                continue
            filtered_result.append(line)
        
        return '\n'.join(filtered_result)
    except Exception as e:
        logger.error(f"Error parsing HTML: {str(e)}")
        return ""

async def process_urls(urls: List[str], max_concurrent: int = 5) -> List[str]:
    """
    Process multiple URLs concurrently using Playwright.

    Args:
        urls: List of URLs to process
        max_concurrent: Maximum number of concurrent requests

    Returns:
        List of processed content strings
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            # Create browser contexts
            n_contexts = min(len(urls), max_concurrent)
            contexts = [await browser.new_context() for _ in range(n_contexts)]
            
            # Create tasks for each URL
            tasks = []
            for i, url in enumerate(urls):
                context = contexts[i % len(contexts)]
                task = fetch_page(url, context)
                tasks.append(task)
            
            # Gather results
            html_contents = await asyncio.gather(*tasks)
            
            # Parse HTML contents in parallel
            with Pool() as pool:
                results = pool.map(parse_html, html_contents)
                
            return results
            
        finally:
            # Cleanup
            for context in contexts:
                await context.close()
            await browser.close()

def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# Main Tool Functions
def perform_search(query: str, max_results: int = 5, max_retries: int = 3) -> str:
    """
    Perform a web search and return formatted results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        max_retries: Maximum number of retry attempts

    Returns:
        Formatted string containing search results or error message
    """
    try:
        results = search_with_retry(query, max_results, max_retries)
        return format_search_results(results)
    except Exception as e:
        return f"Error during search: {e}"

def fetch_web_content(urls: List[str], max_concurrent: int = 3) -> str:
    """
    Fetch and process web content from multiple URLs using Playwright.

    Args:
        urls: List of URLs to fetch and process
        max_concurrent: Maximum number of concurrent requests

    Returns:
        Formatted string containing processed content or error message
    """
    try:
        # Validate URLs
        valid_urls = [url for url in urls if validate_url(url)]
        if not valid_urls:
            return "No valid URLs provided"
        
        # Process URLs
        results = asyncio.run(process_urls(valid_urls, max_concurrent))
        
        # Format output
        output = []
        for url, content in zip(valid_urls, results):
            output.append(f"\n=== Content from {url} ===\n")
            output.append(content)
        
        return "\n".join(output)
    except Exception as e:
        return f"Error during web scraping: {e}"

def create_file(filename: str, content: str) -> str:
    """
    Create a file with the given content and return its content.

    Args:
        filename: Name of the file to create
        content: Content to write to the file

    Returns:
        File content after writing or error message
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        # Read back the content to confirm
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error creating file: {str(e)}"

def execute_python(filename: str) -> str:
    """
    Execute a Python script and return its stdout.

    Args:
        filename: Name of the Python file to execute

    Returns:
        Script output or error message
    """
    try:
        result = subprocess.run(
            ["python", filename],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing Python script: stdout={e.stdout}, stderr={e.stderr}"
    except Exception as e:
        return f"Error executing Python script: {str(e)}" 