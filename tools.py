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
import requests
from bs4 import BeautifulSoup

from common import TokenTracker

logger = logging.getLogger(__name__)

class CachedChatCompletion:
    """Handles chat completions with token usage tracking."""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.token_tracker = TokenTracker()
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 1.0,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Union[str, Dict]] = None,
        reasoning_effort: str = 'high',
    ) -> openai.types.chat.ChatCompletion:
        """Get chat completion with OpenAI's built-in caching."""
        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        # Add reasoning_effort only for models starting with 'o'
        if model.startswith('o'):
            params["reasoning_effort"] = reasoning_effort
            
        if functions:
            params["functions"] = functions
        if function_call:
            params["function_call"] = function_call
            
        # Make API call
        response = self.client.chat.completions.create(**params)
        
        # Update token usage tracking
        usage = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
            'cached_prompt_tokens': getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) if hasattr(response.usage, 'prompt_tokens_details') else 0
        }
        
        # Calculate thinking time and update usage
        thinking_time = 0.0  # This should be passed in from the agent
        self.token_tracker.update_usage(usage, thinking_time, model)
        
        return response
    
    def get_token_usage(self) -> Dict[str, Union[int, float]]:
        """Get current token usage statistics."""
        usage = self.token_tracker.get_total_usage()
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cached_prompt_tokens": usage.cached_prompt_tokens,
            "total_tokens": usage.total_tokens,
            "total_cost": usage.total_cost
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
            if attempt < max_retries - 1:  # If not the last attempt
                logger.info("Waiting 1 second before retry...")
                time.sleep(1)  # Wait 1 second before retry
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

def perform_search(query: str, max_results: int = 10, max_retries: int = 3) -> str:
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
    Create a file with the given content and return a success message.

    Args:
        filename: Name of the file to create
        content: Content to write to the file

    Returns:
        Success message or error message
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully created/updated file: {filename}"
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