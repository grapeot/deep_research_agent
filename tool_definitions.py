"""
Function definitions and descriptions for the interactive chat system tools.
"""

function_definitions = [
    {
        "name": "perform_search",
        "description": "Perform a web search using our native search tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum search results",
                    "default": 10
                },
                "max_retries": {
                    "type": "integer",
                    "description": "Maximum number of retry attempts",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_web_content",
        "description": "Fetch the content of web pages using our web scraper tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to scrape"
                },
                "max_concurrent": {
                    "type": "integer",
                    "description": "Maximum number of concurrent requests",
                    "default": 3
                }
            },
            "required": ["urls"]
        }
    },
    {
        "name": "create_file",
        "description": "Create or change a file with the given content and return its content for verification.",
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
    },
    {
        "name": "execute_command",
        "description": "Execute a terminal command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The terminal command to execute"
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of what the command does"
                }
            },
            "required": ["command", "explanation"]
        }
    }
] 