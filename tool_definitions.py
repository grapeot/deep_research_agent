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
        "name": "execute_python",
        "description": "Execute a Python script and return its stdout. The script should already exist, e.g. it was created by the create_file tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the Python file to execute"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "install_python_package",
        "description": "Install Python packages using pip. Can specify version requirements.",
        "parameters": {
            "type": "object",
            "properties": {
                "packages": {
                    "oneOf": [
                        {
                            "type": "string",
                            "description": "Single package name with optional version specifier (e.g. 'pandas>=2.0.0')"
                        },
                        {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of package names with optional version specifiers"
                        }
                    ],
                    "description": "Package(s) to install"
                },
                "upgrade": {
                    "type": "boolean",
                    "description": "Whether to upgrade the package if it's already installed",
                    "default": False
                }
            },
            "required": ["packages"]
        }
    }
] 