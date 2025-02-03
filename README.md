# Deep Research Agent

A research-focused agentic AI system that helps you conduct comprehensive analysis and research with a combination of tools and methodologies.

## Background

In early 2025, we've seen significant developments in agentic AI systems, with major players like OpenAI's Deep Research and Google's Gemini 1.5 introducing powerful research capabilities. While these systems demonstrate the potential of AI in research assistance, Deep Research Agent takes a unique approach based on our understanding of agentic AI and real-world research needs, particularly in context window management and document-driven communication.

## Core Philosophy

Deep Research Agent is built on several key principles derived from practical experience with context window management and multi-agent systems:

1. **Document-Centric Memory**: Instead of relying on volatile conversation history, the system uses persistent documents to maintain context and track progress. This approach solves the fundamental context window limitation of language models.

2. **Structured Communication**: All important information, including plans, progress, and results, is documented in a shared scratchpad. This ensures no critical information is lost during the research process and allows for easy review and verification.

3. **Tool Augmentation**: Instead of relying solely on pre-trained knowledge, the system leverages specialized tools for up-to-date and accurate information, documenting each tool's usage and results.

4. **User Agency**: The system acts as a collaborative partner rather than an autonomous agent, keeping the user in control through clear documentation and decision points.

## Features & Implementation

- **Document-Driven Communication**: Uses persistent documents (like scratchpad) as the primary communication channel
- **Tool-First Approach**: Focuses on utilizing a carefully curated set of tools rather than trying to be a know-it-all system
- **Structured Planning**: Every research task begins with a clear plan, tracked in a scratch pad
- **Source Tracking**: All research conclusions are accompanied by their sources

The system addresses several key challenges:
1. **Context Window Management**: Using persistent documents to avoid LLM context limitations
2. **Memory Persistence**: Preventing information loss from context truncation
3. **Progress Tracking**: Documenting all steps for reference and review
4. **Reproducibility**: Ensuring research can be reproduced or audited

## Getting Started

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run a research query
python3 deep_research_agent.py "your research query"

# Optional: Specify different model or config
python3 deep_research_agent.py --model "gpt-4" --config "custom_rules.txt" "your query"
```

## System Components

- `deep_research_agent.py`: Main script handling user interaction and tool orchestration
- `tools.py`: Collection of specialized tools for various research tasks
- `tool_definitions.py`: Definitions and parameters for available tools
- `.deep_research_rules`: Configuration file containing system prompts and behavior guidelines
- `scratchpad.txt`: Persistent document for tracking research progress

## Available Tools

1. **Web Search**: Semantic search capabilities using DuckDuckGo for gathering up-to-date information
2. **Web Content Analysis**: Fetch and analyze web content with proper source attribution and HTML parsing
3. **File Operations**: 
   - Create and edit files with content verification
   - Execute Python scripts with user confirmation
   - Manage research artifacts and reports
4. **Package Management**: Install and manage Python packages with user confirmation
5. **Terminal Commands**: Execute system commands with user confirmation and proper explanation

Each tool maintains context through document-driven communication, ensuring all operations are tracked and reproducible.

## Use Cases

- Market Research and Analysis
- Technical Documentation
- Academic Literature Review
- Trend Analysis
- Data Visualization and Reporting
- Source Verification and Fact-Checking

## Future Directions

- Enhanced document management and version control
- Improved context window optimization
- Expanding the tool ecosystem
- Integration with specialized research databases
- Collaborative research features

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Example Usage

Here's an example of using Deep Research Agent to analyze NVIDIA's recent stock performance:

```bash
python deep_research_agent.py "Perform a detailed analysis on the recent (latest 1 month) trend of NVDA stock. How did the stock price change? Visualize it. Did you notice any sudden / big moves? What might have caused it? How about the market sentiment?"
```

The agent follows a structured research process:

1. Creates `scratchpad.md` to plan the analysis:
   ```markdown
   # Analysis Plan
   1. Fetch NVDA stock data for the past month
   2. Create visualization of price movements
   3. Identify significant price changes
   4. Research market events and sentiment
   5. Compile comprehensive report
   ```

2. Generates `nvda_analysis.py` for data analysis and visualization.

   The agent will ask for confirmation before executing the script:
   ```
   Confirm execution of command: python3 nvda_analysis.py
   Explanation: Executing Python script: nvda_analysis.py
   [y/N]: 
   ```

3. Produces `nvda_analysis_report.md` with comprehensive analysis:

Example output files are available in the [examples](examples/) directory:
- [Analysis Report](examples/nvda_analysis_report.md) - Comprehensive stock analysis
- [Price Trend Chart](examples/nvda_trend.png) - Visual representation of stock movement
- [Analysis Script](examples/nvda_analysis.py) - Data processing and visualization code
- [Research Plan](examples/scratchpad.md) - Document-driven planning process

This example demonstrates key features:
- Document-driven research workflow
- Data visualization and analysis
- Market sentiment analysis
- Source verification and fact-checking
- Multi-tool integration (yfinance, matplotlib, web search)
- User control through execution confirmations

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

*Note: This project represents an independent approach to research-focused AI agents, built on practical experience with context window management and real-world research needs.* 