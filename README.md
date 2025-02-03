# Deep Research Agent

A document-driven agentic AI research system that helps conduct comprehensive analysis through persistent context management and tool integration.

## Core Philosophy

1. **Document-Centric Memory**: Uses persistent documents to maintain context and track progress, solving the fundamental context window limitation of language models.
2. **Structured Communication**: Documents all information in a shared scratchpad, ensuring no critical information is lost.
3. **Tool Augmentation**: Leverages specialized tools for up-to-date information gathering and analysis.
4. **User Agency**: Acts as a collaborative partner, keeping users in control through clear documentation and decision points.

## Getting Started

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run a research query
python3 deep_research_agent.py "your research query"
```

## Available Tools

1. **Web Search & Analysis**: Semantic search and content analysis with source attribution
2. **File Operations**: Create/edit files and execute scripts with user confirmation
3. **Package Management**: Install and manage Python packages with user confirmation
4. **Terminal Commands**: Execute system commands with user confirmation

## Example Usage

Analyzing NVIDIA's recent stock performance:

```bash
python3 deep_research_agent.py "Perform a detailed analysis on the recent trend of NVDA stock. How did the stock price change? What might have caused it? How about the market sentiment?"
```

The agent:
1. Creates a research plan in `scratchpad.md`
2. Generates and executes analysis script (with user confirmation):
   ```
   Confirm execution of command: python3 nvda_analysis.py
   Explanation: Executing Python script: nvda_analysis.py
   [y/N]: 
   ```
3. Produces comprehensive analysis in `nvda_analysis_report.md`

Example outputs in [examples/](examples/):
- [Analysis Report](examples/nvda_analysis_report.md)
- [Price Trend Chart](examples/nvda_trend.png)
- [Analysis Script](examples/nvda_analysis.py)

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

*Built on practical experience with context window management and real-world research needs.* 