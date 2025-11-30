# Stock Rating Workflow V2

Enhanced workflow using proper OpenAI Agents SDK MCP integration.

## Key Improvements

**Proper MCP Integration:**
- Uses `HostedMCPTool` to connect to yfinance server
- Agent automatically discovers all 24 tools from MCP server
- No manual tool wrapping needed

**Advanced Features:**
- `AgentHooks` for lifecycle monitoring (on_run_start, on_tool_call, on_error)
- `WebSearchTool` for additional market context
- Structured outputs with Pydantic
- Quality gates between phases
- Auto-approval for tool calls

**Streamlined Flow:**
1. Stock Analyst Agent (with MCP tools) → StockAnalysis
2. Portfolio Manager Agent → FinalRating

## Setup

```bash
# Install
pip install openai-agents

# Set API key
export OPENAI_API_KEY=your_key

# Run yfinance MCP server
uvicorn main:app --host 0.0.0.0 --port 10000 --reload

# Run workflow
python stock_rating_workflow_v2.py
```

## Features

- **Lifecycle Hooks**: Real-time monitoring of agent execution
- **Web Search**: Supplement MCP data with web context
- **Guardrails**: Input validation and quality checks
- **Structured Outputs**: Type-safe with Pydantic models
- **Export**: Save JSON reports with timestamp

## vs V1

**V1**: Manual FastMCP Client calls, 4 agents, verbose
**V2**: HostedMCPTool auto-discovery, 2 agents, hooks, web search

V2 is cleaner, more maintainable, and leverages SDK features properly.
