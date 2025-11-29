# yfinance MCP Server

Stock market data API server built with yfinance and FastMCP, featuring authentication, caching, and monitoring.

## Features

- 📊 Real-time stock quotes & historical data
- 📈 Financial statements & fundamentals
- 📰 News & analyst recommendations
- 🔐 Header-based authentication
- 💾 Smart caching (20s TTL)
- 📝 Structured logging
- ❤️ Health check endpoint
- 🌐 CORS support
- 📊 Built-in metrics

## Quick Start

### Local Development

```bash
# Install dependencies
uv sync

# Run in local stdio mode (for MCP client development)
python main.py --local

# Run in HTTP mode for testing
export PORT=8080
python main.py
```

### Production Deployment

```bash
# Using uvicorn (recommended for production)
uvicorn main:app --host 0.0.0.0 --port 8080

# With multiple workers for better performance
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```

### Deploy to Render

See [DEPLOY.md](DEPLOY.md) for detailed deployment instructions.

## API Endpoints

All tools are available as MCP tools:

- `health` - Server status and metrics
- `quote` - Get current stock price
- `history` - Historical OHLCV data
- `fundamentals` - Company fundamentals
- `financials` - Financial statements
- `holders` - Institutional holders
- `news` - Latest news
- `analyst_recommendations` - Analyst ratings
- `calendar` - Upcoming events
- `options_expirations` - available option dates
- `option_chain` - Options chain data

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_CLIENT_ID` | `stock-mcp-client` | Auth client ID |
| `MCP_CLIENT_SECRET` | `super-secret-key` | Auth secret |
| `PORT` | `8080` | Server port (HTTP mode) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `REQUEST_TIMEOUT` | `30` | yfinance timeout (seconds) |
| `RATE_LIMIT_REQUESTS` | `100` | Requests per minute |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |

### Authentication

Include headers in all requests:
```bash
X-Client-Id: your-client-id
X-Client-Secret: your-secret
```

## Examples

### Using curl

```bash
# Health check
curl -H "X-Client-Id: stock-mcp-client" \
     -H "X-Client-Secret: super-secret-key" \
     http://localhost:8080/health

# Get quote
curl -H "X-Client-Id: stock-mcp-client" \
     -H "X-Client-Secret: super-secret-key" \
     http://localhost:8080/quote?symbol=AAPL
```

## Production Features

✅ Structured logging with configurable levels
✅ Comprehensive error handling
✅ Health check endpoint for monitoring
✅ Request/error/cache metrics
✅ CORS support for web clients
✅ Environment validation on startup
✅ Smart caching to reduce API calls

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run black main.py

# Type check
uv run mypy main.py
```

## License

MIT