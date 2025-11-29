# Deploy yfinance MCP to Render

## Prerequisites
- GitHub account
- Render account (free): https://render.com

## Step 1: Push to GitHub

```bash
cd /Users/matthewwoods/Development/stock-mcp
git init
git add .
git commit -m "Initial commit - yfinance MCP server"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/stock-mcp.git
git push -u origin main
```

## Step 2: Create Render Web Service

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select the `stock-mcp` repository

## Step 3: Configure Service

**Basic Settings:**
- **Name**: `stock-mcp` (or your preferred name)
- **Region**: Choose closest to you
- **Branch**: `main`
- **Runtime**: `Python 3`

**Build & Deploy:**
- **Build Command**: `pip install uv && uv sync`
- **Start Command**: `uv run main.py`

**Plan:**
- Select **Free**

## Step 4: Set Environment Variables

Click **"Advanced"** → **"Add Environment Variable"**

Add these two variables:
- **MCP_CLIENT_ID**: `your-client-id-here` (change this!)
- **MCP_CLIENT_SECRET**: `your-secret-here` (change this!)

> [!IMPORTANT]
> **Change these credentials!** The defaults are not secure for production.

## Step 5: Deploy

1. Click **"Create Web Service"**
2. Wait for deployment (usually 2-3 minutes)
3. Once deployed, you'll get a URL like: `https://stock-mcp.onrender.com`

## Step 6: Test Your Deployment

```bash
# Test without auth (should fail with 401)
curl https://your-app.onrender.com/sse

# Test with auth headers
curl -H "X-Client-Id: your-client-id-here" \\
     -H "X-Client-Secret: your-secret-here" \\
     https://your-app.onrender.com/sse
```

## Important Notes

### Cold Starts
- Free tier spins down after 15 min of inactivity
- First request after sleep takes ~30-60 seconds
- Keep service warm with a cron job (optional)

### Logs
- View logs in Render dashboard → Your Service → Logs
- Server will log: `Starting MCP server in SSE mode on 0.0.0.0:PORT`

### Updating Credentials
1. Go to Render dashboard → Your Service → Environment
2. Update `MCP_CLIENT_ID` and `MCP_CLIENT_SECRET`
3. Service will auto-redeploy

## Troubleshooting

**Service won't start:**
- Check logs for errors
- Verify `pyproject.toml` has all dependencies
- Ensure Python version matches `runtime.txt`

**401 Unauthorized:**
- Verify headers match environment variables
- Check header names: `X-Client-Id` and `X-Client-Secret`

**yfinance API issues:**
- Some rate limiting may occur
- Consider adding retry logic if needed

## Your Deployment URL

After deployment, save your URL and credentials:

```
URL: https://your-app.onrender.com/sse
Client ID: your-client-id-here
Client Secret: your-secret-here
```

## Next Steps

- Update credentials to something secure
- Test all MCP tools via the deployed endpoint
- Consider upgrading to paid tier for zero downtime
