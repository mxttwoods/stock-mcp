import asyncio
from fastmcp import Client
import urllib3

healthurl = "https://stock-mcp-po2g.onrender.com/health"
# client = Client("http://localhost/api/mcp")
client = Client("https://stock-mcp-po2g.onrender.com/api/mcp")


async def call_tool(name: str):
    async with client:
        result = await client.call_tool("greet", {"name": name})
        print(result)


async def health_check():
    print("Checking health...")
    response = urllib3.request("GET", healthurl)
    print(response.data)


asyncio.run(health_check())
asyncio.run(call_tool("world"))
