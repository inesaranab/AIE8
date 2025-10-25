from fastmcp import FastMCP
import os
from dotenv import load_dotenv
import arxiv
load_dotenv()
from tavily import TavilyClient
mcp = FastMCP("my_server")

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@mcp.tool
def web_search(query: str) -> str:
    """Search the web for information"""
    response = tavily_client.search(query)
    return response

@mcp.tool
def arxiv_search(query: str) -> str:
    """Search the arXiv for information"""
    response = arxiv.Search(query=query)
    return response

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)