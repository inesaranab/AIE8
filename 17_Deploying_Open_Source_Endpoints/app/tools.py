import os
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from app.rag import retrieve_information

def get_tool_belt() -> List:
    """Return the list of tools available to the agents"""
    tools = [ArxivQueryRun(), retrieve_information]
    
    # Only add Tavily if API key is available
    if os.environ.get("TAVILY_API_KEY"):
        try:
            tavily_tool = TavilySearchResults(max_results=5)
            tools.insert(0, tavily_tool)
        except Exception as e:
            print(f"Warning: Could not initialize Tavily: {e}")
    
    return tools
