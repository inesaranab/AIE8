from fastmcp import FastMCP
import os
from dotenv import load_dotenv
import arxiv
load_dotenv()
from tavily import TavilyClient
from app.rag import retrieve_information

mcp = FastMCP("my_server")

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@mcp.tool
def web_search(query: str) -> str:
    """Search the web for information"""
    response = tavily_client.search(query)
    
    # Format the response as a readable string
    if 'results' in response and response['results']:
        formatted_results = []
        for i, result in enumerate(response['results'][:3], 1):  # Limit to top 3 results
            formatted_results.append(f"{i}. {result['title']}\n   URL: {result['url']}\n   Content: {result['content'][:200]}...")
        
        return f"Search results for '{query}':\n\n" + "\n\n".join(formatted_results)
    else:
        return f"No results found for '{query}'"

@mcp.tool
def arxiv_search(query: str) -> str:
    """Search the arXiv for information"""
    try:
        search = arxiv.Search(query=query)
        results = list(search.results()[:3])  # Get first 3 results
        
        if results:
            formatted_results = []
            for i, paper in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. {paper.title}\n"
                    f"   Authors: {', '.join([author.name for author in paper.authors])}\n"
                    f"   Published: {paper.published}\n"
                    f"   Summary: {paper.summary[:200]}...\n"
                    f"   URL: {paper.entry_id}"
                )
            return f"arXiv search results for '{query}':\n\n" + "\n\n".join(formatted_results)
        else:
            return f"No arXiv papers found for '{query}'"
    except Exception as e:
        return f"Error searching arXiv for '{query}': {str(e)}"

@mcp.tool
def retrieve_information_tool(query: str) -> str:
    """Use Retrieval Augmented Generation to retrieve information from the RAG database"""
    return retrieve_information.invoke({"query": query})

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)