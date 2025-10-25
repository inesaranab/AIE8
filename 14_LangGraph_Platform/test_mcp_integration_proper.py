#!/usr/bin/env python3
"""
Proper test using langchain-mcp-adapters to validate MCP integration.
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_mcp_integration():
    """Test MCP integration using the proper client."""
    print("🔍 Testing MCP Integration with LangChain Adapters...")
    
    try:
        # Create MCP client
        client = MultiServerMCPClient({
            "my_server": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        })
        
        print(" MCP Client created successfully")
        
        # Get tools
        print("\n📋 Loading MCP tools...")
        tools = await client.get_tools()
        print(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
        
        # Test web search tool
        print("\nTesting web_search tool...")
        web_search_tool = next((tool for tool in tools if tool.name == "web_search"), None)
        
        if web_search_tool:
            print("web_search tool found")
            
            # Test the tool
            result = await web_search_tool.ainvoke({"query": "current temperature in Madrid"})
            print(f"Tool executed successfully!")
            print(f"Response type: {type(result)}")
            print(f"Response preview: {str(result)[:200]}...")
            
            # Check if it's a string
            if isinstance(result, str):
                print("Response is a string (correct format)")
            else:
                print(f"Response is not a string: {type(result)}")
                print(f"Content: {result}")
        else:
            print("web_search tool not found")
        
        # Test arxiv search tool
        print("\nTesting arxiv_search tool...")
        arxiv_search_tool = next((tool for tool in tools if tool.name == "arxiv_search"), None)
        
        if arxiv_search_tool:
            print("arxiv_search tool found")
            
            # Test the tool
            result = await arxiv_search_tool.ainvoke({"query": "machine learning"})
            print(f"Tool executed successfully!")
            print(f"Response type: {type(result)}")
            print(f"Response preview: {str(result)[:200]}...")
            
            # Check if it's a string
            if isinstance(result, str):
                print("Response is a string (correct format)")
            else:
                print(f"Response is not a string: {type(result)}")
                print(f"Content: {result}")
        else:
            print("arxiv_search tool not found")
            
    except Exception as e:
        print(f"Error testing MCP integration: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run the MCP integration test."""
    await test_mcp_integration()
    
    print("\n" + "=" * 60)
    print(" SUMMARY:")
    print("If you see 'Response is a string (correct format)' above,")
    print("then your MCP tools are working correctly and the error you")
    print("encountered earlier should be resolved!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
