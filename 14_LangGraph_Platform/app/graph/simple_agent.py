"""A simplified agent graph using the ToolNode pattern.

The agent decides what tools to use, and ToolNode handles execution automatically.
"""
from __future__ import annotations

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.state import AgentState
from app.models import get_chat_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def make_graph():
    """Create the graph with MCP tools loaded asynchronously."""
    client = MultiServerMCPClient({
        "main-server": {
            "url": "http://localhost:8001/mcp",
            "transport": "streamable_http",
        },
    })
    tools = await client.get_tools()

    def call_model(state: AgentState) -> Dict[str, Any]:
        """Main agent that decides what to do"""
        model = get_chat_model().bind_tools(tools)
        question = state["question"]
        
        # Initialize messages if not present
        messages = state.get("messages", [])
        if not messages:
            # Create comprehensive prompt
            messages = [HumanMessage(content=f"""
            Question: {question}
            
            Please:
            1. Classify sentiment using sentiment_classification tool
            2. If negative sentiment, use retrieve_information_tool to get context
            3. Generate a comprehensive response
            4. Use text_to_speech tool to create audio
            
            Use the available tools as needed.
            """)]
        
        response = model.invoke(messages)
        return {"messages": messages + [response]}

    def route_to_action_or_end(state: AgentState):
        """Decide whether to execute tools or end"""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "end"

    def build_graph():
        """Build the simplified agent graph"""
        graph = StateGraph(AgentState)
        tool_node = ToolNode(tools)  # This handles ALL tool execution automatically!
        
        graph.add_node("agent", call_model)
        graph.add_node("action", tool_node)
        
        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            route_to_action_or_end,
            {"action": "action", "end": END}
        )
        graph.add_edge("action", "agent")  # After tools, back to agent
        
        return graph

    return build_graph().compile()
