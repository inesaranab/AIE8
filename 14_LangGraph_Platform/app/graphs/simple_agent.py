"""A minimal tool-using agent graph.

The graph:
- Calls a chat model bound to the tool belt.
- If the last message requested tool calls, routes to a ToolNode.
- Otherwise, terminates.
"""
from __future__ import annotations

from typing import Dict, Any
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.state import AgentState
from app.models import get_chat_model

from langchain_mcp_adapters.client import MultiServerMCPClient


async def make_graph():
    """Create the graph with MCP tools loaded asynchronously."""
    client = MultiServerMCPClient(
        {
            "my_server": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    
    def _build_model_with_tools():
        """Return a chat model instance bound to the current tool belt."""
        model = get_chat_model()
        return model.bind_tools(tools)

    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with the accumulated messages and append its response."""
        model = _build_model_with_tools()
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        """Route to 'action' if the last message includes tool calls; else END."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END

    def build_graph():
        """Build an agent graph that interleaves model and tool execution."""
        graph = StateGraph(AgentState)
        tool_node = ToolNode(tools)
        graph.add_node("agent", call_model)
        graph.add_node("action", tool_node)
        graph.set_entry_point("agent")
        # Explicitly map END sentinel to avoid KeyError('__end__') in platform runtime
        graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
        graph.add_edge("action", "agent")
        return graph

    return build_graph()


# Export compiled graph for LangGraph Platform
# Note: The graph will be created asynchronously when needed




