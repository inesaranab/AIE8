
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from app.state import AgentState
from app.tools import get_tool_belt
from app.model import get_chat_model
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

def _build_model_with_tools():  
    """Return a chat model instance boud to the current tool belt"""
    model = get_chat_model(model_name="openai/gpt-oss-20b")
    return model.bind_tools(get_tool_belt())

def call_model(state: AgentState) -> Dict[str, Any]:
    model = _build_model_with_tools()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return END

def build_agent():
    graph = StateGraph(AgentState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            END: END,
        }
    )
    graph.add_edge("action", "agent")
    graph.set_entry_point("agent")
    return graph
