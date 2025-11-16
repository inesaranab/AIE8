from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, List

class AgentState(TypedDict):
    """State for the agent"""
    messages: Annotated[List, add_messages]
