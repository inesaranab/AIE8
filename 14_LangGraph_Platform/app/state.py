from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from typing import Annotated, List
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State schema for the agent graph"""  
    messages: Annotated[List, add_messages] 