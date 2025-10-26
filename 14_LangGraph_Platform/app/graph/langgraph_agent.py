
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.models import get_chat_model
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
import os
import re


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

### Setup MCP Client and Tools at Module Level

# We need to run async code at module level, so we'll use a helper
async def make_graph():
    """Create a graph with MCP tools loaded asynchrounously"""
    client = MultiServerMCPClient({
        "main-server": {
            "url": "http://localhost:8001/mcp",
            "transport": "streamable_http",
        },
    })

    # Get tools from MCP servers
    tools = await client.get_tools()

    def _build_model_with_tools():
        """Return a chat model instance bound to the current tool belt."""
        model = get_chat_model()
        return model.bind_tools(tools)

    def call_model_text(state: AgentState):
        """Invoke the model with the accumulated messages and append its response."""
        # Initialize messages if not present
        
        model = _build_model_with_tools()
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}
    

    def route_to_action_or_helpfulness(state: AgentState):
        """Decide whether to execute tools or run the helpfulness evaluator."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "helpfulness"

    def helpfulness_node(state: AgentState):
        """Evaluate helpfulness and generate audio if helpful"""
        # If we've exceeded loop limit, short-circuit with END decision marker
        if len(state["messages"]) > 10:
            return {"messages": [AIMessage(content="HELPFULNESS:END")]}    

        initial_query = state["messages"][0]
        final_response = state["messages"][-1]

        # Safely extract content from messages
        initial_query_content = getattr(initial_query, 'content', str(initial_query))
        final_response_content = getattr(final_response, 'content', str(final_response))

        prompt_template = """
Given an initial query and a final response, determine if the final response is helpful or not.

A response is NOT helpful if it:
- Says "I don't know" without trying to find information
- Gives up immediately without using available tools
- Provides no useful information at all
- Is completely unrelated to the user's question

A response IS helpful if it:
- Provides specific, actionable information
- Directly answers the user's question
- Offers useful guidance or steps
- Contains relevant details or instructions
- Helps the user solve their problem
- Makes an effort to find information using available tools
- Provides context or related information even if not a complete answer

Consider that the model has access to tools for retrieving information. A response that makes an effort to use these tools should generally be considered helpful, even if it doesn't have the complete answer.

Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

Initial Query:
{initial_query}

Final Response:
{final_response}"""

        helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
        helpfulness_check_model = get_chat_model(model_name="gpt-4.1-mini")
        helpfulness_chain = (
            helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()
        )

        helpfulness_response = helpfulness_chain.invoke(
            {
                "initial_query": initial_query_content,
                "final_response": final_response_content,
            }
        )

        decision = "Y" if "Y" in helpfulness_response else "N"
        
        # If helpful, generate audio directly
        if decision == "Y":
            # Extract the final response to convert to audio
            final_response_text = final_response_content
            
            # Create tool call for audio generation
            tool_call = {
                "name": "text_to_speech",
                "args": {"text": final_response_text},
                "id": f"call_audio_{hash(final_response_text)}"
            }
            
            # Return both helpfulness decision and audio tool call
            return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}", tool_calls=[tool_call])]}
        
        return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}

    def helpfulness_decision(state: AgentState):
        """Route to audio if helpful, otherwise continue or end"""
        # Check loop-limit marker
        if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
            return END
        
        last = state["messages"][-1]
        text = getattr(last, "content", "")
        if "HELPFULNESS:Y" in text:
            # Check if there's a tool call for audio generation
            if getattr(last, "tool_calls", None):
                return "audio"
            return END  # Changed from "end" to END
        return "continue"
    

    def build_graph():
        """Build a simplified agent graph with audio generation"""
        graph = StateGraph(AgentState)
        regular_tools = [tool for tool in tools if tool.name != "text_to_speech"]
        audio_tool = [tool for tool in tools if tool.name == "text_to_speech"]
        regular_tool_node = ToolNode(regular_tools)
        audio_tool_node = ToolNode(audio_tool)
        
        graph.add_node("agent_text", call_model_text)
        graph.add_node("action", regular_tool_node)
        graph.add_node("helpfulness", helpfulness_node)
        graph.add_node("audio", audio_tool_node)
        
        graph.set_entry_point("agent_text")
        graph.add_conditional_edges(
            "agent_text",
            route_to_action_or_helpfulness,
            {"action": "action", "helpfulness": "helpfulness"},
        )
        graph.add_edge("action", "agent_text")
        graph.add_conditional_edges(
            "helpfulness",
            helpfulness_decision,
            {"continue": "agent_text", "audio": "audio", END: END},
        )
        graph.add_edge("audio", END)  # Audio goes directly to END

        return graph

    return build_graph().compile()