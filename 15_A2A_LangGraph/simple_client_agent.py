"""Simple LangGraph Agent that calls an A2A Protocol Agent.

This demonstrates how to use LangGraph to create a simple agent that makes
API calls to another agent via the A2A (Agent-to-Agent) protocol.
"""
import asyncio
import logging
from typing import TypedDict
from uuid import uuid4

import httpx
from langgraph.graph import StateGraph, END
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the state for our simple agent
class SimpleAgentState(TypedDict):
    """State schema for the simple agent."""
    query: str  # User's input query
    response: str  # Response from the A2A agent


async def call_a2a(state: SimpleAgentState) -> dict:
    """Node 1: Call the A2A agent with the user's query.

    This node:
    - Connects to the A2A server at localhost:10000
    - Sends the user's query
    - Returns the response
    """
    logger.info(f"Calling A2A agent with query: {state['query']}")

    base_url = 'http://localhost:10000'

    # Create HTTP client with longer timeout for LLM responses
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        # Initialize A2A card resolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        # Fetch the agent card
        agent_card = await resolver.get_agent_card()
        logger.info(f"Connected to agent: {agent_card.name}")

        # Initialize A2A client
        client = A2AClient(
            httpx_client=httpx_client,
            agent_card=agent_card
        )

        # Prepare the message
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': state['query']}
                ],
                'message_id': uuid4().hex,
            },
        }

        # Create request
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**send_message_payload)
        )

        # Send message and get response
        response = await client.send_message(request)

        # Store the full response
        return {"response": response.model_dump(mode='json', exclude_none=True)}


def extract_response(state: SimpleAgentState) -> dict:
    """Node 2: Extract the text content from the A2A response.

    This node parses the A2A response format and extracts the actual
    text content from the agent's reply.
    """
    logger.info("Extracting response content...")

    # Navigate the A2A response structure to get the text
    try:
        response_data = state['response']
        result = response_data.get('result', {})
        artifacts = result.get('artifacts', [])

        # Get the first artifact (the agent's response)
        if artifacts:
            artifact = artifacts[0]
            parts = artifact.get('parts', [])

            # Extract text from all parts
            text_parts = []
            for part in parts:
                if part.get('kind') == 'text':
                    text_parts.append(part.get('text', ''))

            extracted_text = '\n'.join(text_parts)
            logger.info(f"Extracted {len(extracted_text)} characters")

            return {"response": extracted_text}
        else:
            return {"response": "No response artifacts found"}
    except Exception as e:
        logger.error(f"Error extracting response: {e}")
        return {"response": f"Error extracting response: {e}"}


def display(state: SimpleAgentState) -> dict:
    """Node 3: Display the final response.

    This node prints the agent's response to the console.
    """
    print("\n" + "="*80)
    print("QUERY:")
    print("-"*80)
    print(state['query'])
    print("\n" + "="*80)
    print("A2A AGENT RESPONSE:")
    print("-"*80)
    print(state['response'])
    print("="*80 + "\n")

    return state


def build_simple_agent() -> StateGraph:
    """Build the simple LangGraph agent.

    Creates a simple linear graph:
    START → call_a2a → extract_response → display → END
    """
    # Create the graph
    graph = StateGraph(SimpleAgentState)

    # Add nodes
    graph.add_node("call_a2a", call_a2a)
    graph.add_node("extract_response", extract_response)
    graph.add_node("display", display)

    # Define the flow (linear)
    graph.set_entry_point("call_a2a")
    graph.add_edge("call_a2a", "extract_response")
    graph.add_edge("extract_response", "display")
    graph.add_edge("display", END)

    return graph.compile()


async def main():
    """Main function to run the simple agent."""
    print("\n" + "="*80)
    print("SIMPLE LANGGRAPH AGENT WITH A2A PROTOCOL")
    print("="*80)
    print("\nThis agent demonstrates calling another agent via A2A protocol.")
    print("Make sure your A2A server is running on localhost:10000")
    print("(Start it with: uv run python -m app)\n")

    # Build the agent
    agent = build_simple_agent()

    # Define a test query
    test_query = "What are AI agents and how do they work? Give me a brief overview."

    # Create initial state
    initial_state: SimpleAgentState = {
        "query": test_query,
        "response": ""
    }

    # Run the agent
    logger.info("Starting agent execution...")
    try:
        final_state = await agent.ainvoke(initial_state)
        logger.info("Agent execution completed successfully!")
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        print(f"\nERROR: {e}")
        print("\nMake sure your A2A server is running:")
        print("  uv run python -m app")


if __name__ == '__main__':
    asyncio.run(main())
