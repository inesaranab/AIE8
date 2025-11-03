"""Simplified LangGraph agent with consolidated guardrails validation.

This module provides a cleaner agent flow following the helpfulness pattern:
- Agent -> if tool calls -> Action -> Agent
- Agent -> if no tool calls -> Refinement -> Agent (loop until valid)
- Refinement validates: no PII, no invalid topics, grounded response, no competitor mentions
"""

import logging

logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain

try:
    from guardrails.hub import (
        RestrictToTopic,
        DetectJailbreak,
        CompetitorCheck,
        LlmRagEvaluator,
        HallucinationPrompt,
        ProfanityFree,
        GuardrailsPII
    )
    from guardrails import Guard
    print("Guardrails imports successful!")
    guardrails_available = True

except ImportError as e:
    print(f"⚠ Guardrails not available: {e}")
    print("Please follow the setup instructions in the README")
    guardrails_available = False


class AgentState(TypedDict):
    """Enhanced state schema for agent graphs with validation tracking."""
    messages: Annotated[List[BaseMessage], add_messages]
    refinement_count: int  # Track refinement attempts separately
    failed_guards: List[str]  # Track which guards failed


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""

    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"

    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.

    Args:
        rag_chain: Optional RAG chain to include as a tool

    Returns:
        List of tools
    """
    tools = []

    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))

    # Add Arxiv tool
    tools.append(ArxivQueryRun())

    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))

    return tools


def create_guards_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    topic_guard: Optional[Guard] = None,
    jailbreak_guard: Optional[Guard] = None,
    pii_guard: Optional[Guard] = None,
    profanity_guard: Optional[Guard] = None,
    competitor_guard: Optional[Guard] = None,
    factuality_guard: Optional[Guard] = None,
    valid_topics: Optional[List[str]] = None,
    invalid_topics: Optional[List[str]] = None,
    competitors: Optional[List[str]] = None,
    entities: Optional[List[str]] = None,
    max_refinements: int = 5,
):
    """Create a simplified LangGraph agent with consolidated guardrails.

    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        topic_guard: Guard for topic restriction (output validation)
        jailbreak_guard: Guard for jailbreak detection (input validation)
        pii_guard: Guard for PII protection (both input and output)
        profanity_guard: Guard for profanity detection (output validation)
        competitor_guard: Guard for competitor mentions (output validation)
        factuality_guard: Guard for factuality checking (output validation)
        valid_topics: List of valid topics
        invalid_topics: List of invalid topics
        competitors: List of competitor names
        entities: List of PII entity types to detect
        max_refinements: Maximum number of refinement attempts (default: 5)

    Returns:
        Compiled LangGraph agent with simplified flow
    """

    if tools is None:
        tools = get_default_tools(rag_chain)

    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)

    # Store guard configuration for refinement prompts
    guard_config = {
        "valid_topics": valid_topics or ["student loans", "financial aid", "education financing", "loan repayment"],
        "invalid_topics": invalid_topics or ["investment advice", "crypto", "gambling", "politics"],
        "competitors": competitors or ["ChatGPT", "OpenAI", "Gemini", "Claude", "Anthropic", "Google Bard"],
        "has_profanity_guard": profanity_guard is not None,
        "has_competitor_guard": competitor_guard is not None,
        "has_topic_guard": topic_guard is not None,
        "has_pii_guard": pii_guard is not None,
        "has_factuality_guard": factuality_guard is not None,
        "entities": entities or ["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"],
        "max_refinements": max_refinements,
    }

    def preprocessing(
        state: AgentState,
        pii_guard: Optional[Guard] = pii_guard,
        jailbreak_guard: Optional[Guard] = jailbreak_guard,
        topic_guard: Optional[Guard] = topic_guard
    ):
        """Preprocess input with automatic fixes where possible.

        This function:
        1. Processes each HumanMessage through input validation guards
        2. Automatically redacts PII using on_fail="fix"
        3. Checks for jailbreak attempts (fails fast)
        4. Validates topic compliance (returns error if invalid)
        5. Returns processed messages or error message
        6. Initializes refinement_count and failed_guards if not present
        """
        messages = state["messages"]
        processed_messages = []

        # Initialize state fields if not present
        refinement_count = state.get("refinement_count", 0)
        failed_guards = state.get("failed_guards", [])

        if not guardrails_available:
            # If guardrails not available, just pass messages through
            return {
                "messages": messages,
                "refinement_count": refinement_count,
                "failed_guards": failed_guards
            }

        for message in messages:
            if isinstance(message, HumanMessage):
                current_content = message.content

                # Skip processing refinement instructions
                if current_content.startswith("The previous response failed validation"):
                    processed_messages.append(message)
                    continue

                # 1. PII Protection: Automatically redact sensitive information (on_fail="fix")
                if pii_guard:
                    try:
                        pii_response = pii_guard.validate(current_content)
                        # Use the automatically redacted version
                        current_content = pii_response.validated_output
                        # Create new message with redacted content
                        message = HumanMessage(content=current_content)
                    except Exception as e:
                        # If PII validation fails, log but continue with original content
                        logger.warning(f"PII validation error: {e}")

                # 2. Jailbreak Detection: Fail fast for security (on_fail="exception")
                if jailbreak_guard:
                    try:
                        jailbreak_response = jailbreak_guard.validate(current_content)
                        if not jailbreak_response.validation_passed:
                            return {
                                "messages": [AIMessage(content="I'm sorry, but I cannot process requests that attempt to bypass my safety guidelines.")],
                                "refinement_count": refinement_count,
                                "failed_guards": failed_guards
                            }
                    except Exception as e:
                        # Guard raised exception (on_fail="exception" behavior)
                        return {
                            "messages": [AIMessage(content="I'm sorry, but I cannot process requests that attempt to bypass my safety guidelines.")],
                            "refinement_count": refinement_count,
                            "failed_guards": failed_guards
                        }

                # 3. Topic Validation: Ensure query is on-topic (on_fail="exception")
                if topic_guard:
                    try:
                        topic_response = topic_guard.validate(current_content)
                        if not topic_response.validation_passed:
                            # Topic validation failed
                            return {
                                "messages": [AIMessage(content="I'm sorry, but I can only help with questions about student loans and financial aid. Your question appears to be on a different topic.")],
                                "refinement_count": refinement_count,
                                "failed_guards": failed_guards
                            }
                    except Exception as e:
                        # Guard raised exception (on_fail="exception" behavior)
                        error_msg = str(e)
                        if "Invalid topics" in error_msg or "No valid topic" in error_msg:
                            return {
                                "messages": [AIMessage(content="I'm sorry, but I can only help with questions about student loans and financial aid. Your question doesn't appear to be related to these topics.")],
                                "refinement_count": refinement_count,
                                "failed_guards": failed_guards
                            }
                        return {
                            "messages": [AIMessage(content=f"I'm sorry, but I can only help with questions about student loans and financial aid. Topic validation failed.")],
                            "refinement_count": refinement_count,
                            "failed_guards": failed_guards
                        }

                # All validations passed - add the processed message
                processed_messages.append(message)
            else:
                # Keep non-HumanMessages as-is (AIMessages, ToolMessages, etc.)
                processed_messages.append(message)

        # All checks passed - return processed messages
        return {
            "messages": processed_messages,
            "refinement_count": refinement_count,
            "failed_guards": failed_guards
        }

    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def consolidated_refinement(
        state: AgentState,
        pii_guard: Optional[Guard] = pii_guard,
        topic_guard: Optional[Guard] = topic_guard,
        factuality_guard: Optional[Guard] = factuality_guard,
        profanity_guard: Optional[Guard] = profanity_guard,
        competitor_guard: Optional[Guard] = competitor_guard
    ) -> Dict[str, Any]:
        """Consolidated output validation and refinement.

        This function:
        1. Checks refinement count (max attempts)
        2. Validates agent response against all output guards:
           - PII Guard: No PII in output
           - Topic Guard: Response stays on valid topics
           - Factuality Guard: Response is grounded
           - Profanity Guard: No inappropriate language
           - Competitor Guard: No competitor mentions
        3. If all pass: Returns completion marker
        4. If any fail: Creates targeted refinement instruction with specific failures
        5. Updates refinement_count and failed_guards
        """
        messages = state["messages"]
        refinement_count = state.get("refinement_count", 0)
        failed_guards = []

        # Check max refinements
        if refinement_count >= guard_config["max_refinements"]:
            logger.warning(f"Max refinement attempts ({guard_config['max_refinements']}) reached")
            return {
                "messages": [AIMessage(content="I apologize, but I'm having difficulty generating a fully compliant response. Please try rephrasing your question or contact support.")],
                "refinement_count": refinement_count,
                "failed_guards": []
            }

        if not guardrails_available:
            # If guardrails not available, just pass through
            return {
                "messages": [AIMessage(content="REFINEMENT:Y")],
                "refinement_count": refinement_count,
                "failed_guards": []
            }

        # Get the last AI message to validate
        last_message = None
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.content.startswith("REFINEMENT:"):
                last_message = message
                break

        if not last_message:
            # No AI message to validate, shouldn't happen
            logger.warning("No AI message found for validation")
            return {
                "messages": [AIMessage(content="REFINEMENT:Y")],
                "refinement_count": refinement_count,
                "failed_guards": []
            }

        response_content = last_message.content

        # Run all output guards and collect failures

        # 1. PII Guard: Check for PII in output
        if pii_guard:
            try:
                pii_response = pii_guard.validate(response_content)
                if not pii_response.validation_passed:
                    failed_guards.append("PII")
                    logger.info("PII guard failed on output")
            except Exception as e:
                failed_guards.append("PII")
                logger.warning(f"PII guard error: {e}")

        # 2. Topic Guard: Ensure response stays on valid topics
        if topic_guard:
            try:
                topic_response = topic_guard.validate(response_content)
                if not topic_response.validation_passed:
                    failed_guards.append("Topic")
                    logger.info("Topic guard failed on output")
            except Exception as e:
                # Guard raised exception
                error_msg = str(e)
                if "Invalid topics" in error_msg or "No valid topic" in error_msg:
                    failed_guards.append("Topic")
                    logger.info("Topic guard failed on output (exception)")

        # 3. Factuality Guard: Check if response is grounded
        if factuality_guard:
            try:
                factuality_response = factuality_guard.validate(response_content)
                if not factuality_response.validation_passed:
                    failed_guards.append("Factuality")
                    logger.info("Factuality guard failed on output")
            except Exception as e:
                failed_guards.append("Factuality")
                logger.warning(f"Factuality guard error: {e}")

        # 4. Profanity Guard: Check for inappropriate language
        if profanity_guard:
            try:
                profanity_response = profanity_guard.validate(response_content)
                if not profanity_response.validation_passed:
                    failed_guards.append("Profanity")
                    logger.info("Profanity guard failed on output")
            except Exception as e:
                failed_guards.append("Profanity")
                logger.warning(f"Profanity guard error: {e}")

        # 5. Competitor Guard: Check for competitor mentions
        if competitor_guard:
            try:
                competitor_response = competitor_guard.validate(response_content)
                if not competitor_response.validation_passed:
                    failed_guards.append("Competitor")
                    logger.info("Competitor guard failed on output")
            except Exception as e:
                failed_guards.append("Competitor")
                logger.warning(f"Competitor guard error: {e}")

        # If all guards passed, return success
        if not failed_guards:
            logger.info("All guards passed - refinement successful")
            return {
                "messages": [AIMessage(content="REFINEMENT:Y")],
                "refinement_count": refinement_count,
                "failed_guards": []
            }

        # Guards failed - create targeted refinement instruction
        logger.info(f"Guards failed: {', '.join(failed_guards)}. Refinement attempt {refinement_count + 1}")

        # Get initial query
        initial_query = None
        for message in messages:
            if isinstance(message, HumanMessage) and not message.content.startswith("The previous response failed validation"):
                initial_query = message
                break

        if not initial_query:
            initial_query = HumanMessage(content="[Initial query not found]")

        # Build specific requirements based on failed guards
        requirements = []

        if "PII" in failed_guards:
            requirements.append(f"- CRITICAL: Do not include any personally identifiable information (PII) such as: {', '.join(guard_config['entities'])}")

        if "Topic" in failed_guards:
            requirements.append(f"- CRITICAL: Stay strictly on these topics: {', '.join(guard_config['valid_topics'])}")
            requirements.append(f"- CRITICAL: Avoid these topics: {', '.join(guard_config['invalid_topics'])}")

        if "Factuality" in failed_guards:
            requirements.append("- CRITICAL: Base your response ONLY on the retrieved context and factual information")
            requirements.append("- CRITICAL: Do not make up information or provide ungrounded claims")
            requirements.append("- CRITICAL: If you don't have enough information, say so clearly")

        if "Profanity" in failed_guards:
            requirements.append("- CRITICAL: Use only professional and appropriate language")

        if "Competitor" in failed_guards:
            requirements.append(f"- CRITICAL: Do not mention these competitors: {', '.join(guard_config['competitors'])}")

        requirements.append("- Provide accurate and helpful information")
        requirements.append("- Address the user's specific question completely")

        requirements_text = "\n".join(requirements)

        refinement_instruction = f"""The previous response failed validation due to: {', '.join(failed_guards)}

Please provide a NEW response that meets these requirements:

{requirements_text}

Original question: {initial_query.content}

Generate a compliant response now:"""

        # Increment refinement count and return instruction
        return {
            "messages": [HumanMessage(content=refinement_instruction)],
            "refinement_count": refinement_count + 1,
            "failed_guards": failed_guards
        }

    def should_continue(state: AgentState):
        """Route to action if tool calls, otherwise to refinement."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "refinement"

    def refinement_decision(state: AgentState):
        """Decide what to do after refinement."""
        last_message = state["messages"][-1]

        # Check if refinement completed successfully
        if isinstance(last_message, AIMessage) and "REFINEMENT:Y" in last_message.content:
            # Remove the REFINEMENT:Y marker message before ending
            # Find the actual response (second to last message)
            for i in range(len(state["messages"]) - 2, -1, -1):
                msg = state["messages"][i]
                if isinstance(msg, AIMessage) and not msg.content.startswith("REFINEMENT:"):
                    # This is the validated response
                    logger.info("Refinement completed successfully, ending conversation")
                    return "end"
            return "end"

        # Check if we hit max refinements with error message
        if isinstance(last_message, AIMessage) and "having difficulty generating" in last_message.content:
            logger.warning("Max refinements reached, ending with error")
            return "end"

        # Otherwise, continue refining
        return "agent"

    # Build graph with simplified structure
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)

    # Add nodes
    graph.add_node("preprocessing", preprocessing)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("refinement", consolidated_refinement)

    # Set entry point
    graph.set_entry_point("preprocessing")

    # Add edges
    graph.add_edge("preprocessing", "agent")

    # Conditional routing from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            "refinement": "refinement"
        }
    )

    # Action loops back to agent
    graph.add_edge("action", "agent")

    # Conditional routing from refinement
    graph.add_conditional_edges(
        "refinement",
        refinement_decision,
        {
            "agent": "agent",
            "end": END
        }
    )

    return graph.compile()
