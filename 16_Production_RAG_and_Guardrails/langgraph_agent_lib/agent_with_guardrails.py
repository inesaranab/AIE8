"""Production-safe LangGraph agent with Guardrails validation.

This module implements Activity #3: A LangGraph agent with comprehensive
guardrails for input validation, output validation, and error handling.
"""

import logging
from typing import Annotated, List, TypedDict, Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph_agent_lib.rag import ProductionRAGChain

import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun

from langgraph_agent_lib.models import get_openai_model
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import guardrails utilities
from langgraph_agent_lib.guardrails import (
    create_guardrails_guard,
    create_factuality_guard,
    validate_input,
    validate_output
)

# Set up logging
logger = logging.getLogger(__name__)

try:
    from guardrails.hub import (
        RestrictToTopic,
        DetectJailbreak,
        GuardrailsPII,
        HallucinationPrompt,
        ProfanityFree
    )
    from guardrails import Guard
    guardrails_available = True
except ImportError as e:
    logger.warning(f"Guardrails not available: {e}")
    guardrails_available = False


class AgentState(TypedDict):
    """State schema for agent with guardrails."""
    messages: Annotated[List[BaseMessage], add_messages]
    validation_results: Optional[List[Dict[str, Any]]]
    refinement_count: Optional[int]
    input_refinement_count: Optional[int]  # Track input refinement attempts
    error_message: Optional[str]

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
    """Get default tools for the agent with guardrails."""
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

def create_langgraph_agent_with_guardrails(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    valid_topics: Optional[List[str]] = None,
    invalid_topics: Optional[List[str]] = None,
    enable_factuality_check: bool = False,  # Disable by default due to config issues
    max_refinements: int = 2,
    max_input_refinements: int = 0,  # Disable input refinement - reject adversarial inputs outright
    strict_mode: bool = True
):
    """Create a LangGraph agent with comprehensive guardrails.

    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        valid_topics: List of valid topics (default: student loan topics)
        invalid_topics: List of invalid topics to block
        enable_factuality_check: Whether to check response factuality (default: False)
        max_refinements: Maximum number of output refinement attempts
        max_input_refinements: Maximum number of input refinement attempts (0 = reject immediately)
        strict_mode: If True, rejects validation failures immediately

    Returns:
        Compiled LangGraph agent with guardrails

    Note:
        - Input refinement is disabled by default (max_input_refinements=0)
        - Adversarial inputs are rejected immediately without attempting refinement
        - Factuality checking is disabled by default to avoid configuration issues
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)

    # Configure default topics if not provided
    if valid_topics is None:
        valid_topics = ["student loans", "financial aid", "education financing", "loan repayment"]
    if invalid_topics is None:
        invalid_topics = ["investment advice", "crypto", "gambling", "politics", "illegal activities"]
    
    # Create input guard (jailbreak, topic, PII)
    input_guard = None
    if guardrails_available:
        try:
            input_guard = create_guardrails_guard(
                valid_topics=valid_topics,
                invalid_topics=invalid_topics,
                enable_jailbreak_detection=True,
                enable_pii_protection=True,
                enable_profanity_check=False,  # Only check output for profanity
                enable_competitor_check=False # Not mention on exercise
            )
            logger.info("Input guard configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure input guard: {e}")
            if strict_mode:
                raise
    
    # Create output guard (profanity, factuality)
    output_guard = None
    factuality_guard = None
    if guardrails_available:
        try:
            # Profanity guard for output
            profanity_guard = Guard().use(
                ProfanityFree(threshold=0.8, validation_method="sentence", on_fail="exception")
            )
            output_guard = profanity_guard
            
            # Factuality guard (optional, requires RAG context)
            if enable_factuality_check and rag_chain:
                factuality_guard = create_factuality_guard(
                    eval_model="gpt-4o-mini",
                    on_prompt=False  # Check output, not prompt
                )
            logger.info("Output guards configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure output guard: {e}")
            if strict_mode:
                raise

    def refine_input_node(state: AgentState) -> Dict[str, Any]:
        """Refine invalid input by asking the model to suggest a better query.
        
        This node attempts to fix invalid input by having the model:
        - Rephrase the query to be on-topic
        - Remove jailbreak attempts
        - Make the query appropriate
        """
        messages = state.get("messages", [])
        input_refinement_count = state.get("input_refinement_count", 0)
        validation_results = state.get("validation_results", [])
        
        if not messages:
            return {"error_message": "No messages to refine"}
        
        # Get the original invalid input
        original_message = messages[0] if messages else None
        if not isinstance(original_message, HumanMessage):
            return {"error_message": "Cannot refine: no valid user input"}
        
        original_input = original_message.content
        
        # Create a refinement prompt
        refinement_prompt = f"""The following user query was rejected by our validation system:
        
Original query: "{original_input}"

Please suggest a rephrased version of this query that:
1. Stays on-topic (focuses on student loans, financial aid, education financing, or loan repayment)
2. Removes any inappropriate requests or attempts to bypass restrictions
3. Maintains the user's original intent if it's legitimate

Provide only the rephrased query, nothing else:"""
        
        try:
            # Use the model to suggest a refined query
            refinement_messages = [HumanMessage(content=refinement_prompt)]
            refinement_response = model.invoke(refinement_messages)
            refined_input = refinement_response.content.strip()
            
            # Remove quotes if the model added them
            if refined_input.startswith('"') and refined_input.endswith('"'):
                refined_input = refined_input[1:-1]
            
            logger.info(f"Input refined from '{original_input[:50]}...' to '{refined_input[:50]}...'")
            
            # Create new message with refined input
            refined_message = HumanMessage(content=refined_input)
            
            validation_results.append({
                "stage": "input_refinement",
                "attempt": input_refinement_count + 1,
                "original": original_input[:100],
                "refined": refined_input[:100],
                "timestamp": "now"
            })
            
            return {
                "messages": [refined_message],  # Replace with refined message
                "input_refinement_count": input_refinement_count + 1,
                "validation_results": validation_results
            }
            
        except Exception as e:
            logger.error(f"Input refinement error: {str(e)}")
            return {
                "error_message": f"Input refinement failed: {str(e)}",
                "validation_results": validation_results
            }
    
    def validate_input_node(state: AgentState) -> Dict[str, Any]:
        """Pre-processing: Validate user input before agent processes it.
        
        Validates:
        - Jailbreak attempts
        - Topic restrictions
        - PII detection (redacts automatically)
        """
        messages = state.get("messages", [])
        validation_results = state.get("validation_results", [])
        input_refinement_count = state.get("input_refinement_count", 0)
        
        if not messages:
            return {"error_message": "No messages to validate"}
        
        # Get the last message (should be user input)
        last_message = messages[-1]
        
        if not isinstance(last_message, HumanMessage):
            # Not a user input, skip validation
            return {}
        
        user_input = last_message.content
        
        if not guardrails_available or not input_guard:
            logger.warning("Guardrails not available, skipping input validation")
            return {}
        
        try:
            # Validate input
            result = validate_input(
                input_guard,
                user_input,
                raise_on_failure=False  # Don't raise, we'll handle refinement
            )
            
            validation_results.append({
                "stage": "input",
                "passed": result["validation_passed"],
                "error": result.get("error"),
                "timestamp": "now"
            })
            
            if not result["validation_passed"]:
                error_msg = result.get("error", "Input validation failed")
                logger.warning(f"Input validation failed: {error_msg}")
                
                # Check if we should try refinement
                if input_refinement_count < max_input_refinements:
                    # Trigger input refinement
                    logger.info(f"Input validation failed, triggering refinement (attempt {input_refinement_count + 1})")
                    return {
                        "validation_results": validation_results,
                        "input_refinement_count": input_refinement_count
                    }
                else:
                    # Max refinements reached
                    if strict_mode:
                        return {
                            "error_message": f"Input rejected after {max_input_refinements} refinement attempts: {error_msg}",
                            "validation_results": validation_results
                        }
                    else:
                        # In non-strict mode, continue but log the issue
                        logger.warning(f"Continuing despite validation failure: {error_msg}")
            
            # If PII was redacted, update the message
            if result.get("validated_output") != user_input:
                logger.info("PII detected and redacted in input")
                # Create new message with redacted content
                new_message = HumanMessage(content=result["validated_output"])
                return {
                    "messages": [new_message],
                    "validation_results": validation_results
                }
            
            # Validation passed
            return {"validation_results": validation_results}
            
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            validation_results.append({
                "stage": "input",
                "passed": False,
                "error": str(e),
                "timestamp": "now"
            })
            
            # Try refinement on exception too
            if input_refinement_count < max_input_refinements:
                return {
                    "validation_results": validation_results,
                    "input_refinement_count": input_refinement_count
                }
            
            if strict_mode:
                return {
                    "error_message": f"Input validation error: {str(e)}",
                    "validation_results": validation_results
                }
            return {"validation_results": validation_results}

    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages. Handles refinement requests."""
        messages = state.get("messages", [])
        error_message = state.get("error_message")
        refinement_count = state.get("refinement_count", 0)
        validation_results = state.get("validation_results", [])
        
        if error_message:
            # Don't process if there's an error
            return {}
        
        try:
            # If this is a refinement, add instruction to improve the response
            if refinement_count > 0:
                # Get the last AI response that failed validation
                last_ai_response = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        # Skip helpfulness evaluation messages
                        if not msg.content.startswith("HELPFULNESS:") and not getattr(msg, "tool_calls", None):
                            last_ai_response = msg.content
                            break
                
                if last_ai_response:
                    # Add refinement instruction
                    refinement_instruction = HumanMessage(content=(
                        f"The previous response failed validation checks. Please provide a revised, "
                        f"more appropriate response that addresses the validation concerns."
                    ))
                    messages = messages + [refinement_instruction]
            
            response = model_with_tools.invoke(messages)
            return {"messages": [response], "error_message": None}
        except Exception as e:
            logger.error(f"Model invocation error: {str(e)}")
            return {"error_message": f"Model error: {str(e)}"}
    
    def validate_output_node(state: AgentState) -> Dict[str, Any]:
        """Post-processing: Validate agent output before returning.
        
        Validates:
        - Profanity/content moderation
        - Factuality (if RAG context available)
        """
        messages = state.get("messages", [])
        validation_results = state.get("validation_results", [])
        refinement_count = state.get("refinement_count", 0)
        
        if not messages:
            return {}
        
        # Get the last message (should be agent response)
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.content:
            # Not an AI response or no content, skip validation
            return {}
        
        agent_response = last_message.content
        error_message = state.get("error_message")
        
        if error_message:
            # Don't validate if there's already an error
            return {}
        
        validation_failed = False
        validation_error = None
        
        # Validate with output guard (profanity)
        if guardrails_available and output_guard:
            try:
                result = validate_output(
                    output_guard,
                    agent_response,
                    raise_on_failure=False  # Don't raise, we'll handle it
                )
                
                validation_results.append({
                    "stage": "output_profanity",
                    "passed": result["validation_passed"],
                    "error": result.get("error"),
                    "timestamp": "now"
                })
                
                if not result["validation_passed"]:
                    validation_failed = True
                    validation_error = result.get("error", "Output contains inappropriate content")
                    logger.warning(f"Output validation failed: {validation_error}")
            
            except Exception as e:
                logger.error(f"Output validation error: {str(e)}")
                validation_results.append({
                    "stage": "output_profanity",
                    "passed": False,
                    "error": str(e),
                    "timestamp": "now"
                })
        
        # Validate factuality if enabled and we have RAG context
        if guardrails_available and factuality_guard and rag_chain and not validation_failed:
            try:
                # Get context from RAG if available
                # For factuality, we'd need to check if the response aligns with retrieved context
                # This is a simplified version
                result = validate_output(
                    factuality_guard,
                    agent_response,
                    raise_on_failure=False
                )
                
                validation_results.append({
                    "stage": "output_factuality",
                    "passed": result["validation_passed"],
                    "error": result.get("error"),
                    "timestamp": "now"
                })
                
                if not result["validation_passed"]:
                    validation_failed = True
                    validation_error = result.get("error", "Response may contain hallucinations")
                    logger.warning(f"Factuality check failed: {validation_error}")
            
            except Exception as e:
                logger.error(f"Factuality check error: {str(e)}")
                # Don't fail on factuality errors, just log
        
        if validation_failed:
            if refinement_count >= max_refinements:
                # Max refinements reached
                logger.error(f"Max refinements ({max_refinements}) reached. Returning with validation failure.")
                return {
                    "error_message": f"Validation failed after {max_refinements} attempts: {validation_error}",
                    "validation_results": validation_results
                }
            else:
                # Trigger refinement
                logger.info(f"Output validation failed, triggering refinement (attempt {refinement_count + 1})")
                return {
                    "validation_results": validation_results,
                    "refinement_count": refinement_count + 1
                }
        
        # Validation passed
        return {"validation_results": validation_results}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls, otherwise validate output."""
        error_message = state.get("error_message")
        if error_message:
            return END
        
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        # No tool calls, go to output validation
        return "validate_output"
    
    def should_refine(state: AgentState):
        """Decide whether to refine the response or end."""
        refinement_count = state.get("refinement_count", 0)
        error_message = state.get("error_message")
        
        if error_message:
            return END
        
        if refinement_count > 0 and refinement_count <= max_refinements:
            # Refine the response
            return "agent"
        
        # Done (either passed validation or max refinements reached)
        return END
    
    def route_after_input_validation(state: AgentState):
        """Route after input validation: refine, continue to agent, or end on error."""
        error_message = state.get("error_message")
        input_refinement_count = state.get("input_refinement_count", 0)
        validation_results = state.get("validation_results", [])
        
        if error_message:
            return END
        
        # Check if we need to refine (validation failed but haven't exceeded max)
        if validation_results:
            last_result = validation_results[-1]
            if (last_result.get("stage") == "input" and 
                not last_result.get("passed", True) and 
                input_refinement_count < max_input_refinements):
                return "refine_input"
        
        return "agent"
    
    # Build the graph with guardrails
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    # Add nodes
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("refine_input", refine_input_node)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("validate_output", validate_output_node)
    
    # Set entry point to input validation
    graph.set_entry_point("validate_input")
    
    # After input validation, route: refine_input, agent, or END
    graph.add_conditional_edges(
        "validate_input",
        route_after_input_validation,
        {"refine_input": "refine_input", "agent": "agent", END: END}
    )
    
    # After input refinement, re-validate
    graph.add_edge("refine_input", "validate_input")
    
    # After agent, route to tools or validate output
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"action": "action", "validate_output": "validate_output"}
    )
    
    # After tools, return to agent
    graph.add_edge("action", "agent")
    
    # After output validation, decide to refine or end
    graph.add_conditional_edges(
        "validate_output",
        should_refine,
        {"agent": "agent", END: END}
    )
    
    return graph.compile()
