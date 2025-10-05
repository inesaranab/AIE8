# LangGraph Agent-Based RAGAS Synthetic Data Generation

## Overview

This is the **optional assignment** for Assignment 7 that implements RAGAS synthetic data generation using **LangGraph agents** instead of the Knowledge Graph approach.

## What This Implements

### Requirements Met ✅

1. **LangGraph Agent Architecture**: Multi-agent system replacing the Knowledge Graph approach
2. **Evol Instruct Method**: Evolutionary instruction generation for question enhancement
3. **Three Evolution Types**:
   - Simple Evolution
   - Multi-Context Evolution
   - Reasoning Evolution

### Output Format ✅

The implementation produces three structured lists as required:

1. **Evolved Questions**: `List[dict]` containing:
   - `question_id`: Unique identifier
   - `original_question`: Base question
   - `evolved_question`: Enhanced question
   - `evolution_type`: Type of evolution applied

2. **Question Answers**: `List[dict]` containing:
   - `question_id`: Reference to the question
   - `answer`: Generated answer based on context

3. **Question Contexts**: `List[dict]` containing:
   - `question_id`: Reference to the question
   - `contexts`: List of relevant document contexts

## Key Features

### 1. Agent-Based Architecture

Instead of using a Knowledge Graph, this implementation uses specialized LangGraph agents:

- **Simple Evolution Agent**: Refines questions for clarity and specificity
- **Multi-Context Evolution Agent**: Creates questions requiring multiple contexts
- **Reasoning Evolution Agent**: Generates questions requiring deep analysis
- **Answer Generation Agent**: Produces answers based on retrieved contexts
- **Context Retrieval Agent**: Retrieves and validates relevant contexts

### 2. LangGraph Workflows

Three separate workflows are created, one for each evolution type:
- Each workflow follows a pipeline: Evolution → Answer Generation → Context Retrieval
- Workflows are implemented using LangGraph's StateGraph
- All workflows maintain consistent state management

### 3. Evol Instruct Method

The implementation uses evolutionary instruction generation:
- Questions are progressively enhanced based on their evolution type
- Each evolution type has specific prompts designed to encourage different thinking patterns
- The method ensures diverse and challenging questions for evaluation

## How to Use

### 1. Setup

Run the first few cells to:
- Install dependencies
- Set up API keys (OpenAI, LangChain)
- Load and prepare documents
- Create vector store

### 2. Run Generation

Execute the generation cell:
```python
results = generate_synthetic_data(
    base_questions=base_questions,
    num_evolutions_per_type=2
)
```

This will:
- Take each base question
- Apply all three evolution types
- Generate `num_evolutions_per_type` variations per type
- Produce answers and retrieve contexts for each

### 3. Review Results

The notebook includes:
- Formatted display of all generated data
- Validation checks
- Summary statistics
- JSON export of results

## Output Files

Two JSON files are generated:

1. **langgraph_synthetic_data_results.json**: Complete results
2. **langgraph_formatted_results.json**: Formatted output matching assignment requirements

## Comparison with Knowledge Graph Approach

| Aspect | Knowledge Graph | LangGraph Agents |
|--------|----------------|------------------|
| Architecture | Graph-based node relationships | Agent-based workflows |
| Evolution | Node transformations | Agent-driven question enhancement |
| Flexibility | Fixed graph structure | Dynamic agent interactions |
| Complexity | Requires graph building | Direct question evolution |
| Scalability | Limited by graph size | Scales with agent capacity |

## Expected Output

For 5 base questions with 2 evolutions per type, you'll get:
- **30 evolved questions** (5 × 3 types × 2 evolutions)
- **30 answers** (one per evolved question)
- **30 context sets** (one per evolved question)

## Validation

The notebook includes validation to ensure:
- All required output fields are present
- Question IDs are consistent across all lists
- No orphaned answers or contexts
- Proper data structure formatting

## Extension Ideas

To further enhance this implementation:

1. **Add more evolution types**: Create new agents for other question types
2. **Implement feedback loops**: Have agents review and improve each other's outputs
3. **Add quality scoring**: Evaluate question difficulty and relevance
4. **Integrate with LangSmith**: Upload results directly to LangSmith datasets
5. **Parallel processing**: Run multiple evolutions simultaneously

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure OpenAI and LangChain API keys are set correctly
2. **Memory Issues**: Reduce `num_evolutions_per_type` if running out of memory
3. **Rate Limits**: Add delays between API calls if hitting rate limits
4. **Empty Results**: Check that documents are loaded correctly

## Requirements

- Python 3.8+
- langgraph
- langchain
- langchain-openai
- langchain-community
- ragas
- qdrant-client
- langsmith

## Notes

- The implementation uses GPT-4o-mini for generation (cost-effective)
- Vector store is in-memory (Qdrant)
- All LangSmith tracing is enabled for debugging
- Results are reproducible with the same base questions

## Assignment Completion Checklist

- ✅ Reproduce RAGAS Synthetic Data Generation Steps
- ✅ Use LangGraph Agent Graph instead of Knowledge Graph
- ✅ Leverage Evol Instruct method
- ✅ Output: List[dict] of Evolved Questions with IDs and Evolution Types
- ✅ Output: List[dict] of Question IDs and Answers
- ✅ Output: List[dict] of Question IDs and Contexts
- ✅ Handle Simple Evolution
- ✅ Handle Multi-Context Evolution
- ✅ Handle Reasoning Evolution

## Author Notes

This implementation demonstrates how LangGraph can be used as an alternative to traditional Knowledge Graph approaches for synthetic data generation. The agent-based architecture provides more flexibility and is easier to extend with new evolution types or validation steps.

