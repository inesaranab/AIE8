"""
RAG Evaluation: Together AI (gpt-oss) vs OpenAI (gpt-4.1-mini)
Based on AI Makerspace Ragas Evaluation Notebook (2025)

This script:
1. Generates synthetic test data using Ragas
2. Evaluates RAG with gpt-oss (Together AI)
3. Evaluates RAG with gpt-4.1-mini (OpenAI)
4. Compares results

Usage:
    uv run python evaluate_rag_ragas_style.py
"""

import os
from dotenv import load_dotenv
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

print("="*80)
print("RAG EVALUATION: gpt-oss vs gpt-4.1-mini (AI Makerspace Style)")
print("="*80)

# ============================================================================
# STEP 1: Load Documents
# ============================================================================
print("\n📄 Step 1: Loading PDF documents...")

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

path = "data/"
loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()

print(f"✅ Loaded {len(docs)} documents")

# ============================================================================
# STEP 2: Generate Synthetic Test Data with Ragas
# ============================================================================
print("\n🧪 Step 2: Generating synthetic test data with Ragas...")

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Generator models (for creating test data)
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# STEP 2.1: Split documents to 101-450 token range (safely under 512 limit for embeddings)
print("   Splitting documents into 150-400 token chunks (avoids headline extraction)...")
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

# Split to 200 tokens max (safely under 512 token embedding limit)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    length_function=tiktoken_len,
)
split_docs = text_splitter.split_documents(docs)
print(f"   Split into {len(split_docs)} chunks (max 200 tokens each, safely under 512 embedding limit)")

# STEP 2.2: Build Knowledge Graph following official docs
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms

print("   Building Knowledge Graph...")
kg = KnowledgeGraph()
for doc in split_docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

print(f"   Created KG with {len(kg.nodes)} nodes")

# STEP 2.3: Apply default transforms (will use summary path, not headlines)
print("   Applying default transforms (summary-based, no headlines)...")
transforms = default_transforms(
    documents=split_docs,
    llm=generator_llm,
    embedding_model=generator_embeddings
)
apply_transforms(kg, transforms)
print("   ✅ Transforms applied successfully")

# STEP 2.4: Generate test set with ONLY single-hop queries
print("   Generating 10 synthetic test questions (single-hop only)...")
print("   (This may take 2-3 minutes...)")

from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer

# Create a query distribution with ONLY single-hop queries (100% probability)
single_hop_synthesizer = SingleHopSpecificQuerySynthesizer(llm=generator_llm)

generator = TestsetGenerator(
    llm=generator_llm, 
    embedding_model=generator_embeddings, 
    knowledge_graph=kg
)

# Generate with only single-hop queries
# query_distribution expects list of tuples: [(synthesizer, probability), ...]
dataset = generator.generate(
    testset_size=10,
    query_distribution=[(single_hop_synthesizer, 1.0)]  # 100% single-hop queries!
)

print(f"✅ Generated {len(dataset.samples)} test questions")
print("\nSample questions:")
for i, sample in enumerate(dataset.samples[:3], 1):
    print(f"   {i}. {sample.eval_sample.user_input[:80]}...")

# ============================================================================
# STEP 3: Use Your Existing RAG System from app/rag.py
# ============================================================================
print("\n🏗️  Step 3: Using your existing RAG system from app/rag.py...")

from app.rag import _build_rag_graph
import app.model

def build_rag_with_model(model_name: str):
    """Build RAG graph with specified model by temporarily overriding get_chat_model"""
    # Store original function
    original_get_chat_model = app.model.get_chat_model
    
    # Override to force specific model
    def forced_model(*args, **kwargs):
        return original_get_chat_model(model_name=model_name, temperature=0)
    
    app.model.get_chat_model = forced_model
    
    try:
        # Build RAG graph with forced model
        graph = _build_rag_graph(data_dir=path)
        return graph
    finally:
        # Restore original function
        app.model.get_chat_model = original_get_chat_model

print("✅ RAG system ready (will build per model)")

# ============================================================================
# STEP 4: Evaluate with gpt-oss (Together AI)
# ============================================================================
print("\n🔄 Step 4: Evaluating with gpt-oss (Together AI)...")

# Build RAG graph with gpt-oss
graph_oss = build_rag_with_model("openai/gpt-oss-20b")

# Run test questions through gpt-oss
for i, test_row in enumerate(dataset.samples, 1):
    print(f"   [{i}/{len(dataset.samples)}] Processing question...")
    response = graph_oss.invoke({"question": test_row.eval_sample.user_input})
    test_row.eval_sample.response = response["response"]
    # Extract page_content from Document objects
    test_row.eval_sample.retrieved_contexts = [doc.page_content for doc in response["context"]]

print("✅ gpt-oss responses collected")

# Create evaluation dataset for gpt-oss
from ragas import EvaluationDataset
evaluation_dataset_oss = EvaluationDataset.from_pandas(dataset.to_pandas())

# Evaluate with Ragas
print("   Running Ragas evaluation...")
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy
from ragas import evaluate, RunConfig

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
custom_run_config = RunConfig(timeout=360)

gpt_oss_result = evaluate(
    dataset=evaluation_dataset_oss,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy()],
    llm=evaluator_llm,
    run_config=custom_run_config
)

print("✅ gpt-oss evaluation complete")

# ============================================================================
# STEP 5: Evaluate with gpt-4.1-mini (OpenAI)
# ============================================================================
print("\n🔄 Step 5: Evaluating with gpt-4.1-mini (OpenAI)...")

# Use the SAME test questions for a fair comparison
print("   Using the same test questions for gpt-4.1-mini (for fair comparison)...")

# Create a fresh copy of the dataset for gpt-4.1-mini responses
import copy
dataset_gpt4 = copy.deepcopy(dataset)

# Build RAG graph with gpt-4.1-mini
graph_gpt4 = build_rag_with_model("gpt-4.1-mini")

# Run the SAME test questions through gpt-4.1-mini
for i, test_row in enumerate(dataset_gpt4.samples, 1):
    print(f"   [{i}/{len(dataset_gpt4.samples)}] Processing question...")
    response = graph_gpt4.invoke({"question": test_row.eval_sample.user_input})
    test_row.eval_sample.response = response["response"]
    # Extract page_content from Document objects
    test_row.eval_sample.retrieved_contexts = [doc.page_content for doc in response["context"]]

print("✅ gpt-4.1-mini responses collected")

# Create evaluation dataset for gpt-4.1-mini
evaluation_dataset_gpt4 = EvaluationDataset.from_pandas(dataset_gpt4.to_pandas())

# Evaluate with Ragas
print("   Running Ragas evaluation...")

gpt4_result = evaluate(
    dataset=evaluation_dataset_gpt4,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy()],
    llm=evaluator_llm,
    run_config=custom_run_config
)

print("✅ gpt-4.1-mini evaluation complete")

# ============================================================================
# STEP 6: Compare Results
# ============================================================================
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

import numpy as np

# Extract metrics using numpy's nanmean (handles NaN automatically)
metrics_data = {
    "Metric": ["Context Recall", "Faithfulness", "Factual Correctness", "Answer Relevancy"],
    "gpt-oss (Together AI)": [
        np.nanmean(gpt_oss_result["context_recall"]),
        np.nanmean(gpt_oss_result["faithfulness"]),
        np.nanmean(gpt_oss_result["factual_correctness(mode=f1)"]),
        np.nanmean(gpt_oss_result["answer_relevancy"]),
    ],
    "gpt-4.1-mini (OpenAI)": [
        np.nanmean(gpt4_result["context_recall"]),
        np.nanmean(gpt4_result["faithfulness"]),
        np.nanmean(gpt4_result["factual_correctness(mode=f1)"]),
        np.nanmean(gpt4_result["answer_relevancy"]),
    ]
}

comparison_df = pd.DataFrame(metrics_data)
comparison_df["Difference (oss - gpt4)"] = (
    comparison_df["gpt-oss (Together AI)"] - comparison_df["gpt-4.1-mini (OpenAI)"]
)

print("\n" + comparison_df.to_string(index=False))

# Summary
gpt_oss_avg = comparison_df["gpt-oss (Together AI)"].mean()
gpt4_avg = comparison_df["gpt-4.1-mini (OpenAI)"].mean()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Average Score (gpt-oss):      {gpt_oss_avg:.4f}")
print(f"Average Score (gpt-4.1-mini): {gpt4_avg:.4f}")
print(f"Difference:                   {gpt_oss_avg - gpt4_avg:+.4f}")

if gpt_oss_avg > gpt4_avg:
    print(f"\n🏆 Winner: gpt-oss (Together AI) by {(gpt_oss_avg - gpt4_avg):.4f} points")
elif gpt4_avg > gpt_oss_avg:
    print(f"\n🏆 Winner: gpt-4.1-mini (OpenAI) by {(gpt4_avg - gpt_oss_avg):.4f} points")
else:
    print("\n🤝 Tie! Both models performed equally well.")

print("="*80)

# Save results
comparison_df.to_csv('ragas_evaluation_results.csv', index=False)
print("\n✅ Results saved to 'ragas_evaluation_results.csv'")

# Create visualization
try:
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(comparison_df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], comparison_df["gpt-oss (Together AI)"], 
           width, label='gpt-oss (Together AI)', color='#2ecc71')
    ax.bar([i + width/2 for i in x], comparison_df["gpt-4.1-mini (OpenAI)"], 
           width, label='gpt-4.1-mini (OpenAI)', color='#3498db')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('RAG Evaluation: gpt-oss vs gpt-4.1-mini (Ragas Metrics)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Metric"], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ragas_evaluation_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Chart saved as 'ragas_evaluation_comparison.png'")
    plt.close()
except ImportError:
    print("⚠️  matplotlib not available - skipping chart generation")

print("\n🎉 Evaluation complete!")
print("\nFiles created:")
print("  - ragas_evaluation_results.csv (comparison table)")
print("  - ragas_evaluation_comparison.png (visualization)")

