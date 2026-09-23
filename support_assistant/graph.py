import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from ingest import (
    get_collection,
    get_embedding_model
)


# ---------------------------------------------------------
# State
# ---------------------------------------------------------

class SupportState(TypedDict, total=False):

    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


# ---------------------------------------------------------
# Mock LLM setting
# ---------------------------------------------------------

MOCK_LLM = os.getenv(
    "MOCK_LLM",
    "1"
)


# ---------------------------------------------------------
# Classify intent
# ---------------------------------------------------------

def classify_intent(state: SupportState):

    query = state["query"].lower()

    policy_keywords = [
        "zepto",
        "order",
        "delivery",
        "cancel",
        "cancellation",
        "refund",
        "return",
        "payment",
        "wallet",
        "delivery fee",
        "priority delivery",
        "pin code",
        "address",
        "replacement",
        "customer",
        "support"
    ]

    is_policy_question = any(
        keyword in query
        for keyword in policy_keywords
    )

    if is_policy_question:

        return {
            "intent": "policy"
        }

    return {
        "intent": "general"
    }


# ---------------------------------------------------------
# Retrieve and answer
# ---------------------------------------------------------

def retrieve_and_answer(state: SupportState):

    query = state["query"]

    print("Loading embedding model...")

    model = get_embedding_model()

    collection = get_collection()

    # -----------------------------------------------------
    # Create query embedding
    # -----------------------------------------------------

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    ).tolist()

    # -----------------------------------------------------
    # Retrieve top 3 documents
    # -----------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    ids = results.get(
        "ids",
        [[]]
    )[0]

    # -----------------------------------------------------
    # No documents
    # -----------------------------------------------------

    if not documents:

        return {
            "answer": (
                "I could not find relevant information "
                "in the Zepto policy documents."
            ),
            "sources": [],
            "confidence": 0.0
        }

    # -----------------------------------------------------
    # MOCK MODE
    # -----------------------------------------------------

    if MOCK_LLM == "1":

        top_document = documents[0]

        answer = (
            "Based on the retrieved context: "
            + top_document
        )

        return {
            "answer": answer,
            "sources": ids,
            "confidence": 1.0
        }

    # -----------------------------------------------------
    # Optional real LLM mode
    # -----------------------------------------------------

    from langchain_groq import ChatGroq

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is required when "
            "MOCK_LLM=0."
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key
    )

    context = "\n\n".join(
        documents
    )

    prompt = f"""
You are a Zepto customer support assistant.

Use only the supplied context to answer the user's question.

Context:
{context}

User question:
{query}

Rules:
- Answer clearly and concisely.
- Do not invent information.
- If the context does not contain the answer, say that the information is not available.
- Do not answer unrelated questions.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": ids,
        "confidence": 1.0
    }


# ---------------------------------------------------------
# Direct answer
# ---------------------------------------------------------

def direct_answer(state: SupportState):

    if MOCK_LLM == "1":

        return {
            "answer": (
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            "sources": [],
            "confidence": 1.0
        }

    # Optional real LLM path

    from langchain_groq import ChatGroq

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is required when "
            "MOCK_LLM=0."
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key
    )

    response = llm.invoke(
        f"""
You are a Zepto customer support assistant.

The user asked:

{state["query"]}

This question is not related to Zepto policies.

Politely explain that you can only answer Zepto
policy-related questions.
"""
    )

    return {
        "answer": response.content,
        "sources": [],
        "confidence": 1.0
    }


# ---------------------------------------------------------
# Routing
# ---------------------------------------------------------

def route_intent(state: SupportState):

    if state["intent"] == "policy":
        return "retrieve_and_answer"

    return "direct_answer"


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

def build_graph():

    graph = StateGraph(
        SupportState
    )

    # Nodes

    graph.add_node(
        "classify_intent",
        classify_intent
    )

    graph.add_node(
        "retrieve_and_answer",
        retrieve_and_answer
    )

    graph.add_node(
        "direct_answer",
        direct_answer
    )

    # Start

    graph.add_edge(
        START,
        "classify_intent"
    )

    # Conditional routing

    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer":
                "retrieve_and_answer",

            "direct_answer":
                "direct_answer"
        }
    )

    # End

    graph.add_edge(
        "retrieve_and_answer",
        END
    )

    graph.add_edge(
        "direct_answer",
        END
    )

    return graph.compile()


# ---------------------------------------------------------
# Global graph
# ---------------------------------------------------------

build_graph_instance = build_graph()