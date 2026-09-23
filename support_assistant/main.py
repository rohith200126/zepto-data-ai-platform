from fastapi import FastAPI

from models import (
    AskRequest,
    AskResponse
)

from graph import build_graph_instance


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Zepto Support Assistant",
    description="Zepto policy support assistant using RAG and LangGraph",
    version="1.0.0"
)


# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Zepto Support Assistant API is running."
    }


# ---------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------

@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest):

    result = build_graph_instance.invoke(
        {
            "query": request.query
        }
    )

    return {
        "answer": result.get(
            "answer",
            ""
        ),

        "sources": result.get(
            "sources",
            []
        ),

        "confidence": result.get(
            "confidence",
            1.0
        )
    }