# Zepto Support Assistant

## Module 3

This project implements a small Zepto support assistant using document retrieval, embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

## Project structure

support_assistant/

    docs/

        doc_01.txt

        doc_02.txt

        doc_03.txt

        doc_04.txt

        doc_05.txt

        doc_06.txt

        doc_07.txt

        doc_08.txt

    Dockerfile

    graph.py

    ingest.py

    main.py

    models.py

    README.md

    requirements.txt


## Pipeline architecture

Ingestion: ingest.py reads the eight text files from docs/. Each document is used as one chunk because the supplied documents are short enough for this module.

Embedding: ingest.py loads the all-MiniLM-L6-v2 sentence-transformers model and creates an embedding for every chunk.

Vector storage: ChromaDB stores the document chunks and their embeddings in the zepto_policies collection using cosine distance. The local database is stored in support_assistant/chroma_db.

Retrieval: graph.py uses the retrieve_and_answer LangGraph node. It embeds the incoming policy question with the same embedding model and requests the top three most similar chunks from ChromaDB.

Generation: graph.py uses the retrieved chunks to create the final response. In the required mock mode, the top retrieved chunk is used to create a deterministic answer. In the optional real-LLM mode, the structured prompt in graph.py is sent to the LLM.

The data flow is:

docs/*.txt -> ingest.py -> all-MiniLM-L6-v2 -> ChromaDB -> retrieve_and_answer -> answer


## LangGraph flow

The graph has three nodes:

- classify_intent
- retrieve_and_answer
- direct_answer

classify_intent checks the query first. In the default mock mode, it uses the required keyword heuristic. A policy keyword routes the query to retrieve_and_answer. An unrelated query routes to direct_answer.

retrieve_and_answer performs real embedding and ChromaDB retrieval in both mock and real modes. Its final answer generation changes with MOCK_LLM.

direct_answer does not perform retrieval. In mock mode it returns the required fixed response. In real mode it can call the optional LLM directly.

Only the generation decisions branch on MOCK_LLM. The default is MOCK_LLM=1 when the variable is unset. This is the graded path and does not call an LLM provider.

Setting MOCK_LLM=0 enables the optional Groq path and requires GROQ_API_KEY.


## Structured prompt

The prompt in graph.py follows the role-context-task-format-length structure. It also contains an explicit negative constraint and a few-shot example. It is used by the optional real-LLM path.


## Response format

The /ask endpoint returns:

{
    "answer": "...",
    "sources": [
        "doc_01_chunk_01"
    ],
    "confidence": 1.0
}

For general questions, sources is an empty list.


## Run locally

Open a terminal in the support_assistant folder and install the requirements:

pip install -r requirements.txt

Run the application:

uvicorn main:app --reload

The first startup downloads or loads the all-MiniLM-L6-v2 model as required by sentence-transformers.

No LLM API key is needed for the graded mock mode.


## Example call 1: policy question

Request:

curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"query\":\"How long does delivery take?\"}"

Example response:

{
    "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
    "sources": [
        "doc_01_chunk_01",
        "doc_05_chunk_01",
        "doc_04_chunk_01"
    ],
    "confidence": 1.0
}


## Example call 2: general question

Request:

curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"query\":\"What is the capital of France?\"}"

Example response:

{
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
}


## Docker

Build the image from inside the support_assistant folder:

docker build -t zepto-support-assistant .

Run the container:

docker run -d --name zepto-support-assistant -p 8000:7860 zepto-support-assistant

The API is available at:

http://127.0.0.1:8000/ask

Check the running container:

docker ps

The container should show the port mapping:

0.0.0.0:8000->7860/tcp


## Mock mode

MOCK_LLM is left unset by default and therefore uses the required mock path. It can also be explicitly set to 1.

Windows PowerShell:

$env:MOCK_LLM="1"

uvicorn main:app --reload

The optional real-LLM path is enabled with MOCK_LLM=0 and a GROQ_API_KEY environment variable.

The API key must not be hardcoded or committed to the repository.