"""
Retrieval-Augmented Generation (RAG) Pipeline
---------------------------------------------
Loads local text documents, splits them into chunks,
generates embeddings, retrieves relevant chunks and
uses Amazon Bedrock to generate grounded answers.

Features:
- Local document ingestion
- Text chunking with overlap
- Titan text embeddings
- Cosine similarity retrieval
- Bedrock Converse API
- Grounded question answering

Domain: Generative AI, Embeddings, RAG
"""

import json
import math
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

AWS_REGION = "us-east-1"

CHAT_MODEL_ID = "amazon.nova-micro-v1:0"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

DOCUMENTS_DIR = Path(__file__).parent / "documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3

client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)


# --------------------------------------------------
# DOCUMENT LOADING
# --------------------------------------------------

def load_documents():
    """Load all .txt files from the documents directory."""

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Document folder not found: {DOCUMENTS_DIR}"
        )

    documents = []

    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": path.name,
                "text": text
            })

    if not documents:
        raise ValueError(
            "No non-empty .txt documents were found."
        )

    return documents


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def split_into_chunks(text, chunk_size=CHUNK_SIZE,
                       overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping character-based chunks.

    Character-based chunking is a simple starting point.
    Production systems may use token-aware or semantic
    chunking instead.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller "
            "than chunk_size."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def prepare_chunks(documents):
    """Split documents and preserve source information."""

    all_chunks = []

    for document in documents:
        chunks = split_into_chunks(document["text"])

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "source": document["source"],
                "chunk_id": index,
                "text": chunk
            })

    return all_chunks


# --------------------------------------------------
# EMBEDDING GENERATION
# --------------------------------------------------

def generate_embedding(text):
    """Generate a vector embedding using Amazon Titan."""

    response = client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({
            "inputText": text,
            "dimensions": 1024,
            "normalize": True
        }),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    return result["embedding"]


def embed_chunks(chunks):
    """Generate and attach an embedding to every chunk."""

    for chunk in chunks:
        chunk["embedding"] = generate_embedding(chunk["text"])

    return chunks


# --------------------------------------------------
# SIMILARITY SEARCH
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def retrieve_chunks(question, chunks, top_k=TOP_K):
    """Retrieve the most relevant document chunks."""

    question_embedding = generate_embedding(question)

    scored_chunks = []

    for chunk in chunks:
        score = cosine_similarity(
            question_embedding,
            chunk["embedding"]
        )

        scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return scored_chunks[:top_k]


# --------------------------------------------------
# GROUNDED ANSWER GENERATION
# --------------------------------------------------

def generate_answer(question, retrieved_chunks):
    """Generate an answer using retrieved document context."""

    context_parts = []

    for score, chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"Relevance score: {score:.4f}\n"
            f"Content:\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a document-based question-answering assistant. "
        "Answer using only the supplied context. "
        "If the context does not contain enough information, "
        "clearly say that the answer cannot be found in the "
        "provided documents. Do not invent facts. "
        "Mention the source filenames supporting your answer."
    )

    response = client.converse(
        modelId=CHAT_MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[
            {
                "role": "user",
                "content": [{
                    "text": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        "Provide a concise answer based on the context."
                    )
                }]
            }
        ],
        inferenceConfig={
            "maxTokens": 500,
            "temperature": 0.1
        }
    )

    answer = "".join(
        block["text"]
        for block in response["output"]["message"]["content"]
        if "text" in block
    )

    return answer, response.get("usage", {})


# --------------------------------------------------
# MAIN RAG WORKFLOW
# --------------------------------------------------

def run_rag_pipeline(question):
    """Execute document loading, embedding, retrieval and generation."""

    documents = load_documents()
    chunks = prepare_chunks(documents)

    print(f"Loaded {len(documents)} document(s).")
    print(f"Created {len(chunks)} chunk(s).")
    print("Generating embeddings...")

    chunks = embed_chunks(chunks)

    print("Retrieving relevant content...")
    retrieved = retrieve_chunks(question, chunks)

    print("\nRETRIEVED SOURCES")
    for score, chunk in retrieved:
        print(
            f"- {chunk['source']} | "
            f"Chunk {chunk['chunk_id']} | "
            f"Score: {score:.4f}"
        )

    answer, usage = generate_answer(question, retrieved)

    return answer, usage


if __name__ == "__main__":
    question = input("Enter your question: ").strip()

    if not question:
        print("Please enter a question.")
    else:
        try:
            answer, usage = run_rag_pipeline(question)

            print("\nGENERATED ANSWER\n")
            print(answer)

            print("\nTOKEN USAGE\n")
            print(json.dumps(usage, indent=2))

        except (ClientError, FileNotFoundError, ValueError) as error:
            print(f"Pipeline error: {error}")
