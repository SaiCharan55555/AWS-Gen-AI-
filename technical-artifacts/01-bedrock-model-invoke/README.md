
<text color="secondary" size="xs">Retrieval-Augmented Generation Pipeline</text> <button color="secondary" variant="outline" onClick={() => GenUI.copy(`# Retrieval-Augmented Generation (RAG) Pipeline

Overview

A Python-based RAG reference implementation using Amazon Bedrock. It loads local text documents, splits them into chunks, creates embeddings, retrieves relevant content and generates answers using retrieved context.

Features

Local document ingestion

Character-based chunking with overlap

Amazon Titan text embeddings

Cosine similarity search

Top-k retrieval

Grounded answer generation using Amazon Bedrock

Token usage reporting

Architecture

Local documents → Text chunking → Titan embeddings → Similarity search → Retrieved context → Bedrock Converse API → Generated answer

Technologies

Python

Boto3

Amazon Bedrock Runtime

Amazon Titan Text Embeddings

Amazon Nova Micro

Project Structure

```text 03-rag-pipeline/ ├── rag_pipeline.py ├── requirements.txt ├── documents/ │ └── aws_genai_notes.txt └── README.md ```

Setup

Configure AWS credentials.

Ensure both the embedding and chat models are available in your account and region.

Install dependencies:

```bash pip install -r requirements.txt ```

4. Run:

```bash python rag_pipeline.py ```

5. Ask a question about the supplied documents.

How It Works

Load local text documents.

Split documents into overlapping chunks.

Generate an embedding for each chunk.

Embed the user's question.

Rank chunks using cosine similarity.

Select the top-k chunks.

Send the retrieved context and question to the language model.

Display the generated answer and token usage.

Limitations

Uses an in-memory vector store.

Processes plain text files only.

Recomputes document embeddings each run.

Uses simple character-based chunking.

Similarity scores are not calibrated confidence values.

Grounding instructions do not guarantee factual answers.

Future Improvements

Persistent vector storage

PDF and other document formats

Token-aware chunking

Embedding caching

Retrieval evaluation

Answer quality evaluation

Source-level citations in generated answers

Security and Cost

Do not commit AWS credentials or confidential documents. Model inference and embedding requests may incur AWS charges. Review AWS pricing before running large workloads.

Learning Objectives

Understand the RAG workflow, text embeddings, semantic retrieval, context construction and grounded generation.`)}><icon name="copy" size="sm"/> Copy README</button> </box>

5. AWS setup and important notes

Before running either project:

Configure your AWS credentials using the AWS CLI or an IAM role.

Use a region where the chosen models are available.

Make sure your IAM permissions allow bedrock:InvokeModel for the selected models.

Check model availability and pricing before running the scripts.

The RAG implementation uses Amazon Titan Text Embeddings V2, which supports 256-, 512- and 1,024-dimensional vectors. The example uses 1,024 dimensions. 
