"""
Experiment 6: Retrieval-Augmented Generation (RAG) System Using Vector Databases
CS4V48 - GenAI and LLM Lab

AIM: To build a Retrieval-Augmented Generation (RAG) system that retrieves relevant document
     chunks from a vector database and uses an LLM to generate grounded answers.

OBJECTIVE: To understand how embeddings, vector similarity search, and LLM generation are
           combined to reduce hallucination and answer questions using an external knowledge base.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------
DOCUMENTS = [
    "The Eiffel Tower is located in Paris, France and was completed in 1889.",
    "Retrieval-Augmented Generation combines document retrieval with text generation.",
    "Python is a popular high-level programming language used in AI development.",
    "Vector databases store embeddings and support fast similarity search.",
    "Large Language Models are trained on massive text corpora to perform NLP tasks.",
    "FAISS is an open-source library developed by Facebook AI for efficient similarity search.",
]


def build_vector_index(documents: list[str], embed_model: SentenceTransformer):
    """Encode documents and store them in a FAISS flat L2 index."""
    print("Encoding documents and building FAISS index...")
    doc_embeddings = embed_model.encode(documents)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings, dtype="float32"))
    print(f"  Indexed {index.ntotal} document(s) with embedding dimension {dimension}.\n")
    return index, doc_embeddings


def retrieve_top_k(query: str, embed_model: SentenceTransformer,
                   index: faiss.IndexFlatL2, documents: list[str], k: int = 2):
    """Embed the query and retrieve the top-k most similar document chunks."""
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    retrieved = [documents[i] for i in indices[0]]
    return retrieved, distances[0]


def generate_answer(retrieved_chunks: list[str], query: str, generator) -> str:
    """Build an augmented prompt from the retrieved chunks and generate an answer."""
    context = " ".join(retrieved_chunks)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=80)
    return result[0]["generated_text"]


def main():
    print("=" * 60)
    print("Experiment 6: RAG System Using Vector Databases")
    print("=" * 60)

    # Step 1: Load sentence-embedding model
    print("\nLoading sentence-transformer embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2: Build FAISS index from the knowledge base
    index, _ = build_vector_index(DOCUMENTS, embed_model)

    # Step 3: Load the text-generation LLM
    print("Loading text-generation LLM (google/flan-t5-base)...")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    print()

    # Step 4: Define queries and run the RAG pipeline
    queries = [
        "What is RAG in AI?",
        "Where is the Eiffel Tower located?",
        "What is FAISS used for?",
    ]

    for query in queries:
        print(f"Query: {query}")

        retrieved_chunks, distances = retrieve_top_k(query, embed_model, index, DOCUMENTS, k=2)
        print("Retrieved Context:")
        for i, (chunk, dist) in enumerate(zip(retrieved_chunks, distances), 1):
            print(f"  [{i}] (L2={dist:.4f}) {chunk}")

        answer = generate_answer(retrieved_chunks, query, generator)
        print(f"Answer: {answer}")
        print()


if __name__ == "__main__":
    main()
