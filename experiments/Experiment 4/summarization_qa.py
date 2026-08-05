"""
Experiment 4: Text Summarization And Question-Answering System Using Large Language Models
CS4V48 - GenAI and LLM Lab

AIM: To develop a text summarization system and a question-answering system using pre-trained
     Large Language Models (BART and DistilBERT).

OBJECTIVE: To understand abstractive summarization and extractive question-answering pipelines
           built on transformer encoder-decoder and encoder-only architectures respectively.
"""

from transformers import pipeline


# ---------------------------------------------------------------------------
# Shared context passage used for both summarization and QA
# ---------------------------------------------------------------------------
ARTICLE = (
    "Generative AI refers to a class of artificial intelligence models capable of "
    "producing new content such as text, images, audio, and video. Large Language Models (LLMs) "
    "such as GPT and LLaMA are trained on massive text corpora and can perform a wide range of "
    "natural language tasks including translation, summarization, and question answering. These "
    "models are increasingly being deployed in industry applications ranging from customer support "
    "to software development, transforming how humans interact with machines."
)


def run_summarization(article: str) -> str:
    """
    Perform abstractive text summarization using facebook/bart-large-cnn.

    BART uses an encoder-decoder architecture: the encoder reads the full document
    and the decoder generates a new, fluent summary.
    """
    print("Loading BART summarization pipeline...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summary = summarizer(article, max_length=45, min_length=20, do_sample=False)
    return summary[0]["summary_text"]


def run_question_answering(context: str, question: str) -> dict:
    """
    Perform extractive question-answering using distilbert-base-cased-distilled-squad.

    DistilBERT predicts the start and end token positions of the answer span
    within the context passage.
    """
    print("Loading DistilBERT QA pipeline...")
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    answer = qa(question=question, context=context)
    return answer


def main():
    print("=" * 60)
    print("Experiment 4: Text Summarization & Question Answering")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Part A: Text Summarization
    # -----------------------------------------------------------------------
    print("\n--- Part A: Text Summarization (BART) ---\n")
    print("Input Article:\n", ARTICLE, "\n")

    summary_text = run_summarization(ARTICLE)
    print("Summary:\n", summary_text)

    # -----------------------------------------------------------------------
    # Part B: Question Answering
    # -----------------------------------------------------------------------
    print("\n--- Part B: Question Answering (DistilBERT-SQuAD) ---\n")
    question = "What are Large Language Models trained on?"
    answer = run_question_answering(ARTICLE, question)

    print(f"Question: {question}")
    print(f"Answer:   {answer['answer']}  |  Confidence: {round(answer['score'], 3)}")


if __name__ == "__main__":
    main()
