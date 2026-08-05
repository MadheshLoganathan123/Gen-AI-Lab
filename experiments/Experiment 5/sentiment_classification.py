"""
Experiment 5: Sentiment Analysis And Document Classification Using Foundation Models
CS4V48 - GenAI and LLM Lab

AIM: To perform sentiment analysis and multi-class document classification using pre-trained
     foundation models.

OBJECTIVE: To understand how foundation models fine-tuned for classification tasks, and zero-shot
           classification models based on natural language inference, can be applied to categorise
           text without task-specific training.
"""

from transformers import pipeline


def run_sentiment_analysis(reviews: list[str]) -> None:
    """
    Perform binary sentiment analysis using a fine-tuned DistilBERT model
    (trained on SST-2).

    A classification head on top of the pre-trained transformer encoder predicts
    POSITIVE or NEGATIVE sentiment with a confidence score.
    """
    print("Loading sentiment-analysis pipeline...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    print("\n--- Sentiment Analysis Results ---\n")
    for review in reviews:
        result = sentiment_analyzer(review)[0]
        label = result["label"]
        score = round(result["score"], 3)
        print(f"Review : {review}")
        print(f"Result : {label} ({score})\n")


def run_zero_shot_classification(document: str, candidate_labels: list[str]) -> None:
    """
    Perform multi-class document classification using BART-large-MNLI.

    Zero-shot classification reformulates the task as Natural Language Inference (NLI):
    the document is treated as a 'premise' and each candidate label is turned into a
    'hypothesis' (e.g., "This text is about {label}"). The model outputs the entailment
    probability for every label.
    """
    print("Loading zero-shot-classification pipeline (BART-large-MNLI)...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    classification = classifier(document, candidate_labels)

    print("\n--- Zero-Shot Document Classification Results ---\n")
    print(f"Document: {document}\n")
    for label, score in zip(classification["labels"], classification["scores"]):
        print(f"  {label:<12}: {round(score, 3)}")


def main():
    print("=" * 60)
    print("Experiment 5: Sentiment Analysis & Document Classification")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Part A: Sentiment Analysis
    # -----------------------------------------------------------------------
    reviews = [
        "The new smartphone has an amazing camera and battery life!",
        "The delivery was late and the packaging was damaged.",
    ]
    run_sentiment_analysis(reviews)

    # -----------------------------------------------------------------------
    # Part B: Zero-Shot Document Classification
    # -----------------------------------------------------------------------
    document = "The central bank raised interest rates to control rising inflation."
    candidate_labels = ["Politics", "Economy", "Sports", "Technology"]
    run_zero_shot_classification(document, candidate_labels)


if __name__ == "__main__":
    main()
