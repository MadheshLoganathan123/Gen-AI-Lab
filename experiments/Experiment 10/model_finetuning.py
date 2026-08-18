"""
Experiment 10: Fine-Tuning Foundation Models On Domain-Specific Datasets
CS4V48 - GenAI and LLM Lab

AIM: To fine-tune a pre-trained foundation model on a domain-specific dataset using the
     Hugging Face Trainer API, and evaluate its post-training performance metrics.

OBJECTIVE: To understand the end-to-end process of supervised fine-tuning (SFT) for transformer
           models, including dataset tokenization, loss optimization, checkpointing, and
           evaluation with standard metrics (Accuracy, F1).
"""

import os
import torch
import numpy as np
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def prepare_dataset():
    """
    Construct a domain-specific dataset (Customer Support Ticket Intent Classification).
    Classes:
      0: Technical Issue
      1: Billing / Refund
      2: General Inquiry
    """
    train_data = {
        "text": [
            "My application crashes every time I click the login button.",
            "I cannot connect to the database server on port 5432.",
            "The web page is returning a 500 internal server error.",
            "Error: null pointer exception when loading user profile.",
            "I was charged twice on my credit card for the monthly subscription.",
            "Please process a full refund for invoice #49281.",
            "Why did the auto-renewal bill higher than last month?",
            "I want to cancel my active plan and request refund.",
            "What are your business hours during the weekend?",
            "Can you tell me more about your enterprise pricing tier?",
            "Where can I find documentation about the REST API?",
            "Do you offer student discounts on the annual membership?",
        ],
        "label": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    }

    eval_data = {
        "text": [
            "The app freezes upon startup on Windows 11.",
            "I need my money back for the accidental renewal charge.",
            "How do I upgrade my team plan to enterprise?",
        ],
        "label": [0, 1, 2],
    }

    id2label = {0: "Technical Issue", 1: "Billing / Refund", 2: "General Inquiry"}
    label2id = {"Technical Issue": 0, "Billing / Refund": 1, "General Inquiry": 2}

    return Dataset.from_dict(train_data), Dataset.from_dict(eval_data), id2label, label2id


def compute_metrics_builder(accuracy_metric, f1_metric):
    """Build evaluation compute_metrics function."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}
    return compute_metrics


def main():
    print("=" * 60)
    print("Experiment 10: Foundation Model Fine-Tuning & Evaluation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Step 1: Prepare domain-specific dataset
    print("Preparing domain-specific Customer Support Intent dataset...")
    train_dataset, eval_dataset, id2label, label2id = prepare_dataset()
    print(f"  Training samples  : {len(train_dataset)}")
    print(f"  Validation samples: {len(eval_dataset)}\n")

    # Step 2: Load Tokenizer & Tokenize Dataset
    print(f"Loading Tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True)

    # Step 3: Load Model with Classification Head
    print(f"Loading base pre-trained model: {MODEL_NAME} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Step 4: Setup Evaluation Metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    compute_metrics_fn = compute_metrics_builder(accuracy_metric, f1_metric)

    # Step 5: Configure Training Arguments
    checkpoints_dir = os.path.join(OUTPUT_DIR, "fine_tuned_model")
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=1,
        report_to="none",
        use_cpu=(device == "cpu"),
    )

    # Step 6: Initialize Trainer and Train
    print("\nStarting model fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    print("Fine-tuning completed successfully.\n")

    # Step 7: Evaluate Model Post-Training
    print("--- Post-Training Evaluation on Validation Set ---")
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")
    print()

    # Step 8: Test Inference with Fine-Tuned Model
    print("--- Inference on New Unseen Customer Queries ---")
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )

    test_queries = [
        "The software keeps throwing a fatal memory leak error.",
        "Could you refund the subscription fee charged this morning?",
        "Where can I read the terms of service and product roadmap?",
    ]

    for query in test_queries:
        result = classifier(query)[0]
        print(f"Query : \"{query}\"")
        print(f"Intent: {result['label']} (Confidence: {result['score']:.4f})\n")


if __name__ == "__main__":
    main()
