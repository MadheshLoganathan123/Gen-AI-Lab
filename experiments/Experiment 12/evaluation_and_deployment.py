"""
Experiment 12: GenAI Model Evaluation And Interactive Web Application Deployment
CS4V48 - GenAI and LLM Lab

AIM: To quantitatively evaluate Large Language Model generation quality using standard NLP
     metrics (ROUGE, BLEU) and deploy an interactive web-based GenAI application using Gradio.

OBJECTIVE: To understand automated evaluation methodologies for text generation (ROUGE-1, ROUGE-2,
           ROUGE-L, BLEU scores) and build an accessible, responsive web interface for real-time
           user interaction with foundation models.
"""

import os
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def evaluate_generation_quality(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE and BLEU metrics comparing generated text against reference text."""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_scores = bleu.compute(predictions=predictions, references=[[r] for r in references])

    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu": bleu_scores["bleu"],
    }


def create_gradio_interface(generator_fn):
    """Build a Gradio web application interface for interactive text generation."""
    with gr.Blocks(title="GenAI Interactive Assistant") as demo:
        gr.Markdown("# 🤖 Generative AI Multi-Task Assistant")
        gr.Markdown("Enter a prompt or select a task to generate responses in real-time.")

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    lines=4,
                    placeholder="Enter your prompt here (e.g., summarize, translate, or explain a concept)...",
                    label="Input Prompt",
                )
                task_mode = gr.Dropdown(
                    choices=["Text Summarization", "Text Generation", "Concept Explanation"],
                    value="Text Generation",
                    label="Task Mode",
                )
                submit_btn = gr.Button("Generate Output", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(lines=6, label="Model Response")

        submit_btn.click(
            fn=generator_fn,
            inputs=[input_text, task_mode],
            outputs=output_text,
        )

    return demo


def main():
    print("=" * 60)
    print("Experiment 12: Model Evaluation & Web App Deployment")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Part A: Quantitative Evaluation with ROUGE and BLEU
    # -----------------------------------------------------------------------
    print("\n--- Part A: Model Evaluation (ROUGE & BLEU Benchmarking) ---\n")

    print("Loading summarization model (facebook/bart-large-cnn)...")
    bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    def summarize_fn(text: str) -> str:
        inputs = bart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = bart_model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=10,
            max_length=35,
            length_penalty=2.0,
            early_stopping=True,
        )
        return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    test_articles = [
        (
            "Transformers have revolutionized Natural Language Processing by enabling models "
            "to process words in parallel through self-attention mechanisms, surpassing traditional "
            "recurrent neural network architectures.",
            "Transformers revolutionized NLP using self-attention to process words in parallel.",
        ),
        (
            "Reinforcement Learning from Human Feedback (RLHF) aligns large language models "
            "with human preferences, improving helpfulness, accuracy, and safety.",
            "RLHF aligns LLMs with human preferences to improve safety and helpfulness.",
        ),
    ]

    generated_summaries = []
    ground_truth_references = []

    print("Generating summaries for benchmark test set...")
    for idx, (article, reference) in enumerate(test_articles, 1):
        gen = summarize_fn(article)
        generated_summaries.append(gen)
        ground_truth_references.append(reference)
        print(f"\n[Sample {idx}]")
        print(f"  Reference : {reference}")
        print(f"  Generated : {gen}")

    print("\nComputing ROUGE and BLEU scores...")
    metrics = evaluate_generation_quality(generated_summaries, ground_truth_references)

    print("\nEvaluation Results:")
    print(f"  ROUGE-1 Score : {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2 Score : {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L Score : {metrics['rougeL']:.4f}")
    print(f"  BLEU Score    : {metrics['bleu']:.4f}")

    # -----------------------------------------------------------------------
    # Part B: Gradio Web Application Deployment
    # -----------------------------------------------------------------------
    print("\n--- Part B: Interactive Web Application (Gradio) ---\n")
    print("Loading general text generation model (google/flan-t5-base)...")
    flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def app_generator(prompt: str, task: str) -> str:
        if not prompt.strip():
            return "Please enter a non-empty prompt."
        if task == "Text Summarization":
            task_prompt = f"Summarize: {prompt}"
        elif task == "Concept Explanation":
            task_prompt = f"Explain the following concept clearly for beginners: {prompt}"
        else:
            task_prompt = prompt
        inputs = flan_tokenizer(task_prompt, return_tensors="pt")
        outputs = flan_model.generate(**inputs, max_length=120)
        return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Testing Gradio app generator function:")
    sample_res = app_generator("What is deep learning?", "Concept Explanation")
    print(f"  Sample Input : \"What is deep learning?\" [Concept Explanation]")
    print(f"  Sample Output: \"{sample_res}\"\n")

    demo = create_gradio_interface(app_generator)
    print("Gradio Interface constructed successfully.")
    print("To launch the local web server interactively, call `demo.launch()`.")


if __name__ == "__main__":
    main()
