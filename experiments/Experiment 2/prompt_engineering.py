"""
Experiment 2: Prompt Engineering Techniques For Content Generation, Reasoning And Task Automation
CS4V48 - GenAI and LLM Lab

AIM: To implement zero-shot, one-shot, few-shot, and chain-of-thought prompting techniques
     for content generation, reasoning, and task automation using a Large Language Model.

OBJECTIVE: To understand how the structure and content of a prompt influences the output quality
           of an LLM, and to practically apply different prompting strategies to solve content
           generation and reasoning tasks without modifying model weights.
"""

from transformers import pipeline


def run_prompt(generator, name, prompt):
    """Run a prompt through the generator and print the result."""
    # Estimate approximate max length from prompt tokens + 40 for the completion
    max_length = len(prompt.split()) + 40
    output = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=False
    )
    print(f"=== {name} ===")
    print(output[0]["generated_text"])
    print()


def main():
    print("Loading GPT-2 text generation pipeline...")
    generator = pipeline("text-generation", model="gpt2")

    # ------------------------------------------------------------------
    # 1. Zero-Shot Prompt
    # The model receives only the instruction and query — no examples.
    # ------------------------------------------------------------------
    zero_shot_prompt = (
        "Classify the sentiment of this review as Positive or Negative: "
        "'The product quality is excellent!'\nSentiment:"
    )

    # ------------------------------------------------------------------
    # 2. Few-Shot Prompt
    # Provides 2 labeled examples so the model infers the expected pattern.
    # ------------------------------------------------------------------
    few_shot_prompt = (
        "Review: 'I loved this movie, it was fantastic.'\n"
        "Sentiment: Positive\n\n"
        "Review: 'The service was slow and disappointing.'\n"
        "Sentiment: Negative\n\n"
        "Review: 'The product quality is excellent!'\n"
        "Sentiment:"
    )

    # ------------------------------------------------------------------
    # 3. Chain-of-Thought (CoT) Prompt
    # Encourages the model to reason step-by-step before the final answer.
    # ------------------------------------------------------------------
    cot_prompt = (
        "Q: A shop had 15 apples. It sold 6 and then received 10 more. How many apples now?\n"
        "A: Let's think step by step. 15 - 6 = 9. 9 + 10 = 19. The answer is 19.\n\n"
        "Q: A library had 120 books. It lent out 45 and bought 30 new books. How many books now?\n"
        "A: Let's think step by step."
    )

    print("\n--- Running Prompting Experiments ---\n")

    for name, prompt in [
        ("Zero-shot", zero_shot_prompt),
        ("Few-shot", few_shot_prompt),
        ("Chain-of-Thought", cot_prompt),
    ]:
        run_prompt(generator, name, prompt)


if __name__ == "__main__":
    main()
