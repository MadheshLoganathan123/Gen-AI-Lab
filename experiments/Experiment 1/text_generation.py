"""
Experiment 1: Text Generation Using Pre-Trained Foundation Models
CS4V48 - GenAI and LLM Lab

AIM: To develop a text generation application using a pre-trained foundation model (GPT-2)
     with the Hugging Face Transformers library.

OBJECTIVE: To understand how pre-trained foundation models generate coherent text from a given
           prompt, and to explore decoding strategies such as greedy search, sampling, and
           top-k / top-p (nucleus) sampling.
"""

from transformers import pipeline, set_seed


def main():
    # Load the pre-trained GPT-2 text generation pipeline
    print("Loading GPT-2 text generation pipeline...")
    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)

    # Input prompt
    prompt = "Artificial Intelligence will transform the future of"

    print(f"\nInput Prompt: {prompt}\n")

    # Generate text with sampling-based decoding strategies
    outputs = generator(
        prompt,
        max_length=60,
        num_return_sequences=2,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

    # Display generated outputs
    for i, out in enumerate(outputs, 1):
        print(f"--- Generated Text {i} ---")
        print(out["generated_text"])
        print()


if __name__ == "__main__":
    main()
