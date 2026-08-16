"""
Experiment 7: AI-Powered Code Generation And Debugging Assistant
CS4V48 - GenAI and LLM Lab

AIM: To develop an AI-powered assistant that generates code from natural-language descriptions
     and helps identify/fix bugs in existing code using a pre-trained code-generation model.

OBJECTIVE: To understand how transformer models trained on source-code corpora (code LLMs) can
           automate code generation, code explanation, and debugging tasks.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model(model_name: str = "Salesforce/codegen-350M-mono"):
    """Load the CodeGen tokenizer and model."""
    print(f"Loading tokenizer and model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully.\n")
    return tokenizer, model


def generate_code(prompt: str, tokenizer, model, max_new_tokens: int = 80) -> str:
    """
    Generate code continuation for the given prompt.

    The model predicts the most likely token continuation from the prompt,
    effectively 'writing' code given a natural-language instruction or partial snippet.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    print("=" * 60)
    print("Experiment 7: AI-Powered Code Generation & Debugging")
    print("=" * 60)

    tokenizer, model = load_model()

    # ------------------------------------------------------------------
    # Part A: Code Generation from a Natural-Language Instruction
    # ------------------------------------------------------------------
    print("--- Part A: Code Generation ---\n")

    code_gen_prompt = (
        "# Write a Python function to check if a number is prime\n"
        "def is_prime(n):"
    )
    print("Prompt:\n", code_gen_prompt)
    print("\nGenerated Function:")
    generated = generate_code(code_gen_prompt, tokenizer, model, max_new_tokens=80)
    print(generated)
    print()

    # ------------------------------------------------------------------
    # Part B: Debugging a Faulty Code Snippet
    #
    # The bug: factorial() initialises result = 0 instead of result = 1,
    # so every product is 0. The model is asked to suggest the corrected
    # version by completing the `factorial_fixed` stub.
    # ------------------------------------------------------------------
    print("--- Part B: Bug Fix / Debugging ---\n")

    buggy_code = (
        "# The following function should return the factorial of n, but has a bug. Fix it.\n"
        "def factorial(n):\n"
        "    result = 0\n"
        "    for i in range(1, n + 1):\n"
        "        result = result * i\n"
        "    return result\n\n"
        "# Corrected function:\n"
        "def factorial_fixed(n):"
    )
    print("Buggy Code + Fix Instruction:\n", buggy_code)
    print("\nDebug Suggestion:")
    debug_output = generate_code(buggy_code, tokenizer, model, max_new_tokens=60)
    print(debug_output)
    print()


if __name__ == "__main__":
    main()
