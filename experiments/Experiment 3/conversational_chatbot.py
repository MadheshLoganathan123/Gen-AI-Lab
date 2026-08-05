"""
Experiment 3: Conversational AI Chatbot Using Transformer-Based Language Models
CS4V48 - GenAI and LLM Lab

AIM: To build a conversational AI chatbot capable of holding a multi-turn dialogue using a
     transformer-based language model (DialoGPT).

OBJECTIVE: To understand how transformer decoder models maintain conversational context across
           multiple turns using dialogue history encoding, and to implement a simple interactive
           chatbot.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "microsoft/DialoGPT-medium"):
    """Load the DialoGPT tokenizer and model."""
    print(f"Loading {model_name} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def generate_response(
    tokenizer,
    model,
    user_input: str,
    chat_history_ids=None,
):
    """
    Encode the user input, concatenate with chat history, generate a response,
    and return the bot reply along with the updated chat history.
    """
    # Encode the new user input with EOS token appended
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    # Concatenate with the existing chat history (if any)
    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
    )

    # Decode only the newly generated tokens (the bot's reply)
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    return response, chat_history_ids


def run_chatbot(max_turns: int = 5):
    """Interactive chatbot loop for up to `max_turns` conversation turns."""
    tokenizer, model = load_model()
    chat_history_ids = None

    print("\nChatbot ready! Type 'quit' to exit.\n")
    for step in range(max_turns):
        user_input = input(">> User: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        response, chat_history_ids = generate_response(
            tokenizer, model, user_input, chat_history_ids
        )
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    run_chatbot()
