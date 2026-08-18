"""
Experiment 11: Multimodal Generative AI Application (Text-To-Image & Text-To-Speech)
CS4V48 - GenAI and LLM Lab

AIM: To design and develop an integrated multimodal Generative AI pipeline that synthesizes
     creative content across multiple modalities: generating a narrative, synthesizing a visual
     illustration using diffusion models, and producing audio speech narration.

OBJECTIVE: To understand cross-modal generative modeling, combining language generation (LLMs),
           text-to-image synthesis (Stable Diffusion), and text-to-speech synthesis (TTS) into
           a cohesive generative AI application.
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_diffusion_pipeline(model_id: str) -> StableDiffusionPipeline:
    """Load the Stable Diffusion text-to-image pipeline."""
    if torch.cuda.is_available():
        print("CUDA GPU detected — loading Stable Diffusion in float16 mode.")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
    else:
        print("No GPU detected — loading Stable Diffusion in float32 mode (CPU).")
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
    return pipe


def generate_story(story_prompt: str) -> str:
    """Generate a short creative story from a prompt using a pre-trained LLM."""
    print("Generating story using text generation model (google/flan-t5-base)...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    prompt = (
        f"Write a vivid 2-sentence story about: {story_prompt}. "
        "Make it atmospheric and inspiring."
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_image_from_prompt(
    pipe: StableDiffusionPipeline,
    prompt: str,
    output_filename: str = "story_illustration.png",
    num_inference_steps: int = 25,
) -> str:
    """Synthesize an image corresponding to the story visual prompt."""
    print(f"\nGenerating illustration for prompt:\n  \"{prompt}\"")
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.5,
    ).images[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    image.save(save_path)
    print(f"Illustration saved to: {save_path}")
    return save_path


def generate_speech(text: str, output_filename: str = "story_narration.mp3") -> str:
    """Convert story text into speech audio using gTTS."""
    print(f"\nSynthesizing speech narration with gTTS...")
    tts = gTTS(text=text, lang="en", slow=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    tts.save(save_path)
    print(f"Speech audio narration saved to: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("Experiment 11: Multimodal Generative AI Application")
    print("=" * 60)

    story_topic = "an astronaut discovering an ancient bioluminescent crystal on Mars"
    print(f"\nInput Topic: {story_topic}\n")

    # Step 1: Text Generation (Modality 1: Text)
    print("--- Modality 1: Creative Text Generation ---")
    story_text = generate_story(story_topic)
    print(f"Generated Story:\n  \"{story_text}\"\n")

    # Step 2: Image Synthesis (Modality 2: Visual)
    print("--- Modality 2: Text-to-Image Generation (Stable Diffusion) ---")
    image_pipe = load_diffusion_pipeline(MODEL_ID)
    image_prompt = (
        "An astronaut standing in a Martian cave discovering a glowing bioluminescent blue crystal, "
        "cinematic lighting, ultra detailed, 8k"
    )
    image_path = generate_image_from_prompt(
        image_pipe,
        image_prompt,
        output_filename="story_illustration.png",
        num_inference_steps=25,
    )

    # Step 3: Speech Synthesis (Modality 3: Audio)
    print("\n--- Modality 3: Text-to-Speech Audio Generation (TTS) ---")
    audio_path = generate_speech(story_text, output_filename="story_narration.mp3")

    print("\n" + "=" * 60)
    print("Multimodal Pipeline Execution Completed Successfully!")
    print(f"  Text Narrative : \"{story_text}\"")
    print(f"  Visual Output  : {image_path}")
    print(f"  Audio Output   : {audio_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
