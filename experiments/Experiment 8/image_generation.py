"""
Experiment 8: Image Generation Application Using Diffusion Models
CS4V48 - GenAI and LLM Lab

AIM: To implement an image generation application using a pre-trained Diffusion Model
     (Stable Diffusion) that synthesises images from text prompts.

OBJECTIVE: To understand the working principle of diffusion-based generative models —
           the forward noising process and the reverse denoising process — and to use them
           for text-to-image generation.
"""

import os
import torch
from diffusers import StableDiffusionPipeline


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_pipeline(model_id: str) -> StableDiffusionPipeline:
    """
    Load the Stable Diffusion pipeline.

    Uses float16 on CUDA for memory efficiency; falls back to float32 on CPU
    (slow but functional for testing without a GPU).
    """
    if torch.cuda.is_available():
        print("CUDA GPU detected — loading pipeline in float16 mode.")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
    else:
        print("No GPU detected — loading pipeline in float32 mode (CPU, slow).")
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    output_filename: str = "generated_image.png",
) -> str:
    """
    Run the reverse diffusion process guided by the text prompt and save the result.

    Parameters
    ----------
    pipe                : Loaded StableDiffusionPipeline
    prompt              : Descriptive text prompt for the desired image
    num_inference_steps : Number of denoising steps (more = higher quality, slower)
    guidance_scale      : Classifier-free guidance scale (higher = closer to prompt)
    output_filename     : File name to save under the outputs/ directory

    Returns
    -------
    Full path to the saved image file.
    """
    print(f"\nGenerating image for prompt:\n  \"{prompt}\"")
    print(f"  Steps: {num_inference_steps}  |  Guidance scale: {guidance_scale}")

    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    image.save(save_path)
    print(f"Image saved to: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("Experiment 8: Image Generation Using Diffusion Models")
    print("=" * 60)

    # Load the Stable Diffusion pipeline
    print(f"\nLoading Stable Diffusion pipeline ({MODEL_ID})...")
    pipe = load_pipeline(MODEL_ID)
    print("Pipeline loaded.\n")

    # ------------------------------------------------------------------
    # Prompt 1 — Landscape / Architecture
    # ------------------------------------------------------------------
    generate_image(
        pipe,
        prompt="A futuristic city skyline at sunset, digital art, highly detailed",
        num_inference_steps=30,
        guidance_scale=7.5,
        output_filename="generated_city.png",
    )

    # ------------------------------------------------------------------
    # Prompt 2 — Nature scene (demonstrates prompt diversity)
    # ------------------------------------------------------------------
    generate_image(
        pipe,
        prompt="A serene mountain lake at dawn, photorealistic, 8k resolution",
        num_inference_steps=30,
        guidance_scale=8.0,
        output_filename="generated_lake.png",
    )

    print("\nAll images generated successfully.")


if __name__ == "__main__":
    main()
