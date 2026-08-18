"""
Experiment 9: Vision-Language Models For Image Captioning And Visual Question Answering
CS4V48 - GenAI and LLM Lab

AIM: To implement a multimodal Vision-Language application for automatic image captioning
     and Visual Question Answering (VQA) using pre-trained foundation models (BLIP).

OBJECTIVE: To understand how vision and language representations are integrated in multimodal
           transformers to generate natural-language descriptions of images and answer
           contextual questions about visual content.
"""

import os
from PIL import Image, ImageDraw
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def create_sample_image(filepath: str) -> Image.Image:
    """Create and save a synthetic sample image for testing multimodal vision-language tasks."""
    img = Image.new("RGB", (400, 300), color=(135, 206, 235))  # Sky blue background
    draw = ImageDraw.Draw(img)

    # Draw green grass/ground
    draw.rectangle([0, 200, 400, 300], fill=(34, 139, 34))

    # Draw yellow sun
    draw.ellipse([30, 30, 90, 90], fill=(255, 215, 0))

    # Draw a red house
    draw.rectangle([150, 130, 270, 230], fill=(178, 34, 34))

    # Draw house roof (triangle)
    draw.polygon([(135, 130), (210, 70), (285, 130)], fill=(139, 69, 19))

    # Draw a door
    draw.rectangle([190, 175, 230, 230], fill=(101, 67, 33))

    # Draw a window
    draw.rectangle([160, 150, 185, 175], fill=(255, 255, 255))

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img.save(filepath)
    print(f"Sample test image saved to: {filepath}\n")
    return img


def generate_image_caption(image: Image.Image, processor, model, device: str) -> str:
    """Generate a descriptive natural-language caption for the provided image."""
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def answer_visual_question(image: Image.Image, question: str, processor, model, device: str) -> str:
    """Answer a natural-language question grounded in the visual content of the image."""
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer


def main():
    print("=" * 60)
    print("Experiment 9: Image Captioning & Visual Question Answering")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_path = os.path.join(OUTPUT_DIR, "sample_scene.png")
    image = create_sample_image(image_path)

    # -----------------------------------------------------------------------
    # Part A: Image Captioning using BLIP
    # -----------------------------------------------------------------------
    print("--- Part A: Image Captioning (Salesforce/blip-image-captioning-base) ---\n")
    print("Loading BLIP Image Captioning model...")
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    caption_model.eval()

    caption = generate_image_caption(image, caption_processor, caption_model, device)
    print(f"Generated Caption: {caption}\n")

    # -----------------------------------------------------------------------
    # Part B: Visual Question Answering (VQA) using BLIP-VQA
    # -----------------------------------------------------------------------
    print("--- Part B: Visual Question Answering (Salesforce/blip-vqa-base) ---\n")
    print("Loading BLIP VQA model...")
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(device)
    vqa_model.eval()

    questions = [
        "What color is the house?",
        "What is in the sky?",
        "What color is the roof?",
    ]

    for q in questions:
        ans = answer_visual_question(image, q, vqa_processor, vqa_model, device)
        print(f"Question: {q}")
        print(f"Answer:   {ans}\n")


if __name__ == "__main__":
    main()
