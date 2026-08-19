# CS4V48 — GenAI & LLM Laboratory Experiments

This folder contains the shared lab repository for the CS4V48 Generative AI and LLM experiments.
Each experiment is self-contained under its own directory with a script, requirements, and an output file.

## Experiments

| Experiment | Folder | Script | Purpose |
|------------|--------|--------|---------|
| 1 | Experiment 1 | `text_generation.py` | Text generation using GPT-2 |
| 2 | Experiment 2 | `prompt_engineering.py` | Prompt engineering with GPT-2 |
| 3 | Experiment 3 | `conversational_chatbot.py` | DialoGPT chatbot conversation |
| 4 | Experiment 4 | `summarization_qa.py` | Summarization and question answering |
| 5 | Experiment 5 | `sentiment_classification.py` | Sentiment analysis and zero-shot classification |
| 6 | Experiment 6 | `rag_system.py` | Retrieval-Augmented Generation using vector search |
| 7 | Experiment 7 | `code_generation.py` | AI-powered code generation and debugging |
| 8 | Experiment 8 | `image_generation.py` | Image generation using Stable Diffusion |
| 9 | Experiment 9 | `image_captioning_vqa.py` | Vision-Language image captioning & Visual QA |
| 10 | Experiment 10 | `model_finetuning.py` | Supervised fine-tuning & evaluation of foundation models |
| 11 | Experiment 11 | `multimodal_genai.py` | Multimodal text, image, and speech generation pipeline |
| 12 | Experiment 12 | `evaluation_and_deployment.py` | Model evaluation (ROUGE/BLEU) & Gradio web UI deployment |

## Experiment Details

### Experiment 1: Text Generation
- Script: `Experiment 1/text_generation.py`
- Model: GPT-2
- Description: Generates text from a prompt using a pre-trained GPT-2 model and sampling techniques such as top-k, top-p, and temperature control.
- Output: generated text written to `Experiment 1/outputs/output.txt`.

### Experiment 2: Prompt Engineering
- Script: `Experiment 2/prompt_engineering.py`
- Model: GPT-2
- Description: Demonstrates prompt engineering strategies including zero-shot, few-shot, and chain-of-thought prompts to influence model output.
- Output: prompt experiment results written to `Experiment 2/outputs/output.txt`.

### Experiment 3: Conversational Chatbot
- Script: `Experiment 3/conversational_chatbot.py`
- Model: Microsoft DialoGPT-medium
- Description: Builds a multi-turn conversational chatbot that maintains context across turns and responds to user input.
- Output: conversation transcript saved to `Experiment 3/outputs/output.txt`.

### Experiment 4: Summarization & QA
- Script: `Experiment 4/summarization_qa.py`
- Models: BART-large-cnn for summarization, DistilBERT-SQuAD for question answering
- Description: Performs abstractive summarization on a passage and answers a question using an extractive QA pipeline.
- Output: summary and QA results written to `Experiment 4/outputs/output.txt`.

### Experiment 5: Sentiment Classification
- Script: `Experiment 5/sentiment_classification.py`
- Models: DistilBERT fine-tuned on SST-2 for sentiment analysis, BART-large-MNLI for zero-shot classification
- Description: Runs sentiment analysis on sample reviews and performs zero-shot document classification over candidate labels.
- Output: classification results saved to `Experiment 5/outputs/output.txt`.

### Experiment 6: Retrieval-Augmented Generation (RAG)
- Script: `Experiment 6/rag_system.py`
- Models: `all-MiniLM-L6-v2` for embeddings, FAISS for vector indexing, `google/flan-t5-base` for generation
- Description: Implements a RAG pipeline that retrieves top-k relevant knowledge chunks and generates grounded answers.
- Output: retrieved context and grounded generation saved to `Experiment 6/outputs/output.txt`.

### Experiment 7: Code Generation & Debugging
- Script: `Experiment 7/code_generation.py`
- Model: `Salesforce/codegen-350M-mono`
- Description: Generates code from natural language prompts and provides debugging assistance for buggy code snippets.
- Output: code generation and debug suggestions saved to `Experiment 7/outputs/output.txt`.

### Experiment 8: Image Generation Using Diffusion Models
- Script: `Experiment 8/image_generation.py`
- Model: `runwayml/stable-diffusion-v1-5`
- Description: Generates high-quality images from text descriptions using the reverse diffusion process.
- Output: generated image files and execution log saved to `Experiment 8/outputs/output.txt`.

### Experiment 9: Vision-Language Models (Image Captioning & VQA)
- Script: `Experiment 9/image_captioning_vqa.py`
- Models: `Salesforce/blip-image-captioning-base`, `Salesforce/blip-vqa-base`
- Description: Generates natural language image descriptions and answers visual questions about image scenes.
- Output: captioning and VQA results saved to `Experiment 9/outputs/output.txt`.

### Experiment 10: Foundation Model Fine-Tuning & Evaluation
- Script: `Experiment 10/model_finetuning.py`
- Models: `distilbert-base-uncased` with Hugging Face Trainer
- Description: Fine-tunes a transformer model on a custom intent classification dataset and evaluates accuracy and F1 score.
- Output: training metrics and evaluation results saved to `Experiment 10/outputs/output.txt`.

### Experiment 11: Multimodal Generative AI Application
- Script: `Experiment 11/multimodal_genai.py`
- Models: `google/flan-t5-base` (text), `runwayml/stable-diffusion-v1-5` (image), `gTTS` (speech)
- Description: End-to-end multimodal pipeline generating story narrative, visual image illustration, and audio narration.
- Output: generated image, speech mp3, and logs saved to `Experiment 11/outputs/output.txt`.

### Experiment 12: GenAI Model Evaluation & Web App Deployment
- Script: `Experiment 12/evaluation_and_deployment.py`
- Models & Frameworks: BART-large-cnn, Flan-T5, `evaluate` (ROUGE/BLEU), `gradio`
- Description: Benchmarks text generation quality with quantitative metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU) and constructs an interactive Gradio web application.
- Output: benchmark scores and web interface logs saved to `Experiment 12/outputs/output.txt`.

## How to Run an Experiment

1. Activate the shared virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Run any experiment directly:

```powershell
python "Experiment 9/image_captioning_vqa.py"
```

3. Outputs are automatically saved in the respective `Experiment <N>/outputs/` directory.

