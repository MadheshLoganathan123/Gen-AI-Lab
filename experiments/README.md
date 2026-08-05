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

## How to Run an Experiment

1. Open a terminal in this folder.
2. Change to the experiment directory.
3. Install dependencies:

```powershell
cd "Experiment 1"
python -m pip install -r requirements.txt
```

4. Run the main script:

```powershell
python text_generation.py
```

5. Repeat for other experiments.

## Notes

- Each experiment stores its output in `outputs/output.txt`.
- All experiments use Hugging Face transformers and Torch.
- Models are downloaded automatically and require internet access the first time they run.
- Experiment 3 is interactive and reads user input from the terminal.
