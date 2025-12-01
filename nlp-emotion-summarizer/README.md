# NLP Summarizer + Emotion Classifier

This project builds an end-to-end NLP pipeline that:
- Summarizes text using a BART model
- Classifies emotions using a DistilRoBERTa model trained on GoEmotions

## Run

## Project Structure
- `preprocess/` — Data cleaning and tokenization
- `models/` — Emotion classifier & summarizer
- `utils/` — JSON formatting of outputs
- `pipeline.py` — Combines summarization + emotion classification
