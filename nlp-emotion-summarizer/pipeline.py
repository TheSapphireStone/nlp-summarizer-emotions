from preprocess.preprocess_goemotions import load_and_preprocess
from models.emotion_classifier import EmotionClassifier
from models.summarizer import TextSummarizer
from utils.formatting import output_to_json
from config import CONFIG

import torch

class NLPipeline:
    def __init__(self):
        self.emotion_model = EmotionClassifier(CONFIG["emotion_model"])
        self.summarizer = TextSummarizer(CONFIG["summarizer_model"])

    def run(self, text, emotion_labels):
        summary = self.summarizer.summarize(text)
        emotion_probs = self.emotion_model.predict([text])[0].tolist()
        return output_to_json(summary, emotion_probs, emotion_labels)
