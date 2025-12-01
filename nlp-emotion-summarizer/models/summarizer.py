from transformers import pipeline

class TextSummarizer:
    def __init__(self, model_name):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text, max_length=150):
        summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']
