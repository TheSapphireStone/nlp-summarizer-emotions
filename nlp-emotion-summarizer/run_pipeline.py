from pipeline import NLPipeline
from preprocess.preprocess_goemotions import load_and_preprocess

if __name__ == "__main__":
    # Load sample dataset (replace path with your CSV)
    texts, y, mlb = load_and_preprocess("goemotions_sample.csv")

    pipeline = NLPipeline()
    
    # Run on first text for demo
    output = pipeline.run(texts[0], mlb.classes_)
    print(output)
