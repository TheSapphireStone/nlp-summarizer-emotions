import json

def output_to_json(summary, emotions, emotion_labels):
    ranked = {label: float(prob) for label, prob in zip(emotion_labels, emotions)}
    ranked = dict(sorted(ranked.items(), key=lambda x: x[1], reverse=True))
    return json.dumps({
        "summary": summary,
        "emotions": ranked
    }, indent=2)
