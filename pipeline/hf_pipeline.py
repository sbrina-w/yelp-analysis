import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1 
if torch.backends.mps.is_available():
    device = -1

_sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)
_summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

def hf_sentiment(text: str) -> str:
    result = _sentiment_pipeline(text[:512])[0]  
    return label_map[result['label']]

def hf_summarize(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    input_len = len(text.split())
    #skip summarization if it is short
    if input_len < 30:
        return text.strip()
    max_len = min(max(30, int(input_len * 0.8)), 130)  # dynamically scale
    min_len = min(max(10, int(input_len * 0.4)), max_len - 1)

    try: 
        result = _summarization_pipeline(text, max_length=max_len, min_length=min_len, do_sample=False)  
        return result[0]['summary_text'].strip()
    except Exception as e:
        print(f"Summarization error: {e}")
        return text.strip()
