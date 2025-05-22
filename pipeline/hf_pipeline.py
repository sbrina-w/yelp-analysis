import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline

_sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
_summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

def hf_sentiment(text: str) -> str:
    result = _sentiment_pipeline(text[:512])[0]  
    return label_map[result['label']]

def hf_summarize(text: str) -> str:
    input_len = len(text.split())
    max_len = max(30, int(input_len * 0.8))  # dynamically scale
    result = _summarization_pipeline(text[:1024], max_length=max_len)[0]  
    return result['summary_text']
