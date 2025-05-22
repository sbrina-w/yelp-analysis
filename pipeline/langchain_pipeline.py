from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from pipeline.prompt_templates import review_prompt, category_summary_prompt
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def analyze_individual_review(review: str) -> dict:
    messages = [HumanMessage(content=review_prompt.format(review=review))]
    response = llm.invoke(messages).content
    try:
        return json.loads(response)
    except Exception:
        return {"summary": "Parsing error", "issue_type": "Other", "priority": "Low"}

def summarize_by_category(reviews: list[str]) -> dict:
    joined_reviews = "\n".join(reviews)
    messages = [HumanMessage(content=category_summary_prompt.format(reviews=joined_reviews))]
    response = llm.invoke(messages).content  
    try:
        return json.loads(response)
    except Exception:
        return {
            "food": "Parsing error",
            "service": "Parsing error",
            "atmosphere": "Parsing error",
            "noise": "Parsing error",
            "seating": "Parsing error"
        }
