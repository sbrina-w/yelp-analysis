from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from pipeline.prompt_templates import review_prompt, category_summary_prompt
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)

review_chain = review_prompt | llm
category_summary_chain = category_summary_prompt | llm

def analyze_individual_review(review: str) -> dict:
    for attempt in range(3):
        try:
            result = review_chain.invoke({"review": review})
            content = result.content if hasattr(result, "content") else str(result)
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[Attempt {attempt + 1}] Invalid JSON. Retrying...\nResponse:\n{result}\nError: {e}")
        except Exception as e:
            print(f"[Attempt {attempt + 1}] Unexpected error: {e}")
            break
    return {
        "summary": "Parsing error",
        "issue_type": "Other",
        "priority": "Low",
        "sentiment": "neutral"
    }

def summarize_by_category(reviews: list[str]) -> dict:
    joined_reviews = "\n".join(reviews)
    for attempt in range(3):
        try:
            result = category_summary_chain.invoke({"reviews": joined_reviews})
            content = result.content if hasattr(result, "content") else str(result)
            return json.loads(content)
        except Exception:
            print(f"[Attempt {attempt + 1}] Invalid JSON. Retrying...\nResponse:\n{result}\nError: {e}")
        except Exception as e:
            print(f"[Attempt {attempt + 1}] Unexpected error: {e}")
            break
    return {
        "food": "Parsing error",
        "service": "Parsing error",
        "atmosphere": "Parsing error",
        "noise": "Parsing error",
        "seating": "Parsing error"
    }
