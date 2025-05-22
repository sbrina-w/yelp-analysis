import pandas as pd
from pipeline.review_loader import load_grouped_reviews
from pipeline.langchain_pipeline import analyze_individual_review, summarize_by_category
from pipeline.hf_pipeline import hf_sentiment, hf_summarize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm #progress bar for visual 

def analyze_one_review(review_dict: dict):
    review = review_dict.get("text", "")
    if isinstance(review, str) and review.strip():  #skip empty or malformed reviews
            hf_sent = hf_sentiment(review)
            hf_sum = hf_summarize(review)
            #only use OpenAI if HF sentiment is not positive
            if hf_sent in ["neutral", "negative"]:
                openai_result = analyze_individual_review(review)
            else:
                openai_result = {}

            return {
                "review": review,
                "hf_sentiment": hf_sent,
                "hf_summary": hf_sum,
                "openai_analysis": openai_result,
                "date": review_dict.get("date"),
                "user_id": review_dict.get("user_id"),
                "adjusted_rating": review_dict.get("adjusted_rating")
            }
    return None

def analyze_reviews_parallel(reviews: list[dict]):
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for result in tqdm(executor.map(analyze_one_review, reviews), total=len(reviews), desc="Analyzing reviews"):
            if result is not None:
                results.append(result)
    return results

def analyze_reviews_for_restaurant(business_id: str, name: str, all_reviews: list[dict]):
    print(f"\nAnalyzing: {name} ({business_id})")

    individual_analyses = analyze_reviews_parallel(all_reviews)
    review_texts = [r["text"] for r in all_reviews]
    category_summary = summarize_by_category(review_texts)

    return {
        "business_id": business_id,
        "name": name,
        "category_summary": category_summary,
        "individual_analyses": individual_analyses
    }

def main():
    grouped_df = load_grouped_reviews("yelp_data/10_sample_restaurant_reviews.csv")
    all_results = []

    for _, row in tqdm(grouped_df.iterrows(), total=len(grouped_df), desc="Restaurants"):
        business_id = row["business_id"]
        name = row["name"]
        reviews = [
            {
                "text": t,
                "date": d,
                "user_id": u,
                "adjusted_rating": a
            }
            for t, d, u, a in zip(row["text"], row["date"], row["user_id"], row["adjusted_rating"])
        ]
        result = analyze_reviews_for_restaurant(business_id, name, reviews)
        result.update({
            "address": row["address"],
            "city": row["city"],
            "state": row["state"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "adjusted_average": row["adjusted_avg_rating"],
            "adjusted_review_count": row["adjusted_review_count"]
        })
        all_results.append(result)

    pd.DataFrame(all_results).to_json("analysis_output.json", orient="records", indent=2)
    print("\nAnalysis saved to analysis_output.json")

if __name__ == "__main__":
    main()
