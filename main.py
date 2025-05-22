import pandas as pd
from pipeline.review_loader import load_grouped_reviews
from pipeline.langchain_pipeline import analyze_individual_review, summarize_by_category
from pipeline.hf_pipeline import hf_sentiment, hf_summarize

def analyze_reviews_for_restaurant(business_id: str, name: str, all_reviews: list[str]):
    print(f"\nAnalyzing: {name} ({business_id})")

    individual_analyses = []
    for review in all_reviews:
        if isinstance(review, str) and review.strip():  #skip empty or malformed reviews
            hf_sent = hf_sentiment(review)
            hf_sum = hf_summarize(review)
            #only use OpenAI if HF sentiment is not positive
            if hf_sent in ["neutral", "negative"]:
                openai_result = analyze_individual_review(review)
            else:
                openai_result = {}

            individual_analyses.append({
                "review": review,
                "hf_sentiment": hf_sent,
                "hf_summary": hf_sum,
                "openai_analysis": openai_result
            })
        break

    category_summary = summarize_by_category(all_reviews)

    return {
        "business_id": business_id,
        "name": name,
        "category_summary": category_summary,
        "individual_analyses": individual_analyses
    }

def main():
    grouped_df = load_grouped_reviews("yelp_data/10_sample_restaurant_reviews.csv")

    all_results = []

    for _, row in grouped_df.iterrows():
        business_id = row["business_id"]
        name = row["name"]
        reviews = row["text"] 
        result = analyze_reviews_for_restaurant(business_id, name, reviews)
        all_results.append(result)

    pd.DataFrame(all_results).to_json("analysis_output.json", orient="records", indent=2)
    print("\nAnalysis saved to analysis_output.json")

if __name__ == "__main__":
    main()
