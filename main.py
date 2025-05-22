import pandas as pd
from pipeline.review_loader import load_grouped_reviews
from pipeline.langchain_pipeline import analyze_individual_review, summarize_by_category

def analyze_reviews_for_restaurant(business_id: str, name: str, all_reviews: list[str]):
    print(f"\nAnalyzing: {name} ({business_id})")

    individual_analyses = []
    for review in all_reviews:
        if isinstance(review, str) and review.strip():  #skip empty or malformed reviews
            result = analyze_individual_review(review)
            result['review'] = review
            individual_analyses.append(result)

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
        break

    pd.DataFrame(all_results).to_json("analysis_output.json", orient="records", indent=2)
    print("\nAnalysis saved to analysis_output.json")

if __name__ == "__main__":
    main()
