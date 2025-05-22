import pandas as pd

def load_grouped_reviews(csv_path: str):
    df = pd.read_csv(csv_path)
    
    review_cols = ['text', 'date', 'user_id', 'adjusted_rating']
    restaurant_id_cols = [col for col in df.columns if col not in review_cols]

    grouped = df.groupby('business_id').agg({
        **{col: list for col in review_cols}, 
        **{col: 'first' for col in restaurant_id_cols if col != 'business_id'}  #data same for every row on restaurant level
    }).reset_index()

    return grouped
