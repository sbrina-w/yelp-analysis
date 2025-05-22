import pandas as pd

def load_grouped_reviews(csv_path: str):
    df = pd.read_csv(csv_path)
    grouped = df.groupby(['business_id', 'name'])['text'].apply(list).reset_index()
    return grouped
