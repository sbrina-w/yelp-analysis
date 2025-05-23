from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

client = chromadb.PersistentClient(path="./chroma_db")
#avoid duplicate or outdated data when updating
if "yelp_reviews" in [col.name for col in client.list_collections()]:
    client.delete_collection("yelp_reviews")

collection = client.create_collection("yelp_reviews")
reviews_df = pd.read_csv("../yelp_data/300_sample_restaurant_reviews.csv") 

#embeddings
texts = reviews_df['text'].tolist()
metadatas = reviews_df[['business_id', 'name', 'address', 'city', 'latitude', 'longitude', 'adjusted_avg_rating', 'date', 'adjusted_review_count', 'sentiment']].to_dict(orient='records')
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

#store in Chroma
collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=[str(i) for i in reviews_df.index])
