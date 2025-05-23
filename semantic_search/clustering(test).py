import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import chromadb
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
clientOpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("yelp_reviews")

all_data = collection.get(include=["documents", "embeddings", "metadatas"])
documents = all_data["documents"]
embeddings = all_data['embeddings']
metadata = all_data['metadatas']

df = pd.DataFrame(metadata)

kmeans = KMeans(n_clusters=10, random_state=42)
df['topic'] = kmeans.fit_predict(embeddings)
df['text'] = documents
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

#get sample reviews per cluster
cluster_texts = {}
for i in range(df["topic"].nunique()):
    cluster_texts[i] = df[df["topic"] == i]["text"].sample(10, random_state=42).tolist()
def generate_cluster_label(cluster_id, reviews):
    prompt = f"""These are 10 customer reviews from a restaurant review dataset grouped into one cluster. Summarize what this cluster is mostly about in 3–5 words.

Reviews:
{chr(10).join(['- ' + r for r in reviews])}

Label:"""
    try:
        response = clientOpenAI.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error labeling cluster {cluster_id}: {e}")
        return f"Cluster {cluster_id}"

#generate labels
cluster_labels = {}
for cluster_id, reviews in cluster_texts.items():
    label = generate_cluster_label(cluster_id, reviews)
    cluster_labels[cluster_id] = label
    print(f"Cluster {cluster_id}: {label}")

df["topic_label"] = df["topic"].map(cluster_labels)

cluster_sentiment_summary = df.groupby("topic_label")["sentiment"].value_counts(normalize=True).unstack().fillna(0) * 100
print("\nCluster Sentiment Breakdown (%):")
print(cluster_sentiment_summary)
cluster_sentiment_summary.to_csv("plots/cluster_sentiment_summary.csv")

#compare topic trends by city
topic_city_counts = pd.crosstab(df["city"], df["topic_label"], normalize="index") * 100
print("\nTopic Distribution by City (%):")
print(topic_city_counts)
topic_city_counts.to_csv("plots/topic_distribution_by_city.csv")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="pca1", y="pca2", hue="topic_label", palette="tab10")
plt.title("Clustered Reviews (Labeled Topics)")
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/cluster_plot_labeled.png")
plt.show()

df.to_csv("../yelp_data/tagged_reviews_with_topics.csv", index=False)