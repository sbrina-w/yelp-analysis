# Yelp Review Analyzer & Prioritizer

A full-stack GenAI and data science pipeline for analyzing restaurant reviews from Yelp.  
This project leverages LLMs (OpenAI, Hugging Face) and vector embeddings (ChromaDB) to extract, prioritize, and summarize customer feedback — with a semantic search frontend and interactive data visualizations built in Streamlit.

## Features

- **LLM-Based Analysis**
  - Sentiment analysis using OpenAI & Hugging Face
  - Summarization using LangChain + BART
  - Priority issue tagging (e.g., "Food Quality", "Service")

- **Semantic Search & Embeddings**
  - Vectorized review storage with ChromaDB
  - Search similar reviews using Hugging Face embeddings
  - Clustering of reviews for trend detection
  - Visual insights over time and across locations

- **Streamlit Frontend**
  - Three views: Restaurant Explorer, Business Owner, Data Analyst
  - Interactive dashboard: search, filter, explore
  - Charts: pie, bar, line (time trends)

---

## Sample Use Cases

- Business owners can:
  - Detect repeated complaints (e.g. “slow service”)
  - Compare feedback sentiment over time
  - Identify top urgent priorities to act on

- Users can:
  - Browse top-rated food spots in the area
  - Explore overall summarized reviews by category (no need to read through many individual reviews)
  - View visual sentiment summaries

---

## Tech Stack

| Category             | Tool / Library                            |
|----------------------|-------------------------------------------|
| LLM                  | LangChain, OpenAI API, Hugging Face       |
| NLP, Clustering      | pandas, numpy, sklearn, sentence-transformers |
| Embedding DB         | ChromaDB                                  |
| Deployment           | Streamlit, Streamlit Community Cloud      |
| Visualizations       | plotly, folium, seaborn, matplotlib       |
| Data                 | Yelp Open Dataset                         |

---

## Check It Out Here!
https://yelp-analysis.streamlit.app/ 

## Notes
Full summarization and sentiment analysis was done only for a sample of 10 restuarants with 15-20 reviews each, other data analysis (ex. topic clustering) was performed using a sample of 300 restaurants with ~5000 total reviews.
