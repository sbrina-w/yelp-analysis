import streamlit as st
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SemanticSearchEngine:
    def __init__(self, db_path="./semantic_search/chroma_db"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("yelp_reviews")
        
    def search(self, query, n_results=10, restaurant_filter=None, sentiment_filter=None):
        query_embedding = self.model.encode([query])
        filters = []
        if restaurant_filter:
            filters.append({"name": restaurant_filter})
        if sentiment_filter:
            filters.append({"sentiment": sentiment_filter})
        if len(filters) == 1:
            where_clause = filters[0]
        elif len(filters) > 1:
            where_clause = {"$and": filters}
        else:
            where_clause = None
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        return results

class ClusterAnalyzer:
    def __init__(self, db_path="./semantic_search/chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("yelp_reviews")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._load_data()
        
    def _load_data(self):
        all_data = self.collection.get(include=["documents", "embeddings", "metadatas"])
        self.embeddings = np.array(all_data['embeddings'])
        self.documents = all_data["documents"]
        self.metadata = all_data['metadatas']
        self.df = pd.DataFrame(self.metadata)
        self.df['text'] = self.documents
        
    def perform_clustering(self, n_clusters=8, random_state=42):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.df['cluster'] = kmeans.fit_predict(self.embeddings)
        
        pca = PCA(n_components=2, random_state=random_state)
        pca_result = pca.fit_transform(self.embeddings)
        self.df['pca1'] = pca_result[:, 0]
        self.df['pca2'] = pca_result[:, 1]
        
        self._generate_cluster_labels()
        
        return self.df
    
    def _generate_cluster_labels(self):
        cluster_labels = {}
        
        for cluster_id in self.df['cluster'].unique():
            cluster_reviews = self.df[self.df['cluster'] == cluster_id]['text'].sample(
                min(10, len(self.df[self.df['cluster'] == cluster_id])), 
                random_state=42
            ).tolist()
            try:
                label = self._get_openai_label(cluster_id, cluster_reviews)
            except Exception as e:
                print(f"Error generating label for cluster {cluster_id}: {e}")
                label = f"Topic {cluster_id}"
            cluster_labels[cluster_id] = label
        
        self.df['cluster_label'] = self.df['cluster'].map(cluster_labels)
        return cluster_labels
    
    def _get_openai_label(self, cluster_id, reviews):
        prompt = f"""Analyze these restaurant reviews and create a brief 2-4 word label that captures the main theme:

Reviews:
{chr(10).join(['- ' + r[:200] + '...' for r in reviews])}

Label (2-4 words):"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    
    def plot_cluster_scatter(self):
        if 'cluster_label' not in self.df.columns:
            self.perform_clustering()
            
        fig = px.scatter(
            self.df, 
            x='pca1', 
            y='pca2', 
            color='cluster_label',
            hover_data=['name', 'city', 'sentiment'],
            title='Review Clusters - Topic Distribution',
            labels={'pca1': 'PCA Component 1', 'pca2': 'PCA Component 2'},
            width=800,
            height=600
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        return fig
    
    def plot_cluster_by_sentiment(self):
        if 'cluster_label' not in self.df.columns:
            self.perform_clustering()
            
        sentiment_cluster = pd.crosstab(self.df['cluster_label'], self.df['sentiment'], normalize='index') * 100
        
        fig = px.bar(
            sentiment_cluster.reset_index(),
            x='cluster_label',
            y=['positive', 'negative', 'neutral'],
            title='Sentiment Distribution Across Topic Clusters',
            labels={'value': 'Percentage (%)', 'cluster_label': 'Topic Cluster'},
            color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#6b7280'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
        )
        
        return fig
    
    def plot_cluster_by_location(self):
        if 'cluster_label' not in self.df.columns:
            self.perform_clustering()
            
        #get top cities by review count
        top_cities = self.df['city'].value_counts().head(6).index
        city_data = self.df[self.df['city'].isin(top_cities)]
        location_cluster = pd.crosstab(city_data['city'], city_data['cluster_label'], normalize='index') * 100
        
        fig = px.imshow(
            location_cluster.values,
            x=location_cluster.columns,
            y=location_cluster.index,
            color_continuous_scale='Blues',
            title='Topic Distribution by City (%)',
            labels={'color': 'Percentage (%)'}
        )
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_cluster_by_time(self):
        if 'cluster_label' not in self.df.columns:
            self.perform_clustering()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['year_month'] = self.df['date'].dt.to_period('M')
        
        time_cluster = self.df.groupby(['year_month', 'cluster_label']).size().unstack(fill_value=0)
        time_cluster_pct = time_cluster.div(time_cluster.sum(axis=1), axis=0) * 100
        
        fig = go.Figure()
        
        for cluster in time_cluster_pct.columns:
            fig.add_trace(go.Scatter(
                x=time_cluster_pct.index.astype(str),
                y=time_cluster_pct[cluster],
                mode='lines+markers',
                name=cluster,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Topic Trends Over Time (%)',
            xaxis_title='Time Period',
            yaxis_title='Percentage of Reviews (%)',
            height=500,
        )
        
        return fig
    
    def get_cluster_insights(self):
        if 'cluster_label' not in self.df.columns:
            self.perform_clustering()
            
        insights = {}
        
        for cluster_label in self.df['cluster_label'].unique():
            cluster_data = self.df[self.df['cluster_label'] == cluster_label]
            
            insights[cluster_label] = {
                'size': len(cluster_data),
                'avg_rating': cluster_data['adjusted_avg_rating'].mean(),
                'top_cities': cluster_data['city'].value_counts().head(3).to_dict(),
                'sentiment_dist': cluster_data['sentiment'].value_counts(normalize=True).to_dict(),
                'sample_reviews': cluster_data['text'].sample(min(3, len(cluster_data))).tolist()
            }
            
        return insights

def semantic_search_interface():
    st.markdown("### 🔍 Semantic Search")
    st.markdown("Search through reviews using natural language queries")
    
    #init search engine
    if 'search_engine' not in st.session_state:
        with st.spinner("Loading semantic search engine..."):
            st.session_state.search_engine = SemanticSearchEngine()
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search query:", 
            placeholder="e.g., 'slow service', 'great food but noisy', 'cozy atmosphere'",
            key="semantic_query"
        )
    
    with col2:
        n_results = st.selectbox("Results:", [5, 10, 15, 20], index=1)
    
    #filter options
    col1, col2 = st.columns(2)
    with col1:
        restaurant_filter = st.selectbox("Filter by Restaurant:", ["All"] + list(st.session_state.get('restaurant_names', [])))
        restaurant_filter = None if restaurant_filter == "All" else restaurant_filter
    
    with col2:
        sentiment_filter = st.selectbox("Filter by Sentiment:", ["All", "positive", "negative", "neutral"])
        sentiment_filter = None if sentiment_filter == "All" else sentiment_filter
    
    if query:
        with st.spinner("Searching..."):
            results = st.session_state.search_engine.search(
                query, n_results, restaurant_filter, sentiment_filter
            )
        
        if results['documents'][0]:
            st.markdown(f"**Found {len(results['documents'][0])} relevant reviews:**")
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                similarity = 1 / (1 + distance)
                
                st.markdown(f"""
                <div class="review-card" style="border-left: 4px solid #3498db;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <strong>{metadata['name']}</strong>
                        <span style="color: #666;">Similarity: {similarity:.2%}</span>
                    </div>
                    <p style="margin: 0.5rem 0;"><strong>Location:</strong> {metadata['city']}</p>
                    <p style="margin: 0.5rem 0;"><strong>Sentiment:</strong> 
                        <span style="color: {'#10b981' if metadata['sentiment'] == 'positive' else '#ef4444' if metadata['sentiment'] == 'negative' else '#6b7280'};">
                            {metadata['sentiment'].title()}
                        </span>
                    </p>
                    <p style="margin: 0.5rem 0;"><strong>Date:</strong> {metadata['date']}</p>
                    <p style="margin-top: 1rem; font-style: italic;">"{doc[:300]}{'...' if len(doc) > 300 else ''}"</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No results found. Try adjusting your search query or filters.")

def cluster_analysis_interface():
    st.markdown("### 🗺️ Topic Cluster Analysis")
    st.markdown("Explore review topics and their patterns across location, time, and sentiment")
    
    #init cluster analyzer
    if 'cluster_analyzer' not in st.session_state:
        with st.spinner("Loading cluster analysis..."):
            st.session_state.cluster_analyzer = ClusterAnalyzer()
            st.session_state.cluster_analyzer.perform_clustering()
    
    analyzer = st.session_state.cluster_analyzer
    col1, col2 = st.columns([1, 3])
    with col1:
        n_clusters = st.slider("Number of Topics:", 5, 15, 8)
        if st.button("Regenerate Clusters"):
            with st.spinner("Regenerating clusters..."):
                analyzer.perform_clustering(n_clusters=n_clusters)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Topic Overview", 
        "😊 By Sentiment", 
        "🌍 By Location", 
        "📅 By Time",
        "💡 Insights"
    ])
    
    with tab1:
        st.plotly_chart(analyzer.plot_cluster_scatter(), use_container_width=True)
        
        #topic summary
        cluster_summary = analyzer.df['cluster_label'].value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=cluster_summary.values,
                names=cluster_summary.index,
                title='Topic Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Topic Sizes:**")
            for topic, count in cluster_summary.items():
                percentage = count / len(analyzer.df) * 100
                st.markdown(f"**{topic}:** {count} reviews ({percentage:.1f}%)")
    
    with tab2:
        st.plotly_chart(analyzer.plot_cluster_by_sentiment(), use_container_width=True)
        
        #sentiment insights
        sentiment_insights = analyzer.df.groupby('cluster_label')['sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100
        st.markdown("**Key Sentiment Patterns:**")
        
        for topic in sentiment_insights.index:
            pos_pct = sentiment_insights.loc[topic, 'positive']
            neg_pct = sentiment_insights.loc[topic, 'negative']
            
            if pos_pct > 70:
                st.success(f"**{topic}:** Highly positive ({pos_pct:.0f}% positive)")
            elif neg_pct > 50:
                st.error(f"**{topic}:** Concerning ({neg_pct:.0f}% negative)")
            else:
                st.info(f"**{topic}:** Mixed sentiment")
    
    with tab3:
        st.plotly_chart(analyzer.plot_cluster_by_location(), use_container_width=True)
        
        #location-specific insights
        st.markdown("**Location-Specific Topic Patterns:**")
        location_cluster = pd.crosstab(analyzer.df['city'], analyzer.df['cluster_label'], normalize='index') * 100
        
        for city in location_cluster.index[:5]:  
            top_topic = location_cluster.loc[city].idxmax()
            top_pct = location_cluster.loc[city].max()
            st.markdown(f"**{city}:** Most discussed topic is *{top_topic}* ({top_pct:.0f}%)")
    
    with tab4:
        st.plotly_chart(analyzer.plot_cluster_by_time(), use_container_width=True)
        
        #time-based insights
        st.markdown("**Trending Topics:**")
        time_data = analyzer.df.copy()
        time_data['date'] = pd.to_datetime(time_data['date'])
        recent_data = time_data[time_data['date'] >= '2021-01-01']
        
        if not recent_data.empty:
            recent_topics = recent_data['cluster_label'].value_counts(normalize=True) * 100
            st.markdown("**Most discussed topics in recent reviews:**")
            for topic, pct in recent_topics.head(3).items():
                st.markdown(f"• **{topic}:** {pct:.1f}% of recent reviews")
    
    with tab5:
        insights = analyzer.get_cluster_insights()
        
        st.markdown("**Detailed Topic Analysis:**")
        
        for topic, data in insights.items():
            with st.expander(f"📋 {topic} ({data['size']} reviews)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Average Rating", f"{data['avg_rating']:.1f}⭐")
                    
                    st.markdown("**Top Cities:**")
                    for city, count in list(data['top_cities'].items())[:3]:
                        st.markdown(f"• {city}: {count} reviews")
                
                with col2:
                    st.markdown("**Sentiment Breakdown:**")
                    for sentiment, pct in data['sentiment_dist'].items():
                        color = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}[sentiment]
                        st.markdown(f"{color} {sentiment.title()}: {pct:.1%}")
                
                st.markdown("**Sample Reviews:**")
                for i, review in enumerate(data['sample_reviews'], 1):
                    st.markdown(f"{i}. *\"{review[:200]}{'...' if len(review) > 200 else ''}\"*")