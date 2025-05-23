import streamlit as st
import pandas as pd
import seaborn
import plotly.express as px
from plotly.subplots import make_subplots
import json
import folium
from streamlit_folium import st_folium
from semantic_search.trend_analysis import semantic_search_interface, cluster_analysis_interface

st.set_page_config(page_title="Yelp Review Analytics", layout="wide", initial_sidebar_state="expanded")

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
try:
    load_css("styles.css")
except FileNotFoundError:
    pass
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
""", unsafe_allow_html=True)

@st.cache_data
def load_data(path="analysis_output.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error("analysis_output.json file not found")
        return []

data = load_data()

if not data:
    st.stop()

user_type = st.sidebar.radio(
    "Select User Type",
    ["General User (Restaurant Explorer)", "Business Owner (Review Tracker)", "Data Analyst (Deep Insights)"],
    help="Choose your user type to customize the dashboard experience"
)

restaurant_names = [entry["name"] for entry in data]
st.session_state.restaurant_names = restaurant_names

if user_type != "Data Analyst (Deep Insights)":
    selected_name = st.sidebar.selectbox("Select Restaurant", restaurant_names, key="restaurant_select")
    restaurant = next(r for r in data if r["name"] == selected_name)
    
    st.html(f"""
    <div class="main-header">
        <h1>{selected_name}</h1>
    </div>
    """)

@st.cache_data
def process_restaurant_data(restaurant_data):
    df = pd.DataFrame(restaurant_data["individual_analyses"])
    if df.empty:
        return df
    
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period('M')
    
    #extract OpenAI fields
    def extract(row, field):
        return row.get("openai_analysis", {}).get(field, None)
    
    df["openai_sentiment"] = df.apply(lambda r: extract(r, "sentiment"), axis=1)
    df["openai_summary"] = df.apply(lambda r: extract(r, "summary"), axis=1)
    df["issue_type"] = df.apply(lambda r: extract(r, "issue_type"), axis=1)
    df["priority"] = df.apply(lambda r: extract(r, "priority"), axis=1)
    
    return df

if "Data Analyst" in user_type:
    st.html("""
    <div class="main-header">
        <h1>Analytics Dashboard</h1>
        <p>Semantic search and topic clustering across all restaurant reviews</p>
    </div>
    """)
    main_tab1, main_tab2 = st.tabs(["🔍 Semantic Search", "🗺️ Topic Clustering"])
    
    with main_tab1:
        semantic_search_interface()
    with main_tab2:
        cluster_analysis_interface()

elif "General User" in user_type:
    df = process_restaurant_data(restaurant)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.html(f"""
        <div class="metric-card">
            <h3>⭐ Rating</h3>
            <h2>{restaurant.get('adjusted_average', 0):.1f}</h2>
            <p>{restaurant.get('adjusted_review_count', 0)} reviews</p>
        </div>
        """)
    
    with col2:
        sentiment_counts = df["hf_sentiment"].value_counts() if not df.empty else pd.Series()
        positive_pct = (sentiment_counts.get('positive', 0) / len(df) * 100) if not df.empty else 0
        st.html(f"""
        <div class="metric-card">
            <h3>😊 Positive</h3>
            <h2>{positive_pct:.0f}%</h2>
            <p>of reviews</p>
        </div>
        """)
    
    with col3:
        st.html(f"""
        <div class="metric-card">
            <h3>📍 Location</h3>
            <h4>{restaurant.get('city', 'N/A')}, {restaurant.get('state', 'N/A')}</h4>
            <p>{restaurant.get('address', 'N/A')}</p>
        </div>
        """)
    
    with col4:
        recent_reviews = len(df[df['date'] >= '2021-01-01']) if not df.empty else 0
        st.html(f"""
        <div class="metric-card">
            <h3>📅 Recent</h3>
            <h2>{recent_reviews}</h2>
            <p>reviews since 2021</p>
        </div>
        """)
    
    #semantic search widget for General Users
    st.markdown("---")
    st.markdown("### 🔍 Search This Restaurant's Reviews")
    
    search_query = st.text_input(
        "What would you like to know about this restaurant?",
        placeholder="e.g., 'how is the service?', 'is it good for dates?', 'parking availability'"
    )
    
    if search_query:
        #use semantic search for this specific restaurant's reviews
        if 'search_engine' not in st.session_state:
            with st.spinner("Initializing search..."):
                from semantic_search.trend_analysis import SemanticSearchEngine
                st.session_state.search_engine = SemanticSearchEngine()
        
        with st.spinner("Searching reviews..."):
            results = st.session_state.search_engine.search(
                search_query, 
                n_results=5, 
                restaurant_filter=selected_name
            )
        
        if results['documents'][0]:
            st.markdown("**Most relevant reviews:**")
            #show top three relevant results
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0][:3],  
                results['metadatas'][0][:3], 
                results['distances'][0][:3]
            )):
                similarity = 1 / (1 + distance)
                sentiment_color = colors = {'positive': "#14783d",'neutral': "#cca51a",'negative': "#9f1e1e"}[metadata['sentiment']]                
                st.markdown(f"""
                <div class="review-card" style="border-left: 4px solid {sentiment_color};">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>Review</strong></span>
                        <span style="color: #666; font-size: 0.9rem;">{metadata['date']}</span>
                    </div>
                    <p style="margin-top: 1rem; font-style: italic;">"{doc[:250]}{'...' if len(doc) > 250 else ''}"</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No relevant reviews found for your query.")
    
    st.markdown("### 📋 What People Are Saying")
    
    categories = restaurant.get("category_summary", {})
    if categories:
        for category, summary in categories.items():
            icon_map = {
                'food': '🍽️',
                'service': '👥', 
                'atmosphere': '🏢',
                'noise': '🔊',
                'seating': '🪑'
            }
            icon = icon_map.get(category.lower(), '📝')
            
            st.html(f"""
            <div class="category-card">
                <h3>{icon} {category.title()}</h3>
                <p>{summary}</p>
            </div>
            """)
    
    if restaurant.get('latitude') and restaurant.get('longitude'):
        st.markdown("### 🗺️ Location")
        m = folium.Map(
            location=[restaurant['latitude'], restaurant['longitude']], 
            zoom_start=15,
            tiles="Cartodb Positron"
        )
        
        folium.Marker(
            [restaurant['latitude'], restaurant['longitude']],
            popup=f"<b>{restaurant['name']}</b><br>{restaurant.get('address', '')}",
            tooltip=restaurant['name'],
            icon=folium.Icon(color='red', icon='cutlery', prefix='fa')
        ).add_to(m)
        st_folium(m, width="100%", height=400)
    
    st.markdown("---")
    st.markdown("### Restaurant Comparison")
    
    comparison_data = []
    for rest in data:
        rest_df = process_restaurant_data(rest)
        positive_pct = (len(rest_df[rest_df['hf_sentiment'] == 'positive']) / len(rest_df) * 100) if not rest_df.empty else 0
        
        comparison_data.append({
            'Restaurant': rest['name'],
            'Rating': rest.get('adjusted_average', 0),
            'Reviews': rest.get('adjusted_review_count', 0),
            'Positive %': positive_pct,
            'City': rest.get('city', 'N/A'),
            'Latitude': rest.get('latitude'),
            'Longitude': rest.get('longitude')
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_rated = comparison_df.nlargest(5, 'Rating')[['Restaurant', 'Rating', 'Reviews']]
        top_rated = top_rated.sort_values(by='Rating', ascending=True)
        fig = px.bar(top_rated, x='Rating', y='Restaurant', orientation='h',
                    title='Top 5 Highest Rated Restaurants')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        most_positive = comparison_df.nlargest(5, 'Positive %')[['Restaurant', 'Positive %']]
        most_positive = most_positive.sort_values(by='Positive %', ascending=True)
        fig = px.bar(most_positive, x='Positive %', y='Restaurant', orientation='h',
                    title='Most Positive Sentiment (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    if comparison_df['Latitude'].notna().any():
        st.markdown("### 🗺️ All Restaurants Map")

        # center_lat = comparison_df['Latitude'].mean()
        # center_lon = comparison_df['Longitude'].mean()
        
        m = folium.Map(location=[39.9553977,-75.2341756], zoom_start=8)
        
        for _, row in comparison_df.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                #color based on rating
                if row['Rating'] >= 4.0:
                    color = 'green'
                elif row['Rating'] >= 3.0:
                    color = 'orange'
                else:
                    color = 'red'
                
                folium.Marker(
                    [row['Latitude'], row['Longitude']],
                    popup=f"<b>{row['Restaurant']}</b><br>Rating: {row['Rating']:.1f}<br>Reviews: {row['Reviews']}",
                    tooltip=row['Restaurant'],
                    icon=folium.Icon(color=color, icon='cutlery', prefix='fa')
                ).add_to(m)
        
        st_folium(m, width="100%", height=500)
else:
    df = process_restaurant_data(restaurant)
    
    if df.empty:
        st.warning("No review data available for analysis.")
        st.stop()
    
    st.sidebar.markdown("### Filters")
    date_range = st.sidebar.slider(
        "Year Range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(int(df['year'].min()), int(df['year'].max()))
    )
    
    sentiment_filter = st.sidebar.multiselect(
        "Sentiment Filter",
        options=df["hf_sentiment"].unique(),
        default=list(df["hf_sentiment"].unique())
    )
    
    filtered_df = df[
        (df['year'] >= date_range[0]) & 
        (df['year'] <= date_range[1]) &
        (df["hf_sentiment"].isin(sentiment_filter))
    ]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_rating = filtered_df['adjusted_rating'].mean()
        st.html(f"""
        <div class="metric-card">
            <h3>⭐ Avg Rating</h3>
            <h2>{avg_rating:.1f}</h2>
            <p>from {len(filtered_df)} reviews</p>
        </div>
        """)
    
    with col2:
        high_priority = len(filtered_df[filtered_df['priority'] == 'High'])
        st.html(f"""
        <div class="metric-card priority-high">
            <h3>🚨 High Priority</h3>
            <h2>{high_priority}</h2>
            <p>issues to address</p>
        </div>
        """)
    
    with col3:
        negative_reviews = len(filtered_df[filtered_df['hf_sentiment'] == 'negative'])
        st.html(f"""
        <div class="metric-card">
            <h3>😞 Negative Reviews</h3>
            <h2>{negative_reviews}</h2>
            <p>{negative_reviews/len(filtered_df)*100:.0f}% of total</p>
        </div>
        """)
    
    with col4:
        recent_trend = "📈" if filtered_df.tail(5)['adjusted_rating'].mean() > filtered_df.head(5)['adjusted_rating'].mean() else "📉"
        st.html(f"""
        <div class="metric-card">
            <h3>📊 Trend</h3>
            <h2>{recent_trend}</h2>
            <p>Recent vs. Early</p>
        </div>
        """)
    
    st.markdown("---")
    st.markdown("### 🔍 Smart Review Search")
    st.markdown("Find specific feedback to improve your business. *Note: review search is across all franchise locations, whereas analysis is specific to this location.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        business_query = st.text_input(
            "Search for specific feedback:",
            placeholder="e.g., 'staff training issues', 'food quality problems', 'wait times'"
        )
    with col2:
        search_sentiment = st.selectbox("Focus on:", ["All", "Negative", "Positive"], key="business_search_sentiment")
    
    if business_query:
        if 'search_engine' not in st.session_state:
            with st.spinner("Initializing search..."):
                from semantic_search.trend_analysis import SemanticSearchEngine
                st.session_state.search_engine = SemanticSearchEngine()
        
        search_sentiment_filter = None if search_sentiment == "All" else search_sentiment.lower()
        
        with st.spinner("Finding relevant feedback..."):
            results = st.session_state.search_engine.search(
                business_query, 
                n_results=8, 
                restaurant_filter=selected_name,
                sentiment_filter=search_sentiment_filter
            )
        
        if results['documents'][0]:
            st.markdown("**Relevant Customer Feedback:**")
            
            feedback_by_sentiment = {'positive': [], 'negative': [], 'neutral': []}
            
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ):
                sentiment = metadata['sentiment']
                similarity = 1 / (1 + distance)
                feedback_by_sentiment[sentiment].append({
                    'text': doc,
                    'date': metadata['date'],
                    'similarity': similarity
                })
            
            if feedback_by_sentiment['negative']:
                st.markdown("**Issues to Address:**")
                for item in feedback_by_sentiment['negative'][:3]:
                    st.markdown(f"""
                    <div class="review-card" style="border-left: 4px solid #ef4444;">
                        <p style="margin: 0; font-size: 0.9rem; color: #666;">{item['date']} • Relevance: {item['similarity']:.1%}</p>
                        <p style="margin: 0.5rem 0 0 0; font-style: italic;">"{item['text'][:200]}{'...' if len(item['text']) > 200 else ''}"</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if feedback_by_sentiment['positive']:
                st.markdown("**What's Working Well:**")
                for item in feedback_by_sentiment['positive'][:3]:
                    st.markdown(f"""
                    <div class="review-card" style="border-left: 4px solid #10b981;">
                        <p style="margin: 0; font-size: 0.9rem; color: #666;">{item['date']} • Relevance: {item['similarity']:.1%}</p>
                        <p style="margin: 0.5rem 0 0 0; font-style: italic;">"{item['text'][:200]}{'...' if len(item['text']) > 200 else ''}"</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No relevant feedback found. Try different search terms.")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "🎯 Issues Analysis", "📋 Priority Reviews", "📊 Sentiment Overview", "🧠 AI Insights"])
    
    with tab1:

        # Rating trends over time
        monthly_ratings = filtered_df.groupby('year_month')['adjusted_rating'].agg(['mean', 'count']).reset_index()
        monthly_ratings['year_month_str'] = monthly_ratings['year_month'].astype(str)
        
        fig = px.line(monthly_ratings, x='year_month_str', y='mean', 
                     title='Rating Trends Over Time',
                     labels={'mean': 'Average Rating', 'year_month_str': 'Month'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Review volume
        fig2 = px.bar(monthly_ratings, x='year_month_str', y='count',
                     title='Review Volume Over Time',
                     labels={'count': 'Number of Reviews', 'year_month_str': 'Month'})
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    
    with tab2:        
        issues_df = filtered_df[filtered_df['issue_type'].notna()]
        
        if not issues_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                issue_counts = issues_df['issue_type'].value_counts()
                fig = px.pie(values=issue_counts.values, names=issue_counts.index,
                           title='Issue Types Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                priority_counts = issues_df['priority'].value_counts()
                colors = {'High': '#ff4757', 'Medium': '#ffa502', 'Low': '#2ed573'}
                fig = px.bar(x=priority_counts.index, y=priority_counts.values,
                           title='Priority Distribution',
                           color=priority_counts.index,
                           color_discrete_map=colors)
                st.plotly_chart(fig, use_container_width=True)
            
            issue_trends = issues_df.groupby(['year_month', 'issue_type']).size().unstack(fill_value=0)
            if not issue_trends.empty:
                issue_trends_reset = issue_trends.reset_index()
                issue_trends_reset['year_month_str'] = issue_trends_reset['year_month'].astype(str)
                fig = px.line(issue_trends_reset, x='year_month_str', y=issue_trends.columns,
                            title='Issue Trends Over Time',
                            labels={'year_month_str': 'Month'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorized issues found in the selected time period.")
            
    with tab3:        
        priority_issues = filtered_df[filtered_df['priority'].notna()].sort_values(['priority', 'date'], ascending=[True, False])
        
        if not priority_issues.empty:
            priority_filter = st.selectbox("Filter by Priority", 
                                         options=['All'] + list(priority_issues['priority'].unique()))
            
            if priority_filter != 'All':
                display_issues = priority_issues[priority_issues['priority'] == priority_filter]
            else:
                display_issues = priority_issues
            
            for _, row in display_issues.iterrows():
                priority_class = f"priority-{row['priority'].lower()}" if pd.notna(row['priority']) else ""
                sentiment_class = f"sentiment-{row['hf_sentiment']}"
                
                st.markdown(f"""
                <div class="review-card {priority_class} {sentiment_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>Priority: {row['priority']}</strong>
                        <span style="margin-left: auto; color: #666; font-size: 0.9rem;">{row['date'].strftime('%Y-%m-%d')}</span>
                    </div>
                    <p><strong>Issue:</strong> {row['issue_type']}</p>
                    <p><strong>Summary:</strong> {row.get('openai_summary', 'N/A')}</p>
                    <p><strong>Original Review:</strong> {row['review'][:300]}{'...' if len(row['review']) > 300 else ''}</p>
                    <p><strong>User ID:</strong> <code>{row['user_id']}</code></p>
                    <p><strong>Rating:</strong> ⭐ {row['adjusted_rating']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No priority issues found in the selected filters.")
            
    with tab4:        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = filtered_df['hf_sentiment'].value_counts()
            colors = {'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#6b7280'}
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        title='Sentiment Distribution',
                        color=sentiment_counts.index,
                        color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            sentiment_rating = filtered_df.groupby('hf_sentiment')['adjusted_rating'].mean()
            fig = px.bar(x=sentiment_rating.index, y=sentiment_rating.values,
                        title='Average Rating by Sentiment',
                        color=sentiment_rating.index,
                        color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)
        
        sentiment_time = filtered_df.groupby(['year_month', 'hf_sentiment']).size().unstack(fill_value=0)
        if not sentiment_time.empty:
            sentiment_time_reset = sentiment_time.reset_index()
            sentiment_time_reset['year_month_str'] = sentiment_time_reset['year_month'].astype(str)
            fig = px.area(sentiment_time_reset, x='year_month_str', y=sentiment_time.columns,
                         title='Sentiment Trends Over Time',
                         labels={'year_month_str': 'Month'})
            st.plotly_chart(fig, use_container_width=True)    
    
    with tab5:
        st.markdown("### AI-Powered Business Insights")
        st.markdown("Analytics to help improve your restaurant")
        
        if st.button("🔍 Analyze Review Topics for This Restaurant"):
            with st.spinner("Analyzing review topics..."):
                from semantic_search.trend_analysis import ClusterAnalyzer
                
                if 'restaurant_analyzer' not in st.session_state:
                    st.session_state.restaurant_analyzer = ClusterAnalyzer()
                
                analyzer = st.session_state.restaurant_analyzer
                if hasattr(analyzer, 'df') and not analyzer.df.empty:
                    if selected_name in analyzer.df['name'].values:
                        try: 
                            restaurant_subset = analyzer.df[analyzer.df['name'] == selected_name]
                            n_clusters = min(6, max(2, len(restaurant_subset) // 5))
                            analyzer.perform_clustering(n_clusters=n_clusters)
                            restaurant_data = analyzer.df[analyzer.df['name'] == selected_name]
                            if not restaurant_data.empty:
                                restaurant_clusters = restaurant_data['cluster_label'].value_counts()
                            
                                st.markdown("**Main Topics in Your Reviews:**")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig = px.pie(
                                        values=restaurant_clusters.values,
                                        names=restaurant_clusters.index,
                                        title=f'Review Topics for {selected_name}'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.markdown("**Topic Breakdown:**")
                                    for topic, count in restaurant_clusters.items():
                                        pct = count / len(restaurant_data) * 100
                                        topic_sentiment = restaurant_data[restaurant_data['cluster_label'] == topic]['sentiment'].mode()
                                        sentiment_emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
                                        emoji = sentiment_emoji.get(topic_sentiment.iloc[0] if not topic_sentiment.empty else 'neutral', '😐')
                                        
                                        st.markdown(f"{emoji} **{topic}**: {count} reviews ({pct:.1f}%)")
                                
                                negative_topics = restaurant_data[restaurant_data['sentiment'] == 'negative']['cluster_label'].value_counts()
                                if not negative_topics.empty:
                                    st.markdown("**⚠️ Priority Areas for Improvement:**")
                                    for topic, count in negative_topics.head(3).items():
                                        st.error(f"**{topic}**: {count} negative reviews - Consider addressing this area")
                        
                            else:
                                st.info("Not enough data for topic analysis. This feature works better with restaurants that have more reviews in the dataset.")
                        except Exception as e:
                            st.error(f"Error during clustering: {e}")
                    else:
                        st.info("No reviews found for the selected restaurant.")
                else:
                    st.warning("Data not yet loaded or empty. Please try again.")