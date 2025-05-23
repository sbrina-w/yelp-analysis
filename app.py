import streamlit as st
import pandas as pd
import seaborn
import plotly.express as px
from plotly.subplots import make_subplots
import json
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Yelp Review Analytics", layout="wide", initial_sidebar_state="expanded")

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("styles.css")
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
    ["General User (Restaurant Explorer)", "Business Owner (Review Tracker)"],
    help="Choose your user type to customize the dashboard experience"
)

restaurant_names = [entry["name"] for entry in data]
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

df = process_restaurant_data(restaurant)


if "General User" in user_type:
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
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🎯 Issues Analysis", "📋 Priority Reviews", "📊 Sentiment Overview"])
    
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
                    <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>Priority: {row['priority']}</strong>
                        <span style="margin-left: auto; color: #666; font-size: 0.9rem;">{row['date'].strftime('%Y-%m-%d')}</span>
                    </div>
                    <p><strong>Issue:</strong> {row['issue_type']}</p>
                    <p><strong>Summary:</strong> {row.get('openai_summary', 'N/A')}</p>
                    <p><strong>Original Review:</strong> {row['review']}...</p>
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
