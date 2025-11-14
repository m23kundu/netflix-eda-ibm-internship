import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page Config (Fancy)
st.set_page_config(
    page_title="Netflix EDA Pro",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Stylish Look)
st.markdown("""
<style>
    .main {background-color: #0e1117; color: white;}
    .stApp {background-color: #0e1117;}
    .css-1d391kg {background-color: #1a1a1a;}
    .css-1v0mbdj {color: #ff4b4b;}
    .metric-card {
        background-color: #1f1f1f;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.2);
        text-align: center;
    }
    .stPlotlyChart {border-radius: 15px; overflow: hidden;}
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    df['director'].fillna('Unknown', inplace=True)
    df['cast'].fillna('Unknown', inplace=True)
    df['country'].fillna('Unknown', inplace=True)
    df['rating'].fillna(df['rating'].mode()[0], inplace=True)
    df['duration'].fillna('Unknown', inplace=True)
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year

    def extract_duration(d):
        if pd.isna(d) or d == 'Unknown': return np.nan
        if 'min' in d: return int(d.split()[0])
        elif 'Season' in d: return int(d.split()[0])
        return np.nan
    df['duration_num'] = df['duration'].apply(extract_duration)
    df['duration_type'] = df['duration'].apply(lambda x: 'min' if 'min' in str(x) else 'seasons' if 'Season' in str(x) else 'unknown')
    return df

df = load_data()

# Title
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🎬 Netflix Content Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Interactive EDA Dashboard | IBM Internship Project</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.markdown("### 🎛️ Filters")

search = st.sidebar.text_input("🔍 Search Title", "")
type_filter = st.sidebar.multiselect("Content Type", ['Movie', 'TV Show'], default=['Movie', 'TV Show'])
year_range = st.sidebar.slider("Year Added", 2008, 2021, (2015, 2021))
rating_filter = st.sidebar.multiselect("Rating", df['rating'].unique(), default=df['rating'].unique()[:3])
duration_range = st.sidebar.slider("Duration", 0, 400, (50, 200))

# Apply Filters
filtered = df[df['type'].isin(type_filter)]
filtered = filtered[filtered['year_added'].between(year_range[0], year_range[1])]
filtered = filtered[filtered['rating'].isin(rating_filter)]
filtered = filtered[filtered['duration_num'].between(duration_range[0], duration_range[1])]
if search:
    filtered = filtered[filtered['title'].str.contains(search, case=False, na=False)]

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><h3>{len(filtered)}</h3><p>Total Titles</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h3>{len(filtered[filtered['type']=='Movie'])}</h3><p>Movies</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h3>{len(filtered[filtered['type']=='TV Show'])}</h3><p>TV Shows</p></div>", unsafe_allow_html=True)
with col4:
    avg_dur = filtered[filtered['type']=='Movie']['duration_num'].mean()
    st.markdown(f"<div class='metric-card'><h3>{avg_dur:.0f} min</h3><p>Avg Movie Length</p></div>", unsafe_allow_html=True)

# Row 1: Content Trend + Pie
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Content Added Over Years")
    year_count = filtered['year_added'].value_counts().sort_index()
    fig1 = px.line(year_count, markers=True, color_discrete_sequence=['#ff4b4b'])
    fig1.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🍰 Content Type Split")
    type_count = filtered['type'].value_counts()
    fig2 = px.pie(values=type_count.values, names=type_count.index, color_discrete_sequence=['#ff4b4b', '#4b4bff'])
    fig2.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig2, use_container_width=True)

# Row 2: World Map + Top Directors
col1, col2 = st.columns(2)
with col1:
    st.subheader("🌍 Content by Country")
    country_count = filtered[filtered['country'] != 'Unknown']['country'].value_counts().head(20)
    fig3 = px.choropleth(
        pd.DataFrame({'country': country_count.index, 'count': country_count.values}),
        locations='country', locationmode='country names', color='count',
        color_continuous_scale='Reds', template='plotly_dark'
    )
    fig3.update_layout(height=450)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("🎥 Top 10 Directors")
    dir_count = filtered[filtered['director'] != 'Unknown']['director'].value_counts().head(10)
    fig4 = px.bar(y=dir_count.index, x=dir_count.values, orientation='h', color_discrete_sequence=['#ff4b4b'])
    fig4.update_layout(template='plotly_dark', height=450, xaxis_title="Titles", yaxis_title="Director")
    st.plotly_chart(fig4, use_container_width=True)

# Row 3: Rating + Duration
col1, col2 = st.columns(2)
with col1:
    st.subheader("⭐ Rating Distribution")
    rating_count = filtered['rating'].value_counts()
    fig5 = px.bar(x=rating_count.index, y=rating_count.values, color_discrete_sequence=['#4b4bff'])
    fig5.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    st.subheader("⏱️ Duration Distribution")
    movie_dur = filtered[filtered['type']=='Movie']['duration_num'].dropna()
    fig6 = px.histogram(movie_dur, nbins=30, color_discrete_sequence=['#ff4b4b'])
    fig6.update_layout(template='plotly_dark', height=400, xaxis_title="Duration (min)", yaxis_title="Count")
    st.plotly_chart(fig6, use_container_width=True)

# Word Cloud
st.subheader("☁️ Most Common Genres")
genres = ' '.join(filtered['listed_in'].dropna())
wordcloud = WordCloud(width=1000, height=500, background_color='#0e1117', color_func=lambda *args, **kwargs: "#ff4b4b").generate(genres)
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Download
st.download_button(
    "📥 Download Filtered Data",
    filtered.to_csv(index=False),
    "netflix_filtered.csv",
    "text/csv"
)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Made with ❤️ using Streamlit | IBM Internship 2025</p>", unsafe_allow_html=True)