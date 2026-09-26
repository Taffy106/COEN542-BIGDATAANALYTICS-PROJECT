import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient

st.set_page_config(page_title="Social Sentiment & Trends", layout="wide")
client = MongoClient("mongodb://localhost:27017/")
db = client["socialmedia"]

posts = pd.DataFrame(list(db.processed_posts.find({}, {"_id": 0})))
trends = pd.DataFrame(list(db.trends.find({}, {"_id": 0})))

st.title("Scalable Sentiment & Trend Analytics")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Sentiment Distribution")
    sentiment_counts = posts["sentiment_label"].value_counts().reset_index()
    fig = px.pie(sentiment_counts, names="sentiment_label", values="count")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top Trending Hashtags")
    top_trends = trends.sort_values("mentions", ascending=False).head(15)
    fig2 = px.bar(top_trends, x="token", y="mentions")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Sample Posts")
st.dataframe(posts[["text", "sentiment_label", "sentiment_score"]].head(50))

st.subheader("Scalability Experiment")
with open("scalability_data_fraction.html", "r") as f:
    st.components.v1.html(f.read(), height=500)
with open("scalability_cores.html", "r") as f:
    st.components.v1.html(f.read(), height=500)