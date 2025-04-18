import streamlit as st
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt

st.set_page_config(page_title="Financial News Sentiment", layout="wide")
st.title("\U0001F4C8 Financial News Sentiment Dashboard")

# Sidebar: Input API key
api_key = st.sidebar.text_input("Enter your NewsAPI Key", type="password")

if not api_key:
    st.warning("Please enter your NewsAPI key in the sidebar.")
    st.stop()

# Fetch news
url = "https://newsapi.org/v2/everything"
params = {
    "q": "stock market OR finance OR bitcoin OR earnings",
    "language": "en",
    "sortBy": "publishedAt",
    "pageSize": 15,
    "apiKey": api_key
}

response = requests.get(url, params=params)
if response.status_code != 200:
    st.error("Failed to fetch articles. Check your API key or try again later.")
    st.stop()

articles = response.json().get("articles", [])
analyzer = SentimentIntensityAnalyzer()

data = []
for article in articles:
    content = (article.get("title") or "") + " " + (article.get("description") or "")
    sentiment = analyzer.polarity_scores(content)
    sentiment_class = "positive" if sentiment["compound"] > 0.05 else "negative" if sentiment["compound"] < -0.05 else "neutral"
    data.append({
        "Title": article.get("title"),
        "Description": article.get("description"),
        "URL": article.get("url"),
        "Sentiment": sentiment_class,
        "Compound": sentiment["compound"],
        "Source": article.get("source", {}).get("name", ""),
        "Published At": article.get("publishedAt")
    })

df = pd.DataFrame(data)

# Show article table with clickable titles
st.subheader("\U0001F4F0 Articles & Sentiment")
def make_clickable(link, text):
    return f"<a href='{link}' target='_blank'>{text}</a>"

styled_df = df.copy()
styled_df["Title"] = styled_df.apply(lambda x: make_clickable(x["URL"], x["Title"] if x["Title"] else "View Article"), axis=1)

st.write("Click on a title to read the full article.")
st.write(styled_df[["Title", "Sentiment", "Compound", "Source", "Published At"]].to_html(escape=False, index=False), unsafe_allow_html=True)

# Pie chart of sentiment
st.subheader("\U0001F4CA Sentiment Distribution")
sentiment_counts = df["Sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

chart = alt.Chart(sentiment_counts).mark_arc(innerRadius=50).encode(
    theta="Count",
    color="Sentiment",
    tooltip=["Sentiment", "Count"]
).properties(width=400, height=400)

st.altair_chart(chart, use_container_width=True)

# Bar chart of compound score by article
st.subheader("\U0001F4C9 Article Sentiment Scores")
bar_chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("Title:N", sort='-y', title="Article"),
    y="Compound:Q",
    color="Sentiment:N",
    tooltip=["Title", "Compound", "Sentiment"]
).properties(height=400)

st.altair_chart(bar_chart, use_container_width=True)

# Time-series sentiment trend
st.subheader("\U0001F4C5 Sentiment Over Time")
df["Published At"] = pd.to_datetime(df["Published At"])
df_sorted = df.sort_values("Published At")
line_chart = alt.Chart(df_sorted).mark_line(point=True).encode(
    x="Published At:T",
    y="Compound:Q",
    color="Sentiment:N",
    tooltip=["Title", "Published At", "Compound", "Sentiment"]
)

st.altair_chart(line_chart, use_container_width=True)

# Optional CSV download
st.download_button("Download CSV", df.to_csv(index=False), "financial_news_sentiment.csv", "text/csv")

# Custom URL sentiment analysis
st.subheader("\U0001F50D Analyze Custom News URL")
custom_url = st.text_input("Paste a news article URL to analyze its sentiment")
if custom_url:
    try:
        from newspaper import Article
        article = Article(custom_url)
        article.download()
        article.parse()
        article_text = article.title + " " + article.text
        result = analyzer.polarity_scores(article_text)
        sentiment_label = "positive" if result["compound"] > 0.05 else "negative" if result["compound"] < -0.05 else "neutral"

        st.markdown(f"**Title:** {article.title}")
        st.markdown(f"**Sentiment:** `{sentiment_label}` with score `{result['compound']}`")
        with st.expander("View full article text"):
            st.write(article.text)
    except Exception as e:
        st.error(f"Failed to analyze the article. Reason: {str(e)}")