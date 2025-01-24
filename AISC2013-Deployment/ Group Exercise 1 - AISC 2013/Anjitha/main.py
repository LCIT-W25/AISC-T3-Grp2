import sqlite3
import requests

# Create SQLite database and table
db_name = "news.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Create a table for storing news articles
cursor.execute("""
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    author TEXT,
    title TEXT,
    description TEXT,
    content TEXT,
    url TEXT,
    publish_time TEXT,
    category TEXT
)
""")
conn.commit()

# Function to insert news into the SQLite database
def insert_news(data, category):
    for article in data[:5]:  # Only insert the first 5 articles
        cursor.execute("""
        INSERT INTO news (source, author, title, description, content, url, publish_time, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article.get("source", "Unknown"),
            article.get("author", "Unknown"),
            article.get("title", ""),
            article.get("description", ""),
            article.get("content", ""),
            article.get("url", ""),
            article.get("publishedAt", ""),
            category
        ))
    conn.commit()

# Function to fetch NewsAPI data
def fetch_newsapi(category):
    api_key = 'YOUR_NEWSAPI_KEY'  # Replace with your own NewsAPI key
    url = f'https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={api_key}'
   
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        # Add the source as "NewsAPI"
        for article in news_data.get('articles', [])[:5]:  # Get the top 5 articles
            article['source'] = "NewsAPI"
        insert_news(news_data.get('articles', []), category)
    else:
        print(f"Failed to fetch NewsAPI for category {category}, status code: {response.status_code}")

# Categories to fetch from NewsAPI
categories = ["business", "technology", "entertainment", "sports", "science", "health"]

# Fetch and store data for each category
for category in categories:
    fetch_newsapi(category)

# **Verify data insertion**
cursor.execute("SELECT COUNT(*) FROM news")
count = cursor.fetchone()[0]
print(f"Total rows in the news table: {count}")

# Fetch and display top 5 results from SQLite database
print("Top 5 news articles from SQLite database:")
cursor.execute("SELECT * FROM news LIMIT 5")
for row in cursor.fetchall():
    print(row)

# Close the database connection
conn.close()