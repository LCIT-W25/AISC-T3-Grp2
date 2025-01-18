import requests
import sqlite3
from bs4 import BeautifulSoup

# SQLite Database setup
conn = sqlite3.connect("news_data.db")
cursor = conn.cursor()

# Create a table
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
    category TEXT)
""")
conn.commit()

# Function to scrape Google News
def scrape_google_news():
    categories = ["Canada", "Local", "World", "Business", "Tech", "Entertainmen>
    base_url = "https://news.google.com/search?q={category}&hl=en-CA&gl=CA&ceid>

    for category in categories:
        response = requests.get(base_url.format(category=category))
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("article")[:5]  # Top 5 articles

        for article in articles:
            title = article.find("h3").text if article.find("h3") else None
            url = article.find("a")["href"] if article.find("a") else None
            description = article.find("p").text if article.find("p") else None

            cursor.execute("""            INSERT INTO news (source, author, title, description, content, url,>
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ("GNews", None, title, description, None, url, None, category))
    conn.commit()

# Function to fetch news from NewsAPI
def fetch_newsapi_data():
    api_key = "YOUR_NEWSAPI_KEY"
    categories = ["general", "business", "technology", "entertainment", "sports>
    base_url = "https://newsapi.org/v2/top-headlines?category={category}&countr>

    for category in categories:
        response = requests.get(base_url.format(category=category))
        if response.status_code == 200:
            data = response.json()
            for article in data["articles"][:5]:
                cursor.execute("""
                INSERT INTO news (source, author, title, description, content, >
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, ("NewsAPI", article["author"], article["title"], article[">                      
                         article["content"], article["url"], article["publishedAt">
    conn.commit()

# Run the functions
scrape_google_news()
fetch_newsapi_data()

# Verify data in SQLite
cursor.execute("SELECT * FROM news LIMIT 5")
print(cursor.fetchall())

conn.close()
