import sqlite3
import requests
from bs4 import BeautifulSoup

# SQLite Database setup
DB_NAME = 'news.db'
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Function to clear all data from the news table
def clear_news_table():
    print("Clearing all data from the news table...")
    cursor.execute('DELETE FROM news')
    conn.commit()
    print("All data has been cleared.")

# Function to scrape Google News
def scrape_google_news(category):
    print(f"Scraping Google News for category: {category}...")
    base_url = "https://news.google.com"
    category_url = f"{base_url}/search?q={category}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(category_url)

    if response.status_code != 200:
        print(f"Error fetching Google News: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.find_all("article", limit=5)
    news_items = []

    for article in articles:
        title_tag = article.find("h3")
        title = title_tag.text if title_tag else "Unknown Title"
        link_tag = article.find("a")
        url = base_url + link_tag["href"][1:] if link_tag else "Unknown URL"
        description_tag = article.find_next("p")
        description = description_tag.text if description_tag else "No description available"
        news_items.append({
            "source": "Google News",
            "author": None,
            "title": title,
            "description": description,
            "content": None,
            "url": url,
            "publish_time": None
        })

    return news_items

# Function to fetch data from NewsAPI
def fetch_newsapi_data(category):
    print(f"Fetching NewsAPI data for category: {category}...")
    api_key = "eafeaacbdf124cdb8a3caeb5afd96ae9"  # Replace with your valid NewsAPI key
    url = f"https://newsapi.org/v2/top-headlines?category={category.lower()}&apiKey={api_key}"
    response = requests.get(url)

    if response.status_code == 401:
        print("Error: Invalid API key for NewsAPI.")
        return []

    if response.status_code != 200:
        print(f"Error fetching NewsAPI data: {response.status_code}")
        return []

    articles = response.json().get("articles", [])
    news_items = []

    for article in articles[:5]:
        news_items.append({
            "source": "NewsAPI",
            "author": article.get("author"),
            "title": article.get("title"),
            "description": article.get("description"),
            "content": article.get("content"),
            "url": article.get("url"),
            "publish_time": article.get("publishedAt")
        })

    return news_items

# Function to save data to SQLite
def save_to_sqlite(news_items):
    for item in news_items:
        cursor.execute('''
        INSERT INTO news (source, author, title, description, content, url, publish_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            item['source'], item['author'], item['title'],
            item['description'], item['content'], item['url'], 
            item['publish_time']
        ))
    conn.commit()

# Main process
categories = ["Canada", "Local", "World", "Business", "Tech", "Entertainment", "Sports", "Science", "Health"]

# Clear all data from the table before adding new data
clear_news_table()

for category in categories:
    # Scrape Google News
    google_news = scrape_google_news(category)
    save_to_sqlite(google_news)

    # Fetch NewsAPI Data
    newsapi_data = fetch_newsapi_data(category)
    save_to_sqlite(newsapi_data)

print("All news has been successfully saved to the database!")

# Close SQLite connection
conn.close()
