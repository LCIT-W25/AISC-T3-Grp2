import sqlite3
import requests

class NewsDatabase:
    def _init_(self, db_name="news.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        """Create the news table if it doesn't exist."""
        self.cursor.execute("""
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
        self.conn.commit()

    def insert_news(self, data, category):
        """Insert news data into the SQLite database."""
        for article in data[:5]:  # Only insert the first 5 articles
            self.cursor.execute("""
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
        self.conn.commit()

    def fetch_top_articles(self):
        """Fetch and display the top 5 results from the SQLite database."""
        self.cursor.execute("SELECT * FROM news LIMIT 5")
        return self.cursor.fetchall()

    def close(self):
        """Close the database connection."""
        self.conn.close()


class NewsAPIClient:
    def _init_(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/top-headlines"
        self.categories = ["business", "technology", "entertainment", "sports", "science", "health"]

    def fetch_news(self, category):
        """Fetch top news articles from NewsAPI for a given category."""
        url = f'{self.base_url}?country=us&category={category}&apiKey={self.api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json()
            # Add the source as "NewsAPI"
            for article in news_data.get('articles', [])[:5]:  # Get the top 5 articles
                article['source'] = "NewsAPI"
            return news_data.get('articles', [])
        else:
            print(f"Failed to fetch NewsAPI for category {category}, status code: {response.status_code}")
            return []


class NewsScraper:
    def _init_(self, db_name, api_key):
        self.db = NewsDatabase(db_name)
        self.news_api_client = NewsAPIClient(api_key)

    def scrape_and_store_news(self):
        """Scrape and store news articles for each category."""
        for category in self.news_api_client.categories:
            print(f"Fetching news for category: {category}")
            news_data = self.news_api_client.fetch_news(category)
            self.db.insert_news(news_data, category)

    def display_top_articles(self):
        """Display the top 5 news articles from the SQLite database."""
        print("Top 5 news articles from SQLite database:")
        top_articles = self.db.fetch_top_articles()
        for row in top_articles:
            print(row)

    def close(self):
        """Close the database connection."""
        self.db.close()


def main():
    # Your NewsAPI key here
    api_key = 'f7d97478e7d5438aab133108205e32b9'  # Replace with your actual NewsAPI key

    # Initialize scraper
    scraper = NewsScraper(db_name="news.db", api_key=api_key)

    # Scrape and store news articles
    scraper.scrape_and_store_news()

    # Display top 5 articles
    scraper.display_top_articles()

    # Close the connection
    scraper.close()


if _name_ == "_main_":
    main()
