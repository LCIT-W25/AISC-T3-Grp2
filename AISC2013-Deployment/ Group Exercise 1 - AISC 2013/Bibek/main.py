import requests
import sqlite3
from bs4 import BeautifulSoup
from datetime import datetime


class NewsScraper:
    def __init__(self, db_file, newsapi_key):
        """Initialize the Newscraper Instance"""
        self.db_file = db_file
        self.newsapi_key = newsapi_key
        self.conn = None

    def initialize_db(self):
        """Initialize SQLite database and create a table if it doesn't exist."""
        self.conn = sqlite3.connect(self.db_file)
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS News (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            source TEXT,
                            author TEXT,
                            title TEXT,
                            description TEXT,
                            content TEXT,
                            url TEXT,
                            publish_time TEXT,
                            category TEXT
                          )''')
        self.conn.commit()

    def scrape_google_news(self, categories):
        """Scrape Google News for the given categories."""
        base_url = "https://news.google.com/topics/"
        results = []


        for category, topic_id in categories.items():
            try:
                print(f"Fetching Google News for category: {category}")
                url = f"{base_url}{topic_id}?hl=en-CA&gl=CA&ceid=CA:en"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
                }
                response = requests.get(url, headers=headers)
                response.raise_for_status()

                # Parse HTML response
                soup = BeautifulSoup(response.text, "html.parser")
                articles = soup.select("article")
                for article in soup.find_all("article")[:5]:  # Limit to 5 articles per category
                    try:
                        h3_tag = article.find("h3")
                        title = h3_tag.text if h3_tag else "N/A"

                        p_tag = article.find("p")
                        description = p_tag.text if p_tag else "N/A"

                        link_tag = article.find("a", href=True)
                        link = link_tag["href"] if link_tag else "N/A"
                        full_url = f"https://news.google.com{link}" if not link.startswith("http") else link

                        results.append({
                            "Source": "GNews",
                            "Author": "N/A",
                            "Title": title,
                            "Description": description,
                            "Content": description,
                            "URL": full_url,
                            "PublishTime": datetime.utcnow().isoformat(),
                            "Category": category
                        })
                    except Exception as article_error:
                        print(f"Error parsing article in category '{category}': {article_error}")
            except Exception as e:
                print(f"Unexpected error while fetching Google News for category '{category}': {e}")

        return results

    def fetch_newsapi_news(self, categories):
        base_url = "https://newsapi.org/v2/top-headlines"
        results = []

        for category in categories:
            try:
                print(f"Fetching NewsAPI data for category: {category}")
                params = {
                    "apiKey": self.newsapi_key,
                    "category": category.lower(),
                    "pageSize": 5,  # Fetch 5 articles per category
                    "country": "ca"  # Canadian news
                }
                response = requests.get(base_url, params=params)
                response.raise_for_status()

                articles = response.json().get("articles", [])
                for article in articles:
                    results.append({
                        "Source": "NewsAPI",
                        "Author": article.get("author"),
                        "Title": article.get("title"),
                        "Description": article.get("description"),
                        "Content": article.get("content"),
                        "URL": article.get("url"),
                        "PublishTime": article.get("publishedAt"),
                        "Category": category
                    })
            except Exception as e:
                print(f"Unexpected error while fetching NewsAPI data for category '{category}': {e}")

        return results

    def save_to_db(self, data):
        """Save news data to the SQLite database."""
        cursor = self.conn.cursor()
        for item in data:
            cursor.execute('''INSERT INTO News (source, author, title, description, content, url, publish_time, category)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                           (item["Source"], item["Author"], item["Title"], item["Description"],
                            item["Content"], item["URL"], item["PublishTime"], item["Category"]))
        self.conn.commit()

    def display_top_results(self, limit=5):
        """Display the top N results from the SQLite database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM News LIMIT ?", (limit,))
        for row in cursor.fetchall():
            print(row)

    def close_connection(self):
        """Close the SQLite database connection."""
        if self.conn:
            self.conn.close()


def main():
    # News Categories with Google News topic IDs
    CATEGORIES = {
        "Canada": "CAAqKAgKIiJDQkFTRXdvSkwyMHZNRFp1YlY4d1pYTXpMM1F4TmpRekVnSmxiaWdBUAE",
        "Local": "CAAqLAgKIiZDQkFTRXdvSkwyMHZNRFp1YlY4d1pYTXpMM1F4TmpRekVnSmxiaWdBUAE",
        "World": "CAAqLAgKIiZDQkFTRXdvSkwyMHZNRFp1YlY4d1pYTXpMM1F4TmpRekVnSmxiaWdBUAE",
        "Business": "CAAqKAgKIiJDQkFTRXdvSkwyMHZNRFp1YlY4d1pYTXpMM1F4TmpRekVnSmxiaWdBUAE",
        "Tech": "CAAqLAgKIiZDQkFTRXdvSkwyMHZNRFp1YlY4d1pYTXpMM1F4TmpRekVnSmxiaWdBUAE",
        "Entertainment": "CAAqLAgKIiZDQkFTRXdvSkwyMHZNRFp1YlY4d1pYTXpMM1F4TmpRekVnSmxiaWdBUAE",
        "Sports": "CAAqKAgKIiJDQkFTRXdvSkwyMHZNRFp1YlY4d1pYTXpMM1F4TmpRekVnSmxiaWdBUAE",
        "Science": "CAAqKAgKIiJDQkFTRXdvSkwyMHZNRFp1YlY4d1pYTXpMM1F4TmpRekVnSmxiaWdBUAE",
        "Health": "CAAqLAgKIiZDQkFTRXdvSkwyMHZNRFp1YlY4d1pYTXpMM1F4TmpRekVnSmxiaWdBUAE"
    }


    # Initialize and run the scraper
    db_file = "news_data.db"
    newsapi_key = ""  # Replace with your API key
    scraper = NewsScraper(db_file=db_file, newsapi_key=newsapi_key)
    scraper.initialize_db()


    # Scrape Google News
    print("Scraping Google News...")
    google_news_data = scraper.scrape_google_news(CATEGORIES)


    # Fetch NewsAPI data
    print("Fetching NewsAPI data...")
    newsapi_categories = list(CATEGORIES.keys())
    newsapi_data = scraper.fetch_newsapi_news(newsapi_categories)


    # Save data to SQLite
    print("Saving data to SQLite database...")
    scraper.save_to_db(google_news_data)
    scraper.save_to_db(newsapi_data)
    print("Data saved successfully!")


    # Display top results
    print("Top 5 results from SQLite database:")
    scraper.display_top_results(limit=5)


    # Close database connection
    scraper.close_connection()



if __name__ == "__main__":
    main()



