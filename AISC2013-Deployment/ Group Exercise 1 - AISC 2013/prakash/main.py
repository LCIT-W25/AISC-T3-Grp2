import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import datetime

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "google_news"
COLLECTION_NAME = "news"

# API Configuration
GOOGLE_NEWS_BASE_URL = "https://news.google.com/topics/"
NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"
NEWSAPI_KEY = ""

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


def connect_to_mongodb(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]


def scrape_google_news(category, category_id):
    url = f"{GOOGLE_NEWS_BASE_URL}{category_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []

        for article in soup.find_all('article')[:5]:
            title = article.find('h3').text if article.find('h3') else "N/A"
            description = article.find(
                'p').text if article.find('p') else "N/A"
            link = article.find('a', href=True)['href'] if article.find(
                'a', href=True) else "N/A"
            articles.append({
                "Source": "GNews",
                "Author": "N/A",
                "Title": title,
                "Description": description,
                "Content": description,
                "URL": f"https://news.google.com{link}",
                "PublishTime": datetime.utcnow().isoformat(),
                "Category": category
            })
        return articles
    except Exception as e:
        print(f"Error scraping Google News for category {category}: {e}")
        return []


def fetch_newsapi_data(category):
    params = {
        'apiKey': NEWSAPI_KEY,
        'category': category.lower(),
        'country': 'us',
        'pageSize': 5
    }
    try:
        response = requests.get(NEWSAPI_URL, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        return [
            {
                "Source": article.get('source', {}).get('name', 'NewsAPI'),
                "Author": article.get('author', 'N/A'),
                "Title": article.get('title', 'N/A'),
                "Description": article.get('description', 'N/A'),
                "Content": article.get('content', 'N/A'),
                "URL": article.get('url', 'N/A'),
                "PublishTime": article.get('publishedAt', datetime.utcnow().isoformat()),
                "Category": category
            } for article in articles
        ]
    except Exception as e:
        print(f"Error fetching data from NewsAPI for category {category}: {e}")
        return []


def save_to_mongodb(collection, data):
    if data:
        collection.insert_many(data)
        print(f"Inserted {len(data)} articles into MongoDB.")
    else:
        print("No data to insert into MongoDB.")


def main():
    collection = connect_to_mongodb(MONGO_URI, DB_NAME, COLLECTION_NAME)

    all_news = []

    # Scrape news from Google News
    for category, topic_id in CATEGORIES.items():
        print(f"Scraping Google News for category: {category}...")
        google_news = scrape_google_news(category, topic_id)
        all_news.extend(google_news)

    # Fetch news from NewsAPI
    for category in CATEGORIES.keys():
        print(f"Fetching NewsAPI data for category: {category}...")
        newsapi_data = fetch_newsapi_data(category)
        all_news.extend(newsapi_data)

    # Save all news to MongoDB
    save_to_mongodb(collection, all_news)


if __name__ == "__main__":
    main()
