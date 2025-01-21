import sqlite3
import requests
from bs4 import BeautifulSoup
from datetime import datetime


def create_database():
    conn = sqlite3.connect('news_data.db')
    cursor = conn.cursor()
    cursor.execute('''
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
    ''')
    conn.commit()
    conn.close()


def scrape_google_news():

    categories = {
    "Canada": "CAAqJggKIiBDQkFTRWdvSkwyMHZNR1F3TmpCbkVnVmxiaTFIUWlnQVAB",
    "Local": "CAAqHAgKIhZDQklTQ2pvSWJHOWpZV3hmZGpJb0FBUAE",
    "World": "CAAqKggKIiRDQkFTRlFvSUwyMHZNRGx1YlY4U0JXVnVMVWRDR2dKRFFTZ0FQAQ",
    "Business": "CAAqKggKIiRDQkFTRlFvSUwyMHZNRGx6TVdZU0JXVnVMVWRDR2dKRFFTZ0FQAQ",
    "Technology": "CAAqKggKIiRDQkFTRlFvSUwyMHZNRGRqTVhZU0JXVnVMVWRDR2dKRFFTZ0FQAQ",
    "Entertainment": "CAAqKggKIiRDQkFTRlFvSUwyMHZNREpxYW5RU0JXVnVMVWRDR2dKRFFTZ0FQAQ",
    "Sports": "CAAqKggKIiRDQkFTRlFvSUwyMHZNRFp1ZEdvU0JXVnVMVWRDR2dKRFFTZ0FQAQ",
    "Science": "CAAqKggKIiRDQkFTRlFvSUwyMHZNRFp0Y1RjU0JXVnVMVWRDR2dKRFFTZ0FQAQ",
    "Health": "CAAqJQgKIh9DQkFTRVFvSUwyMHZNR3QwTlRFU0JXVnVMVWRDS0FBUAE"
}

    
    base_url = "https://news.google.com/topics"
    base_google_url = "https://news.google.com"  # For constructing full URLs

    news_data = []
    for category, topic_id in categories.items():
        url = f"{base_url}/{topic_id}?hl=en-CA&gl=CA&ceid=CA%3Aen"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', limit=5)

        for article in articles:
            title_tags = [tag for tag in article.find_all('a', href=True) if 'WwrzSb' not in tag.get('class', [])]
            title_tag = title_tags[0] if title_tags else None  # Get the first matching tag
            title = title_tag.text.strip() if title_tag and title_tag.text else 'No Title'
            # Extract description (optional, might not always be present)
            description = 'No Description'  # Placeholder as it isn't clearly in the HTML

            # Extract URL
            url_relative = title_tag['href'] if title_tag and title_tag.has_attr('href') else 'No URL'
            url_full = f"{base_google_url}{url_relative}" if url_relative.startswith('./') else url_relative

            source = 'Gnews'

            # Extract publish time
            time_tag = article.find('time', class_='hvbAAd')
            publish_time = time_tag['datetime'] if time_tag and time_tag.has_attr('datetime') else 'No Time'

            # Extract author
            author_tag = article.find('div', class_='bInasb')
            author = author_tag.text.strip().replace('By ', '') if author_tag else 'Unknown'
        
            content = fetch_article_content(url_full)

            news_data.append((source, author, title, description, content, url_full, publish_time, category))

    return news_data

def fetch_article_content(article_url):
    try:
        response = requests.get(article_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        #with open("soup_output.txt", "w", encoding="utf-8") as file:
        #    file.write(soup.prettify())
        paragraphs = soup.find_all('p')  
        content = ' '.join([p.text.strip() for p in paragraphs if p.text.strip()])
        return content
    
    except Exception as e:
        print(f"Error fetching content from {article_url}: {e}")
        return 'No Content'

def fetch_newsapi_data(api_key):
    categories = ["general", "Business", "Technology", "Entertainment", "Sports", "Science", "Health"]
    base_url = "https://newsapi.org/v2/top-headlines"
    news_data = []

    for category in categories:
        response = requests.get(base_url, params={'category': category.lower(), 'apiKey': api_key})
        if response.status_code == 200:
            articles = response.json().get('articles', [])[:5]
            for article in articles:
                news_data.append((
                    article.get('source', {}).get('name', 'Unknown'),
                    article.get('author', 'Unknown'),
                    article.get('title', 'No Title'),
                    article.get('description', 'No Description'),
                    article.get('content', 'No Content'),
                    article.get('url', 'No URL'),
                    article.get('publishedAt', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    category
                ))
    
    return news_data

def save_to_database(news_data):
    conn = sqlite3.connect('news_data.db')
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT INTO news (source, author, title, description, content, url, publish_time, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', news_data)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Initialize the database
    create_database()

    # Scrape Google News
    google_news = scrape_google_news()
    save_to_database(google_news)

    # Fetch NewsAPI Data
    API_KEY = '4163696f64184843a677e891166e5d86' 
    newsapi_news = fetch_newsapi_data(API_KEY)
    save_to_database(newsapi_news)

    print("Data has been saved to SQLite database.")


