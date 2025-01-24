import sqlite3
import requests
from bs4 import BeautifulSoup
from datetime import datetime


def initialize_database():
    """
    Create a new database (if it doesn't exist) with a 'news_articles' table.
    """
    conn = sqlite3.connect('collected_news.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_articles (
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


def gnews_scraper():
    """
    Scrape top 5 articles from Google News for each defined topic.
    Returns a list of tuples corresponding to the news_articles schema.
    """

    # Google News topic IDs mapped to categories:
    TOPIC_IDS = {
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
    google_main = "https://news.google.com"

    fetched_articles = []
    for category_name, topic_id in TOPIC_IDS.items():
        gnews_url = f"{base_url}/{topic_id}?hl=en-CA&gl=CA&ceid=CA%3Aen"
        try:
            resp = requests.get(gnews_url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Limit to top 5 articles
            articles_section = soup.find_all('article', limit=5)

            for entry in articles_section:
                # Try extracting the <a> tag that holds the title
                title_candidates = [
                    link for link in entry.find_all('a', href=True)
                    if 'WwrzSb' not in link.get('class', [])
                ]
                title_tag = title_candidates[0] if title_candidates else None
                article_title = title_tag.text.strip() if (title_tag and title_tag.text) else 'N/A'

                # Description is not clearly available from the snippet
                article_description = "N/A"

                # Construct absolute article URL
                rel_url = title_tag['href'] if (title_tag and title_tag.has_attr('href')) else ''
                if rel_url.startswith('./'):
                    full_link = f"{google_main}{rel_url[1:]}"  # remove '.' to make it /xyz...
                else:
                    full_link = rel_url or "N/A"

                # Google as the source
                source_origin = "GoogleNews"

                # Extract publish time from <time> if available
                time_tag = entry.find('time', class_='hvbAAd')
                pub_time = time_tag['datetime'] if (time_tag and time_tag.has_attr('datetime')) else 'N/A'

                # Attempt extracting author text
                author_tag = entry.find('div', class_='bInasb')
                article_author = author_tag.text.strip().replace('By ', '') if author_tag else 'Unknown'

                # Optionally fetch the entire article text
                article_body = get_article_text(full_link)

                # Construct the tuple
                fetched_articles.append((
                    source_origin,
                    article_author,
                    article_title,
                    article_description,
                    article_body,
                    full_link,
                    pub_time,
                    category_name
                ))
        except requests.RequestException as e:
            print(f"Error scraping category {category_name}: {e}")

    return fetched_articles


def get_article_text(page_url):
    """
    Fetch the body text from an article by making a separate request.
    """
    if not page_url or page_url == 'N/A':
        return 'No Content'

    try:
        req = requests.get(page_url, timeout=10)
        parser = BeautifulSoup(req.text, 'html.parser')
        paragraphs = parser.find_all('p')
        text_content = ' '.join(
            p.text.strip() for p in paragraphs
            if p.text.strip()
        )
        return text_content if text_content else 'No Content'
    except Exception as err:
        print(f"Failed to retrieve article text from {page_url}: {err}")
        return 'No Content'


def pull_newsapi_items(api_token):
    """
    Fetch top 5 headlines from NewsAPI for several categories.
    Returns a list of tuples matching the news_articles schema.
    """

    # Some categories recognized by the NewsAPI
    newsapi_cats = ["general", "Business", "Technology", "Entertainment", "Sports", "Science", "Health"]
    api_endpoint = "https://newsapi.org/v2/top-headlines"

    collected_data = []
    for cat in newsapi_cats:
        try:
            resp = requests.get(api_endpoint, params={
                'category': cat.lower(),
                'apiKey': api_token,
                'country': 'ca'  # For Canada
            })
            if resp.status_code == 200:
                # Limit each category to the top 5 articles
                entries = resp.json().get('articles', [])[:5]
                for art in entries:
                    collected_data.append((
                        art.get('source', {}).get('name', 'Unknown'),
                        art.get('author') if art.get('author') else 'N/A',
                        art.get('title') if art.get('title') else 'N/A',
                        art.get('description') if art.get('description') else 'N/A',
                        art.get('content') if art.get('content') else 'N/A',
                        art.get('url') if art.get('url') else 'N/A',
                        art.get('publishedAt') if art.get('publishedAt') else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        cat  # category name
                    ))
            else:
                print(f"NewsAPI responded with status code {resp.status_code} for category {cat}.")
        except requests.RequestException as exc:
            print(f"Request error for NewsAPI category {cat}: {exc}")

    return collected_data


def store_in_database(articles):
    """
    Persist a list of article tuples into the 'news_articles' table in SQLite.
    """
    conn = sqlite3.connect('collected_news.db')
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT INTO news_articles (source, author, title, description, content, url, publish_time, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', articles)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # 1. Setup database
    initialize_database()

    # 2. Scrape Google News
    google_articles = gnews_scraper()
    store_in_database(google_articles)

    # 3. Fetch from NewsAPI
    NEWSAPI_KEY = '4163696f64184843a677e891166e5d86'  # Replace with your valid NewsAPI key
    newsapi_articles = pull_newsapi_items(NEWSAPI_KEY)
    store_in_database(newsapi_articles)

    print("All news data successfully saved into 'collected_news.db'.")
