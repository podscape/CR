import requests
from bs4 import BeautifulSoup
import tweepy
import feedparser
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
import logging
from urllib.parse import quote_plus
from fake_useragent import UserAgent
from sqlalchemy.orm import sessionmaker
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

from database_model import Article, initialize_database

class ContentAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def analyze_content(self, text: str) -> Dict:
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Text summarization
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        # Extract important sentences
        important_sentences = []
        for sent in sentences:
            if (len(list(sent.ents)) > 0 or
                len([chunk for chunk in sent.noun_chunks]) > 1):
                important_sentences.append(sent.text)
        
        summary = " ".join(important_sentences[:3])
        
        # Extract keywords
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = zip(feature_names, tfidf_matrix.toarray()[0])
            keywords = dict(sorted(scores, key=lambda x: x[1], reverse=True)[:10])
        except:
            keywords = {}
        
        return {
            'sentiment_score': sentiment.polarity,
            'sentiment_subjectivity': sentiment.subjectivity,
            'summary': summary,
            'keywords': keywords
        }

class MultiSourceRetriever:
    def __init__(self, db_url: str = "sqlite:///articles.db", twitter_bearer_token: Optional[str] = None):
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.ua = UserAgent()
        self.sources = self._initialize_sources()
        
        # Initialize database
        self.engine = initialize_database(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize analyzers
        self.analyzer = ContentAnalyzer()
        
        # Initialize Twitter client
        self.twitter_client = None
        if twitter_bearer_token:
            self.twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)

    def _initialize_sources(self) -> Dict:
        return {
            'associated_press': {
                'name': 'Associated Press',
                'rss_url': 'https://feeds.apnews.com/rss/topnews',
                'selectors': {
                    'title': 'h1',
                    'content': 'div.Article',
                    'author': 'span.byline'
                }
            },
            'reuters': {
                'name': 'Reuters',
                'rss_url': 'https://www.rss.reuters.com/news',
                'selectors': {
                    'title': 'h1.article-header__title',
                    'content': 'div.article-body__content',
                    'author': 'a.author-name'
                }
            }
        }

    def store_article(self, article_data: Dict):
        try:
            session = self.Session()
            
            # Perform content analysis if content exists
            if 'content' in article_data:
                analysis = self.analyzer.analyze_content(article_data['content'])
                article_data.update(analysis)
            
            # Create article object
            article = Article(
                url=article_data['url'],
                title=article_data.get('title', ''),
                content=article_data.get('content', ''),
                source=article_data['source'],
                topic=article_data.get('topic', ''),
                timestamp=article_data['timestamp'],
                sentiment_score=article_data.get('sentiment_score'),
                sentiment_subjectivity=article_data.get('sentiment_subjectivity'),
                summary=article_data.get('summary', ''),
                keywords=article_data.get('keywords', {}),
                article_metadata=article_data.get('article_metadata', {})  # Updated field name
            )
            
            session.merge(article)
            session.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing article: {str(e)}")
            session.rollback()
        finally:
            session.close()

    # Rest of the methods remain the same...
def retrieve_topic_content(self, topic: str, include_sources: List[str] = None) -> List[Dict]:
    """
    Retrieve content about a topic from multiple sources.
    
    Args:
        topic: Topic to search for
        include_sources: List of specific sources to include (None for all)
        
    Returns:
        List of articles with content and analysis
    """
    all_content = []
    
    try:
        # Fetch from Google News
        self.logger.info(f"Fetching Google News content for topic: {topic}")
        google_articles = self.fetch_google_news(topic)
        if google_articles:
            all_content.extend(google_articles)
        
        # Fetch from Twitter if available
        if self.twitter_client:
            self.logger.info(f"Fetching Twitter content for topic: {topic}")
            twitter_content = self.fetch_twitter_content(topic)
            if twitter_content:
                all_content.extend(twitter_content)
        
        # Fetch from other sources
        for source_name, source_info in self.sources.items():
            if include_sources is None or source_name in include_sources:
                self.logger.info(f"Fetching content from {source_name}")
                try:
                    # Parse RSS feed
                    feed = feedparser.parse(source_info['rss_url'])
                    
                    for entry in feed.entries:
                        article = {
                            'title': entry.title,
                            'url': entry.link,
                            'source': source_info['name'],
                            'timestamp': datetime.fromtimestamp(
                                time.mktime(entry.published_parsed)
                            ),
                            'topic': topic
                        }
                        
                        # Fetch full article content
                        full_article = self.fetch_article_content(article)
                        if full_article:
                            all_content.append(full_article)
                            # Store in database
                            self.store_article(full_article)
                            
                except Exception as e:
                    self.logger.error(f"Error fetching from {source_name}: {str(e)}")
                    continue
        
        # Sort by timestamp
        all_content.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return all_content
        
    except Exception as e:
        self.logger.error(f"Error retrieving content: {str(e)}")
        return []

def fetch_google_news(self, topic: str, max_results: int = 10) -> List[Dict]:
    """Fetch articles from Google News."""
    try:
        encoded_topic = quote_plus(topic)
        url = f"https://news.google.com/rss/search?q={encoded_topic}"
        
        feed = feedparser.parse(url)
        articles = []
        
        for entry in feed.entries[:max_results]:
            article = {
                'title': entry.title,
                'url': entry.link,
                'source': 'Google News',
                'timestamp': datetime.fromtimestamp(time.mktime(entry.published_parsed)),
                'topic': topic
            }
            
            # Fetch full article content
            full_article = self.fetch_article_content(article)
            if full_article:
                articles.append(full_article)
                # Store in database
                self.store_article(full_article)
        
        return articles
        
    except Exception as e:
        self.logger.error(f"Error fetching Google News for {topic}: {str(e)}")
        return []

def fetch_article_content(self, article: Dict) -> Optional[Dict]:
    """Fetch full content for an article."""
    try:
        headers = {'User-Agent': self.ua.random}
        response = requests.get(article['url'], headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try source-specific selectors first
        source_domain = article['source'].lower().replace(' ', '_')
        if source_domain in self.sources:
            selectors = self.sources[source_domain]['selectors']
            content_elem = soup.select_one(selectors['content'])
            if content_elem:
                article['content'] = content_elem.get_text().strip()
        
        # Fallback to general content extraction
        if 'content' not in article:
            # Get all paragraphs
            paragraphs = soup.find_all('p')
            article['content'] = ' '.join([p.get_text().strip() for p in paragraphs])
        
        # Analyze content
        if article['content']:
            analysis = self.analyzer.analyze_content(article['content'])
            article.update(analysis)
        
        return article
        
    except Exception as e:
        self.logger.error(f"Error fetching content for {article['url']}: {str(e)}")
        return None
    
def main():
    # Initialize retriever
    retriever = MultiSourceRetriever(db_url="sqlite:///articles.db")
    
    # Test with a topic
    topic = "artificial intelligence"
    print(f"\nRetrieving content about: {topic}")
    
    # Get content
    try:
        results = retriever.retrieve_topic_content(
            topic=topic,
            include_sources=['associated_press', 'reuters']
        )
        
        # Print results
        for article in results[:5]:  # Show first 5 articles
            print(f"\nTitle: {article.get('title', 'No title')}")
            print(f"Source: {article['source']}")
            print(f"URL: {article['url']}")
            print(f"Timestamp: {article['timestamp']}")
            if 'sentiment_score' in article:
                print(f"Sentiment: {article['sentiment_score']:.2f}")
            if 'summary' in article:
                print(f"Summary: {article['summary'][:200]}...")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    main()