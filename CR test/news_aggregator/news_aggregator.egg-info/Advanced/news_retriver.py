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
from sqlalchemy import create_engine, Column, String, DateTime, Float, JSON, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Initialize the SQLAlchemy base
Base = declarative_base()

class Article(Base):
    """SQLAlchemy model for storing articles"""
    __tablename__ = 'articles'
    
    url = Column(String, primary_key=True)
    title = Column(String)
    content = Column(Text)
    source = Column(String)
    topic = Column(String)
    timestamp = Column(DateTime)
    sentiment_score = Column(Float)
    sentiment_subjectivity = Column(Float)
    summary = Column(Text)
    keywords = Column(JSON)
    article_metadata = Column(JSON)

class ContentAnalyzer:
    """Handles content analysis tasks"""
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def analyze_content(self, text: str) -> Dict:
        """Analyze content and return metrics"""
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Text summarization
        doc = self.nlp(text[:10000])  # Limit text length for processing
        sentences = list(doc.sents)
        
        # Extract important sentences
        important_sentences = []
        for sent in sentences:
            if len(list(sent.ents)) > 0 or len([chunk for chunk in sent.noun_chunks]) > 1:
                important_sentences.append(sent.text)
        
        summary = " ".join(important_sentences[:3])
        
        # Extract keywords
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
            keywords = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10])
        except:
            keywords = {}
        
        return {
            'sentiment_score': sentiment.polarity,
            'sentiment_subjectivity': sentiment.subjectivity,
            'summary': summary,
            'keywords': keywords
        }

class NewsRetriever:
    """Main class for retrieving and analyzing news content"""
    def __init__(self, db_url: str = "sqlite:///articles.db"):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.ua = UserAgent()
        self.sources = self._initialize_sources()
        
        # Initialize database
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize content analyzer
        self.analyzer = ContentAnalyzer()
        
    def _initialize_sources(self) -> Dict:
        """Initialize default news sources"""
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

    def get_topic_content(self, topic: str, include_sources: List[str] = None) -> List[Dict]:
        """
        Get content about a specific topic
        """
        all_content = []
        
        try:
            # Get Google News content
            google_articles = self._fetch_google_news(topic)
            all_content.extend(google_articles)
            
            # Get content from other sources
            for source_name, source_info in self.sources.items():
                if include_sources and source_name not in include_sources:
                    continue
                    
                source_articles = self._fetch_source_content(source_name, topic)
                all_content.extend(source_articles)
            
            # Sort by timestamp
            all_content.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return all_content
            
        except Exception as e:
            self.logger.error(f"Error getting topic content: {str(e)}")
            return []

    def _fetch_google_news(self, topic: str, max_results: int = 10) -> List[Dict]:
        """Fetch articles from Google News"""
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
                
                full_article = self._fetch_article_content(article)
                if full_article:
                    articles.append(full_article)
                    self._store_article(full_article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching Google News: {str(e)}")
            return []

    def _fetch_source_content(self, source_name: str, topic: str) -> List[Dict]:
        """Fetch content from a specific source"""
        try:
            source = self.sources[source_name]
            feed = feedparser.parse(source['rss_url'])
            
            articles = []
            for entry in feed.entries:
                # Basic keyword matching for topic relevance
                if topic.lower() not in entry.title.lower():
                    continue
                    
                article = {
                    'title': entry.title,
                    'url': entry.link,
                    'source': source['name'],
                    'timestamp': datetime.fromtimestamp(time.mktime(entry.published_parsed)),
                    'topic': topic
                }
                
                full_article = self._fetch_article_content(article)
                if full_article:
                    articles.append(full_article)
                    self._store_article(full_article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching from {source_name}: {str(e)}")
            return []

    def _fetch_article_content(self, article: Dict) -> Optional[Dict]:
        """Fetch and process full article content"""
        try:
            headers = {'User-Agent': self.ua.random}
            response = requests.get(article['url'], headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get main content
            content = soup.find_all('p')
            article['content'] = ' '.join([p.get_text().strip() for p in content])
            
            # Analyze content
            if article['content']:
                analysis = self.analyzer.analyze_content(article['content'])
                article.update(analysis)
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error fetching article content: {str(e)}")
            return None

    def _store_article(self, article_data: Dict):
        """Store article in database"""
        try:
            session = self.Session()
            
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
                article_metadata=article_data.get('article_metadata', {})
            )
            
            session.merge(article)
            session.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing article: {str(e)}")
            session.rollback()
        finally:
            session.close()

def main():
    # Initialize retriever
    retriever = NewsRetriever(db_url="sqlite:///articles.db")
    
    # Test with a topic
    topic = "artificial intelligence"
    print(f"\nRetrieving content about: {topic}")
    
    try:
        # Get content
        results = retriever.get_topic_content(
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