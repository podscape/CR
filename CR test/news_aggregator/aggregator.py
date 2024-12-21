import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Set
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import time
import hashlib
from urllib.parse import urlparse, quote_plus
import logging
import json
import feedparser
from fake_useragent import UserAgent

class NewsAggregator:
    def __init__(self, vector_dimension: int = 384, max_cache_size: int = 1000):
        self.vector_dimension = vector_dimension
        self.max_cache_size = max_cache_size
        self.sources = {}
        self.ua = UserAgent()
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load default news sources
        self._load_default_sources()
        
        # Queues for different types of tasks
        self.source_queue = queue.Queue()  # For RSS/direct news sources
        self.google_queue = queue.Queue()  # For Google News queries
        self.update_queue = queue.Queue()  # For article processing
        
        # Article storage
        self.article_vectors = np.zeros((0, vector_dimension))
        self.articles = []
        self.article_timestamps = []
        self.processed_urls = set()
        
        # Start worker threads
        self.running = True
        self.threads = [
            threading.Thread(target=self._process_sources),
            threading.Thread(target=self._process_google_news),
            threading.Thread(target=self._process_updates)
        ]
        for thread in self.threads:
            thread.daemon = True
            thread.start()

    def _load_default_sources(self):
        """Load default news sources with their selectors."""
        self.sources = {
            'reuters': {
                'name': 'Reuters',
                'rss_url': 'https://www.rss.reuters.com/news',
                'selectors': {
                    'title': 'h1.article-header__title',
                    'content': 'div.article-body__content',
                    'author': 'a.author-name',
                }
            },
            'ap': {
                'name': 'Associated Press',
                'rss_url': 'https://feeds.apnews.com/rss/topnews',
                'selectors': {
                    'title': 'h1',
                    'content': 'div.Article',
                    'author': 'span.byline',
                }
            },
            # Add more default sources as needed
        }

    def add_news_source(self, name: str, rss_url: str, selectors: Dict[str, str]):
        """Add a new news source configuration."""
        self.sources[name] = {
            'name': name,
            'rss_url': rss_url,
            'selectors': selectors
        }
        self.source_queue.put(name)

    def track_topic(self, topic: str, update_interval: int = 3600):
        """
        Track a topic on Google News.
        
        Args:
            topic: Topic to track
            update_interval: Update interval in seconds
        """
        self.google_queue.put({
            'topic': topic,
            'interval': update_interval,
            'last_update': datetime.now() - timedelta(seconds=update_interval)
        })

    def _fetch_google_news(self, topic: str) -> List[Dict]:
        """Fetch articles from Google News for a given topic."""
        try:
            # Encode topic for URL
            encoded_topic = quote_plus(topic)
            
            # Use Google News RSS feed
            url = f"https://news.google.com/rss/search?q={encoded_topic}"
            
            # Parse RSS feed
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:10]:  # Limit to top 10 results
                article = {
                    'url': entry.link,
                    'title': entry.title,
                    'timestamp': datetime.fromtimestamp(time.mktime(entry.published_parsed)),
                    'source': 'Google News',
                    'topic': topic
                }
                articles.append(article)
            
            return articles
                
        except Exception as e:
            self.logger.error(f"Error fetching Google News for {topic}: {str(e)}")
            return []

    def _process_google_news(self):
        """Process Google News topics."""
        topics = {}  # Store topics and their update times
        
        while self.running:
            try:
                # Check for new topics
                while not self.google_queue.empty():
                    topic_info = self.google_queue.get_nowait()
                    topics[topic_info['topic']] = topic_info
                
                # Process topics that need updating
                current_time = datetime.now()
                for topic, info in topics.items():
                    if current_time - info['last_update'] >= timedelta(seconds=info['interval']):
                        articles = self._fetch_google_news(topic)
                        
                        for article in articles:
                            if article['url'] not in self.processed_urls:
                                self._process_article(article)
                        
                        info['last_update'] = current_time
                
                time.sleep(10)  # Prevent excessive CPU usage
                
            except Exception as e:
                self.logger.error(f"Error in Google News processing: {str(e)}")
                time.sleep(5)

    def _fetch_rss_articles(self, source_name: str) -> List[Dict]:
        """Fetch articles from an RSS feed."""
        source = self.sources[source_name]
        try:
            feed = feedparser.parse(source['rss_url'])
            
            articles = []
            for entry in feed.entries:
                article = {
                    'url': entry.link,
                    'title': entry.title,
                    'timestamp': datetime.fromtimestamp(time.mktime(entry.published_parsed)),
                    'source': source['name']
                }
                articles.append(article)
            
            return articles
                
        except Exception as e:
            self.logger.error(f"Error fetching RSS feed for {source_name}: {str(e)}")
            return []

    def _process_article(self, article: Dict):
        """Process a single article."""
        try:
            # Fetch full article content
            headers = {'User-Agent': self.ua.random}
            response = requests.get(article['url'], headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get source selectors if available
            source_domain = urlparse(article['url']).netloc
            source = next((s for s in self.sources.values() if source_domain in s['rss_url']), None)
            
            if source:
                # Extract content using source-specific selectors
                content = soup.select_one(source['selectors']['content'])
                if content:
                    article['content'] = content.get_text().strip()
                
                author = soup.select_one(source['selectors']['author'])
                if author:
                    article['author'] = author.get_text().strip()
            else:
                # Generic content extraction
                article['content'] = ' '.join([p.get_text() for p in soup.find_all('p')])
            
            # Generate embedding
            if 'content' in article:
                vector = self._generate_simple_embedding(f"{article['title']} {article['content']}")
                
                self.update_queue.put({
                    'operation': 'add',
                    'article': article,
                    'vector': vector
                })
                
                self.processed_urls.add(article['url'])
            
        except Exception as e:
            self.logger.error(f"Error processing article {article['url']}: {str(e)}")

    def _process_sources(self):
        """Process RSS news sources."""
        while self.running:
            try:
                # Process each source
                for source_name in self.sources:
                    articles = self._fetch_rss_articles(source_name)
                    
                    for article in articles:
                        if article['url'] not in self.processed_urls:
                            self._process_article(article)
                    
                    time.sleep(5)  # Rate limiting between sources
                
                time.sleep(300)  # Wait 5 minutes before next round
                
            except Exception as e:
                self.logger.error(f"Error in source processing: {str(e)}")
                time.sleep(5)

    def _generate_simple_embedding(self, text: str) -> np.ndarray:
        """Generate a simple embedding vector for text."""
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        vector = np.zeros(self.vector_dimension)
        for i in range(min(32, self.vector_dimension)):
            vector[i] = float(hash_bytes[i])
            
        return vector / np.linalg.norm(vector)

    def _process_updates(self):
        """Process article updates."""
        while self.running:
            try:
                update = self.update_queue.get(timeout=1)
                
                if update['operation'] == 'add':
                    self.article_vectors = np.vstack([
                        self.article_vectors,
                        update['vector'].reshape(1, -1)
                    ])
                    self.articles.append(update['article'])
                    self.article_timestamps.append(update['article']['timestamp'])
                    
                    if len(self.articles) > self.max_cache_size:
                        self.article_vectors = self.article_vectors[1:]
                        self.articles.pop(0)
                        self.article_timestamps.pop(0)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing update: {str(e)}")

    def retrieve_articles(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant articles for a query."""
        if len(self.article_vectors) == 0:
            return []
        
        similarities = np.dot(self.article_vectors, query_vector)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            article = self.articles[idx].copy()
            article['similarity'] = float(similarities[idx])
            results.append(article)
        
        return results

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        for thread in self.threads:
            thread.join()

# Example usage
def example_usage():
    # Initialize aggregator
    aggregator = NewsAggregator()
    
    # Add custom news source
    aggregator.add_news_source(
        name="TechNews",
        rss_url="https://example.com/tech/rss",
        selectors={
            'title': 'h1.title',
            'content': 'div.article-content',
            'author': 'span.author'
        }
    )
    
    # Track topics
    aggregator.track_topic("artificial intelligence")
    aggregator.track_topic("climate change")
    
    # Wait for articles to be collected
    time.sleep(60)
    
    # Query for similar articles
    query_vector = np.random.randn(384)  # Replace with proper query embedding
    results = aggregator.retrieve_articles(query_vector, top_k=5)
    
    # Print results
    for article in results:
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print(f"URL: {article['url']}")
        print(f"Similarity: {article['similarity']:.3f}")
        print("---")
    
    # Clean up
    aggregator.cleanup()

if __name__ == "__main__":
    example_usage()