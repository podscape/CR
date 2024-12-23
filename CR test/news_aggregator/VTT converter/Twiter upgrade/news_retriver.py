from typing import List, Dict, Optional
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging

class NewsRetriever:
    """Enhanced async news retriever"""
    def __init__(self, db_url: str = "sqlite:///news_analysis.db"):
        self.db_url = db_url
        self.session = None
        self.logger = logging.getLogger(__name__)
    
    async def init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def fetch_article(self, url: str) -> Optional[Dict]:
        """Fetch article content asynchronously"""
        try:
            await self.init_session()
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    content = ' '.join([p.text for p in soup.find_all('p')])
                    return {
                        'url': url,
                        'content': content,
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            self.logger.error(f"Error fetching article {url}: {str(e)}")
            return None
    
    async def monitor_news(self, topics: List[str], interval_minutes: int = 15):
        """Monitor news for given topics"""
        while True:
            try:
                for topic in topics:
                    articles = await self.get_topic_articles(topic)
                    # Process and store articles
                    await self.process_articles(articles)
                
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                self.logger.error(f"Error monitoring news: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

class EnhancedNewsRetriever(NewsRetriever):
    """Enhanced version with social media integration"""
    def __init__(self, db_url: str, twitter_integration: Optional['TwitterIntegration'] = None):
        super().__init__(db_url)
        self.twitter = twitter_integration
    
    async def process_articles(self, articles: List[Dict]):
        """Process and potentially tweet about articles"""
        for article in articles:
            # Store in database
            await self.store_article(article)
            
            # Maybe tweet about interesting articles
            if self.twitter and self.is_interesting_article(article):
                await self.twitter.tweet_about_article(article)
    
    def is_interesting_article(self, article: Dict) -> bool:
        """Determine if article is worth tweeting about"""
        # Add your criteria here
        return True  # Placeholder