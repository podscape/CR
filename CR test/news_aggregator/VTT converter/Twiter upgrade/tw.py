import asyncio
import os
import random
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
import pytz
import json
import aiohttp
import tweepy
import feedparser
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from fake_useragent import UserAgent
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, DateTime, Float, JSON, Text, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from textblob import TextBlob
from utils.llama_hf import AsyncTextGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize SQLAlchemy
Base = declarative_base()

# Twitter prompts
BASE_PROMPT = "You are a knowledgeable crypto and AI expert. Create an engaging tweet about: "

tweet_prompts = {
    "ai_agents": [
        "What's the latest breakthrough in AI?",
        "How are AI agents transforming business?",
        "Discuss emerging AI trends"
    ],
    "crypto": [
        "What's moving the crypto market today?",
        "Latest developments in blockchain",
        "Crypto market analysis"
    ],
    "NFTs": [
        "Notable NFT projects to watch",
        "NFT market trends",
        "Innovation in digital collectibles"
    ],
    "Solana": [
        "Solana ecosystem updates",
        "SOL performance analysis",
        "New Solana projects launch"
    ],
    "engagement": [
        "What are your thoughts on this?",
        "Share your experience with",
        "How do you see this developing?"
    ],
    "memecoins": [
        "Latest memecoin movements",
        "Community-driven token updates",
        "Viral crypto trends"
    ],
    "humor": [
        "Lighter side of crypto",
        "Tech humor of the day",
        "Fun blockchain facts"
    ]
}

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

class TwitterIntegration:
    """Enhanced Twitter/X integration with automated posting"""
    def __init__(self, 
                 consumer_key: str = os.getenv('TWITTER_API_KEY'),
                 consumer_secret: str = os.getenv('TWITTER_API_SECRET_KEY'),
                 access_token: str = os.getenv('TWITTER_ACCESS_TOKEN'),
                 access_token_secret: str = os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        # Initialize Twitter client
        self.client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )
        
        # Initialize LLaMA
        self.generator = AsyncTextGenerator(model_name)
        
        # Configure timezone
        self.timezone = pytz.timezone('US/Eastern')
        
        # Initialize monitored accounts
        self.monitored_accounts = {
            'marionawfal': None,
            'shawmakesmagic': None,
            'tonybetw': None
        }
        self._initialize_user_ids()
        
        # Define content categories and schedules
        self.categories = {
            "morning": ["ai_agents", "engagement"],
            "afternoon": ["memecoins", "humor"],
            "evening": ["crypto", "engagement"],
            "night": ["NFTs", "Solana"]
        }

    def _initialize_user_ids(self):
        """Initialize user IDs for monitored accounts"""
        try:
            for username in self.monitored_accounts.keys():
                user = self.client.get_user(username=username)
                if user.data:
                    self.monitored_accounts[username] = user.data.id
        except Exception as e:
            logger.error(f"Error initializing Twitter user IDs: {str(e)}")

    def get_est_hour(self) -> int:
        """Get current hour in EST"""
        utc_now = datetime.now(pytz.utc)
        est_now = utc_now.astimezone(self.timezone)
        return est_now.hour

    def get_next_category(self, hour: int) -> str:
        """Determine content category based on time"""
        if 6 <= hour < 12:
            return random.choice(self.categories["morning"])
        elif 12 <= hour < 17:
            return random.choice(self.categories["afternoon"])
        elif 17 <= hour < 22:
            return random.choice(self.categories["evening"])
        else:
            return random.choice(self.categories["night"])

    async def generate_tweet(self, base_prompt: str) -> str:
        """Generate tweet content using LLaMA"""
        try:
            current_hour = self.get_est_hour()
            category = self.get_next_category(current_hour)
            category_prompt = random.choice(tweet_prompts[category])
            
            print(f"Generating tweet for category: {category}")
            print(f"Prompt: {category_prompt}")
            
            response = await self.generator.generate_text(base_prompt + category_prompt)
            
            if len(response.response) > 280:
                response = await self.generator.generate_text(
                    "Shorten this to 280 characters: " + response.response
                )
            
            return response.response
            
        except Exception as e:
            logger.error(f"Error generating tweet: {str(e)}")
            return None

    async def post_tweet(self, content: str):
        """Post tweet with error handling"""
        try:
            if content:
                self.client.create_tweet(text=content)
                logger.info(f"Posted tweet: {content}")
        except Exception as e:
            logger.error(f"Error posting tweet: {str(e)}")

    async def get_user_tweets(self, username: str, max_results: int = 10) -> List[Dict]:
        """Fetch recent tweets from a user"""
        try:
            user_id = self.monitored_accounts.get(username)
            if not user_id:
                return []
            
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics', 'context_annotations']
            )
            
            results = []
            for tweet in tweets.data or []:
                tweet_data = {
                    'content': tweet.text,
                    'timestamp': tweet.created_at,
                    'metrics': tweet.public_metrics,
                    'source': f'Twitter-{username}',
                    'url': f"https://twitter.com/{username}/status/{tweet.id}",
                    'title': f"Tweet by {username}"
                }
                results.append(tweet_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching tweets from {username}: {str(e)}")
            return []

    async def auto_post_schedule(self):
        """Automatically post tweets on schedule"""
        while True:
            try:
                # Generate and post tweet
                tweet_content = await self.generate_tweet(BASE_PROMPT)
                if tweet_content:
                    await self.post_tweet(tweet_content)
                
                # Wait random time between 30-60 minutes
                wait_time = random.randint(30, 60) * 60
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error in auto posting: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def monitor_topics(self, topics: List[str]):
        """Monitor topics and generate relevant content"""
        while True:
            for topic in topics:
                try:
                    tweet_content = await self.generate_tweet(f"Create an informative tweet about {topic}: ")
                    if tweet_content:
                        await self.post_tweet(tweet_content)
                    await asyncio.sleep(random.randint(2, 5) * 60)
                except Exception as e:
                    logger.error(f"Error monitoring topic {topic}: {str(e)}")
            
            await asyncio.sleep(1800)  # Wait 30 minutes between cycles

async def main():
    try:
        # Create Twitter integration
        twitter = TwitterIntegration()
        
        # Define topics to monitor
        topics = [
            "artificial_intelligence",
            "cryptocurrency",
            "financial_markets",
            "geopolitics"
        ]
        
        # Create tasks
        tasks = [
            asyncio.create_task(twitter.auto_post_schedule()),
            asyncio.create_task(twitter.monitor_topics(topics))
        ]
        
        # Run tasks
        await asyncio.gather(*tasks)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        logger.info("Shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping all tasks...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")