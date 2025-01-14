
import asyncio
import os
import random
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pytz
import tweepy
import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from urllib.parse import urljoin
import logging
from dotenv import load_dotenv

load_dotenv()

class ContentAggregator:
    """Aggregates content from multiple sources for AI training"""
    def __init__(self):
        self.setup_directories()
        self.setup_twitter_client()
        
        # Content categories and keywords
        self.categories = {
            'ai_agents': [
                'ai agent', 'autonomous agent', 'agent ecosystem',
                'ai assistant', 'language model', 'llm'
            ],
            'blockchain': [
                'blockchain', 'web3', 'cryptocurrency', 'smart contract',
                'defi', 'dao', 'nft'
            ],
            'tech_news': [
                'artificial intelligence', 'machine learning',
                'technology innovation', 'tech startup'
            ]
        }
        
        # Configure news sources
        self.news_sources = {
            'ai_news': [
                'https://artificialintelligence-news.com/feed/',
                'https://www.unite.ai/feed/',
            ],
            'crypto_news': [
                'https://cointelegraph.com/rss',
                'https://decrypt.co/feed',
                'https://cryptoslate.com/feed/'
            ],
            'tech_news': [
                'https://techcrunch.com/feed/',
                'https://venturebeat.com/feed/'
            ]
        }
        
        # Twitter accounts to monitor
        self.twitter_accounts = {
            'ai_experts': [
                'sama',         # Sam Altman
                'ylecun',      # Yann LeCun
                'AndrewYNg'    # Andrew Ng
            ],
            'crypto_experts': [
                'vitalikbuterin',  # Vitalik Buterin
                'SBF_FTX',        # Sam Bankman-Fried
                'cz_binance'      # Changpeng Zhao
            ]
        }

    def setup_directories(self):
        """Initialize directory structure"""
        self.data_dir = os.path.join('data', 'content')
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def setup_twitter_client(self):
        """Initialize Twitter API client"""
        try:
            self.twitter = tweepy.Client(
                bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
                consumer_key=os.getenv('TWITTER_API_KEY'),
                consumer_secret=os.getenv('TWITTER_API_SECRET_KEY'),
                access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
                access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
            )
        except Exception as e:
            logging.error(f"Error initializing Twitter client: {str(e)}")
            self.twitter = None

    async def fetch_tweets(self, username: str, category: str) -> List[Dict]:
        """Fetch recent tweets from a user"""
        tweets = []
        try:
            user = self.twitter.get_user(username=username)
            if user.data:
                response = self.twitter.get_users_tweets(
                    id=user.data.id,
                    max_results=100,
                    tweet_fields=['created_at', 'public_metrics', 'context_annotations']
                )
                
                if response.data:
                    for tweet in response.data:
                        if self._is_relevant_content(tweet.text, category):
                            tweets.append({
                                'type': 'tweet',
                                'category': category,
                                'author': username,
                                'content': tweet.text,
                                'timestamp': tweet.created_at,
                                'metrics': tweet.public_metrics,
                                'url': f"https://twitter.com/{username}/status/{tweet.id}"
                            })
        except Exception as e:
            logging.error(f"Error fetching tweets from {username}: {str(e)}")
        
        return tweets

    async def fetch_rss_feed(self, feed_url: str, category: str) -> List[Dict]:
        """Fetch articles from RSS feed"""
        articles = []
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                if self._is_relevant_content(entry.title + " " + entry.summary, category):
                    articles.append({
                        'type': 'article',
                        'category': category,
                        'title': entry.title,
                        'content': entry.summary,
                        'url': entry.link,
                        'timestamp': datetime.fromtimestamp(time.mktime(entry.published_parsed)),
                        'author': entry.get('author', 'Unknown'),
                        'source': feed_url
                    })
        except Exception as e:
            logging.error(f"Error fetching RSS feed {feed_url}: {str(e)}")
        
        return articles

    def _is_relevant_content(self, text: str, category: str) -> bool:
        """Check if content is relevant for a category"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.categories[category])

    async def aggregate_content(self) -> pd.DataFrame:
        """Aggregate content from all sources"""
        all_content = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Fetch tweets
        for category, users in self.twitter_accounts.items():
            for username in users:
                tweets = await self.fetch_tweets(username, category)
                all_content.extend(tweets)

        # Fetch news
        for category, feeds in self.news_sources.items():
            for feed_url in feeds:
                articles = await self.fetch_rss_feed(feed_url, category)
                all_content.extend(articles)

        # Save raw data
        raw_file = os.path.join(self.raw_dir, f'raw_content_{timestamp}.json')
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(all_content, f, default=str)

        # Convert to DataFrame and save processed data
        df = pd.DataFrame(all_content)
        processed_file = os.path.join(self.processed_dir, f'processed_content_{timestamp}.csv')
        df.to_csv(processed_file, index=False)

        return df

    def get_latest_content(self, category: Optional[str] = None, days: int = 1) -> pd.DataFrame:
        """Get recent content, optionally filtered by category"""
        try:
            files = os.listdir(self.processed_dir)
            if not files:
                return pd.DataFrame()

            latest_file = max(files)
            df = pd.read_csv(os.path.join(self.processed_dir, latest_file))
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff_date]
            
            # Filter by category if specified
            if category:
                df = df[df['category'] == category]
            
            return df
            
        except Exception as e:
            logging.error(f"Error retrieving content: {str(e)}")
            return pd.DataFrame()