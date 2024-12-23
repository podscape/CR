import asyncio
import os
import random
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pytz
import tweepy
from utils.llama_hf import AsyncTextGenerator
from twitter_prompts import tweet_prompts

class TwitterIntegration:
    """Enhanced async Twitter integration"""
    def __init__(self, 
                 consumer_key: str,
                 consumer_secret: str,
                 access_token: str,
                 access_token_secret: str,
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
        
        # Set up timezone
        self.timezone = pytz.timezone('US/Eastern')
        
        # Initialize monitored accounts
        self.monitored_accounts = {
            'marionawfal': None,
            'shawmakesmagic': None,
            'tonybetw': None
        }
        self._initialize_user_ids()
    
    def _initialize_user_ids(self):
        """Initialize user IDs for monitored accounts"""
        for username in self.monitored_accounts.keys():
            try:
                user = self.client.get_user(username=username)
                if user.data:
                    self.monitored_accounts[username] = user.data.id
            except Exception as e:
                logging.error(f"Error initializing user ID for {username}: {str(e)}")
    
    async def generate_tweet(self, base_prompt: str) -> str:
        """Generate tweet content using LLaMA"""
        current_hour = self.get_est_hour()
        category = self.get_next_category(current_hour)
        category_prompt = random.choice(tweet_prompts[category])
        
        response = await self.generator.generate_text(base_prompt + category_prompt)
        
        if len(response.response) > 280:
            response = await self.generator.generate_text(
                "Shorten this to 280 characters: " + response.response
            )
        
        return response.response
    
    async def post_tweet(self, content: str):
        """Post tweet with error handling"""
        try:
            self.client.create_tweet(text=content)
            logging.info(f"Posted tweet: {content}")
        except Exception as e:
            logging.error(f"Error posting tweet: {str(e)}")
    
    async def auto_post_schedule(self, base_prompt: str):
        """Automatically post tweets on schedule"""
        while True:
            try:
                # Generate and post tweet
                tweet_content = await self.generate_tweet(base_prompt)
                await self.post_tweet(tweet_content)
                
                # Wait random time between 30-60 minutes
                wait_time = random.randint(30, 60) * 60
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logging.error(f"Error in auto posting: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def monitor_topics(self, topics: List[str]):
        """Monitor topics and generate relevant content"""
        while True:
            for topic in topics:
                try:
                    tweets = await self.fetch_topic_tweets(topic)
                    await self.process_tweets(tweets)
                    await asyncio.sleep(random.randint(60, 180))
                except Exception as e:
                    logging.error(f"Error monitoring topic {topic}: {str(e)}")
            
            await asyncio.sleep(900)  # Wait 15 minutes between full cycles