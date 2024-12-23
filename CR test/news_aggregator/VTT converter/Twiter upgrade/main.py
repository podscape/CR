
import asyncio
import os
import random
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
import pytz
import json
import tweepy
from dotenv import load_dotenv
from utils.llama_hf import AsyncTextGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define tweet prompts and categories
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

class TwitterBot:
    """Enhanced Twitter/X bot with automated posting"""
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
        
        # Define content categories and schedules
        self.categories = {
            "morning": ["ai_agents", "engagement"],
            "afternoon": ["memecoins", "humor"],
            "evening": ["crypto", "engagement"],
            "night": ["NFTs", "Solana"]
        }
        
        # Initialize user IDs
        self._initialize_user_ids()
        logger.info("Twitter bot initialized successfully")

    def _initialize_user_ids(self):
        """Initialize user IDs for monitored accounts"""
        try:
            for username in self.monitored_accounts.keys():
                user = self.client.get_user(username=username)
                if user.data:
                    self.monitored_accounts[username] = user.data.id
                    logger.info(f"Initialized user ID for {username}")
        except Exception as e:
            logger.error(f"Error initializing Twitter user IDs: {str(e)}")

    def get_est_hour(self) -> int:
        """Get current hour in EST"""
        utc_now = datetime.now(pytz.utc)
        est_now = utc_now.astimezone(self.timezone)
        return est_now.hour

    def get_next_category(self) -> str:
        """Determine content category based on current time"""
        hour = self.get_est_hour()
        if 6 <= hour < 12:
            return random.choice(self.categories["morning"])
        elif 12 <= hour < 17:
            return random.choice(self.categories["afternoon"])
        elif 17 <= hour < 22:
            return random.choice(self.categories["evening"])
        else:
            return random.choice(self.categories["night"])

    async def generate_tweet(self, base_prompt: str = BASE_PROMPT) -> str:
        """Generate tweet content using LLaMA"""
        try:
            category = self.get_next_category()
            prompt = random.choice(tweet_prompts[category])
            
            logger.info(f"Generating tweet for category: {category}")
            logger.info(f"Using prompt: {prompt}")
            
            full_prompt = f"{base_prompt}{prompt}"
            response = await self.generator.generate_text(full_prompt)
            
            if len(response.response) > 280:
                response = await self.generator.generate_text(
                    f"Shorten this to 280 characters while maintaining style: {response.response}"
                )
            
            logger.info(f"Generated tweet: {response.response}")
            return response.response
            
        except Exception as e:
            logger.error(f"Error generating tweet: {str(e)}")
            return None

    async def post_tweet(self, content: str) -> bool:
        """Post tweet with error handling"""
        try:
            if not content:
                return False
                
            self.client.create_tweet(text=content)
            logger.info(f"Successfully posted tweet: {content}")
            return True
            
        except Exception as e:
            logger.error(f"Error posting tweet: {str(e)}")
            return False

    async def auto_post_schedule(self, interval_minutes: Tuple[int, int] = (30, 60)):
        """Automatically post tweets on schedule"""
        while True:
            try:
                # Generate and post tweet
                tweet_content = await self.generate_tweet()
                if await self.post_tweet(tweet_content):
                    # Random wait between min and max interval
                    wait_time = random.randint(interval_minutes[0], interval_minutes[1]) * 60
                    logger.info(f"Waiting {wait_time/60:.1f} minutes until next tweet")
                    await asyncio.sleep(wait_time)
                else:
                    # Short wait on failure
                    logger.warning("Tweet failed, waiting 5 minutes")
                    await asyncio.sleep(300)
                    
            except Exception as e:
                logger.error(f"Error in auto posting schedule: {str(e)}")
                await asyncio.sleep(300)

    async def monitor_topics(self, topics: List[str]):
        """Monitor topics and generate relevant content"""
        while True:
            for topic in topics:
                try:
                    logger.info(f"Generating content for topic: {topic}")
                    tweet_content = await self.generate_tweet(
                        f"Create an informative tweet about {topic}: "
                    )
                    
                    if await self.post_tweet(tweet_content):
                        # Random wait between topics
                        wait_time = random.randint(2, 5) * 60
                        await asyncio.sleep(wait_time)
                        
                except Exception as e:
                    logger.error(f"Error monitoring topic {topic}: {str(e)}")
            
            # Wait before next cycle
            await asyncio.sleep(1800)

async def main():
    try:
        # Create Twitter bot
        bot = TwitterBot()
        
        # Define topics to monitor
        topics = [
            "artificial_intelligence",
            "cryptocurrency",
            "financial_markets",
            "geopolitics"
        ]
        
        # Create tasks
        tasks = [
            asyncio.create_task(bot.auto_post_schedule()),
            asyncio.create_task(bot.monitor_topics(topics))
        ]
        
        # Run tasks
        logger.info("Starting bot tasks...")
        await asyncio.gather(*tasks)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        logger.info("Shutting down bot...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping bot tasks...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")