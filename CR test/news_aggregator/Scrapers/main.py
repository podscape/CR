# main.py
import asyncio
import logging
from datetime import datetime
from scrapers.ai_agent_scraper import AIAgentScraper
from scrapers.content_aggregator import ContentAggregator

async def main():
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize content aggregator
        aggregator = ContentAggregator()
        
        # Fetch and process content
        logging.info("Starting content aggregation...")
        content_df = await aggregator.aggregate_content()
        logging.info(f"Aggregated {len(content_df)} content items")
        
        # Scrape AI agent info
        agent_url = "https://docs.dolosdiary.com"
        agent_name = "dolos"
        
        logging.info(f"Starting AI agent scraping for {agent_name}...")
        scraper = AIAgentScraper(agent_url, agent_name)
        docs = scraper.crawl().load()
        scraper.save_data()
        logging.info(f"Scraped {len(docs)} pages for {agent_name}")
        
        # Get recent AI-related content
        ai_content = aggregator.get_latest_content('ai_agents', days=1)
        logging.info(f"Found {len(ai_content)} AI-related items in the last 24 hours")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())