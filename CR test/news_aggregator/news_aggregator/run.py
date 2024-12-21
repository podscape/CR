from news_aggregator import NewsAggregator
import time

def main():
    # Initialize aggregator
    aggregator = NewsAggregator()
    
    # Add some topics to track
    aggregator.track_topic("artificial intelligence")
    aggregator.track_topic("technology")
    
    try:
        print("News aggregator is running...")
        print("Press Ctrl+C to stop")
        
        while True:
            # Get latest articles every 5 minutes
            results = aggregator.retrieve_articles(
                aggregator._generate_simple_embedding("latest news"),
                top_k=5
            )
            
            print("\nLatest articles:")
            for article in results:
                print(f"\nTitle: {article['title']}")
                print(f"Source: {article['source']}")
                print(f"URL: {article['url']}")
            
            time.sleep(300)  # Wait 5 minutes
            
    except KeyboardInterrupt:
        print("\nStopping news aggregator...")
        aggregator.cleanup()

if __name__ == "__main__":
    main()