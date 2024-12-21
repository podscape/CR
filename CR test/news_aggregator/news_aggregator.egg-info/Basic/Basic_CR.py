from retriever import MultiSourceRetriever
import config

def main():
    # Initialize with Twitter credentials if available
    retriever = MultiSourceRetriever(
        twitter_bearer_token=config.TWITTER_BEARER_TOKEN
    )
    
    # Define topics to track
    topics = [
        "artificial intelligence",
        "climate change",
        "space exploration"
    ]
    
    # Retrieve content for each topic
    for topic in topics:
        print(f"\nContent for topic: {topic}")
        results = retriever.retrieve_topic_content(topic)
        
        for article in results:
            print(f"\nTitle: {article.get('title', 'No title')}")
            print(f"Source: {article['source']}")
            print(f"URL: {article['url']}")
            print(f"Timestamp: {article['timestamp']}")
            print("-" * 50)

if __name__ == "__main__":
    main()