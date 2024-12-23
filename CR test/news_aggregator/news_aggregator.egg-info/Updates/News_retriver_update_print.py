
import requests
from bs4 import BeautifulSoup
import tweepy
import feedparser
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging
from urllib.parse import quote_plus
from fake_useragent import UserAgent
from sqlalchemy import create_engine, Column, String, DateTime, Float, JSON, Text, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
import pandas as pd

def create_fresh_database(db_name: str = "news_analysis.db"):
    """Create a fresh database with the complete schema"""
    db_url = f"sqlite:///{db_name}"
    engine = create_engine(db_url)
    
    # Drop all existing tables if they exist
    Base.metadata.drop_all(engine)
    
    # Create all tables with new schema
    Base.metadata.create_all(engine)
    
    return db_url

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
    cluster_id = Column(Integer, nullable=True)
    embedding = Column(JSON, nullable=True)

class ContentClusterer:
    """Handles content clustering and similarity analysis"""
    def __init__(self, embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(max_features=embedding_dim)
        self.clusterer = DBSCAN(eps=0.3, min_samples=2)
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF embeddings for texts"""
        return self.vectorizer.fit_transform(texts).toarray()
    
    def cluster_content(self, articles: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        """Cluster articles based on content similarity"""
        # Extract text content
        texts = [f"{article['title']} {article.get('content', '')}" for article in articles]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Perform clustering
        clusters = self.clusterer.fit_predict(embeddings)
        
        # Add cluster information to articles
        for i, article in enumerate(articles):
            article['cluster_id'] = int(clusters[i])
            article['embedding'] = embeddings[i].tolist()
        
        return articles, embeddings

    def find_similar_articles(self, target_article: Dict, all_articles: List[Dict], 
                            threshold: float = 0.7) -> List[Dict]:
        """Find articles similar to target article"""
        if 'embedding' not in target_article:
            return []
            
        target_embedding = np.array(target_article['embedding'])
        similar_articles = []
        
        for article in all_articles:
            if 'embedding' in article and article['url'] != target_article['url']:
                similarity = 1 - cosine(target_embedding, np.array(article['embedding']))
                if similarity > threshold:
                    article['similarity_score'] = similarity
                    similar_articles.append(article)
        
        return sorted(similar_articles, key=lambda x: x['similarity_score'], reverse=True)

class TopicVisualizer:
    """Handles visualization of topic relationships and content clusters"""
    def __init__(self):
        self.graph = nx.Graph()
        
    def create_topic_network(self, topic_manager: 'TopicManager') -> None:
        """Create network visualization of topic relationships"""
        # Add nodes for each topic
        for topic, info in topic_manager.topics.items():
            self.graph.add_node(topic, size=len(info['keywords']))
            
            # Add edges for related topics
            for related in info['related_topics']:
                self.graph.add_edge(topic, related, weight=1)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color='lightblue',
                             node_size=[v * 500 for v in nx.degree_centrality(self.graph).values()])
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos)
        
        # Add labels
        nx.draw_networkx_labels(self.graph, pos)
        
        plt.title("Topic Relationship Network")
        plt.axis('off')
        
    def visualize_content_clusters(self, articles: List[Dict], embeddings: np.ndarray) -> go.Figure:
        """Create visualization of content clusters"""
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(embeddings)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'cluster': [article['cluster_id'] for article in articles],
            'title': [article.get('title', 'No title') for article in articles]
        })
        
        # Create interactive scatter plot
        fig = go.Figure()
        
        for cluster_id in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster_id]
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=cluster_data['title'],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title="Content Clusters Visualization",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            showlegend=True
        )
        
        return fig
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

class TwitterIntegration:
    """Handles Twitter/X content retrieval"""
    def __init__(self, bearer_token: str):
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.monitored_accounts = {
            'marionawfal': None,  # Will be filled with user IDs
            'shawmakesmagic': None,
            'tonybetw': None
        }
        self._initialize_user_ids()
    
    def _initialize_user_ids(self):
        """Initialize user IDs for monitored accounts"""
        try:
            for username in self.monitored_accounts.keys():
                user = self.client.get_user(username=username)
                if user.data:
                    self.monitored_accounts[username] = user.data.id
        except Exception as e:
            logging.error(f"Error initializing Twitter user IDs: {str(e)}")
    
    def get_user_tweets(self, username: str, max_results: int = 10) -> List[Dict]:
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
                    'title': f"Tweet by {username}",  # For consistency with article format
                }
                results.append(tweet_data)
            
            return results
            
        except Exception as e:
            logging.error(f"Error fetching tweets from {username}: {str(e)}")
            return []

class TopicManager:
    """Manages multiple topics and their relationships"""
    def __init__(self):
        self.topics = {
            'artificial_intelligence': {
                'keywords': [
                    'AI', 'machine learning', 'deep learning', 'neural networks',
                    'artificial intelligence', 'LLM', 'GPT', 'AI models'
                ],
                'related_topics': ['technology', 'automation', 'cryptocurrency']
            },
            'cryptocurrency': {
                'keywords': [
                    'crypto', 'bitcoin', 'ethereum', 'blockchain', 'web3',
                    'defi', 'nft', 'cryptocurrency', 'tokens', 'mining'
                ],
                'related_topics': ['financial_markets', 'technology']
            },
            'financial_markets': {
                'keywords': [
                    'stocks', 'trading', 'market analysis', 'investment',
                    'forex', 'commodities', 'bonds', 'market trends'
                ],
                'related_topics': ['cryptocurrency', 'economics', 'geopolitics']
            },
            'geopolitics': {
                'keywords': [
                    'international relations', 'political risk', 'global policy',
                    'sanctions', 'trade war', 'diplomacy', 'foreign policy'
                ],
                'related_topics': ['cryptocurrency', 'financial_markets']
            },
            'technology': {
                'keywords': [
                    'tech innovation', 'startups', 'digital transformation',
                    'cloud computing', 'cybersecurity', 'fintech', 'biotech'
                ],
                'related_topics': ['artificial_intelligence', 'cryptocurrency']
            }
        }
        
        # Initialize topic relationships graph
        self.topic_graph = nx.Graph()
        self._build_topic_graph()
    
    def _build_topic_graph(self):
        """Build network graph of topic relationships"""
        for topic, info in self.topics.items():
            self.topic_graph.add_node(topic, keywords=info['keywords'])
            for related in info['related_topics']:
                self.topic_graph.add_edge(topic, related)
    
    def get_related_topics(self, topic: str) -> List[str]:
        """Get related topics for content expansion"""
        if topic in self.topic_graph:
            return list(self.topic_graph.neighbors(topic))
        return []
    
    def get_topic_keywords(self, topic: str) -> List[str]:
        """Get keywords for a topic"""
        return self.topics.get(topic, {}).get('keywords', [])
    
    def calculate_topic_relevance(self, text: str, topic: str) -> float:
        """Calculate relevance score of text to topic"""
        keywords = self.get_topic_keywords(topic)
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return keyword_count / len(keywords)

class NewsRetriever:
    """Base class for retrieving and analyzing news content"""
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
            },
            'benzinga': {
                'name': 'Benzinga',
                'rss_url': 'https://www.benzinga.com/feed',
                'selectors': {
                    'title': 'h1.article-title',
                    'content': 'div.article-content-body',
                    'author': 'span.author-name'
                }
            },
            'cryptoslate': {
                'name': 'CryptoSlate',
                'rss_url': 'https://cryptoslate.com/feed/',
                'selectors': {
                    'title': 'h1.entry-title',
                    'content': 'div.entry-content',
                    'author': 'span.author'
                }
            },
            'cointelegraph': {
                'name': 'CoinTelegraph',
                'rss_url': 'https://cointelegraph.com/rss',
                'selectors': {
                    'title': 'h1.post__title',
                    'content': 'div.post__content',
                    'author': 'div.post__author'
                }
            },
            'ai_news': {
                'name': 'AI News',
                'rss_url': 'https://www.artificialintelligence-news.com/feed/',
                'selectors': {
                    'title': 'h1.entry-title',
                    'content': 'div.entry-content',
                    'author': 'span.author'
                }
            },
            'techcrunch_ai': {
                'name': 'TechCrunch AI',
                'rss_url': 'https://techcrunch.com/category/artificial-intelligence/feed/',
                'selectors': {
                    'title': 'h1.article__title',
                    'content': 'div.article-content',
                    'author': 'span.article__author'
                }
            },
            'decrypt': {
                'name': 'Decrypt',
                'rss_url': 'https://decrypt.co/feed',
                'selectors': {
                    'title': 'h1.entry-title',
                    'content': 'div.entry-content',
                    'author': 'span.author'
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

class EnhancedNewsRetriever(NewsRetriever):
    """Enhanced version with multi-source and multi-topic support"""
    def __init__(self, db_url: str = "sqlite:///articles.db", twitter_bearer_token: Optional[str] = None):
        super().__init__(db_url)
        self.topic_manager = TopicManager()
        self.twitter = TwitterIntegration(twitter_bearer_token) if twitter_bearer_token else None
        self.clusterer = ContentClusterer()
    
    def get_comprehensive_content(self, 
                                primary_topic: str,
                                include_related: bool = True,
                                include_twitter: bool = True) -> List[Dict]:
        """Get comprehensive content including related topics and social media"""
        all_content = []
        
        # Get content for primary topic
        print(f"\nFetching content for primary topic: {primary_topic}")
        primary_content = self.get_topic_content(primary_topic)
        all_content.extend(primary_content)
        
        # Get content from related topics
        if include_related:
            related_topics = self.topic_manager.get_related_topics(primary_topic)
            for related_topic in related_topics:
                print(f"Fetching content for related topic: {related_topic}")
                related_content = self.get_topic_content(related_topic)
                all_content.extend(related_content)
        
        # Get Twitter content
        if include_twitter and self.twitter:
            print("Fetching Twitter content...")
            for username in self.twitter.monitored_accounts:
                tweets = self.twitter.get_user_tweets(username)
                all_content.extend(tweets)
        
        # Sort by timestamp
        all_content.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return all_content

def print_article_details(article: Dict, show_similar: bool = False, similar_articles: List[Dict] = None):
    """Print detailed information about an article"""
    print("\n" + "="*80)
    print(f"TITLE: {article.get('title', 'No title')}")
    print("="*80)
    print(f"SOURCE: {article['source']}")
    print(f"URL: {article['url']}")
    print(f"TIMESTAMP: {article['timestamp']}")
    
    if 'cluster_id' in article:
        print(f"CLUSTER: {article['cluster_id']}")
    
    if 'sentiment_score' in article:
        sentiment = article['sentiment_score']
        sentiment_label = "POSITIVE" if sentiment > 0 else "NEGATIVE" if sentiment < 0 else "NEUTRAL"
        print(f"SENTIMENT: {sentiment_label} ({sentiment:.2f})")
    
    if 'keywords' in article and article['keywords']:
        print("\nKEY TERMS:")
        for keyword, score in list(article['keywords'].items())[:5]:
            print(f"- {keyword}: {score:.3f}")
    
    if 'summary' in article and article['summary']:
        print("\nSUMMARY:")
        print(article['summary'][:300] + "...")
    
    if show_similar and similar_articles:
        print("\nSIMILAR ARTICLES:")
        for sim_article in similar_articles[:3]:
            print(f"- {sim_article['title']} (similarity: {sim_article['similarity_score']:.2f})")
    
    print("-"*80)

def print_cluster_summary(articles: List[Dict]):
    """Print summary of content clusters"""
    clusters = {}
    for article in articles:
        cluster_id = article.get('cluster_id')
        if cluster_id is not None:
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(article)
    
    print("\n" + "="*80)
    print("CLUSTER SUMMARY")
    print("="*80)
    
    for cluster_id, cluster_articles in clusters.items():
        if cluster_id == -1:
            continue  # Skip noise cluster
            
        print(f"\nCluster {cluster_id} ({len(cluster_articles)} articles):")
        # Get common keywords
        all_keywords = Counter()
        for article in cluster_articles:
            if 'keywords' in article:
                all_keywords.update(article['keywords'].keys())
        
        print("Common themes:", ", ".join([k for k, v in all_keywords.most_common(5)]))
        print("Articles:")
        for article in cluster_articles[:3]:  # Show top 3 articles per cluster
            print(f"- {article['title']}")
    print("-"*80)

def main():
    # Initialize retriever with Twitter credentials
    db_url = create_fresh_database("news_analysis_v2.db") 
    retriever = EnhancedNewsRetriever(
        db_url=db_url,
        twitter_bearer_token="YOUR_TWITTER_BEARER_TOKEN"
    )
    
    # Create visualizer
    visualizer = TopicVisualizer()
    
    # Create and save topic network visualization
    print("Creating topic network visualization...")
    visualizer.create_topic_network(retriever.topic_manager)
    plt.savefig('topic_network.png')
    print("Topic network saved as 'topic_network.png'")
    
    # Define topics to track
    topics = [
        "artificial_intelligence",
        "cryptocurrency",
        "financial_markets",
        "geopolitics"
    ]
    
    print("\nStarting content retrieval and analysis...")
    all_articles = []
    
    # Process each topic
    for topic in topics:
        print(f"\nProcessing topic: {topic}")
        results = retriever.get_comprehensive_content(
            primary_topic=topic,
            include_related=True,
            include_twitter=True
        )
        print(f"Found {len(results)} items for {topic}")
        all_articles.extend(results)
    
    # Perform clustering
    print("\nPerforming content clustering...")
    clustered_articles, embeddings = retriever.clusterer.cluster_content(all_articles)
    
    # Create and save cluster visualization
    print("Creating cluster visualization...")
    cluster_viz = visualizer.visualize_content_clusters(clustered_articles, embeddings)
    cluster_viz.write_html('content_clusters.html')
    print("Cluster visualization saved as 'content_clusters.html'")
    
    # Print cluster summary
    print_cluster_summary(clustered_articles)
    
    # Print detailed results for top articles
    print("\nMOST RECENT ARTICLES BY TOPIC:")
    for topic in topics:
        topic_articles = [a for a in clustered_articles if a.get('topic') == topic]
        if topic_articles:
            print(f"\nTOP ARTICLES FOR {topic.upper()}:")
            for article in topic_articles[:3]:  # Show top 3 per topic
                similar_articles = retriever.clusterer.find_similar_articles(article, clustered_articles)
                print_article_details(article, show_similar=True, similar_articles=similar_articles)
    
    # Print statistics
    print("\n" + "="*80)
    print("CONTENT ANALYSIS STATISTICS")
    print("="*80)
    print(f"Total articles processed: {len(clustered_articles)}")
    print(f"Number of clusters: {len(set(a['cluster_id'] for a in clustered_articles if a.get('cluster_id', -1) != -1))}")
    
    # Sentiment distribution
    sentiments = [a.get('sentiment_score', 0) for a in clustered_articles]
    print(f"Average sentiment: {np.mean(sentiments):.2f}")
    print(f"Sentiment distribution:")
    print(f"- Positive: {sum(1 for s in sentiments if s > 0)} articles")
    print(f"- Neutral: {sum(1 for s in sentiments if s == 0)} articles")
    print(f"- Negative: {sum(1 for s in sentiments if s < 0)} articles")

if __name__ == "__main__":
    main()
