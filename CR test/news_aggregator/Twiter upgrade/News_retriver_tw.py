
import asyncio
import os
import random
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
import json

import asyncio
import requests
from bs4 import BeautifulSoup
import tweepy
import feedparser
import numpy as np
from urllib.parse import quote_plus
from fake_useragent import UserAgent
import pytz
from dotenv import load_dotenv
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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the SQLAlchemy base
Base = declarative_base()

def create_fresh_database(db_name: str = "news_analysis.db"):
    """Create a fresh database with the complete schema"""
    db_url = f"sqlite:///{db_name}"
    engine = create_engine(db_url)
    
    # Drop all existing tables if they exist
    Base.metadata.drop_all(engine)
    
    # Create all tables with new schema
    Base.metadata.create_all(engine)
    
    return db_url

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
        try:
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
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
            keywords = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10])
            
            return {
                'sentiment_score': sentiment.polarity,
                'sentiment_subjectivity': sentiment.subjectivity,
                'summary': summary,
                'keywords': keywords
            }
        
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return {
                'sentiment_score': 0.0,
                'sentiment_subjectivity': 0.0,
                'summary': '',
                'keywords': {}
            }

class ContentClusterer:
    """Handles content clustering and similarity analysis"""
    def __init__(self, embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(max_features=embedding_dim)
        self.clusterer = DBSCAN(eps=0.3, min_samples=2)
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF embeddings for texts"""
        try:
            return self.vectorizer.fit_transform(texts).toarray()
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return np.zeros((len(texts), self.embedding_dim))
    
    def cluster_content(self, articles: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        """Cluster articles based on content similarity"""
        try:
            # Extract text content
            texts = [f"{article.get('title', '')} {article.get('content', '')}" for article in articles]
            
            # Create embeddings
            embeddings = self.create_embeddings(texts)
            
            # Perform clustering
            clusters = self.clusterer.fit_predict(embeddings)
            
            # Add cluster information to articles
            for i, article in enumerate(articles):
                article['cluster_id'] = int(clusters[i])
                article['embedding'] = embeddings[i].tolist()
            
            return articles, embeddings
            
        except Exception as e:
            logger.error(f"Error clustering content: {str(e)}")
            # Return original articles without clustering
            return articles, np.zeros((len(articles), self.embedding_dim))

    def find_similar_articles(self, target_article: Dict, all_articles: List[Dict], 
                            threshold: float = 0.7) -> List[Dict]:
        """Find articles similar to target article"""
        try:
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
            
        except Exception as e:
            logger.error(f"Error finding similar articles: {str(e)}")
            return []

class TopicVisualizer:
    """Handles visualization of topic relationships and content clusters"""
    def __init__(self):
        self.graph = nx.Graph()
        
    def create_topic_network(self, topic_manager: 'TopicManager') -> None:
        """Create network visualization of topic relationships"""
        try:
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
            
        except Exception as e:
            logger.error(f"Error creating topic network visualization: {str(e)}")
    
    def visualize_content_clusters(self, articles: List[Dict], embeddings: np.ndarray) -> Optional[go.Figure]:
        """Create visualization of content clusters"""
        try:
            # Reduce dimensionality for visualization
            tsne = TSNE(n_components=2, random_state=42)
            coords = tsne.fit_transform(embeddings)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'x': coords[:, 0],
                'y': coords[:, 1],
                'cluster': [article.get('cluster_id', -1) for article in articles],
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
            
        except Exception as e:
            logger.error(f"Error creating cluster visualization: {str(e)}")
            return None

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
        try:
            for topic, info in self.topics.items():
                self.topic_graph.add_node(topic, keywords=info['keywords'])
                for related in info['related_topics']:
                    self.topic_graph.add_edge(topic, related)
        except Exception as e:
            logger.error(f"Error building topic graph: {str(e)}")
    
    def get_related_topics(self, topic: str) -> List[str]:
        """Get related topics for content expansion"""
        try:
            if topic in self.topic_graph:
                return list(self.topic_graph.neighbors(topic))
            return []
        except Exception as e:
            logger.error(f"Error getting related topics: {str(e)}")
            return []
    
    def get_topic_keywords(self, topic: str) -> List[str]:
        """Get keywords for a topic"""
        return self.topics.get(topic, {}).get('keywords', [])
    
    def calculate_topic_relevance(self, text: str, topic: str) -> float:
        """Calculate relevance score of text to topic"""
        try:
            keywords = self.get_topic_keywords(topic)
            if not keywords:
                return 0.0
            
            text_lower = text.lower()
            keyword_count = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            return keyword_count / len(keywords)
        except Exception as e:
            logger.error(f"Error calculating topic relevance: {str(e)}")
            return 0.0
