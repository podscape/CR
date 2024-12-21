from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine, Column, String, DateTime, Float, JSON, Text

# Create the base class
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
    article_metadata = Column(JSON)  # Changed from 'metadata' to 'article_metadata'

def initialize_database(db_url: str = "sqlite:///articles.db"):
    """Initialize the database and create tables"""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine

