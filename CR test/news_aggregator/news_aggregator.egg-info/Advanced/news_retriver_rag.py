import json
from datetime import datetime
import numpy as np
from typing import List, Dict, Optional
import tiktoken
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

class LLMDataProcessor:
    """Processes articles for LLM consumption"""
    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.gpt_tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's tokenizer
        
    def prepare_for_llm(self, articles: List[Dict], 
                       max_length: int = 2048,
                       include_metadata: bool = True) -> List[Dict]:
        """
        Prepare articles for LLM training or RAG.
        
        Args:
            articles: List of articles
            max_length: Maximum token length for each chunk
            include_metadata: Whether to include metadata in output
        """
        prepared_data = []
        
        for article in tqdm(articles, desc="Processing articles"):
            # Create base document
            doc = {
                "text": article['content'],
                "metadata": {
                    "title": article['title'],
                    "source": article['source'],
                    "url": article['url'],
                    "timestamp": article['timestamp'].isoformat(),
                    "topic": article.get('topic', ''),
                    "sentiment": article.get('sentiment_score', 0),
                }
            }
            
            # Generate chunks if content is too long
            chunks = self.chunk_text(doc['text'], max_length)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "text": chunk,
                    "chunk_id": i,
                    "embedding": self.generate_embedding(chunk),
                }
                
                if include_metadata:
                    chunk_doc["metadata"] = doc["metadata"]
                
                prepared_data.append(chunk_doc)
        
        return prepared_data
    
    def chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks of maximum token length"""
        tokens = self.gpt_tokenizer.encode(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for token in tokens:
            if current_length + 1 > max_tokens:
                # Convert current chunk to text
                chunk_text = self.gpt_tokenizer.decode(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
            
            current_chunk.append(token)
            current_length += 1
        
        # Add final chunk if any
        if current_chunk:
            chunk_text = self.gpt_tokenizer.decode(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text using the model"""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling for sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].tolist()

    def save_for_training(self, data: List[Dict], output_file: str):
        """Save processed data in JSONL format for training"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    def save_for_vector_db(self, data: List[Dict], output_file: str):
        """Save processed data in format suitable for vector databases"""
        vector_data = []
        for item in data:
            vector_item = {
                "id": f"{item['metadata']['source']}_{item['chunk_id']}",
                "values": item['embedding'],
                "metadata": {
                    "text": item['text'],
                    **item['metadata']
                }
            }
            vector_data.append(vector_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, ensure_ascii=False, indent=2)

class NewsRetrieverLLM(NewsRetriever):
    """Extended NewsRetriever with LLM data processing capabilities"""
    def __init__(self, db_url: str = "sqlite:///articles.db"):
        super().__init__(db_url)
        self.llm_processor = LLMDataProcessor()
    
    def get_training_data(self, topic: str, 
                         max_articles: int = 100,
                         max_length: int = 2048) -> List[Dict]:
        """
        Retrieve and process articles for LLM training/RAG
        """
        # Get articles
        articles = self.get_topic_content(topic)[:max_articles]
        
        # Process for LLM
        processed_data = self.llm_processor.prepare_for_llm(
            articles,
            max_length=max_length
        )
        
        return processed_data
    
    def export_training_data(self, topic: str, 
                           output_file: str = "training_data.jsonl",
                           vector_file: str = "vector_data.json"):
        """Export processed data in formats suitable for different uses"""
        # Get and process data
        processed_data = self.get_training_data(topic)
        
        # Save in different formats
        self.llm_processor.save_for_training(processed_data, output_file)
        self.llm_processor.save_for_vector_db(processed_data, vector_file)
        
        return {
            'training_file': output_file,
            'vector_file': vector_file,
            'num_chunks': len(processed_data)
        }

def main():
    # Initialize retriever with LLM capabilities
    retriever = NewsRetrieverLLM(db_url="sqlite:///articles.db")
    
    # Test with a topic
    topic = "artificial intelligence"
    print(f"\nRetrieving and processing content about: {topic}")
    
    try:
        # Export data in different formats
        result = retriever.export_training_data(
            topic=topic,
            output_file=f"{topic.replace(' ', '_')}_training.jsonl",
            vector_file=f"{topic.replace(' ', '_')}_vectors.json"
        )
        
        print(f"\nProcessed {result['num_chunks']} chunks of text")
        print(f"Training data saved to: {result['training_file']}")
        print(f"Vector data saved to: {result['vector_file']}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    main()