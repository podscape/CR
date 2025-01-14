
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
from datetime import datetime
import logging
from typing import Dict, List, Optional
import csv

class AIAgentScraper:
    """Scraper for AI agent documentation and specifications"""
    def __init__(self, base_url: str, agent_name: str):
        self.base_url = base_url
        self.agent_name = agent_name
        self.found_urls = set()
        self.docs = []
        self.structured_data = {
            'name': agent_name,
            'features': [],
            'capabilities': [],
            'use_cases': [],
            'updates': [],
            'integrations': [],
            'metrics': {}
        }
        
        # Initialize directories
        self.data_dir = os.path.join('data', 'agents')
        self.docs_dir = os.path.join(self.data_dir, 'documents')
        self.structured_dir = os.path.join(self.data_dir, 'structured')
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.structured_dir, exist_ok=True)

    def get_links(self, url: str) -> List[str]:
        """Get all valid links from a page"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            return [
                urljoin(self.base_url, link.get('href'))
                for link in soup.find_all('a')
                if link.get('href') and self.base_url in urljoin(self.base_url, link.get('href'))
            ]
        except Exception as e:
            logging.error(f"Error fetching links from {url}: {str(e)}")
            return []

    def extract_structured_data(self, soup: BeautifulSoup) -> Dict:
        """Extract structured data from page content"""
        data = {}
        
        # Extract features
        features = soup.find_all(['h2', 'h3'], string=lambda s: 'feature' in s.lower() if s else False)
        data['features'] = [
            feature.find_next('p').text.strip()
            for feature in features
            if feature.find_next('p')
        ]
        
        # Extract capabilities
        capabilities = soup.find_all(['h2', 'h3'], string=lambda s: 'capability' in s.lower() if s else False)
        data['capabilities'] = [
            cap.find_next('p').text.strip()
            for cap in capabilities
            if cap.find_next('p')
        ]
        
        # Extract use cases
        use_cases = soup.find_all(['h2', 'h3'], string=lambda s: 'use case' in s.lower() if s else False)
        data['use_cases'] = [
            case.find_next('p').text.strip()
            for case in use_cases
            if case.find_next('p')
        ]
        
        # Extract integrations
        integrations = soup.find_all(['h2', 'h3'], string=lambda s: 'integration' in s.lower() if s else False)
        data['integrations'] = [
            integration.find_next('p').text.strip()
            for integration in integrations
            if integration.find_next('p')
        ]
        
        return data

    def crawl(self):
        """Crawl the website and collect data"""
        urls = set([self.base_url])
        while urls:
            url = urls.pop()
            if url not in self.found_urls:
                try:
                    logging.info(f"Processing: {url}")
                    self.found_urls.add(url)
                    
                    response = requests.get(url)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract data
                    page_data = self.extract_structured_data(soup)
                    for key in self.structured_data.keys():
                        if key in page_data:
                            self.structured_data[key].extend(page_data[key])
                    
                    # Get new URLs
                    new_urls = self.get_links(url)
                    urls.update(set(new_urls) - self.found_urls)
                    
                except Exception as e:
                    logging.error(f"Error processing {url}: {str(e)}")
        
        return self

    def load(self):
        """Load document content"""
        try:
            loader = WebBaseLoader(list(self.found_urls))
            self.docs = loader.load()
            return self.docs
        except Exception as e:
            logging.error(f"Error loading documents: {str(e)}")
            return []

    def save_data(self):
        """Save both document content and structured data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save documents
        doc_file = os.path.join(self.docs_dir, f"{self.agent_name}_{timestamp}.txt")
        try:
            with open(doc_file, 'w', encoding='utf-8') as f:
                for doc in self.docs:
                    f.write(f"URL: {doc.metadata.get('source', 'Unknown')}\n")
                    f.write("-" * 80 + "\n")
                    f.write(doc.page_content + "\n\n")
        except Exception as e:
            logging.error(f"Error saving documents: {str(e)}")

        # Save structured data
        csv_file = os.path.join(self.structured_dir, f"{self.agent_name}_{timestamp}.csv")
        try:
            rows = []
            
            # Add each type of data
            for data_type, items in self.structured_data.items():
                if isinstance(items, list):
                    for item in items:
                        rows.append([data_type, item, '', datetime.now()])
                elif isinstance(items, dict):
                    for key, value in items.items():
                        rows.append([data_type, key, value, datetime.now()])
            
            # Write to CSV
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['type', 'content', 'value', 'date'])
                writer.writerows(rows)
            
        except Exception as e:
            logging.error(f"Error saving structured data: {str(e)}")