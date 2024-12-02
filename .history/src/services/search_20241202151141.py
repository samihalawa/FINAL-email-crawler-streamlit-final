import asyncio
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime
import random
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.ua = UserAgent()
        self.session = None
        self.domains_processed = set()
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def optimize_search_term(self, term: str, language: str) -> str:
        """Optimize search term based on language"""
        if language == 'EN':
            return f'"{term}" email OR contact OR "get in touch" site:.com'
        elif language == 'ES':
            return f'"{term}" correo OR contacto OR "ponte en contacto" site:.es'
        return term
        
    def shuffle_keywords(self, term: str) -> str:
        """Randomly shuffle keywords in search term"""
        words = term.split()
        random.shuffle(words)
        return ' '.join(words)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_google(self, term: str, num_results: int = 10) -> List[str]:
        """Perform Google search and return URLs"""
        headers = {'User-Agent': self.ua.random}
        search_url = f"https://www.google.com/search?q={term}&num={num_results}"
        
        async with self.session.get(search_url, headers=headers) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                search_results = []
                for result in soup.select('.tF2Cxc'):
                    link = result.select_one('.yuRUbf a')
                    if link and 'href' in link.attrs:
                        search_results.append(link['href'])
                return search_results
            else:
                logger.error(f"Search failed with status {response.status}")
                return []
                
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def extract_contact_info(self, url: str) -> Dict[str, Any]:
        """Extract contact information from webpage"""
        headers = {'User-Agent': self.ua.random}
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract emails
                    emails = set()
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    text_content = soup.get_text()
                    found_emails = re.findall(email_pattern, text_content)
                    emails.update(found_emails)
                    
                    # Extract other contact info
                    name = soup.find('meta', {'name': 'author'})
                    name = name['content'] if name else ''
                    
                    company = soup.find('meta', {'property': 'og:site_name'})
                    company = company['content'] if company else ''
                    
                    return {
                        'emails': list(emails),
                        'name': name,
                        'company': company,
                        'url': url
                    }
                return None
        except Exception as e:
            logger.error(f"Error extracting contact info from {url}: {str(e)}")
            return None
            
    async def perform_search(
        self,
        search_terms: List[str],
        num_results: int = 10,
        language: str = 'EN',
        optimize_english: bool = False,
        optimize_spanish: bool = False,
        shuffle_keywords: bool = False
    ) -> List[Dict[str, Any]]:
        """Main search function that coordinates the entire search process"""
        all_results = []
        self.domains_processed.clear()
        
        async with self:  # Create aiohttp session
            for term in search_terms:
                # Optimize term if requested
                if optimize_english and language == 'EN':
                    term = self.optimize_search_term(term, 'EN')
                elif optimize_spanish and language == 'ES':
                    term = self.optimize_search_term(term, 'ES')
                
                # Shuffle keywords if requested
                if shuffle_keywords:
                    term = self.shuffle_keywords(term)
                
                # Perform search
                urls = await self.search_google(term, num_results)
                
                # Process URLs in parallel
                tasks = []
                for url in urls:
                    domain = urlparse(url).netloc
                    if domain not in self.domains_processed:
                        self.domains_processed.add(domain)
                        tasks.append(self.extract_contact_info(url))
                
                # Gather results
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, dict) and result.get('emails'):
                            for email in result['emails']:
                                all_results.append({
                                    'email': email,
                                    'name': result['name'],
                                    'company': result['company'],
                                    'url': result['url'],
                                    'found_at': datetime.utcnow().isoformat()
                                })
        
        return all_results 