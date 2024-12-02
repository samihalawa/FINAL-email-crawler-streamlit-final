import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import json
from datetime import datetime
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from googlesearch import search as google_search

from core.config import settings
from core.database import get_db, SearchProcess, SearchTerm, Lead

logger = logging.getLogger(__name__)

class SearchService:
    """Optimized search service based on reference implementation"""
    
    def __init__(self):
        self.session = None
        self.user_agent = UserAgent()
        self.search_config = {
            'max_results_per_term': 100,
            'request_timeout': 30,
            'max_retries': 3,
            'batch_size': 10,
            'delay_between_requests': 2
        }
    
    async def initialize(self):
        """Initialize aiohttp session with retry logic"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.search_config['request_timeout'])
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_request(self, url: str) -> Optional[str]:
        """Make HTTP request with retry logic"""
        headers = {'User-Agent': self.user_agent.random}
        try:
            async with self.session.get(url, headers=headers, ssl=False) as response:
                if response.status == 200:
                    return await response.text()
                logger.warning(f"Request failed with status {response.status}: {url}")
                return None
        except Exception as e:
            logger.error(f"Request error for {url}: {str(e)}")
            raise
    
    async def process_search_terms(self, search_process_id: int) -> Dict[str, Any]:
        """Process search terms in batches"""
        async with get_db() as db:
            search_process = await db.get(SearchProcess, search_process_id)
            if not search_process:
                raise ValueError(f"Search process {search_process_id} not found")
            
            search_terms = search_process.search_terms
            total_results = []
            
            # Process in batches
            for i in range(0, len(search_terms), self.search_config['batch_size']):
                batch = search_terms[i:i + self.search_config['batch_size']]
                batch_tasks = [self.search_term(term) for term in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for term, results in zip(batch, batch_results):
                    if isinstance(results, Exception):
                        logger.error(f"Search failed for term '{term}': {str(results)}")
                        continue
                    total_results.extend(results)
                
                # Update process status
                search_process.total_leads_found = len(total_results)
                search_process.updated_at = datetime.utcnow()
                await db.commit()
                
                # Respect rate limits
                await asyncio.sleep(self.search_config['delay_between_requests'])
            
            return {
                'total_results': len(total_results),
                'results': total_results,
                'failed_terms': [t for t, r in zip(search_terms, batch_results) if isinstance(r, Exception)]
            }
    
    async def search_term(self, term: str) -> List[Dict[str, Any]]:
        """Search single term across multiple sources"""
        await self.initialize()
        results = []
        
        try:
            # Google search
            search_results = list(google_search(
                term, 
                num_results=self.search_config['max_results_per_term'],
                lang=settings.DEFAULT_SEARCH_SETTINGS['language']
            ))
            
            # Process URLs in parallel
            tasks = [self.process_url(url) for url in search_results]
            url_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter and clean results
            for result in url_results:
                if isinstance(result, Exception) or not result:
                    continue
                results.extend(result)
            
            # Remove duplicates
            seen_emails = set()
            unique_results = []
            for r in results:
                if r.get('email') and r['email'] not in seen_emails:
                    seen_emails.add(r['email'])
                    unique_results.append(r)
            
            return unique_results
            
        except Exception as e:
            logger.error(f"Search failed for term '{term}': {str(e)}")
            raise
    
    async def process_url(self, url: str) -> List[Dict[str, Any]]:
        """Process single URL to extract leads"""
        try:
            html_content = await self._make_request(url)
            if not html_content:
                return []
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract emails
            emails = self.extract_emails(html_content)
            if not emails:
                return []
            
            # Extract metadata
            title = soup.title.string if soup.title else ''
            description = soup.find('meta', {'name': 'description'})
            description = description['content'] if description else ''
            
            results = []
            for email in emails:
                lead = {
                    'email': email,
                    'source_url': url,
                    'title': title,
                    'description': description,
                    'found_at': datetime.utcnow().isoformat()
                }
                results.append(lead)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {str(e)}")
            return []
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract valid email addresses from text"""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        
        # Basic validation
        valid_emails = []
        for email in emails:
            if (
                len(email) <= 254 and  # RFC 5321
                re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) and
                not any(c.isspace() for c in email)
            ):
                valid_emails.append(email.lower())
        
        return list(set(valid_emails))