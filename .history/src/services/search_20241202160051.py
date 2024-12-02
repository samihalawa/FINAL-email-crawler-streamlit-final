from typing import List, Dict, Optional
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging
import re
from datetime import datetime

from core.database import get_db, Lead, SearchTerm, SearchProcess
from services.ai import AIService
from utils.validation import validate_email

ai_service = AIService()

class SearchService:
    def __init__(self):
        self.session = aiohttp.ClientSession()
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def extract_emails_from_url(self, url: str) -> List[str]:
        """Extract emails from a given URL"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    return []
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                
                # Find all email addresses
                emails = set(self.email_pattern.findall(text))
                
                # Validate each email
                valid_emails = [email for email in emails if validate_email(email)]
                
                return valid_emails
                
        except Exception as e:
            logging.error(f"Error extracting emails from {url}: {str(e)}")
            return []

    async def process_url(self, url: str, search_term_id: int, excluded_domains: List[str]) -> List[Dict]:
        """Process a single URL to extract leads"""
        domain = urlparse(url).netloc
        
        if any(excluded in domain for excluded in excluded_domains):
            return []
            
        emails = await self.extract_emails_from_url(url)
        results = []
        
        async with get_db() as session:
            for email in emails:
                # Check if lead already exists
                existing = await session.execute(
                    "SELECT id FROM leads WHERE email = :email",
                    {'email': email}
                )
                if not existing.scalar():
                    lead = Lead(
                        email=email,
                        source_url=url,
                        search_term_id=search_term_id,
                        created_at=datetime.utcnow()
                    )
                    session.add(lead)
                    results.append({
                        'email': email,
                        'url': url,
                        'search_term_id': search_term_id
                    })
            
            await session.commit()
        
        return results

    async def batch_google_search(self, search_term: str, num_results: int = 10) -> List[str]:
        """Perform Google search in batches"""
        # Note: Implement actual Google search logic here
        # This is a placeholder that should be replaced with actual search implementation
        return []

    async def execute_search(self, search_term_id: int, process_id: Optional[int] = None) -> Dict:
        """Execute search for a given search term"""
        results = []
        total_leads = 0
        
        async with get_db() as session:
            search_term = await session.get(SearchTerm, search_term_id)
            if not search_term:
                return {'error': 'Search term not found'}
            
            # Update process status if provided
            if process_id:
                process = await session.get(SearchProcess, process_id)
                if process:
                    process.status = 'running'
                    process.started_at = datetime.utcnow()
                    await session.commit()
            
            try:
                # Get project settings
                project = await session.get_project(search_term.project_id)
                excluded_domains = project.settings.get('excluded_domains', [])
                
                # Perform search in batches
                urls = await self.batch_google_search(search_term.term)
                
                # Process URLs concurrently
                tasks = [
                    self.process_url(url, search_term_id, excluded_domains)
                    for url in urls
                ]
                
                url_results = await asyncio.gather(*tasks)
                
                # Flatten results
                for result in url_results:
                    results.extend(result)
                    total_leads += len(result)
                
                # Update process status
                if process_id:
                    process = await session.get(SearchProcess, process_id)
                    if process:
                        process.status = 'completed'
                        process.completed_at = datetime.utcnow()
                        process.results = {
                            'total_leads': total_leads,
                            'results': results
                        }
                        await session.commit()
                
            except Exception as e:
                error_msg = f"Error in search execution: {str(e)}"
                logging.error(error_msg)
                
                if process_id:
                    process = await session.get(SearchProcess, process_id)
                    if process:
                        process.status = 'failed'
                        process.error = error_msg
                        await session.commit()
                
                return {'error': error_msg}
        
        return {
            'total_leads': total_leads,
            'results': results
        }

    async def optimize_search_terms(self, base_terms: List[str], kb_info: Dict) -> List[str]:
        """Optimize search terms using AI"""
        return await ai_service.optimize_search_terms(base_terms, kb_info)