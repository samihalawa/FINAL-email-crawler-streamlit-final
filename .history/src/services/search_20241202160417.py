import asyncio
from typing import List, Dict, Any
import aiohttp
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import json
from datetime import datetime

from core.logging import app_logger
from services.ai import AIService

ai_service = AIService()

class SearchService:
    """Optimized search service with parallel processing"""
    
    def __init__(self):
        self.session = None
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        self.current_agent = 0
    
    async def initialize(self):
        """Initialize the search service"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the search service"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _get_next_user_agent(self) -> str:
        """Get next user agent in rotation"""
        self.current_agent = (self.current_agent + 1) % len(self.user_agents)
        return self.user_agents[self.current_agent]
    
    async def search(self, query: str, excluded_domains: List[str] = None) -> List[Dict[str, Any]]:
        """Perform search with parallel processing"""
        await self.initialize()
        
        # Prepare search
        excluded_domains = excluded_domains or []
        results = []
        
        # Search multiple sources in parallel
        tasks = [
            self.search_google(query, excluded_domains),
            self.search_linkedin(query, excluded_domains),
            self.search_other_sources(query, excluded_domains)
        ]
        
        # Gather results
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for source_results in all_results:
            if isinstance(source_results, Exception):
                app_logger.error("Search error", exc_info=source_results)
                continue
            results.extend(source_results)
        
        # Remove duplicates
        seen_emails = set()
        unique_results = []
        for result in results:
            if result['email'] not in seen_emails:
                seen_emails.add(result['email'])
                unique_results.append(result)
        
        return unique_results
    
    async def search_google(self, query: str, excluded_domains: List[str]) -> List[Dict[str, Any]]:
        """Search using Google"""
        results = []
        
        try:
            # Prepare search URL
            search_url = f"https://www.google.com/search?q={query}&num=100"
            
            # Make request
            headers = {'User-Agent': self._get_next_user_agent()}
            async with self.session.get(search_url, headers=headers) as response:
                if response.status != 200:
                    return results
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract results
                for result in soup.select('.g'):
                    try:
                        link = result.find('a')['href']
                        if any(domain in link for domain in excluded_domains):
                            continue
                        
                        # Process result in parallel
                        results.extend(await self.process_search_result(link))
                        
                    except Exception as e:
                        app_logger.error(f"Error processing Google result: {str(e)}")
        
        except Exception as e:
            app_logger.error("Google search error", exc_info=e)
        
        return results
    
    async def search_linkedin(self, query: str, excluded_domains: List[str]) -> List[Dict[str, Any]]:
        """Search using LinkedIn"""
        results = []
        
        try:
            # Prepare search URL
            search_url = f"https://www.linkedin.com/search/results/people/?keywords={query}"
            
            # Make request
            headers = {
                'User-Agent': self._get_next_user_agent(),
                'Accept': 'application/json'
            }
            async with self.session.get(search_url, headers=headers) as response:
                if response.status != 200:
                    return results
                
                data = await response.json()
                
                # Process profiles in parallel
                tasks = []
                for profile in data.get('elements', []):
                    if profile.get('publicIdentifier'):
                        tasks.append(self.process_linkedin_profile(profile['publicIdentifier']))
                
                # Gather results
                profile_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in profile_results:
                    if isinstance(result, Exception):
                        continue
                    if result:
                        results.append(result)
        
        except Exception as e:
            app_logger.error("LinkedIn search error", exc_info=e)
        
        return results
    
    async def search_other_sources(self, query: str, excluded_domains: List[str]) -> List[Dict[str, Any]]:
        """Search other sources (e.g., company websites, directories)"""
        results = []
        
        # Add other search sources here
        # Example: Business directories, industry-specific sites, etc.
        
        return results
    
    async def process_search_result(self, url: str) -> List[Dict[str, Any]]:
        """Process a search result URL"""
        results = []
        
        try:
            # Make request
            headers = {'User-Agent': self._get_next_user_agent()}
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    return results
                
                html = await response.text()
                
                # Extract information
                emails = self.extract_emails(html)
                names = self.extract_names(html)
                company = self.extract_company(html, url)
                
                # Create lead entries
                for email in emails:
                    # Use AI to validate and enrich lead data
                    lead_data = await ai_service.enrich_lead_data({
                        'email': email,
                        'names': names,
                        'company': company,
                        'url': url
                    })
                    
                    if lead_data:
                        results.append(lead_data)
        
        except Exception as e:
            app_logger.error(f"Error processing URL {url}: {str(e)}")
        
        return results
    
    async def process_linkedin_profile(self, profile_id: str) -> Dict[str, Any]:
        """Process a LinkedIn profile"""
        try:
            # Make request
            url = f"https://www.linkedin.com/in/{profile_id}"
            headers = {'User-Agent': self._get_next_user_agent()}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                
                # Extract profile data
                data = {
                    'linkedin_url': url,
                    'first_name': self.extract_linkedin_field(html, 'firstName'),
                    'last_name': self.extract_linkedin_field(html, 'lastName'),
                    'company': self.extract_linkedin_field(html, 'company'),
                    'position': self.extract_linkedin_field(html, 'position')
                }
                
                # Use AI to find email
                email = await ai_service.find_email(data)
                if email:
                    data['email'] = email
                    return data
        
        except Exception as e:
            app_logger.error(f"Error processing LinkedIn profile {profile_id}: {str(e)}")
        
        return None
    
    def extract_emails(self, html: str) -> List[str]:
        """Extract email addresses from HTML"""
        # Basic email regex pattern
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(pattern, html)))
    
    def extract_names(self, html: str) -> List[str]:
        """Extract names from HTML"""
        # This is a basic implementation
        # Consider using AI/NLP for better name extraction
        soup = BeautifulSoup(html, 'html.parser')
        names = []
        
        # Look for common name patterns
        for tag in soup.find_all(['h1', 'h2', 'h3', 'strong']):
            text = tag.get_text().strip()
            if len(text.split()) == 2:  # First and last name
                names.append(text)
        
        return names
    
    def extract_company(self, html: str, url: str) -> str:
        """Extract company name from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try meta tags first
            meta_org = soup.find('meta', property='og:site_name')
            if meta_org:
                return meta_org['content']
            
            # Try the title
            title = soup.title.string if soup.title else ''
            if title:
                # Remove common suffixes
                title = re.sub(r'\s*[-|]\s*.+$', '', title)
                return title.strip()
            
            # Use domain name as fallback
            domain = url.split('/')[2]
            return domain.split('.')[0].capitalize()
            
        except Exception:
            return ''
    
    def extract_linkedin_field(self, html: str, field: str) -> str:
        """Extract field from LinkedIn profile HTML"""
        try:
            # Look for JSON-LD data
            soup = BeautifulSoup(html, 'html.parser')
            scripts = soup.find_all('script', type='application/ld+json')
            
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if field in data:
                        return data[field]
                except:
                    continue
            
            # Fallback to HTML parsing
            if field == 'firstName':
                tag = soup.find('h1', class_='text-heading-xlarge')
                if tag:
                    return tag.get_text().split()[0]
            elif field == 'lastName':
                tag = soup.find('h1', class_='text-heading-xlarge')
                if tag:
                    return tag.get_text().split()[-1]
            elif field == 'company':
                tag = soup.find('h2', class_='text-heading-small')
                if tag:
                    return tag.get_text().strip()
            elif field == 'position':
                tag = soup.find('div', class_='text-body-medium')
                if tag:
                    return tag.get_text().strip()
        
        except Exception:
            pass
        
        return ''