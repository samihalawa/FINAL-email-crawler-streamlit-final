import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
import re
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import concurrent.futures
import asyncio
import aiohttp

def get_domain_from_url(url):
    return urlparse(url).netloc

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def extract_emails_from_html(html_content):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, html_content)

def extract_info_from_page(soup):
    name = soup.find('meta', {'name': 'author'})
    name = name['content'] if name else ''

    company = soup.find('meta', {'property': 'og:site_name'})
    company = company['content'] if company else ''

    job_title = soup.find('meta', {'name': 'job_title'})
    job_title = job_title['content'] if job_title else ''

    return name, company, job_title

# Configure session for better performance
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.1)
adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Create a UserAgent instance
ua = UserAgent()

async def fetch_url_async(url, headers):
    """Fetch URL asynchronously"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=10, ssl=False) as response:
                if response.status == 200:
                    return await response.text()
        except Exception:
            pass
    return None

def process_urls_batch(urls, batch_size=10):
    """Process URLs in batches"""
    headers = {'User-Agent': ua.random}
    
    async def process_batch(batch_urls):
        tasks = [fetch_url_async(url, headers) for url in batch_urls]
        return await asyncio.gather(*tasks)
    
    results = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        batch_results = loop.run_until_complete(process_batch(batch))
        results.extend(batch_results)
        loop.close()
    
    return results 