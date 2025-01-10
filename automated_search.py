import os
import sys
import json
import logging
import asyncio
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import aiohttp
import re
from fake_useragent import UserAgent
from urllib.parse import urlparse, quote
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_search_urls(term, num_results=10):
    """Generate URLs based on common patterns without using search engines"""
    term_encoded = quote(term)
    base_urls = [
        f"https://www.linkedin.com/jobs/search?keywords={term_encoded}",
        f"https://www.glassdoor.com/Job/spain-{term_encoded}-jobs-SRCH_IL.0,5_IN219",
        f"https://www.indeed.es/jobs?q={term_encoded}",
        f"https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword={term_encoded}",
        f"https://www.tecnoempleo.com/busqueda-empleo.php?te={term_encoded}",
        f"https://www.jobfluent.com/jobs-{term_encoded}",
        f"https://www.welcometothejungle.com/es/jobs?query={term_encoded}",
        f"https://www.talent.com/jobs?k={term_encoded}&l=Spain",
        f"https://es.trabajo.org/empleo-{term_encoded}",
        f"https://www.workday.com/jobs-{term_encoded}"
    ]
    
    # Add company career pages
    companies = [
        "telefonica", "bbva", "santander", "repsol", "iberdrola", 
        "inditex", "mercadona", "caixabank", "mapfre", "naturgy"
    ]
    for company in companies:
        base_urls.append(f"https://www.{company}.com/careers")
        base_urls.append(f"https://www.{company}.es/empleo")
    
    return base_urls[:num_results]

async def fetch_url(session, url, ua):
    try:
        headers = {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        async with session.get(url, headers=headers, ssl=False, timeout=10) as response:
            if response.status != 200:
                return None
            return await response.text()
    except Exception as e:
        return None

def extract_emails(html_content):
    if not html_content:
        return []
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return list(set(re.findall(email_pattern, html_content)))

def is_valid_email(email):
    if not email or '@' not in email:
        return False
    
    # Quick pattern match before more expensive checks
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False
        
    # Common exclusions using fast string operations
    invalid_patterns = ['.png@', '.jpg@', '.jpeg@', '.gif@', '.css@', '.js@',
                       '@example.com', '@test.com', '@sample.com',
                       'noreply@', 'no-reply@', 'donotreply@',
                       'admin@', 'administrator@', 'webmaster@', 'info@', 'contact@', 'support@']
    
    email_lower = email.lower()
    return not any(pattern in email_lower for pattern in invalid_patterns)

def extract_company_info(soup, url):
    if not soup:
        return {}
    
    info = {
        'title': '',
        'description': '',
        'company': '',
        'phones': []
    }
    
    # Extract title
    if soup.title:
        info['title'] = soup.title.string.strip() if soup.title.string else ''
    
    # Extract meta description
    meta_desc = soup.find('meta', {'name': 'description'})
    if meta_desc and 'content' in meta_desc.attrs:
        info['description'] = meta_desc['content'].strip()
    
    # Try multiple methods to get company name
    company_meta = soup.find('meta', {'property': 'og:site_name'})
    if company_meta and 'content' in company_meta.attrs:
        info['company'] = company_meta['content'].strip()
    if not info['company']:
        domain = urlparse(url).netloc.split('.')[-2]
        info['company'] = domain.capitalize()
    
    # Extract phone numbers
    phone_pattern = r'\b(?:\+?1[-.]?)?\s*(?:\([0-9]{3}\)|[0-9]{3})[-.]?\s*[0-9]{3}[-.]?\s*[0-9]{4}\b'
    text_content = ' '.join([text for text in soup.stripped_strings])
    info['phones'] = list(set(re.findall(phone_pattern, text_content)))
    
    return info

async def process_search_results(search_terms, max_results=10):
    ua = UserAgent()
    results = []
    domains_processed = set()
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        for term in search_terms:
            logging.info(f"Processing search term: {term}")
            try:
                urls = generate_search_urls(term, max_results)
                
                if not urls:
                    continue
                
                # Process all URLs at once
                tasks = []
                for url in urls:
                    domain = urlparse(url).netloc
                    if domain not in domains_processed:
                        domains_processed.add(domain)
                        tasks.append(fetch_url(session, url, ua))
                
                if tasks:
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for url, response in zip(urls, responses):
                        if isinstance(response, Exception) or not response:
                            continue
                            
                        soup = BeautifulSoup(response, 'html.parser')
                        emails = extract_emails(response)
                        valid_emails = [email for email in emails if is_valid_email(email)]
                        
                        if valid_emails:
                            info = extract_company_info(soup, url)
                            for email in valid_emails:
                                result = {
                                    'search_term': term,
                                    'url': url,
                                    'email': email,
                                    'company': info.get('company', ''),
                                    'title': info.get('title', ''),
                                    'description': info.get('description', ''),
                                    'phones': info.get('phones', [])
                                }
                                results.append(result)
                                logging.info(f"Found valid email: {email} from {url}")
            
            except Exception as e:
                logging.error(f"Error processing term '{term}': {str(e)}")
                continue
    
    return results

def save_results(results, filename='search_results.csv'):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    logging.info(f"Results saved to {filename}")
    return df

def main():
    # Test search terms
    search_terms = [
        "software engineer spain",
        "data scientist barcelona",
        "tech startup madrid",
        "CTO spain startup",
        "developer spain remote"
    ]
    
    # Run the search
    logging.info("Starting automated search test...")
    results = asyncio.run(process_search_results(search_terms, max_results=5))
    
    # Save and analyze results
    if results:
        df = save_results(results)
        
        print("\nSearch Results Summary:")
        print(f"Total results found: {len(results)}")
        
        print("\nResults by search term:")
        term_counts = df['search_term'].value_counts()
        print(term_counts)
        
        print("\nTop domains found:")
        domains = df['url'].apply(lambda x: urlparse(x).netloc)
        print(domains.value_counts().head())
        
        print("\nSample Results:")
        for result in results[:5]:
            print(f"\nEmail: {result['email']}")
            print(f"URL: {result['url']}")
            print(f"Company: {result['company']}")
            print(f"Title: {result['title'][:100]}...")
            print("-" * 50)
        
        # Additional analysis
        print("\nEmail patterns:")
        email_domains = df['email'].apply(lambda x: x.split('@')[1])
        print(email_domains.value_counts().head())
        
    else:
        print("No results found.")

if __name__ == "__main__":
    main() 