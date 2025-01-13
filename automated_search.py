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
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_search_urls(term, num_results=10):
    """Generate URLs based on common patterns without using search engines"""
    term_encoded = quote(term)
    base_urls = [
        f"https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword={term_encoded}",
        f"https://www.tecnoempleo.com/busqueda-empleo.php?te={term_encoded}",
        f"https://www.talent.com/jobs?k={term_encoded}&l=Spain",
        f"https://es.trabajo.org/empleo-{term_encoded}",
        f"https://www.welcometothejungle.com/es/jobs?query={term_encoded}",
        f"https://www.jobfluent.com/jobs-{term_encoded}"
    ]
    
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
        
        async with session.get(url, headers=headers, ssl=False, timeout=30) as response:
            if response.status != 200:
                logging.warning(f"Failed to fetch {url} - Status: {response.status}")
                return None
            content = await response.text()
            logging.info(f"Successfully fetched {url} - Content length: {len(content)}")
            return content
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return None

def extract_emails(html_content):
    if not html_content:
        return []
    # More comprehensive email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, html_content, re.IGNORECASE)
    # Additional cleaning
    cleaned_emails = []
    for email in emails:
        email = email.lower().strip()
        if email.endswith('.'):
            email = email[:-1]
        cleaned_emails.append(email)
    return list(set(cleaned_emails))

def is_valid_email(email):
    if not email or '@' not in email:
        return False
    
    # Quick pattern match before more expensive checks
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False
        
    # Common exclusions using fast string operations
    invalid_patterns = ['.png@', '.jpg@', '.jpeg@', '.gif@', '.css@', '.js@',
                       '@example.com', '@test.com', '@sample.com',
                       'noreply@', 'no-reply@', 'donotreply@']
    
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
    meta_desc = soup.find('meta', {'name': ['description', 'Description']})
    if meta_desc and 'content' in meta_desc.attrs:
        info['description'] = meta_desc['content'].strip()
    
    # Try multiple methods to get company name
    company_selectors = [
        {'property': 'og:site_name'},
        {'name': 'author'},
        {'class': 'company-name'},
        {'class': 'employer'},
        {'class': 'organization'}
    ]
    
    for selector in company_selectors:
        element = soup.find(attrs=selector)
        if element:
            if 'content' in element.attrs:
                info['company'] = element['content'].strip()
            else:
                info['company'] = element.text.strip()
            break
    
    if not info['company']:
        domain = urlparse(url).netloc.split('.')[-2]
        info['company'] = domain.capitalize()
    
    # Extract phone numbers - Spanish format included
    phone_pattern = r'\b(?:\+?34[-.]?)?\s*(?:\d{3}[-.]?\s*\d{3}[-.]?\s*\d{3}|\d{9})\b'
    text_content = ' '.join([text for text in soup.stripped_strings])
    info['phones'] = list(set(re.findall(phone_pattern, text_content)))
    
    return info

def parse_infojobs(soup, url):
    """Extract job listings from InfoJobs search results"""
    jobs = []
    try:
        # Find all job listing containers - updated selectors
        job_items = soup.find_all('li', class_='ij-List-item')
        
        for item in job_items:
            job = {}
            
            # Get job title and URL
            title_elem = item.find('a', class_='ij-OfferCard-titleLink')
            if title_elem:
                job['title'] = title_elem.text.strip()
                job['url'] = 'https://www.infojobs.net' + title_elem.get('href', '')
            
            # Get company name
            company_elem = item.find('a', class_='ij-OfferCard-companyLogo')
            if company_elem:
                job['company'] = company_elem.get('title', '').strip()
            
            # Get location
            location_elem = item.find('span', class_='ij-OfferCard-location')
            if location_elem:
                job['location'] = location_elem.text.strip()
            
            # Get salary if available
            salary_elem = item.find('span', class_='ij-OfferCard-salaryPeriod')
            if salary_elem:
                job['salary'] = salary_elem.text.strip()
            
            # Get contract type
            contract_elem = item.find('span', class_='ij-OfferCard-contractType')
            if contract_elem:
                job['contract_type'] = contract_elem.text.strip()
            
            # Get experience
            exp_elem = item.find('span', class_='ij-OfferCard-experienceMin')
            if exp_elem:
                job['experience'] = exp_elem.text.strip()
            
            if job.get('title'):  # Only add if we at least got a title
                jobs.append(job)
                logging.info(f"Found job: {job['title']} at {job.get('company', 'Unknown Company')}")
                
        return jobs
    except Exception as e:
        logging.error(f"Error parsing InfoJobs page: {str(e)}")
        return []

async def process_search_results(search_terms, max_results=10):
    ua = UserAgent()
    results = []
    jobs = []  # New list to store job listings
    domains_processed = set()
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
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
                        
                        # Special handling for InfoJobs
                        if 'infojobs.net' in url:
                            job_listings = parse_infojobs(soup, url)
                            for job in job_listings:
                                job['search_term'] = term
                                jobs.append(job)
                            continue
                            
                        # Regular email extraction for other sites
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
                
                # Add a small delay between search terms to avoid overwhelming servers
                await asyncio.sleep(1)
            
            except Exception as e:
                logging.error(f"Error processing term '{term}': {str(e)}")
                continue
    
    # Save job listings to a separate file
    if jobs:
        df_jobs = pd.DataFrame(jobs)
        df_jobs.to_csv('job_listings.csv', index=False)
        logging.info(f"Saved {len(jobs)} job listings to job_listings.csv")
    
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
    results = asyncio.run(process_search_results(search_terms, max_results=6))
    
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
        
        # Try to load and display job listings
        try:
            df_jobs = pd.read_csv('job_listings.csv')
            print("\nJob Listings Summary:")
            print(f"Total jobs found: {len(df_jobs)}")
            print("\nSample Jobs:")
            for _, job in df_jobs.head().iterrows():
                print(f"\nTitle: {job['title']}")
                print(f"Company: {job['company']}")
                print(f"Location: {job.get('location', 'N/A')}")
                print(f"URL: {job.get('url', 'N/A')}")
                print("-" * 50)
        except Exception as e:
            print("No job listings file found")
        
    else:
        print("No results found.")

if __name__ == "__main__":
    main() 