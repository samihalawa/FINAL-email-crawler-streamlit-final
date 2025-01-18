import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import random
import time
import urllib3
import logging

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    ]
    return random.choice(user_agents)

def google_search(query, num_results=10):
    logger.info(f"Searching for: {query}")
    
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    search_url = f"https://www.google.com/search?q={query}&num={num_results}"
    logger.info(f"Search URL: {search_url}")
    
    try:
        response = requests.get(search_url, headers=headers, verify=False)
        response.raise_for_status()
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response length: {len(response.text)}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = []
        
        # Find all search result divs
        results = soup.find_all('div', {'class': ['g', 'tF2Cxc']})
        logger.info(f"Found {len(results)} raw results")
        
        for result in results:
            try:
                link = result.find('a')
                if not link:
                    continue
                    
                url = link.get('href')
                if not url or not url.startswith('http'):
                    continue
                
                title = result.find('h3')
                title = title.get_text() if title else ''
                
                snippet = result.find('div', {'class': 'VwiC3b'})
                snippet = snippet.get_text() if snippet else ''
                
                search_results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet
                })
                logger.info(f"Added result: {title} - {url}")
                
            except Exception as e:
                logger.error(f"Error parsing result: {e}")
                continue
                
        logger.info(f"Returning {len(search_results)} processed results")
        return search_results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def extract_emails(url):
    logger.info(f"Extracting emails from: {url}")
    try:
        headers = {"User-Agent": get_random_user_agent()}
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        logger.info(f"Got response from {url}, length: {len(response.text)}")
        
        # Extract emails using regex
        email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
        emails = re.findall(email_pattern, response.text)
        logger.info(f"Found {len(emails)} raw emails")
        
        # Filter valid emails
        valid_emails = []
        for email in emails:
            if '@' in email and '.' in email.split('@')[1]:
                if not any(x in email.lower() for x in ['noreply', 'no-reply', 'donotreply']):
                    valid_emails.append(email)
        
        logger.info(f"Found {len(valid_emails)} valid emails")
        return list(set(valid_emails))  # Remove duplicates
        
    except Exception as e:
        logger.error(f"Error extracting emails from {url}: {e}")
        return []

if __name__ == "__main__":
    # Test the search
    query = "software engineer contact email site:.com"
    results = google_search(query, num_results=5)
    
    print("\nSearch Results:")
    for result in results:
        print(f"\nTitle: {result['title']}")
        print(f"URL: {result['url']}")
        
        # Extract emails from each result
        emails = extract_emails(result['url'])
        if emails:
            print("Emails found:", emails)
        
        # Small delay between requests
        time.sleep(2)
