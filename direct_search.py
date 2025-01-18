import requests
from bs4 import BeautifulSoup
import re
import random
import time
import urllib3
import logging
from urllib.parse import urlparse

# Disable SSL warnings and set up logging
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO)

def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    ]
    return random.choice(user_agents)

def should_skip_domain(domain: str):
    skip_domains = {
        "www.airbnb.es",
        "www.airbnb.com", 
        "www.linkedin.com",
        "es.linkedin.com",
        "www.idealista.com",
        "www.facebook.com",
        "www.instagram.com",
        "www.youtube.com",
        "youtu.be",
    }
    return domain in skip_domains

def get_domain_from_url(url: str):
    return urlparse(url).netloc

def google_search(query: str, num_results=10, lang="es"):
    """Perform a Google search and return results."""
    try:
        print(f"Starting search for query: {query}")
        results = []
        
        # Configure session with simpler headers
        session = requests.Session()
        session.headers.update({
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": f"{lang},en-US;q=0.5"
        })
        
        # Perform search with direct URL
        search_url = f"https://www.google.com/search?q={query}&num={num_results}&hl={lang}"
        response = session.get(search_url, verify=False, timeout=10)
        
        if response.status_code != 200:
            print(f"Google search failed with status code: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all search result divs with simplified selectors
        search_divs = soup.find_all('div', {'class': ['g', 'jtfYYd']})
        
        for div in search_divs:
            try:
                # Find the main link
                link_elem = div.find('a')
                if not link_elem:
                    continue
                    
                url = link_elem.get('href', '')
                if not url or not url.startswith('http'):
                    continue
                    
                domain = get_domain_from_url(url).lower()
                if should_skip_domain(domain):
                    continue
                    
                # Get title and snippet with simplified selectors
                title = ""
                snippet = ""
                
                title_elem = div.find('h3')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                
                snippet_elem = div.find('div', {'class': 'VwiC3b'})
                if snippet_elem:
                    snippet = snippet_elem.get_text(strip=True)
                
                results.append({
                    'link': url,
                    'title': title,
                    'snippet': snippet,
                    'domain': domain
                })
                
            except Exception as e:
                print(f"Error parsing search result: {str(e)}")
                continue
                
        return results[:num_results]
        
    except Exception as e:
        print(f"Google search error for query '{query}': {str(e)}")
        return []

def extract_emails_from_html(html_content: str):
    pattern = r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
    return re.findall(pattern, html_content)

def is_valid_contact_email(email: str):
    email_low = email.lower()
    invalid_patterns = [
        "sentry", "noreply", "no-reply", "donotreply", "do-not-reply", "automated",
        "notification", "alert", "system", "admin@", "postmaster", "mailer-daemon",
        "webmaster", "hostmaster", "support@", "error@", "report@", "test@",
        "office@", "mail@", "email@"
    ]
    if any(p in email_low for p in invalid_patterns):
        return False
    invalid_domains = ["example.com", "test.com", "sample.com", "mail.com", "website.com"]
    domain = email_low.split("@")[-1]
    if domain in invalid_domains:
        return False
    return True

def extract_emails_from_url(url: str):
    try:
        headers = {"User-Agent": get_random_user_agent()}
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        emails = extract_emails_from_html(response.text)
        valid_emails = [e for e in emails if is_valid_contact_email(e)]
        
        return list(set(valid_emails))  # Remove duplicates
        
    except Exception as e:
        print(f"Error extracting emails from {url}: {e}")
        return []

if __name__ == "__main__":
    # Test search
    query = "software engineer barcelona email contact"
    results = google_search(query, num_results=5)
    
    print("\nSearch Results:")
    for result in results:
        print(f"\nTitle: {result['title']}")
        print(f"URL: {result['link']}")
        print(f"Domain: {result['domain']}")
        
        # Extract emails
        emails = extract_emails_from_url(result['link'])
        if emails:
            print("Emails found:", emails)
        
        # Small delay between requests
        time.sleep(2)
