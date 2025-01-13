import aiohttp
import asyncio
import re
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

async def fetch_page(url):
    ua = UserAgent()
    headers = {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, ssl=False) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
    return None

def extract_emails(html):
    if not html:
        return []
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, html, re.IGNORECASE)
    return list(set(emails))

def is_valid_email(email):
    invalid_patterns = ['.png@', '.jpg@', '.jpeg@', '.gif@', '.css@', '.js@',
                       '@example.com', '@test.com', '@sample.com',
                       'noreply@', 'no-reply@', 'donotreply@']
    
    email_lower = email.lower()
    return not any(pattern in email_lower for pattern in invalid_patterns)

async def main():
    # Search terms that worked well
    search_terms = [
        "software engineer spain",
        "desarrollador senior",
        "tech recruiter barcelona",
        "IT recruiter madrid",
        "desarrollador python"
    ]
    
    all_results = []
    for term in search_terms:
        url = f"https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword={term}"
        logging.info(f"Searching: {term}")
        
        html = await fetch_page(url)
        if html:
            emails = extract_emails(html)
            valid_emails = [email for email in emails if is_valid_email(email)]
            
            for email in valid_emails:
                result = {
                    'search_term': term,
                    'email': email,
                    'source': 'InfoJobs'
                }
                all_results.append(result)
                logging.info(f"Found email: {email}")
        
        # Be nice to the server
        await asyncio.sleep(2)
    
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv('recruiter_emails.csv', index=False)
        print("\nResults Summary:")
        print(f"Total emails found: {len(all_results)}")
        print("\nEmails by domain:")
        print(df['email'].apply(lambda x: x.split('@')[1]).value_counts().head())
        print("\nSample emails:")
        print(df['email'].head())
    else:
        print("No results found")

if __name__ == "__main__":
    asyncio.run(main()) 