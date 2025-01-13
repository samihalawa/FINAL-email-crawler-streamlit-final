from playwright.sync_api import sync_playwright
import re
import pandas as pd
import logging
import time
import random

logging.basicConfig(level=logging.INFO)

def extract_emails(text):
    if not text:
        return []
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text, re.IGNORECASE)
    return list(set(emails))

def is_valid_email(email):
    invalid_patterns = ['.png@', '.jpg@', '.jpeg@', '.gif@', '.css@', '.js@',
                       '@example.com', '@test.com', '@sample.com',
                       'noreply@', 'no-reply@', 'donotreply@']
    
    email_lower = email.lower()
    return not any(pattern in email_lower for pattern in invalid_patterns)

def scrape_infojobs():
    search_terms = [
        "software engineer spain",
        "desarrollador senior",
        "tech recruiter barcelona",
        "IT recruiter madrid",
        "desarrollador python"
    ]
    
    all_results = []
    
    with sync_playwright() as p:
        # Launch browser with more realistic settings
        browser = p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='es-ES'
        )
        
        page = context.new_page()
        
        # Enable JavaScript console logging
        page.on("console", lambda msg: logging.debug(f"Browser console: {msg.text}"))
        
        for term in search_terms:
            try:
                logging.info(f"Searching for: {term}")
                
                # Go to search page
                url = f"https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword={term}"
                page.goto(url, wait_until='networkidle')
                
                # Wait for content to load
                page.wait_for_selector('main', timeout=10000)
                
                # Scroll to load more content
                for _ in range(3):
                    page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                    time.sleep(1)
                
                # Get all job cards
                job_cards = page.query_selector_all('article.ij-OfferCard')
                
                for card in job_cards:
                    try:
                        # Click to open job details
                        card.click()
                        time.sleep(random.uniform(1, 2))
                        
                        # Get all text content
                        content = page.content()
                        
                        # Extract emails
                        emails = extract_emails(content)
                        valid_emails = [email for email in emails if is_valid_email(email)]
                        
                        for email in valid_emails:
                            result = {
                                'search_term': term,
                                'email': email,
                                'source': 'InfoJobs'
                            }
                            all_results.append(result)
                            logging.info(f"Found email: {email}")
                        
                    except Exception as e:
                        logging.error(f"Error processing job card: {str(e)}")
                        continue
                
                # Random delay between searches
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                logging.error(f"Error processing search term '{term}': {str(e)}")
                continue
        
        browser.close()
    
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
    scrape_infojobs() 