import streamlit as st
from dotenv import load_dotenv
import logging
import os
from sqlalchemy import create_engine, func, text, distinct
from sqlalchemy.orm import sessionmaker
from models import Base, Lead, LeadSource, SearchTerm, EmailCampaign, EmailTemplate
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from urllib.parse import urlparse
import re
from datetime import datetime

# Must be first Streamlit command
st.set_page_config(
    page_title="AutoclientAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def initialize_settings():
    try:
        load_dotenv()
        DATABASE_URL = os.getenv('DATABASE_URL')
        if not DATABASE_URL:
            st.error("DATABASE_URL not found in environment variables")
            return False

        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return True

    except Exception as e:
        logging.exception(f"Error in initialize_settings: {str(e)}")
        return False

def get_domain_from_url(url):
    """Extract domain from URL."""
    return urlparse(url).netloc

def is_valid_email(email):
    """Validate email format."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def extract_emails_from_html(html_content):
    """Extract email addresses from HTML content."""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, html_content)

def get_page_title(html_content):
    """Extract page title from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.title.string if soup.title else ''

def get_page_description(html_content):
    """Extract meta description from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    meta = soup.find('meta', {'name': 'description'})
    return meta.get('content', '') if meta else ''

def extract_info_from_page(soup):
    """Extract information from page."""
    name = soup.find('meta', {'name': 'author'})
    name = name['content'] if name else ''

    company = soup.find('meta', {'property': 'og:site_name'})
    company = company['content'] if company else ''

    job_title = soup.find('meta', {'name': 'job_title'})
    job_title = job_title['content'] if job_title else ''

    return name, company, job_title

def save_lead(session, email, first_name=None, company=None, job_title=None, url=None, search_term_id=None, created_at=None):
    """Save a new lead to database."""
    try:
        existing_lead = session.query(Lead).filter_by(email=email).first()
        if existing_lead:
            return existing_lead

        new_lead = Lead(
            email=email,
            first_name=first_name,
            company=company,
            job_title=job_title,
            created_at=created_at
        )
        session.add(new_lead)
        session.commit()

        if url and search_term_id:
            lead_source = LeadSource(
                lead_id=new_lead.id,
                search_term_id=search_term_id,
                url=url,
                domain=get_domain_from_url(url),
                created_at=created_at
            )
            session.add(lead_source)
            session.commit()

        return new_lead
    except Exception as e:
        logging.error(f"Error saving lead: {str(e)}")
        session.rollback()
        return None

def manual_search(session, terms, num_results, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False, language='ES', enable_email_sending=True, log_container=None, from_email=None, reply_to=None, email_template=None):
    """Perform manual search for leads."""
    ua = UserAgent()
    results = []
    total_leads = 0
    domains_processed = set()
    processed_emails_per_domain = {}

    for original_term in terms:
        try:
            # Add the term to search terms if it doesn't exist
            existing_term = session.query(SearchTerm).filter_by(term=original_term).first()
            search_term_id = existing_term.id if existing_term else None

            if log_container:
                log_container.write(f"Searching for '{original_term}'")

            for url in search(original_term, num_results=num_results, lang=language):
                domain = get_domain_from_url(url)
                if ignore_previously_fetched and domain in domains_processed:
                    continue

                try:
                    if not url.startswith(('http://', 'https://')):
                        url = 'http://' + url

                    response = requests.get(url, timeout=10, verify=False, headers={'User-Agent': ua.random})
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')

                    valid_emails = [email for email in extract_emails_from_html(response.text) if is_valid_email(email)]

                    if not valid_emails:
                        continue

                    name, company, job_title = extract_info_from_page(soup)
                    page_title = get_page_title(response.text)
                    page_description = get_page_description(response.text)

                    if domain not in processed_emails_per_domain:
                        processed_emails_per_domain[domain] = set()

                    for email in valid_emails:
                        if email in processed_emails_per_domain[domain]:
                            continue

                        processed_emails_per_domain[domain].add(email)

                        lead = save_lead(
                            session=session,
                            email=email,
                            first_name=name,
                            company=company,
                            job_title=job_title,
                            url=url,
                            search_term_id=search_term_id,
                            created_at=datetime.utcnow()
                        )

                        if lead:
                            total_leads += 1
                            results.append({
                                'Email': email,
                                'URL': url,
                                'Lead Source': original_term,
                                'Title': page_title,
                                'Description': page_description,
                                'Name': name,
                                'Company': company,
                                'Job Title': job_title
                            })

                            if log_container:
                                log_container.write(f"Saved lead: {email}")

                    domains_processed.add(domain)

                except requests.RequestException as e:
                    if log_container:
                        log_container.error(f"Error processing URL {url}: {str(e)}")

        except Exception as e:
            if log_container:
                log_container.error(f"Error processing term '{original_term}': {str(e)}")

    if log_container:
        log_container.info(f"Total leads found: {total_leads}")

    return {"total_leads": total_leads, "results": results}

def main():
    if not initialize_settings():
        st.error("Failed to initialize application. Please check the logs and configuration.")
        return

    st.title("Welcome to AutoclientAI")

if __name__ == "__main__":
    main()