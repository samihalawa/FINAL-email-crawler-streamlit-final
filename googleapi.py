import os
import json
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, BigInteger, Text, DateTime, ForeignKey, func, Boolean, JSON, text
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from email_validator import validate_email, EmailNotValidError
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import boto3
from botocore.exceptions import ClientError
import uuid
import logging
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Configuration - Load from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
MAX_SEARCHES_PER_DAY = int(os.getenv("MAX_SEARCHES_PER_DAY", 100))

# Database Configuration - Load from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

if not all([API_KEY, SEARCH_ENGINE_ID, DATABASE_URL]):
    raise ValueError("Missing required environment variables for API or database.")

engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI App
app = FastAPI()

# Database Models
class SearchQuota(Base):
    __tablename__ = 'search_quota'
    id = Column(BigInteger, primary_key=True)
    date = Column(DateTime(timezone=True), unique=True)
    searches_used = Column(BigInteger, default=0)

class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text)
    job_title = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    lead_sources = relationship("LeadSource", back_populates="lead")

class LeadSource(Base):
    __tablename__ = 'lead_sources'
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    url = Column(Text)
    page_title = Column(Text)
    meta_description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    lead = relationship("Lead", back_populates="lead_sources")

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    id = Column(BigInteger, primary_key=True)
    provider = Column(Text)
    aws_access_key_id = Column(Text)
    aws_secret_access_key = Column(Text)
    aws_region = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    template_name = Column(Text)
    subject = Column(Text)
    body_content = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    template_id = Column(BigInteger, ForeignKey('email_templates.id'))
    status = Column(Text)
    sent_at = Column(DateTime(timezone=True))
    message_id = Column(Text)
    tracking_id = Column(Text)
    original_subject = Column(Text)
    original_content = Column(Text)
    lead = relationship("Lead")
    template = relationship("EmailTemplate")

class Project(Base):
    __tablename__ = 'projects'
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class Campaign(Base):
    __tablename__ = 'campaigns'
    id = Column(BigInteger, primary_key=True)
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    campaign_name = Column(Text)
    campaign_type = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    auto_send = Column(Boolean, default=False)
    loop_automation = Column(Boolean, default=False)
    ai_customization = Column(Boolean, default=False)
    max_emails_per_group = Column(BigInteger)
    loop_interval = Column(BigInteger)

class CampaignLead(Base):
    __tablename__ = 'campaign_leads'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    status = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    term = Column(Text)
    category = Column(Text)
    language = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class SearchTermGroup(Base):
    __tablename__ = 'search_term_groups'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    id = Column(BigInteger, primary_key=True)
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    kb_name = Column(Text)
    kb_bio = Column(Text)
    kb_values = Column(Text)
    contact_name = Column(Text)
    contact_role = Column(Text)
    contact_email = Column(Text)
    company_description = Column(Text)
    company_mission = Column(Text)
    company_target_market = Column(Text)
    company_other = Column(Text)
    product_name = Column(Text)
    product_description = Column(Text)
    product_target_customer = Column(Text)
    product_other = Column(Text)
    other_context = Column(Text)
    example_email = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

# API Response Models
class SearchResult(BaseModel):
    emails_found: List[str]
    total_results: int
    sources: List[str]
    is_cached: bool = False

class BatchSearchResult(BaseModel):
    total_emails: int
    results: List[SearchResult]
    execution_time: float
    searches_remaining: int

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_search_quota(db: Session) -> int:
    """Get the remaining search quota for the current day."""
    today = datetime.now().date()
    try:
        quota = db.query(SearchQuota).filter(
            func.date(SearchQuota.date) == today
        ).first()
        
        if not quota:
            quota = SearchQuota(date=today, searches_used=0)
            db.add(quota)
            db.commit()
        
        return MAX_SEARCHES_PER_DAY - quota.searches_used
    except SQLAlchemyError as e:
        logger.error(f"Error getting search quota: {e}")
        return 0

def increment_search_quota(db: Session) -> None:
    """Increment the search quota for the current day."""
    today = datetime.now().date()
    try:
        quota = db.query(SearchQuota).filter(
            func.date(SearchQuota.date) == today
        ).first()
        
        if quota:
            quota.searches_used += 1
            db.commit()
    except SQLAlchemyError as e:
        logger.error(f"Error incrementing search quota: {e}")

def search_google(query: str, db: Session) -> Dict[str, Any]:
    """Search Google using Custom Search API."""
    try:
        logger.info(f"Starting Google search for query: {query}")
        
        # Check daily quota
        remaining_quota = get_search_quota(db)
        if remaining_quota <= 0:
            logger.warning("Daily search quota exceeded")
            return {"error": "Daily search quota exceeded", "items": []}
        
        # Perform search
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": query,
            "num": 10
        }
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)
        
        logger.info(f"Making request to Google API with params: {params}")
        response = http.get(url, params=params, timeout=10)
        logger.info(f"Google API response status: {response.status_code}")
        
        if response.status_code == 429:
            logger.error("Google API quota exceeded")
            return {"error": "Google API quota exceeded", "items": []}
        
        if response.status_code != 200:
            logger.error(f"Google API error: {response.text}")
            return {"error": f"Google API error: {response.text}", "items": []}
        
        # Update quota
        increment_search_quota(db)
        
        result = response.json()
        if "error" in result:
            logger.error(f"Google API error in response: {result['error']}")
            return {"error": result["error"].get("message", "Unknown error"), "items": []}
        
        logger.info(f"Got {len(result.get('items', []))} results from Google")
        return result
        
    except Exception as e:
        logger.error(f"Error in search_google: {str(e)}")
        return {"error": str(e), "items": []}

def process_url(url: str) -> Optional[Dict[str, Any]]:
    """Fetch and process a URL to extract relevant information."""
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch URL {url}, status code: {response.status_code}")
            return None
        
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.title.string if soup.title else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag:
            meta_desc = meta_tag.get("content", "")
        
        return {
            "title": title,
            "description": meta_desc,
            "content": response.text
        }
        
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return None

def manual_search_worker(
    db: Session,
    term: str,
    num_results: int,
    ignore_previous: bool,
    optimize_english: bool,
    optimize_spanish: bool,
    shuffle_keywords: bool,
    language: str
) -> dict:
    try:
        logger.info(f"Starting manual search worker for term: {term}")
        
        # Get search results from Google
        search_results = search_google(term, db)
        
        # Check for errors
        if "error" in search_results:
            error_msg = search_results["error"]
            logger.error(f"Search error: {error_msg}")
            return {
                'results': [],
                'search_logs': [f"Search error: {error_msg}"],
                'email_logs': [],
                'error': error_msg
            }
            
        logger.info(f"Got {len(search_results.get('items', []))} results from Google")
        
        # Extract emails and process results
        results = {
            'results': [],
            'search_logs': [],
            'email_logs': []
        }
        
        processed_urls = set()
        processed_domains = set()
        
        for item in search_results.get('items', []):
            try:
                url = item.get('link')
                if not url:
                    logger.debug(f"Skipping item with no URL: {item}")
                    continue
                    
                logger.info(f"Processing URL: {url}")
                domain = get_domain_from_url(url)
                
                if should_skip_domain(domain):
                    logger.debug(f"Skipping domain: {domain}")
                    continue
                    
                if url in processed_urls or domain in processed_domains:
                    logger.debug(f"Already processed URL/domain: {url}")
                    continue
                    
                # Process the URL
                page_info = process_url(url)
                if not page_info:
                    logger.debug(f"Failed to process URL: {url}")
                    continue
                    
                # Extract and validate emails
                emails = extract_emails(page_info.get('content', ''))
                logger.info(f"Found {len(emails)} emails on {url}")
                
                company_name = page_info.get('title', '').split('|')[0].strip()
                
                for email in emails:
                    if not is_valid_email(email):
                        logger.debug(f"Invalid email: {email}")
                        continue
                        
                    logger.info(f"Processing valid email: {email}")
                    
                    # Save to database
                    lead = save_lead(
                        db,
                        email=email,
                        url=url,
                        company=company_name,
                        page_title=page_info.get('title'),
                        meta_description=page_info.get('description')
                    )
                    
                    if lead:
                        logger.info(f"Saved lead: {email}")
                        results['results'].append({
                            'Email': email,
                            'Company': company_name,
                            'URL': url
                        })
                    else:
                        logger.warning(f"Failed to save lead: {email}")
                
                processed_urls.add(url)
                processed_domains.add(domain)
                
            except Exception as e:
                error_msg = f'Error processing URL {url}: {str(e)}'
                logger.error(error_msg)
                results['search_logs'].append(error_msg)
                continue
                
        logger.info(f"Search worker completed. Found {len(results['results'])} leads")
        return results
        
    except Exception as e:
        error_msg = f"Error in manual search worker: {str(e)}"
        logger.error(error_msg)
        return {
            'results': [],
            'search_logs': [error_msg],
            'email_logs': [],
            'error': error_msg
        }

def extract_emails(html_content: str) -> List[str]:
    """Extract email addresses from HTML content using multiple patterns."""
    emails = set()
    
    # Basic email pattern
    basic_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    emails.update(re.findall(basic_pattern, html_content))
    
    # Look for obfuscated emails
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Check text content
    text_content = soup.get_text()
    emails.update(re.findall(basic_pattern, text_content))
    
    # Check mailto links
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('mailto:'):
            email = href.replace('mailto:', '').split('?')[0]
            if re.match(basic_pattern, email):
                emails.add(email)
    
    # Check for common email patterns in text
    contact_texts = soup.find_all(text=re.compile(r'contacto|contact|email|correo|e-mail'))
    for text in contact_texts:
        parent = text.parent
        if parent:
            emails.update(re.findall(basic_pattern, str(parent)))
    
    return list(emails)

def save_lead(
    db: Session,
    email: str,
    url: str,
    company: Optional[str] = None,
    page_title: Optional[str] = None,
    meta_description: Optional[str] = None
) -> Optional[Lead]:
    """Save lead to the database."""
    try:
        if not is_valid_email(email):
            logger.warning(f"Invalid email: {email}")
            return None
        
        existing_lead = db.query(Lead).filter_by(email=email).first()
        if existing_lead:
            logger.info(f"Existing lead: {email}")
            lead_id = existing_lead.id
        else:
            new_lead = Lead(email=email, company=company)
            db.add(new_lead)
            db.commit()
            db.refresh(new_lead)
            lead_id = new_lead.id
            logger.info(f"New lead: {email}")

        lead_source = LeadSource(
            lead_id=lead_id,
            url=url,
            page_title=page_title,
            meta_description=meta_description
        )
        db.add(lead_source)
        db.commit()
        logger.info(f"Saved source: {url}")
        return new_lead if not existing_lead else existing_lead
    except SQLAlchemyError as e:
        logger.error(f"Database error saving lead: {str(e)}")
        db.rollback()
        return None

def get_random_user_agent() -> str:
    """Get a random user agent string."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59",
    ]
    return random.choice(user_agents)

def should_skip_domain(domain: str) -> bool:
    """Check if a domain should be skipped."""
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

def get_domain_from_url(url: str) -> str:
    """Extract the domain from a URL."""
    return urlparse(url).netloc

def is_valid_email(email: str) -> bool:
    """Validate an email address."""
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

# Define the /search endpoint
class SearchRequest(BaseModel):
    term: str
    num_results: int = 10
    ignore_previous: bool = False
    optimize_english: bool = False
    optimize_spanish: bool = False
    shuffle_keywords: bool = False
    language: str = "en"

@app.post("/search")
async def search_endpoint(search_request: SearchRequest, db: Session = Depends(get_db)):
    """Endpoint to perform a manual search."""
    results = manual_search_worker(
        db=db,
        term=search_request.term,
        num_results=search_request.num_results,
        ignore_previous=search_request.ignore_previous,
        optimize_english=search_request.optimize_english,
        optimize_spanish=search_request.optimize_spanish,
        shuffle_keywords=search_request.shuffle_keywords,
        language=search_request.language
    )
    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
