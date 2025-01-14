# Standard library imports
import os
import json
import re
import logging
import asyncio
import time
import random
import html
import smtplib
import uuid
import threading
import signal
import subprocess
from datetime import datetime, timedelta
from contextlib import contextmanager
from threading import local
from urllib.parse import urlparse, urlencode
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict, Any, Tuple

# Third-party imports
import requests
import pandas as pd
import streamlit as st
import openai
import boto3
import aiohttp
import urllib3
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import (
    func, create_engine, Column, BigInteger, Text, DateTime, 
    ForeignKey, Boolean, JSON, select, text, distinct, and_, 
    or_, inspect
)
from sqlalchemy.orm import (
    declarative_base, sessionmaker, relationship, Session, 
    joinedload, configure_mappers
)
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from openai import OpenAI
from streamlit_tags import st_tags
import plotly.express as px
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure requests retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

# Database configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Default search settings
DEFAULT_SEARCH_SETTINGS = {
    'num_results': 50,
    'ignore_previously_fetched': True,
    'optimize_english': False,
    'optimize_spanish': False,
    'shuffle_keywords_option': False,
    'language': 'ES',
    'enable_email_sending': True
}

# Add thread-local storage for database sessions
thread_local = local()

def get_db_session():
    """Get or create a database session for the current thread."""
    if not hasattr(thread_local, "session"):
        with db_session() as session:
            thread_local.session = session
    return thread_local.session

# Initialize SQLAlchemy Base
Base = declarative_base()

# Database connection with retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
    reraise=True
)
def get_db_connection():
    """Get a database connection with retry logic"""
    try:
        engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
        engine.connect()
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

# Validate database configuration
if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

class Project(Base):
    __tablename__ = 'projects'
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text, default="Default Project")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaigns = relationship("Campaign", back_populates="project")
    knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False)

class Campaign(Base):
    __tablename__ = 'campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_name = Column(Text, default="Default Campaign")
    campaign_type = Column(Text, default="Email")
    project_id = Column(BigInteger, ForeignKey('projects.id'), default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    auto_send = Column(Boolean, default=False)
    loop_automation = Column(Boolean, default=False)
    ai_customization = Column(Boolean, default=False)
    max_emails_per_group = Column(BigInteger, default=40)
    loop_interval = Column(BigInteger, default=60)
    project = relationship("Project", back_populates="campaigns")
    email_campaigns = relationship("EmailCampaign", back_populates="campaign")
    search_terms = relationship("SearchTerm", back_populates="campaign")
    campaign_leads = relationship("CampaignLead", back_populates="campaign")

class CampaignLead(Base):
    __tablename__ = 'campaign_leads'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    status = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="campaign_leads")
    campaign = relationship("Campaign", back_populates="campaign_leads")

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    id = Column(BigInteger, primary_key=True)
    project_id = Column(BigInteger, ForeignKey('projects.id'), nullable=False)
    kb_name, kb_bio, kb_values, contact_name, contact_role, contact_email = [Column(Text) for _ in range(6)]
    company_description, company_mission, company_target_market, company_other = [Column(Text) for _ in range(4)]
    product_name, product_description, product_target_customer, product_other = [Column(Text) for _ in range(4)]
    other_context, example_email = Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    project = relationship("Project", back_populates="knowledge_base")

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']}

# Update the Lead model to remove the domain attribute
class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    phone = Column(Text)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text)
    job_title = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead")
    email_campaigns = relationship("EmailCampaign", back_populates="lead")

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'phone': self.phone,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'company': self.company,
            'job_title': self.job_title,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    template_name, subject, body_content = Column(Text), Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False)
    language = Column(Text, default='ES')  # Add language column
    campaign = relationship("Campaign")
    email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    template_id = Column(BigInteger, ForeignKey('email_templates.id'))
    customized_subject = Column(Text)
    customized_content = Column(Text)  # Make sure this column exists
    original_subject = Column(Text)
    original_content = Column(Text)
    status = Column(Text)
    engagement_data = Column(JSON)
    message_id = Column(Text)
    tracking_id = Column(Text, unique=True)
    sent_at = Column(DateTime(timezone=True))
    ai_customized = Column(Boolean, default=False)
    opened_at = Column(DateTime(timezone=True))
    clicked_at = Column(DateTime(timezone=True))
    open_count = Column(BigInteger, default=0)
    click_count = Column(BigInteger, default=0)
    campaign = relationship("Campaign", back_populates="email_campaigns")
    lead = relationship("Lead", back_populates="email_campaigns")
    template = relationship("EmailTemplate", back_populates="email_campaigns")

class OptimizedSearchTerm(Base):
    __tablename__ = 'optimized_search_terms'
    id = Column(BigInteger, primary_key=True)
    original_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    term = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    original_term = relationship("SearchTerm", back_populates="optimized_terms")

class SearchTermEffectiveness(Base):
    __tablename__ = 'search_term_effectiveness'
    id = Column(BigInteger, primary_key=True)
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    total_results, valid_leads, irrelevant_leads, blogs_found, directories_found = [Column(BigInteger) for _ in range(5)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_term = relationship("SearchTerm", back_populates="effectiveness")

class SearchTermGroup(Base):
    __tablename__ = 'search_term_groups'
    id = Column(BigInteger, primary_key=True)
    name, email_template, description = Column(Text), Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_terms = relationship("SearchTerm", back_populates="group")

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    id = Column(BigInteger, primary_key=True)
    group_id = Column(BigInteger, ForeignKey('search_term_groups.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    term, category = Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    language = Column(Text, default='ES')  # Add language column
    group = relationship("SearchTermGroup", back_populates="search_terms")
    campaign = relationship("Campaign", back_populates="search_terms")
    optimized_terms = relationship("OptimizedSearchTerm", back_populates="original_term")
    lead_sources = relationship("LeadSource", back_populates="search_term")
    effectiveness = relationship("SearchTermEffectiveness", back_populates="search_term", uselist=False)

class LeadSource(Base):
    __tablename__ = 'lead_sources'
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    url, domain, page_title, meta_description, scrape_duration = [Column(Text) for _ in range(5)]
    meta_tags, phone_numbers, content, tags = [Column(Text) for _ in range(4)]
    http_status = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="lead_sources")
    search_term = relationship("SearchTerm", back_populates="lead_sources")

class AIRequestLog(Base):
    __tablename__ = 'ai_request_logs'
    id = Column(BigInteger, primary_key=True)
    function_name, prompt, response, model_used = [Column(Text) for _ in range(4)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    email_campaign_id = Column(BigInteger, ForeignKey('email_campaigns.id'))
    lead = relationship("Lead")
    email_campaign = relationship("EmailCampaign")

class AutomationLog(Base):
    __tablename__ = 'automation_logs'
    id = Column(BigInteger, primary_key=True, index=True)
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime)
    status = Column(Text, default='running')
    logs = Column(JSON, default=list)
    leads_gathered = Column(BigInteger, default=0)
    emails_sent = Column(BigInteger, default=0)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    campaign = relationship("Campaign", backref="automation_logs")

# Replace the existing EmailSettings class with this unified Settings class
class Settings(Base):
    __tablename__ = 'settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    setting_type = Column(Text, nullable=False)  # 'general', 'email', etc.
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    email = Column(Text, nullable=False)
    provider = Column(Text, nullable=False)  # 'ses' or 'smtp'
    smtp_server = Column(Text)
    smtp_port = Column(BigInteger)
    smtp_username = Column(Text)
    smtp_password = Column(Text)
    aws_access_key_id = Column(Text)
    aws_secret_access_key = Column(Text)
    aws_region = Column(Text, default='us-east-1')
    daily_limit = Column(BigInteger, default=50000)  # AWS SES default limit
    hourly_limit = Column(BigInteger, default=5)  # Start with a conservative limit
    is_active = Column(Boolean, default=True)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

@contextmanager
def db_session():
    """Database session context manager with retry logic"""
    engine = get_db_connection()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

def settings_page():
    st.title("Settings")
    
    try:
        with db_session() as session:
            # Email Settings
            st.subheader("Email Settings")
            email_settings = fetch_email_settings(session)
            
            # Rest of the settings page code...
            
    except Exception as e:
        st.error(f"Error loading settings: {str(e)}")

def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    """Send email using either AWS SES or SMTP based on email settings."""
    try:
        # Get email settings for the from_email
        email_settings = session.query(EmailSettings).filter_by(
            email=from_email, 
            is_active=True
        ).first()
        
        if not email_settings:
            logging.error(f"No active email settings found for {from_email}")
            return None, None

        tracking_id = str(uuid.uuid4())
        tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
        wrapped_body = wrap_email_body(body)
        tracked_body = wrapped_body.replace('</body>', f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/></body>')

        # Add click tracking
        soup = BeautifulSoup(tracked_body, 'html.parser')
        for a in soup.find_all('a', href=True):
            original_url = a['href']
            tracked_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'click', 'url': original_url})}"
            a['href'] = tracked_url
        tracked_body = str(soup)

        if email_settings.provider.lower() == 'ses':
            if not all([email_settings.aws_access_key_id, email_settings.aws_secret_access_key, email_settings.aws_region]):
                logging.error("Missing AWS credentials for SES")
                return None, None

            try:
                if ses_client is None:
                    ses_client = boto3.client(
                        'ses',
                        aws_access_key_id=email_settings.aws_access_key_id,
                        aws_secret_access_key=email_settings.aws_secret_access_key,
                        region_name=email_settings.aws_region
                    )

                response = ses_client.send_email(
                    Source=from_email,
                    Destination={'ToAddresses': [to_email]},
                    Message={
                        'Subject': {'Data': subject, 'Charset': charset},
                        'Body': {'Html': {'Data': tracked_body, 'Charset': charset}}
                    },
                    ReplyToAddresses=[reply_to] if reply_to else []
                )
                return response, tracking_id

            except Exception as e:
                logging.error(f"SES error: {str(e)}")
                return None, None

        elif email_settings.provider.lower() == 'smtp':
            try:
                msg = MIMEMultipart()
                msg['From'] = from_email
                msg['To'] = to_email
                msg['Subject'] = subject
                if reply_to:
                    msg['Reply-To'] = reply_to
                msg.attach(MIMEText(tracked_body, 'html', charset))

                with smtplib.SMTP(email_settings.smtp_server, email_settings.smtp_port) as server:
                    server.starttls()
                    server.login(email_settings.smtp_username, email_settings.smtp_password)
                    server.send_message(msg)
                return {'MessageId': f'smtp-{tracking_id}'}, tracking_id

            except Exception as e:
                logging.error(f"SMTP error: {str(e)}")
                return None, None

        else:
            logging.error(f"Unknown email provider: {email_settings.provider}")
            return None, None

    except Exception as e:
        logging.error(f"Error in send_email_ses: {str(e)}")
        return None, None

def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    try:
        lead = session.query(Lead).filter_by(email=lead_email).first()
        if not lead:
            logging.error(f"Lead with email {lead_email} not found.")
            return

        new_campaign = EmailCampaign(
            lead_id=lead.id,
            template_id=template_id,
            status=status,
            sent_at=sent_at,
            customized_subject=subject or "No subject",
            message_id=message_id or f"unknown-{uuid.uuid4()}",
            customized_content=email_body or "No content",
            campaign_id=get_active_campaign_id(),
            tracking_id=str(uuid.uuid4())
        )
        session.add(new_campaign)
        session.commit()
    except Exception as e:
        logging.error(f"Error saving email campaign: {str(e)}")
        session.rollback()
        return None
    return new_campaign

def update_log(log_container, message, level='info', search_process_id=None):
    """Update log container with message."""
    if not log_container:
        return
        
    timestamp = datetime.now().strftime('%H:%M:%S')
    icon = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'email': '📧'
    }.get(level, 'ℹ️')
    
    log_message = f"{icon} [{timestamp}] {message}"
    log_container.markdown(log_message)
    
    if search_process_id:
        with db_session() as session:
            try:
                automation_log = session.query(AutomationLog).get(search_process_id)
                if automation_log:
                    automation_log.logs.append({
                        'timestamp': datetime.now().isoformat(),
                        'level': level,
                        'message': message
                    })
                    session.commit()
            except:
                pass

def optimize_search_term(search_term, language):
    """Optimize search term based on language."""
    try:
        # Remove special characters and extra spaces
        term = re.sub(r'[^\w\s]', ' ', search_term)
        term = ' '.join(term.split())
        
        if language.lower() == 'english':
            # Add common English business terms
            business_terms = [
                'company', 'business', 'enterprise', 'corporation',
                'firm', 'organization', 'agency', 'office'
            ]
            term = f"{term} {random.choice(business_terms)}"
        elif language.lower() == 'spanish':
            # Add common Spanish business terms
            business_terms = [
                'empresa', 'negocio', 'compañía', 'corporación',
                'firma', 'organización', 'agencia', 'oficina'
            ]
            term = f"{term} {random.choice(business_terms)}"
            
        return term
    except Exception as e:
        logging.error(f"Error optimizing search term: {str(e)}")
        return search_term

def shuffle_keywords(term):
    """Shuffle keywords in search term."""
    try:
        words = term.split()
        random.shuffle(words)
        return ' '.join(words)
    except Exception as e:
        logging.error(f"Error shuffling keywords: {str(e)}")
        return term

def get_domain_from_url(url):
    """Extract domain from URL."""
    try:
        return urlparse(url).netloc
    except Exception as e:
        logging.error(f"Error extracting domain: {str(e)}")
        return url

def is_valid_email(email):
    """Validate email address."""
    if email is None:
        return False
        
    try:
        # Check for common invalid patterns
        invalid_patterns = [
            r".*\.(png|jpg|jpeg|gif|css|js)$",
            r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
            r"^(email|info|contact|support|hello|hola|hi|salutations|greetings|inquiries|questions)@.*",
            r"^email@email\.com$",
            r".*@example\.com$",
            r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"
        ]
        
        # Check for common typo domains
        typo_domains = [
            "gmil.com", "gmal.com", "gmaill.com", "gnail.com"
        ]
        
        # Check against invalid patterns
        if any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns):
            return False
            
        # Check against typo domains
        if any(email.lower().endswith(f"@{domain}") for domain in typo_domains):
            return False
            
        # Validate email format
        validate_email(email)
        return True
        
    except EmailNotValidError:
        return False
    except Exception as e:
        logging.error(f"Error validating email: {str(e)}")
        return False

def extract_emails_from_html(html_content):
    """Extract email addresses from HTML content."""
    try:
        # Find all email addresses using regex
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, html_content)
        
        # Filter and validate emails
        valid_emails = [email for email in emails if is_valid_email(email)]
        
        return list(set(valid_emails))  # Remove duplicates
    except Exception as e:
        logging.error(f"Error extracting emails: {str(e)}")
        return []

def extract_info_from_page(soup):
    """Extract useful information from BeautifulSoup object."""
    try:
        info = {
            'Title': None,
            'Description': None,
            'Content': None,
            'Emails': []
        }
        
        # Get title
        if soup.title:
            info['Title'] = soup.title.string
            
        # Get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            info['Description'] = meta_desc.get('content')
            
        # Get main content
        content = []
        for tag in soup.find_all(['p', 'div', 'span']):
            if tag.string:
                content.append(tag.string.strip())
        info['Content'] = ' '.join(content)
        
        # Extract emails from the entire HTML
        info['Emails'] = extract_emails_from_html(str(soup))
        
        return info
    except Exception as e:
        logging.error(f"Error extracting page info: {str(e)}")
        return {}

def manual_search(session, terms, num_results, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False, language='ES', enable_email_sending=True, log_container=None, from_email=None, reply_to=None, email_template=None):
    ua, results, total_leads, domains_processed = UserAgent(), [], 0, set()
    processed_emails_per_domain = {}  # Track processed emails per domain
    
    for original_term in terms:
        try:
            search_term_id = add_or_get_search_term(session, original_term, get_active_campaign_id())
            search_term = shuffle_keywords(original_term) if shuffle_keywords_option else original_term
            search_term = optimize_search_term(search_term, 'english' if optimize_english else ('spanish' if optimize_spanish else None)) if optimize_english or optimize_spanish else search_term
            update_log(log_container, f"Searching for '{original_term}' (Used '{search_term}')")
            
            for url in google_search(search_term, num_results, lang=language):
                domain = get_domain_from_url(url)
                if ignore_previously_fetched and domain in domains_processed:
                    # update_log(log_container, f"Skipping Previously Fetched: {domain}", 'warning')
                    continue
                
                # update_log(log_container, f"Fetching: {url}") # Remove verbose logging
                try:
                    if not url.startswith(('http://', 'https://')):
                        url = 'http://' + url
                    
                    response = requests.get(url, timeout=10, verify=False, headers={'User-Agent': ua.random})
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract all emails from the page
                    valid_emails = [email for email in extract_emails_from_html(response.text) if is_valid_email(email)]
                    # update_log(log_container, f"Found {len(valid_emails)} valid email(s) on {url}", 'success') # Remove verbose logging
                    
                    if not valid_emails:
                        continue
                        
                    # Extract page info once for all leads from this URL
                    name, company, job_title = extract_info_from_page(soup)
                    page_title = get_page_title(response.text)
                    page_description = get_page_description(response.text)
                    
                    # Initialize set for this domain if not exists
                    if domain not in processed_emails_per_domain:
                        processed_emails_per_domain[domain] = set()
                    
                    # Process each valid email from this URL
                    for email in valid_emails:
                        # Skip if we've already processed this email for this domain
                        if email in processed_emails_per_domain[domain]:
                            continue
                            
                        processed_emails_per_domain[domain].add(email)
                        
                        lead = save_lead(session, email=email, first_name=name, company=company, job_title=job_title, url=url, search_term_id=search_term_id, created_at=datetime.utcnow())
                        if lead:
                            total_leads += 1
                            results.append({
                                'Email': email,
                                'URL': url,
                                'Lead Source': original_term,
                                'Title': page_title,
                                'Description': page_description,
                                'Tags': [],
                                'Name': name,
                                'Company': company,
                                'Job Title': job_title,
                                'Search Term ID': search_term_id
                            })
                            update_log(log_container, f"Saved lead: {email}", 'success')

                            if enable_email_sending:
                                if not from_email or not email_template:
                                    update_log(log_container, "Email sending is enabled but from_email or email_template is not provided", 'error')
                                    continue

                                template = session.query(EmailTemplate).filter_by(id=int(email_template.split(":")[0])).first()
                                if not template:
                                    update_log(log_container, "Email template not found", 'error')
                                    continue

                                wrapped_content = wrap_email_body(template.body_content)
                                response, tracking_id = send_email_ses(session, from_email, email, template.subject, wrapped_content, reply_to=reply_to)
                                if response:
                                    update_log(log_container, f"Sent email to: {email}", 'email_sent')
                                    save_email_campaign(session, email, template.id, 'Sent', datetime.utcnow(), template.subject, response['MessageId'], wrapped_content)
                                else:
                                    update_log(log_container, f"Failed to send email to: {email}", 'error')
                                    save_email_campaign(session, email, template.id, 'Failed', datetime.utcnow(), template.subject, None, wrapped_content)
                    
                    # Add domain to processed list after processing all its emails
                    domains_processed.add(domain)
                    
                except requests.RequestException as e:
                    update_log(log_container, f"Error processing URL {url}: {str(e)}", 'error')
        except Exception as e:
            update_log(log_container, f"Error processing term '{original_term}': {str(e)}", 'error')
    
    update_log(log_container, f"Total leads found: {total_leads}", 'info')
    return {"total_leads": total_leads, "results": results}

def generate_or_adjust_email_template(prompt, kb_info=None, current_template=None):
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in creating and refining email templates for marketing campaigns. Always respond with a JSON object containing 'subject' and 'body' keys. The 'body' should contain HTML formatted content suitable for insertion into an email body."},
        {"role": "user", "content": f"""{'Adjust the following email template based on the given instructions:' if current_template else 'Create an email template based on the following prompt:'} {prompt}

        {'Current Template:' if current_template else 'Guidelines:'}
                            save_email_campaign(session, result['Email'], template.id, 'sent', datetime.utcnow(), template.subject, response.get('MessageId', 'Unknown'), wrapped_content)

        Respond with a JSON object containing 'subject' and 'body' keys. The 'body' should contain HTML formatted content suitable for insertion into an email body.

        Follow these guidelines:
        1. Use proper HTML tags for structuring the email content (e.g., <p>, <h1>, <h2>, etc.).
        2. Include inline CSS for styling where appropriate.
        3. Ensure the content is properly structured and easy to read.
        4. Include a call-to-action button or link with appropriate styling.
        5. Make the design responsive for various screen sizes.
        6. Do not include <html>, <head>, or <body> tags.

        Example structure:
        {{
          "subject": "Your compelling subject line here",
          "body": "<h1>Welcome!</h1><p>Your email content here...</p><a href='#' style='display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px;'>Call to Action</a>"
        }}"""}
    ]
    if kb_info:
        messages.append({"role": "user", "content": f"Consider this knowledge base information: {json.dumps(kb_info)}"})

    response = openai_chat_completion(messages, function_name="generate_or_adjust_email_template")
    
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "subject": "AI Generated Subject",
                "body": f"<p>{response}</p>"
            }
    elif isinstance(response, dict):
        return response
    else:
        return {"subject": "", "body": "<p>Failed to generate email content.</p>"}

def fetch_leads_with_sources(session):
    try:
        query = session.query(Lead, func.string_agg(LeadSource.url, ', ').label('sources'), func.max(EmailCampaign.sent_at).label('last_contact'), func.string_agg(EmailCampaign.status, ', ').label('email_statuses')).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(Lead.id)
        return pd.DataFrame([{**{k: getattr(lead, k) for k in ['id', 'email', 'first_name', 'last_name', 'company', 'job_title', 'created_at']}, 'Source': sources, 'Last Contact': last_contact, 'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted', 'Delete': False} for lead, sources, last_contact, email_statuses in query.all()])
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_leads_with_sources: {str(e)}")
        return pd.DataFrame()

def fetch_search_terms_with_lead_count(session):
    query = session.query(SearchTerm.term, func.count(distinct(Lead.id)).label('lead_count'), func.count(distinct(EmailCampaign.id)).label('email_count')).join(LeadSource, SearchTerm.id == LeadSource.search_term_id).join(Lead, LeadSource.lead_id == Lead.id).outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id).group_by(SearchTerm.term)
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

def add_search_term(session, term, campaign_id):
    try:
        new_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=datetime.utcnow())
        session.add(new_term)
        session.commit()
        return new_term.id
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")
        raise

def update_search_term_group(session, group_id, updated_terms):
    try:
        current_term_ids = set(int(term.split(":")[0]) for term in updated_terms)
        existing_terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all()
        for term in existing_terms:
            term.group_id = None if term.id not in current_term_ids else group_id
        for term_str in updated_terms:
            term = session.query(SearchTerm).get(int(term_str.split(":")[0]))
            if term: term.group_id = group_id
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_search_term_group: {str(e)}")

def add_new_search_term(session, term, campaign_id, group_id):
    try:
        group_id = int(group_id.split(":")[0]) if group_id != "None" else None
        new_term = SearchTerm(term=term, campaign_id=campaign_id, group_id=group_id, created_at=datetime.utcnow())
        session.add(new_term)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")
        raise

def ai_group_search_terms(session, ungrouped_terms):
    # Implement the logic to group search terms using AI
    # This is just a placeholder example
    grouped_terms = {}
    for term in ungrouped_terms:
        group_name = f"Group {random.randint(1, 3)}"
        if group_name not in grouped_terms:
            grouped_terms[group_name] = []
        grouped_terms[group_name].append(term)
    return grouped_terms

def update_search_term_groups(session, grouped_terms):
    for group_name, terms in grouped_terms.items():
        group = session.query(SearchTermGroup).filter_by(name=group_name).first()
        if not group:
            group = SearchTermGroup(name=group_name)
            session.add(group)
            session.flush()
        for term in terms:
            term.group_id = group.id
    session.commit()

def create_search_term_group(session, group_name):
    try:
        session.add(SearchTermGroup(name=group_name))
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating search term group: {str(e)}")

def delete_search_term_group(session, group_id):
    try:
        group = session.query(SearchTermGroup).get(group_id)
        if group:
            session.query(SearchTerm).filter(SearchTerm.group_id == group_id).update({SearchTerm.group_id: None})
            session.delete(group)
            session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting search term group: {str(e)}")

def ai_automation_loop(session, log_container, leads_container):
    """Run automated search and email sending loop."""
    automation_logs, total_search_terms, total_emails_sent = [], 0, 0
    
    try:
        # Initialize automation log entry
        automation_log = AutomationLog(
            campaign_id=get_active_campaign_id(),
            status='running',
            start_time=datetime.utcnow()
        )
        session.add(automation_log)
        session.commit()
        
        while st.session_state.get('automation_status', False):
            try:
                log_container.info("Starting automation cycle")
                
                # Get knowledge base info
                kb_info = get_knowledge_base_info(session, get_active_project_id())
                if not kb_info:
                    message = "Knowledge Base not found. Skipping cycle."
                    log_container.warning(message)
                    automation_logs.append({"message": message, "level": "warning"})
                    time.sleep(3600)
                    continue

                # Get and optimize search terms
                base_terms = [term.term for term in session.query(SearchTerm)
                            .filter_by(project_id=get_active_project_id()).all()]
                
                if not base_terms:
                    message = "No search terms found. Please add some search terms first."
                    log_container.warning(message)
                    automation_logs.append({"message": message, "level": "warning"})
                    time.sleep(3600)
                    continue
                
                optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
                total_search_terms = len(optimized_terms)
                
                # Display optimized terms
                st.subheader("Optimized Search Terms")
                st.write(", ".join(optimized_terms))
                
                # Initialize progress tracking
                progress_bar = st.progress(0)
                new_leads_all = []
                
                # Process each search term
                for idx, term in enumerate(optimized_terms):
                    try:
                        # Perform search
                        results = manual_search(session, [term], 10, ignore_previously_fetched=True)
                        
                        # Process results and save leads
                        new_leads = []
                        for res in results.get('results', []):
                            lead = save_lead(session, res['Email'], url=res.get('URL'))
                            if lead:
                                new_leads.append((lead.id, lead.email))
                                new_leads_all.append((lead.email, res.get('URL', '')))
                        
                        # Send emails if we have leads and a template
                        if new_leads:
                            template = session.query(EmailTemplate)\
                                .filter_by(project_id=get_active_project_id())\
                                .first()
                            
                            if template:
                                from_email = kb_info.get('contact_email', 'hello@indosy.com')
                                reply_to = kb_info.get('contact_email', 'eugproductions@gmail.com')
                                
                                logs, sent_count = bulk_send_emails(
                                    session, 
                                    template.id,
                                    from_email,
                                    reply_to,
                                    [{'Email': email} for _, email in new_leads]
                                )
                                
                                automation_logs.extend(logs)
                                total_emails_sent += sent_count
                        
                        # Update progress and display
                        progress_bar.progress((idx + 1) / total_search_terms)
                        if new_leads:
                            leads_container.text_area(
                                "New Leads Found", 
                                "\n".join([email for _, email in new_leads]),
                                height=200
                            )
                            
                    except Exception as term_error:
                        error_msg = f"Error processing term '{term}': {str(term_error)}"
                        log_container.error(error_msg)
                        automation_logs.append({"message": error_msg, "level": "error"})
                        continue
                
                # Display cycle summary
                success_msg = f"Automation cycle completed. Terms: {total_search_terms}, Emails: {total_emails_sent}"
                st.success(success_msg)
                automation_logs.append({"message": success_msg, "level": "success"})
                
                # Update automation log
                automation_log.leads_gathered = len(new_leads_all)
                automation_log.emails_sent = total_emails_sent
                automation_log.logs = automation_logs
                session.commit()
                
                # Wait before next cycle
                time.sleep(3600)
                
            except Exception as cycle_error:
                error_msg = f"Critical error in automation cycle: {str(cycle_error)}"
                log_container.error(error_msg)
                automation_logs.append({"message": error_msg, "level": "error"})
                time.sleep(300)
        
        # Update final status
        automation_log.status = 'completed'
        automation_log.end_time = datetime.utcnow()
        session.commit()
        
        log_container.info("Automation stopped")
        
        # Update session state
        st.session_state.update({
            "automation_logs": automation_logs,
            "total_leads_found": total_search_terms,
            "total_emails_sent": total_emails_sent
        })
        
    except Exception as e:
        error_msg = f"Fatal error in automation loop: {str(e)}"
        log_container.error(error_msg)
        if 'automation_log' in locals():
            automation_log.status = 'failed'
            automation_log.end_time = datetime.utcnow()
            automation_log.logs = automation_logs + [{"message": error_msg, "level": "error"}]
            session.commit()

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        # Try to get AI settings first, fall back to general settings if needed
        ai_settings = session.query(Settings).filter_by(setting_type='ai').first()
        if ai_settings and ai_settings.value:
            settings = ai_settings.value
        else:
            general_settings = session.query(Settings).filter_by(setting_type='general').first()
            if not general_settings or 'openai_api_key' not in general_settings.value:
                st.error("AI settings not configured. Please configure them in the settings page.")
                return None
            settings = general_settings.value

        # Initialize OpenAI client with proper settings
        client = OpenAI(
            api_key=settings.get('api_key') or settings.get('openai_api_key'),
            base_url=settings.get('api_base_url', 'https://api.openai.com/v1')
        )
        model = settings.get('model_name') or settings.get('openai_model', "gpt-4")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=settings.get('max_tokens', 1500)
        )
        result = response.choices[0].message.content
        with db_session() as session:
            log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id, model)
        
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    except Exception as e:
        st.error(f"Error in OpenAI API call: {str(e)}")
        with db_session() as session:
            log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id, model)
        return None

def log_ai_request(session, function_name, prompt, response, lead_id=None, email_campaign_id=None, model_used=None):
    session.add(AIRequestLog(
        function_name=function_name,
        prompt=json.dumps(prompt),
        response=json.dumps(response) if response else None,
        lead_id=lead_id,
        email_campaign_id=email_campaign_id,
        model_used=model_used
    ))
    session.commit()

def save_lead(session, email, first_name=None, last_name=None, company=None, job_title=None, phone=None, url=None, search_term_id=None, created_at=None):
    try:
        existing_lead = session.query(Lead).filter_by(email=email).first()
        if existing_lead:
            for attr in ['first_name', 'last_name', 'company', 'job_title', 'phone', 'created_at']:
                if locals()[attr]: setattr(existing_lead, attr, locals()[attr])
            lead = existing_lead
        else:
            lead = Lead(email=email, first_name=first_name, last_name=last_name, company=company, job_title=job_title, phone=phone, created_at=created_at or datetime.utcnow())
            session.add(lead)
        session.flush()
        lead_source = LeadSource(lead_id=lead.id, url=url, search_term_id=search_term_id)
        session.add(lead_source)
        campaign_lead = CampaignLead(campaign_id=get_active_campaign_id(), lead_id=lead.id, status="Not Contacted", created_at=datetime.utcnow())
        session.add(campaign_lead)
        session.commit()
        return lead
    except Exception as e:
        logging.error(f"Error saving lead: {str(e)}")
        session.rollback()
        return None

def save_lead_source(session, lead_id, search_term_id, url, http_status, scrape_duration, page_title=None, meta_description=None, content=None, tags=None, phone_numbers=None):
    session.add(LeadSource(lead_id=lead_id, search_term_id=search_term_id, url=url, http_status=http_status, scrape_duration=scrape_duration, page_title=page_title or get_page_title(url), meta_description=meta_description or get_page_description(url), content=content or extract_visible_text(BeautifulSoup(requests.get(url).text, 'html.parser')), tags=tags, phone_numbers=phone_numbers))
    session.commit()

def get_page_title(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "No title found"
        return title.strip()
    except Exception as e:
        logging.error(f"Error getting page title for {url}: {str(e)}")
        return "Error fetching title"

def extract_visible_text(soup):
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return ' '.join(chunk for chunk in chunks if chunk)

def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
    session.add(SearchTermEffectiveness(term=term, total_results=total_results, valid_leads=valid_leads, irrelevant_leads=total_results - valid_leads, blogs_found=blogs_found, directories_found=directories_found))
    session.commit()

get_active_project_id = lambda: st.session_state.get('active_project_id', 1)
get_active_campaign_id = lambda: st.session_state.get('active_campaign_id', 1)
set_active_project_id = lambda project_id: st.session_state.__setitem__('active_project_id', project_id)
set_active_campaign_id = lambda campaign_id: st.session_state.__setitem__('active_campaign_id', campaign_id)

def add_or_get_search_term(session, term, campaign_id, created_at=None):
    search_term = session.query(SearchTerm).filter_by(term=term, campaign_id=campaign_id).first()
    if not search_term:
        search_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=created_at or datetime.utcnow())
        session.add(search_term)
        session.commit()
        session.refresh(search_term)
    return search_term.id

def fetch_campaigns(session):
    return [f"{camp.id}: {camp.campaign_name}" for camp in session.query(Campaign).all()]

def fetch_projects(session):
    return [f"{project.id}: {project.project_name}" for project in session.query(Project).all()]

def fetch_email_templates(session):
    return [f"{t.id}: {t.template_name}" for t in session.query(EmailTemplate).all()]

def create_or_update_email_template(session, template_name, subject, body_content, template_id=None, is_ai_customizable=False, created_at=None, language='ES'):
    # Ensure campaign_id is an integer
    campaign_id = int(get_active_campaign_id())
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=campaign_id, created_at=created_at or datetime.utcnow())
    if template_id: template.template_name, template.subject, template.body_content, template.is_ai_customizable = template_name, subject, body_content, is_ai_customizable
    template.language = language
    session.add(template)
    session.commit()
    return template.id

safe_datetime_compare = lambda date1, date2: False if date1 is None or date2 is None else date1 > date2

def fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted):
    try:
        query = session.query(Lead)
        if send_option == "Specific Email":
            query = query.filter(Lead.email == specific_email)
        elif send_option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and selected_terms:
            # Fetch SearchTerm IDs based on the selected terms
            search_term_ids = [
                term.id
                for term in session.query(SearchTerm.id)
                .filter(SearchTerm.term.in_(selected_terms))
                .all()
            ]
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.id.in_(search_term_ids))
        
        if exclude_previously_contacted:
            subquery = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None)).subquery()
            query = query.outerjoin(subquery, Lead.id == subquery.c.lead_id).filter(subquery.c.lead_id.is_(None))
        
        return [{"Email": lead.email, "ID": lead.id} for lead in query.all()]
    except Exception as e:
        logging.error(f"Error fetching leads: {str(e)}")
        return []

def update_display(container, items, title, item_key):
    container.markdown(
        f"""
        <style>
            .container {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
            background-color: rgba(49, 51, 63, 0.1);
        }}
        .entry {{
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 0.25rem;
            }}
        </style>
        <div class="container">
            <h4>{title} ({len(items)})</h4>
            {"".join(f'<div class="entry">{item[item_key]}</div>' for item in items[-20:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def get_domain_from_url(url): return urlparse(url).netloc

def display_search_settings():
    """Display and collect search settings from the user."""
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search terms input
        search_terms = st_tags(
            label='Enter search terms:',
            text='Press enter to add more',
            suggestions=['software engineer', 'data scientist', 'product manager'],
            maxtags=10,
            key='search_terms_input'
        )
        
        # Results per term slider
        num_results = st.slider(
            "Results per term", 
            min_value=1, 
            max_value=50000, 
            value=10,
            help="Number of results to fetch per search term"
        )

    with col2:
        st.subheader("Search Options")
        
        # Main options
        enable_email_sending = st.checkbox(
            "Enable email sending", 
            value=True,
            help="Automatically send emails to found leads"
        )
        
        ignore_previously_fetched = st.checkbox(
            "Ignore fetched domains", 
            value=True,
            help="Skip domains that have been previously searched"
        )
        
        # Advanced options in an expander
        with st.expander("Advanced Options", expanded=False):
            shuffle_keywords_option = st.checkbox(
                "Shuffle Keywords", 
                value=True,
                help="Randomly reorder keywords for better results"
            )
            
            optimize_english = st.checkbox(
                "Optimize (English)", 
                value=False,
                help="Optimize search terms for English results"
            )
            
            optimize_spanish = st.checkbox(
                "Optimize (Spanish)", 
                value=False,
                help="Optimize search terms for Spanish results"
            )
            
            language = st.selectbox(
                "Select Language", 
                options=["ES", "EN"], 
                index=0,
                help="Choose search language"
            )

    # Return all settings as a dictionary
    return {
        'search_terms': search_terms,
        'num_results': num_results,
        'enable_email_sending': enable_email_sending,
        'ignore_previously_fetched': ignore_previously_fetched,
        'shuffle_keywords_option': shuffle_keywords_option,
        'optimize_english': optimize_english,
        'optimize_spanish': optimize_spanish,
        'language': language
    }

def display_email_options(email_settings, email_templates):
    """Display and handle email settings and template selection."""
    if not email_settings or not email_templates:
        return None, None

    # Create columns for email settings
    col1, col2 = st.columns(2)
    
    with col1:
        # Email template selection - Fix for template object handling
        template_options = [{'id': t['id'] if isinstance(t, dict) else t.id, 
                           'name': t['template_name'] if isinstance(t, dict) else t.template_name} 
                          for t in email_templates]
        selected_template = st.selectbox(
            "Email Template",
            options=template_options,
            format_func=lambda x: x['name'],
            key='email_template_select'
        )
        selected_template_id = selected_template['id'] if selected_template else None

    with col2:
        # Email settings selection - Fix for settings object handling
        setting_options = [{'id': s['id'] if isinstance(s, dict) else s.id, 
                          'name': f"{s['name'] if isinstance(s, dict) else s.name} ({s['email'] if isinstance(s, dict) else s.email})"} 
                         for s in email_settings]
        selected_setting = st.selectbox(
            "From Email",
            options=setting_options,
            format_func=lambda x: x['name'],
            key='email_setting_select'
        )
        selected_setting_id = selected_setting['id'] if selected_setting else None

    return selected_setting_id, selected_template_id

def manual_search_page():
    """Page for manual search and email sending."""
    st.title("🔍 Manual Search")

    # Initialize search state if needed
    if 'search_state' not in st.session_state:
        st.session_state.search_state = {
            'is_searching': False,
            'current_term_index': 0,
            'total_terms': 0,
            'results_cache': [],
            'last_search_params': None,
            'background_processes': {},
            'last_update': time.time()
        }

    with db_session() as session:
        project_id = get_active_project_id()
        if project_id is None:
            st.warning("Please select a project first.")
            return

        campaign_id = get_active_campaign_id()
        if campaign_id is None:
            st.warning("Please select a campaign first.")
            return

        # Show active processes first
        active_processes = session.query(AutomationLog).filter(
            AutomationLog.status.in_(['running', 'completed'])
        ).order_by(AutomationLog.start_time.desc()).limit(5).all()
        
        if active_processes:
            st.subheader("Active Search Processes")
            for process in active_processes:
                with st.expander(
                    f"Process {process.id} - {process.status.title()} - "
                    f"Started: {process.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    expanded=True
                ):
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Status", process.status.title())
                    col2.metric("Leads Found", process.leads_gathered or 0)
                    col3.metric("Emails Sent", process.emails_sent or 0)
                    
                    # Display logs
                    if process.logs:
                        for log in process.logs:
                            level = log.get('level', 'info')
                            icon = {
                                'info': 'ℹ️',
                                'success': '✅',
                                'warning': '⚠️',
                                'error': '❌',
                                'email': '📧'
                            }.get(level, 'ℹ️')
                            st.write(f"{icon} {log.get('message', '')}")

        # Fetch email settings and templates only once
        email_settings = fetch_email_settings(session)
        email_templates = fetch_email_templates(session)

        # Display search settings
        search_settings = display_search_settings()

        # Display email settings and template selection if email sending is enabled
        selected_email_setting_id = None
        selected_template_id = None
        if search_settings['enable_email_sending']:
            if not email_templates:
                st.error("No email templates available. Please create a template first.")
                return
            if not email_settings:
                st.error("No email settings available. Please add email settings first.")
                return
            selected_email_setting_id, selected_template_id = display_email_options(email_settings, email_templates)

        # Create containers for dynamic content
        status_container = st.empty()
        progress_container = st.empty()
        results_container = st.empty()
        log_container = st.empty()

        # Search button and results
        if st.button("Start Search", type="primary"):
            if not search_settings['search_terms']:
                st.error("Please enter at least one search term.")
                return

            if search_settings['enable_email_sending'] and (not selected_email_setting_id or not selected_template_id):
                st.error("Please select both an email setting and an email template.")
                return

            # Create automation log
            automation_log = AutomationLog(
                campaign_id=campaign_id,
                start_time=datetime.now(),
                status='running',
                leads_gathered=0,
                emails_sent=0,
                logs=[{
                    'timestamp': datetime.now().isoformat(),
                    'level': 'info',
                    'message': 'Starting search process'
                }]
            )
            session.add(automation_log)
            session.commit()

            try:
                with st.spinner('Searching...'):
                    # Perform the search
                    search_results = perform_search(
                        session,
                        search_settings['search_terms'],
                        project_id,
                        campaign_id,
                        search_settings['num_results'],
                        search_settings['language'],
                        search_settings['ignore_previously_fetched'],
                        search_settings['optimize_english'],
                        search_settings['optimize_spanish'],
                        search_settings['shuffle_keywords_option']
                    )

                    # Display search results
                    if search_results:
                        results_df = pd.DataFrame(search_results)
                        results_container.dataframe(results_df)

                        # Process and display new leads
                        new_leads_all = []
                        for result in search_results:
                            new_leads = process_search_result(session, result, project_id, campaign_id)
                            new_leads_all.extend(new_leads)

                        if new_leads_all:
                            st.subheader("New Leads")
                            leads_df = pd.DataFrame(new_leads_all)
                            st.dataframe(leads_df)

                            # Update automation log
                            automation_log.leads_gathered = len(new_leads_all)
                            automation_log.logs.append({
                                'timestamp': datetime.now().isoformat(),
                                'level': 'success',
                                'message': f'Found {len(new_leads_all)} new leads'
                            })

                            # Send emails if enabled
                            if search_settings['enable_email_sending']:
                                email_setting = next((s for s in email_settings if s.id == selected_email_setting_id), None)
                                template = next((t for t in email_templates if t.id == selected_template_id), None)

                                if email_setting and template:
                                    email_limit_check = check_email_limits(session, email_setting)
                                    if email_limit_check['can_send']:
                                        with st.spinner('Sending emails...'):
                                            logs, sent_count = bulk_send_emails(
                                                session,
                                                template.id,
                                                email_setting.email,
                                                email_setting.email,  # Using same email as reply-to
                                                new_leads_all,
                                                progress_container,
                                                status_container,
                                                results_container,
                                                log_container
                                            )
                                            automation_log.emails_sent = sent_count
                                            automation_log.logs.extend([{
                                                'timestamp': datetime.now().isoformat(),
                                                'level': 'email',
                                                'message': log
                                            } for log in logs])
                                    else:
                                        st.error(email_limit_check['message'])
                                        automation_log.logs.append({
                                            'timestamp': datetime.now().isoformat(),
                                            'level': 'error',
                                            'message': email_limit_check['message']
                                        })
                        else:
                            st.info("No new leads found.")
                            automation_log.logs.append({
                                'timestamp': datetime.now().isoformat(),
                                'level': 'info',
                                'message': 'No new leads found'
                            })
                    else:
                        st.info("No results found for the given search terms.")
                        automation_log.logs.append({
                            'timestamp': datetime.now().isoformat(),
                            'level': 'info',
                            'message': 'No results found'
                        })

            except Exception as e:
                error_msg = f"An error occurred during the search: {str(e)}"
                st.error(error_msg)
                automation_log.logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'level': 'error',
                    'message': error_msg
                })
            finally:
                # Update automation log status
                automation_log.status = 'completed'
                automation_log.end_time = datetime.now()
                session.commit()

def update_lead(session, lead_id, updated_data):
    try:
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            for key, value in updated_data.items():
                setattr(lead, key, value)
            return True
    except SQLAlchemyError as e:
        logging.error(f"Error updating lead {lead_id}: {str(e)}")
        session.rollback()
    return False

def delete_lead(session, lead_id):
    try:
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            session.delete(lead)
            return True
    except SQLAlchemyError as e:
        logging.error(f"Error deleting lead {lead_id}: {str(e)}")
        session.rollback()
    return False

def is_valid_email(email):
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False



def view_leads_page():
    st.title("Lead Management Dashboard")
    with db_session() as session:
        if 'leads' not in st.session_state or st.button("Refresh Leads"):
            st.session_state.leads = fetch_leads_with_sources(session)
        if not st.session_state.leads.empty:
            total_leads = len(st.session_state.leads)
            contacted_leads = len(st.session_state.leads[st.session_state.leads['Last Contact'].notna()])
            conversion_rate = (st.session_state.leads['Last Email Status'] == 'sent').mean()

            st.columns(3)[0].metric("Total Leads", f"{total_leads:,}")
            st.columns(3)[1].metric("Contacted Leads", f"{contacted_leads:,}")
            st.columns(3)[2].metric("Conversion Rate", f"{conversion_rate:.2%}")

            st.subheader("Leads Table")
            search_term = st.text_input("Search leads by email, name, company, or source")
            filtered_leads = st.session_state.leads[st.session_state.leads.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]

            leads_per_page, page_number = 20, st.number_input("Page", min_value=1, value=1)
            start_idx, end_idx = (page_number - 1) * leads_per_page, page_number * leads_per_page

            edited_df = st.data_editor(
                filtered_leads.iloc[start_idx:end_idx],
                column_config={
                    "ID": st.column_config.NumberColumn("ID", disabled=True),
                    "Email": st.column_config.TextColumn("Email"),
                    "First Name": st.column_config.TextColumn("First Name"),
                    "Last Name": st.column_config.TextColumn("Last Name"),
                    "Company": st.column_config.TextColumn("Company"),
                    "Job Title": st.column_config.TextColumn("Job Title"),
                    "Source": st.column_config.TextColumn("Source", disabled=True),
                    "Last Contact": st.column_config.DatetimeColumn("Last Contact", disabled=True),
                    "Last Email Status": st.column_config.TextColumn("Last Email Status", disabled=True),
                    "Delete": st.column_config.CheckboxColumn("Delete")
                },
                disabled=["ID", "Source", "Last Contact", "Last Email Status"],
                hide_index=True,
                num_rows="dynamic"
            )

            if st.button("Save Changes", type="primary"):
                for index, row in edited_df.iterrows():
                    if row['Delete']:
                        if delete_lead_and_sources(session, row['ID']):
                            st.success(f"Deleted lead: {row['Email']}")
                    else:
                        updated_data = {k: row[k] for k in ['Email', 'First Name', 'Last Name', 'Company', 'Job Title']}
                        if update_lead(session, row['ID'], updated_data):
                            st.success(f"Updated lead: {row['Email']}")
                st.rerun()

            st.download_button(
                "Export Filtered Leads to CSV",
                filtered_leads.to_csv(index=False).encode('utf-8'),
                "exported_leads.csv",
                "text/csv"
            )

            st.subheader("Lead Growth")
            if 'Created At' in st.session_state.leads.columns:
                lead_growth = st.session_state.leads.groupby(pd.to_datetime(st.session_state.leads['Created At']).dt.to_period('M')).size().cumsum()
                st.line_chart(lead_growth)
            else:
                st.warning("Created At data is not available for lead growth chart.")

            st.subheader("Email Campaign Performance")
            email_status_counts = st.session_state.leads['Last Email Status'].value_counts()
            st.plotly_chart(px.pie(
                values=email_status_counts.values,
                names=email_status_counts.index,
                title="Distribution of Email Statuses"
            ), use_container_width=True)
        else:
            st.info("No leads available. Start by adding some leads to your campaigns.")

def fetch_leads_with_sources(session):
    try:
        query = session.query(Lead, func.string_agg(LeadSource.url, ', ').label('sources'), func.max(EmailCampaign.sent_at).label('last_contact'), func.string_agg(EmailCampaign.status, ', ').label('email_statuses')).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(Lead.id)
        return pd.DataFrame([{**{k: getattr(lead, k) for k in ['id', 'email', 'first_name', 'last_name', 'company', 'job_title', 'created_at']}, 'Source': sources, 'Last Contact': last_contact, 'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted', 'Delete': False} for lead, sources, last_contact, email_statuses in query.all()])
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_leads_with_sources: {str(e)}")
        return pd.DataFrame()

def delete_lead_and_sources(session, lead_id):
    try:
        session.query(LeadSource).filter(LeadSource.lead_id == lead_id).delete()
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            session.delete(lead)
            return True
    except SQLAlchemyError as e:
        logging.error(f"Error deleting lead {lead_id} and its sources: {str(e)}")
        session.rollback()
    return False

def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term, 
                           func.count(distinct(Lead.id)).label('lead_count'),
                           func.count(distinct(EmailCampaign.id)).label('email_count'))
             .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
             .join(Lead, LeadSource.lead_id == Lead.id)
             .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
             .group_by(SearchTerm.term))
    df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    return df

def add_search_term(session, term, campaign_id):
    try:
        new_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=datetime.utcnow())
        session.add(new_term)
        session.commit()
        return new_term.id
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")
        raise

def get_active_campaign_id():
    return st.session_state.get('active_campaign_id', 1)

def search_terms_page():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Search Terms Dashboard</h1>", unsafe_allow_html=True)
    with db_session() as session:
        search_terms_df = fetch_search_terms_with_lead_count(session)
        if not search_terms_df.empty:
            st.columns(3)[0].metric("Total Search Terms", len(search_terms_df))
            st.columns(3)[1].metric("Total Leads", search_terms_df['Lead Count'].sum())
            st.columns(3)[2].metric("Total Emails Sent", search_terms_df['Email Count'].sum())
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Search Term Groups", "Performance", "Add New Term", "AI Grouping", "Manage Groups"])
            
            with tab1:
                groups = session.query(SearchTermGroup).all()
                groups.append("Ungrouped")
                for group in groups:
                    with st.expander(group.name if isinstance(group, SearchTermGroup) else group, expanded=True):
                        group_id = group.id if isinstance(group, SearchTermGroup) else None
                        terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all() if group_id else session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
                        updated_terms = st_tags(
                            label="",
                            text="Add or remove terms",
                            value=[f"{term.id}: {term.term}" for term in terms],
                            suggestions=[term for term in search_terms_df['Term'] if term not in [f"{t.id}: {t.term}" for t in terms]],
                            key=f"group_{group_id}"
                        )
                        if st.button("Update", key=f"update_{group_id}"):
                            update_search_term_group(session, group_id, updated_terms)
                            st.success("Group updated successfully")
                            st.rerun()
            
            with tab2:
                col1, col2 = st.columns([3, 1])
                with col1:
                    chart_type = st.radio("Chart Type", ["Bar", "Pie"], horizontal=True)
                    fig = px.bar(search_terms_df.nlargest(10, 'Lead Count'), x='Term', y=['Lead Count', 'Email Count'], title='Top 10 Search Terms', labels={'value': 'Count', 'variable': 'Type'}, barmode='group') if chart_type == "Bar" else px.pie(search_terms_df, values='Lead Count', names='Term', title='Lead Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(search_terms_df.nlargest(5, 'Lead Count')[['Term', 'Lead Count', 'Email Count']], use_container_width=True)
            
            with tab3:
                col1, col2, col3 = st.columns([2,1,1])
                new_term = col1.text_input("New Search Term")
                campaign_id = get_active_campaign_id()
                group_for_new_term = col2.selectbox("Assign to Group", ["None"] + [f"{g.id}: {g.name}" for g in groups if isinstance(g, SearchTermGroup)], format_func=lambda x: x.split(":")[1] if ":" in x else x)
                if col3.button("Add Term", use_container_width=True) and new_term:
                    add_new_search_term(session, new_term, campaign_id, group_for_new_term)
                    st.success(f"Added: {new_term}")
                    st.rerun()

            with tab4:
                st.subheader("AI-Powered Search Term Grouping")
                ungrouped_terms = session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
                if ungrouped_terms:
                    st.write(f"Found {len(ungrouped_terms)} ungrouped search terms.")
                    if st.button("Group Ungrouped Terms with AI"):
                        with st.spinner("AI is grouping terms..."):
                            grouped_terms = ai_group_search_terms(session, ungrouped_terms)
                            update_search_term_groups(session, grouped_terms)
                            st.success("Search terms have been grouped successfully!")
                            st.rerun()
                else:
                    st.info("No ungrouped search terms found.")

            with tab5:
                st.subheader("Manage Search Term Groups")
                col1, col2 = st.columns(2)
                with col1:
                    new_group_name = st.text_input("New Group Name")
                    if st.button("Create New Group") and new_group_name:
                        create_search_term_group(session, new_group_name)
                        st.success(f"Created new group: {new_group_name}")
                        st.rerun()
                with col2:
                    group_to_delete = st.selectbox("Select Group to Delete", 
                                                   [f"{g.id}: {g.name}" for g in groups if isinstance(g, SearchTermGroup)],
                                                   format_func=lambda x: x.split(":")[1])
                    if st.button("Delete Group") and group_to_delete:
                        group_id = int(group_to_delete.split(":")[0])
                        delete_search_term_group(session, group_id)
                        st.success(f"Deleted group: {group_to_delete.split(':')[1]}")
                        st.rerun()

        else:
            st.info("No search terms available. Add some to your campaigns.")

def update_search_term_group(session, group_id, updated_terms):
    for term_id in [int(term.split(":")[0]) for term in updated_terms]:
        term = session.query(SearchTerm).get(term_id)
        term.group_id = group_id
    session.commit()

def add_new_search_term(session, term, campaign_id, group_id):
    try:
        group_id = int(group_id.split(":")[0]) if group_id != "None" else None
        new_term = SearchTerm(term=term, campaign_id=campaign_id, group_id=group_id, created_at=datetime.utcnow())
        session.add(new_term)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")
        raise

def ai_group_search_terms(session, ungrouped_terms):
    # Implement the logic to group search terms using AI
    # This is just a placeholder example
    grouped_terms = {}
    for term in ungrouped_terms:
        group_name = f"Group {random.randint(1, 3)}"
        if group_name not in grouped_terms:
            grouped_terms[group_name] = []
        grouped_terms[group_name].append(term)
    return grouped_terms

def update_search_term_groups(session, grouped_terms):
    for group_name, terms in grouped_terms.items():
        group = session.query(SearchTermGroup).filter_by(name=group_name).first()
        if not group:
            group = SearchTermGroup(name=group_name)
            session.add(group)
            session.flush()
        for term in terms:
            term.group_id = group.id
    session.commit()

def create_search_term_group(session, group_name):
    try:
        session.add(SearchTermGroup(name=group_name))
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating search term group: {str(e)}")

def delete_search_term_group(session, group_id):
    try:
        group = session.query(SearchTermGroup).get(group_id)
        if group:
            # Set group_id to None for all search terms in this group
            session.query(SearchTerm).filter(SearchTerm.group_id == group_id).update({SearchTerm.group_id: None})
            session.delete(group)
            session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting search term group: {str(e)}")

def email_templates_page():
    st.header("Email Templates")
    with db_session() as session:
        templates = session.query(EmailTemplate).all()
        with st.expander("Create New Template", expanded=False):
            new_template_name = st.text_input("Template Name", key="new_template_name")
            use_ai = st.checkbox("Use AI to generate template", key="use_ai")
            if use_ai:
                ai_prompt = st.text_area("Enter prompt for AI template generation", key="ai_prompt")
                use_kb = st.checkbox("Use Knowledge Base information", key="use_kb")
                if st.button("Generate Template", key="generate_ai_template"):
                    with st.spinner("Generating template..."):
                        kb_info = get_knowledge_base_info(session, get_active_project_id()) if use_kb else None
                        generated_template = generate_or_adjust_email_template(ai_prompt, kb_info)
                        new_template_subject = generated_template.get("subject", "AI Generated Subject")
                        new_template_body = generated_template.get("body", "")
                        
                        if new_template_name:
                            new_template = EmailTemplate(
                                template_name=new_template_name,
                                subject=new_template_subject,
                                body_content=new_template_body,
                                campaign_id=get_active_campaign_id()
                            )
                            session.add(new_template)
                            session.commit()
                            st.success("AI-generated template created and saved!")
                            templates = session.query(EmailTemplate).all()
                        else:
                            st.warning("Please provide a name for the template before generating.")
                        
                        st.subheader("Generated Template")
                        st.text(f"Subject: {new_template_subject}")
                        st.components.v1.html(wrap_email_body(new_template_body), height=400, scrolling=True)
            else:
                new_template_subject = st.text_input("Subject", key="new_template_subject")
                new_template_body = st.text_area("Body Content", height=200, key="new_template_body")

            if st.button("Create Template", key="create_template_button"):
                if all([new_template_name, new_template_subject, new_template_body]):
                    new_template = EmailTemplate(
                        template_name=new_template_name,
                        subject=new_template_subject,
                        body_content=new_template_body,
                        campaign_id=get_active_campaign_id()
                    )
                    session.add(new_template)
                    session.commit()
                    st.success("New template created successfully!")
                    templates = session.query(EmailTemplate).all()
                else:
                    st.warning("Please fill in all fields to create a new template.")

        if templates:
            st.subheader("Existing Templates")
            for template in templates:
                with st.expander(f"Template: {template.template_name}", expanded=False):
                    col1, col2 = st.columns(2)
                    edited_subject = col1.text_input("Subject", value=template.subject, key=f"subject_{template.id}")
                    is_ai_customizable = col2.checkbox("AI Customizable", value=template.is_ai_customizable, key=f"ai_{template.id}")
                    edited_body = st.text_area("Body Content", value=template.body_content, height=200, key=f"body_{template.id}")
                    
                    ai_adjustment_prompt = st.text_area("AI Adjustment Prompt", key=f"ai_prompt_{template.id}", placeholder="E.g., Make it less marketing-like and mention our main features")
                    
                    col3, col4 = st.columns(2)
                    if col3.button("Apply AI Adjustment", key=f"apply_ai_{template.id}"):
                        with st.spinner("Applying AI adjustment..."):
                            kb_info = get_knowledge_base_info(session, get_active_project_id())
                            adjusted_template = generate_or_adjust_email_template(ai_adjustment_prompt, kb_info, current_template=edited_body)
                            edited_subject = adjusted_template.get("subject", edited_subject)
                            edited_body = adjusted_template.get("body", edited_body)
                            st.success("AI adjustment applied. Please review and save changes.")
                    
                    if col4.button("Save Changes", key=f"save_{template.id}"):
                        template.subject = edited_subject
                        template.body_content = edited_body
                        template.is_ai_customizable = is_ai_customizable
                        session.commit()
                        st.success("Template updated successfully!")
                    
                    st.markdown("### Preview")
                    st.text(f"Subject: {edited_subject}")
                    st.components.v1.html(wrap_email_body(edited_body), height=400, scrolling=True)
                    
                    if st.button("Delete Template", key=f"delete_{template.id}"):
                        session.delete(template)
                        session.commit()
                        st.success("Template deleted successfully!")
                        st.rerun()
        else:
            st.info("No email templates found. Create a new template to get started.")

def get_email_preview(session, template_id, from_email, reply_to):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if template:
        wrapped_content = wrap_email_body(template.body_content)
        return wrapped_content
    return "<p>Template not found</p>"

def fetch_all_search_terms(session):
    return session.query(SearchTerm).all()

def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

def get_email_template_by_name(session, template_name):
    return session.query(EmailTemplate).filter_by(template_name=template_name).first()

def bulk_send_page():
    st.title("Bulk Email Sending")
    with db_session() as session:
        templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)
        if not templates or not email_settings:
            st.error("No email templates or settings available. Please set them up first.")
            return

        template_option = st.selectbox("Email Template", options=templates, format_func=lambda x: x.split(":")[1].strip())
        template_id = int(template_option.split(":")[0])
        template = session.query(EmailTemplate).filter_by(id=template_id).first()

        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input("Subject", value=template.subject if template else "")
            email_setting_option = st.selectbox("From Email", options=email_settings, format_func=lambda x: f"{x['name']} ({x['email']})")
            if email_setting_option:
                from_email = email_setting_option['email']
                reply_to = st.text_input("Reply To", email_setting_option['email'])
            else:
                st.error("Selected email setting not found. Please choose a valid email setting.")
                return

        with col2:
            send_option = st.radio("Send to:", ["All Leads", "Specific Email", "Leads from Chosen Search Terms", "Leads from Search Term Groups"])
            specific_email = None
            selected_terms = None
            if send_option == "Specific Email":
                specific_email = st.text_input("Enter email")
            elif send_option == "Leads from Chosen Search Terms":
                search_terms_with_counts = fetch_search_terms_with_lead_count(session)
                selected_terms = st.multiselect("Select Search Terms", options=search_terms_with_counts['Term'].tolist())
                selected_terms = [term.split(" (")[0] for term in selected_terms]
            elif send_option == "Leads from Search Term Groups":
                groups = fetch_search_term_groups(session)
                selected_groups = st.multiselect("Select Search Term Groups", options=groups)
                if selected_groups:
                    group_ids = [int(group.split(':')[0]) for group in selected_groups]
                    selected_terms = fetch_search_terms_for_groups(session, group_ids)

        exclude_previously_contacted = st.checkbox("Exclude Previously Contacted Domains", value=True)

        st.markdown("### Email Preview")
        st.text(f"From: {from_email}\nReply-To: {reply_to}\nSubject: {subject}")
        st.components.v1.html(get_email_preview(session, template_id, from_email, reply_to), height=600, scrolling=True)

        leads = fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted)
        total_leads = len(leads)
        eligible_leads = [lead for lead in leads if lead.get('language', template.language) == template.language]
        contactable_leads = [lead for lead in eligible_leads if not (exclude_previously_contacted and lead.get('domain_contacted', False))]

        st.info(f"Total leads: {total_leads}\n"
                f"Leads matching template language ({template.language}): {len(eligible_leads)}\n"
                f"Leads to be contacted: {len(contactable_leads)}")

        if st.button("Send Emails", type="primary"):
            if not contactable_leads:
                st.warning("No leads found matching the selected criteria.")
                return
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            log_container = st.empty()
            logs, sent_count = bulk_send_emails(session, template_id, from_email, reply_to, contactable_leads, progress_bar, status_text, results, log_container)
            st.success(f"Emails sent successfully to {sent_count} leads.")
            st.subheader("Sending Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            success_rate = (results_df['Status'] == 'sent').mean()
            st.metric("Email Sending Success Rate", f"{success_rate:.2%}")

def fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted):
    try:
        query = session.query(Lead)
        if send_option == "Specific Email":
            query = query.filter(Lead.email == specific_email)
        elif send_option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and selected_terms:
            # Fetch SearchTerm IDs based on the selected terms
            search_term_ids = [
                term.id
                for term in session.query(SearchTerm.id)
                .filter(SearchTerm.term.in_(selected_terms))
                .all()
            ]
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.id.in_(search_term_ids))
        
        if exclude_previously_contacted:
            subquery = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None)).subquery()
            query = query.outerjoin(subquery, Lead.id == subquery.c.lead_id).filter(subquery.c.lead_id.is_(None))
        
        return [{"Email": lead.email, "ID": lead.id} for lead in query.all()]
    except Exception as e:
        logging.error(f"Error fetching leads: {str(e)}")
        return []

def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        return []

def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term, 
                           func.count(distinct(Lead.id)).label('lead_count'),
                           func.count(distinct(EmailCampaign.id)).label('email_count'))
             .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
             .join(Lead, LeadSource.lead_id == Lead.id)
             .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
             .group_by(SearchTerm.term))
    df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    return df

def fetch_search_term_groups(session):
    return [f"{group.id}: {group.name}" for group in session.query(SearchTermGroup).all()]

def fetch_search_terms_for_groups(session, group_ids):
    terms = session.query(SearchTerm).filter(SearchTerm.group_id.in_(group_ids)).all()
    return [term.term for term in terms]

def ai_automation_loop(session, log_container, leads_container):
    automation_logs, total_search_terms, total_emails_sent = [], 0, 0
    while st.session_state.get('automation_status', False):
        try:
            log_container.info("Starting automation cycle")
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                log_container.warning("Knowledge Base not found. Skipping cycle.")
                time.sleep(3600)
                continue
            base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
            optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            st.subheader("Optimized Search Terms")
            st.write(", ".join(optimized_terms))

            total_search_terms = len(optimized_terms)
            progress_bar = st.progress(0)
            for idx, term in enumerate(optimized_terms):
                results = manual_search(session, [term], 10, ignore_previously_fetched=True)
                new_leads = []
                for res in results['results']:
                    lead = save_lead(session, res['Email'], url=res['URL'])
                    if lead:
                        new_leads.append((lead.id, lead.email))
                if new_leads:
                    template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                    if template:
                        from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                        reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                        logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': email} for _, email in new_leads])
                        automation_logs.extend(logs)
                        total_emails_sent += sent_count
                leads_container.text_area("New Leads Found", "\n".join([email for _, email in new_leads]), height=200)
                progress_bar.progress((idx + 1) / len(optimized_terms))
            st.success(f"Automation cycle completed. Total search terms: {total_search_terms}, Total emails sent: {total_emails_sent}")
            time.sleep(3600)
        except Exception as e:
            log_container.error(f"Critical error in automation cycle: {str(e)}")
            time.sleep(300)
    log_container.info("Automation stopped")
    st.session_state.automation_logs = automation_logs
    st.session_state.total_leads_found = total_search_terms
    st.session_state.total_emails_sent = total_emails_sent

def display_search_results(results, key_suffix):
    if not results: return st.warning("No results to display.")
    with st.expander("Search Results", expanded=True):
        st.markdown(f"### Total Leads Found: **{len(results)}**")
        for i, res in enumerate(results):
            with st.expander(f"Lead: {res['Email']}", key=f"lead_expander_{key_suffix}_{i}"):
                st.markdown(f"**URL:** [{res['URL']}]({res['URL']})  \n**Title:** {res['Title']}  \n**Description:** {res['Description']}  \n**Tags:** {', '.join(res['Tags'])}  \n**Lead Source:** {res['Lead Source']}  \n**Lead Email:** {res['Email']}")

def perform_quick_scan(session):
    with st.spinner("Performing quick scan..."):
        terms = session.query(SearchTerm).order_by(func.random()).limit(3).all()
        email_setting = fetch_email_settings(session)[0] if fetch_email_settings(session) else None
        from_email = email_setting['email'] if email_setting else None
        reply_to = from_email
        email_template = session.query(EmailTemplate).first()
        res = manual_search(session, [term.term for term in terms], 10, True, False, False, True, "EN", True, st.empty(), from_email, reply_to, f"{email_template.id}: {email_template.template_name}" if email_template else None)
    st.success(f"Quick scan completed! Found {len(res['results'])} new leads.")
    return {"new_leads": len(res['results']), "terms_used": [term.term for term in terms]}

def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"Optimize and expand these search terms for lead generation:\n{', '.join(base_terms)}\n\nConsider:\n1. Relevance to business and target market\n2. Potential for high-quality leads\n3. Variations and related terms\n4. Industry-specific jargon\n\nRespond with a JSON array of optimized terms."
    response = openai_chat_completion([{"role": "system", "content": "You're an AI specializing in optimizing search terms for lead generation. Be concise and effective."}, {"role": "user", "content": prompt}], function_name="generate_optimized_search_terms")
    return response.get('optimized_terms', base_terms) if isinstance(response, dict) else base_terms

def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term, 
                           func.count(distinct(Lead.id)).label('lead_count'),
                           func.count(distinct(EmailCampaign.id)).label('email_count'))
             .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
             .join(Lead, LeadSource.lead_id == Lead.id)
             .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
             .group_by(SearchTerm.term))
    df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    return df

def fetch_leads_for_search_terms(session, search_term_ids) -> List[Lead]:
    return session.query(Lead).distinct().join(LeadSource).filter(LeadSource.search_term_id.in_(search_term_ids)).all()

def projects_campaigns_page():
    with db_session() as session:
        st.header("Projects and Campaigns")
        
        # Get all projects
        projects = session.query(Project).all()
        
        # Set default project if none selected
        if not get_active_project_id() and projects:
            set_active_project_id(projects[0].id)
        
        # Get current project campaigns
        if get_active_project_id():
            campaigns = session.query(Campaign).filter_by(project_id=get_active_project_id()).all()
            if campaigns and not get_active_campaign_id():
                set_active_campaign_id(campaigns[0].id)

        # Project selection
        st.subheader("Set Active Project and Campaign")
        if projects:
            project_options = [(p.id, p.project_name) for p in projects]  # Changed from p.name to p.project_name
            default_index = next((i for i, (pid, _) in enumerate(project_options) if pid == get_active_project_id()), 0)
            selected_project = st.selectbox(
                "Select Active Project", 
                options=project_options,
                format_func=lambda x: x[1],
                index=default_index
            )
            
            active_project_id = selected_project[0]
            set_active_project_id(active_project_id)
            
            # Campaign selection
            active_project_campaigns = session.query(Campaign).filter_by(project_id=active_project_id).all()
            if active_project_campaigns:
                campaign_options = [(c.id, c.campaign_name) for c in active_project_campaigns]  # Changed from c.name to c.campaign_name
                default_campaign_index = next(
                    (i for i, (cid, _) in enumerate(campaign_options) if cid == get_active_campaign_id()), 
                    0
                )
                selected_campaign = st.selectbox(
                    "Select Active Campaign", 
                    options=campaign_options,
                    format_func=lambda x: x[1],
                    index=default_campaign_index
                )
                
                active_campaign_id = selected_campaign[0]
                set_active_campaign_id(active_campaign_id)
                
                st.success(f"Active Project: {selected_project[1]}, Active Campaign: {selected_campaign[1]}")
            else:
                st.warning(f"No campaigns available for {selected_project[1]}. Please add a campaign.")
        else:
            st.warning("No projects found. Please add a project first.")

def knowledge_base_page():
    st.title("Knowledge Base")
    with db_session() as session:
        project_options = fetch_projects(session)
        if not project_options: return st.warning("No projects found. Please create a project first.")
        selected_project = st.selectbox("Select Project", options=project_options)
        project_id = int(selected_project.split(":")[0])
        set_active_project_id(project_id)
        kb_entry = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        with st.form("knowledge_base_form"):
            fields = ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']
            form_data = {field: st.text_input(field.replace('_', ' ').title(), value=getattr(kb_entry, field, '')) if field in ['kb_name', 'contact_name', 'contact_role', 'contact_email', 'product_name'] else st.text_area(field.replace('_', ' ').title(), value=getattr(kb_entry, field, '')) for field in fields}
            if st.form_submit_button("Save Knowledge Base"):
                try:
                    form_data.update({'project_id': project_id, 'created_at': datetime.utcnow()})
                    if kb_entry:
                        for k, v in form_data.items(): setattr(kb_entry, k, v)
                    else: session.add(KnowledgeBase(**form_data))
                    session.commit()
                    st.success("Knowledge Base saved successfully!", icon="✅")
                except Exception as e: st.error(f"An error occurred while saving the Knowledge Base: {str(e)}")

def autoclient_ai_page():
    st.header("AutoclientAI - Automated Lead Generation")
    with st.expander("Knowledge Base Information", expanded=False):
        with db_session() as session:
            kb_info = get_knowledge_base_info(session, get_active_project_id())
        if not kb_info:
            return st.error("Knowledge Base not found for the active project. Please set it up first.")
        st.json(kb_info)
    user_input = st.text_area("Enter additional context or specific goals for lead generation:", help="This information will be used to generate more targeted search terms.")
    if st.button("Generate Optimized Search Terms", key="generate_optimized_terms"):
        with st.spinner("Generating optimized search terms..."):
            with db_session() as session:
                base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
                optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            if optimized_terms:
                st.session_state.optimized_terms = optimized_terms
                st.success("Search terms optimized successfully!")
                st.subheader("Optimized Search Terms")
                st.write(", ".join(optimized_terms))
            else:
                st.error("Failed to generate optimized search terms. Please try again.")
    if st.button("Start Automation", key="start_automation"):
        st.session_state.update({"automation_status": True, "automation_logs": [], "total_leads_found": 0, "total_emails_sent": 0})
        st.success("Automation started!")
    if st.session_state.get('automation_status', False):
        st.subheader("Automation in Progress")
        progress_bar, log_container, leads_container, analytics_container = st.progress(0), st.empty(), st.empty(), st.empty()
        try:
            with db_session() as session:
                ai_automation_loop(session, log_container, leads_container)
        except Exception as e:
            st.error(f"An error occurred in the automation process: {str(e)}")
            st.session_state.automation_status = False
    if not st.session_state.get('automation_status', False) and st.session_state.get('automation_logs'):
        st.subheader("Automation Results")
        st.metric("Total Leads Found", st.session_state.total_leads_found)
        st.metric("Total Emails Sent", st.session_state.total_emails_sent)
        st.subheader("Automation Logs")
        st.text_area("Logs", "\n".join(st.session_state.automation_logs), height=300)
    if 'email_logs' in st.session_state:
        st.subheader("Email Sending Logs")
        df_logs = pd.DataFrame(st.session_state.email_logs)
        st.dataframe(df_logs)
        success_rate = (df_logs['Status'] == 'sent').mean() * 100
        st.metric("Email Sending Success Rate", f"{success_rate:.2f}%")
    st.subheader("Debug Information")
    st.json(st.session_state)
    st.write("Current function:", autoclient_ai_page.__name__)
    st.write("Session state keys:", list(st.session_state.keys()))

def update_search_terms(session, classified_terms):
    for group, terms in classified_terms.items():
        for term in terms:
            existing_term = session.query(SearchTerm).filter_by(term=term, project_id=get_active_project_id()).first()
            if existing_term:
                existing_term.group = group
            else:
                session.add(SearchTerm(term=term, group=group, project_id=get_active_project_id()))
    session.commit()

def update_results_display(results_container, results):
    results_container.markdown(
        f"""
        <style>
        .results-container {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
            background-color: rgba(49, 51, 63, 0.1);
        }}
        .result-entry {{
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 0.25rem;
        }}
        </style>
        <div class="results-container">
            <h4>Found Leads ({len(results)})</h4>
            {"".join(f'<div class="result-entry"><strong>{res["Email"]}</strong><br>{res["URL"]}</div>' for res in results[-10:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def automation_control_panel_page():
    st.title("Automation Control Panel")

    with db_session() as session:
        active_project_id = get_active_project_id()
        active_campaign_id = get_active_campaign_id()

        # Fetch the active project and campaign details
        project = session.query(Project).get(active_project_id)
        campaign = session.query(Campaign).get(active_campaign_id)

        if project:
            st.subheader(f"Active Project: {project.project_name}")  # Changed from project.name
            # Remove description reference since it doesn't exist in the model
            
        if campaign:
            st.subheader(f"Active Campaign: {campaign.campaign_name}")  # Changed from campaign.name
            # Remove description reference since it doesn't exist in the model

    # ... rest of automation_control_panel_page() ...

def get_knowledge_base_info(session, project_id):
    kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb.to_dict() if kb else None

def generate_optimized_search_terms(session, base_terms, kb_info):
    ai_prompt = f"Generate 5 optimized search terms based on: {', '.join(base_terms)}. Context: {kb_info}"
    return get_ai_response(ai_prompt).split('\n')

def update_display(container, items, title, item_type):
    container.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True)
    for item in items[-10:]:
        container.text(item)

def get_search_terms(session):
    return [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]

def get_ai_response(prompt):
    return openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100).choices[0].text.strip()

def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        return []

def bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_bar=None, status_container=None, results_container=None, log_container=None):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        logging.error(f"Email template with ID {template_id} not found.")
        return [], 0

    email_subject = template.subject
    email_content = template.body_content

    logs, sent_count = [], 0
    total_leads = len(leads)

    for index, lead in enumerate(leads):
        try:
            validate_email(lead['Email'])
            response, tracking_id = send_email_ses(session, from_email, lead['Email'], email_subject, email_content, reply_to=reply_to)
            if response:
                status = 'sent'
                message_id = response.get('MessageId', f"sent-{uuid.uuid4()}")
                sent_count += 1
                log_message = f"✅ Email sent to: {lead['Email']}"
            else:
                status = 'failed'
                message_id = f"failed-{uuid.uuid4()}"
                log_message = f"❌ Failed to send email to: {lead['Email']}"
            
            save_email_campaign(session, lead['Email'], template_id, status, datetime.utcnow(), email_subject, message_id, email_content)
            logs.append(log_message)

            if progress_bar:
                progress_bar.progress((index + 1) / total_leads)
            if status_container:
                status_container.text(f"Processed {index + 1}/{total_leads} leads")
            if results_container:
                results_container.write(f"{log_message}")
            if log_container:
                log_container.write(log_message)

        except EmailNotValidError:
            log_message = f"❌ Invalid email address: {lead['Email']}"
            logs.append(log_message)
            if log_container:
                log_container.write(log_message)
        except Exception as e:
            error_message = f"Error sending email to {lead['Email']}: {str(e)}"
            logging.error(error_message)
            save_email_campaign(session, lead['Email'], template_id, 'failed', datetime.utcnow(), email_subject, f"error-{uuid.uuid4()}", email_content)
            logs.append(f"❌ Error sending email to: {lead['Email']} (Error: {str(e)})")

    return logs, sent_count

def wrap_email_body(body_content):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Template</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
        </style>
    </head>
    <body>
        {body_content}
    </body>
    </html>
    """

def fetch_sent_email_campaigns(session):
    try:
        email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({
            'ID': [ec.id for ec in email_campaigns],
            'Sent At': [ec.sent_at.strftime("%Y-%m-%d %H:%M:%S") if ec.sent_at else "" for ec in email_campaigns],
            'Email': [ec.lead.email for ec in email_campaigns],
            'Template': [ec.template.template_name for ec in email_campaigns],
            'Subject': [ec.customized_subject or "No subject" for ec in email_campaigns],
            'Content': [ec.customized_content or "No content" for ec in email_campaigns],
            'Status': [ec.status for ec in email_campaigns],
            'Message ID': [ec.message_id or "No message ID" for ec in email_campaigns],
            'Campaign ID': [ec.campaign_id for ec in email_campaigns],
            'Lead Name': [f"{ec.lead.first_name or ''} {ec.lead.last_name or ''}".strip() or "Unknown" for ec in email_campaigns],
            'Lead Company': [ec.lead.company or "Unknown" for ec in email_campaigns]
        })
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_sent_email_campaigns: {str(e)}")
        return pd.DataFrame()

def display_logs(log_container, logs, selected_filter, auto_scroll):
    """Display logs with improved styling, filtering, and auto-scroll"""
    if not logs:
        return

    filtered_logs = []
    for log_entry in logs:
        if isinstance(log_entry, dict):
            log_message = log_entry.get('message', '')
            log_level = log_entry.get('level', 'info')
            log_timestamp = log_entry.get('timestamp', '')
        elif isinstance(log_entry, str):
            log_message = log_entry
            log_level = 'info'
            log_timestamp = ''
        else:
            continue

        if selected_filter == 'all' or selected_filter == log_level:
            filtered_logs.append((log_timestamp, log_level, log_message))

    log_html = ""
    for timestamp, level, message in filtered_logs:
        icon = "ℹ️"
        color = "black"
        if level == 'error':
            icon = "❌"
            color = "red"
        elif level == 'success':
            icon = "✅"
            color = "green"
        elif level == 'email':
            icon = "📧"
            color = "blue"
        elif level == 'search':
            icon = "🔍"
            color = "purple"

        log_html += f"<div style='color: {color}; margin-bottom: 5px;'><small>{timestamp}</small> {icon} {message}</div>"

    log_container.markdown(
        f"""
        <div style='
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
            height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.9em;
            line-height: 1.4;
        '>
            {log_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    if auto_scroll:
        js = f"""
        <script>
            var element = window.parent.document.querySelectorAll('div.stMarkdown')[{st.session_state.worker_log_state['update_counter']}];
            element[0].scrollTop = element[0.0].scrollHeight;
        </script>
        """
        st.components.v1.html(js, height=0)

def view_sent_email_campaigns():
    st.header("Sent Email Campaigns")
    try:
        with db_session() as session:
            email_campaigns = fetch_sent_email_campaigns(session)
        if not email_campaigns.empty:
            st.dataframe(email_campaigns)
            st.subheader("Detailed Content")
            selected_campaign = st.selectbox("Select a campaign to view details", email_campaigns['ID'].tolist())
            if selected_campaign:
                campaign_content = email_campaigns[email_campaigns['ID'] == selected_campaign]['Content'].iloc[0]
                st.text_area("Content", campaign_content if campaign_content else "No content available", height=300)
        else:
            st.info("No sent email campaigns found.")
    except Exception as e:
        st.error(f"An error occurred while fetching sent email campaigns: {str(e)}")
        logging.error(f"Error in view_sent_email_campaigns: {str(e)}")

def fetch_logs_for_automation_log(session, automation_log_id):
    automation_log = session.query(AutomationLog).get(automation_log_id)
    if automation_log and automation_log.logs:
        return automation_log.logs
    else:
        return []

def run_automated_search(automation_log_id):
    """Start an automated search process and return the process object."""
    try:
        with db_session() as session:
            automation_log = session.query(AutomationLog).get(automation_log_id)
            if not automation_log:
                logging.error(f"No automation log found with ID {automation_log_id}")
                return None

            # Start the background search process
            thread = threading.Thread(
                target=run_background_search,
                args=(session, automation_log_id),
                daemon=True
            )
            thread.start()
            return thread

    except Exception as e:
        logging.error(f"Error starting automated search: {str(e)}")
        return None

def manual_search_worker_page():
    """Page for manual search worker control."""
    st.title("⚙️ Manual Search Worker")

    # Check for active project and campaign
    if not get_active_project_id() or not get_active_campaign_id():
        st.warning("Please select a project and campaign first.")
        return

    st.markdown(
        """
        This page allows you to run and monitor automated search processes. 
        Configure your search settings below and start a worker to begin searching.
        """
    )

    try:
        with db_session() as session:
            active_project_id = get_active_project_id()
            active_campaign_id = get_active_campaign_id()

            # Fetch the active project and campaign details
            project = session.query(Project).get(active_project_id)
            campaign = session.query(Campaign).get(active_campaign_id)

            if project:
                st.subheader(f"Active Project: {project.project_name}")

            if campaign:
                st.subheader(f"Active Campaign: {campaign.campaign_name}")

                # Fetch search terms for the campaign
                search_terms = session.query(SearchTerm).filter_by(campaign_id=campaign.id).all()
                if not search_terms:
                    st.warning("No search terms found for this campaign. Please add search terms first.")
                    return

                # Display search terms
                st.subheader("Campaign Search Terms")
                terms_df = pd.DataFrame([{
                    'Term': term.term,
                    'Category': term.category,
                    'Language': term.language
                } for term in search_terms])
                st.dataframe(terms_df)

                # Add search configuration options
                st.subheader("Search Configuration")
                num_results = st.number_input("Results per search term", min_value=1, value=50)
                ignore_previously_fetched = st.checkbox("Ignore previously fetched domains", value=True)
                language = st.selectbox("Search Language", ["ES", "EN"], index=0)
                
                # Advanced options in expander
                with st.expander("Advanced Options"):
                    optimize_english = st.checkbox("Optimize English terms", value=False)
                    optimize_spanish = st.checkbox("Optimize Spanish terms", value=False)
                    shuffle_keywords = st.checkbox("Shuffle keywords", value=False)

                # Email settings if needed
                with st.expander("Email Settings"):
                    enable_email_sending = st.checkbox("Enable Email Sending", value=False)
                    if enable_email_sending:
                        email_templates = session.query(EmailTemplate).filter_by(campaign_id=campaign.id).all()
                        if email_templates:
                            template_options = {f"{t.template_name} (ID: {t.id})": t.id for t in email_templates}
                            selected_template = st.selectbox("Select Email Template", list(template_options.keys()))
                            template_id = template_options[selected_template]
                            
                            email_settings = session.query(EmailSettings).filter_by(is_active=True).all()
                            if email_settings:
                                from_email = st.selectbox("From Email", 
                                    [setting.email for setting in email_settings])
                                reply_to = st.text_input("Reply-To Email (optional)")
                            else:
                                st.warning("No active email settings found. Please configure email settings first.")
                                enable_email_sending = False
                        else:
                            st.warning("No email templates found. Please create an email template first.")
                            enable_email_sending = False

                # Button to start the automated search
                if st.button("Start Automated Search"):
                    # Create automation log entry
                    automation_log = AutomationLog(
                        campaign_id=active_campaign_id,
                        start_time=datetime.now(),
                        status="running",
                        logs=[{
                            'timestamp': datetime.now().isoformat(),
                            'level': 'info',
                            'message': 'Starting automated search',
                            'search_settings': {
                                'num_results': num_results,
                                'ignore_previously_fetched': ignore_previously_fetched,
                                'language': language,
                                'optimize_english': optimize_english,
                                'optimize_spanish': optimize_spanish,
                                'shuffle_keywords_option': shuffle_keywords,
                                'enable_email_sending': enable_email_sending,
                                'from_email': from_email if enable_email_sending else None,
                                'reply_to': reply_to if enable_email_sending else None,
                                'template_id': template_id if enable_email_sending else None,
                                'search_terms': [term.term for term in search_terms]
                            }
                        }]
                    )
                    session.add(automation_log)
                    session.commit()

                    # Start the search process
                    process = run_automated_search(automation_log.id)
                    if process:
                        st.session_state.search_process = process
                        st.session_state.automation_log_id = automation_log.id
                        st.success(f"Automated search started with log ID: {automation_log.id}")
                    else:
                        st.error("Failed to start automated search.")

                # Display logs if an automation log ID exists
                if 'automation_log_id' in st.session_state:
                    automation_log = session.query(AutomationLog).get(st.session_state.automation_log_id)
                    if automation_log:
                        st.subheader("Search Progress")
                        
                        # Display status and metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Status", automation_log.status)
                        col2.metric("Leads Gathered", automation_log.leads_gathered or 0)
                        col3.metric("Emails Sent", automation_log.emails_sent or 0)
                        
                        # Display logs
                        st.subheader("Search Logs")
                        log_container = st.container()
                        with log_container:
                            if automation_log.logs:
                                for log in automation_log.logs:
                                    if isinstance(log, dict):
                                        timestamp = log.get('timestamp', '')
                                        level = log.get('level', 'info')
                                        message = log.get('message', '')
                                        
                                        icon = {
                                            'info': 'ℹ️',
                                            'success': '✅',
                                            'warning': '⚠️',
                                            'error': '❌'
                                        }.get(level, 'ℹ️')
                                        
                                        st.markdown(f"{icon} `{timestamp}` {message}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("Error in manual search worker page")

def get_active_project_id():
    """Get the currently active project ID from session state"""
    if not st.session_state.get("is_initialized"):
        initialize_session_state()
    validate_active_ids()
    # Ensure the returned value is an integer
    return int(st.session_state.get("current_project_id"))

def set_active_project_id(project_id):
    """Set the active project ID in session state"""
    # Ensure the ID is stored as an integer
    st.session_state.current_project_id = int(project_id)

def get_active_campaign_id():
    """Get the currently active campaign ID from session state"""
    if not st.session_state.get("is_initialized"):
        initialize_session_state()
    validate_active_ids()
    # Ensure the returned value is an integer
    return int(st.session_state.get("current_campaign_id"))

def set_active_campaign_id(campaign_id):
    """Set the active campaign ID in session state"""
    # Ensure the ID is stored as an integer
    st.session_state.current_campaign_id = int(campaign_id)

def safe_datetime_compare(date1, date2):
    """Safely compare two datetime objects, handling None values"""
    if date1 is None and date2 is None:
        return 0
    if date1 is None:
        return -1
    if date2 is None:
        return 1
    return -1 if date1 < date2 else 1 if date1 > date2 else 0

def validate_and_clean_email(email):
    """Validate and clean an email address"""
    try:
        if not email:
            return None
        # Remove any whitespace
        email = email.strip()
        # Validate email
        valid = validate_email(email)
        return valid.email
    except EmailNotValidError:
        return None

def process_email_template(template_content, lead_info=None, kb_info=None):
    """Process an email template with lead and knowledge base information"""
    if not template_content:
        return ""
    
    try:
        # Replace lead-specific placeholders
        if lead_info:
            template_content = template_content.replace("{first_name}", lead_info.get('first_name', ''))
            template_content = template_content.replace("{last_name}", lead_info.get('last_name', ''))
            template_content = template_content.replace("{company}", lead_info.get('company', ''))
            template_content = template_content.replace("{job_title}", lead_info.get('job_title', ''))
        
        # Replace knowledge base placeholders
        if kb_info:
            template_content = template_content.replace("{company_name}", kb_info.get('company_name', ''))
            template_content = template_content.replace("{product_name}", kb_info.get('product_name', ''))
            template_content = template_content.replace("{contact_name}", kb_info.get('contact_name', ''))
            template_content = template_content.replace("{contact_role}", kb_info.get('contact_role', ''))
        
        return template_content
    except Exception as e:
        logging.error(f"Error processing email template: {str(e)}")
        return template_content

def check_email_limits(session, email_settings):
    """Check if email sending is within configured limits"""
    try:
        if not email_settings:
            return {'can_send': False, 'daily_remaining': 0, 'hourly_remaining': 0}
        
        # Get current time
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        
        # Count emails sent today
        daily_sent = session.query(EmailCampaign).filter(
            EmailCampaign.sent_at >= today_start
        ).count()
        
        # Count emails sent this hour
        hourly_sent = session.query(EmailCampaign).filter(
            EmailCampaign.sent_at >= hour_start
        ).count()
        
        # Check against limits
        daily_remaining = email_settings.daily_limit - daily_sent if email_settings.daily_limit else float('inf')
        hourly_remaining = email_settings.hourly_limit - hourly_sent if email_settings.hourly_limit else float('inf')
        
        can_send = daily_remaining > 0 and hourly_remaining > 0
        
        return {
            'can_send': can_send,
            'daily_remaining': daily_remaining if daily_remaining != float('inf') else 'unlimited',
            'hourly_remaining': hourly_remaining if hourly_remaining != float('inf') else 'unlimited'
        }
        
    except Exception as e:
        logging.error(f"Error checking email limits: {str(e)}")
        return {'can_send': False, 'daily_remaining': 0, 'hourly_remaining': 0}

def initialize_pages():
    return {
        "🔍 Manual Search": manual_search_page,
        "📦 Bulk Send": bulk_send_page,
        "👥 View Leads": view_leads_page,
        "🔑 Search Terms": search_terms_page,
        "✉️ Email Templates": email_templates_page,
        "📚 Knowledge Base": knowledge_base_page,
        "🤖 AutoclientAI": autoclient_ai_page,
        "⚙️ Automation Control": automation_control_panel_page,
        "⚙️ Manual Search Worker": manual_search_worker_page,
        "📨 Email Logs": view_campaign_logs,
        "🔄 Settings": settings_page,
        "📨 Sent Campaigns": view_sent_email_campaigns,
        "🚀 Projects & Campaigns": projects_campaigns_page
    }

def initialize_session_state():
    """Initialize session state and ensure all required data exists."""
    # Initialize basic session state
    st.session_state.active_page = st.session_state.get('active_page', 'Manual Search')
    st.session_state.search_results = st.session_state.get('search_results', [])
    st.session_state.edit_template_id = st.session_state.get('edit_template_id', None)
    st.session_state.is_initialized = True
    
    # Ensure required database objects exist
    with db_session() as session:
        # Create default project if none exists
        default_project = session.query(Project).first()
        if not default_project:
            default_project = Project(project_name="Default Project")
            session.add(default_project)
            session.commit()
        
        # Create default campaign if none exists
        default_campaign = session.query(Campaign).first()
        if not default_campaign:
            default_campaign = Campaign(
                campaign_name="Default Campaign",
                project_id=default_project.id
            )
            session.add(default_campaign)
            session.commit()
        
        # Create default email settings if none exist
        default_email = session.query(EmailSettings).first()
        if not default_email:
            default_email = EmailSettings(
                name="Default Email",
                email="noreply@example.com",
                provider="ses",
                aws_region="us-east-1",
                is_active=True
            )
            session.add(default_email)
            session.commit()
        
        # Create default email template if none exists
        default_template = session.query(EmailTemplate).first()
        if not default_template:
            default_template = EmailTemplate(
                campaign_id=default_campaign.id,
                template_name="Default Template",
                subject="Hello from AutoclientAI",
                body_content="This is a default email template."
            )
            session.add(default_template)
            session.commit()
        
        # Create default knowledge base if none exists
        default_kb = session.query(KnowledgeBase).first()
        if not default_kb:
            default_kb = KnowledgeBase(
                project_id=default_project.id,
                kb_name="Default Knowledge Base",
                company_description="Default company description"
            )
            session.add(default_kb)
            session.commit()
        
        # Always set session state to first available IDs
        st.session_state.current_project_id = default_project.id
        st.session_state.current_campaign_id = default_campaign.id

def check_database_connection():
    """Check if database connection is working."""
    try:
        with db_session() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        return False

def check_email_service():
    """Check if email service is available."""
    try:
        with db_session() as session:
            settings = session.query(EmailSettings).first()
            if not settings:
                return False
            
            if settings.provider.lower() == 'ses':
                try:
                    aws_session = boto3.Session(
                        aws_access_key_id=settings.aws_access_key_id,
                        aws_secret_access_key=settings.aws_secret_access_key,
                        region_name=settings.aws_region or 'us-east-1'
                    )
                    ses_client = aws_session.client('ses')
                    # Try to verify email identity instead of checking quota
                    ses_client.verify_email_identity(EmailAddress=settings.email)
                    return True
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    if error_code == 'AccessDenied':
                        # Log the error but don't fail the check
                        logging.warning(f"SES permissions limited: {str(e)}")
                        return True
                    elif error_code == 'MessageRejected':
                        logging.error(f"Email verification failed: {str(e)}")
                        return False
                    else:
                        logging.error(f"SES error: {str(e)}")
                        return False
            else:  # SMTP provider
                try:
                    with smtplib.SMTP(settings.smtp_server, settings.smtp_port, timeout=5) as server:
                        server.starttls()
                        server.login(settings.smtp_username, settings.smtp_password)
                    return True
                except Exception as e:
                    logging.error(f"SMTP error: {str(e)}")
                    return False
    except Exception as e:
        logging.error(f"Email service check error: {str(e)}")
        return False

def initialize_settings():
    """Initialize database and required settings."""
    try:
        with db_session() as session:
            # Create default project and campaign if they don't exist
            default_project = session.query(Project).filter_by(id=1).first()
            if not default_project:
                try:
                    default_project = Project(
                        id=1,
                        project_name="Default Project",
                        created_at=datetime.utcnow()
                    )
                    session.add(default_project)
                    session.commit()
                except SQLAlchemyError as e:
                    logging.error(f"Failed to create default project: {str(e)}")
                    return False

            default_campaign = session.query(Campaign).filter_by(id=1).first()
            if not default_campaign:
                try:
                    default_campaign = Campaign(
                        id=1,
                        campaign_name="Default Campaign",
                        project_id=1,
                        created_at=datetime.utcnow()
                    )
                    session.add(default_campaign)
                    session.commit()
                except SQLAlchemyError as e:
                    logging.error(f"Failed to create default campaign: {str(e)}")
                    return False

            # Initialize session state
            initialize_session_state()
            return True

    except Exception as e:
        logging.error(f"Failed to initialize settings: {str(e)}")
        st.error(f"Failed to initialize settings: {str(e)}")
        return False

def initialize_pages():
    return {
        "🔍 Manual Search": manual_search_page,
        "📦 Bulk Send": bulk_send_page,
        "👥 View Leads": view_leads_page,
        "🔑 Search Terms": search_terms_page,
        "✉️ Email Templates": email_templates_page,
        "📚 Knowledge Base": knowledge_base_page,
        "🤖 AutoclientAI": autoclient_ai_page,
        "⚙️ Automation Control": automation_control_panel_page,
        "⚙️ Manual Search Worker": manual_search_worker_page,
        "📨 Email Logs": view_campaign_logs,
        "🔄 Settings": settings_page,
        "📨 Sent Campaigns": view_sent_email_campaigns,
        "🚀 Projects & Campaigns": projects_campaigns_page
    }

def main():
    st.set_page_config(
        page_title="AutoclientAI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check database connection first
    if not check_database_connection():
        st.error("Failed to connect to database. Please check your configuration.")
        return

    # Initialize settings and check database state
    if not initialize_settings():
        st.error("Failed to initialize application. Please check the logs and configuration.")
        return

    # Initialize session state with defaults
    initialize_session_state()

    # Initialize pages
    pages = initialize_pages()

    # Create navigation menu
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=list(pages.keys()),
            icons=["search", "box-seam", "people", "key", "envelope", "book", "robot", 
                  "gear", "tools", "journal-text", "sliders", "envelope-paper", "rocket"],
            menu_icon="house",
            default_index=0
        )

    # Check if we have default project and campaign
    with db_session() as session:
        if not get_active_project_id():
            set_active_project_id(1)
        if not get_active_campaign_id():
            set_active_campaign_id(1)

        # Verify the defaults exist
        project = session.query(Project).get(get_active_project_id())
        campaign = session.query(Campaign).get(get_active_campaign_id())

        if not project or not campaign:
            if selected not in ["🔄 Settings", "🚀 Projects & Campaigns"]:
                st.warning("Please set up a project and campaign first.")
                pages["🚀 Projects & Campaigns"]()
                return

        # Add project name and description to Project model if needed
        if project:
            st.sidebar.markdown(f"**Active Project:** {project.project_name}")  # Changed from project.name
            if campaign:
                st.sidebar.markdown(f"**Active Campaign:** {campaign.campaign_name}")  # Changed from campaign.name

    try:
        pages[selected]()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("An error occurred in the main function")

def verify_ses_setup(session, email_settings_id):
    """Verify SES configuration and permissions."""
    try:
        email_settings = session.query(EmailSettings).get(email_settings_id)
        if not email_settings or email_settings.provider.lower() != 'ses':
            return False, "Invalid email settings or not SES provider"

        ses_client = boto3.client(
            'ses',
            aws_access_key_id=email_settings.aws_access_key_id,
            aws_secret_access_key=email_settings.aws_secret_access_key,
            region_name=email_settings.aws_region
        )

        # Test SES permissions
        try:
            quota = ses_client.get_send_quota()
            logging.info(f"SES quota retrieved successfully: {quota}")
        except Exception as e:
            logging.error(f"Failed to get SES quota: {str(e)}")
            return False, f"SES permissions error: {str(e)}"

        # Verify email identity if not already verified
        try:
            verification = ses_client.get_identity_verification_attributes(
                Identities=[email_settings.email]
            )
            status = verification['VerificationAttributes'].get(
                email_settings.email, {}
            ).get('VerificationStatus', 'NotVerified')

            if status.lower() != 'success':
                ses_client.verify_email_identity(EmailAddress=email_settings.email)
                return False, f"Email {email_settings.email} verification pending. Check your email."

        except Exception as e:
            logging.error(f"Failed to verify email identity: {str(e)}")
            return False, f"Email verification error: {str(e)}"

        return True, "SES setup verified successfully"

    except Exception as e:
        logging.error(f"Error verifying SES setup: {str(e)}")
        return False, f"Setup verification error: {str(e)}"

def validate_active_ids():
    """Validate that current project and campaign IDs exist in database."""
    with db_session() as session:
        project_id = st.session_state.get('current_project_id')
        campaign_id = st.session_state.get('current_campaign_id')
        
        project = session.query(Project).filter_by(id=project_id).first()
        campaign = session.query(Campaign).filter_by(id=campaign_id).first()
        
        if not project or not campaign:
            # Reset to defaults if invalid
            initialize_session_state()
            return False
        return True

def get_active_project_id():
    """Get the currently active project ID from session state"""
    if not st.session_state.get('is_initialized'):
        initialize_session_state()
    validate_active_ids()
    return st.session_state.get('current_project_id')

def get_active_campaign_id():
    """Get the currently active campaign ID from session state"""
    if not st.session_state.get('is_initialized'):
        initialize_session_state()
    validate_active_ids()
    return st.session_state.get('current_campaign_id')

def set_active_campaign_id(campaign_id):
    """Set the active campaign ID in session state"""
    st.session_state.current_campaign_id = campaign_id

def perform_search(session, search_terms, project_id, campaign_id, num_results, language, 
                  ignore_previously_fetched=True, optimize_english=False, 
                  optimize_spanish=False, shuffle_keywords_option=False):
    """Perform the search operation with the given parameters."""
    results = []
    
    for term in search_terms:
        # Save search term
        search_term = add_or_get_search_term(session, term, campaign_id)
        
        # Optimize term if requested
        if optimize_english and language == "EN":
            term = optimize_search_term(term, "EN")
        elif optimize_spanish and language == "ES":
            term = optimize_search_term(term, "ES")
            
        # Shuffle keywords if requested
        if shuffle_keywords_option:
            term = shuffle_keywords(term)
            
        # Perform the search
        try:
            search_results = manual_search(
                session=session,
                terms=[term],
                num_results=num_results,
                ignore_previously_fetched=ignore_previously_fetched,
                optimize_english=optimize_english,
                optimize_spanish=optimize_spanish,
                shuffle_keywords_option=shuffle_keywords_option,
                language=language
            )
            
            if search_results and 'results' in search_results:
                results.extend(search_results['results'])
                
                # Log effectiveness
                log_search_term_effectiveness(
                    session,
                    search_term,
                    len(search_results['results']),
                    len([r for r in search_results['results'] if r.get('Email')]),
                    len([r for r in search_results['results'] if 'blog' in r.get('URL', '').lower()]),
                    len([r for r in search_results['results'] if 'directory' in r.get('URL', '').lower()])
                )
                
        except Exception as e:
            st.error(f"Error searching for term '{term}': {str(e)}")
            continue
            
    return results

def process_search_result(session, result, project_id, campaign_id):
    """Process a single search result and save lead information."""
    new_leads = []
    
    if 'Email' in result and result['Email'] and is_valid_email(result['Email']):
        try:
            # Save the lead
            lead = save_lead(
                session=session,
                email=result['Email'],
                url=result.get('URL'),
                company=result.get('Company'),
                first_name=result.get('FirstName'),
                last_name=result.get('LastName'),
                job_title=result.get('JobTitle')
            )
            
            if lead:
                # Create campaign lead association
                campaign_lead = CampaignLead(
                    campaign_id=campaign_id,
                    lead_id=lead.id,
                    status='new'
                )
                session.add(campaign_lead)
                
                # Save lead source information
                if result.get('URL'):
                    # Fetch the SearchTerm object using the ID
                    search_term_id = result.get('Search Term ID')
                    search_term = session.query(SearchTerm).get(search_term_id)
                    if search_term:
                        lead_source = LeadSource(
                            lead_id=lead.id,
                            search_term_id=search_term.id,
                            url=result['URL'],
                            domain=get_domain_from_url(result['URL']),
                            page_title=result.get('Title'),
                            meta_description=result.get('Description'),
                            content=result.get('Content'),
                            http_status=result.get('Status', 200),
                            scrape_duration=result.get('ScrapeDuration', 0)
                        )
                        session.add(lead_source)
                
                session.commit()
                
                # Add to new leads list
                new_leads.append({
                    'id': lead.id,
                    'email': lead.email,
                    'company': lead.company,
                    'url': result.get('URL'),
                    'first_name': lead.first_name,
                    'last_name': lead.last_name,
                    'job_title': lead.job_title
                })
                
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error processing search result: {str(e)}")
        except Exception as e:
            session.rollback()
            logging.error(f"Error processing search result: {str(e)}")
    
    return new_leads

def start_background_process():
    """Start or resume background search process"""
    if 'background_process' not in st.session_state:
        st.session_state.background_process = {
            'is_running': False,
            'pid': None,
            'start_time': None,
            'last_update': None
        }

def get_background_state():
    """Get current state of background process"""
    try:
        if os.path.exists('.search_pid'):
            with open('.search_pid', 'r') as f:
                pid = int(f.read().strip())
                try:
                    os.kill(pid, 0)  # Check if process exists
                    return {
                        'is_running': True,
                        'pid': pid,
                        'start_time': st.session_state.background_process.get('start_time'),
                        'last_update': st.session_state.background_process.get('last_update')
                    }
                except OSError:
                    cleanup_background_process()
    except:
        pass
    return None

def cleanup_background_process():
    """Clean up background process state"""
    try:
        if os.path.exists('.search_pid'):
            os.remove('.search_pid')
        if 'background_process' in st.session_state:
            st.session_state.background_process = {
                'is_running': False,
                'pid': None,
                'start_time': None,
                'last_update': None
            }
    except:
        pass

def run_background_search(session, automation_log_id):
    """Run search process in background"""
    try:
        automation_log = session.query(AutomationLog).get(automation_log_id)
        if not automation_log:
            return
        
        automation_log.status = 'running'
        automation_log.start_time = datetime.now()
        session.commit()
        
        # Initialize logs list if None
        if automation_log.logs is None:
            automation_log.logs = []
            
        # Get search settings from logs
        search_settings = {}
        if automation_log.logs:
            for log in reversed(automation_log.logs):
                if isinstance(log, dict) and 'search_settings' in log:
                    search_settings = log['search_settings']
                    break
        
        search_terms = search_settings.get('search_terms', [])
        if not search_terms:
            automation_log.logs.append({
                'timestamp': datetime.now().isoformat(),
                'level': 'error',
                'message': 'No search terms found'
            })
            automation_log.status = 'error'
            session.commit()
            return
        
        for i, term in enumerate(search_terms):
            try:
                if automation_log.status != 'running':
                    break
                
                results = manual_search(
                    session=session,
                    terms=[term],
                    num_results=search_settings.get('num_results', 10),
                    ignore_previously_fetched=search_settings.get('ignore_previously_fetched', True),
                    optimize_english=search_settings.get('optimize_english', False),
                    optimize_spanish=search_settings.get('optimize_spanish', False),
                    shuffle_keywords_option=search_settings.get('shuffle_keywords_option', False),
                    language=search_settings.get('language', 'ES'),
                    enable_email_sending=search_settings.get('enable_email_sending', False),
                    from_email=search_settings.get('from_email'),
                    reply_to=search_settings.get('reply_to'),
                    email_template=search_settings.get('email_template')
                )
                
                automation_log.leads_gathered = (automation_log.leads_gathered or 0) + len(results.get('results', []))
                if results.get('emails_sent'):
                    automation_log.emails_sent = (automation_log.emails_sent or 0) + results['emails_sent']
                
                automation_log.logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'level': 'info',
                    'message': f"Processed term '{term}': Found {len(results.get('results', []))} leads"
                })
                
                session.commit()
                
            except Exception as e:
                automation_log.logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'level': 'error',
                    'message': f"Error processing term '{term}': {str(e)}"
                })
                session.commit()
                continue
            
            time.sleep(2)  # Rate limiting
        
        automation_log.status = 'completed'
        automation_log.end_time = datetime.now()
        automation_log.logs.append({
            'timestamp': datetime.now().isoformat(),
            'level': 'success',
            'message': 'Search completed successfully'
        })
        session.commit()
        
    except Exception as e:
        try:
            automation_log.status = 'error'
            automation_log.logs.append({
                'timestamp': datetime.now().isoformat(),
                'level': 'error',
                'message': f"Critical error: {str(e)}"
            })
            session.commit()
        except:
            pass

def check_email_limits(session, email_setting):
    """Check if email sending limits are reached"""
    try:
        # Get count of emails sent in last 24h
        yesterday = datetime.utcnow() - timedelta(days=1)
        sent_count = session.query(EmailCampaign).filter(
            EmailCampaign.sent_at >= yesterday,
            EmailCampaign.status == 'sent'
        ).count()
        
        # Default daily limit
        daily_limit = email_setting.daily_limit if hasattr(email_setting, 'daily_limit') else 100
        
        if sent_count >= daily_limit:
            return {
                'can_send': False,
                'message': f"Daily email limit reached ({sent_count}/{daily_limit})"
            }
        
        return {
            'can_send': True,
            'message': f"Can send emails ({sent_count}/{daily_limit} used)"
        }
        
    except Exception as e:
        return {
            'can_send': False,
            'message': f"Error checking limits: {str(e)}"
        }

def bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_container=None, status_container=None, results_container=None, log_container=None):
    """Send emails in bulk to leads"""
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        return [], 0
    
    logs, sent_count = [], 0
    total_leads = len(leads)
    
    for index, lead in enumerate(leads):
        try:
            if not lead or 'Email' not in lead:
                continue
                
            validate_email(lead['Email'])
            
            # Prepare email content
            wrapped_content = wrap_email_body(template.body_content)
            
            # Send email
            response, tracking_id = send_email_ses(
                session, 
                from_email, 
                lead['Email'], 
                template.subject, 
                wrapped_content, 
                reply_to=reply_to
            )
            
            if response:
                status = 'sent'
                message_id = response.get('MessageId', f"sent-{uuid.uuid4()}")
                sent_count += 1
                log_message = f"✅ Email sent to: {lead['Email']}"
            else:
                status = 'failed'
                message_id = f"failed-{uuid.uuid4()}"
                log_message = f"❌ Failed to send email to: {lead['Email']}"
            
            # Save campaign
            save_email_campaign(
                session,
                lead['Email'],
                template_id,
                status,
                datetime.utcnow(),
                template.subject,
                message_id,
                wrapped_content
            )
            
            logs.append(log_message)
            
            # Update UI if containers provided
            if progress_container:
                progress_container.progress((index + 1) / total_leads)
            if status_container:
                status_container.text(f"Processed {index + 1}/{total_leads} leads")
            if results_container:
                results_container.write(f"{log_message}")
            if log_container:
                log_container.write(log_message)
                
        except EmailNotValidError:
            log_message = f"❌ Invalid email address: {lead['Email']}"
            logs.append(log_message)
            if log_container:
                log_container.write(log_message)
        except Exception as e:
            log_message = f"❌ Error sending email to {lead['Email']}: {str(e)}"
            logs.append(log_message)
            if log_container:
                log_container.write(log_message)
            
            try:
                save_email_campaign(
                    session,
                    lead['Email'],
                    template_id,
                    'failed',
                    datetime.utcnow(),
                    template.subject,
                    f"error-{uuid.uuid4()}",
                    wrapped_content
                )
            except:
                pass
    
    return logs, sent_count

def update_search_state(session, automation_log_id, **kwargs):
    """Update search state in database."""
    try:
        automation_log = session.query(AutomationLog).get(automation_log_id)
        if automation_log:
            for key, value in kwargs.items():
                setattr(automation_log, key, value)
            session.commit()
    except Exception as e:
        logging.error(f"Error updating search state: {str(e)}")

def get_search_state(session, automation_log_id):
    """Get current search state from database."""
    try:
        automation_log = session.query(AutomationLog).get(automation_log_id)
        if automation_log:
            return {
                'status': automation_log.status,
                'leads_gathered': automation_log.leads_gathered,
                'emails_sent': automation_log.emails_sent,
                'start_time': automation_log.start_time,
                'end_time': automation_log.end_time,
                'logs': automation_log.logs
            }
    except Exception as e:
        logging.error(f"Error getting search state: {str(e)}")
    return None

def get_page_description(html_content):
    """Extract meta description from HTML content."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '').strip()
        # Fallback to first paragraph if no meta description
        first_p = soup.find('p')
        if first_p:
            return first_p.get_text().strip()[:200]  # Limit to 200 chars
        return ''
    except Exception as e:
        logging.error(f"Error getting page description: {str(e)}")
        return ''

def view_campaign_logs():
    """Display email campaign logs."""
    st.title("Email Campaign Logs")
    
    try:
        with db_session() as session:
            # Fetch campaign logs with related data
            logs = session.query(EmailCampaign)\
                .join(Lead)\
                .join(EmailTemplate)\
                .order_by(EmailCampaign.sent_at.desc())\
                .limit(1000)\
                .all()
            
            if not logs:
                st.info("No campaign logs found.")
                return
            
            # Create DataFrame for display
            df = pd.DataFrame([{
                'Date': log.sent_at,
                'Email': log.lead.email if log.lead else 'Unknown',
                'Template': log.template.template_name if log.template else 'Unknown',
                'Status': log.status,
                'Opens': log.open_count,
                'Clicks': log.click_count
            } for log in logs])
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Emails", len(logs))
            col2.metric("Success Rate", f"{(df['Status'] == 'sent').mean():.1%}")
            col3.metric("Open Rate", f"{(df['Opens'] > 0).mean():.1%}")
            
            # Add filters
            status_filter = st.multiselect(
                "Filter by Status",
                options=sorted(df['Status'].unique()),
                default=sorted(df['Status'].unique())
            )
            
            # Apply filters
            filtered_df = df[df['Status'].isin(status_filter)]
            
            # Display logs
            st.dataframe(
                filtered_df,
                column_config={
                    "Date": st.column_config.DatetimeColumn("Date", format="D MMM YYYY, HH:mm"),
                    "Email": st.column_config.TextColumn("Email"),
                    "Template": st.column_config.TextColumn("Template"),
                    "Status": st.column_config.TextColumn("Status"),
                    "Opens": st.column_config.NumberColumn("Opens"),
                    "Clicks": st.column_config.NumberColumn("Clicks")
                },
                hide_index=True
            )
            
            # Display charts
            st.subheader("Campaign Analytics")
            col1, col2 = st.columns(2)
            
            with col1:
                status_counts = df['Status'].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Email Status Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                daily_sends = df.resample('D', on='Date')['Email'].count()
                fig = px.line(
                    x=daily_sends.index,
                    y=daily_sends.values,
                    title="Daily Email Volume"
                )
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"An error occurred while loading campaign logs: {str(e)}")
        logging.error(f"Error in view_campaign_logs: {str(e)}")

if __name__ == "__main__":
    main()