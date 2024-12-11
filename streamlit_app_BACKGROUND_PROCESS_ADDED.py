import os
import json
import re
import logging
import time
import requests
import pandas as pd
import streamlit as st
import openai
import boto3
import uuid
import urllib3
import random
import threading
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import (
    func, create_engine, Column, BigInteger, Text, 
    DateTime, ForeignKey, Boolean, JSON, select, 
    text, distinct, and_
)
from sqlalchemy.orm import (
    declarative_base, sessionmaker, relationship, 
    Session, joinedload
)
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from openai import OpenAI
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import smtplib
import sentry_sdk
from prometheus_client import start_http_server, Counter, Gauge
import healthcheck
from functools import wraps

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Initialize SQLAlchemy
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Monitoring and metrics
TOTAL_SEARCHES = Counter('total_searches', 'Total number of searches performed')
TOTAL_LEADS = Counter('total_leads', 'Total number of leads found')
ACTIVE_SEARCHES = Gauge('active_searches', 'Number of active search processes')
ERROR_COUNT = Counter('error_count', 'Total number of errors encountered')

def setup_monitoring():
    """Initialize monitoring and error tracking."""
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Initialize Sentry for error tracking
    sentry_dsn = os.getenv('SENTRY_DSN')
    if sentry_dsn:
        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=1.0,
            environment=os.getenv('ENVIRONMENT', 'production')
        )

def monitor_metrics(func):
    """Decorator to monitor function metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                TOTAL_LEADS.inc(result.get('total_leads', 0))
                ERROR_COUNT.inc(result.get('error_count', 0))
            return result
        finally:
            duration = time.time() - start_time
            logging.info(f"Function {func.__name__} took {duration:.2f} seconds")
    return wrapper

@contextmanager
def db_session():
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Database Models
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
    max_emails_per_group = Column(BigInteger, default=500)
    loop_interval = Column(BigInteger, default=60)
    project = relationship("Project", back_populates="campaigns")
    email_campaigns = relationship("EmailCampaign", back_populates="campaign")
    search_terms = relationship("SearchTerm", back_populates="campaign")
    campaign_leads = relationship("CampaignLead", back_populates="campaign")

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    id = Column(BigInteger, primary_key=True)
    project_id = Column(BigInteger, ForeignKey('projects.id'), nullable=False)
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    project = relationship("Project", back_populates="knowledge_base")

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in [
            'kb_name', 'kb_bio', 'kb_values', 'contact_name', 
            'contact_role', 'contact_email', 'company_description', 
            'company_mission', 'company_target_market', 'company_other', 
            'product_name', 'product_description', 'product_target_customer', 
            'product_other', 'other_context', 'example_email'
        ]}

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

class CampaignLead(Base):
    __tablename__ = 'campaign_leads'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    status = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="campaign_leads")
    campaign = relationship("Campaign", back_populates="campaign_leads")

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    template_name = Column(Text)
    subject = Column(Text)
    body_content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False)
    language = Column(Text, default='ES')
    campaign = relationship("Campaign")
    email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    template_id = Column(BigInteger, ForeignKey('email_templates.id'))
    customized_subject = Column(Text)
    customized_content = Column(Text)
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

class SearchProcess(Base):
    __tablename__ = 'search_processes'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    search_terms = Column(JSON)
    num_results = Column(BigInteger)
    status = Column(Text)
    error_message = Column(Text)
    total_leads_found = Column(BigInteger, default=0)
    settings = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    logs = Column(JSON, default=list)

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    id = Column(BigInteger, primary_key=True)
    term = Column(Text, nullable=False)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaign = relationship("Campaign", back_populates="search_terms")

class Settings(Base):
    __tablename__ = 'settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    setting_type = Column(Text, nullable=False)
    value = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    email = Column(Text, nullable=False)
    provider = Column(Text, nullable=False)  # 'smtp' or 'ses'
    # SMTP settings
    smtp_server = Column(Text)
    smtp_port = Column(BigInteger)
    smtp_username = Column(Text)
    smtp_password = Column(Text)
    # AWS SES settings
    aws_access_key_id = Column(Text)
    aws_secret_access_key = Column(Text)
    aws_region = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class LeadSource(Base):
    __tablename__ = 'lead_sources'
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    url = Column(Text)
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="lead_sources")
    search_term = relationship("SearchTerm")

# Create all tables
Base.metadata.create_all(bind=engine)

# Utility Functions
def wrap_email_body(body_content):
    """Wraps email body content in proper HTML structure."""
    return f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px;">
        {body_content}
    </body>
    </html>
    """

def get_domain_from_url(url):
    """Extracts domain from URL."""
    return urlparse(url).netloc

def is_valid_email(email):
    """Validates email format."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def extract_emails_from_html(html_content):
    """Extracts email addresses from HTML content."""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, html_content)

def get_page_title(html_content):
    """Extracts page title from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.find('title')
    return title.text.strip() if title else ''

def get_page_description(html_content):
    """Extracts page description from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_desc = soup.find('meta', {'name': 'description'})
    return meta_desc['content'].strip() if meta_desc and 'content' in meta_desc.attrs else ''

def extract_info_from_page(soup):
    """Extracts relevant information from page."""
    name = soup.find('meta', {'name': 'author'})
    name = name['content'] if name else ''
    
    company = soup.find('meta', {'property': 'og:site_name'})
    company = company['content'] if company else ''
    
    job_title = soup.find('meta', {'name': 'job_title'})
    job_title = job_title['content'] if job_title else ''
    
    return name, company, job_title

def update_log(log_container, message, level='info', search_process_id=None):
    """Updates log messages in UI and database."""
    icon = {
        'info': '🔵',
        'success': '🟢',
        'warning': '🟠',
        'error': '🔴',
        'email_sent': '🟣'
    }.get(level, '⚪')
    
    log_entry = f"{icon} {message}"
    print(f"{icon} {message.split('<')[0]}")
    
    if search_process_id:
        with db_session() as session:
            search_process = session.query(SearchProcess).get(search_process_id)
            if search_process:
                if not search_process.logs:
                    search_process.logs = []
                search_process.logs.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': level,
                    'message': message
                })
                session.commit()
    
    if log_container:
        if 'log_entries' not in st.session_state:
            st.session_state.log_entries = []
        
        html_log_entry = f"{icon} {message}"
        st.session_state.log_entries.append(html_log_entry)
        log_html = f"<div style='height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.8em; line-height: 1.2;'>{'<br>'.join(st.session_state.log_entries)}</div>"
        log_container.markdown(log_html, unsafe_allow_html=True)

def optimize_search_term(search_term, language):
    """Optimizes search term based on language."""
    if language == 'english':
        return f'"{search_term}" email OR contact OR "get in touch" site:.com'
    elif language == 'spanish':
        return f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es'
    return search_term

def shuffle_keywords(term):
    """Shuffles keywords in a search term."""
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

class SearchWorkerThread(threading.Thread):
    """Background worker thread for search processes."""
    def __init__(self, engine, session_maker):
        super().__init__()
        self.engine = engine
        self.session_maker = session_maker
        self.daemon = True
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            with db_session() as session:
                pending_process = session.query(SearchProcess).filter_by(status='pending').first()
                if pending_process:
                    try:
                        self.process_search(pending_process.id)
                    except Exception as e:
                        logging.error(f"Error in search worker: {str(e)}")
            time.sleep(1)

    def process_search(self, process_id):
        with db_session() as session:
            process = session.query(SearchProcess).get(process_id)
            if not process:
                return

            try:
                process.status = 'running'
                process.started_at = datetime.utcnow()
                session.commit()

                results = manual_search(
                    session,
                    process.search_terms,
                    process.num_results,
                    **process.settings
                )

                process.total_leads_found = results.get('total_leads', 0)
                process.status = 'completed'
                process.completed_at = datetime.utcnow()
                session.commit()

            except Exception as e:
                process.status = 'failed'
                process.error_message = str(e)
                process.completed_at = datetime.utcnow()
                session.commit()
                raise

    def stop(self):
        self._stop_event.set()

# Email Functions
def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    """Sends email using AWS SES or SMTP."""
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings:
        logging.error(f"No email settings found for {from_email}")
        return None, None

    tracking_id = str(uuid.uuid4())
    tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
    wrapped_body = wrap_email_body(body)
    tracked_body = wrapped_body.replace('</body>', f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/></body>')

    soup = BeautifulSoup(tracked_body, 'html.parser')
    for a in soup.find_all('a', href=True):
        original_url = a['href']
        tracked_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'click', 'url': original_url})}"
        a['href'] = tracked_url
    tracked_body = str(soup)

    try:
        if email_settings.provider == 'ses':
            if ses_client is None:
                aws_session = boto3.Session(
                    aws_access_key_id=email_settings.aws_access_key_id,
                    aws_secret_access_key=email_settings.aws_secret_access_key,
                    region_name=email_settings.aws_region
                )
                ses_client = aws_session.client('ses')
            
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

        elif email_settings.provider == 'smtp':
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            if reply_to:
                msg['Reply-To'] = reply_to
            msg.attach(MIMEText(tracked_body, 'html'))

            with smtplib.SMTP(email_settings.smtp_server, email_settings.smtp_port) as server:
                server.starttls()
                server.login(email_settings.smtp_username, email_settings.smtp_password)
                server.send_message(msg)
            return {'MessageId': f'smtp-{uuid.uuid4()}'}, tracking_id

        else:
            logging.error(f"Unknown email provider: {email_settings.provider}")
            return None, None

    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        return None, None

def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    """Saves email campaign details to database."""
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
            campaign_id=st.session_state.get('active_campaign_id', 1),
            tracking_id=str(uuid.uuid4())
        )
        session.add(new_campaign)
        session.commit()
        return new_campaign
    except Exception as e:
        logging.error(f"Error saving email campaign: {str(e)}")
        session.rollback()
        return None

# Core Search Function
def manual_search(session, terms, num_results, ignore_previously_fetched=True, 
                 optimize_english=False, optimize_spanish=False, 
                 shuffle_keywords_option=False, language='ES', 
                 enable_email_sending=True, log_container=None, 
                 from_email=None, reply_to=None, email_template=None, 
                 search_process_id=None):
    """
    Performs manual search for leads based on given terms and settings.
    Includes rate limiting, error handling, and retry logic.
    """
    # Initialize variables
    ua = UserAgent()
    results = []
    total_leads = 0
    domains_processed = set()
    processed_emails_per_domain = {}
    error_count = 0
    MAX_ERRORS = 50
    DELAY_BETWEEN_REQUESTS = 1.0  # seconds
    
    try:
        for original_term in terms:
            if error_count >= MAX_ERRORS:
                update_log(log_container, "Too many errors encountered. Stopping search.", 'error', search_process_id)
                break
                
            try:
                # Get or create search term
                search_term_id = add_or_get_search_term(session, original_term, st.session_state.get('active_campaign_id', 1))
                
                # Process search term
                search_term = shuffle_keywords(original_term) if shuffle_keywords_option else original_term
                search_term = optimize_search_term(search_term, 'english' if optimize_english else 'spanish') if optimize_english or optimize_spanish else search_term
                
                update_log(log_container, f"Searching for '{original_term}' (Used '{search_term}')", search_process_id=search_process_id)
                
                # Perform Google search with retry logic
                try:
                    search_results = list(google_search(search_term, num_results, lang=language))
                except Exception as e:
                    update_log(log_container, f"Search error for term '{original_term}': {str(e)}", 'error', search_process_id)
                    error_count += 1
                    continue
                
                for url in search_results:
                    if error_count >= MAX_ERRORS:
                        break
                        
                    domain = get_domain_from_url(url)
                    
                    if ignore_previously_fetched and domain in domains_processed:
                        update_log(log_container, f"Skipping Previously Fetched: {domain}", 'warning', search_process_id)
                        continue
                    
                    update_log(log_container, f"Fetching: {url}", search_process_id=search_process_id)
                    
                    try:
                        # Normalize URL
                        if not url.startswith(('http://', 'https://')):
                            url = 'http://' + url
                        
                        # Make request with retry logic and timeout
                        for attempt in range(3):
                            try:
                                response = requests.get(
                                    url,
                                    timeout=10,
                                    verify=False,
                                    headers={'User-Agent': ua.random}
                                )
                                response.raise_for_status()
                                break
                            except requests.RequestException as e:
                                if attempt == 2:  # Last attempt
                                    raise
                                time.sleep(2 ** attempt)  # Exponential backoff
                        
                        html_content = response.text
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Extract emails with validation
                        emails = extract_emails_from_html(html_content)
                        valid_emails = []
                        for email in emails:
                            try:
                                if is_valid_email(email):
                                    valid_emails.append(email)
                            except Exception:
                                continue
                        
                        update_log(log_container, f"Found {len(valid_emails)} valid email(s) on {url}", 'success', search_process_id)
                        
                        if not valid_emails:
                            continue
                        
                        # Extract page info safely
                        try:
                            name, company, job_title = extract_info_from_page(soup)
                            page_title = get_page_title(html_content)
                            page_description = get_page_description(html_content)
                        except Exception as e:
                            logging.warning(f"Error extracting page info: {str(e)}")
                            name, company, job_title = '', '', ''
                            page_title = ''
                            page_description = ''
                        
                        # Initialize domain tracking
                        if domain not in processed_emails_per_domain:
                            processed_emails_per_domain[domain] = set()
                        
                        # Process each valid email
                        for email in valid_emails:
                            if email in processed_emails_per_domain[domain]:
                                continue
                            
                            processed_emails_per_domain[domain].add(email)
                            
                            # Save lead with retry logic
                            for attempt in range(3):
                                try:
                                    lead = save_lead(
                                        session,
                                        email=email,
                                        first_name=name,
                                        company=company,
                                        job_title=job_title,
                                        url=url,
                                        search_term_id=search_term_id,
                                        created_at=datetime.utcnow()
                                    )
                                    break
                                except Exception as e:
                                    if attempt == 2:  # Last attempt
                                        raise
                                    time.sleep(2 ** attempt)
                            
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
                                
                                update_log(log_container, f"Saved lead: {email}", 'success', search_process_id)
                                
                                # Handle email sending if enabled
                                if enable_email_sending and from_email and email_template:
                                    try:
                                        template = session.query(EmailTemplate).filter_by(id=int(email_template.split(":")[0])).first()
                                        
                                        if template:
                                            wrapped_content = wrap_email_body(template.body_content)
                                            response, tracking_id = send_email_ses(
                                                session,
                                                from_email,
                                                email,
                                                template.subject,
                                                wrapped_content,
                                                reply_to=reply_to
                                            )
                                            
                                            if response:
                                                update_log(log_container, f"Sent email to: {email}", 'email_sent', search_process_id)
                                                save_email_campaign(
                                                    session,
                                                    email,
                                                    template.id,
                                                    'Sent',
                                                    datetime.utcnow(),
                                                    template.subject,
                                                    response['MessageId'],
                                                    wrapped_content
                                                )
                                            else:
                                                update_log(log_container, f"Failed to send email to: {email}", 'error', search_process_id)
                                                save_email_campaign(
                                                    session,
                                                    email,
                                                    template.id,
                                                    'Failed',
                                                    datetime.utcnow(),
                                                    template.subject,
                                                    None,
                                                    wrapped_content
                                                )
                                    except Exception as e:
                                        logging.error(f"Error sending email: {str(e)}")
                                        error_count += 1
                        
                        domains_processed.add(domain)
                        time.sleep(DELAY_BETWEEN_REQUESTS)  # Rate limiting
                        
                    except Exception as e:
                        error_count += 1
                        update_log(log_container, f"Error processing URL {url}: {str(e)}", 'error', search_process_id)
                        
            except Exception as e:
                error_count += 1
                update_log(log_container, f"Error processing term '{original_term}': {str(e)}", 'error', search_process_id)
        
        update_log(log_container, f"Total leads found: {total_leads}", 'info', search_process_id)
        if error_count > 0:
            update_log(log_container, f"Total errors encountered: {error_count}", 'warning', search_process_id)
            
        return {
            "total_leads": total_leads,
            "results": results,
            "error_count": error_count,
            "domains_processed": len(domains_processed)
        }
        
    except Exception as e:
        logging.error(f"Critical error in manual search: {str(e)}")
        update_log(log_container, f"Critical error: {str(e)}", 'error', search_process_id)
        return {
            "total_leads": total_leads,
            "results": results,
            "error_count": error_count + 1,
            "domains_processed": len(domains_processed)
        }

# Streamlit UI Components
@monitor_metrics
def manual_search_page():
    """Renders the manual search page in Streamlit with monitoring."""
    TOTAL_SEARCHES.inc()
    ACTIVE_SEARCHES.inc()
    try:
        return manual_search_page_impl()
    finally:
        ACTIVE_SEARCHES.dec()

def manual_search_page_impl():
    """Implementation of manual search page."""
    st.title("Manual Search")
    
    # Initialize worker thread if needed
    if not hasattr(st.session_state, 'worker_thread') or not st.session_state.worker_thread.is_alive():
        st.session_state.worker_thread = SearchWorkerThread(engine, SessionLocal)
        st.session_state.worker_thread.start()
    
    # Add system status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        if check_database_connection():
            st.success("Database: Connected")
        else:
            st.error("Database: Disconnected")
    
    with col2:
        if check_email_service():
            st.success("Email Service: Available")
        else:
            st.error("Email Service: Unavailable")
    
    with col3:
        worker_status = "Active" if hasattr(st.session_state, 'worker_thread') and st.session_state.worker_thread.is_alive() else "Inactive"
        st.info(f"Worker Status: {worker_status}")
    
    with db_session() as session:
        # Show active processes
        active_processes = session.query(SearchProcess).filter(
            SearchProcess.status.in_(['pending', 'running', 'completed'])
        ).order_by(SearchProcess.created_at.desc()).limit(5).all()
        
        if active_processes:
            st.subheader("Background Processes")
            for process in active_processes:
                with st.expander(f"Process {process.id} - {process.status.title()} - Started at {process.started_at.strftime('%Y-%m-%d %H:%M:%S') if process.started_at else 'Not started'}"):
                    display_process_logs(process.id)

        # Get recent searches and templates
        recent_searches = session.query(SearchTerm).order_by(SearchTerm.created_at.desc()).limit(5).all()
        recent_search_terms = [term.term for term in recent_searches]
        email_templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)

    # Search interface
    col1, col2 = st.columns([2, 1])

    with col1:
        search_terms = st_tags(
            label='Enter search terms:',
            text='Press enter to add more',
            value=recent_search_terms,
            suggestions=['software engineer', 'data scientist', 'product manager'],
            maxtags=10,
            key='search_terms_input'
        )
        num_results = st.slider("Results per term", 1, 50000, 10)

    with col2:
        enable_email_sending = st.checkbox("Enable email sending", value=True)
        ignore_previously_fetched = st.checkbox("Ignore fetched domains", value=True)
        shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=True)
        optimize_english = st.checkbox("Optimize (English)", value=False)
        optimize_spanish = st.checkbox("Optimize (Spanish)", value=False)
        language = st.selectbox("Select Language", options=["ES", "EN"], index=0)
        run_in_background = st.checkbox("Run in background", value=False)

    # Email settings if enabled
    if enable_email_sending:
        if not email_templates:
            st.error("No email templates available. Please create a template first.")
            return
        if not email_settings:
            st.error("No email settings available. Please add email settings first.")
            return

        col3, col4 = st.columns(2)
        with col3:
            email_template = st.selectbox(
                "Email template", 
                options=email_templates, 
                format_func=lambda x: x.split(":")[1].strip()
            )
        with col4:
            email_setting_option = st.selectbox(
                "From Email", 
                options=email_settings, 
                format_func=lambda x: f"{x['name']} ({x['email']})"
            )
            if email_setting_option:
                from_email = email_setting_option['email']
                reply_to = st.text_input("Reply To", email_setting_option['email'])
            else:
                st.error("No email setting selected. Please select an email setting.")
                return

    # Search button and execution
    if st.button("Search"):
        if not search_terms:
            st.warning("Enter at least one search term.")
            return

        with db_session() as session:
            # Create search process
            search_process = SearchProcess(
                campaign_id=st.session_state.get('active_campaign_id', 1),
                search_terms=search_terms,
                num_results=num_results,
                status='pending',
                settings={
                    'ignore_previously_fetched': ignore_previously_fetched,
                    'optimize_english': optimize_english,
                    'optimize_spanish': optimize_spanish,
                    'shuffle_keywords_option': shuffle_keywords_option,
                    'language': language,
                    'enable_email_sending': enable_email_sending,
                    'from_email': from_email if enable_email_sending else None,
                    'reply_to': reply_to if enable_email_sending else None,
                    'email_template': email_template if enable_email_sending else None
                }
            )
            session.add(search_process)
            session.commit()
            
            st.success(f"Search process {search_process.id} submitted to worker")
            display_process_logs(search_process.id)

def display_process_logs(process_id):
    """Displays logs for a search process."""
    with db_session() as session:
        process = session.query(SearchProcess).get(process_id)
        if process and process.logs:
            for log in process.logs:
                icon = {
                    'info': '🔵',
                    'success': '🟢',
                    'warning': '🟠',
                    'error': '🔴',
                    'email_sent': '🟣'
                }.get(log.get('level', 'info'), '⚪')
                st.write(f"{icon} {log.get('message', '')}")

def fetch_email_templates(session):
    """Fetches available email templates."""
    templates = session.query(EmailTemplate).all()
    return [f"{t.id}: {t.template_name}" for t in templates]

def fetch_email_settings(session):
    """Fetches available email settings."""
    settings = session.query(EmailSettings).all()
    return [{'id': s.id, 'name': s.name, 'email': s.email} for s in settings]

def add_or_get_search_term(session, term, campaign_id):
    """Adds a new search term or gets existing one."""
    existing_term = session.query(SearchTerm).filter_by(term=term, campaign_id=campaign_id).first()
    if existing_term:
        return existing_term.id
    
    new_term = SearchTerm(term=term, campaign_id=campaign_id)
    session.add(new_term)
    session.commit()
    return new_term.id

def save_lead(session, email, **kwargs):
    """Saves a new lead or updates existing one."""
    try:
        lead = session.query(Lead).filter_by(email=email).first()
        if not lead:
            lead = Lead(email=email, **kwargs)
            session.add(lead)
        else:
            for key, value in kwargs.items():
                if value:
                    setattr(lead, key, value)
        session.commit()
        return lead
    except Exception as e:
        logging.error(f"Error saving lead: {str(e)}")
        session.rollback()
        return None

def settings_page():
    """Renders the settings page in Streamlit."""
    st.title("Settings")
    
    with db_session() as session:
        # General Settings
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings:
            general_settings = Settings(
                name='General Settings',
                setting_type='general',
                value={}
            )
        
        st.header("General Settings")
        with st.form("general_settings_form"):
            openai_api_key = st.text_input(
                "OpenAI API Key",
                value=general_settings.value.get('openai_api_key', ''),
                type="password"
            )
            openai_api_base = st.text_input(
                "OpenAI API Base URL",
                value=general_settings.value.get('openai_api_base', 'https://api.openai.com/v1')
            )
            openai_model = st.text_input(
                "OpenAI Model",
                value=general_settings.value.get('openai_model', 'gpt-4-1106-preview')
            )
            
            if st.form_submit_button("Save General Settings"):
                general_settings.value = {
                    'openai_api_key': openai_api_key,
                    'openai_api_base': openai_api_base,
                    'openai_model': openai_model
                }
                session.add(general_settings)
                session.commit()
                st.success("General settings saved successfully!")

        # Email Settings
        st.header("Email Settings")
        email_settings = session.query(EmailSettings).all()
        
        # Display existing settings
        for setting in email_settings:
            with st.expander(f"{setting.name} ({setting.email})"):
                st.write(f"Provider: {setting.provider}")
                st.write(
                    f"{'SMTP Server: ' + setting.smtp_server if setting.provider == 'smtp' else 'AWS Region: ' + setting.aws_region}"
                )
                if st.button(f"Delete {setting.name}", key=f"delete_{setting.id}"):
                    session.delete(setting)
                    session.commit()
                    st.success(f"Deleted {setting.name}")
                    st.rerun()

        # Edit or create new settings
        edit_id = st.selectbox(
            "Edit existing setting",
            ["New Setting"] + [f"{s.id}: {s.name}" for s in email_settings]
        )
        edit_setting = session.query(EmailSettings).get(int(edit_id.split(":")[0])) if edit_id != "New Setting" else None
        
        with st.form("email_setting_form"):
            name = st.text_input(
                "Name",
                value=edit_setting.name if edit_setting else "",
                placeholder="e.g., Company Gmail"
            )
            email = st.text_input(
                "Email",
                value=edit_setting.email if edit_setting else "",
                placeholder="your.email@example.com"
            )
            provider = st.selectbox(
                "Provider",
                ["smtp", "ses"],
                index=0 if edit_setting and edit_setting.provider == "smtp" else 1
            )
            
            if provider == "smtp":
                smtp_server = st.text_input(
                    "SMTP Server",
                    value=edit_setting.smtp_server if edit_setting else "",
                    placeholder="smtp.gmail.com"
                )
                smtp_port = st.number_input(
                    "SMTP Port",
                    min_value=1,
                    max_value=65535,
                    value=edit_setting.smtp_port if edit_setting else 587
                )
                smtp_username = st.text_input(
                    "SMTP Username",
                    value=edit_setting.smtp_username if edit_setting else "",
                    placeholder="your.email@gmail.com"
                )
                smtp_password = st.text_input(
                    "SMTP Password",
                    type="password",
                    value=edit_setting.smtp_password if edit_setting else "",
                    placeholder="Your SMTP password"
                )
            else:
                aws_access_key_id = st.text_input(
                    "AWS Access Key ID",
                    value=edit_setting.aws_access_key_id if edit_setting else "",
                    placeholder="AKIAIOSFODNN7EXAMPLE"
                )
                aws_secret_access_key = st.text_input(
                    "AWS Secret Access Key",
                    type="password",
                    value=edit_setting.aws_secret_access_key if edit_setting else "",
                    placeholder="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
                )
                aws_region = st.text_input(
                    "AWS Region",
                    value=edit_setting.aws_region if edit_setting else "",
                    placeholder="us-west-2"
                )
            
            if st.form_submit_button("Save Email Setting"):
                try:
                    setting_data = {
                        'name': name,
                        'email': email,
                        'provider': provider
                    }
                    
                    if provider == 'smtp':
                        setting_data.update({
                            'smtp_server': smtp_server,
                            'smtp_port': smtp_port,
                            'smtp_username': smtp_username,
                            'smtp_password': smtp_password
                        })
                    else:
                        setting_data.update({
                            'aws_access_key_id': aws_access_key_id,
                            'aws_secret_access_key': aws_secret_access_key,
                            'aws_region': aws_region
                        })
                    
                    if edit_setting:
                        for k, v in setting_data.items():
                            setattr(edit_setting, k, v)
                    else:
                        new_setting = EmailSettings(**setting_data)
                        session.add(new_setting)
                    
                    session.commit()
                    st.success("Email setting saved successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error saving email setting: {str(e)}")
                    session.rollback()

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
            
            if settings.provider == 'ses':
                aws_session = boto3.Session(
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.aws_region
                )
                ses_client = aws_session.client('ses')
                ses_client.get_send_quota()
            else:
                with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
                    server.starttls()
                    server.login(settings.smtp_username, settings.smtp_password)
            return True
    except Exception as e:
        logging.error(f"Email service check error: {str(e)}")
        return False

# Main Streamlit app
def main():
    """Main Streamlit application with production features."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    
    # Setup monitoring
    setup_monitoring()
    
    # Configure Streamlit
    st.set_page_config(
        page_title="Email Lead Finder",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'active_campaign_id' not in st.session_state:
        st.session_state.active_campaign_id = 1
    if 'active_project_id' not in st.session_state:
        st.session_state.active_project_id = 1
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    
    try:
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Manual Search", "Settings", "Monitoring"],
            icons=["search", "gear", "graph-up"],
            orientation="horizontal",
        )
        
        if selected == "Manual Search":
            manual_search_page()
        elif selected == "Settings":
            settings_page()
        elif selected == "Monitoring":
            monitoring_page()
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if sentry_sdk.Hub.current.client:
            sentry_sdk.capture_exception(e)
        logging.exception("Unhandled exception in main app")

def monitoring_page():
    """Display monitoring information."""
    st.title("System Monitoring")
    
    # System Status
    st.header("System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Searches", ACTIVE_SEARCHES._value.get())
    with col2:
        st.metric("Total Searches", TOTAL_SEARCHES._value.get())
    with col3:
        st.metric("Total Leads", TOTAL_LEADS._value.get())
    
    # Error Rate
    st.header("Error Statistics")
    error_rate = (ERROR_COUNT._value.get() / max(TOTAL_SEARCHES._value.get(), 1)) * 100
    st.metric("Error Rate", f"{error_rate:.2f}%")
    
    # Recent Logs
    st.header("Recent Logs")
    try:
        with open('app.log', 'r') as f:
            logs = f.readlines()[-50:]  # Last 50 lines
            for log in logs:
                st.text(log.strip())
    except Exception as e:
        st.error(f"Error reading logs: {str(e)}")
    
    # Database Stats
    st.header("Database Statistics")
    with db_session() as session:
        try:
            lead_count = session.query(func.count(Lead.id)).scalar()
            campaign_count = session.query(func.count(Campaign.id)).scalar()
            email_sent_count = session.query(func.count(EmailCampaign.id)).filter_by(status='Sent').scalar()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Leads", lead_count)
            with col2:
                st.metric("Active Campaigns", campaign_count)
            with col3:
                st.metric("Emails Sent", email_sent_count)
        except Exception as e:
            st.error(f"Error fetching database stats: {str(e)}")

if __name__ == "__main__":
    main()
