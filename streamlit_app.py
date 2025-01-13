import os, json, re, logging, asyncio, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid, aiohttp, urllib3, random, html, smtplib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import (
    func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, 
    Boolean, JSON, select, text, distinct, and_, or_, inspect
)
from sqlalchemy.orm import (
    declarative_base, sessionmaker, relationship, Session, joinedload,
    configure_mappers
)
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from openai import OpenAI 
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
import plotly.express as px
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import subprocess
import threading
from threading import local
import signal

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
    project_id = Column(BigInteger, ForeignKey('projects.id'))
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
    provider = Column(Text, nullable=False)
    smtp_server = Column(Text)
    smtp_port = Column(BigInteger)
    smtp_username = Column(Text)
    smtp_password = Column(Text)
    aws_access_key_id = Column(Text)
    aws_secret_access_key = Column(Text)
    aws_region = Column(Text)
    daily_limit = Column(BigInteger)  # Add daily limit column
    hourly_limit = Column(BigInteger)  # Add hourly limit column
    is_active = Column(Boolean, default=True)  # Add is_active column

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
        raise
    finally:
        session.close()

def settings_page():
    st.title("Settings")
    with db_session() as session:
        try:
            # Email Settings
            st.subheader("Email Settings")
            
            if st.button("Create Default Settings"):
                try:
                    initialize_settings()
                    st.success("Default settings created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving email setting: {str(e)}")
                    session.rollback()
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")

def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
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
        test_response = requests.get(f"https://autoclient-email-analytics.trigox.workers.dev/test", timeout=5)
        if test_response.status_code != 200:
            logging.warning("Analytics worker is down. Using original URLs.")
            tracked_body = wrapped_body
    except requests.RequestException:
        logging.warning("Failed to reach analytics worker. Using original URLs.")
        tracked_body = wrapped_body

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

def update_log(log_container, message, level='info'):
    # Set icon based on level
    icon = {
        'info': '📝',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    }.get(level, '📝')
    
    log_entry = f"{icon} {message}"
    
    # Simple console logging without HTML
    print(f"{icon} {message.split('<')[0]}")  # Only print the first part of the message before any HTML tags
    
    # Initialize log entries in session state if not present
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    
    # HTML-formatted log entry for Streamlit display
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_html = f"""
    <div style='margin-bottom: 5px; padding: 8px; border-radius: 4px; background-color: {"#e8f4ea" if level == "success" else "#fff3e6" if level == "warning" else "#ffe6e6" if level == "error" else "#f0f2f6"}'>
        <span style='opacity: 0.6'>{timestamp}</span> {icon} {message}
    </div>
    """
    
    try:
        # Add to session state
        st.session_state.log_entries.append(log_html)
        
        # Update display
        log_container.markdown(log_html, unsafe_allow_html=True)
    except Exception as e:
        print(f"Error updating log: {str(e)}")

def optimize_search_term(search_term, language):
    """Optimizes search term based on language."""
    if language == 'english':
        # Remove site restriction and add more contact-related terms
        return f'"{search_term}" (email OR contact OR "get in touch" OR "reach out" OR "contact us" OR "contact form" OR "send message" OR "send us a message" OR "write to us" OR "get in contact")'
    elif language == 'spanish':
        # Remove site restriction and add more contact-related terms
        return f'"{search_term}" (correo OR contacto OR email OR "ponte en contacto" OR "formulario de contacto" OR "enviar mensaje" OR "envíanos un mensaje" OR "escríbenos" OR "contacta con nosotros")'
    return search_term

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def get_domain_from_url(url):
    return urlparse(url).netloc

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def extract_emails_from_html(html_content):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, html_content)

def extract_info_from_page(soup):
    name = soup.find('meta', {'name': 'author'})
    name = name['content'] if name else ''
    
    company = soup.find('meta', {'property': 'og:site_name'})
    company = company['content'] if company else ''
    
    job_title = soup.find('meta', {'name': 'job_title'})
    job_title = job_title['content'] if job_title else ''
    
    return name, company, job_title

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
                    html_content, soup = response.text, BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract all emails from the page
                    emails = extract_emails_from_html(html_content)
                    valid_emails = [email for email in emails if is_valid_email(email)]
                    update_log(log_container, f"Found {len(valid_emails)} valid email(s) on {url}", 'success')
                    
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

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:
            st.error("OpenAI API key not set. Please configure it in the settings.")
            return None

        client = OpenAI(api_key=general_settings.value['openai_api_key'])
        model = general_settings.value.get('openai_model', "gpt-4o-mini")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
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
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=get_active_campaign_id(), created_at=created_at or datetime.utcnow())
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
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))
        
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

def manual_search_page():
    try:
        st.title("Manual Search")
        
        with st.form("search_form"):
            search_terms = st_tags(
                label='Enter Search Terms:',
                text='Press enter to add more',
                value=[],
                maxtags=10,
                key='manual_search_terms'
            )

            col1, col2 = st.columns(2)
            with col1:
                shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=False)
                optimize_english = st.checkbox("Optimize for English", value=False)
            with col2:
                optimize_spanish = st.checkbox("Optimize for Spanish", value=False)

            submitted = st.form_submit_button("Start Search")

        if submitted and search_terms:
            with st.spinner("Searching..."):
                log_container = st.empty()
                results_container = st.container()
                
                try:
                    with db_session() as session:
                        results = perform_search(
                            session,
                            search_terms,
                            log_container,
                            shuffle_keywords_option,
                            optimize_english,
                            optimize_spanish
                        )
                        
                        if results["total_leads"] > 0:
                            with results_container:
                                st.success(f"Found {results['total_leads']} new leads!")
                                df = pd.DataFrame(results["results"])
                                st.dataframe(df)
                        else:
                            st.warning("No new leads found.")
                            
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error loading manual search page: {str(e)}")

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

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:
            st.error("OpenAI API key not set. Please configure it in the settings.")
            return None

        client = OpenAI(api_key=general_settings.value['openai_api_key'])
        model = general_settings.value.get('openai_model', "gpt-4o-mini")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
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
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=get_active_campaign_id(), created_at=created_at or datetime.utcnow())
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
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))
        
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

def manual_search_page():
    try:
        st.title("Manual Search")
        
        with st.form("search_form"):
            search_terms = st_tags(
                label='Enter Search Terms:',
                text='Press enter to add more',
                value=[],
                maxtags=10,
                key='manual_search_terms'
            )

            col1, col2 = st.columns(2)
            with col1:
                shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=False)
                optimize_english = st.checkbox("Optimize for English", value=False)
            with col2:
                optimize_spanish = st.checkbox("Optimize for Spanish", value=False)

            submitted = st.form_submit_button("Start Search")

        if submitted and search_terms:
            with st.spinner("Searching..."):
                log_container = st.empty()
                results_container = st.container()
                
                try:
                    with db_session() as session:
                        results = perform_search(
                            session,
                            search_terms,
                            log_container,
                            shuffle_keywords_option,
                            optimize_english,
                            optimize_spanish
                        )
                        
                        if results["total_leads"] > 0:
                            with results_container:
                                st.success(f"Found {results['total_leads']} new leads!")
                                df = pd.DataFrame(results["results"])
                                st.dataframe(df)
                        else:
                            st.warning("No new leads found.")
                            
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error loading manual search page: {str(e)}")

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

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:
            st.error("OpenAI API key not set. Please configure it in the settings.")
            return None

        client = OpenAI(api_key=general_settings.value['openai_api_key'])
        model = general_settings.value.get('openai_model', "gpt-4o-mini")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
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
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=get_active_campaign_id(), created_at=created_at or datetime.utcnow())
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
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))
        
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

def manual_search_page():
    try:
        st.title("Manual Search")
        
        with st.form("search_form"):
            search_terms = st_tags(
                label='Enter Search Terms:',
                text='Press enter to add more',
                value=[],
                maxtags=10,
                key='manual_search_terms'
            )

            col1, col2 = st.columns(2)
            with col1:
                shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=False)
                optimize_english = st.checkbox("Optimize for English", value=False)
            with col2:
                optimize_spanish = st.checkbox("Optimize for Spanish", value=False)

            submitted = st.form_submit_button("Start Search")

        if submitted and search_terms:
            with st.spinner("Searching..."):
                log_container = st.empty()
                results_container = st.container()
                
                try:
                    with db_session() as session:
                        results = perform_search(
                            session,
                            search_terms,
                            log_container,
                            shuffle_keywords_option,
                            optimize_english,
                            optimize_spanish
                        )
                        
                        if results["total_leads"] > 0:
                            with results_container:
                                st.success(f"Found {results['total_leads']} new leads!")
                                df = pd.DataFrame(results["results"])
                                st.dataframe(df)
                        else:
                            st.warning("No new leads found.")
                            
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error loading manual search page: {str(e)}")

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

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:
            st.error("OpenAI API key not set. Please configure it in the settings.")
            return None

        client = OpenAI(api_key=general_settings.value['openai_api_key'])
        model = general_settings.value.get('openai_model', "gpt-4o-mini")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
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
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=get_active_campaign_id(), created_at=created_at or datetime.utcnow())
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
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))
        
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

def manual_search_page():
    try:
        st.title("Manual Search")
        
        with st.form("search_form"):
            search_terms = st_tags(
                label='Enter Search Terms:',
                text='Press enter to add more',
                value=[],
                maxtags=10,
                key='manual_search_terms'
            )

            col1, col2 = st.columns(2)
            with col1:
                shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=False)
                optimize_english = st.checkbox("Optimize for English", value=False)
            with col2:
                optimize_spanish = st.checkbox("Optimize for Spanish", value=False)

            submitted = st.form_submit_button("Start Search")

        if submitted and search_terms:
            with st.spinner("Searching..."):
                log_container = st.empty()
                results_container = st.container()
                
                try:
                    with db_session() as session:
                        results = perform_search(
                            session,
                            search_terms,
                            log_container,
                            shuffle_keywords_option,
                            optimize_english,
                            optimize_spanish
                        )
                        
                        if results["total_leads"] > 0:
                            with results_container:
                                st.success(f"Found {results['total_leads']} new leads!")
                                df = pd.DataFrame(results["results"])
                                st.dataframe(df)
                        else:
                            st.warning("No new leads found.")
                            
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error loading manual search page: {str(e)}")

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

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:
            st.error("OpenAI API key not set. Please configure it in the settings.")
            return None

        client = OpenAI(api_key=general_settings.value['openai_api_key'])
        model = general_settings.value.get('openai_model', "gpt-4o-mini")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
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
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=get_active_campaign_id(), created_at=created_at or datetime.utcnow())
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
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))
        
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

def manual_search_page():
    try:
        st.title("Manual Search")
        
        with st.form("search_form"):
            search_terms = st_tags(
                label='Enter Search Terms:',
                text='Press enter to add more',
                value=[],
                maxtags=10,
                key='manual_search_terms'
            )

            col1, col2 = st.columns(2)
            with col1:
                shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=False)
                optimize_english = st.checkbox("Optimize for English", value=False)
            with col2:
                optimize_spanish = st.checkbox("Optimize for Spanish", value=False)

            submitted = st.form_submit_button("Start Search")

        if submitted and search_terms:
            with st.spinner("Searching..."):
                log_container = st.empty()
                results_container = st.container()
                
                try:
                    with db_session() as session:
                        results = perform_search(
                            session,
                            search_terms,
                            log_container,
                            shuffle_keywords_option,
                            optimize_english,
                            optimize_spanish
                        )
                        
                        if results["total_leads"] > 0:
                            with results_container:
                                st.success(f"Found {results['total_leads']} new leads!")
                                df = pd.DataFrame(results["results"])
                                st.dataframe(df)
                        else:
                            st.warning("No new leads found.")
                            
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error loading manual search page: {str(e)}")

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

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:
            st.error("OpenAI API key not set. Please configure it in the settings.")
            return None

        client = OpenAI(api_key=general_settings.value['openai_api_key'])
        model = general_settings.value.get('openai_model', "gpt-4o-mini")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
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
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=get_active_campaign_id(), created_at=created_at or datetime.utcnow())
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
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))
        
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

def manual_search_page():
    try:
        st.title("Manual Search")
        
        with st.form("search_form"):
            search_terms = st_tags(
                label='Enter Search Terms:',
                text='Press enter to add more',
                value=[],
                maxtags=10,
                key='manual_search_terms'
            )

            col1, col2 = st.columns(2)
            with col1:
                shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=False)
                optimize_english = st.checkbox("Optimize for English", value=False)
            with col2:
                optimize_spanish = st.checkbox("Optimize for Spanish", value=False)

            submitted = st.form_submit_button("Start Search")

        if submitted and search_terms:
            with st.spinner("Searching..."):
                log_container = st.empty()
                results_container = st.container()
                
                try:
                    with db_session() as session:
                        results = perform_search(
                            session,
                            search_terms,
                            log_container,
                            shuffle_keywords_option,
                            optimize_english,
                            optimize_spanish
                        )
                        
                        if results["total_leads"] > 0:
                            with results_container:
                                st.success(f"Found {results['total_leads']} new leads!")
                                df = pd.DataFrame(results["results"])
                                st.dataframe(df)
                        else:
                            st.warning("No new leads found.")
                            
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error loading manual search page: {str(e)}")

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

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        st.error(f"An error occurred: {str(e)}")
        logging.exception("An error occurred in the main function")
        st.write("Please try refreshing the page or contact support if the issue persists.")

    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")

if __name__ == "__main__":
    main()