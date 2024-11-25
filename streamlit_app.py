import os, json, logging, time, pandas as pd, streamlit as st, openai, boto3, urllib3
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select, text, and_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags
import plotly.express as px
from contextlib import contextmanager
from typing import Optional, Dict, List, Any, Union
from sqlalchemy.sql.expression import distinct
from email_validator import validate_email, EmailNotValidError
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode
import uuid
import threading
from ratelimit import limits, RateLimiter
import requests
from fake_useragent import UserAgent
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import json
import threading
from typing import Dict, List
from bs4 import BeautifulSoup
import plotly.express as px
from streamlit_extras.metric_cards import metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stoggle import stoggle
from streamlit_extras.let_it_rain import rain
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.dataframe_explorer import dataframe_explorer

# Add at top of file
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()
DB_CONFIG = {
    key: os.getenv(f"SUPABASE_DB_{key}")
    for key in ["HOST", "NAME", "USER", "PASSWORD", "PORT"]
}

if not all(DB_CONFIG.values()):
    raise ValueError("Missing database configuration")

DATABASE_URL = "postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{NAME}".format(**DB_CONFIG)

# Single engine instance with optimal configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 10}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

# Add connection pooling and retry logic
def get_db_engine():
    return create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,  # Recycle connections every 30 minutes
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10}
    )

# Add connection health check
def check_db_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logging.error(f"Database connection check failed: {e}")
        return False

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
    phone, first_name, last_name, company, job_title = [Column(Text) for _ in range(5)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Remove the domain column
    campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead")
    email_campaigns = relationship("EmailCampaign", back_populates="lead")

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
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    leads_gathered, emails_sent = Column(BigInteger), Column(BigInteger)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    status, logs = Column(Text), Column(JSON)
    campaign = relationship("Campaign")
    search_term = relationship("SearchTerm")

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

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set")

@contextmanager
def safe_db_session():
    session = None
    try:
        session = SessionLocal()
        yield session
        session.commit()
    except Exception as e:
        if session:
            session.rollback()
        logging.error(f"Database error: {str(e)}")
        raise
    finally:
        if session:
            session.close()

def settings_page():
    st.header("Settings")
    with st.form("settings"):
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        email_provider = st.selectbox("Email Provider", ["SMTP", "AWS SES"])
        if email_provider == "SMTP":
            smtp_server = st.text_input("SMTP Server")
            smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535)
            smtp_user = st.text_input("SMTP Username")
            smtp_pass = st.text_input("SMTP Password", type="password")
        else:
            aws_key = st.text_input("AWS Access Key")
            aws_secret = st.text_input("AWS Secret Key", type="password")
            aws_region = st.text_input("AWS Region")
        if st.form_submit_button("Save Settings"):
            with safe_db_session() as session:
                settings = Settings(
                    name="email_settings",
                    setting_type="email",
                    value=json.dumps({
                        "provider": email_provider,
                        "smtp_server": smtp_server if email_provider == "SMTP" else None,
                        "smtp_port": smtp_port if email_provider == "SMTP" else None,
                        "smtp_username": smtp_user if email_provider == "SMTP" else None,
                        "smtp_password": smtp_pass if email_provider == "SMTP" else None,
                        "aws_access_key_id": aws_key if email_provider == "AWS SES" else None,
                        "aws_secret_access_key": aws_secret if email_provider == "AWS SES" else None,
                        "aws_region": aws_region if email_provider == "AWS SES" else None
                    })
                )
                session.add(settings)
                session.commit()
                st.success("Settings saved successfully!")

def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings:
        logging.error(f"No email settings found for {from_email}")
        return None, None

    tracking_id = str(uuid.uuid4())
    tracked_body = add_tracking_to_email(body, tracking_id)

    try:
        if email_settings.provider == 'ses':
            if not all([email_settings.aws_access_key_id, email_settings.aws_secret_access_key, email_settings.aws_region]):
                raise ValueError("Missing AWS credentials")
                
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

        elif email_settings.provider == 'smtp':
            success, error = send_email_smtp(
                server=email_settings.smtp_server,
                port=email_settings.smtp_port,
                username=email_settings.smtp_username,
                password=email_settings.smtp_password,
                from_email=from_email,
                to_email=to_email,
                subject=subject,
                body=tracked_body,
                reply_to=reply_to
            )
            
            if success:
                return {'MessageId': f'smtp-{uuid.uuid4()}'}, tracking_id
            else:
                logging.error(f"SMTP error: {error}")
                return None, None
                
        else:
            raise ValueError(f"Unknown email provider: {email_settings.provider}")

    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        return None, None

def send_email_smtp(server, port, username, password, from_email, to_email, subject, body, reply_to=None):
    """Enhanced SMTP sending function with proper TLS handling"""
    try:
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        if reply_to:
            msg['Reply-To'] = reply_to
        msg.attach(MIMEText(body, 'html'))

        # Handle different SMTP configurations
        if port == 465:  # SSL
            smtp = smtplib.SMTP_SSL(server, port, timeout=10)
        else:  # TLS or plain
            smtp = smtplib.SMTP(server, port, timeout=10)
            if port == 587:  # Explicit TLS
                smtp.starttls()
        
        # Some servers need EHLO
        smtp.ehlo()
        smtp.login(username, password)
        smtp.send_message(msg)
        smtp.quit()
        
        return True, None
    except Exception as e:
        logging.error(f"SMTP error: {str(e)}")
        return False, str(e)

def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    try:
        with session.begin_nested():  # Use savepoint for atomic operation
            lead = session.query(Lead).filter_by(email=lead_email).first()
            if not lead:
                raise ValueError(f"Lead not found: {lead_email}")
            
            campaign = EmailCampaign(
                lead_id=lead.id,
                template_id=template_id,
                status=status,
                sent_at=sent_at,
                message_id=message_id or f"error-{uuid.uuid4()}",
                customized_subject=subject,
                customized_content=email_body,
                campaign_id=st.session_state.get('active_campaign_id'),
                tracking_id=str(uuid.uuid4())
            )
            session.add(campaign)
            session.flush()
            return campaign.id
    except Exception as e:
        logging.error(f"Failed to save campaign: {str(e)}")
        raise

def update_log(log_container, message, level='info'):
    icons = {
        'info': '🔵',
        'success': '🟢',
        'warning': '⚠️',
        'error': '🔴',
        'email_sent': '🟣'
    }
    icon = icons.get(level, '⚪')
    log_entry = f"{icon} {message}"
    
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    
    st.session_state.log_entries = st.session_state.log_entries[-99:] + [log_entry]  # Keep last 100 entries
    log_html = f"""
    <div style='height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.8em; line-height: 1.2;'>
        {'<br>'.join(st.session_state.log_entries)}
    </div>
    """
    log_container.markdown(log_html, unsafe_allow_html=True)
    print(f"{icon} {message.split('<')[0]}")  # Console logging without HTML

def optimize_search_term(search_term, language):
    if language == 'english':
        return f'"{search_term}" email OR contact OR "get in touch" site:.com'
    elif language == 'spanish':
        return f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es'
    return search_term

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def get_domain_from_url(url):
    return urlparse(url).netloc

def is_valid_email(email):
    if not email or not isinstance(email, str):
        return False
        
    # Add more comprehensive validation patterns
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js|exe|zip|rar|pdf)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply|no-reply|donotreply)@.*",
        r"^[0-9]+@.*",
        r".*@(example|test|localhost|invalid)\.",
        r".*@.*\.(test|local|invalid|example)$",
        r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$",
        r".*@(gmil|gmal|gmaill|gnail)\.com$",
        r".*@.*\.(local|test|example|invalid)$"
    ]
    
    try:
        # Use email_validator library
        email_info = validate_email(email, check_deliverability=True)
        normalized_email = email_info.normalized
        
        # Check against patterns
        if any(re.match(pattern, normalized_email, re.IGNORECASE) for pattern in invalid_patterns):
            return False
            
        return True
    except EmailNotValidError:
        return False

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

# Add rate limiting for API calls
class APIRateLimiter:
    def __init__(self, calls_per_second=2):
        self.rate_limiter = RateLimiter(max_calls=calls_per_second, period=1)
        
    @contextmanager
    def limit(self):
        try:
            with self.rate_limiter:
                yield
        except Exception as e:
            logging.error(f"Rate limit exceeded: {e}")
            raise

# Update manual_search function
def manual_search(session, terms, num_results, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False, language='ES', enable_email_sending=True, log_container=None, from_email=None, reply_to=None, email_template=None):
    rate_limiter = APIRateLimiter()
    with rate_limiter.limit():
        ua = UserAgent()
        results = []
        total_leads = 0
        domains_processed = set()

        for term in terms:
            try:
                search_term_id = add_or_get_search_term(session, term, get_active_campaign_id())
                search_term = shuffle_keywords(term) if shuffle_keywords_option else term
                search_term = optimize_search_term(search_term, 'english' if optimize_english else 'spanish') if optimize_english or optimize_spanish else search_term
                
                if log_container:
                    update_log(log_container, f"Searching for '{term}' (Used '{search_term}')")

                for url in google_search(search_term, num_results, lang=language):
                    domain = get_domain_from_url(url)
                    if ignore_previously_fetched and domain in domains_processed:
                        if log_container:
                            update_log(log_container, f"Skipping Previously Fetched: {domain}", 'warning')
                        continue

                    if log_container:
                        update_log(log_container, f"Fetching: {url}")

                    try:
                        if not url.startswith(('http://', 'https://')):
                            url = 'http://' + url
                        response = requests.get(url, timeout=10, verify=False, headers={'User-Agent': ua.random})
                        response.raise_for_status()
                        html_content = response.text
                        soup = BeautifulSoup(html_content, 'html.parser')
                        emails = extract_emails_from_html(html_content)

                        if log_container:
                            update_log(log_container, f"Found {len(emails)} email(s) on {url}", 'success')

                        for email in filter(is_valid_email, emails):
                            if domain not in domains_processed:
                                name, company, job_title = extract_info_from_page(soup)
                                lead = save_lead(session, email=email, first_name=name, company=company, 
                                              job_title=job_title, url=url, search_term_id=search_term_id)
                                
                                if lead:
                                    total_leads += 1
                                    results.append({
                                        'Email': email,
                                        'URL': url,
                                        'Lead Source': term,
                                        'Title': get_page_title(html_content),
                                        'Description': get_page_description(html_content),
                                        'Tags': [],
                                        'Name': name,
                                        'Company': company,
                                        'Job Title': job_title,
                                        'Search Term ID': search_term_id
                                    })

                                    if log_container:
                                        update_log(log_container, f"Saved lead: {email}", 'success')

                                    domains_processed.add(domain)

                                    if enable_email_sending and from_email and email_template:
                                        template = session.query(EmailTemplate).filter_by(id=int(email_template.split(":")[0])).first()
                                        if template:
                                            wrapped_content = wrap_email_body(template.body_content)
                                            response, tracking_id = send_email_ses(session, from_email, email, 
                                                                                template.subject, wrapped_content, 
                                                                                reply_to=reply_to)
                                            if response:
                                                if log_container:
                                                    update_log(log_container, f"Sent email to: {email}", 'email_sent')
                                                save_email_campaign(session, email, template.id, 'Sent', 
                                                         datetime.utcnow(), template.subject, 
                                                         response['MessageId'], wrapped_content)
                                            else:
                                                if log_container:
                                                    update_log(log_container, f"Failed to send email to: {email}", 'error')
                                                save_email_campaign(session, email, template.id, 'Failed', 
                                                         datetime.utcnow(), template.subject, None, 
                                                         wrapped_content)
                                    break

                    except requests.RequestException as e:
                        if log_container:
                            update_log(log_container, f"Error processing URL {url}: {str(e)}", 'error')

            except Exception as e:
                if log_container:
                    update_log(log_container, f"Error processing term '{term}': {str(e)}", 'error')

        if log_container:
            update_log(log_container, f"Total leads found: {total_leads}", 'info')

        return {"total_leads": total_leads, "results": results}

def generate_or_adjust_email_template(prompt, kb_info=None, current_template=None):
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in creating and refining email templates for marketing campaigns. Always respond with a JSON object containing 'subject' and 'body' keys. The 'body' should contain HTML formatted content suitable for insertion into an email body."},
        {"role": "user", "content": f"""{'Adjust the following email template based on the given instructions:' if current_template else 'Create an email template based on the following prompt:'} {prompt}

        {'Current Template:' if current_template else 'Guidelines:'}
        {current_template if current_template else '1. Focus on benefits to the reader, address potential customer doubts, include clear CTAs, use a natural tone, and be concise.'}

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
        # Split the query into multiple lines for readability
        query = (session.query(
            Lead,
            func.string_agg(LeadSource.url, ', ').label('sources'),
            func.max(EmailCampaign.sent_at).label('last_contact'),
            func.string_agg(EmailCampaign.status, ', ').label('email_statuses')
        )
        .outerjoin(LeadSource)
        .outerjoin(EmailCampaign)
        .group_by(Lead.id))
        
        results = query.all()
        
        # Process results in a more readable way
        lead_data = []
        for lead, sources, last_contact, email_statuses in results:
            lead_info = {
                k: getattr(lead, k) for k in [
                    'id', 'email', 'first_name', 'last_name',
                    'company', 'job_title', 'created_at'
                ]
            }
            lead_info.update({
                'Source': sources,
                'Last Contact': last_contact,
                'Last Email Status': (
                    email_statuses.split(', ')[-1] 
                    if email_statuses else 'Not Contacted'
                ),
                'Delete': False
            })
            lead_data.append(lead_info)
            
        return pd.DataFrame(lead_data)
        
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_leads_with_sources: {str(e)}")
        return pd.DataFrame()

def fetch_search_terms_with_lead_count(session):
    try:
        query = (session.query(
            SearchTerm.term,
            func.count(distinct(Lead.id)).label('lead_count'),
            func.count(distinct(EmailCampaign.id)).label('email_count'))
            .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
            .join(Lead, LeadSource.lead_id == Lead.id)
            .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
            .group_by(SearchTerm.term))
        
        return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_search_terms_with_lead_count: {str(e)}")
        return pd.DataFrame(columns=['Term', 'Lead Count', 'Email Count'])

def add_search_term(session, term, campaign_id):
    try:
        new_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=datetime.utcnow())
        session.add(new_term)
        session.commit()
        return new_term.id
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error adding search term: {e}")
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

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        group_id = None
        if group_for_new_term != "None":
            group_id = int(group_for_new_term.split(":")[0])
            
        new_search_term = SearchTerm(
            term=new_term,
            campaign_id=campaign_id,
            created_at=datetime.utcnow(),
            group_id=group_id
        )
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
    existing_groups = session.query(SearchTermGroup).all()
    prompt = f"Categorize these search terms into existing groups or suggest new ones:\n{', '.join([term.term for term in ungrouped_terms])}\n\nExisting groups: {', '.join([group.name for group in existing_groups])}\n\nRespond with a JSON object: {{group_name: [term1, term2, ...]}}"
    messages = [{"role": "system", "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."}, {"role": "user", "content": prompt}]
    response = openai_chat_completion(messages, function_name="ai_group_search_terms")
    return response if isinstance(response, dict) else {}

def update_search_term_groups(session, grouped_terms):
    for group_name, terms in grouped_terms.items():
        group = session.query(SearchTermGroup).filter_by(name=group_name).first() or SearchTermGroup(name=group_name)
        if not group.id: session.add(group); session.flush()
        for term in terms:
            search_term = session.query(SearchTerm).filter_by(term=term).first()
            if search_term: search_term.group_id = group.id
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
    if not hasattr(st.session_state, 'automation_lock'):
        st.session_state.automation_lock = threading.Lock()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    
    while st.session_state.get('automation_status', False):
        try:
            with st.session_state.automation_lock:
                with safe_db_session() as new_session:
                    if not check_db_connection():
                        raise ConnectionError("Database connection lost")
                        
                    kb_info = get_knowledge_base_info(new_session, get_active_project_id())
                    if not kb_info:
                        update_log(log_container, "Knowledge Base not found", 'warning')
                        time.sleep(60)
                        continue
                    
                    # Process in smaller batches with error handling
                    batch_size = 5
                    for term_batch in chunks(get_search_terms(new_session), batch_size):
                        if not st.session_state.get('automation_status'):
                            break
                            
                        try:
                            results = manual_search(new_session, term_batch, 5)
                            if results and results.get('results'):
                                update_results_display(leads_container, results['results'])
                        except Exception as e:
                            logging.error(f"Batch processing error: {e}")
                            continue
                            
                    time.sleep(5)
                    
        except Exception as e:
            logging.error(f"Automation error: {e}")
            time.sleep(30)

def openai_chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    function_name: Optional[str] = None,
    lead_id: Optional[int] = None,
    email_campaign_id: Optional[int] = None
) -> Union[Dict, str, None]:
    try:
        with get_db_session() as session:
            settings = session.query(Settings).filter_by(
                setting_type='general'
            ).first()
            
            if not settings or 'openai_api_key' not in settings.value:
                raise ValueError("OpenAI API key not configured")
            
            client = openai.OpenAI(api_key=settings.value['openai_api_key'])
            model = settings.value.get('openai_model', 'gpt-4')
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            
            result = response.choices[0].message.content
            
            log_ai_request(
                session, function_name, messages,
                result, lead_id, email_campaign_id, model
            )
            
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
                
    except Exception as e:
        logger.exception("OpenAI API error")
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
    # Add error handling and better text cleaning
    try:
        for script in soup(["script", "style", "meta", "noscript"]):
            script.extract()
        text = ' '.join(soup.stripped_strings)
        return ' '.join(text.split())
    except Exception:
        return ""

def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
    session.add(SearchTermEffectiveness(term=term, total_results=total_results, valid_leads=valid_leads, irrelevant_leads=total_results - valid_leads, blogs_found=blogs_found, directories_found=directories_found))
    session.commit()

get_active_project_id = lambda: int(st.session_state.get('active_project_id', 1))
get_active_campaign_id = lambda: int(st.session_state.get('active_campaign_id', 1))
def set_active_project_id(project_id):
    with threading.Lock():
        st.session_state['active_project_id'] = int(project_id)
def set_active_campaign_id(campaign_id):
    with threading.Lock():
        st.session_state['active_campaign_id'] = int(campaign_id)

def add_or_get_search_term(session, term, campaign_id, created_at=None):
    try:
        with session.begin_nested():
            search_term = session.query(SearchTerm).filter_by(
                term=term, 
                campaign_id=campaign_id
            ).with_for_update().first()
            if not search_term:
                search_term = SearchTerm(
                    term=term,
                    campaign_id=campaign_id,
                    created_at=created_at or datetime.utcnow()
                )
                session.add(search_term)
                session.flush()
            return search_term.id
    except SQLAlchemyError:
        session.rollback()
        raise

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
def get_domain_from_url(url):
    return urlparse(url).netloc

def manual_search_page():
    # Only reload on form submission
    with st.form("manual_search_form"):
        search_terms = st_tags(label="Enter Search Terms", text="Press enter after each term")
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.number_input("Results per term", 1, 100, 10)
            optimize_english = st.checkbox("Optimize for English")
            optimize_spanish = st.checkbox("Optimize for Spanish")
        with col2:
            shuffle_keywords = st.checkbox("Shuffle Keywords")
            ignore_previous = st.checkbox("Ignore Previously Fetched", value=True)
        submit_button = st.form_submit_button("Search")
    
    # Move email settings outside form since they need immediate effect
    with safe_db_session() as session:
        email_settings = fetch_email_settings(session)
        from_email = st.selectbox("From Email", options=[f"{s['name']} ({s['email']})" for s in email_settings])
    
    # Results display (no reload needed)
    if 'search_results' in st.session_state:
        display_search_results(st.session_state.search_results)

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

def ai_automation_loop(session, log_container, leads_container):
    if not hasattr(st.session_state, 'automation_lock'):
        st.session_state.automation_lock = threading.Lock()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    
    while st.session_state.get('automation_status', False):
        try:
            with st.session_state.automation_lock:
                with safe_db_session() as new_session:
                    if not check_db_connection():
                        raise ConnectionError("Database connection lost")
                        
                    kb_info = get_knowledge_base_info(new_session, get_active_project_id())
                    if not kb_info:
                        update_log(log_container, "Knowledge Base not found", 'warning')
                        time.sleep(60)
                        continue
                    
                    # Process in smaller batches with error handling
                    batch_size = 5
                    for term_batch in chunks(get_search_terms(new_session), batch_size):
                        if not st.session_state.get('automation_status'):
                            break
                            
                        try:
                            results = manual_search(new_session, term_batch, 5)
                            if results and results.get('results'):
                                update_results_display(leads_container, results['results'])
                        except Exception as e:
                            logging.error(f"Batch processing error: {e}")
                            continue
                            
                    time.sleep(5)
                    
        except Exception as e:
            logging.error(f"Automation error: {e}")
            time.sleep(30)

# Make sure all database operations are performed within a session context
def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def get_page_description(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    return meta_desc['content'] if meta_desc else "No description found"

def is_valid_email(email):
    if not email or not isinstance(email, str):
        return False
    
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js|exe|zip|rar|pdf|doc|docx)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply|no-reply|donotreply|webmaster|postmaster)@.*",
        r"^(email|info|contact|support|hello|hola|hi|test|example|admin|administrator)@.*",
        r"^[0-9]+@.*",
        r".*@(example|test|localhost|invalid|temp|temporary)\.",
        r".*@.*\.(test|local|invalid|example|temp)$",
        r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL|php|html)$",
        r".*@(gmil|gmal|gmaill|gnail|gmale|gamil|gmai|gemail)\.com$",
        r".*@.*\.(local|test|example|invalid|temp|tmp)$"
    ]
    
    try:
        email_info = validate_email(email, check_deliverability=True)
        normalized_email = email_info.normalized
        return not any(re.match(pattern, normalized_email, re.IGNORECASE) for pattern in invalid_patterns)
    except EmailNotValidError:
        return False

def remove_invalid_leads(session):
    invalid_leads = session.query(Lead).filter(
        ~Lead.email.op('~')(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$') |
        Lead.email.op('~')(r'.*\.(png|jpg|jpeg|gif|css|js)$') |
        Lead.email.op('~')(r'^(nr|bootstrap|jquery|core|icon-|noreply)@.*') |
        Lead.email == 'email@email.com' |
        Lead.email.like('%@example.com') |
        Lead.email.op('~')(r'.*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$') |
        Lead.email.like('%@gmil.com') |
        Lead.email.like('%@gmal.com') |
        Lead.email.like('%@gmaill.com') |
        Lead.email.like('%@gnail.com')
    ).all()

    for lead in invalid_leads:
        session.query(LeadSource).filter(LeadSource.lead_id == lead.id).delete()
        session.delete(lead)

    session.commit()
    return len(invalid_leads)


def fetch_email_settings(session):
    """Fetch email settings from the database"""
    try:
        settings = session.query(EmailSettings).all()
        return [{"email": s.email, "name": s.name} for s in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {str(e)}")
        return []

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

def bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_bar=None, status_text=None, results=None, log_container=None):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        raise ValueError(f"Email template {template_id} not found")
    
    email_limiter = RateLimiter(max_calls=50, period=60)  # 50 emails per minute
    logs, sent_count = [], 0
    
    for index, lead in enumerate(leads):
        try:
            with email_limiter:
                response, tracking_id = send_email_ses(session, from_email, lead['Email'], 
                                                     template.subject, template.body_content, 
                                                     reply_to=reply_to)
                if response:
                    with safe_db_session() as new_session:  # Use new session for each save
                        save_email_campaign(new_session, lead['Email'], template_id, 'sent',
                                         datetime.utcnow(), template.subject, 
                                         response.get('MessageId'), template.body_content)
                    sent_count += 1
                
                if progress_bar:
                    progress_bar.progress((index + 1) / len(leads))
        except Exception as e:
            logging.error(f"Failed to send email to {lead['Email']}: {str(e)}")
            continue
            
    return logs, sent_count

def view_campaign_logs():
    st.header("Email Logs")
    with safe_db_session() as session:
        logs = fetch_all_email_logs(session)
        if logs.empty:
            st.info("No email logs found.")
        else:
            st.write(f"Total emails sent: {len(logs)}")
            st.write(f"Success rate: {(logs['Status'] == 'sent').mean():.2%}")

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=logs['Sent At'].min().date())
            with col2:
                end_date = st.date_input("End Date", value=logs['Sent At'].max().date())

            filtered_logs = logs[(logs['Sent At'].dt.date >= start_date) & (logs['Sent At'].dt.date <= end_date)]

            search_term = st.text_input("Search by email or subject")
            if search_term:
                filtered_logs = filtered_logs[filtered_logs['Email'].str.contains(search_term, case=False) | 
                                              filtered_logs['Subject'].str.contains(search_term, case=False)]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Emails Sent", len(filtered_logs))
            with col2:
                st.metric("Unique Recipients", filtered_logs['Email'].nunique())
            with col3:
                st.metric("Success Rate", f"{(filtered_logs['Status'] == 'sent').mean():.2%}")

            daily_counts = filtered_logs.resample('D', on='Sent At')['Email'].count()
            st.bar_chart(daily_counts)

            st.subheader("Detailed Email Logs")
            for _, log in filtered_logs.iterrows():
                with st.expander(f"{log['Sent At'].strftime('%Y-%m-%d %H:%M:%S')} - {log['Email']} - {log['Status']}"):
                    st.write(f"**Subject:** {log['Subject']}")
                    st.write(f"**Content Preview:** {log['Content'][:100]}...")
                    if st.button("View Full Email", key=f"view_email_{log['ID']}"):
                        st.components.v1.html(wrap_email_body(log['Content']), height=400, scrolling=True)
                    if log['Status'] != 'sent':
                        st.error(f"Status: {log['Status']}")

            logs_per_page = 20
            total_pages = (len(filtered_logs) - 1) // logs_per_page + 1
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            start_idx = (page - 1) * logs_per_page
            end_idx = start_idx + logs_per_page

            st.table(filtered_logs.iloc[start_idx:end_idx][['Sent At', 'Email', 'Subject', 'Status']])

            if st.button("Export Logs to CSV"):
                csv = filtered_logs.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="email_logs.csv",
                    mime="text/csv"
                )

def fetch_all_email_logs(session):
    try:
        email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({
            'ID': [ec.id for ec in email_campaigns],
            'Sent At': [ec.sent_at for ec in email_campaigns],
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
        logging.error(f"Database error in fetch_all_email_logs: {str(e)}")
        return pd.DataFrame()

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
    # Use session state for filters
    if 'lead_filters' not in st.session_state:
        st.session_state.lead_filters = {}
    
    # Only reload on filter apply
    with st.form("lead_filters_form"):
        st.session_state.lead_filters.update({
            'email': st.text_input("Filter by Email"),
            'company': st.text_input("Filter by Company"),
            'date_range': st.date_input("Date Range")
        })
        apply_filters = st.form_submit_button("Apply Filters")
    
    # Display leads (no reload needed)
    if 'filtered_leads' in st.session_state:
        display_leads(st.session_state.filtered_leads)

def fetch_leads_with_sources(session):
    try:
        query = session.query(Lead, func.string_agg(LeadSource.url, ', ').label('sources'), func.max(EmailCampaign.sent_at).label('last_contact'), func.string_agg(EmailCampaign.status, ', ').label('email_statuses')).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(Lead.id))
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

def get_active_campaign_id():
    return st.session_state.get('active_campaign_id', 1)

def search_terms_page():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Search Terms Dashboard</h1>", unsafe_allow_html=True)
    with safe_db_session() as session:
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
    try:
        current_term_ids = set(int(term.split(":")[0]) for term in updated_terms)
        existing_terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all()
        for term in existing_terms: term.group_id = None if term.id not in current_term_ids else group_id
        for term_str in updated_terms:
            term_id = int(term_str.split(":")[0])
            term = session.query(SearchTerm).get(term_id)
            if term: term.group_id = group_id
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_search_term_group: {str(e)}")

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        new_search_term = SearchTerm(term=new_term, campaign_id=campaign_id, created_at=datetime.utcnow())
        if group_for_new_term != "None": new_search_term.group_id = int(group_for_new_term.split(":")[0])
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
    existing_groups = session.query(SearchTermGroup).all()
    prompt = f"""Categorize these search terms into existing groups or suggest new ones:
    {', '.join([term.term for term in ungrouped_terms])}
    Existing groups: {', '.join([group.name for group in existing_groups])}
    Respond with a JSON object: {{group_name: [term1, term2, ...]}}"""
    messages = [
        {"role": "system", "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."},
        {"role": "user", "content": prompt}
    ]
    response = openai_chat_completion(messages, function_name="ai_group_search_terms")
    return response if isinstance(response, dict) else {}

def update_search_term_groups(session, grouped_terms):
    for group_name, terms in grouped_terms.items():
        group = session.query(SearchTermGroup).filter_by(name=group_name).first()
        if not group:
            group = SearchTermGroup(name=group_name)
            session.add(group)
            session.flush()
        for term in terms:
            search_term = session.query(SearchTerm).filter_by(term=term).first()
            if search_term: search_term.group_id = group.id
    session.commit()

def create_search_term_group(session, group_name):
    try:
        new_group = SearchTermGroup(name=group_name)
        session.add(new_group)
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
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting search term group: {str(e)}")

def email_templates_page():
    # Combine template creation into one form
    with st.form("template_form"):
        template_name = st.text_input("Template Name")
        subject = st.text_input("Subject")
        body = st.text_area("Body")
        is_ai_customizable = st.checkbox("Enable AI Customization")
        submit_button = st.form_submit_button("Save Template")
    
    # Preview updates (no reload needed)
    if 'template_preview' in st.session_state:
        st.markdown(st.session_state.template_preview, unsafe_allow_html=True)

def validate_email_settings(settings):
    if settings['provider'] == 'ses': required_fields = ['aws_access_key_id', 'aws_secret_access_key', 'aws_region']
    else: required_fields = ['smtp_server', 'smtp_port', 'smtp_username', 'smtp_password']
    missing_fields = [field for field in required_fields if not settings.get(field)]
    if missing_fields: raise ValueError(f"Missing required fields for {settings['provider']}: {', '.join(missing_fields)}")
    return True

def is_valid_email_info(email):
    if email is None: return False, ["Email is empty"]
    warnings = []
    patterns = {
        r".*\.(png|jpg|jpeg|gif|css|js)$": "Email ends with file extension",
        r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*": "Generic/System email prefix",
        r"^(email|info|contact|support|hello|hola|hi|salutations|greetings|inquiries|questions)@.*": "Generic business email",
        r"^email@email\.com$": "Placeholder email",
        r".*@example\.com$": "Example domain email"
    }
    for pattern, warning in patterns.items():
        if re.match(pattern, email, re.IGNORECASE): warnings.append(warning)
    try:
        validate_email(email)
        return True, warnings
    except EmailNotValidError as e:
        warnings.append(str(e))
        return False, warnings

def add_tracking_to_email(body: str, tracking_id: str) -> str:
    tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
    soup = BeautifulSoup(body, 'html.parser')
    tracking_pixel = f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/>'
    for link in soup.find_all('a', href=True):
        original_url = link['href']
        tracked_url = (f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'click', 'url': original_url})}")
        link['href'] = tracked_url
    modified_body = str(soup)
    return modified_body.replace('</body>', f'{tracking_pixel}</body>') if '</body>' in modified_body else modified_body + tracking_pixel

def test_email_connection(settings):
    try:
        if settings['provider'] == 'smtp':
            import smtplib
            if settings['smtp_port'] == 465: server = smtplib.SMTP_SSL(settings['smtp_server'], settings['smtp_port'], timeout=5)
            else:
                server = smtplib.SMTP(settings['smtp_server'], settings['smtp_port'], timeout=5)
                if settings['smtp_port'] == 587: server.starttls()
            server.ehlo()
            server.login(settings['smtp_username'], settings['smtp_password'])
            server.quit()
            return True, None
        elif settings['provider'] == 'ses':
            import boto3
            ses_client = boto3.client('ses', aws_access_key_id=settings['aws_access_key_id'], aws_secret_access_key=settings['aws_secret_access_key'], region_name=settings['aws_region'])
            ses_client.get_send_quota()
            return True, None
        else: return False, "Unknown provider"
    except Exception as e: return False, str(e)

def get_email_preview(session, template_id, from_email, reply_to):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if template: return template.body_content
    return "<p>Template not found</p>"

def fetch_all_search_terms(session): return session.query(SearchTerm).all()

def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

def get_email_template_by_name(session, template_name): return session.query(EmailTemplate).filter_by(template_name=template_name).first()

def bulk_send_page():
    # Combine all settings into one form
    with st.form("bulk_send_form"):
        template_option = st.selectbox("Email Template", options=templates)
        email_setting = st.selectbox("From Email", options=email_settings)
        send_option = st.radio("Send to:", ["All Leads", "Specific Email", "Leads from Search Terms"])
        
        if send_option == "Specific Email":
            specific_email = st.text_input("Enter email")
        elif send_option == "Leads from Search Terms":
            selected_terms = st.multiselect("Select Search Terms", options=search_terms)
            
        exclude_previous = st.checkbox("Exclude Previously Contacted")
        submit_button = st.form_submit_button("Send Emails")

def display_search_results(results, key_suffix):
    if not results: return st.warning("No results to display.")
    with st.expander("Search Results", expanded=True):
        st.markdown(f"### Total Leads Found: **{len(results)}**")
        for i, res in enumerate(results):
            with st.expander(f"Lead: {res['Email']}", key=f"lead_expander_{key_suffix}_{i}"):
                st.markdown(f"**URL:** [{res['URL']}]({res['URL']})\n**Title:** {res['Title']}\n**Description:** {res['Description']}\n**Tags:** {', '.join(res['Tags'])}\n**Lead Source:** {res['Lead Source']}\n**Lead Email:** {res['Email']}")

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
    with safe_db_session() as session:
        st.header("Projects and Campaigns")
        st.subheader("Add New Project")
        with st.form("add_project_form"):
            project_name = st.text_input("Project Name")
            if st.form_submit_button("Add Project"):
                if project_name.strip():
                    try:
                        session.add(Project(project_name=project_name, created_at=datetime.utcnow()))
                        session.commit()
                        st.success(f"Project '{project_name}' added successfully.")
                    except SQLAlchemyError as e:
                        st.error(f"Error adding project: {str(e)}")
                else:
                    st.warning("Please enter a project name.")
        st.subheader("Existing Projects and Campaigns")
        projects = session.query(Project).all()
        for project in projects:
            with st.expander(f"Project: {project.project_name}"):
                st.info("Campaigns share resources and settings within a project.")
                with st.form(f"add_campaign_form_{project.id}"):
                    campaign_name = st.text_input("Campaign Name", key=f"campaign_name_{project.id}")
                    if st.form_submit_button("Add Campaign"):
                        if campaign_name.strip():
                            try:
                                session.add(Campaign(campaign_name=campaign_name, project_id=project.id, created_at=datetime.utcnow()))
                                session.commit()
                                st.success(f"Campaign '{campaign_name}' added to '{project.project_name}'.")
                            except SQLAlchemyError as e:
                                st.error(f"Error adding campaign: {str(e)}")
                        else:
                            st.warning("Please enter a campaign name.")
                campaigns = session.query(Campaign).filter_by(project_id=project.id).all()
                st.write("Campaigns:" if campaigns else f"No campaigns for {project.project_name} yet.")
                for campaign in campaigns:
                    st.write(f"- {campaign.campaign_name}")
        st.subheader("Set Active Project and Campaign")
        project_options = [p.project_name for p in projects]
        if project_options:
            active_project = st.selectbox("Select Active Project", options=project_options, index=0)
            active_project_id = session.query(Project.id).filter_by(project_name=active_project).scalar()
            set_active_project_id(active_project_id)
            active_project_campaigns = session.query(Campaign).filter_by(project_id=active_project_id).all()
            if active_project_campaigns:
                campaign_options = [c.campaign_name for c in active_project_campaigns]
                active_campaign = st.selectbox("Select Active Campaign", options=campaign_options, index=0)
                active_campaign_id = session.query(Campaign.id).filter_by(campaign_name=active_campaign, project_id=active_project_id).scalar()
                set_active_campaign_id(active_campaign_id)
                st.success(f"Active Project: {active_project}, Active Campaign: {active_campaign}")
            else:
                st.warning(f"No campaigns available for {active_project}. Please add a campaign.")
        else:
            st.warning("No projects found. Please add a project first.")


def fetch_projects(session):
    """Fetch all projects and return them formatted as 'id: name' options"""
    projects = session.query(Project).all()
    return [f"{p.id}: {p.project_name}" for p in projects]

@contextmanager
def get_db_session():
    session = None
    try:
        session = SessionLocal()
        yield session
        session.commit()
    except SQLAlchemyError as e:
        if session:
            session.rollback()
        logging.error(f"Database error: {str(e)}")
        raise
    finally:
        if session:
            session.close()


def knowledge_base_page():
    st.title("Knowledge Base")
    
    with get_db_session() as session:
        project_options = fetch_projects(session)
        if not project_options:
            st.warning("No projects found. Please create a project first.")
            return
            
        selected_project = st.selectbox("Select Project", options=project_options)
        try:
            project_id = int(selected_project.split(":")[0])
        except (ValueError, AttributeError):
            st.error("Invalid project selection")
            return
        
        kb_entry = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        
        with st.form("knowledge_base_form"):
            text_fields = ['kb_name', 'contact_name', 'contact_role', 'contact_email', 'product_name']
            form_data = {}
            
            fields = [
                'kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role',
                'contact_email', 'company_description', 'company_mission',
                'company_target_market', 'company_other', 'product_name',
                'product_description', 'product_target_customer', 'product_other',
                'other_context', 'example_email'
            ]
            
            for field in fields:
                current_value = getattr(kb_entry, field, '') if kb_entry else ''
                field_label = field.replace('_', ' ').title()
                form_data[field] = (
                    st.text_input(field_label, value=current_value)
                    if field in text_fields
                    else st.text_area(field_label, value=current_value)
                )
            
            if st.form_submit_button("Save Knowledge Base"):
                try:
                    form_data.update({
                        'project_id': project_id,
                        'created_at': datetime.utcnow()
                    })
                    
                    if kb_entry:
                        for key, value in form_data.items():
                            setattr(kb_entry, key, value)
                    else:
                        session.add(KnowledgeBase(**form_data))
                    
                    session.commit()
                    st.success("Knowledge Base saved successfully!", icon="✅")
                except Exception as e:
                    session.rollback()
                    st.error(f"Error saving Knowledge Base: {str(e)}")

def autoclient_ai_page():
    st.header("AutoclientAI - Automated Lead Generation")
    with st.expander("Knowledge Base Information", expanded=False):
        with get_db_session() as session:
            kb_info = get_knowledge_base_info(session, get_active_project_id())
        if not kb_info:
            return st.error("Knowledge Base not found for the active project. Please set it up first.")
        st.json(kb_info)
    user_input = st.text_area("Enter additional context or specific goals for lead generation:", help="This information will be used to generate more targeted search terms.")
    if st.button("Generate Optimized Search Terms", key="generate_optimized_terms"):
        with st.spinner("Generating optimized search terms..."):
            with get_db_session() as session:
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
        with st.form("automation_confirm"):
            confirm = st.form_submit_button("Confirm Start")
            if confirm:
                st.session_state.update({
                    "automation_status": True,
                    "automation_logs": [],
                    "total_leads_found": 0,
                    "total_emails_sent": 0
                })
    if st.session_state.get('automation_status', False):
        st.subheader("Automation in Progress")
        progress_bar, log_container, leads_container, analytics_container = st.progress(0), st.empty(), st.empty(), st.empty()
        try:
            with get_db_session() as session:
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
    metric_cards([
        {"title": "Total Leads", "value": len(results)},
        {"title": "New Leads", "value": len([r for r in results if r.get("is_new")])},
    ])

    results_html = f"""
    <div class="results-container">
        {''.join(
            f'<div class="result-entry">'
            f'<strong>{res["Email"]}</strong><br>'
            f'<a href="{res["URL"]}" target="_blank">{res["URL"]}</a>'
            f'</div>'
            for res in results[-10:]
        )}
    </div>
    """
    results_container.markdown(results_html, unsafe_allow_html=True)

import streamlit_extras
#We add all misisng functions
def update_automation_status(status_container, log_container):
    from streamlit_extras.stoggle import stoggle
    
    status = "Running" if st.session_state.automation_status else "Stopped"
    status_container.metric("Automation Status", status)
    
    stoggle("View Logs", "\n".join(st.session_state.automation_logs))

def automation_control_panel_page():
    from streamlit_extras.switch_page_button import switch_page
    
    current_status = st.session_state.get('automation_status', False)
    if st.button("Toggle Automation"): 
        st.session_state.automation_status = not current_status
        st.rerun()
        
    if st.session_state.get('automation_status'):
        status_container = st.empty()
        log_container = st.empty()
        while st.session_state.automation_status:
            update_automation_status(status_container, log_container)
            time.sleep(1)
            
    if st.button("Back to Main"):
        switch_page("Main")

def get_knowledge_base_info(session, project_id):
    kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb.to_dict() if kb else None

def generate_optimized_search_terms(session, base_terms, kb_info):
    ai_prompt = f"Generate 5 optimized search terms based on: {', '.join(base_terms)}. Context: {kb_info}"
    return get_ai_response(ai_prompt).split('\n')

def update_display(container, items, title, item_type):
    from streamlit_extras.colored_header import colored_header
    colored_header(title, description=f"Latest {len(items)} {item_type}")
    
    for item in items[-10:]:
        container.text(item)

def get_search_terms(session):
    return [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]

def get_ai_response(prompt):
    return openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100).choices[0].text.strip()

def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{
            "id": setting.id,
            "name": setting.name,
            "email": setting.email,
            "provider": setting.provider,
            "smtp_server": setting.smtp_server,
            "smtp_port": setting.smtp_port,
            "smtp_username": setting.smtp_username,
            "smtp_password": setting.smtp_password,
            "aws_access_key_id": setting.aws_access_key_id,
            "aws_secret_access_key": setting.aws_secret_access_key,
            "aws_region": setting.aws_region
        } for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        return []

def bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_bar=None, status_text=None, results=None, log_container=None):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        raise ValueError(f"Email template {template_id} not found")
    
    email_limiter = RateLimiter(max_calls=50, period=60)
    logs, sent_count = [], 0
    
    for index, lead in enumerate(leads):
        try:
            with email_limiter:
                response, tracking_id = send_email_ses(session, from_email, lead['Email'], 
                                                     template.subject, template.body_content, 
                                                     reply_to=reply_to)
                if response:
                    with safe_db_session() as new_session:
                        save_email_campaign(new_session, lead['Email'], template_id, 'sent',
                                         datetime.utcnow(), template.subject, 
                                         response.get('MessageId'), template.body_content)
                    sent_count += 1
                
                if progress_bar:
                    progress_bar.progress((index + 1) / len(leads))
        except Exception as e:
            logging.error(f"Failed to send email to {lead['Email']}: {str(e)}")
            continue
            
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

def display_logs(log_container, logs):
    if not logs:
        log_container.info("No logs to display yet.")
        return

    from streamlit_extras.let_it_rain import rain
    rain(emoji="📝", font_size=20, falling_speed=5, animation_length="infinite")

    log_container.markdown(
        """
        <style>
        .log-container {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
        }
        .log-entry {
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            border-radius: 0.25rem;
            background-color: rgba(28, 131, 225, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    log_entries = "".join(f'<div class="log-entry">{log}</div>' for log in logs[-20:])
    log_container.markdown(f'<div class="log-container">{log_entries}</div>', unsafe_allow_html=True)

def view_sent_email_campaigns():
    st.header("Sent Email Campaigns")
    try:
        with safe_db_session() as session:
            email_campaigns = fetch_sent_email_campaigns(session)
        if not email_campaigns.empty:
            from streamlit_extras.dataframe_explorer import dataframe_explorer
            filtered_df = dataframe_explorer(email_campaigns)
            st.dataframe(filtered_df)
            
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

def create_custom_menu():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Manual Search"
    
    menu_items = {
        "Manual Search": manual_search_page,
        "Bulk Send": bulk_send_page,
        "View Leads": view_leads_page,
        "Search Terms": search_terms_page,
        "Email Templates": email_templates_page,
        "Projects & Campaigns": projects_campaigns_page,
        "Knowledge Base": knowledge_base_page,
        "AutoclientAI": autoclient_ai_page,
        "Automation Control": automation_control_panel_page,
        "Email Logs": view_campaign_logs,
        "Settings": settings_page,
        "Sent Campaigns": view_sent_email_campaigns
    }
    
    from streamlit_option_menu import option_menu
    selected_page = option_menu(
        "Navigation",
        options=list(menu_items.keys()),
        icons=['search', 'send', 'people', 'tags', 'envelope', 'folder', 'book', 'robot', 'toggles', 'journal', 'gear', 'envelope-check'],
        menu_icon="cast",
        default_index=0,
    )
    
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()
    
    return menu_items[selected_page]

def main():
    try:
        init_session_state()
        st.set_page_config(
            page_title="Autoclient.ai",
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🤖"
        )
        current_page_func = create_custom_menu()
        current_page_func()
    except Exception as e:
        logger.exception("Error in main function")
        st.error(f"An error occurred: {str(e)}")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Add this right before the ai_automation_loop function

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'automation_status': False,
        'automation_logs': [],
        'total_leads_found': 0,
        'total_emails_sent': 0,
        'current_page': "Manual Search",
        'search_results': [],
        'log_entries': [],
        'optimized_terms': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Add this near the top of the file, before the main function

def google_search(query, num_results=10, lang='ES'):
    """Perform a Google search and return URLs"""
    try:
        headers = {'User-Agent': UserAgent().random}
        search_url = f"https://www.google.com/search?q={query}&num={num_results}&hl={lang}"
        response = requests.get(search_url, headers=headers, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        search_results = []
        for result in soup.find_all('a'):
            href = result.get('href', '')
            if href.startswith('/url?q='):
                url = href.split('/url?q=')[1].split('&')[0]
                if not any(x in url for x in ['google.', 'youtube.', 'facebook.', 'linkedin.']):
                    search_results.append(url)
                    
        return search_results[:num_results]
    except Exception as e:
        logging.error(f"Error in google_search: {str(e)}")
        return []

# Add this before the manual_search function

if __name__ == "__main__":
    main()
