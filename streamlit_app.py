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
    try:
        if not hasattr(thread_local, "session"):
            engine = get_db_connection()
            Session = sessionmaker(bind=engine)
            thread_local.session = Session()
        return thread_local.session
    finally:
        if hasattr(thread_local, "session"):
            thread_local.session.close()

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
    knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False, cascade="all, delete-orphan")

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
    email_campaigns = relationship("EmailCampaign", back_populates="campaign", cascade="all, delete-orphan")
    search_terms = relationship("SearchTerm", back_populates="campaign", cascade="all, delete-orphan")
    campaign_leads = relationship("CampaignLead", back_populates="campaign", cascade="all, delete-orphan")

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
        return {
            'kb_name': self.kb_name,
            'kb_bio': self.kb_bio,
            'kb_values': self.kb_values,
            'contact_name': self.contact_name,
            'contact_role': self.contact_role,
            'contact_email': self.contact_email,
            'company_description': self.company_description,
            'company_mission': self.company_mission,
            'company_target_market': self.company_target_market,
            'company_other': self.company_other,
            'product_name': self.product_name,
            'product_description': self.product_description,
            'product_target_customer': self.product_target_customer,
            'product_other': self.product_other,
            'other_context': self.other_context,
            'example_email': self.example_email,
            'domain': self.domain
        }

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
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'domain': self.domain
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
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'), nullable=True)
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
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'), nullable=True)
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
    lead_id = Column(BigInteger, ForeignKey('leads.id'), nullable=True)
    email_campaign_id = Column(BigInteger, ForeignKey('email_campaigns.id'), nullable=True)
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
    name = Column(Text, nullable=False, unique=True)
    setting_type = Column(Text, nullable=False)  # 'general', 'email', etc.
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False, unique=True)
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

    def fetch_email_settings(session):
        return session.query(EmailSettings).all()

    with db_session() as session:
        # Email Settings
        st.subheader("Email Settings")
        email_settings = fetch_email_settings(session)

        if not email_settings:
            st.warning("No email settings found. Please add email settings.")
        else:
            selected_email_setting_id = st.selectbox("Select Email Setting", options=[setting.id for setting in email_settings], format_func=lambda x: next((setting.name for setting in email_settings if setting.id == x), ''))

            selected_email_setting = next((setting for setting in email_settings if setting.id == selected_email_setting_id), None)

            if selected_email_setting:
                if st.button("Edit"):
                    st.session_state.edit_email_setting = selected_email_setting

                if st.button("Delete"):
                    session.delete(selected_email_setting)
                    session.commit()
                    st.success("Email setting deleted successfully.")
                    st.experimental_rerun()

                if st.button("Set as Active"):
                    for setting in email_settings:
                        setting.is_active = False
                    selected_email_setting.is_active = True
                    session.commit()
                    st.success(f"Email setting '{selected_email_setting.name}' set as active.")
                    st.experimental_rerun()

                st.write(f"**Name:** {selected_email_setting.name}")
                st.write(f"**Email:** {selected_email_setting.email}")
                st.write(f"**Provider:** {selected_email_setting.provider}")
                if selected_email_setting.provider.lower() == 'smtp':
                    st.write(f"**SMTP Server:** {selected_email_setting.smtp_server}")
                    st.write(f"**SMTP Port:** {selected_email_setting.smtp_port}")
                st.write(f"**Active:** {selected_email_setting.is_active}")

        if 'edit_email_setting' in st.session_state:
            with st.form(key='edit_email_settings_form'):
                st.subheader("Edit Email Settings")
                setting = st.session_state.edit_email_setting
                setting.name = st.text_input("Name", value=setting.name)
                setting.email = st.text_input("Email", value=setting.email)
                setting.provider = st.selectbox("Provider", options=['SES', 'SMTP'], index=0 if setting.provider.lower() == 'ses' else 1)
                if setting.provider.lower() == 'smtp':
                    setting.smtp_server = st.text_input("SMTP Server", value=setting.smtp_server)
                    setting.smtp_port = st.number_input("SMTP Port", value=setting.smtp_port, min_value=0, max_value=65535)
                    setting.smtp_username = st.text_input("SMTP Username", value=setting.smtp_username)
                    setting.smtp_password = st.text_input("SMTP Password", value=setting.smtp_password, type="password")
                else:
                    setting.aws_access_key_id = st.text_input("AWS Access Key ID", value=setting.aws_access_key_id)
                    setting.aws_secret_access_key = st.text_input("AWS Secret Access Key", value=setting.aws_secret_access_key, type="password")
                    setting.aws_region = st.text_input("AWS Region", value=setting.aws_region)
                setting.daily_limit = st.number_input("Daily Limit", value=setting.daily_limit, min_value=1)
                setting.hourly_limit = st.number_input("Hourly Limit", value=setting.hourly_limit, min_value=1)

                if st.form_submit_button("Save Changes"):
                    session.commit()
                    del st.session_state.edit_email_setting
                    st.experimental_rerun()

        if st.button("Add New Email Setting"):
            st.session_state.show_add_email_form = True

        if st.session_state.get('show_add_email_form', False):
            with st.form(key='add_email_settings_form'):
                st.subheader("Add New Email Settings")
                new_name = st.text_input("Name")
                new_email = st.text_input("Email")
                new_provider = st.selectbox("Provider", options=['SES', 'SMTP'])
                if new_provider.lower() == 'smtp':
                    new_smtp_server = st.text_input("SMTP Server")
                    new_smtp_port = st.number_input("SMTP Port", min_value=0, max_value=65535)
                    new_smtp_username = st.text_input("SMTP Username")
                    new_smtp_password = st.text_input("SMTP Password", type="password")
                else:
                    new_aws_access_key_id = st.text_input("AWS Access Key ID")
                    new_aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password")
                    new_aws_region = st.text_input("AWS Region")
                new_daily_limit = st.number_input("Daily Limit", min_value=1, value=50000)
                new_hourly_limit = st.number_input("Hourly Limit", min_value=1, value=5)

                if st.form_submit_button("Add"):
                    new_email_setting = EmailSettings(
                        name=new_name,
                        email=new_email,
                        provider=new_provider,
                        smtp_server=new_smtp_server if new_provider.lower() == 'smtp' else None,
                        smtp_port=new_smtp_port if new_provider.lower() == 'smtp' else None,
                        smtp_username=new_smtp_username if new_provider.lower() == 'smtp' else None,
                        smtp_password=new_smtp_password if new_provider.lower() == 'smtp' else None,
                        aws_access_key_id=new_aws_access_key_id if new_provider.lower() == 'ses' else None,
                        aws_secret_access_key=new_aws_secret_access_key if new_provider.lower() == 'ses' else None,
                        aws_region=new_aws_region if new_provider.lower() == 'ses' else None,
                        daily_limit=new_daily_limit,
                        hourly_limit=new_hourly_limit,
                        is_active=False
                    )
                    session.add(new_email_setting)
                    session.commit()
                    st.success("New email setting added successfully.")
                    st.session_state.show_add_email_form = False
                    st.experimental_rerun()

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

def update_log(log_container, message, level='info'):
    icon = {'info': '🔵', 'success': '🟢', 'warning': '🟠', 'error': '🔴', 'email_sent': '🟣'}.get(level, '⚪')
    log_entry = f"{icon} {message}"
    
    # Initialize log entries in session state if not present
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    
    # HTML-formatted log entry for Streamlit display
    html_log_entry = f"{icon} {message}"
    st.session_state.log_entries.append(html_log_entry)
    
    # Update the Streamlit display with all logs
    log_html = f"<div style='height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.8em; line-height: 1.2;'>{'<br>'.join(st.session_state.log_entries)}</div>"
    log_container.markdown(log_html, unsafe_allow_html=True)

def optimize_search_term(search_term, language):
    """Optimizes search term based on language."""
    if language == 'english':
        # Remove site restriction and add more contact-related terms
        return f'"{search_term}" (email OR contact OR "get in touch" OR "reach out" OR "contact us" OR "contact form" OR "send message" OR "send us a message" OR "write to us" OR "get in contact")'
    elif language == 'spanish':
        # Remove site restriction and add more contact-related terms
        return f'"{search_term}" (correo OR contacto OR email OR "ponte en contacto" OR "formulario de contacto" OR "enviar mensaje" OR "envíanos un mensaje" OR "escríbenos" OR "contacta con nosotros")'
    elif language == 'french':
        return f'"{search_term}" (email OR contact OR "entrer en contact" OR "contactez-nous" OR "formulaire de contact" OR "envoyer un message" OR "envoyez-nous un message" OR "écrivez-nous" OR "entrer en contact")'
    elif language == 'german':
        return f'"{search_term}" (E-Mail OR Kontakt OR "in Kontakt treten" OR "kontaktieren Sie uns" OR "Kontaktformular" OR "Nachricht senden" OR "senden Sie uns eine Nachricht" OR "schreiben Sie uns" OR "in Kontakt treten")'
    elif language == 'italian':
        return f'"{search_term}" (email OR contatto OR "entrare in contatto" OR "contattaci" OR "modulo di contatto" OR "invia messaggio" OR "inviaci un messaggio" OR "scrivici" OR "entrare in contatto")'
    elif language == 'portuguese':
        return f'"{search_term}" (email OR contato OR "entrar em contato" OR "entre em contato" OR "formulário de contato" OR "enviar mensagem" OR "envie-nos uma mensagem" OR "escreva para nós" OR "entrar em contato")'
    elif language == 'russian':
        return f'"{search_term}" (электронная почта OR контакт OR "связаться" OR "свяжитесь с нами" OR "контактная форма" OR "отправить сообщение" OR "отправьте нам сообщение" OR "напишите нам" OR "связаться")'
    elif language == 'chinese':
        return f'"{search_term}" (电子邮件 OR 联系 OR "取得联系" OR "联系我们" OR "联系表格" OR "发送消息" OR "给我们发送消息" OR "写信给我们" OR "取得联系")'
    elif language == 'japanese':
        return f'"{search_term}" (Eメール OR 連絡先 OR "連絡する" OR "お問い合わせ" OR "お問い合わせフォーム" OR "メッセージを送信" OR "メッセージを送信してください" OR "お問い合わせ" OR "連絡する")'
    elif language == 'korean':
        return f'"{search_term}" (이메일 OR 연락처 OR "연락하기" OR "문의하기" OR "문의 양식" OR "메시지 보내기" OR "메시지 보내기" OR "문의하기" OR "연락하기")'
    elif language == 'arabic':
        return f'"{search_term}" (البريد الإلكتروني OR الاتصال OR "الاتصال" OR "اتصل بنا" OR "نموذج الاتصال" OR "إرسال رسالة" OR "أرسل لنا رسالة" OR "اكتب إلينا" OR "الاتصال")'
    elif language == 'hindi':
        return f'"{search_term}" (ईमेल OR संपर्क OR "संपर्क करें" OR "हमसे संपर्क करें" OR "संपर्क फ़ॉर्म" OR "संदेश भेजें" OR "हमें संदेश भेजें" OR "हमें लिखें" OR "संपर्क करें")'
    elif language == 'bengali':
        return f'"{search_term}" (ইমেল OR যোগাযোগ OR "যোগাযোগ করুন" OR "আমাদের সাথে যোগাযোগ করুন" OR "যোগাযোগ ফর্ম" OR "বার্তা পাঠান" OR "আমাদের বার্তা পাঠান" OR "আমাদের লিখুন" OR "যোগাযোগ করুন")'
    elif language == 'punjabi':
        return f'"{search_term}" (ਈਮੇਲ OR ਸੰਪਰਕ OR "ਸੰਪਰਕ ਕਰੋ" OR "ਸਾਡੇ ਨਾਲ ਸੰਪਰਕ ਕਰੋ" OR "ਸੰਪਰਕ ਫਾਰਮ" OR "ਸੁਨੇਹਾ ਭੇਜੋ" OR "ਸਾਨੂੰ ਸੁਨੇਹਾ ਭੇਜੋ" OR "ਸਾਨੂੰ ਲਿਖੋ" OR "ਸੰਪਰਕ ਕਰੋ")'
    elif language == 'telugu':
        return f'"{search_term}" (ఇమెయిల్ OR సంప్రదించండి OR "సంప్రదించండి" OR "మమ్మల్ని సంప్రదించండి" OR "సంప్రదింపు ఫారం" OR "సందేశం పంపండి" OR "మాకు సందేశం పంపండి" OR "మాకు వ్రాయండి" OR "సంప్రదించండి")'
    elif language == 'marathi':
        return f'"{search_term}" (ईमेल OR संपर्क OR "संपर्क करा" OR "आमच्याशी संपर्क साधा" OR "संपर्क फॉर्म" OR "संदेश पाठवा" OR "आम्हाला संदेश पाठवा" OR "आम्हाला लिहा" OR "संपर्क करा")'
    elif language == 'tamil':
        return f'"{search_term}" (மின்னஞ்சல் OR தொடர்பு OR "தொடர்பு கொள்ளவும்" OR "எங்களைத் தொடர்பு கொள்ளவும்" OR "தொடர்பு படிவம்" OR "செய்தி அனுப்பவும்" OR "எங்களுக்கு செய்தி அனுப்பவும்" OR "எங்களுக்கு எழுதவும்" OR "தொடர்பு கொள்ளவும்")'
    elif language == 'turkish':
        return f'"{search_term}" (e-posta OR iletişim OR "iletişime geçin" OR "bize ulaşın" OR "iletişim formu" OR "mesaj gönder" OR "bize mesaj gönder" OR "bize yazın" OR "iletişime geçin")'
    elif language == 'urdu':
        return f'"{search_term}" (ای میل OR رابطہ OR "رابطہ کریں" OR "ہم سے رابطہ کریں" OR "رابطہ فارم" OR "پیغام بھیجیں" OR "ہمیں پیغام بھیجیں" OR "ہمیں لکھیں" OR "رابطہ کریں")'
    elif language == 'gujarati':
        return f'"{search_term}" (ઈમેલ OR સંપર્ક OR "સંપર્ક કરો" OR "અમારો સંપર્ક કરો" OR "સંપર્ક ફોર્મ" OR "સંદેશ મોકલો" OR "અમને સંદેશ મોકલો" OR "અમને લખો" OR "સંપર્ક કરો")'
    elif language == 'malayalam':
        return f'"{search_term}" (ഇമെയിൽ OR ബന്ധപ്പെടുക OR "ബന്ധപ്പെടുക" OR "ഞങ്ങളെ ബന്ധപ്പെടുക" OR "ബന്ധപ്പെടാനുള്ള ഫോം" OR "സന്ദേശം അയയ്ക്കുക" OR "ഞങ്ങൾക്ക് സന്ദേശം അയയ്ക്കുക" OR "ഞങ്ങൾക്ക് എഴുതുക" OR "ബന്ധപ്പെടുക")'
    elif language == 'kannada':
        return f'"{search_term}" (ಇಮೇಲ್ OR ಸಂಪರ್ಕಿಸಿ OR "ಸಂಪರ್ಕಿಸಿ" OR "ನಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸಿ" OR "ಸಂಪರ್ಕ ಫಾರ್ಮ್" OR "ಸಂದೇಶ ಕಳುಹಿಸಿ" OR "ನಮಗೆ ಸಂದೇಶ ಕಳುಹಿಸಿ" OR "ನಮಗೆ ಬರೆಯಿರಿ" OR "ಸಂಪರ್ಕಿಸಿ")'
    elif language == 'odia':
        return f'"{search_term}" (ଇମେଲ୍ OR ଯୋଗାଯୋଗ OR "ଯୋଗାଯୋଗ କରନ୍ତୁ" OR "ଆମ ସହିତ ଯୋଗାଯୋଗ କରନ୍ତୁ" OR "ଯୋଗାଯୋଗ ଫର୍ମ" OR "ବାର୍ତ୍ତା ପଠାନ୍ତୁ" OR "ଆମକୁ ବାର୍ତ୍ତା ପଠାନ୍ତୁ" OR "ଆମକୁ ଲେଖନ୍ତୁ" OR "ଯୋଗାଯୋଗ କରନ୍ତୁ")'
    elif language == 'thai':
        return f'"{search_term}" (อีเมล OR ติดต่อ OR "ติดต่อ" OR "ติดต่อเรา" OR "แบบฟอร์มติดต่อ" OR "ส่งข้อความ" OR "ส่งข้อความถึงเรา" OR "เขียนถึงเรา" OR "ติดต่อ")'
    elif language == 'vietnamese':
        return f'"{search_term}" (email OR liên hệ OR "liên hệ" OR "liên hệ với chúng tôi" OR "biểu mẫu liên hệ" OR "gửi tin nhắn" OR "gửi tin nhắn cho chúng tôi" OR "viết thư cho chúng tôi" OR "liên hệ")'
    elif language == 'indonesian':
        return f'"{search_term}" (email OR kontak OR "hubungi" OR "hubungi kami" OR "formulir kontak" OR "kirim pesan" OR "kirim pesan kepada kami" OR "tulis kepada kami" OR "hubungi")'
    elif language == 'filipino':
        return f'"{search_term}" (email OR contact OR "makipag-ugnayan" OR "makipag-ugnayan sa amin" OR "form ng contact" OR "magpadala ng mensahe" OR "magpadala sa amin ng mensahe" OR "sumulat sa amin" OR "makipag-ugnayan")'
    elif language == 'dutch':
        return f'"{search_term}" (e-mail OR contact OR "neem contact op" OR "neem contact met ons op" OR "contactformulier" OR "bericht sturen" OR "stuur ons een bericht" OR "schrijf ons" OR "neem contact op")'
    elif language == 'greek':
        return f'"{search_term}" (email OR επικοινωνία OR "επικοινωνήστε" OR "επικοινωνήστε μαζί μας" OR "φόρμα επικοινωνίας" OR "στείλτε μήνυμα" OR "στείλτε μας ένα μήνυμα" OR "γράψτε μας" OR "επικοινωνήστε")'
    elif language == 'polish':
        return f'"{search_term}" (email OR kontakt OR "skontaktuj się" OR "skontaktuj się z nami" OR "formularz kontaktowy" OR "wyślij wiadomość" OR "wyślij do nas wiadomość" OR "napisz do nas" OR "skontaktuj się")'
    elif language == 'swedish':
        return f'"{search_term}" (e-post OR kontakt OR "kontakta" OR "kontakta oss" OR "kontaktformulär" OR "skicka meddelande" OR "skicka ett meddelande till oss" OR "skriv till oss" OR "kontakta")'
    elif language == 'danish':
        return f'"{search_term}" (e-mail OR kontakt OR "kontakt" OR "kontakt os" OR "kontaktformular" OR "send besked" OR "send os en besked" OR "skriv til os" OR "kontakt")'
    elif language == 'norwegian':
        return f'"{search_term}" (e-post OR kontakt OR "kontakt" OR "kontakt oss" OR "kontaktskjema" OR "send melding" OR "send oss en melding" OR "skriv til oss" OR "kontakt")'
    elif language == 'finnish':
        return f'"{search_term}" (sähköposti OR yhteystiedot OR "ota yhteyttä" OR "ota meihin yhteyttä" OR "yhteydenottolomake" OR "lähetä viesti" OR "lähetä meille viesti" OR "kirjoita meille" OR "ota yhteyttä")'
    elif language == 'czech':
        return f'"{search_term}" (email OR kontakt OR "kontaktujte" OR "kontaktujte nás" OR "kontaktní formulář" OR "odeslat zprávu" OR "odešlete nám zprávu" OR "napište nám" OR "kontaktujte")'
    elif language == 'hungarian':
        return f'"{search_term}" (e-mail OR kapcsolat OR "lépjen kapcsolatba" OR "lépjen kapcsolatba velünk" OR "kapcsolatfelvételi űrlap" OR "üzenet küldése" OR "küldjön üzenetet" OR "írjon nekünk" OR "kapcsolat")'

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
    """Page for manual search and email sending."""
    st.title("🔍 Manual Search")

    with db_session() as session:
        project_id = get_active_project_id()
        if project_id is None:
            st.warning("Please select a project first.")
            return

        campaign_id = get_active_campaign_id()
        if campaign_id is None:
            st.warning("Please select a campaign first.")
            return

        # Fetch email settings and templates only once
        email_settings = fetch_email_settings(session)
        email_templates = fetch_email_templates(session)

        # Display search settings
        search_settings = display_search_settings()

        # Display email settings and template selection
        selected_email_setting_id, selected_template_id = display_email_options(email_settings, email_templates)

        # Search button and results
        if st.button("Search"):
            if not search_settings['search_terms']:
                st.error("Please enter at least one search term.")
            return

            # Fetch email setting and template based on selected IDs
            selected_email_setting = next((setting for setting in email_settings if setting['id'] == selected_email_setting_id), None)
            selected_template = next((template for template in email_templates if template['id'] == selected_template_id), None)

            if search_settings['enable_email_sending'] and (not selected_email_setting or not selected_template):
                st.error("Please select both an email setting and an email template.")
                return

            # Perform the search
            try:
                with st.spinner('Searching...'):
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
                    st.subheader("Search Results")
                    if search_results:
                        st.dataframe(pd.DataFrame(search_results))

                        # Process and display new leads
                        new_leads_all = []
                        for result in search_results:
                            new_leads = process_search_result(session, result, project_id, campaign_id)
                            new_leads_all.extend(new_leads)

                        if new_leads_all:
                            st.subheader("New Leads")
                            leads_df = pd.DataFrame(new_leads_all, columns=['Email', 'URL'])
                            st.dataframe(leads_df, hide_index=True)

                            # Send emails if enabled
                            if search_settings['enable_email_sending']:
                                st.subheader("Email Sending Results")
                                email_limit_check = check_email_limits(session, session.query(EmailSettings).get(selected_email_setting_id))

                                if email_limit_check['can_send']:
                                    with st.spinner('Sending emails...'):
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()
                                        results_container = st.empty()
                                        log_container = st.empty()

                                        logs, sent_count = bulk_send_emails(
                                            session,
                                            selected_template_id,
                                            selected_email_setting['email'],
                                            selected_email_setting['email'],
                                            leads_df.to_dict('records'),
                                            progress_bar,
                                            status_text,
                                            results_container,
                                            log_container
                                        )

                                        st.success(f"Successfully sent {sent_count} emails.")
                                        if logs:
                                            st.write("Email Logs:")
                                            for log in logs:
                                                st.text(log)
                                else:
                                    st.error("Email sending is currently limited. Please check your email settings or try again later.")
                            else:
                                st.info("No new leads found.")
                        else:
                            st.info("No results found for the given search terms.")

            except Exception as e:
                st.error(f"An error occurred during the search: {e}")

# Update other functions that might be accessing detached objects

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

    col1, col2 = st.columns([2, 1])
    with col1:
        status = "Active" if st.session_state.get('automation_status', False) else "Inactive"
        st.metric("Automation Status", status)
    with col2:
        button_text = "Stop Automation" if st.session_state.get('automation_status', False) else "Start Automation"
        if st.button(button_text, use_container_width=True):
            st.session_state.automation_status = not st.session_state.get('automation_status', False)
            if st.session_state.automation_status:
                st.session_state.automation_logs = []
            st.rerun()

    if st.button("Perform Quick Scan", use_container_width=True):
        with st.spinner("Performing quick scan..."):
            try:
                with db_session() as session:
                    new_leads = session.query(Lead).filter(Lead.is_processed == False).count()
                    session.query(Lead).filter(Lead.is_processed == False).update({Lead.is_processed: True})
                    session.commit()
                    st.success(f"Quick scan completed! Found {new_leads} new leads.")
            except Exception as e:
                st.error(f"An error occurred during quick scan: {str(e)}")

    st.subheader("Real-Time Analytics")
    try:
        with db_session() as session:
            total_leads = session.query(Lead).count()
            emails_sent = session.query(EmailCampaign).count()
            col1, col2 = st.columns(2)
            col1.metric("Total Leads", total_leads)
            col2.metric("Emails Sent", emails_sent)
    except Exception as e:
        st.error(f"An error occurred while displaying analytics: {str(e)}")

    st.subheader("Automation Logs")
    log_container = st.empty()
    update_display(log_container, st.session_state.get('automation_logs', []), "Latest Logs", "log")

    st.subheader("Recently Found Leads")
    leads_container = st.empty()

    if st.session_state.get('automation_status', False):
        st.info("Automation is currently running in the background.")
        try:
            with db_session() as session:
                while st.session_state.get('automation_status', False):
                    kb_info = get_knowledge_base_info(session, get_active_project_id())
                    if not kb_info:
                        st.session_state.automation_logs.append("Knowledge Base not found. Skipping cycle.")
                        time.sleep(3600)
                        continue

                    base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
                    optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)

                    new_leads_all = []
                    for term in optimized_terms:
                        results = manual_search(session, [term], 10)
                        new_leads = [(res['Email'], res['URL']) for res in results['results'] if save_lead(session, res['Email'], url=res['URL'])]
                        new_leads_all.extend(new_leads)

                        if new_leads:
                            template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                            if template:
                                from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                                reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                                logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': email} for email, _ in new_leads])
                                st.session_state.automation_logs.extend(logs)

                    if new_leads_all:
                        leads_df = pd.DataFrame(new_leads_all, columns=['Email', 'URL'])
                        leads_container.dataframe(leads_df, hide_index=True)
                    else:
                        leads_container.info("No new leads found in this cycle.")

                    update_display(log_container, st.session_state.get('automation_logs', []), "Latest Logs", "log")
                    time.sleep(3600)
        except Exception as e:
            st.error(f"An error occurred in the automation process: {str(e)}")

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

def bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_bar=None, status_text=None, results=None, log_container=None):
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
            if status_text:
                status_text.text(f"Processed {index + 1}/{total_leads} leads")
            if results is not None:
                results.append({"Email": lead['Email'], "Status": status})

            if log_container:
                log_container.text(log_message)

        except EmailNotValidError:
            log_message = f"❌ Invalid email address: {lead['Email']}"
            logs.append(log_message)
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
    """Runs the automated_search.py script."""
    try:
        process = subprocess.Popen(
            ["python", "automated_search.py", str(automation_log_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return process
    except Exception as e:
        logging.error(f"Failed to start automated search: {str(e)}")
        return None

def cleanup_search_state():
    """Cleans up search state if it's been too long since the last update"""
    if 'worker_log_state' in st.session_state:
        current_time = time.time()
        if current_time - st.session_state.worker_log_state['last_update'] > 3600:  # 1 hour timeout
            st.session_state.worker_log_state = {
                'buffer': [],
                'last_count': 0,
                'last_update': current_time,
                'update_counter': 0,
                'auto_scroll': True
            }
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

    with db_session() as session:
        active_project_id = get_active_project_id()
        active_campaign_id = get_active_campaign_id()

        # Fetch the active project and campaign details
        project = session.query(Project).get(active_project_id)
        campaign = session.query(Campaign).get(active_campaign_id)

        if project:
            st.subheader(f"Active Project: {project.name}")
            st.write(f"Description: {project.description}")

        if campaign:
            st.subheader(f"Active Campaign: {campaign.name}")
            st.write(f"Description: {campaign.description}")

        # Button to start the automated search
        if st.button("Start Automated Search"):
            automation_log = AutomationLog(
                project_id=active_project_id,
                campaign_id=active_campaign_id,
                start_time=datetime.now(),
                status="running"
            )
            session.add(automation_log)
            session.flush()  # Ensure automation_log has an ID
            session.commit()

            # Run the automated search process
            process = run_automated_search(automation_log.id)
            if process:
                st.session_state.search_process = process
                st.session_state.automation_log_id = automation_log.id
                st.success(f"Automated search started with log ID: {automation_log.id}")
            else:
                st.error("Failed to start automated search.")

        # Display logs if an automation log ID is in the session state
        if 'automation_log_id' in st.session_state:
            automation_log_id = st.session_state.automation_log_id
            logs = fetch_logs_for_automation_log(session, automation_log_id)
            if logs:
                st.subheader("Search Logs:")
                for log in logs:
                    st.text(log)

        # Button to stop the automated search
        if 'search_process' in st.session_state and st.session_state.search_process:
            if st.button("Stop Automated Search"):
                st.session_state.search_process.terminate()
                st.session_state.search_process = None

                # Update the automation log status
                automation_log = session.query(AutomationLog).get(st.session_state.automation_log_id)
                if automation_log:
                    automation_log.status = "stopped"
                    automation_log.end_time = datetime.now()
                    session.commit()
                    st.success("Automated search stopped.")

        # Display worker logs
        if 'worker_log_state' in st.session_state:
            st.subheader("Worker Logs:")
            worker_log_state = st.session_state.worker_log_state
            if worker_log_state['buffer']:
                for log in worker_log_state['buffer']:
                    st.text(log)
            else:
                st.info("No logs available.")

            # Auto-scroll functionality
            if worker_log_state['auto_scroll']:
                js = f"""
                <script>
                    var element = window.parent.document.getElementById('worker-logs');
                    if (element) {{
                        element.scrollTop = element.scrollHeight;
                    }}
                </script>
                """
                st.components.v1.html(js)

            # Button to toggle auto-scroll
            if st.button("Toggle Auto-Scroll"):
                worker_log_state['auto_scroll'] = not worker_log_state['auto_scroll']
                st.experimental_rerun()

        # Button to refresh logs
        if st.button("Refresh Logs"):
            st.experimental_rerun()

        # Button to clear logs
        if st.button("Clear Logs"):
            if 'worker_log_state' in st.session_state:
                st.session_state.worker_log_state['buffer'] = []
                st.session_state.worker_log_state['last_count'] = 0
                st.session_state.worker_log_state['update_counter'] = 0
            st.experimental_rerun()

        # Cleanup search state if it's been too long since the last update
        cleanup_search_state()

def get_active_project_id():
    """Get the currently active project ID from session state"""
    if not st.session_state.get("is_initialized"):
        initialize_session_state()
    validate_active_ids()
    return st.session_state.get("current_project_id")

def set_active_project_id(project_id):
    """Set the active project ID in session state"""
    st.session_state.current_project_id = project_id

def get_active_campaign_id():
    """Get the currently active campaign ID from session state"""
    if not st.session_state.get("is_initialized"):
        initialize_session_state()
    validate_active_ids()
    return st.session_state.get("current_campaign_id")

def set_active_campaign_id(campaign_id):
    """Set the active campaign ID in session state"""
    st.session_state.current_campaign_id = campaign_id

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
            set_active_project_id(1)  # Set default project
        if not get_active_campaign_id():
            set_active_campaign_id(1)  # Set default campaign

        # Verify the defaults exist
        project = session.query(Project).get(get_active_project_id())
        campaign = session.query(Campaign).get(get_active_campaign_id())

        if not project or not campaign:
            if selected not in ["🔄 Settings", "🚀 Projects & Campaigns"]:
                st.warning("Please set up a project and campaign first.")
                pages["🚀 Projects & Campaigns"]()
                return

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

if __name__ == "__main__":
    main()