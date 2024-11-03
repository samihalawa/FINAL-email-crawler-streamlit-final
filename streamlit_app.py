import os, json, re, logging, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid, random, smtplib
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from streamlit_option_menu import option_menu
from openai import OpenAI
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
from contextlib import contextmanager
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import distinct
import urllib3
from dotenv import load_dotenv
from email_validator import validate_email, EmailNotValidError
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import plotly.express as px
import asyncio, aiohttp
from contextlib import contextmanager
from sqlalchemy.pool import QueuePool
from typing import Generator
from functools import lru_cache
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

# Initialize database connection once
load_dotenv()  # Move this to the top before accessing env vars

DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("Database configuration incomplete. Check .env file.")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Initialize engine with proper configuration
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

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
    is_processed = Column(Boolean, default=False)  # Add this line
    campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead")
    email_campaigns = relationship("EmailCampaign", back_populates="lead")

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    project_id = Column(BigInteger, ForeignKey('projects.id'))  # Add project relationship
    template_name, subject, body_content = Column(Text), Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False)
    language = Column(Text, default='ES')
    campaign = relationship("Campaign")
    project = relationship("Project")  # Add project relationship
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

class TemplateCache:
    def __init__(self):
        self.last_refresh = datetime.now()
        self.cache_ttl = timedelta(minutes=5)
        self.templates = {}
    
    def should_refresh(self) -> bool:
        return datetime.now() - self.last_refresh > self.cache_ttl
    
    @lru_cache(maxsize=100)
    def get_template(self, template_id: int) -> Optional[Dict]:
        """Get single template with LRU caching"""
        if template_id in self.templates:
            return self.templates[template_id]
        return None

    def refresh_cache(self, session: Session) -> None:
        """Refresh template cache"""
        try:
            templates = session.query(EmailTemplate).all()
            self.templates = {t.id: {
                'name': t.template_name,
                'subject': t.subject,
                'body': t.body_content,
                'is_ai_customizable': t.is_ai_customizable
            } for t in templates}
            self.last_refresh = datetime.now()
        except Exception as e:
            logging.error(f"Template cache refresh failed: {e}")

template_cache = TemplateCache()

@contextmanager
def safe_db_connection() -> Generator[Session, None, None]:
    """Enhanced database connection with advanced pool management and monitoring"""
    session = None
    try:
        session = SessionLocal()
        # Add connection health check
        session.execute("SELECT 1")
        yield session
        session.commit()
    except SQLAlchemyError as e:
        if session:
            session.rollback()
        error_id = str(uuid.uuid4())
        logging.error(f"Database error {error_id}: {str(e)}")
        raise SQLAlchemyError(f"Database operation failed (Error ID: {error_id})")
    finally:
        if session:
            session.close()

def get_pool_status() -> dict:
    """Monitor connection pool status"""
    return {
        'pool_size': engine.pool.size(),
        'checkedin': engine.pool.checkedin(),
        'checkedout': engine.pool.checkedout(),
        'overflow': engine.pool.overflow()
    }

def settings_page():
    st.title("⚙️ Settings")
    
    # Add keyboard shortcuts
    st.markdown("*Press 'S' to save settings, 'R' to refresh*")
    
    tabs = st.tabs(["📧 Email Settings", "🔧 General Settings", "🗄️ Database Settings"])
    
    with tabs[0]:
        with safe_db_connection() as session:
            st.subheader("Email Configuration")
            
            # Add quick test button
            if st.button("🔍 Test All Email Settings"):
                with st.spinner("Testing email configurations..."):
                    test_results = test_email_settings(session)
                    for result in test_results:
                        if result['success']:
                            st.success(f"✓ {result['name']}: Connection successful")
                        else:
                            st.error(f"⚠ {result['name']}: {result['error']}")
            
            settings = session.query(EmailSettings).all()
            for setting in settings:
                with st.expander(f"📧 {setting.name} ({setting.email})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        name = st.text_input("Name", setting.name, key=f"n_{setting.id}")
                        email = st.text_input("Email", setting.email, key=f"e_{setting.id}")
                        if email and not is_valid_email(email):
                            st.warning("⚠ Please enter a valid email address")
                        
                        provider = st.selectbox("Provider", ["smtp", "ses"], 
                                             index=0 if setting.provider == "smtp" else 1,
                                             key=f"p_{setting.id}")
                    
                    with col2:
                        if provider == "smtp":
                            server = st.text_input("SMTP Server", setting.smtp_server, key=f"s_{setting.id}")
                            port = st.number_input("SMTP Port", value=setting.smtp_port or 587, key=f"pt_{setting.id}")
                            username = st.text_input("Username", setting.smtp_username, key=f"u_{setting.id}")
                            password = st.text_input("Password", type="password", key=f"pw_{setting.id}")
                            
                            # Add SMTP test button
                            if st.button("Test SMTP Connection", key=f"test_{setting.id}"):
                                with st.spinner("Testing SMTP connection..."):
                                    success, message = test_smtp_connection(server, port, username, password)
                                    if success:
                                        st.success("✓ SMTP connection successful!")
                                    else:
                                        st.error(f"⚠ SMTP connection failed: {message}")
                        else:
                            key_id = st.text_input("AWS Access Key ID", setting.aws_access_key_id, key=f"ak_{setting.id}")
                            secret = st.text_input("AWS Secret Key", type="password", key=f"sk_{setting.id}")
                            region = st.text_input("AWS Region", setting.aws_region, key=f"r_{setting.id}")
                            
                            # Add AWS test button
                            if st.button("Test AWS Connection", key=f"test_{setting.id}"):
                                with st.spinner("Testing AWS connection..."):
                                    success, message = test_aws_connection(key_id, secret, region)
                                    if success:
                                        st.success("✓ AWS connection successful!")
                                    else:
                                        st.error(f"⚠ AWS connection failed: {message}")
                    
                    col1, col2 = st.columns([3,1])
                    with col1:
                        if st.button("💾 Update", key=f"upd_{setting.id}", type="primary"):
                            with st.spinner("Saving settings..."):
                                if update_email_settings(session, setting.id, locals()):
                                    st.success("✓ Settings updated successfully!")
                                    time.sleep(0.5)
                                    st.rerun()
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{setting.id}", type="secondary"):
                            if st.checkbox("Confirm delete?", key=f"confirm_{setting.id}"):
                                delete_email_settings(session, setting.id)
                                st.success("Settings deleted!")
                                time.sleep(0.5)
                                st.rerun()
            
            # Enhanced new settings form
            with st.expander("➕ Add New Email Settings"):
                with st.form("new_email_settings"):
                    st.markdown("### Add New Email Configuration")
                    col1, col2 = st.columns(2)
                    with col1:
                        name = st.text_input("Name", placeholder="My Email Settings")
                        email = st.text_input("Email", placeholder="email@domain.com")
                        provider = st.selectbox("Provider", ["smtp", "ses"])
                    with col2:
                        if provider == "smtp":
                            server = st.text_input("SMTP Server", placeholder="smtp.gmail.com")
                            port = st.number_input("SMTP Port", value=587)
                            username = st.text_input("Username")
                            password = st.text_input("Password", type="password")
                        else:
                            key_id = st.text_input("AWS Access Key ID")
                            secret = st.text_input("AWS Secret Key", type="password")
                            region = st.text_input("AWS Region", placeholder="us-east-1")
                    
                    if st.form_submit_button("Add Email Settings", type="primary"):
                        with st.spinner("Adding new settings..."):
                            if add_new_email_settings(session, locals()):
                                st.success("✓ New settings added successfully!")
                                time.sleep(0.5)
                                st.rerun()

def update_email_settings(session, setting_id, data):
    try:
        setting = session.query(EmailSettings).get(setting_id)
        if setting:
            setting.name = data['name']
            setting.email = data['email']
            setting.provider = data['provider']
            
            if data['provider'] == 'smtp':
                setting.smtp_server = data['server']
                setting.smtp_port = data['port']
                setting.smtp_username = data['username']
                if data['password']:
                    setting.smtp_password = data['password']
            else:
                setting.aws_access_key_id = data['key_id']
                if data['secret']:
                    setting.aws_secret_access_key = data['secret']
                setting.aws_region = data['region']
            
            session.commit()
            st.success("Settings updated!")
            st.rerun()
    except Exception as e:
        session.rollback()
        st.error(f"Error updating settings: {str(e)}")

def add_new_email_settings(session):
    with st.form("new_email_settings"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name")
            email = st.text_input("Email")
            provider = st.selectbox("Provider", ["smtp", "ses"])
        with col2:
            if provider == "smtp":
                server = st.text_input("SMTP Server")
                port = st.number_input("SMTP Port", value=587)
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
            else:
                key_id = st.text_input("AWS Access Key ID")
                secret = st.text_input("AWS Secret Key", type="password")
                region = st.text_input("AWS Region")
        
        if st.form_submit_button("Add Email Settings", type="primary"):
            try:
                setting = EmailSettings(
                    name=name,
                    email=email,
                    provider=provider,
                    smtp_server=server if provider == "smtp" else None,
                    smtp_port=port if provider == "smtp" else None,
                    smtp_username=username if provider == "smtp" else None,
                    smtp_password=password if provider == "smtp" else None,
                    aws_access_key_id=key_id if provider == "ses" else None,
                    aws_secret_access_key=secret if provider == "ses" else None,
                    aws_region=region if provider == "ses" else None
                )
                session.add(setting)
                session.commit()
                st.success("Email settings added!")
                st.rerun()
            except Exception as e:
                st.error(f"Error adding settings: {str(e)}")

# 1. Optimized send_email_ses
def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    """Enhanced email sending with SES and SMTP support"""
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

# 2. Optimized save_email_campaign  
def save_email_campaign(session: Session, email: str, template_id: int, status: str, 
                       sent_at: datetime, subject: str, message_id: str, content: str) -> bool:
    """Save email campaign with enhanced error handling"""
    try:
        lead = session.query(Lead).filter_by(email=email).first()
        if not lead:
            logging.error(f"Lead not found for email: {email}")
            return False
            
        campaign = EmailCampaign(
            campaign_id=get_active_campaign_id(),
            lead_id=lead.id,
            template_id=template_id,
            status=status,
            sent_at=sent_at,
            original_subject=subject,
            original_content=content,
            message_id=message_id,
            tracking_id=str(uuid.uuid4())
        )
        session.add(campaign)
        session.commit()
        return True
    except Exception as e:
        logging.error(f"Error saving email campaign: {e}")
        session.rollback()
        return False

# 3. Optimized is_valid_email
def is_valid_email(email):
    """Single consolidated email validation function"""
    if email is None:
        return False
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^(test|prueba)@.*",
        r"^email@email\.com$",
        r".*@example\.com$", 
        r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"
    ]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    try:
        validate_email(email)
        return not (any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns) or 
                   any(email.lower().endswith(f"@{domain}") for domain in typo_domains))
    except EmailNotValidError:
        return False

# 4. Optimized extract_info_from_page
def extract_info_from_page(s):
    try: return {k:s.find('meta',{'name':k})['content'] if s.find('meta',{'name':k}) else '' 
                for k in ['author','og:site_name','job_title','description']}
    except Exception as e: logging.error(f"Extract error: {e}"); return {}

# 5. Optimized get_knowledge_base_info
def get_knowledge_base_info(session: Session, project_id: int) -> Optional[Dict[str,str]]:
    """Get knowledge base info for project"""
    try:
        kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        return kb.to_dict() if kb else None
    except Exception as e:
        logging.error(f"KB error: {e}")
        return None

# 6. Optimized update_search_terms
def update_search_terms(session: Session, terms_dict: Dict[str, List[str]]) -> None:
    """Update search terms with proper error handling"""
    try:
        for group, terms in terms_dict.items():
            for term in terms:
                existing = session.query(SearchTerm).filter_by(
                    term=term,
                    project_id=get_active_project_id()
                ).first()
                if existing:
                    existing.group = group
                else:
                    session.add(SearchTerm(
                        term=term,
                        group=group,
                        project_id=get_active_project_id()
                    ))
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error updating search terms: {e}")

# 7. Optimized fetch_email_settings
def fetch_email_settings(session: Session) -> List[Dict[str, Any]]:
    """Fetch email settings with error handling"""
    try:
        return [
            {"id": s.id, "name": s.name, "email": s.email}
            for s in session.query(EmailSettings).all()
        ]
    except Exception as e:
        logging.error(f"Error fetching settings: {e}")
        return []

# 8. Optimized save_lead
def save_lead(session, email, first_name=None, last_name=None, company=None, job_title=None, phone=None, url=None, search_term_id=None, created_at=None):
    """Enhanced lead saving with proper error handling"""
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
        
        if url and search_term_id:
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

# 9. Optimized wrap_email_body
def wrap_email_body(body: str, tracking_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
    """Enhanced email body wrapper with tracking and metadata support"""
    metadata_html = ""
    if metadata:
        metadata_html = f"""
        <div style='margin-bottom:20px;color:#666'>
            {chr(10).join(f'<strong>{k}:</strong> {v}<br>' for k,v in metadata.items())}
        </div>
        """
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <style>
        body {{
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #ffffff;
        }}
        a {{
            color: #007bff;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .button {{
            display: inline-block;
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .button:hover {{
            background-color: #0056b3;
        }}
    </style>
</head>
<body>
    {metadata_html}
    {body}
    {f"""<img src="https://autoclient-email-analytics.trigox.workers.dev/track?id={tracking_id}&type=open" 
         width="1" height="1" style="display:none"/>""" if tracking_id else ""}
</body>
</html>'''

# 10. Optimized fetch_leads
def fetch_leads(
    session: Session,
    template_id: int,
    option: str,
    email: str,
    search_terms: List[str],
    exclude_contacted: bool
) -> List[Dict[str, Any]]:
    try:
        query = session.query(Lead)
        if option == "Specific Email":
            query = query.filter(Lead.email == email)
        elif option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and search_terms:
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(search_terms))
        
        if exclude_contacted:
            contacted_leads = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None)).subquery()
            query = query.outerjoin(contacted_leads, Lead.id == contacted_leads.c.lead_id).filter(contacted_leads.c.lead_id.is_(None))
        
        return [{"Email": lead.email, "ID": lead.id} for lead in query.all()]
    except Exception as e:
        logging.error(f"Error fetching leads: {str(e)}")
        return []

def update_log(c, msg, lvl='info'): 
    if 'log_entries' not in st.session_state: st.session_state.log_entries = []
    st.session_state.log_entries.append(f"{{'info':'🔵','success':'🟢','warning':'','error':'🔴','email_sent':'🟣'}}.get(lvl,'⚪') {msg}")
    c.markdown(f"<div style='height:300px;overflow-y:auto;font-family:monospace'><br>".join(st.session_state.log_entries), unsafe_allow_html=True)

def optimize_search_term(term, lang): return f'"{term}" {{"en":"email OR contact site:.com","es":"correo OR contacto site:.es"}}.get(lang,term)'

def shuffle_keywords(term): return ' '.join(random.sample(term.split(),len(term.split())))

def get_domain_from_url(url): return urlparse(url).netloc

def manual_search(
    session: Session, 
    terms: List[str],
    num_results: int,
    ignore_previously_fetched: bool = True,
    optimize_english: bool = False,
    optimize_spanish: bool = False,
    shuffle_keywords_option: bool = False,
    language: str = 'ES',
    enable_email_sending: bool = True,
    log_container: Any = None,
    from_email: Optional[str] = None,
    reply_to: Optional[str] = None, 
    email_template: Optional[str] = None
) -> Dict[str, Any]:
    """Perform manual search for leads"""
    results = []
    try:
        for term in terms:
            # Apply optimizations
            if optimize_english and language == 'EN':
                term = f"{term} email OR contact site:.com"
            elif optimize_spanish and language == 'ES':
                term = f"{term} correo OR contacto site:.es"
            
            if shuffle_keywords_option:
                term = shuffle_keywords(term)

            # Perform search
            for url in google_search(term, num_results=num_results, lang=language):
                if ignore_previously_fetched:
                    domain = get_domain_from_url(url)
                    if session.query(LeadSource).filter(LeadSource.domain == domain).first():
                        continue
                
                # Extract info and save lead
                try:
                    page_info = extract_info_from_page(url)
                    if email := page_info.get('email'):
                        if is_valid_email(email):
                            lead = save_lead(session, email, url=url)
                            if lead:
                                results.append({
                                    'Email': email,
                                    'URL': url,
                                    'Company': page_info.get('company', '')
                                })
                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")
                    continue

        return {'results': results}
    except Exception as e:
        logging.error(f"Manual search error: {e}")
        return {'results': []}

def generate_or_adjust_email_template(
    prompt: str,
    kb_info: Optional[Dict[str, Any]] = None,
    current_template: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Generate or adjust email template using AI"""
    try:
        messages = [
            {"role": "system", "content": "You are an AI email template generator."},
            {"role": "user", "content": prompt}
        ]
        
        if kb_info:
            messages.append({
                "role": "user",
                "content": f"Context: {json.dumps(kb_info)}"
            })
            
        if current_template:
            messages.append({
                "role": "user", 
                "content": f"Adjust: {current_template}"
            })

        response = openai_chat_completion(
            messages, 
            function_name="generate_or_adjust_email_template"
        )
        
        if response and isinstance(response, str):
            template_data = json.loads(response)
            if all(k in template_data for k in ['subject', 'body']):
                return template_data
            
        return {"subject": "", "body": ""}
        
    except Exception as e:
        logging.error(f"Failed to generate template: {str(e)}")
        st.error(f"Failed to generate template: {str(e)}")
        return {"subject": "", "body": ""}
def fetch_leads_with_sources(session):
    try:
        results = session.query(
            Lead,
            func.string_agg(LeadSource.url, ', ').label('sources'),
            func.max(EmailCampaign.sent_at).label('last_contact'), 
            func.string_agg(EmailCampaign.status, ', ').label('email_statuses')
        ).outerjoin(
            LeadSource
        ).outerjoin(
            EmailCampaign
        ).filter(
            Lead.id.in_(
                session.query(CampaignLead.lead_id).filter(
                    CampaignLead.campaign_id == get_active_campaign_id()
                )
            )
        ).group_by(
            Lead.id
        ).all()
        
        return pd.DataFrame([{
            'id': lead.id,
            'email': lead.email,
            'first_name': lead.first_name,
            'last_name': lead.last_name,
            'company': lead.company,
            'job_title': lead.job_title,
            'created_at': lead.created_at,
            'Source': sources,
            'Last Contact': last_contact,
            'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted',
            'Delete': False
        } for lead, sources, last_contact, email_statuses in results])
    except Exception as e:
        st.error(f"Error fetching leads: {str(e)}")
        return pd.DataFrame()
def update_lead(session, lead_id, data):
    try:
        update_data = {
            k: v for k, v in data.items() 
            if k in ['email', 'first_name', 'last_name', 'company', 'job_title']
            and v is not None
        }
        if update_data:
            session.query(Lead).filter(Lead.id == lead_id).update(update_data)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        st.error(f"Error updating lead: {str(e)}")
        return False

def view_leads_page():
    st.title("Lead Management")
    
    with safe_db_connection() as session:
        leads_df = fetch_leads_with_sources(session)
        
        if not leads_df.empty:
            st.markdown("### Lead Database")
            
            filters = st.columns(3)
            with filters[0]:
                email_filter = st.text_input("Filter by Email", key="email_filter")
            with filters[1]:    
                company_filter = st.text_input("Filter by Company", key="company_filter")
            with filters[2]:
                date_filter = st.date_input("Filter by Date", key="date_filter")
            
            filtered_df = leads_df
            if email_filter:
                filtered_df = filtered_df[filtered_df['email'].str.contains(email_filter, case=False, na=False)]
            if company_filter:
                filtered_df = filtered_df[filtered_df['company'].str.contains(company_filter, case=False, na=False)]
            if date_filter:
                filtered_df = filtered_df[pd.to_datetime(filtered_df['created_at']).dt.date == date_filter]
            
            edited_df = st.data_editor(
                filtered_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "email": st.column_config.TextColumn("Email", width="medium"),
                    "company": st.column_config.TextColumn("Company", width="medium"),
                    "job_title": st.column_config.TextColumn("Job Title", width="medium"),
                    "created_at": st.column_config.DatetimeColumn("Created", width="small"),
                    "Source": st.column_config.TextColumn("Source URL", width="large"),
                    "Last Contact": st.column_config.DatetimeColumn("Last Contact", width="small"),
                    "Last Email Status": st.column_config.TextColumn("Status", width="small"),
                    "Delete": st.column_config.CheckboxColumn("Delete", width="small")
                }
            )
            
            if st.button("Update Selected", type="primary", key="update_leads"):
                changes = edited_df[edited_df != leads_df].dropna(how='all')
                if not changes.empty:
                    with st.spinner("Updating leads..."):
                        for idx, row in changes.iterrows():
                            if row['Delete']:
                                delete_lead(session, row['id'])
                            else:
                                update_lead(session, row['id'], row.to_dict())
                        st.success("Leads updated successfully!")
                        st.rerun()

def bulk_send_page():
    st.title("Bulk Email Campaign")
    
    with safe_db_connection() as session:
        templates = session.query(EmailTemplate).filter_by(campaign_id=get_active_campaign_id()).all()
        email_settings = [{'name': s.name, 'email': s.email} for s in session.query(EmailSettings).all()]

        if not templates or not email_settings:
            st.error("No email templates or settings available. Please set them up first.")
            return

        col1, col2 = st.columns([2,1])
        
        with col1:
            template = session.query(EmailTemplate).get(
                int(st.selectbox("Email Template", 
                    options=[f"{t.id}: {t.template_name}" for t in templates],
                    format_func=lambda x: x.split(":")[1].strip()
                ).split(":")[0])
            )


            st.markdown("### Preview")
            st.text(f"Subject: {template.subject}")
            st.components.v1.html(wrap_email_body(template.body_content), height=300, scrolling=True)

            lead_selection = st.radio("Select Recipients", 
                ["All Leads", "Specific Email", "Leads from Search Terms"])
            
            leads_data = []
            if lead_selection == "Specific Email":
                email = st.text_input("Enter Email")
                if email:
                    if is_valid_email(email):
                        leads_data = [{'Email': email}]
                    else:
                        st.error("Invalid email address")
            elif lead_selection == "Leads from Search Terms":
                terms = [t.term for t in session.query(SearchTerm).all()]
                selected_terms = st.multiselect("Select Search Terms", options=terms)
                if selected_terms:
                    leads_data = [{'Email': l.email} for l in session.query(Lead)
                        .join(LeadSource).join(SearchTerm)
                        .filter(SearchTerm.term.in_(selected_terms)).all()]
            else:
                leads_data = [{'Email': l.email} for l in session.query(Lead).all()]

        with col2:
            email_setting = st.selectbox("From Email", options=email_settings,
                format_func=lambda x: f"{x['name']} ({x['email']})")
            
            if email_setting:
                from_email = email_setting['email']
                reply_to = st.text_input("Reply To", email_setting['email'])
            else:
                st.error("No email setting selected")
                return

        preview_col, send_col = st.columns(2)
        with preview_col:
            if st.button("Preview Campaign", type="secondary", use_container_width=True):
                if leads_data:
                    df = pd.DataFrame(leads_data)
                    st.success(f"Campaign will send to {len(df)} leads")
                    st.dataframe(df, hide_index=True)
                else:
                    st.warning("No leads match the selected criteria")

        with send_col:
            if st.button("Send Campaign", type="primary", use_container_width=True):
                if not st.session_state.get('confirm_send'):
                    st.session_state.confirm_send = True
                    st.warning("Click again to confirm sending")
                else:
                    with st.spinner("Sending campaign..."):
                        progress = st.progress(0)
                        status = st.empty()
                        
                        sent, failed = 0, 0
                        for i, lead in enumerate(leads_data):
                            try:
                                email_content = wrap_email_body(template.body_content)
                                msg_id = send_email_ses(session, from_email, lead['Email'], 
                                    template.subject, email_content, reply_to)['MessageId']
                                
                                session.add(EmailCampaign(
                                    lead_email=lead['Email'],
                                    template_id=template.id,
                                    status='sent',
                                    sent_at=datetime.utcnow(),
                                    subject=template.subject,
                                    message_id=msg_id,
                                    content=template.body_content
                                ))
                                sent += 1
                                status.success(f"Sent to {lead['Email']}")
                            except Exception as e:
                                failed += 1
                                status.error(f"Failed to send to {lead['Email']}: {str(e)}")
                            progress.progress((i + 1) / len(leads_data))
                            
                        session.commit()
                        if sent:
                            st.balloons()
                            st.success(f"Campaign sent to {sent} leads ({failed} failed)")
                        else:
                            st.error("Campaign failed to send")
                        st.session_state.confirm_send = False

# Add these page functions after the existing ones

def view_campaign_logs():
    st.header("Email Logs")
    with safe_db_connection() as session:
        # Get logs from database
        logs = pd.read_sql(
            session.query(EmailCampaign).statement,
            session.bind
        )
        
        if logs.empty:
            st.info("No email logs found.")
            return
            
        # Date filters
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=logs['sent_at'].min().date())
        with col2:
            end_date = st.date_input("End Date", value=logs['sent_at'].max().date())

        # Filter by date range
        logs = logs[
            (logs['sent_at'].dt.date >= start_date) & 
            (logs['sent_at'].dt.date <= end_date)
        ]

        # Search filter
        search = st.text_input("Search by email or subject")
        if search:
            logs = logs[
                logs['lead_email'].str.contains(search, case=False) | 
                logs['subject'].str.contains(search, case=False)
            ]

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Emails Sent", len(logs))
        with col2:
            st.metric("Unique Recipients", logs['lead_email'].nunique())
        with col3:
            st.metric("Success Rate", f"{(logs['status'] == 'sent').mean():.2%}")

        # Daily volume chart
        daily_counts = logs.resample('D', on='sent_at')['lead_email'].count()
        st.bar_chart(daily_counts)

        # Paginated logs table
        per_page = 20
        pages = (len(logs) - 1) // per_page + 1
        page = st.number_input("Page", 1, pages, 1)
        start = (page - 1) * per_page
        
        st.table(logs.iloc[start:start+per_page][['sent_at', 'lead_email', 'subject', 'status']])

        # Export to CSV
        if st.button("Export Logs"):
            st.download_button(
                "Download CSV",
                logs.to_csv(index=False),
                "email_logs.csv",
                "text/csv"
            )

def projects_campaigns_page():
    with safe_db_connection() as session:
        st.header("Projects and Campaigns")
        
        # Add project form
        st.subheader("Add Project")
        with st.form("new_project"):
            name = st.text_input("Project Name")
            if st.form_submit_button("Add"):
                if name:
                    session.add(Project(
                        project_name=name,
                        created_at=datetime.utcnow()
                    ))
                    session.commit()
                    st.success(f"Added project: {name}")
                else:
                    st.warning("Enter a project name")

        # List projects
        projects = session.query(Project).all()
        for project in projects:
            with st.expander(f"Project: {project.project_name}"):
                # Add campaign form
                with st.form(f"new_campaign_{project.id}"):
                    name = st.text_input("Campaign Name")
                    if st.form_submit_button("Add"):
                        if name:
                            session.add(Campaign(
                                campaign_name=name,
                                project_id=project.id,
                                created_at=datetime.utcnow()
                            ))
                            session.commit()
                            st.success(f"Added campaign: {name}")
                        else:
                            st.warning("Enter a campaign name")

                # List campaigns
                campaigns = session.query(Campaign).filter_by(project_id=project.id).all()
                if campaigns:
                    st.write("Campaigns:")
                    for c in campaigns:
                        st.write(f"- {c.campaign_name}")
                else:
                    st.write("No campaigns yet")

        # Set active project/campaign
        if projects:
            project = st.selectbox(
                "Active Project",
                projects,
                format_func=lambda p: p.project_name
            )
            st.session_state.active_project_id = project.id

            campaigns = session.query(Campaign).filter_by(project_id=project.id).all()
            if campaigns:
                campaign = st.selectbox(
                    "Active Campaign",
                    campaigns,
                    format_func=lambda c: c.campaign_name
                )
                st.session_state.active_campaign_id = campaign.id
                st.success(f"Active: {project.project_name} / {campaign.campaign_name}")
            else:
                st.warning("Add a campaign first")
        else:
            st.warning("Add a project first")

def knowledge_base_page():
    st.title("Knowledge Base")
    
    with safe_db_connection() as session:
        # Get projects
        projects = session.query(Project).all()
        if not projects:
            return st.warning("Create a project first")
            
        # Select project
        project = st.selectbox(
            "Project",
            projects,
            format_func=lambda p: p.project_name
        )
        st.session_state.active_project_id = project.id
        
        # Get/create KB entry
        kb = session.query(KnowledgeBase).filter_by(project_id=project.id).first()
        if not kb:
            kb = KnowledgeBase(project_id=project.id)
            
        # KB form
        with st.form("kb_form"):
            fields = {
                'kb_name': st.text_input,
                'kb_bio': st.text_area,
                'kb_values': st.text_area,
                'contact_name': st.text_input,
                'contact_role': st.text_input,
                'contact_email': st.text_input,
                'company_description': st.text_area,
                'company_mission': st.text_area,
                'company_target_market': st.text_area,
                'company_other': st.text_area,
                'product_name': st.text_input,
                'product_description': st.text_area,
                'product_target_customer': st.text_area,
                'product_other': st.text_area,
                'other_context': st.text_area,
                'example_email': st.text_area
            }
            
            # Render fields
            data = {}
            for field, input_type in fields.items():
                label = field.replace('_', ' ').title()
                data[field] = input_type(label, getattr(kb, field, ''))
            
            if st.form_submit_button("Save"):
                try:
                    for k, v in data.items():
                        setattr(kb, k, v)
                    if not kb.id:
                        kb.created_at = datetime.utcnow()
                        session.add(kb)
                    session.commit()
                    st.success("Saved!")
                except Exception as e:
                    st.error(f"Error: {e}")

def delete_lead(session: Session, lead_id: int) -> bool:
    """Delete a lead and all associated records"""
    try:
        with session.begin_nested():
            # Delete all associated records in correct order
            for model in [CampaignLead, LeadSource, EmailCampaign, AIRequestLog]:
                session.query(model).filter_by(lead_id=lead_id).delete()
            session.query(Lead).filter_by(id=lead_id).delete()
            return True
    except Exception as e:
        logging.error(f"Error deleting lead {lead_id}: {e}")
        return False

def get_active_project_id() -> Optional[int]:
    return st.session_state.get('active_project_id')

def get_active_campaign_id() -> Optional[int]:
    return st.session_state.get('active_campaign_id')

def test_email_settings(session: Session) -> List[Dict[str, Any]]:
    """Test all email configurations"""
    return [{
        'name': s.name,
        'success': test_smtp_connection(s.smtp_server, s.smtp_port, s.smtp_username, s.smtp_password)[0] 
            if s.provider == 'smtp' 
            else test_aws_connection(s.aws_access_key_id, s.aws_secret_access_key, s.aws_region)[0],
        'error': None
    } for s in session.query(EmailSettings).all()]

def test_smtp_connection(server: str, port: int, username: str, password: str) -> Tuple[bool, str]:
    """Test SMTP server connection"""
    try:
        with smtplib.SMTP(server, port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            return True, "Success"
    except Exception as e:
        return False, str(e)

def test_aws_connection(key_id: str, secret: str, region: str) -> Tuple[bool, str]:
    """Test AWS SES connection"""
    try:
        session = boto3.Session(
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name=region
        )
        session.client('ses').get_send_quota()
        return True, "Success"
    except Exception as e:
        return False, str(e)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def send_email_ses(session: Session, from_email: str, to_email: str, 
                  subject: str, body: str, reply_to: Optional[str] = None) -> Dict[str, Any]:
    """Send email via SES with retries"""
    settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not settings:
        raise ValueError(f"No settings found for {from_email}")
        
    tracking_id = str(uuid.uuid4())
    tracked_body = add_tracking(body, tracking_id)
    
    if settings.provider == 'ses':
        client = boto3.client('ses',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        response = client.send_email(
            Source=from_email,
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {'Html': {'Data': tracked_body}}
            },
            ReplyToAddresses=[reply_to] if reply_to else []
        )
        return {'MessageId': response['MessageId'], 'TrackingId': tracking_id}
    else:
        raise ValueError(f"Unsupported provider: {settings.provider}")

def initialize_session_state() -> None:
    """Initialize all session state variables"""
    defaults = {
        'active_campaign_id': None,
        'active_project_id': None,
        'automation_status': False,
        'automation_logs': [],
        'confirm_send': False,
        'leads': pd.DataFrame(),
        'template_cache': {},
        'last_refresh': datetime.now()
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def setup_navigation() -> str:
    """Setup navigation menu and return selected page"""
    pages = {
        "Dashboard": "house",
        "Manual Search": "search", 
        "Automation": "robot",
        "Leads": "people",
        "Templates": "envelope",
        "Campaigns": "send",
        "Settings": "gear",
        "Logs": "journal-text"
    }
    return option_menu(
        None,
        list(pages.keys()),
        list(pages.values()),
        "horizontal",
        "main_menu"
    )

def manual_search_page():
    st.title("Manual Search")
    
    with safe_db_connection() as session:
        # Fetch recent searches
        recent_searches = session.query(SearchTerm).order_by(SearchTerm.created_at.desc()).limit(5).all()
        recent_search_terms = [term.term for term in recent_searches]
        
        email_templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)

    col1, col2 = st.columns([2, 1])
    with col1:
        search_terms = st_tags(
            label='Enter search terms:',
            text='Press enter to add more',
            value=recent_search_terms,
            suggestions=['software engineer', 'data scientist'],
            maxtags=10,
            key='search_terms_input'
        )
        num_results = st.slider("Results per term", 1, 50, 10)

    with col2:
        enable_email_sending = st.checkbox("Enable email sending", value=True)
        ignore_previously_fetched = st.checkbox("Ignore fetched domains", value=True)
        shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=True)
        optimize_english = st.checkbox("Optimize (English)", value=False)
        optimize_spanish = st.checkbox("Optimize (Spanish)", value=False)
        language = st.selectbox("Select Language", options=["ES", "EN"], index=0)

    # Rest of the manual search page implementation...

def add_tracking(body: str, tracking_id: str) -> str:
    """Add tracking pixel and link tracking to email body"""
    tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
    soup = BeautifulSoup(body, 'html.parser')
    
    # Add tracking to links
    for a in soup.find_all('a', href=True):
        original_url = a['href']
        tracked_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'click', 'url': original_url})}"
        a['href'] = tracked_url
    
    # Add tracking pixel
    if soup.body:
        pixel = soup.new_tag('img', src=tracking_pixel_url, width="1", height="1", style="display:none;")
        soup.body.append(pixel)
    
    return str(soup)

def fetch_email_templates(session: Session) -> List[str]:
    """Fetch all email templates"""
    return [f"{t.id}: {t.template_name}" for t in session.query(EmailTemplate).all()]

def fetch_search_term_groups(session: Session) -> List[str]:
    """Fetch all search term groups"""
    return [f"{g.id}: {g.name}" for g in session.query(SearchTermGroup).all()]

def fetch_search_terms_for_groups(session: Session, group_ids: List[int]) -> List[str]:
    """Fetch search terms for given groups"""
    return [t.term for t in session.query(SearchTerm).filter(SearchTerm.group_id.in_(group_ids)).all()]

def log_ai_request(session: Session, function_name: str, prompt: str, 
                  response: str, lead_id: Optional[int] = None, 
                  email_campaign_id: Optional[int] = None, 
                  model_used: Optional[str] = None) -> None:
    """Log AI request with enhanced error handling"""
    try:
        log = AIRequestLog(
            function_name=function_name,
            prompt=json.dumps(prompt),
            response=json.dumps(response) if response else None,
            lead_id=lead_id,
            email_campaign_id=email_campaign_id,
            model_used=model_used
        )
        session.add(log)
        session.commit()
    except Exception as e:
        logging.error(f"Error logging AI request: {e}")
        session.rollback()

def autoclient_ai_page():
    """AI automation control page"""
    st.header("AutoclientAI - Automated Lead Generation")
    
    # Add missing AI automation controls
    with st.expander("Knowledge Base Information", expanded=False):
        with safe_db_connection() as session:
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                return st.error("Knowledge Base not found. Please set it up first.")
            st.json(kb_info)

    # Add missing AI controls
    user_input = st.text_area("Enter additional context or specific goals:", 
                             help="This will help generate targeted search terms.")
    
    if st.button("Generate Optimized Terms"):
        with st.spinner("Generating optimized search terms..."):
            with safe_db_connection() as session:
                base_terms = [term.term for term in session.query(SearchTerm)
                            .filter_by(project_id=get_active_project_id()).all()]
                optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
                if optimized_terms:
                    st.session_state.optimized_terms = optimized_terms
                    st.success("Terms optimized successfully!")
                    st.write(", ".join(optimized_terms))

def email_templates_page():
    """Email template management page"""
    st.header("Email Templates")
    
    with safe_db_connection() as session:
        # Add missing template management UI
        with st.expander("Create New Template", expanded=False):
            new_template_name = st.text_input("Template Name")
            use_ai = st.checkbox("Use AI Generation")
            
            if use_ai:
                ai_prompt = st.text_area("AI Generation Prompt")
                use_kb = st.checkbox("Use Knowledge Base")
                if st.button("Generate Template"):
                    with st.spinner("Generating..."):
                        kb_info = get_knowledge_base_info(session, get_active_project_id()) if use_kb else None
                        generated = generate_or_adjust_email_template(ai_prompt, kb_info)
                        if generated:
                            st.success("Template generated!")
                            st.code(generated['body'], language='html')

def search_terms_page():
    """Search terms management page"""
    st.header("Search Terms Management")
    
    with safe_db_connection() as session:
        # Add missing search term management features
        col1, col2 = st.columns([2,1])
        with col1:
            terms = st_tags(
                label='Add Search Terms:',
                text='Press enter to add more',
                value=[],
                suggestions=['software engineer', 'data scientist'],
                maxtags=10
            )
        
        with col2:
            if st.button("Optimize Terms", type="primary"):
                with st.spinner("Optimizing..."):
                    optimized = optimize_search_terms(session, terms)
                    st.success(f"Optimized {len(optimized)} terms")

def analytics_dashboard():
    """Analytics dashboard page"""
    st.header("Campaign Analytics")
    
    with safe_db_connection() as session:
        # Add missing analytics visualizations
        col1, col2, col3 = st.columns(3)
        with col1:
            total_leads = session.query(Lead).count()
            st.metric("Total Leads", total_leads)
        with col2:
            emails_sent = session.query(EmailCampaign).count()
            st.metric("Emails Sent", emails_sent)
        with col3:
            success_rate = session.query(EmailCampaign).filter_by(status='sent').count() / emails_sent if emails_sent else 0
            st.metric("Success Rate", f"{success_rate:.1%}")

def generate_optimized_search_terms(session: Session, base_terms: List[str], kb_info: Dict) -> List[str]:
    """Generate optimized search terms using AI"""
    prompt = f"Optimize these search terms: {', '.join(base_terms)}\nContext: {json.dumps(kb_info)}"
    response = openai_chat_completion(
        messages=[{"role": "system", "content": prompt}],
        function_name="optimize_search_terms"
    )
    return response.get('terms', base_terms) if isinstance(response, dict) else base_terms

def get_campaign_analytics(session: Session, campaign_id: int) -> Dict[str, Any]:
    """Get campaign analytics data"""
    return {
        'total_leads': session.query(Lead).join(CampaignLead)
            .filter(CampaignLead.campaign_id == campaign_id).count(),
        'emails_sent': session.query(EmailCampaign)
            .filter(EmailCampaign.campaign_id == campaign_id).count(),
        'success_rate': session.query(EmailCampaign)
            .filter(EmailCampaign.campaign_id == campaign_id, 
                   EmailCampaign.status == 'sent').count()
    }

if __name__ == "__main__":
    st.set_page_config("Email Campaign Manager", "📧", "wide", "expanded")
    initialize_session_state()
    selected = setup_navigation()
    
    # Route to selected page
    page_funcs = {
        "Dashboard": lambda: st.title("Dashboard"),
        "Manual Search": lambda: st.title("Manual Search"),
        "Automation": lambda: st.title("Automation"),
        "Leads": view_leads_page,
        "Templates": lambda: st.title("Templates"),
        "Campaigns": lambda: st.title("Campaigns"),
        "Settings": settings_page,
        "Logs": view_campaign_logs
    }
    page_funcs[selected]()
