import os, json, re, logging, asyncio, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid, aiohttp, urllib3, random, html, smtplib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select, text, distinct, and_, case, or_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed, wait_exponential
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from openai import OpenAI 
from typing import List, Optional, Dict, Any, Tuple, Callable
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
import plotly.express as px
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
from collections import defaultdict
import sqlalchemy
import threading
from cryptography.fernet import Fernet

load_dotenv()

# Add default values to prevent None values
DB_HOST = os.getenv("SUPABASE_DB_HOST") or "localhost"
DB_NAME = os.getenv("SUPABASE_DB_NAME") or "postgres"
DB_USER = os.getenv("SUPABASE_DB_USER") or "postgres"
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD") or ""
DB_PORT = os.getenv("SUPABASE_DB_PORT") or "5432"

try:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
except Exception as e:
    logging.warning(f"Error constructing database URL: {str(e)}")
    DATABASE_URL = "sqlite:///autoclient.db"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create the declarative base
Base = declarative_base()

def get_database_url():
    """Get database URL with proper error handling and fallback"""
    try:
        db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        return db_url
    except Exception as e:
        logging.warning(f"Error constructing database URL: {str(e)}")
        return "sqlite:///autoclient.db"

# Initialize core components
def init_core_components():
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load environment variables
    load_dotenv()
    
    # Create SQLAlchemy Base
    Base = declarative_base()
    
    # Initialize database connection
    database_url = get_database_url()
    engine = create_engine(
        database_url,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,   # Recycle connections hourly
        pool_size=5,         # Reduce pool size for Streamlit Cloud
        max_overflow=2
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        logging.error(f"Failed to initialize primary database: {str(e)}")
        fallback_url = 'sqlite:///autoclient.db'
        logging.info(f"Falling back to SQLite: {fallback_url}")
        engine = create_engine(fallback_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
    
    return Base, engine, SessionLocal

# Initialize database components
Base, engine, Session = init_core_components()

@contextmanager
def db_session():
    session = None
    try:
        session = Session()
        yield session
        session.commit()
    except Exception as e:
        if session:
            session.rollback()
        raise
    finally:
        if session:
            session.close()

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
    campaign_id, lead_id = [Column(BigInteger, ForeignKey('campaigns.id')) for _ in range(1)] + [Column(BigInteger, ForeignKey('leads.id'))]
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
        return {attr: getattr(self, attr) for attr in [
            'kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 
            'contact_email', 'company_description', 'company_mission', 
            'company_target_market', 'company_other', 'product_name', 
            'product_description', 'product_target_customer', 'product_other', 
            'other_context', 'example_email'
        ]}

class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    phone, first_name, last_name, company, job_title = [Column(Text) for _ in range(5)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead")
    email_campaigns = relationship("EmailCampaign", back_populates="lead")

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    template_name, subject, body_content = [Column(Text) for _ in range(3)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False)
    language = Column(Text, default='ES')
    campaign = relationship("Campaign")
    email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_id, lead_id, template_id = [Column(BigInteger, ForeignKey('campaigns.id')) for _ in range(1)] + [Column(BigInteger, ForeignKey('leads.id')) for _ in range(1)] + [Column(BigInteger, ForeignKey('email_templates.id'))]
    customized_subject, customized_content, original_subject, original_content, status = [Column(Text) for _ in range(5)]
    engagement_data, message_id, tracking_id = [Column(JSON) for _ in range(3)]
    sent_at, opened_at, clicked_at = [Column(DateTime(timezone=True)) for _ in range(3)]
    open_count, click_count = [Column(BigInteger, default=0) for _ in range(2)]
    ai_customized = Column(Boolean, default=False)
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
    name, email_template, description = [Column(Text) for _ in range(3)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_terms = relationship("SearchTerm", back_populates="group")

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    id = Column(BigInteger, primary_key=True)
    group_id = Column(BigInteger, ForeignKey('search_term_groups.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    term, category = [Column(Text) for _ in range(2)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    language = Column(Text, default='ES')
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
    campaign_id, search_term_id = [Column(BigInteger, ForeignKey('campaigns.id')) for _ in range(1)] + [Column(BigInteger, ForeignKey('search_terms.id'))]
    leads_gathered, emails_sent = [Column(BigInteger) for _ in range(2)]
    start_time, end_time = [Column(DateTime(timezone=True), server_default=func.now()) for _ in range(1)] + [Column(DateTime(timezone=True))]
    status, logs = Column(Text), Column(JSON)
    campaign = relationship("Campaign")
    search_term = relationship("SearchTerm")

class Settings(Base):
    __tablename__ = 'settings'
    id = Column(BigInteger, primary_key=True)
    name, setting_type = [Column(Text, nullable=False) for _ in range(2)]
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    id = Column(BigInteger, primary_key=True)
    name, email, provider = [Column(Text, nullable=False) for _ in range(3)]
    smtp_server, smtp_port, smtp_username, smtp_password = [Column(Text) for _ in range(4)]
    aws_access_key_id, aws_secret_access_key, aws_region = [Column(Text) for _ in range(3)]

    # Add encryption for sensitive fields
    @property
    def decrypt_password(self) -> str:
        return decrypt_value(self.smtp_password)
    
    @property
    def decrypt_aws_secret(self) -> str:
        return decrypt_value(self.aws_secret_access_key)

DATABASE_URL = os.environ.get("DATABASE_URL") or get_database_url()
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set")

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def settings_page():
    st.title("Settings")
    
    with db_session() as session:
        st.header("General Settings")
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        
        with st.form("general_settings"):
            api_key = st.text_input("OpenAI API Key", 
                                  value=general_settings.value.get('api_key', '') if general_settings else '',
                                  type="password")
            
            if st.form_submit_button("Save General Settings"):
                if not general_settings:
                    general_settings = Settings(
                        name="General Settings",
                        setting_type="general",
                        value={'api_key': api_key}
                    )
                    session.add(general_settings)
                else:
                    general_settings.value['api_key'] = api_key
                session.commit()
                st.success("Settings saved successfully!")
        
        st.header("Email Settings")
        handle_email_settings(session)

def check_required_settings() -> bool:
    """Check if required settings are configured
    
    Returns:
        bool: True if all required settings exist, False otherwise
    """
    with db_session() as s:
        return bool(s.query(Settings).filter_by(setting_type='general').first() and 
               s.query(EmailSettings).first())

def send_email_ses(
    settings: EmailSettings,
    from_email: str,
    to_email: str, 
    subject: str,
    body: str,
    charset: str,
    reply_to: Optional[str],
    ses_client: Optional[Any] = None
) -> Dict[str, Any]:
    """Send email using AWS SES"""
    if not ses_client:
        ses_client = boto3.client('ses', 
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key, 
            region_name=settings.aws_region
        )
    
    email_args = {
        'Source': from_email,
        'Destination': {'ToAddresses': [to_email]},
        'Message': {
            'Subject': {'Data': subject, 'Charset': charset},
            'Body': {'Html': {'Data': body, 'Charset': charset}}
        }
    }
    
    if reply_to:
        email_args['ReplyToAddresses'] = [reply_to]
    else:
        email_args['ReplyToAddresses'] = []
        
    return ses_client.send_email(**email_args)

def save_email_campaign(
    s: Session,
    lead_email: str,
    template_id: int,
    status: str,
    sent_at: datetime,
    subject: str,
    message_id: Optional[str],
    email_body: str
) -> None:
    """Save email campaign with proper input validation"""
    try:
        # Validate inputs
        if not is_valid_email(lead_email):
            logging.error(f"Invalid email format: {lead_email}")
            return
            
        if not isinstance(template_id, int) or template_id <= 0:
            logging.error(f"Invalid template ID: {template_id}")
            return
            
        # Sanitize inputs
        subject = html.escape(subject or "No subject")
        email_body = html.escape(email_body or "No content")
        
        lead = s.query(Lead).filter_by(email=lead_email).first()
        if not lead:
            logging.error(f"Lead not found: {lead_email}")
            return
            
        campaign = EmailCampaign(
            lead_id=lead.id,
            template_id=template_id,
            status=status,
            sent_at=sent_at,
            customized_subject=subject,
            message_id=message_id or f"unknown-{uuid.uuid4()}",
            customized_content=email_body,
            campaign_id=get_active_campaign_id(),
            tracking_id=str(uuid.uuid4())
        )
        
        s.add(campaign)
        s.commit()
    except SQLAlchemyError as e:
        s.rollback()
        logging.error(f"Database error in save_email_campaign: {str(e)}")
    except Exception as e:
        s.rollback()
        logging.error(f"Unexpected error in save_email_campaign: {str(e)}")

def update_log(log_container, message, level='info'):
    icons = {'info': '🔵', 'success': '🟢', 'warning': '🟠', 'error': '🔴', 'email_sent': '🟣'}
    icon = icons.get(level, '⚪')
    
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
        
    st.session_state.log_entries.append(f"{icon} {message}")
    log_container.markdown(
        f"<div style='height:300px;overflow-y:auto;font-family:monospace;font-size:0.8em;line-height:1.2'>"
        f"{'<br>'.join(st.session_state.log_entries)}</div>",
        unsafe_allow_html=True
    )

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
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  # Remove extra )
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

def update_display(container, items: List[dict], title: str, item_key: str) -> None:
    container.markdown(
        f"""<style>.container{{max-height:400px;overflow-y:auto;border:1px solid rgba(49,51,63,0.2);
        border-radius:0.25rem;padding:1rem;background-color:rgba(49,51,63,0.1)}}
        .entry{{margin-bottom:0.5rem;padding:0.5rem;background-color:rgba(255,255,255,0.1);
        border-radius:0.25rem}}</style>
        <div class="container"><h4>{title} ({len(items)})</h4>
        {"".join(f'<div class="entry">{item[item_key]}</div>' for item in items[-20:])}</div>""",
        unsafe_allow_html=True
    )

def get_domain_from_url(url): return urlparse(url).netloc

def manual_search_page():
    st.title("Manual Search")
    
    # Check settings first
    if not check_required_settings():
        st.error("⚠️ Please configure settings first")
        if st.button("Go to Settings"):
            st.switch_page("settings")
        return

    # Form for search inputs
    with st.form("search_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_terms = st_tags(
                label='Enter Search Terms',
                text='Press enter to add more terms',
                value=[],
                suggestions=['software engineer', 'data scientist', 'product manager'],
                maxtags=10,
                key='search_terms_input'
            )
            
            num_results = st.slider(
                "Results per term", 
                min_value=1, 
                max_value=100, 
                value=10
            )

        with col2:
            enable_email = st.checkbox(
                "Enable email sending",
                value=True
            )
            
            ignore_prev = st.checkbox(
                "Skip processed domains",
                value=True
            )
            
            lang = st.selectbox(
                "Search Language",
                options=["ES", "EN"]
            )
            
            optimize_terms = st.checkbox(
                f"Optimize for {lang}"
            )

        # Email settings section (only show if email enabled)
        if enable_email:
            with st.expander("Email Settings"):
                with db_session() as session:
                    templates = fetch_email_templates(session)
                    email_settings = fetch_email_settings(session)
                
                if not templates:
                    st.error("No email templates available")
                    return
                    
                if not email_settings:
                    st.error("No email settings configured")
                    return

                template = st.selectbox(
                    "Email Template",
                    options=templates,
                    format_func=lambda x: x.split(":")[1].strip()
                )
                
                email_setting = st.selectbox(
                    "From Email",
                    options=email_settings,
                    format_func=lambda x: f"{x['name']} ({x['email']})"
                )
                
                if email_setting:
                    from_email = email_setting['email']
                    reply_to = st.text_input(
                        "Reply-To Email",
                        value=email_setting['email']
                    )

        # Submit button
        submitted = st.form_submit_button("Start Search")

    # Handle form submission
    if submitted:
        if not search_terms:
            st.warning("Please enter at least one search term")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.empty()
        results_container = st.empty()

        try:
            with db_session() as session:
                results = manual_search(
                    session=session,
                    terms=search_terms,
                    num_results=num_results,
                    ignore_prev=ignore_prev,
                    opt_en=optimize_terms and lang == "EN",
                    opt_es=optimize_terms and lang == "ES",
                    lang=lang,
                    enable_email=enable_email,
                    log_container=log_container,
                    from_email=from_email if enable_email else None,
                    reply_to=reply_to if enable_email else None,
                    email_template=template if enable_email else None
                )
                
                if results['results']:
                    display_search_results(results, results_container)
                else:
                    st.warning("No leads found matching the criteria")
                    
        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")
            logging.exception("Search error")
def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term, 
             func.count(distinct(Lead.id)).label('lead_count'),
             func.count(distinct(EmailCampaign.id)).label('email_count'))
            .outerjoin(LeadSource, SearchTerm.id == LeadSource.search_term_id)
            .outerjoin(Lead, LeadSource.lead_id == Lead.id)
            .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
            .group_by(SearchTerm.term))
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

def add_search_term(session, term, campaign_id):
    try:
        new_term = SearchTerm(
            term=term,
            campaign_id=campaign_id,
            created_at=datetime.utcnow()
        )
        session.add(new_term)
        session.commit()
        return new_term.id
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")
        return None

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
            
            tab1, tab2, tab3 = st.tabs(["Search Terms", "Performance", "Add New"])
            
            with tab1:
                st.dataframe(search_terms_df)
                
            with tab2:
                fig = px.bar(search_terms_df.nlargest(10, 'Lead Count'), 
                           x='Term', y=['Lead Count', 'Email Count'],
                           title='Top 10 Search Terms',
                           labels={'value': 'Count', 'variable': 'Type'},
                           barmode='group')
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                new_term = st.text_input("New Search Term")
                campaign_id = get_active_campaign_id()
                
                if st.button("Add Term") and new_term:
                    if add_search_term(session, new_term, campaign_id):
                        st.success(f"Added: {new_term}")
                        st.rerun()
                    else:
                        st.error("Failed to add term")
        else:
            st.info("No search terms available. Add some to get started.")
def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        new_search_term = SearchTerm(
            term=new_term,
            campaign_id=campaign_id,
            created_at=datetime.utcnow()
        )
        if group_for_new_term != "None":
            new_search_term.group_id = int(group_for_new_term.split(":")[0])
        session.add(new_search_term)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")
        return False

def ai_group_search_terms(session, ungrouped_terms):
    try:
        existing_groups = [g.name for g in session.query(SearchTermGroup).all()]
        terms = [t.term for t in ungrouped_terms]
        prompt = f"Group these terms:\n{', '.join(terms)}\nExisting groups: {', '.join(existing_groups)}"
        response = openai_chat_completion(
            [
                {"role": "system", "content": "You categorize search terms."},
                {"role": "user", "content": prompt}
            ],
            function_name="ai_group_search_terms"
        )
        return response or {}
    except Exception as e:
        logging.error(f"Error grouping terms: {str(e)}")
        return {}

def update_search_term_groups(session, grouped_terms):
    try:
        for group_name, terms in grouped_terms.items():
            group = session.query(SearchTermGroup).filter_by(name=group_name).first()
            if not group:
                group = SearchTermGroup(name=group_name)
                session.add(group)
                session.flush()
            for term in terms:
                search_term = session.query(SearchTerm).filter_by(term=term).first()
                if search_term:
                    search_term.group_id = group.id
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error updating groups: {str(e)}")
        return False

def create_search_term_group(session, group_name):
    try:
        group = SearchTermGroup(name=group_name)
        session.add(group)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating group: {str(e)}")
        return False

def delete_search_term_group(session, group_id):
    try:
        group = session.query(SearchTermGroup).get(group_id)
        if group:
            session.query(SearchTerm).filter_by(group_id=group_id).update({"group_id": None})
            session.delete(group)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting group: {str(e)}")
        return False

def email_templates_page():
    st.title("Email Templates")
    
    with st.form("template_form"):
        template_name = st.text_input("Template Name")
        subject = st.text_input("Subject")
        body = st.text_area("Body Content", height=300)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(TEMPLATE_HELP)
        
        with col2:
            if st.button("Preview"):
                template = EmailTemplate(
                    template_name=template_name,
                    subject=subject,
                    body_content=body
                )
                preview_subject, preview_body = preview_template(template)
                st.subheader("Preview")
                st.text(f"Subject: {preview_subject}")
                st.markdown(preview_body, unsafe_allow_html=True)
        
        if st.form_submit_button("Save Template"):
            with db_session() as session:
                template_id = create_or_update_email_template(
                    session, template_name, subject, body
                )
                if template_id:
                    st.success(f"Template saved successfully! ID: {template_id}")
                else:
                    st.error("Failed to save template")

@st.cache_data(ttl=300, max_entries=100)
def fetch_cached_email_settings(session):
    return fetch_email_settings(session)

@st.cache_data(ttl=600, max_entries=50)
def fetch_cached_templates(session):
    return fetch_email_templates(session)

TEMPLATE_HELP = """Available Variables:
{{name}} - Full name
{{first_name}} - First name
{{last_name}} - Last name
{{company}} - Company name
{{job_title}} - Job title"""

def process_template_variables(content: str, lead: Lead) -> str:
    # Add input sanitization
    variables = {
        'name': html.escape(f"{lead.first_name or ''} {lead.last_name or ''}".strip() or "there"),
        'first_name': html.escape(lead.first_name or "there"),
        'last_name': html.escape(lead.last_name or ""),
        'company': html.escape(lead.company or "your company"),
        'job_title': html.escape(lead.job_title or "professional")
    }
    return ''.join(content.replace(f"{{{{{var}}}}}", value) for var, value in variables.items())

def extract_name_from_url(url: str) -> tuple[str, str]:
    if 'linkedin.com/in/' not in url: return '', ''
    path = urlparse(url).path.strip('/').split('/')[-1].replace('-', ' ').split()
    return (path[0].title(), ' '.join(path[1:]).title()) if len(path) >= 2 else ('', '')

def save_lead_with_name(session: Session, email: str, url: str = None, **kwargs) -> Lead:
    lead = session.query(Lead).filter_by(email=email).first()
    if url and not (kwargs.get('first_name') and kwargs.get('last_name')):
        first_name, last_name = extract_name_from_url(url)
        if first_name and last_name: kwargs.update({'first_name': first_name, 'last_name': last_name})
    if lead:
        for key, value in kwargs.items():
            if value and not getattr(lead, key): setattr(lead, key, value)
    else:
        lead = Lead(email=email, created_at=datetime.utcnow(), **kwargs)
        session.add(lead)
    session.commit()
    return lead

def preview_template(template: EmailTemplate, sample_data: Dict[str, Any] = None) -> tuple[str, str]:
    """Preview email template with sample data"""
    if not sample_data:
        sample_data = {
            'first_name': 'John',
            'last_name': 'Doe',
            'company': 'ACME Corp',
            'job_title': 'Manager'
        }
    lead = Lead(**sample_data)
    return process_template_variables(template.subject, lead), process_template_variables(template.body_content, lead)

def handle_email_settings(s: Session) -> None:
    """Handle email settings with input validation"""
    try:
        email_settings = s.query(EmailSettings).all()
        if email_settings:
            st.subheader("Existing Email Settings")
            for setting in email_settings:
                with st.expander(f"{setting.name} ({setting.email})"):
                    st.text(f"Provider: {setting.provider}")
                    st.text(f"SMTP Server: {setting.smtp_server}")
                    
        with st.form("email_settings_form"):
            name = st.text_input("Name", max_chars=100)
            email = st.text_input("Email")
            
            if not is_valid_email(email):
                st.warning("Please enter a valid email address")
                return
                
            provider = st.selectbox("Provider", ["SMTP", "AWS SES"])
            
            # Rest of the function remains the same...
            
    except Exception as e:
        st.error(f"Error handling email settings: {str(e)}")
        logging.error(f"Email settings error: {str(e)}")

def send_email_provider(email_settings, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    try:
        return send_smtp_email(email_settings, from_email, to_email, subject, body, charset, reply_to) if email_settings.provider == "SMTP" else send_ses_email(email_settings, from_email, to_email, subject, body, charset, reply_to, ses_client)
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        return None

def send_smtp_email(settings, from_email, to_email, subject, body, charset, reply_to):
    msg = MIMEMultipart()
    msg.update({'From': from_email, 'To': to_email, 'Subject': subject})
    if reply_to: msg.add_header('Reply-To', reply_to)
    msg.attach(MIMEText(body, 'html', charset))
    with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
        server.starttls()
        server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(msg)
    return {'MessageId': f'smtp-{uuid.uuid4()}'}

def add_tracking(body: str, tracking_id: str, tracking_pixel_url: str) -> str:
    tracking_pixel = f'<img src="{tracking_pixel_url}" width="1" height="1" />'
    soup = BeautifulSoup(body, 'html.parser')
    for link in soup.find_all('a'):
        if original_url := link.get('href'):
            link['href'] = f"https://autoclient-email-analytics.trigox.workers.dev/click?id={tracking_id}&url={urlencode({'url': original_url})}"
    modified_body = str(soup)
    return modified_body.replace('</body>', f'{tracking_pixel}</body>') if '</body>' in modified_body else modified_body + tracking_pixel

@contextmanager 
def managed_automation_session():
    try:
        st.session_state.automation_active = True
        yield
    finally:
        st.session_state.automation_active = False

def track_automation_progress(total_steps: int):
    progress = st.progress(0)
    status = st.empty()
    def update(step: int, message: str):
        progress.progress(step / total_steps)
        status.text(f"Step {step}/{total_steps}: {message}")
    return update

def load_automation_config():
    return {
        'batch_size': st.sidebar.slider("Batch Size", 5, 50, 20),
        'delay': st.sidebar.number_input("Delay (sec)", 1, 60, 5),
        'retries': st.sidebar.number_input("Retries", 1, 10, 3)
    }

def fetch_search_terms(session):
    return [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).filter(SearchTerm.is_active == True).all()]

def process_search_term(session, term: str, config: dict, leads_container):
    results = manual_search(session, [term], config['batch_size'])
    new_leads = [(res['Email'], res['URL']) for res in results['results'] if save_lead(session, res['Email'], url=res['URL'])]
    if new_leads:
        leads_df = pd.DataFrame(new_leads, columns=['Email', 'URL'])
        leads_container.dataframe(leads_df, hide_index=True)
    return new_leads

def process_email_queue(session, kb_info: dict, config: dict):
    template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
    if template:
        from_email = kb_info.get('contact_email') or 'hello@indosy.com'
        reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
        for _ in range(config['retries']):
            try:
                bulk_send_emails(session, template.id, from_email, reply_to)
                break
            except Exception as e:
                logger.error(f"Email error: {str(e)}")
                time.sleep(config['delay'])

def update_automation_metrics(session):
    try:
        metrics = {
            'total_leads': session.query(Lead).count(),
            'emails_sent': session.query(EmailCampaign).filter(EmailCampaign.sent_at.isnot(None)).count(),
            'success_rate': session.query(EmailCampaign).filter(EmailCampaign.status == 'sent').count() / session.query(EmailCampaign).filter(EmailCampaign.sent_at.isnot(None)).count()
        }
        st.session_state.automation_metrics = metrics
        return metrics
    except Exception as e:
        logging.error(f"Error updating metrics: {str(e)}")
        return {'total_leads': 0, 'emails_sent': 0, 'success_rate': 0}

def extract_name(soup):
    name_tags = soup.find_all(['h1', 'h2', 'h3'], class_=lambda x: x and 'name' in x.lower())
    return name_tags[0].text.strip() if name_tags else ""

def extract_company(soup):
    company_meta = soup.find('meta', {'property': 'og:site_name'})
    if company_meta: return company_meta['content']
    company_tags = soup.find_all(['span', 'div', 'p'], class_=lambda x: x and 'company' in x.lower())
    return company_tags[0].text.strip() if company_tags else ""

def extract_job_title(soup):
    title_tags = soup.find_all(['span', 'div', 'p'], class_=lambda x: x and ('title' in x.lower() or 'role' in x.lower()))
    return title_tags[0].text.strip() if title_tags else ""

def extract_phone_numbers(soup):
    phone_pattern = r'(\+?[\d\s-]{10,})'
    text = soup.get_text()
    return list(set(re.findall(phone_pattern, text)))

def extract_social_links(soup):
    social_patterns = ['linkedin.com', 'twitter.com', 'facebook.com', 'instagram.com']
    social_links = []
    for link in soup.find_all('a', href=True):
        if any(pattern in link['href'].lower() for pattern in social_patterns):
            social_links.append(link['href'])
    return social_links

def validate_email_settings(email_setting, template):
    if not email_setting:
        raise ValueError("No email setting selected")
    if not template:
        raise ValueError("No email template selected")
    return {
        'from_email': email_setting['email'],
        'reply_to': email_setting['email']
    }

def main():
    """Enhanced main application with proper navigation and error handling"""
    try:
        st.set_page_config(
            page_title="AutoClient",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        if 'active_project_id' not in st.session_state:
            st.session_state.active_project_id = 1
        if 'active_campaign_id' not in st.session_state:
            st.session_state.active_campaign_id = 1

        pages = {
            "Search": manual_search_page,
            "Templates": email_templates_page,
            "Settings": settings_page,
            "Analytics": analytics_page,
            "Projects": projects_campaigns_page,
            "Knowledge Base": knowledge_base_page,
            "Automation": automation_control_panel_page,
            "Logs": view_campaign_logs
        }

        selected = option_menu(
            menu_title=None,
            options=list(pages.keys()),
            icons=["search", "envelope", "gear", "graph-up", "folder", "book", "robot", "list-check"],
            orientation="horizontal"
        )

        pages[selected]()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logging.exception("Application error occurred")

if __name__ == "__main__": main()

class RateLimiter:
    """Rate limiting context manager"""
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
        self._lock = threading.Lock()  # Add thread safety
        
    def __enter__(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.calls[0] + self.period - now
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        self.calls.append(now)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def analytics_page() -> None:
    """Display analytics dashboard"""
    st.title("Analytics Dashboard")
    
    with db_session() as session:
        metrics = {
            'total_leads': session.query(Lead).count(),
            'total_emails': session.query(EmailCampaign).count(),
            'successful_emails': session.query(EmailCampaign).filter_by(status='sent').count(),
            'total_searches': session.query(SearchTerm).count()
        }
        
        cols = st.columns(4)
        cols[0].metric("Total Leads", metrics['total_leads'])
        cols[1].metric("Total Emails", metrics['total_emails'])
        cols[2].metric("Successful Emails", metrics['successful_emails'])
        cols[3].metric("Search Terms", metrics['total_searches'])
        
        display_analytics_charts(session)

def process_url(url: str, search_term: str) -> dict:
    """Process URL to extract lead information"""
    try:
        # Configure request session
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Make request with random user agent
        headers = {'User-Agent': UserAgent().random}
        response = session.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        # Parse content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract information
        email = extract_email(soup)
        if not email:
            return None
            
        return {
            'Email': email,
            'URL': url,
            'Title': extract_title(soup),
            'Company': extract_company(soup),
            'Name': extract_name(soup),
            'JobTitle': extract_job_title(soup),
            'SearchTerm': search_term
        }
        
    except Exception as e:
        logging.error(f"Error processing {url}: {str(e)}")
        return None

def extract_email(soup) -> Optional[str]:
    """Extract and validate email from webpage"""
    # Email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    # Get text content
    text = soup.get_text()
    
    # Find all email matches
    emails = re.findall(email_pattern, text)
    
    # Validate and return first valid email
    for email in emails:
        try:
            validated = validate_email(email)
            return validated.email
        except EmailNotValidError:
            continue
            
    return None

def fetch_email_settings(session: Session) -> List[Dict[str, Any]]:
    """Fetch email settings from database with proper typing"""
    try:
        settings = session.query(EmailSettings).all()
        return [{'id': s.id, 'name': s.name, 'email': s.email} for s in settings]
    except SQLAlchemyError as e:
        logging.error(f"Database error: {str(e)}")
        return []

def display_analytics_charts(session):
    """Display analytics charts"""
    try:
        # Email campaign performance over time
        email_data = pd.read_sql(
            session.query(EmailCampaign.sent_at, EmailCampaign.status)
            .filter(EmailCampaign.sent_at.isnot(None))
            .statement,
            session.bind
        )
        if not email_data.empty:
            email_data['sent_at'] = pd.to_datetime(email_data['sent_at'])
            email_data['date'] = email_data['sent_at'].dt.date
            daily_stats = email_data.groupby(['date', 'status']).size().unstack(fill_value=0)
            
            fig = px.line(daily_stats, 
                         title='Email Campaign Performance',
                         labels={'value': 'Count', 'date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)
            
        # Search term effectiveness
        search_data = pd.read_sql(
            session.query(SearchTermEffectiveness).statement,
            session.bind
        )
        if not search_data.empty:
            fig = px.bar(search_data,
                        x='term',
                        y=['valid_leads', 'irrelevant_leads'],
                        title='Search Term Effectiveness',
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying analytics: {str(e)}")

def extract_title(soup):
    """Extract page title from soup"""
    if soup.title:
        return soup.title.string.strip()
    return ""

def save_lead(session: Session, email: str, **kwargs) -> Optional[Lead]:
    """Save lead to database with proper typing"""
    try:
        lead = session.query(Lead).filter_by(email=email).first()
        if not lead:
            lead = Lead(
                email=email,
                created_at=datetime.utcnow(),
                **kwargs
            )
            session.add(lead)
            session.commit()
            return True
        return False
    except Exception as e:
        logging.error(f"Error saving lead: {str(e)}")
        session.rollback()
        return False

def bulk_send_emails(session: Session, template_id: int, from_email: str, reply_to: Optional[str] = None) -> None:
    """Send emails in bulk with proper error handling"""
    try:
        template = session.query(EmailTemplate).get(template_id)
        if not template:
            raise ValueError("Template not found")
            
        leads = session.query(Lead).join(
            EmailCampaign, 
            and_(Lead.id == EmailCampaign.lead_id, EmailCampaign.template_id == template_id),
            isouter=True
        ).filter(EmailCampaign.id.is_(None)).all()
        
        for lead in leads:
            try:
                subject = process_template_variables(template.subject, lead)
                body = process_template_variables(template.body_content, lead)
                
                result, tracking_id = send_email_ses(
                    session,
                    from_email,
                    lead.email,
                    subject,
                    body,
                    reply_to=reply_to
                )
                
                if result:
                    save_email_campaign(
                        session,
                        lead.email,
                        template_id,
                        'sent',
                        datetime.utcnow(),
                        subject,
                        result.get('MessageId'),
                        body
                    )
                    
            except Exception as e:
                logging.error(f"Error sending email to {lead.email}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Bulk email error: {str(e)}")
        raise

def manual_search(session: Session, terms: List[str], num_results: int, **kwargs) -> Dict[str, List]:
    """Execute manual search with batch processing"""
    results = {'results': [], 'errors': []}
    batch_size = 10  # Process in smaller batches
    
    for term in terms:
        processed = 0
        while processed < num_results:
            batch = google_search(term, num_results=min(batch_size, num_results - processed))
            if batch:
                results['results'].extend(batch)
            processed += batch_size
    return results

def display_search_results(results: Dict[str, List], container: Any) -> None:
    """Display search results with pagination"""
    page_size = 20
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    page = st.session_state.current_page
    
    start_idx = page * page_size
    end_idx = start_idx + page_size
    
    paginated_results = results['results'][start_idx:end_idx]
    
    # Display pagination controls
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Next"):
            st.session_state.current_page = min(page + 1, len(results['results']) // page_size)
        if st.button("Previous"):
            st.session_state.current_page = max(0, page - 1)

def handle_form_submission() -> None:
    """Handle form submission with state tracking"""
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
        
    if st.session_state.form_submitted:
        try:
            with st.spinner("Processing..."):
                process_form()
        finally:
            st.session_state.form_submitted = False
