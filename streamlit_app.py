import os, json, re, logging, asyncio, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid, aiohttp, urllib3, random, html, smtplib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select, text, distinct, and_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from openai import OpenAI 
from typing import List, Optional
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
import plotly.express as px
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager

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

# Configure SQLAlchemy
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize OpenAI client
client = OpenAI()

# Configure session state
if 'automation_running' not in st.session_state:
    st.session_state['automation_running'] = False

if 'active_campaign_id' not in st.session_state:
    st.session_state['active_campaign_id'] = None

# ... existing code ...

class Project(Base):
    """Project model for organizing campaigns and knowledge bases."""
    __tablename__ = 'projects'
    
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text, default="Default Project")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    campaigns = relationship("Campaign", back_populates="project", cascade="all, delete-orphan")
    knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'project_name': self.project_name,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.project_name}')>"

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
        return {attr: getattr(self, attr) for attr in [
            'kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email',
            'company_description', 'company_mission', 'company_target_market', 'company_other',
            'product_name', 'product_description', 'product_target_customer', 'product_other',
            'other_context', 'example_email'
        ]}

class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True, index=True)
    phone = Column(Text)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text, index=True)
    job_title = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
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
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'), index=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'), index=True)
    template_id = Column(BigInteger, ForeignKey('email_templates.id'), index=True)
    customized_subject = Column(Text)
    customized_content = Column(Text)
    original_subject = Column(Text)
    original_content = Column(Text)
    status = Column(Text, index=True)
    engagement_data = Column(JSON)
    message_id = Column(Text, index=True)
    tracking_id = Column(Text, unique=True, index=True)
    sent_at = Column(DateTime(timezone=True), index=True)
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
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    leads_gathered, emails_sent = Column(BigInteger), Column(BigInteger)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    status, logs = Column(Text), Column(JSON)
    campaign = relationship("Campaign")
    search_term = relationship("SearchTerm")

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
    daily_limit = Column(BigInteger, default=999999999)
    hourly_limit = Column(BigInteger, default=999999999)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

@contextmanager
def db_session():
    """Create a new database session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

@contextmanager
def safe_db_session():
    """Create a new database session with error handling."""
    session = None
    try:
        session = SessionLocal()
        yield session
        session.commit()
    except SQLAlchemyError as e:
        if session:
            session.rollback()
        st.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        if session:
            session.rollback()
        st.error(f"An error occurred: {str(e)}")
        raise
    finally:
        if session:
            session.close()

def check_required_settings(session):
    """Check if all required settings are present in the database."""
    try:
        # Check database connection first
        session.execute(text('SELECT 1'))
        
        # Get settings from database
        settings = {
            'ai': session.query(Settings).filter_by(setting_type='ai').first(),
            'automation': session.query(Settings).filter_by(setting_type='automation').first(),
            'email_defaults': session.query(Settings).filter_by(setting_type='email_defaults').first()
        }
        
        # Check if all required settings exist
        missing_settings = [setting_type for setting_type, setting in settings.items() if not setting]
        if missing_settings:
            raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")
        
        return True
    except Exception as e:
        raise ValueError(f"Database connection failed: {str(e)}")

def initialize_settings():
    """Initialize database and required settings."""
    with safe_db_session() as session:
        try:
            # Create tables if they don't exist
            Base.metadata.create_all(bind=engine)
            
            # Create default AWS settings if they don't exist
            aws_settings = session.query(Settings).filter_by(setting_type='aws').first()
            if not aws_settings:
                default_aws_settings = {
                    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
                    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', ''),
                    'aws_region': os.getenv('AWS_REGION', 'us-east-1'),
                    'aws_ses_enabled': True
                }
                session.add(Settings(name='AWS Settings', setting_type='aws', value=default_aws_settings))
                session.commit()
            
            # Create default database settings if they don't exist
            db_settings = session.query(Settings).filter_by(setting_type='database').first()
            if not db_settings:
                default_db_settings = {
                    'max_connections': 20,
                    'pool_size': 5,
                    'pool_timeout': 30,
                    'pool_recycle': 1800
                }
                session.add(Settings(name='Database Settings', setting_type='database', value=default_db_settings))
                session.commit()
            
            # Create default automation settings if they don't exist
            automation_settings = session.query(Settings).filter_by(setting_type='automation').first()
            if not automation_settings:
                default_automation_settings = {
                    'max_leads_per_term': 100,
                    'max_emails_per_batch': 50,
                    'min_interval_seconds': 60,
                    'max_retries': 3,
                    'timeout_seconds': 30
                }
                session.add(Settings(name='Automation Settings', setting_type='automation', value=default_automation_settings))
                session.commit()
            
            # Create default AI settings if they don't exist
            ai_settings = session.query(Settings).filter_by(setting_type='ai').first()
            if not ai_settings:
                default_ai_settings = {
                    'model': 'gpt-4',
                    'temperature': 0.7,
                    'max_tokens': 2000,
                    'presence_penalty': 0.0,
                    'frequency_penalty': 0.0
                }
                session.add(Settings(name='AI Settings', setting_type='ai', value=default_ai_settings))
                session.commit()
            
            # Create default email settings if they don't exist
            email_settings = session.query(Settings).filter_by(setting_type='email_defaults').first()
            if not email_settings:
                default_email_settings = {
                    'from_email': os.getenv('DEFAULT_FROM_EMAIL', ''),
                    'reply_to': os.getenv('DEFAULT_REPLY_TO', ''),
                    'daily_limit': 2000,
                    'hourly_limit': 200,
                    'batch_size': 50,
                    'delay_seconds': 1
                }
                session.add(Settings(name='Email Default Settings', setting_type='email_defaults', value=default_email_settings))
                session.commit()
            
            return True
        except Exception as e:
            st.error(f"Failed to initialize settings: {str(e)}")
            return False

@safe_button_operation
def handle_automation_start():
    """Start the automation process."""
    if not st.session_state.get('active_campaign_id'):
        st.error("Please select a campaign first")
        return
    
    st.session_state['automation_running'] = True
    st.experimental_rerun()

@safe_button_operation
def handle_automation_stop():
    """Stop the automation process."""
    st.session_state['automation_running'] = False
    st.experimental_rerun()

def safe_button_operation(func):
    """Decorator for safe button operations with error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Operation failed: {str(e)}")
            return None
    return wrapper

def settings_page():
    """Settings page for configuring the application."""
    st.title("Settings")
    
    with safe_db_session() as session:
        try:
            # AWS Settings
            st.header("AWS Settings")
            aws_settings = session.query(Settings).filter_by(setting_type='aws').first()
            if aws_settings:
                aws_form = st.form("aws_settings")
                with aws_form:
                    aws_access_key = st.text_input("AWS Access Key ID", value=aws_settings.value.get('aws_access_key_id', ''), type='password')
                    aws_secret_key = st.text_input("AWS Secret Access Key", value=aws_settings.value.get('aws_secret_access_key', ''), type='password')
                    aws_region = st.text_input("AWS Region", value=aws_settings.value.get('aws_region', 'us-east-1'))
                    aws_ses_enabled = st.checkbox("Enable AWS SES", value=aws_settings.value.get('aws_ses_enabled', True))
                    
                    if st.form_submit_button("Save AWS Settings"):
                        aws_settings.value.update({
                            'aws_access_key_id': aws_access_key,
                            'aws_secret_access_key': aws_secret_key,
                            'aws_region': aws_region,
                            'aws_ses_enabled': aws_ses_enabled
                        })
            session.commit()
                        st.success("AWS settings saved successfully!")
            
            # Email Settings
            st.header("Email Settings")
            email_settings = session.query(Settings).filter_by(setting_type='email_defaults').first()
            if email_settings:
                email_form = st.form("email_settings")
                with email_form:
                    from_email = st.text_input("Default From Email", value=email_settings.value.get('from_email', ''))
                    reply_to = st.text_input("Default Reply-To", value=email_settings.value.get('reply_to', ''))
                    daily_limit = st.number_input("Daily Email Limit", value=email_settings.value.get('daily_limit', 2000), min_value=1)
                    hourly_limit = st.number_input("Hourly Email Limit", value=email_settings.value.get('hourly_limit', 200), min_value=1)
                    batch_size = st.number_input("Email Batch Size", value=email_settings.value.get('batch_size', 50), min_value=1)
                    delay_seconds = st.number_input("Delay Between Emails (seconds)", value=email_settings.value.get('delay_seconds', 1), min_value=1)
                    
                    if st.form_submit_button("Save Email Settings"):
                        email_settings.value.update({
                            'from_email': from_email,
                            'reply_to': reply_to,
                            'daily_limit': daily_limit,
                            'hourly_limit': hourly_limit,
                            'batch_size': batch_size,
                            'delay_seconds': delay_seconds
                        })
        session.commit()
                        st.success("Email settings saved successfully!")
            
            # AI Settings
            st.header("AI Settings")
            ai_settings = session.query(Settings).filter_by(setting_type='ai').first()
            if ai_settings:
                ai_form = st.form("ai_settings")
                with ai_form:
                    model = st.selectbox("OpenAI Model", ['gpt-4', 'gpt-3.5-turbo'], index=0 if ai_settings.value.get('model') == 'gpt-4' else 1)
                    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=ai_settings.value.get('temperature', 0.7), step=0.1)
                    max_tokens = st.number_input("Max Tokens", value=ai_settings.value.get('max_tokens', 2000), min_value=1)
                    presence_penalty = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, value=ai_settings.value.get('presence_penalty', 0.0), step=0.1)
                    frequency_penalty = st.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, value=ai_settings.value.get('frequency_penalty', 0.0), step=0.1)
                    
                    if st.form_submit_button("Save AI Settings"):
                        ai_settings.value.update({
                            'model': model,
                            'temperature': temperature,
                            'max_tokens': max_tokens,
                            'presence_penalty': presence_penalty,
                            'frequency_penalty': frequency_penalty
                        })
        session.commit()
                        st.success("AI settings saved successfully!")
            
            # Automation Settings
            st.header("Automation Settings")
        automation_settings = session.query(Settings).filter_by(setting_type='automation').first()
            if automation_settings:
                automation_form = st.form("automation_settings")
                with automation_form:
                    max_leads = st.number_input("Max Leads per Search Term", value=automation_settings.value.get('max_leads_per_term', 100), min_value=1)
                    max_emails = st.number_input("Max Emails per Batch", value=automation_settings.value.get('max_emails_per_batch', 50), min_value=1)
                    min_interval = st.number_input("Min Interval Between Cycles (seconds)", value=automation_settings.value.get('min_interval_seconds', 60), min_value=1)
                    max_retries = st.number_input("Max Retries", value=automation_settings.value.get('max_retries', 3), min_value=1)
                    timeout = st.number_input("Request Timeout (seconds)", value=automation_settings.value.get('timeout_seconds', 30), min_value=1)
                    
                    if st.form_submit_button("Save Automation Settings"):
                        automation_settings.value.update({
                            'max_leads_per_term': max_leads,
                            'max_emails_per_batch': max_emails,
                            'min_interval_seconds': min_interval,
                            'max_retries': max_retries,
                            'timeout_seconds': timeout
                        })
                    session.commit()
                        st.success("Automation settings saved successfully!")
                        
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")

def unified_automation_page():
    """Unified automation page for managing search and email campaigns."""
    st.title("Automation Dashboard")
    
    with safe_db_session() as session:
        try:
            # Project and Campaign Selection
            projects = fetch_projects(session)
            if not projects:
                st.warning("No projects found. Please create a project first.")
                return
            
            selected_project = st.selectbox(
                "Select Project",
                options=[(p.id, p.project_name) for p in projects],
                format_func=lambda x: x[1]
            )
            
            campaigns = fetch_campaigns(session, selected_project[0])
            if not campaigns:
                st.warning("No campaigns found for this project. Please create a campaign first.")
            return

            selected_campaign = st.selectbox(
                "Select Campaign",
                options=[(c.id, c.campaign_name) for c in campaigns],
                format_func=lambda x: x[1]
            )
            
            st.session_state['active_campaign_id'] = selected_campaign[0]
            
            # Campaign Status and Controls
            st.header("Campaign Status")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.get('automation_running', False):
                    if st.button("Stop Automation", type="secondary"):
                        handle_automation_stop()
                    else:
                    if st.button("Start Automation", type="primary"):
                        handle_automation_start()

        with col2:
                st.write("Status:", "Running" if st.session_state.get('automation_running', False) else "Stopped")
            
            # Logs and Results
            st.header("Automation Logs")
            log_container = st.empty()
            leads_container = st.empty()
            
            # Start automation if running
            if st.session_state.get('automation_running', False):
                ai_automation_loop(session, log_container, leads_container)
                
    except Exception as e:
            st.error(f"Error in automation page: {str(e)}")

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Email Campaign Automation",
        page_icon="📧",
        layout="wide"
    )
    
    # Initialize settings if needed
    try:
        initialize_settings()
            except Exception as e:
        st.error(f"Failed to initialize settings: {str(e)}")
        return

    # Navigation
        selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Search Terms", "Email Templates", "Leads", "Settings"],
        icons=["house", "search", "envelope", "person", "gear"],
            menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )
    
    # Page routing
    if selected == "Dashboard":
        unified_automation_page()
    elif selected == "Search Terms":
        search_terms_page()
    elif selected == "Email Templates":
        email_templates_page()
    elif selected == "Leads":
        view_leads_page()
    elif selected == "Settings":
        settings_page()

def get_page_title(soup):
    """Extract page title from BeautifulSoup object."""
    try:
        return soup.title.string.strip() if soup.title else ''
    except Exception:
        return ''

def extract_visible_text(soup):
    """Extract visible text content from BeautifulSoup object."""
    try:
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        return ' '.join(chunk for chunk in chunks if chunk)
    except Exception:
        return ''

def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
    """Log the effectiveness of a search term."""
    try:
        effectiveness = SearchTermEffectiveness(
            search_term_id=term.id,
            total_results=total_results,
            valid_leads=valid_leads,
            blogs_found=blogs_found,
            directories_found=directories_found
        )
        session.add(effectiveness)
        session.commit()
    except Exception as e:
        session.rollback()
        st.error(f"Error logging search term effectiveness: {str(e)}")

def add_or_get_search_term(session, term, campaign_id, created_at=None):
    """Add a new search term or get existing one."""
    try:
        existing_term = session.query(SearchTerm).filter_by(term=term, campaign_id=campaign_id).first()
        if existing_term:
            return existing_term
        
        new_term = SearchTerm(
            term=term,
            campaign_id=campaign_id,
            created_at=created_at or datetime.now()
        )
        session.add(new_term)
        session.commit()
        return new_term
    except Exception as e:
        session.rollback()
        st.error(f"Error adding search term: {str(e)}")
        return None

def fetch_campaigns(session, project_id=None):
    """Fetch campaigns, optionally filtered by project."""
    try:
        query = session.query(Campaign)
        if project_id:
            query = query.filter_by(project_id=project_id)
        return query.all()
    except Exception as e:
        st.error(f"Error fetching campaigns: {str(e)}")
        return []

def fetch_projects(session):
    """Fetch all projects."""
    try:
        return session.query(Project).all()
    except Exception as e:
        st.error(f"Error fetching projects: {str(e)}")
        return []

def fetch_email_templates(session):
    """Fetch all email templates."""
    try:
        return session.query(EmailTemplate).all()
    except Exception as e:
        st.error(f"Error fetching email templates: {str(e)}")
        return []

def create_or_update_email_template(session, template_name, subject, body_content, template_id=None, is_ai_customizable=False, created_at=None, language='ES'):
    """Create a new email template or update existing one."""
    try:
        if template_id:
            template = session.query(EmailTemplate).get(template_id)
            if template:
                template.template_name = template_name
                template.subject = subject
                template.body_content = body_content
                template.is_ai_customizable = is_ai_customizable
                template.language = language
        else:
            template = EmailTemplate(
                template_name=template_name,
                subject=subject,
                body_content=body_content,
                is_ai_customizable=is_ai_customizable,
                created_at=created_at or datetime.now(),
                language=language
            )
            session.add(template)
        
        session.commit()
        return template
    except Exception as e:
        session.rollback()
        st.error(f"Error saving email template: {str(e)}")
        return None

def fetch_leads(session, send_option, specific_email, selected_terms, exclude_previously_contacted, campaign_id):
    """Fetch leads based on various filters."""
    try:
        query = session.query(Lead)
        
        if send_option == "Specific Email":
            if specific_email:
                query = query.filter(Lead.email == specific_email)
            else:
                return []
        elif send_option == "Search Term":
            if selected_terms:
                query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.id.in_(selected_terms))
        
        if exclude_previously_contacted:
            contacted_leads = session.query(EmailCampaign.lead_id).filter_by(campaign_id=campaign_id).distinct()
            query = query.filter(~Lead.id.in_(contacted_leads))
        
        return query.all()
    except Exception as e:
        st.error(f"Error fetching leads: {str(e)}")
        return []

def update_display(container, items, title, item_key):
    """Update display container with items."""
    if items:
        container.markdown(f"### {title}")
        for item in items:
            container.markdown(f"- {item[item_key]}")
    else:
        container.markdown(f"### No {title} found")

def get_knowledge_base_info(session, project_id):
    """Get knowledge base information for a project."""
    try:
        kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        return kb.to_dict() if kb else None
    except Exception as e:
        st.error(f"Error fetching knowledge base: {str(e)}")
        return None

def is_valid_email(email):
    """Validate email address."""
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def remove_invalid_leads(session):
    """Remove leads with invalid email addresses."""
    try:
        leads = session.query(Lead).all()
        removed_count = 0
        
        for lead in leads:
            if not is_valid_email(lead.email):
                session.delete(lead)
                removed_count += 1
        
        if removed_count > 0:
            session.commit()
            st.success(f"Removed {removed_count} invalid leads")
        else:
            st.info("No invalid leads found")
    except Exception as e:
        session.rollback()
        st.error(f"Error removing invalid leads: {str(e)}")

def perform_quick_scan(session):
    """Perform a quick scan of the database for issues."""
    try:
        # Check for invalid emails
        invalid_emails = session.query(Lead).filter(
            ~Lead.email.regexp_match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        ).count()
        
        # Check for duplicate emails
        duplicate_emails = session.query(Lead.email).group_by(Lead.email).having(func.count(Lead.id) > 1).count()
        
        # Check for orphaned records
        orphaned_sources = session.query(LeadSource).filter(
            ~LeadSource.lead_id.in_(session.query(Lead.id))
        ).count()
        
        return {
            'invalid_emails': invalid_emails,
            'duplicate_emails': duplicate_emails,
            'orphaned_sources': orphaned_sources
        }
    except Exception as e:
        st.error(f"Error performing quick scan: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")

# ... existing code ...