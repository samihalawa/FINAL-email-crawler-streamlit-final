import os, json, re, logging, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid, urllib3, random, smtplib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup, SoupStrainer
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select, text, distinct, and_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from typing import List, Optional, Dict
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
import plotly.express as px
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import traceback
from logging.handlers import RotatingFileHandler
#database info
load_dotenv()

# Database configuration
DB_HOST = os.environ.get("SUPABASE_DB_HOST")
DB_NAME = os.environ.get("SUPABASE_DB_NAME")
DB_USER = os.environ.get("SUPABASE_DB_USER")
DB_PASSWORD = os.environ.get("SUPABASE_DB_PASSWORD")
DB_PORT = os.environ.get("SUPABASE_DB_PORT")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={
        'connect_timeout': 10,
        'keepalives': 1,
        'keepalives_idle': 30,
        'keepalives_interval': 10,
        'keepalives_count': 5
    }
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")


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
        """Convert project to dictionary format."""
        return {
            'id': self.id,
            'project_name': self.project_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'campaign_count': len(self.campaigns) if self.campaigns else 0,
            'has_knowledge_base': bool(self.knowledge_base)
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
        return {attr: getattr(self, attr) for attr in ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']}

# Update the Lead model to remove the domain attribute
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
    daily_limit = Column(BigInteger, default=999999999)
    hourly_limit = Column(BigInteger, default=999999999)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

@contextmanager
def db_session():
    """Optimized database session context manager with proper error handling and connection pooling"""
    session = None
    try:
        session = SessionLocal()
        yield session
        session.commit()
    except SQLAlchemyError as e:
        if session:
            session.rollback()
        error_msg = str(e)
        logging.error(f"Database error: {error_msg}")
        if "connection" in error_msg.lower():
            st.error("Database connection error. Please try again.")
        elif "deadlock" in error_msg.lower():
            st.error("Database conflict detected. Please retry your operation.")
        else:
            st.error("A database error occurred. Please try again.")
        raise
    except Exception as e:
        if session:
            session.rollback()
        logging.error(f"Unexpected error in database session: {str(e)}")
        st.error("An unexpected error occurred. Please try again.")
        raise
    finally:
        if session:
            try:
                session.close()
            except Exception as e:
                logging.error(f"Error closing database session: {str(e)}")

@contextmanager
def safe_db_session():
    """Enhanced database session with retry logic for transient errors"""
    max_retries = 3
    retry_delay = 1
    last_error = None
    
    for attempt in range(max_retries):
        try:
            with db_session() as session:
                yield session
                return
        except SQLAlchemyError as e:
            last_error = e
            if "connection" in str(e).lower() and attempt < max_retries - 1:
                logging.warning(f"Database connection error, attempt {attempt + 1} of {max_retries}")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            raise last_error
        except Exception as e:
            logging.error(f"Critical database error: {str(e)}")
            raise

def check_required_settings(session):
    """Check all required settings and environment variables at startup."""
    # Check database connection first
    try:
        session.execute(text('SELECT 1'))
    except Exception as e:
        raise ValueError(f"Database connection failed: {str(e)}")
    
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
                    'database_url': os.getenv('DATABASE_URL', ''),
                    'max_connections': 20,
                    'pool_size': 5
                }
                session.add(Settings(name='Database Settings', setting_type='database', value=default_db_settings))
                session.commit()
            
            # Create default AI settings if they don't exist
            ai_settings = session.query(Settings).filter_by(setting_type='ai').first()
            if not ai_settings:
                default_ai_settings = {
                    'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
                    'api_base_url': 'https://api.openai.com/v1',
                    'model_name': 'gpt-4',
                    'max_tokens': 1500,
                    'temperature': 0.7
                }
                session.add(Settings(name='AI Settings', setting_type='ai', value=default_ai_settings))
                session.commit()
            
            # Create default automation settings if they don't exist
            automation_settings = session.query(Settings).filter_by(setting_type='automation').first()
            if not automation_settings:
                default_automation_settings = {
                    'max_runtime_hours': 24,
                    'cycle_interval_seconds': 3600,
                    'error_retry_seconds': 300,
                    'results_per_search': 10,
                    'max_leads_per_cycle': 500
                }
                session.add(Settings(name='Automation Settings', setting_type='automation', value=default_automation_settings))
                session.commit()
            
            # Create default email settings if they don't exist
            email_defaults_settings = session.query(Settings).filter_by(setting_type='email_defaults').first()
            if not email_defaults_settings:
                default_email_defaults_settings = {
                    'default_from_email': '',
                    'default_reply_to': '',
                    'email_batch_size': 100
                }
                session.add(Settings(name='Email Defaults', setting_type='email_defaults', value=default_email_defaults_settings))
                session.commit()
            
            # Check required settings
            check_required_settings(session)
            
            # Verify database connection
            session.execute(text("SELECT 1"))
            session.commit()
            
        except Exception as e:
            logging.error(f"Failed to initialize settings: {str(e)}")
            raise

def safe_button_operation(func):
    """Decorator to prevent multiple clicks and handle errors"""
    def wrapper(*args, **kwargs):
        button_key = kwargs.get('key', 'default_button')
        if f'processing_{button_key}' in st.session_state:
            st.warning('Operation in progress, please wait...')
            return
            
        try:
            st.session_state[f'processing_{button_key}'] = True
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            st.error(f"Operation failed: {str(e)}")
            logging.exception("Button operation error")
        finally:
            if f'processing_{button_key}' in st.session_state:
                del st.session_state[f'processing_{button_key}']
    return wrapper

def settings_page():
    st.title("Settings")
    with db_session() as session:
        try:
            # Email Settings
            st.subheader("Email Settings")
            
            # Fetch existing settings
            email_settings = session.query(EmailSettings).all()
            
            # Display existing settings
            if email_settings:
                settings_data = []
                for setting in email_settings:
                    settings_data.append({
                        'ID': setting.id,
                        'Name': setting.name,
                        'Email': setting.email,
                        'Provider': setting.provider,
                        'Daily Limit': setting.daily_limit or 'No limit',
                        'Hourly Limit': setting.hourly_limit or 'No limit',
                        'Active': '✓' if setting.is_active else '✗'
                    })
                
                df = pd.DataFrame(settings_data)
                st.dataframe(
                    df,
                    hide_index=True,
                    column_config={
                        'ID': st.column_config.NumberColumn('ID'),
                        'Name': st.column_config.TextColumn('Name'),
                        'Email': st.column_config.TextColumn('Email'),
                        'Provider': st.column_config.TextColumn('Provider'),
                        'Daily Limit': st.column_config.TextColumn('Daily Limit'),
                        'Hourly Limit': st.column_config.TextColumn('Hourly Limit'),
                        'Active': st.column_config.TextColumn('Active')
                    }
                )

            # Add new email setting
            st.subheader("Add Email Setting")
            with st.form("email_setting_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Setting Name")
                    email = st.text_input("Email Address")
                    provider = st.selectbox("Provider", ["AWS SES", "SMTP"])
                    
                with col2:
                    daily_limit = st.number_input("Daily Email Limit", min_value=0, value=1000)
                    hourly_limit = st.number_input("Hourly Email Limit", min_value=0, value=100)
                    is_active = st.checkbox("Active", value=True)
                
                # Provider-specific settings
                if provider == "AWS SES":
                    aws_access_key = st.text_input("AWS Access Key ID")
                    aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
                    aws_region = st.text_input("AWS Region", value="us-east-1")
                else:
                    smtp_server = st.text_input("SMTP Server")
                    smtp_port = st.number_input("SMTP Port", value=587)
                    smtp_username = st.text_input("SMTP Username")
                    smtp_password = st.text_input("SMTP Password", type="password")

                if st.form_submit_button("Add Email Setting"):
                    try:
                        new_setting = EmailSettings(
                            name=name,
                            email=email,
                            provider=provider,
                            daily_limit=daily_limit,
                            hourly_limit=hourly_limit,
                            is_active=is_active,
                            project_id=get_active_project_id()
                        )
                        
                        if provider == "AWS SES":
                            new_setting.aws_access_key_id = aws_access_key
                            new_setting.aws_secret_access_key = aws_secret_key
                            new_setting.aws_region = aws_region
                        else:
                            new_setting.smtp_server = smtp_server
                            new_setting.smtp_port = smtp_port
                            new_setting.smtp_username = smtp_username
                            new_setting.smtp_password = smtp_password
                        
                        session.add(new_setting)
                        session.commit()
                        st.success("Email setting added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding email setting: {str(e)}")
                        session.rollback()

            # Application Settings
            st.subheader("Application Settings")
            
            # Fetch current settings
            app_settings = session.query(Settings).filter_by(setting_type='application').first()
            current_settings = app_settings.value if app_settings else {}
            
            # General settings
            with st.form("app_settings_form"):
                enable_ai = st.checkbox(
                    "Enable AI Features",
                    value=current_settings.get('enable_ai', True)
                )
                
                debug_mode = st.checkbox(
                    "Debug Mode",
                    value=current_settings.get('debug_mode', False)
                )

                default_language = st.selectbox(
                    "Default Language",
                    options=["ES", "EN"],
                    index=0 if current_settings.get('default_language', 'ES') == 'ES' else 1
                )
                
                max_search_results = st.number_input(
                    "Max Search Results per Term",
                    min_value=1,
                    value=current_settings.get('max_search_results', 50)
                )
                
                email_batch_size = st.number_input(
                    "Email Batch Size",
                    min_value=1,
                    value=current_settings.get('email_batch_size', 10)
                )
                
                if st.form_submit_button("Save Settings"):
                    try:
                        new_settings = {
                            'enable_ai': enable_ai,
                            'debug_mode': debug_mode,
                            'default_language': default_language,
                            'max_search_results': max_search_results,
                            'email_batch_size': email_batch_size
                        }
                        
                        if app_settings:
                            app_settings.value = new_settings
                        else:
                            app_settings = Settings(
                                name='application_settings',
                                setting_type='application',
                                value=new_settings
                            )
                            session.add(app_settings)
                        
                        session.commit()
                        st.success("Application settings saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving application settings: {str(e)}")

            # AI Settings
            st.subheader("AI Settings")
            ai_settings = session.query(Settings).filter_by(setting_type='ai').first()
            current_ai_settings = ai_settings.value if ai_settings else {}

            with st.form("ai_settings_form"):
                openai_api_key = st.text_input(
                    "OpenAI API Key",
                    value=current_ai_settings.get('openai_api_key', ''),
                    type="password"
                )

                openai_api_base = st.text_input(
                    "OpenAI API Base URL",
                    value=current_ai_settings.get('api_base_url', 'https://api.openai.com/v1')
                )

                openai_model = st.selectbox(
                    "OpenAI Model",
                    options=["gpt-4", "gpt-3.5-turbo"],
                    index=0 if current_ai_settings.get('model_name', 'gpt-4') == 'gpt-4' else 1
                )

                max_tokens = st.number_input(
                    "Max Tokens per Request",
                    min_value=100,
                    max_value=4000,
                    value=int(current_ai_settings.get('max_tokens', 1500))
                )

                if st.form_submit_button("Save AI Settings"):
                    try:
                        new_ai_settings = {
                            'openai_api_key': openai_api_key,
                            'api_base_url': openai_api_base,
                            'model_name': openai_model,
                            'max_tokens': max_tokens
                        }

                        if ai_settings:
                            ai_settings.value = new_ai_settings
                        else:
                            ai_settings = Settings(
                                name='ai_settings',
                                setting_type='ai',
                                value=new_ai_settings
                            )
                            session.add(ai_settings)

                        session.commit()
                        st.success("AI settings saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving AI settings: {str(e)}")

            # Database Information
            st.subheader("Database Information")
            
            # Get table statistics
            stats = {
                'Projects': session.query(Project).count(),
                'Campaigns': session.query(Campaign).count(),
                'Leads': session.query(Lead).count(),
                'Search Terms': session.query(SearchTerm).count(),
                'Email Templates': session.query(EmailTemplate).count(),
                'Email Campaigns': session.query(EmailCampaign).count()
            }
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            for i, (table, count) in enumerate(stats.items()):
                with [col1, col2, col3][i % 3]:
                    st.metric(table, count)
            
            # Database maintenance
            st.subheader("Database Maintenance")
            
            if st.button("Check Database Health"):
                try:
                    check_database_state()
                    st.success("Database health check completed successfully!")
                except Exception as e:
                    st.error(f"Database health check failed: {str(e)}")
            
            if st.button("Create Default Settings"):
                try:
                    initialize_settings()
                    st.success("Default settings created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating default settings: {str(e)}")

        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")

def get_random_user_agent():
    """Get a random user agent to avoid blocking"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59'
    ]
    return random.choice(user_agents)

def should_skip_domain(domain):
    """Check if domain should be skipped"""
    skip_domains = {
        'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
        'youtube.com', 'github.com', 'medium.com', 'reddit.com'
    }
    return domain in skip_domains

def save_lead(session, url, search_term):
    """Save lead with improved extraction"""
    try:
        # Get page content with retries
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(url, timeout=10, verify=False, headers=headers)
        response.raise_for_status()
        
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract information
        emails = extract_emails_from_html(html_content)
        name, _, job_title = extract_info_from_page(soup)
        company = extract_company_name(soup, url)
        
        # Track processed emails to avoid duplicates
        processed_emails = set()
        leads = []
        
        # Get or create search term object if string was passed
        search_term_obj = None
        if isinstance(search_term, str):
            search_term_obj = session.query(SearchTerm).filter_by(term=search_term).first()
            if not search_term_obj:
                search_term_obj = SearchTerm(
                    term=search_term,
                    campaign_id=get_active_campaign_id()
                )
                session.add(search_term_obj)
                session.flush()
        else:
            search_term_obj = search_term
        
        for email in emails:
            if email in processed_emails:
                continue
                
            if is_valid_email(email) and is_valid_contact_email(email):
                processed_emails.add(email)
                
                # Check if lead already exists
                lead = session.query(Lead).filter_by(email=email).first()
                if not lead:
                    lead = Lead(
                        email=email,
                        first_name=name,
                        company=company,
                        job_title=job_title
                    )
                    session.add(lead)
                    session.flush()
                
                # Save lead source
                save_lead_source(
                    session,
                    lead_id=lead.id,
                    search_term_id=search_term_obj.id if search_term_obj else None,
                    url=url,
                    http_status=response.status_code,
                    scrape_duration=str(response.elapsed.total_seconds()),
                    page_title=soup.title.string if soup.title else None,
                    meta_description=soup.find('meta', {'name': 'description'}).get('content') if soup.find('meta', {'name': 'description'}) else None,
                    content=str(soup)[:1000],
                    tags=None,
                    phone_numbers=None
                )
                leads.append(lead)
        
        session.commit()
        return leads[0] if leads else None
        
    except requests.exceptions.RequestException as e:
        if '403' in str(e):
            logging.warning(f"Access forbidden for {url} - site may be blocking scraping")
        elif '429' in str(e):
            logging.warning(f"Rate limited by {url} - too many requests")
        else:
            logging.error(f"Error saving lead from {url}: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error saving lead from {url}: {str(e)}")
        return None

def send_email_ses(session, from_email, to_email, subject, body, reply_to=None):
    """Send email with better error handling"""
    try:
        # Get email settings
        settings = session.query(EmailSettings).first()
        if not settings or not settings.provider or settings.provider.lower() != 'ses':
            raise ValueError("SES email settings not configured")
            
        # Configure SES client
        ses_client = boto3.client(
            'ses',
            aws_access_key_id=settings.aws_access_key_id,  # Fixed field name
            aws_secret_access_key=settings.aws_secret_access_key,  # Fixed field name
            region_name=settings.aws_region or 'eu-west-1'
        )
        
        # Prepare email
        email_data = {
            'Source': from_email,
            'Destination': {'ToAddresses': [to_email]},
            'Message': {
                'Subject': {'Data': subject},
                'Body': {'Text': {'Data': body}}
            }
        }
        if reply_to:
            email_data['ReplyToAddresses'] = [reply_to]
            
        # Send email
        response = ses_client.send_email(**email_data)
        return response, response.get('MessageId')
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'MessageRejected':
            logging.error(f"Email rejected: {str(e)}")
        elif error_code == 'InvalidParameterValue':
            logging.error(f"Invalid email parameter: {str(e)}")
        else:
            logging.error(f"SES error: {str(e)}")
        return None, None
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        return None, None

def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    try:
        lead = session.query(Lead).filter_by(email=lead_email).first()
        if not lead:
            return None

        template = session.query(EmailTemplate).get(template_id)
        if not template:
            return None

        campaign = session.query(Campaign).get(template.campaign_id)
        if not campaign:
            return None

        email_campaign = EmailCampaign(
            campaign_id=campaign.id,
            lead_id=lead.id,
            template_id=template_id,
            status=status,
            sent_at=sent_at,
            original_subject=subject,
            original_content=email_body,
            message_id=message_id,
            tracking_id=str(uuid.uuid4())
        )
        
        session.add(email_campaign)
        return email_campaign
    except Exception as e:
        logging.error(f"Error saving email campaign: {str(e)}")
        return None

def update_log(log_container, message, level='info'):
    icon = {'info': '🔵', 'success': '🟢', 'warning': '🟠', 'error': '🔴', 'email_sent': '🟣'}.get(level, '⚪')
    log_entry = f"{icon} {message}"
    
    # Simple console logging without HTML
    print(f"{icon} {message.split('<')[0]}")  # Only print the first part of the message before any HTML tags
    
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    
    # HTML-formatted log entry for Streamlit display
    html_log_entry = f"{icon} {message}"
    st.session_state.log_entries.append(html_log_entry)
    
    # Update the Streamlit display with all logs
    log_html = f"<div style='height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.8em; line-height: 1.2;'>{'<br>'.join(st.session_state.log_entries)}</div>"
    log_container.markdown(log_html, unsafe_allow_html=True)

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

def manual_search(session, terms, num_results, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False, language='ES', enable_email_sending=True, log_container=None, from_email=None, reply_to=None, email_template=None):
    """Perform manual search for leads using provided terms"""
    if not terms:
        if log_container:
            update_log(log_container, "No search terms provided", level='error')
        return
        
    search_params = {
        'stop': num_results,
        'lang': language.lower(),
        'pause': 2.0  # Add pause between searches to avoid rate limiting
    }
    
    for search_term in terms:
        if not search_term:
            continue
            
        if optimize_english and language == 'ES':
            search_term = optimize_search_term(search_term, 'EN')
        elif optimize_spanish and language == 'EN':
            search_term = optimize_search_term(search_term, 'ES')
            
        if shuffle_keywords_option:
            search_term = shuffle_keywords(search_term)
            
        if log_container:
            update_log(log_container, f"Searching for: {search_term}")
            
        try:
            search_results = list(google_search(search_term, **search_params))
            if not search_results:
                if log_container:
                    update_log(log_container, f"No results found for: {search_term}", level='warning')
                continue
                
            for url in search_results:
                if not url:
                    continue
                    
                if ignore_previously_fetched:
                    domain = get_domain_from_url(url)
                    if not domain or should_skip_domain(domain):
                        if log_container:
                            update_log(log_container, f"Skipping {domain}: Previously fetched or invalid", level='warning')
                        continue
                    
                try:
                    lead = save_lead(session, url=url, search_term=search_term)
                    if not lead:
                        if log_container:
                            update_log(log_container, f"Failed to save lead for {url}", level='warning')
                        continue
                        
                    if lead.email and enable_email_sending and email_template:
                        try:
                            response, _ = send_email_ses(
                                session,
                                from_email,
                                lead.email,
                                email_template.subject,
                                email_template.body_content,
                                reply_to=reply_to
                            )
                            if log_container:
                                update_log(log_container, f"Email sent to: {lead.email}", level='success')
                        except Exception as e:
                            if log_container:
                                update_log(log_container, f"Email error ({lead.email}): {str(e)}", level='error')
                except Exception as e:
                    if log_container:
                        update_log(log_container, f"Error processing {url}: {str(e)}", level='error')
                    continue
                    
        except Exception as e:
            if log_container:
                update_log(log_container, f"Search error for term '{search_term}': {str(e)}", level='error')
            continue

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
        # Get IDs of selected terms
        selected_term_ids = [int(term.split(':')[0]) for term in updated_terms]
        
        # Update all terms that should be in this group
        session.query(SearchTerm)\
            .filter(SearchTerm.id.in_(selected_term_ids))\
            .update({SearchTerm.group_id: group_id}, synchronize_session=False)
        
        # Remove group_id from terms that were unselected
        session.query(SearchTerm)\
            .filter(SearchTerm.group_id == group_id)\
            .filter(~SearchTerm.id.in_(selected_term_ids))\
            .update({SearchTerm.group_id: None}, synchronize_session=False)
        
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_search_term_group: {str(e)}")
        raise

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        group_id = None
        if group_for_new_term != "None":
            group_id = int(group_for_new_term.split(':')[0])
            
        new_search_term = SearchTerm(
            term=new_term,
            campaign_id=campaign_id,
            group_id=group_id,
            created_at=datetime.utcnow()
        )
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")
        raise

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
    # Get automation settings
    automation_settings = session.query(Settings).filter_by(setting_type='automation').first()
    if not automation_settings:
        log_container.error("Automation settings not found. Please check your configuration.")
        return
        
    # Get email settings
    email_settings = session.query(EmailSettings).first()
    if not email_settings:
        log_container.error("Email settings not found. Please check your configuration.")
        return

    # Initialize tracking variables
    automation_logs, total_search_terms, total_emails_sent = [], 0, 0
    start_time = time.time()
    max_runtime = automation_settings.value.get('max_runtime_hours', 24) * 3600
    cycle_interval = automation_settings.value.get('cycle_interval_seconds', 3600)
    error_retry_interval = automation_settings.value.get('error_retry_seconds', 300)
    results_per_search = automation_settings.value.get('results_per_search', 10)
    max_leads_per_cycle = automation_settings.value.get('max_leads_per_cycle', 500)
    
    while st.session_state.get('automation_status', False):
        current_time = time.time()
        if current_time - start_time > max_runtime:
            log_container.warning("Maximum runtime reached. Stopping automation.")
            st.session_state.automation_status = False
            break
            
        try:
            log_container.info("Starting automation cycle")
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                log_container.warning("Knowledge Base not found. Skipping cycle.")
                time.sleep(cycle_interval)
                continue
                
            base_terms = [term.term for term in session.query(SearchTerm).filter_by(campaign_id=get_active_campaign_id()).all()]
            optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            st.subheader("Optimized Search Terms")
            st.write(", ".join(optimized_terms))

            total_search_terms = len(optimized_terms)
            progress_bar = st.progress(0)
            cycle_leads = 0
            
            for idx, term in enumerate(optimized_terms):
                if cycle_leads >= max_leads_per_cycle:
                    log_container.info(f"Reached maximum leads per cycle ({max_leads_per_cycle}). Moving to next cycle.")
                    break
                    
                results = manual_search(session, [term], results_per_search, ignore_previously_fetched=True, log_container=log_container)
                new_leads = []
                for res in results['results']:
                    lead = save_lead(session, res['Email'], url=res['URL'])
                    if lead:
                        new_leads.append((lead.id, lead.email))
                        cycle_leads += 1
                        
                if new_leads:
                    template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                    if template:
                        from_email = kb_info.get('contact_email') or email_settings.value.get('default_from_email')
                        reply_to = kb_info.get('contact_email') or email_settings.value.get('default_reply_to')
                        
                        if not from_email or not reply_to:
                            log_container.error("Missing email configuration")
                            continue
                            
                        logs, sent_count = bulk_send_emails(
                            session, 
                            template.id, 
                            from_email, 
                            reply_to, 
                            [{'Email': email} for _, email in new_leads],
                            batch_size=email_settings.value.get('email_batch_size', 100),
                            log_container=log_container
                        )
                        automation_logs.extend(logs)
                        total_emails_sent += sent_count
                        
                leads_container.text_area("New Leads Found", "\n".join([email for _, email in new_leads]), height=200)
                progress_bar.progress((idx + 1) / len(optimized_terms))
                
            st.success(f"Automation cycle completed. Total search terms: {total_search_terms}, Total emails sent: {total_emails_sent}")
            time.sleep(cycle_interval)
            
        except Exception as e:
            log_container.error(f"Critical error in automation cycle: {str(e)}")
            time.sleep(error_retry_interval)
            
    log_container.info("Automation stopped")
    st.session_state.automation_logs = automation_logs
    st.session_state.total_leads_found = total_search_terms
    st.session_state.total_emails_sent = total_emails_sent

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        # First try to get AI settings
        ai_settings = session.query(Settings).filter_by(setting_type='ai').first()
        
        # If AI settings don't exist, try to get general settings as fallback
        if not ai_settings:
            general_settings = session.query(Settings).filter_by(setting_type='general').first()
            if general_settings and 'openai_api_key' in general_settings.value:
                # Convert general settings to AI settings format
                ai_settings = Settings(
                    name='ai_settings',
                    setting_type='ai',
                    value={
                        'openai_api_key': general_settings.value['openai_api_key'],
                        'api_base_url': general_settings.value.get('openai_api_base', 'https://api.openai.com/v1'),
                        'model_name': general_settings.value.get('openai_model', 'gpt-4'),
                        'max_tokens': 1500
                    }
                )
                session.add(ai_settings)
                session.commit()
        
        if not ai_settings or 'openai_api_key' not in ai_settings.value:
            st.error("OpenAI API key not set. Please configure it in the AI settings.")
            return None

        client = OpenAI(
            api_key=ai_settings.value['openai_api_key'],
            base_url=ai_settings.value.get('api_base_url', 'https://api.openai.com/v1')
        )
        model = ai_settings.value.get('model_name', 'gpt-4')
        max_tokens = int(ai_settings.value.get('max_tokens', 1500))
        
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
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
    """Save lead source with error handling"""
    try:
        lead_source = LeadSource(
            lead_id=lead_id,
            search_term_id=search_term_id if search_term_id is not None else None,
            url=url,
            http_status=http_status,
            scrape_duration=scrape_duration,
            page_title=page_title or get_page_title(url),
            meta_description=meta_description or get_page_description(url),
            content=content or extract_visible_text(BeautifulSoup(requests.get(url).text, 'html.parser')),
            tags=tags,
            phone_numbers=phone_numbers
        )
        session.add(lead_source)
        session.commit()
    except Exception as e:
        logging.error(f"Error saving lead source: {str(e)}")
        session.rollback()

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

def fetch_campaigns(session, project_id=None):
    if project_id:
        return [f"{c.id}: {c.campaign_name}" for c in session.query(Campaign).filter_by(project_id=project_id).all()]
    else:
        return [f"{c.id}: {c.campaign_name}" for c in session.query(Campaign).all()]

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

def fetch_leads(session, send_option, specific_email, selected_terms, exclude_previously_contacted, campaign_id):
    try:
        query = session.query(Lead)
        if send_option == "Specific Email":
            query = query.filter(Lead.email == specific_email)
        elif send_option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and selected_terms:
            # Join through LeadSource to get leads from specific search terms
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))
        
        if exclude_previously_contacted:
            # Use a subquery to exclude previously contacted leads
            contacted_leads = session.query(EmailCampaign.lead_id)\
                .filter(EmailCampaign.sent_at.isnot(None))\
                .filter(EmailCampaign.campaign_id == campaign_id)\
                .subquery()
            query = query.outerjoin(contacted_leads, Lead.id == contacted_leads.c.lead_id)\
                .filter(contacted_leads.c.lead_id.is_(None))
        
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
    st.title("Manual Search")
    
    # Create main log container
    log_container = st.empty()
    
    # Initialize session state for logs if not exists
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    
    def update_logs():
        # Format all logs with colors and emojis
        formatted_logs = []
        for log in st.session_state.log_entries[-100:]:  # Keep last 100 logs
            level = log.get('level', 'info')
            msg = log.get('message', '')
            timestamp = log.get('timestamp', datetime.now().strftime('%H:%M:%S'))
            
            # Color coding and icons
            icon = {
                'info': '🔵',
                'success': '🟢',
                'warning': '🟠',
                'error': '🔴',
                'email_sent': '🟣'
            }.get(level, '⚪')
            
            # Add log entry with monospace font and proper spacing
            formatted_logs.append(
                f'<div style="font-family: monospace; padding: 2px 0;">{icon} [{timestamp}] {msg}</div>'
            )
        
        # Create scrollable container with auto-scroll
        log_html = f"""
        <div style='
            background: rgba(0,0,0,0.05);
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        '>
            <div id="log-container" style='
                height: 300px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 0.9em;
                line-height: 1.4;
                padding: 10px;
                background: rgba(255,255,255,0.8);
                border-radius: 3px;
            '>
                {''.join(formatted_logs)}
            </div>
        </div>
        <script>
            function scrollToBottom() {{
                var container = document.getElementById('log-container');
                if (container) {{
                    container.scrollTop = container.scrollHeight;
                }}
            }}
            // Initial scroll
            scrollToBottom();
            // Set up a mutation observer to watch for changes
            var observer = new MutationObserver(scrollToBottom);
            var container = document.getElementById('log-container');
            if (container) {{
                observer.observe(container, {{ childList: true, subtree: true }});
            }}
        </script>
        """
        log_container.markdown(log_html, unsafe_allow_html=True)
    
    def custom_log_update(message, level='info'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        st.session_state.log_entries.append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })
        update_logs()
    
    # Initialize session state
    if 'domains_processed' not in st.session_state:
        st.session_state.domains_processed = set()
    
    # Reset domains button
    if st.button("Reset Processed Domains"):
        st.session_state.domains_processed = set()
        st.success("Domain list reset successfully!")
    
    with db_session() as session:
        recent_searches = session.query(SearchTerm).order_by(SearchTerm.created_at.desc()).limit(5).all()
        recent_search_terms = [term.term for term in recent_searches]
        
        email_templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)

    col1, col2 = st.columns([2, 1])
    
    # Initialize email variables
    from_email = None
    reply_to = None
    email_template = None

    with col1:
        # Store current terms in session state if not exists
        if 'current_search_terms' not in st.session_state:
            st.session_state.current_search_terms = recent_search_terms

        search_terms = st_tags(
            label='Enter search terms:',
            text='Press enter to add more',
            value=st.session_state.current_search_terms,
            suggestions=['software engineer', 'data scientist', 'product manager'],
            maxtags=50,
            key='search_terms_input'
        )

        col_gen, col_clear = st.columns([1, 1])
        with col_gen:
            if st.button("🤖 Generate 10 More with AI", use_container_width=True):
                with st.spinner("Generating search terms with AI..."):
                    with db_session() as session:
                        kb_info = get_knowledge_base_info(session, get_active_project_id())
                        if not kb_info:
                            st.warning("Please set up your Knowledge Base first to use AI generation.")
                            return
                        
                        new_terms = generate_optimized_search_terms(session, search_terms, kb_info)
                        if new_terms:
                            # Combine existing and new terms, remove duplicates
                            combined_terms = list(set(search_terms + new_terms[:10]))
                            st.session_state.current_search_terms = combined_terms
                            st.success(f"Added {len(new_terms)} new search terms!")
                            st.rerun()
                        else:
                            st.error("Failed to generate new terms. Please try again.")
        
        with col_clear:
            if st.button("🗑️ Clear All Terms", use_container_width=True):
                st.session_state.current_search_terms = []
                st.rerun()

        num_results = st.slider("Results per term", 1, 50000, 10)

    with col2:
        enable_email_sending = st.checkbox("Enable email sending", value=True)
        ignore_previously_fetched = st.checkbox("Ignore fetched domains", value=True)
        shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=True)
        optimize_english = st.checkbox("Optimize (English)", value=False)
        optimize_spanish = st.checkbox("Optimize (Spanish)", value=False)
        language = st.selectbox("Select Language", options=["ES", "EN"], index=0)

    if enable_email_sending:
        if not email_templates:
            st.error("No email templates available. Please create a template first.")
            return
        if not email_settings:
            st.error("No email settings available. Please add email settings first.")
            return

        col3, col4 = st.columns(2)
        with col3:
            email_template = st.selectbox("Email template", options=email_templates, format_func=lambda x: x.split(":")[1].strip())
        with col4:
            email_setting_option = st.selectbox("From Email", options=email_settings, format_func=lambda x: f"{x['name']} ({x['email']})")
            if email_setting_option:
                from_email = email_setting_option['email']
                reply_to = st.text_input("Reply To", email_setting_option['email'])
            else:
                st.error("No email setting selected. Please select an email setting.")
                return

    if st.button("Search"):
        if not search_terms:
            return st.warning("Enter at least one search term.")

        progress_bar = st.progress(0)
        status_text = st.empty()
        email_status = st.empty()
        results = []

        leads_container = st.empty()
        leads_found, emails_sent = [], []

        log_container = st.empty()

        for i, term in enumerate(search_terms):
            status_text.text(f"Searching: '{term}' ({i+1}/{len(search_terms)})")

            with db_session() as session:
                term_results = manual_search(
                    session, [term], num_results, 
                    ignore_previously_fetched, 
                    optimize_english, 
                    optimize_spanish, 
                    shuffle_keywords_option, 
                    language, 
                    enable_email_sending, 
                    log_container, 
                    from_email, 
                    reply_to, 
                    email_template
                )
                results.extend(term_results['results'])

                leads_found.extend([f"{res['Email']} - {res['Company']}" for res in term_results['results']])

                if enable_email_sending:
                    template = session.query(EmailTemplate).filter_by(id=int(email_template.split(":")[0])).first()
                    for result in term_results['results']:
                        if not result or 'Email' not in result or not is_valid_email(result['Email']):
                            status_text.text(f"Skipping invalid result or email: {result.get('Email') if result else 'None'}")
                            continue
                        wrapped_content = wrap_email_body(template.body_content)
                        response, tracking_id = send_email_ses(session, from_email, result['Email'], template.subject, wrapped_content, reply_to=reply_to)
                        if response:
                            save_email_campaign(session, result['Email'], template.id, 'sent', datetime.utcnow(), template.subject, response.get('MessageId', 'Unknown'), wrapped_content)
                            emails_sent.append(f"✅ {result['Email']}")
                            status_text.text(f"Email sent to: {result['Email']}")
                        else:
                            save_email_campaign(session, result['Email'], template.id, 'failed', datetime.utcnow(), template.subject, None, wrapped_content)
                            emails_sent.append(f"❌ {result['Email']}")
                            status_text.text(f"Failed to send email to: {result['Email']}")

            leads_container.dataframe(pd.DataFrame({"Leads Found": leads_found, "Emails Sent": emails_sent + [""] * (len(leads_found) - len(emails_sent))}))
            progress_bar.progress((i + 1) / len(search_terms))

        # Display final results
        st.subheader("Search Results")
        st.dataframe(pd.DataFrame(results))

        if enable_email_sending:
            st.subheader("Email Sending Results")
            success_rate = sum(1 for email in emails_sent if email.startswith("✅")) / len(emails_sent) if emails_sent else 0
            st.metric("Email Sending Success Rate", f"{success_rate:.2%}")

        st.download_button(
            label="Download CSV",
            data=pd.DataFrame(results).to_csv(index=False).encode('utf-8'),
            file_name="search_results.csv",
            mime="text/csv",
        )

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

def ai_automation_loop(session, log_container, leads_container):
    # Get automation settings
    automation_settings = session.query(Settings).filter_by(setting_type='automation').first()
    if not automation_settings:
        log_container.error("Automation settings not found. Please check your configuration.")
        return
        
    # Get email settings
    email_settings = session.query(EmailSettings).first()
    if not email_settings:
        log_container.error("Email settings not found. Please check your configuration.")
        return

    # Initialize tracking variables
    automation_logs, total_search_terms, total_emails_sent = [], 0, 0
    start_time = time.time()
    max_runtime = automation_settings.value.get('max_runtime_hours', 24) * 3600
    cycle_interval = automation_settings.value.get('cycle_interval_seconds', 3600)
    error_retry_interval = automation_settings.value.get('error_retry_seconds', 300)
    results_per_search = automation_settings.value.get('results_per_search', 10)
    max_leads_per_cycle = automation_settings.value.get('max_leads_per_cycle', 500)
    
    while st.session_state.get('automation_status', False):
        current_time = time.time()
        if current_time - start_time > max_runtime:
            log_container.warning("Maximum runtime reached. Stopping automation.")
            st.session_state.automation_status = False
            break
            
        try:
            log_container.info("Starting automation cycle")
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                log_container.warning("Knowledge Base not found. Skipping cycle.")
                time.sleep(cycle_interval)
                continue
                
            base_terms = [term.term for term in session.query(SearchTerm).filter_by(campaign_id=get_active_campaign_id()).all()]
            optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            st.subheader("Optimized Search Terms")
            st.write(", ".join(optimized_terms))

            total_search_terms = len(optimized_terms)
            progress_bar = st.progress(0)
            cycle_leads = 0
            
            for idx, term in enumerate(optimized_terms):
                if cycle_leads >= max_leads_per_cycle:
                    log_container.info(f"Reached maximum leads per cycle ({max_leads_per_cycle}). Moving to next cycle.")
                    break
                    
                results = manual_search(session, [term], results_per_search, ignore_previously_fetched=True, log_container=log_container)
                new_leads = []
                for res in results['results']:
                    lead = save_lead(session, res['Email'], url=res['URL'])
                    if lead:
                        new_leads.append((lead.id, lead.email))
                        cycle_leads += 1
                        
                if new_leads:
                    template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                    if template:
                        from_email = kb_info.get('contact_email') or email_settings.value.get('default_from_email')
                        reply_to = kb_info.get('contact_email') or email_settings.value.get('default_reply_to')
                        
                        if not from_email or not reply_to:
                            log_container.error("Missing email configuration")
                            continue
                            
                        logs, sent_count = bulk_send_emails(
                            session, 
                            template.id, 
                            from_email, 
                            reply_to, 
                            [{'Email': email} for _, email in new_leads],
                            batch_size=email_settings.value.get('email_batch_size', 100),
                            log_container=log_container
                        )
                        automation_logs.extend(logs)
                        total_emails_sent += sent_count
                        
                leads_container.text_area("New Leads Found", "\n".join([email for _, email in new_leads]), height=200)
                progress_bar.progress((idx + 1) / len(optimized_terms))
                
            st.success(f"Automation cycle completed. Total search terms: {total_search_terms}, Total emails sent: {total_emails_sent}")
            time.sleep(cycle_interval)
            
        except Exception as e:
            log_container.error(f"Critical error in automation cycle: {str(e)}")
            time.sleep(error_retry_interval)
            
    log_container.info("Automation stopped")
    st.session_state.automation_logs = automation_logs
    st.session_state.total_leads_found = total_search_terms
    st.session_state.total_emails_sent = total_emails_sent

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
    if email is None: return False
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^(email|info|contact|support|hello|hola|hi|salutations|greetings|inquiries|questions)@.*",
        r"^email@email\.com$",
        r".*@example\.com$",
        r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"
    ]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    if any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns): return False
    if any(email.lower().endswith(f"@{domain}") for domain in typo_domains): return False
    try: validate_email(email); return True
    except EmailNotValidError: return False

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
    sent_count = 0
    logs = []
    
    template = session.query(EmailTemplate).get(template_id)
    if not template:
        if log_container:
            log_container.error("Email template not found")
        return [], 0
        
    wrapped_content = wrap_email_body(template.body_content)
    
    for index, lead in enumerate(leads):
        try:
            validate_email(lead['Email'])
            response, tracking_id = send_email_ses(session, from_email, lead['Email'], template.subject, wrapped_content, reply_to=reply_to)
            
            if response:
                status = 'sent'
                message_id = response.get('MessageId', f"sent-{uuid.uuid4()}")
                sent_count += 1
                log_message = f"✅ Email sent to: {lead['Email']}"
            else:
                status = 'failed'
                message_id = None
                log_message = f"❌ Failed to send email to: {lead['Email']}"
            
            email_campaign = save_email_campaign(session, lead['Email'], template_id, status, datetime.utcnow(), template.subject, message_id, wrapped_content)
            if email_campaign:
                session.commit()
            logs.append(log_message)

            if progress_bar:
                progress_bar.progress((index + 1) / len(leads))
            if status_text:
                status_text.text(f"Processed {index + 1}/{len(leads)} leads ({sent_count} sent)")
            if results is not None:
                results.append({"Email": lead['Email'], "Status": status})
            if log_container:
                log_container.text(log_message)
            
        except EmailNotValidError:
            log_message = f"❌ Invalid email address: {lead['Email']}"
            logs.append(log_message)
            if log_container:
                log_container.warning(log_message)
        except Exception as e:
            error_message = f"Error sending email to {lead['Email']}: {str(e)}"
            logs.append(error_message)  # Add this line
            logging.error(error_message)
            if log_container:
                log_container.error(error_message)
                
    return logs, sent_count

def view_campaign_logs():
    st.header("Email Logs")
    with db_session() as session:
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
    st.title("View Leads")
    with db_session() as session:
        leads_df = fetch_leads_with_sources(session)
        if not leads_df.empty:
            st.session_state.leads = leads_df  # Store leads in session state
            st.dataframe(leads_df, hide_index=True)
            
            st.subheader("Filter Leads")
            col1, col2 = st.columns(2)
            with col1:
                search_term_filter = st.text_input("Filter by Search Term")
            with col2:
                email_status_filter = st.selectbox("Filter by Email Status", options=["All", "Sent", "Failed", "Not Contacted"])

            filtered_leads = leads_df.copy()
            if search_term_filter:
                filtered_leads = filtered_leads[filtered_leads['Source'].str.contains(search_term_filter, case=False, na=False)]
            if email_status_filter != "All":
                if email_status_filter == "Not Contacted":
                    filtered_leads = filtered_leads[filtered_leads['Last Email Status'] == "Not Contacted"]
                else:
                    filtered_leads = filtered_leads[filtered_leads['Last Email Status'] == email_status_filter]

            st.markdown(f"### Showing {len(filtered_leads)} of {len(leads_df)} leads")
            
            edited_df = st.data_editor(
                filtered_leads,
                column_config={
                    "Delete": st.column_config.CheckboxColumn(required=True),
                    "ID": st.column_config.Column(disabled=True),
                    "Email": st.column_config.Column(width="medium"),
                    "First Name": st.column_config.Column(width="medium"),
                    "Last Name": st.column_config.Column(width="medium"),
                    "Company": st.column_config.Column(width="medium"),
                    "Job Title": st.column_config.Column(width="medium"),
                    "Source": st.column_config.Column(width="large", disabled=True),
                    "Last Contact": st.column_config.Column(width="medium", disabled=True),
                    "Last Email Status": st.column_config.Column(width="medium", disabled=True)
                },
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

            # Add visualization section
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
    st.title("Search Terms Management")
    
    with db_session() as session:
        # Create new group section
        with st.expander("Create New Group", expanded=False):
            new_group_name = st.text_input("New Group Name")
            if st.button("Create Group"):
                if new_group_name.strip():
                    create_search_term_group(session, new_group_name)
                    st.success(f"Group '{new_group_name}' created successfully!")
                    st.rerun()
                else:
                    st.warning("Please enter a group name")

        # Manage existing groups
        groups = session.query(SearchTermGroup).all()
        if groups:
            st.subheader("Existing Groups")
            for group in groups:
                with st.expander(f"Group: {group.name}", expanded=False):
                    # Get all search terms
                    all_terms = session.query(SearchTerm).filter_by(campaign_id=get_active_campaign_id()).all()
                    
                    # Get terms currently in this group
                    group_terms = [term for term in all_terms if term.group_id == group.id]
                    
                    # Create options for multiselect
                    term_options = [f"{term.id}:{term.term}" for term in all_terms]
                    default_values = [f"{term.id}:{term.term}" for term in group_terms]
                    
                    # Display multiselect for terms
                    selected_terms = st.multiselect(
                        "Select terms for this group",
                        options=term_options,
                        default=default_values,
                        format_func=lambda x: x.split(':')[1]
                    )
                    
                    if st.button("Update Group", key=f"update_{group.id}"):
                        update_search_term_group(session, group.id, selected_terms)
                        st.success("Group updated successfully!")
                        st.rerun()
                    
                    if st.button("Delete Group", key=f"delete_{group.id}"):
                        delete_search_term_group(session, group.id)
                        st.success("Group deleted successfully!")
                        st.rerun()

        # Add new search terms section
        st.subheader("Add New Search Term")
        with st.form("add_search_term_form"):
            new_term = st.text_input("New Search Term")
            group_options = ["None"] + [f"{g.id}:{g.name}" for g in groups]
            group_for_new_term = st.selectbox("Assign to Group", options=group_options)
            
            if st.form_submit_button("Add Term"):
                if new_term.strip():
                    sanitized_term = re.sub(r'[^\w\s]', '', new_term)  # Remove special characters
                    if sanitized_term and sanitized_term not in st.session_state.search_terms:
                        add_new_search_term(session, new_term, get_active_campaign_id(), group_for_new_term)
                        st.success(f"Term '{new_term}' added successfully!")
                        st.rerun()
                    else:
                        st.warning("Invalid or duplicate search term.")
                else:
                    st.warning("Please enter a search term")

def update_search_term_group(session, group_id, updated_terms):
    try:
        # Get IDs of selected terms
        selected_term_ids = [int(term.split(':')[0]) for term in updated_terms]
        
        # Update all terms that should be in this group
        session.query(SearchTerm)\
            .filter(SearchTerm.id.in_(selected_term_ids))\
            .update({SearchTerm.group_id: group_id}, synchronize_session=False)
        
        # Remove group_id from terms that were unselected
        session.query(SearchTerm)\
            .filter(SearchTerm.group_id == group_id)\
            .filter(~SearchTerm.id.in_(selected_term_ids))\
            .update({SearchTerm.group_id: None}, synchronize_session=False)
        
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_search_term_group: {str(e)}")
        raise

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        group_id = None
        if group_for_new_term != "None":
            group_id = int(group_for_new_term.split(':')[0])
            
        new_search_term = SearchTerm(
            term=new_term,
            campaign_id=campaign_id,
            group_id=group_id,
            created_at=datetime.utcnow()
        )
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")
        raise

def ai_group_search_terms(session, ungrouped_terms):
    existing_groups = session.query(SearchTermGroup).all()
    
    prompt = f"""
    Categorize these search terms into existing groups or suggest new ones:
    {', '.join([term.term for term in ungrouped_terms])}

    Existing groups: {', '.join([group.name for group in existing_groups])}

    Respond with a JSON object: {{group_name: [term1, term2, ...]}}
    """

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
            if search_term:
                search_term.group_id = group.id

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
            # Set group_id to None for all search terms in this group
            session.query(SearchTerm).filter(SearchTerm.group_id == group_id).update({SearchTerm.group_id: None})
            session.delete(group)
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting search term group: {str(e)}")

# Add new functions for lead management
def delete_lead_and_sources(session, lead_id):
    try:
        lead = session.query(Lead).get(lead_id)
        if lead:
            session.query(LeadSource).filter(LeadSource.lead_id == lead_id).delete()
            session.query(CampaignLead).filter(CampaignLead.lead_id == lead_id).delete()
            session.query(EmailCampaign).filter(EmailCampaign.lead_id == lead_id).delete()
            session.delete(lead)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting lead and sources: {str(e)}")
        return False

def update_lead(session, lead_id, updated_data):
    try:
        lead = session.query(Lead).get(lead_id)
        if lead:
            for key, value in updated_data.items():
                setattr(lead, key.lower().replace(' ', '_'), value)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        logging.error(f"Error updating lead: {str(e)}")
        return False

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

def wrap_email_body(body_content):
    """Wrap email content in a properly structured HTML template with sanitization"""
    try:
        # Basic HTML sanitization
        soup = BeautifulSoup(body_content, 'html.parser')
        
        # Remove potentially dangerous tags/attributes
        for tag in soup.find_all(True):
            if tag.name in ['script', 'iframe', 'object', 'embed']:
                tag.decompose()
            for attr in list(tag.attrs):
                if attr.startswith('on') or attr in ['style', 'class']:
                    del tag[attr]
        
        sanitized_content = str(soup)
        
        # Wrap in responsive template
        template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Email Preview</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                @media only screen and (max-width: 600px) {{
                    body {{
                        padding: 10px;
                    }}
                }}
            </style>
        </head>
        <body>
            {sanitized_content}
        </body>
        </html>
        """
        return template
    except Exception as e:
        logging.error(f"Error wrapping email content: {str(e)}")
        return f"""
        <!DOCTYPE html>
        <html>
        <body>
            <p style="color: red;">Error processing template: {str(e)}</p>
            <hr>
            <pre>{html.escape(body_content)}</pre>
        </body>
        </html>
        """

def get_email_preview(session, template_id, from_email, reply_to):
    """Get a preview of the email template with proper error handling and sanitization"""
    try:
        template = session.query(EmailTemplate).filter_by(id=template_id).first()
        if not template:
            return "<p>Template not found</p>"
        
        if not template.body_content:
            return "<p>Template content is empty</p>"
            
        try:
            wrapped_content = wrap_email_body(template.body_content)
            # Basic HTML validation
            soup = BeautifulSoup(wrapped_content, 'html.parser')
            if not soup.find('body'):
                return "<p>Invalid HTML structure in template</p>"
                
            # Add preview info
            info_div = soup.new_tag('div')
            info_div['style'] = 'background-color: #f8f9fa; padding: 10px; margin-bottom: 20px; border-radius: 5px;'
            info_div.string = f"From: {from_email}\nReply-To: {reply_to}\nSubject: {template.subject}"
            soup.body.insert(0, info_div)
            
            return str(soup)
        except Exception as e:
            logging.error(f"Error wrapping email content: {str(e)}")
            return f"<p>Error processing template: {str(e)}</p>"
    except SQLAlchemyError as e:
        logging.error(f"Database error in get_email_preview: {str(e)}")
        return "<p>Error accessing template</p>"
    except Exception as e:
        logging.error(f"Unexpected error in get_email_preview: {str(e)}")
        return "<p>Unexpected error occurred</p>"

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

            leads = fetch_leads(session, send_option, specific_email, selected_terms, exclude_previously_contacted, template.campaign_id)
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

def fetch_leads(session, send_option, specific_email, selected_terms, exclude_previously_contacted, campaign_id):
    try:
        query = session.query(Lead)
        if send_option == "Specific Email":
            query = query.filter(Lead.email == specific_email)
        elif send_option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and selected_terms:
            # Join through LeadSource to get leads from specific search terms
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))
        
        if exclude_previously_contacted:
            # Use a subquery to exclude previously contacted leads
            contacted_leads = session.query(EmailCampaign.lead_id)\
                .filter(EmailCampaign.sent_at.isnot(None))\
                .filter(EmailCampaign.campaign_id == campaign_id)\
                .subquery()
            query = query.outerjoin(contacted_leads, Lead.id == contacted_leads.c.lead_id)\
                .filter(contacted_leads.c.lead_id.is_(None))
        
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
            base_terms = [term.term for term in session.query(SearchTerm).filter_by(campaign_id=get_active_campaign_id()).all()]
            optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            st.subheader("Optimized Search Terms")
            st.write(", ".join(optimized_terms))
            total_search_terms = len(optimized_terms)
            progress_bar = st.progress(0)
            for idx, term in enumerate(optimized_terms):
                results = manual_search(session, [term], 10, ignore_previously_fetched=True)
                new_leads = [(save_lead(session, res['Email'], url=res['URL']).id, res['Email']) for res in results['results'] if save_lead(session, res['Email'], url=res['URL'])]
                if new_leads:
                    template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                    if template:
                        from_email = kb_info.get('contact_email', 'hello@indosy.com')
                        reply_to = kb_info.get('contact_email', 'eugproductions@gmail.com')
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
    st.session_state.update({"automation_logs": automation_logs, "total_leads_found": total_search_terms, "total_emails_sent": total_emails_sent})

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
    """Generate optimized search terms using AI based on knowledge base info"""
    if not kb_info:
        logging.error("No knowledge base info available")
        return []
        
    try:
        prompt = f"""Generate 5 optimized search terms based on:
        Base terms: {', '.join(base_terms)}
        Company: {str(kb_info.get('company_description', ''))}
        Target Market: {str(kb_info.get('company_target_market', ''))}
        Product: {str(kb_info.get('product_description', ''))}
        
        Format: Return only the terms, one per line, no numbering or extra text.
        Example:
        software engineer spain
        tech lead barcelona
        senior developer madrid
        """
        
        response = openai_chat_completion(
            messages=[
                {"role": "system", "content": "You are a search term optimization assistant. Generate targeted search terms for lead generation."},
                {"role": "user", "content": prompt}
            ],
            function_name="generate_search_terms"
        )
        
        if isinstance(response, str):
            terms = [term.strip() for term in response.split('\n') if term.strip()]
            return terms[:5]  # Ensure we only return max 5 terms
        return []
    except Exception as e:
        logging.error(f"Error generating search terms: {str(e)}")
        return base_terms[:5] if base_terms else []

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

def get_search_terms(session):
    return [term.term for term in session.query(SearchTerm).filter_by(campaign_id=get_active_campaign_id()).all()]

def get_ai_response(prompt):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='ai').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:
            return ""
        
        client = OpenAI(api_key=general_settings.value['openai_api_key'])
        try:
            response = client.chat.completions.create(
                model=general_settings.value.get('openai_model', 'gpt-4'),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API error: {str(e)}")
            return ""

def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        return []

def bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_bar=None, status_text=None, results=None, log_container=None):
    MAX_EMAILS_PER_MINUTE = 30
    email_count = 0
    last_email_time = time.time()
    total_leads = len(leads)
    sent_count = 0
    logs = []
    
    template = session.query(EmailTemplate).get(template_id)
    if not template:
        if log_container:
            log_container.error("Email template not found")
        return [], 0
        
    wrapped_content = wrap_email_body(template.body_content)
    
    for index, lead in enumerate(leads):
        if email_count >= MAX_EMAILS_PER_MINUTE:
            sleep_time = 60 - (time.time() - last_email_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            email_count = 0
            last_email_time = time.time()
            
        try:
            validate_email(lead['Email'])
            response, tracking_id = send_email_ses(session, from_email, lead['Email'], template.subject, wrapped_content, reply_to=reply_to)
            
            if response:
                status = 'sent'
                message_id = response.get('MessageId', f"sent-{uuid.uuid4()}")
                sent_count += 1
                log_message = f"✅ Email sent to: {lead['Email']}"
            else:
                status = 'failed'
                message_id = None
                log_message = f"❌ Failed to send email to: {lead['Email']}"
            
            email_campaign = save_email_campaign(session, lead['Email'], template_id, status, datetime.utcnow(), template.subject, message_id, wrapped_content)
            if email_campaign:
                session.commit()
            logs.append(log_message)

            if progress_bar:
                progress_bar.progress((index + 1) / total_leads)
            if status_text:
                status_text.text(f"Processed {index + 1}/{total_leads} leads ({sent_count} sent)")
            if results is not None:
                results.append({"Email": lead['Email'], "Status": status})
            if log_container:
                log_container.text(log_message)

            email_count += 1
            
        except EmailNotValidError:
            log_message = f"❌ Invalid email address: {lead['Email']}"
            logs.append(log_message)
            if log_container:
                log_container.warning(log_message)
        except Exception as e:
            error_message = f"Error sending email to {lead['Email']}: {str(e)}"
            logs.append(error_message)  # Add this line
            logging.error(error_message)
            if log_container:
                log_container.error(error_message)
                
    return logs, sent_count

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

def initialize_settings():
    """Initialize database and required settings."""
    with safe_db_session() as session:
        try:
            # Create tables if they don't exist
            Base.metadata.create_all(bind=engine)
            
            # Check and create default settings
            check_required_settings(session)
            
            # Create default AI settings if they don't exist
            ai_settings = session.query(Settings).filter_by(setting_type='ai').first()
            if not ai_settings:
                default_ai_settings = {
                    'openai_api_key': '',
                    'api_base_url': 'https://api.openai.com/v1',
                    'model_name': 'gpt-4',
                    'max_tokens': 1500,
                    'temperature': 0.7
                }
                session.add(Settings(name='AI Settings', setting_type='ai', value=default_ai_settings))
                session.commit()
            
            # Create default automation settings if they don't exist
            automation_settings = session.query(Settings).filter_by(setting_type='automation').first()
            if not automation_settings:
                default_automation_settings = {
                    'max_runtime_hours': 24,
                    'cycle_interval_seconds': 3600,
                    'error_retry_seconds': 300,
                    'results_per_search': 10,
                    'max_leads_per_cycle': 500
                }
                session.add(Settings(name='Automation Settings', setting_type='automation', value=default_automation_settings))
                session.commit()
            
            # Create default email settings if they don't exist
            email_defaults_settings = session.query(Settings).filter_by(setting_type='email_defaults').first()
            if not email_defaults_settings:
                default_email_defaults_settings = {
                    'default_from_email': '',
                    'default_reply_to': '',
                    'email_batch_size': 100
                }
                session.add(Settings(name='Email Defaults', setting_type='email_defaults', value=default_email_defaults_settings))
                session.commit()
            
            # Verify database connection
            session.execute(text("SELECT 1"))
            session.commit()
            
        except Exception as e:
            logging.error(f"Failed to initialize settings: {str(e)}")
            raise

def get_openai_settings():
    """Get OpenAI settings from database"""
    with safe_db_session() as session:
        settings = session.query(Settings).filter_by(setting_type='openai').first()
        if not settings:
            initialize_settings()
            settings = session.query(Settings).filter_by(setting_type='openai').first()
        return settings.value

def run_automation_cycle(session, settings, log_container=None):
    """Core automation logic separated from UI concerns"""
    results = {
        'new_leads': [],
        'emails_sent': 0,
        'logs': [],
        'active_template': None,
        'active_project': None,
        'active_campaign': None,
        'search_terms_used': [],
        'errors': []
    }
    
    def log(message, level='info'):
        results['logs'].append({'message': message, 'level': level, 'timestamp': datetime.utcnow()})
        if log_container:
            log_container.markdown(f"{'🔵' if level=='info' else '🟢' if level=='success' else '🔴' if level=='error' else '🟡'} {message}")
    
    try:
        # Get active project and campaign
        project = session.query(Project).get(get_active_project_id())
        campaign = session.query(Campaign).get(get_active_campaign_id())
        if not project or not campaign:
            log("No active project or campaign found", 'error')
            results['errors'].append("No active project or campaign")
            return results
        
        results['active_project'] = project.project_name
        results['active_campaign'] = campaign.campaign_name
        log(f"Running automation for project: {project.project_name}, campaign: {campaign.campaign_name}")
        
        # Get knowledge base info
        kb_info = get_knowledge_base_info(session, project.id)
        if not kb_info:
            log("No knowledge base found for active project", 'error')
            results['errors'].append("Missing knowledge base")
            return results
        
        # Get email settings and template
        email_settings = session.query(EmailSettings).first()
        template = session.query(EmailTemplate).filter_by(campaign_id=campaign.id).first()
        if settings.get('auto_email', True) and (not email_settings or not template):
            log("Email settings or template not found", 'error')
            results['errors'].append("Missing email configuration")
            return results
        
        results['active_template'] = template.template_name if template else None
        
        # Get and optimize search terms
        base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=project.id).all()]
        if not base_terms:
            log("No search terms found", 'error')
            results['errors'].append("No search terms configured")
            return results
        
        optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
        results['search_terms_used'] = optimized_terms
        log(f"Generated {len(optimized_terms)} optimized search terms", 'info')
        
        # Process each search term
        for term in optimized_terms:
            if len(results['new_leads']) >= settings['max_leads_per_cycle']:
                log(f"Reached maximum leads limit ({settings['max_leads_per_cycle']})")
                break
                
            log(f"Searching for term: {term}")
            search_results = manual_search(
                session=session,
                terms=[term],
                num_results=settings['results_per_search'],
                ignore_previously_fetched=True,
                optimize_english=settings.get('optimize_english', False),
                optimize_spanish=settings.get('optimize_spanish', False),
                language=settings.get('language', 'ES'),
                log_container=log_container
            )
            
            # Process search results
            for res in search_results['results']:
                lead = save_lead(session, res['Email'], url=res['URL'])
                if lead:
                    results['new_leads'].append(lead)
                    log(f"Found new lead: {lead.email}", 'success')
                    
                    # Send email if enabled
                    if settings.get('auto_email', True) and template and email_settings:
                        try:
                            wrapped_content = wrap_email_body(template.body_content)
                            email_defaults = session.query(Settings).filter_by(setting_type='email_defaults').first()
                            from_email = email_defaults.value.get('default_from_email') if email_defaults else email_settings.email
                            reply_to = email_defaults.value.get('default_reply_to') if email_defaults else email_settings.email
                            response, tracking_id = send_email_ses(
                                session, 
                                from_email,
                                lead.email, 
                                template.subject, 
                                wrapped_content, 
                                reply_to=reply_to
                            )
                            
                            if response:
                                results['emails_sent'] += 1
                                log(f"Email sent to {lead.email}", 'success')
                                save_email_campaign(
                                    session, 
                                    lead.email, 
                                    template.id, 
                                    'sent', 
                                    datetime.utcnow(), 
                                    template.subject, 
                                    response.get('MessageId'), 
                                    wrapped_content
                                )
                            else:
                                log(f"Failed to send email to {lead.email}", 'error')
                                results['errors'].append(f"Email sending failed: {lead.email}")
                        except Exception as e:
                            log(f"Error sending email to {lead.email}: {str(e)}", 'error')
                            results['errors'].append(f"Email error: {str(e)}")
                            
            # Save automation log
            current_search_term = session.query(SearchTerm).filter_by(term=term).first()
            session.add(AutomationLog(
                campaign_id=campaign.id,
                search_term_id=current_search_term.id if current_search_term else None,
                leads_gathered=len(results['new_leads']),
                emails_sent=results['emails_sent'],
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                status='completed',
                logs=results['logs']
            ))
            session.commit()
            
    except Exception as e:
        error_msg = f"Critical error in automation cycle: {str(e)}"
        log(error_msg, 'error')
        results['errors'].append(error_msg)
        
    return results

@safe_button_operation
def handle_automation_start():
    """Safe handler for automation start button"""
    with safe_db_session() as session:
        settings_ok, error_msg = check_required_settings(session)
        if not settings_ok:
            st.error(error_msg)
            return False
            
        st.session_state.update({
            "automation_status": True,
            "automation_logs": [],
            "total_leads_found": 0,
            "total_emails_sent": 0,
            "automation_start_time": datetime.utcnow(),
            "automation_active": True
        })
        return True

@safe_button_operation
def handle_automation_stop():
    """Safe handler for automation stop button"""
    st.session_state.automation_status = False
    if 'automation_thread' in st.session_state and st.session_state.automation_thread:
        st.session_state.automation_thread = None
    st.session_state.automation_active = False
    return True

def unified_automation_page():
    """Unified automation page with improved error handling."""
    st.title("AI Automation")
    
    # Initialize containers
    status_container = st.empty()
    metrics_container = st.empty()
    log_container = st.empty()
    leads_container = st.empty()
    
    try:
        with safe_db_session() as session:
            # Check required settings first
            settings_ok, error_msg = check_required_settings(session)
            if not settings_ok:
                st.error(f"Configuration error: {error_msg}")
                return
            
            # Fetch automation settings
            automation_settings = get_automation_settings(session)
            if not automation_settings:
                st.warning("Automation settings not found. Using defaults.")
                automation_settings = create_default_automation_settings(session)
    except ValueError as e:
        st.error(f"Settings error: {str(e)}")
        return
    except Exception as e:
        logging.error(f"Database error in automation page: {str(e)}")
        st.error("Failed to load automation settings. Please check the database connection.")
        return
    
    # Display automation controls
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            if st.button("▶️ Start Automation", use_container_width=True):
                handle_automation_start()
        except Exception as e:
            st.error(f"Failed to start automation: {str(e)}")
    
    with col2:
        try:
            if st.button("⏹️ Stop Automation", use_container_width=True):
                handle_automation_stop()
        except Exception as e:
            st.error(f"Failed to stop automation: {str(e)}")
    
    with col3:
        try:
            if st.button("🔄 Quick Scan", use_container_width=True):
                perform_quick_scan(session)
        except Exception as e:
            st.error(f"Quick scan failed: {str(e)}")
    
    # Display automation metrics
    try:
        display_automation_metrics(metrics_container)
    except Exception as e:
        logging.error(f"Failed to display metrics: {str(e)}")
        metrics_container.error("Failed to load automation metrics")
    
    # Display automation logs
    try:
        display_automation_logs(log_container)
    except Exception as e:
        logging.error(f"Failed to display logs: {str(e)}")
        log_container.error("Failed to load automation logs")
    
    # Display found leads
    try:
        display_found_leads(leads_container)
    except Exception as e:
        logging.error(f"Failed to display leads: {str(e)}")
        leads_container.error("Failed to load found leads")

def main():
    """Main application entry point."""
    if not initialize_settings():
        st.error("Failed to initialize application settings. Please check your configuration.")
        return
        
    st.set_page_config(
        page_title="Email Lead Generator",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def google_search(query, stop=10, lang='es'):
    """
    Perform a Google search with the given query and parameters.
    
    Args:
        query (str): The search query
        stop (int): Number of results to return
        lang (str): Language code for search results
        
    Returns:
        list: List of search result URLs
    """
    try:
        # Use googlesearch-python package directly
        search_results = google_search(query, stop=stop, lang=lang)
        return list(search_results)
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

def is_valid_contact_email(email):
    """Check if an email is valid for contacting"""
    try:
        validate_email(email)
        return not any(pattern in email.lower() for pattern in BLOCKED_EMAIL_PATTERNS)
    except EmailNotValidError:
        return False

def extract_company_name(soup, url):
    """Extract company name from page"""
    # Try meta tags first
    company = soup.find('meta', {'property': 'og:site_name'})
    if company:
        return company['content']
    
    # Try domain name
    domain = get_domain_from_url(url)
    if domain:
        domain = domain.replace('www.', '').split('.')[0].upper()
        return domain
    
    return 'Unknown'

def save_lead(session, url, search_term):
    """Save lead with improved extraction"""
    try:
        # Get page content with retries
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(url, timeout=10, verify=False, headers=headers)
        response.raise_for_status()
        
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract information
        emails = extract_emails_from_html(html_content)
        name, _, job_title = extract_info_from_page(soup)
        company = extract_company_name(soup, url)
        
        # Track processed emails to avoid duplicates
        processed_emails = set()
        leads = []
        
        # Get or create search term object if string was passed
        search_term_obj = None
        if isinstance(search_term, str):
            search_term_obj = session.query(SearchTerm).filter_by(term=search_term).first()
            if not search_term_obj:
                search_term_obj = SearchTerm(
                    term=search_term,
                    campaign_id=get_active_campaign_id()
                )
                session.add(search_term_obj)
                session.flush()
        else:
            search_term_obj = search_term
        
        for email in emails:
            if email in processed_emails:
                continue
                
            if is_valid_email(email) and is_valid_contact_email(email):
                processed_emails.add(email)
                
                # Check if lead already exists
                lead = session.query(Lead).filter_by(email=email).first()
                if not lead:
                    lead = Lead(
                        email=email,
                        first_name=name,
                        company=company,
                        job_title=job_title
                    )
                    session.add(lead)
                    session.flush()
                
                # Save lead source
                save_lead_source(
                    session,
                    lead_id=lead.id,
                    search_term_id=search_term_obj.id if search_term_obj else None,
                    url=url,
                    http_status=response.status_code,
                    scrape_duration=str(response.elapsed.total_seconds()),
                    page_title=soup.title.string if soup.title else None,
                    meta_description=soup.find('meta', {'name': 'description'}).get('content') if soup.find('meta', {'name': 'description'}) else None,
                    content=str(soup)[:1000],
                    tags=None,
                    phone_numbers=None
                )
                leads.append(lead)
        
        session.commit()
        return leads[0] if leads else None
        
    except requests.exceptions.RequestException as e:
        if '403' in str(e):
            logging.warning(f"Access forbidden for {url} - site may be blocking scraping")
        elif '429' in str(e):
            logging.warning(f"Rate limited by {url} - too many requests")
        else:
            logging.error(f"Error saving lead from {url}: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error saving lead from {url}: {str(e)}")
        return None

def get_domain_from_url(url: str) -> str:
    """
    Extract the domain from a URL.
    
    Args:
        url (str): The URL to extract the domain from
        
    Returns:
        str: The extracted domain name
        
    Example:
        >>> get_domain_from_url('https://www.example.com/path')
        'example.com'
    """
    try:
        # Use urllib.parse for reliable URL parsing
        from urllib.parse import urlparse
        parsed = urlparse(url)
        # Get domain without www prefix
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception as e:
        logging.error(f"Error extracting domain from URL {url}: {str(e)}")
        return url  # Return original URL if parsing fails

def should_skip_domain(domain: str) -> bool:
    """
    Check if a domain should be skipped based on predefined rules.
    
    Args:
        domain (str): The domain to check
        
    Returns:
        bool: True if domain should be skipped, False otherwise
    """
    # Common domains to skip
    SKIP_DOMAINS = {
        'linkedin.com',
        'facebook.com',
        'twitter.com',
        'instagram.com',
        'youtube.com',
        'google.com',
        'blogspot.com',
        'wordpress.com',
        'medium.com'
    }
    
    return any(
        domain.endswith(skip_domain) or
        domain == skip_domain
        for skip_domain in SKIP_DOMAINS
    )

if __name__ == "__main__":
    initialize_settings()
    main()
