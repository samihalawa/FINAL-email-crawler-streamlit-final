

# Standard library imports
import os
import json
import re
import logging
import logging.config
import asyncio
import time
import random
import html
import smtplib
import uuid
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager, wraps
from threading import local, Lock
from urllib.parse import urlparse, urlencode
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional

# Third-party imports
try:
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
        Index, inspect, Float
    )
    from sqlalchemy.orm import (
        declarative_base, sessionmaker, relationship, 
        Session, joinedload
    )
    from sqlalchemy.exc import SQLAlchemyError
    from botocore.exceptions import ClientError
    from tenacity import (
        retry, stop_after_attempt, wait_random_exponential, 
        wait_fixed, wait_exponential
    )
    from email_validator import validate_email, EmailNotValidError
    from streamlit_option_menu import option_menu
    from openai import OpenAI
    from streamlit_tags import st_tags
    import plotly.express as px
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry
except ImportError as e:
    print(f"Error importing required packages: {str(e)}")
    print("Please install required packages using: pip install -r requirements.txt")
    raise

# Load environment variables
load_dotenv()

# Ensure LOGGING_CONFIG is defined
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'  # Add missing formatter reference
        }
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Disable urllib3 warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Database configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST", "aws-0-eu-central-1.pooler.supabase.com")
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres.whwiyccyyfltobvqxiib")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "SamiHalawa1996")
DB_PORT = os.getenv("SUPABASE_DB_PORT", "6543")  # Using transaction mode pooler port

# Ensure all required environment variables are set
if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Configure database engine with retry mechanism
def get_engine():
    return create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        connect_args={
            'connect_timeout': 10,
            'application_name': 'autoclient_app',
            'options': '-c statement_timeout=30000'
        }
    )

# Initialize engine with retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def initialize_engine():
    engine = get_engine()
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine

try:
    engine = initialize_engine()
except Exception as e:
    logger.error(f"Failed to initialize database engine: {str(e)}")
    st.error("Database connection failed. Please check your credentials and connection.")
    raise

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=Session
)

Base = declarative_base()

# Add cleanup handler
import atexit
@atexit.register
def cleanup_engine():
    if 'engine' in globals():
        engine.dispose()
        logger.info("Database engine disposed")

# Add thread-local storage for database sessions
thread_local = local()

@contextmanager
def get_db():
    """Get a database session with automatic cleanup."""
    session = None
    try:
        session = SessionLocal()
        yield session
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        if session:
            session.close()
            if hasattr(thread_local, "session"):
                del thread_local.session

def get_db_session() -> Session:
    """Get or create a database session for the current thread."""
    if not hasattr(thread_local, "session"):
        with get_db() as session:
            thread_local.session = session
    try:
        return thread_local.session
    finally:
        # Ensure session is cleaned up
        if hasattr(thread_local, "session"):
            thread_local.session.close()
            del thread_local.session

@contextmanager
def db_session():
    """Provide a transactional scope around a series of operations."""
    session = None
    try:
        # Try to get existing session from thread local storage
        if hasattr(thread_local, "session"):
            session = thread_local.session
        else:
            # Create new session if none exists
            session = SessionLocal()
            thread_local.session = session
        
        yield session
        session.commit()
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        if session:
            session.close()
            if hasattr(thread_local, "session"):
                del thread_local.session

# Update the database configuration with more conservative settings
engine = create_engine(
    DATABASE_URL,
    pool_size=10,  # Increased from 5 to allow more concurrent connections
    max_overflow=2,  # Reduced from 10 to limit total connections
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={
        'connect_timeout': 10,
        'application_name': 'autoclient_app',
        'options': '-c statement_timeout=30000'
    }
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=Session
)

Base = declarative_base()

# Ensure all required environment variables are set
if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

try:
    # Test database connection
    with db_session() as session:
        session.execute(text("SELECT 1"))  # This line is safe as it does not involve user input
except SQLAlchemyError as e:
    st.error(f"Failed to connect to database: {str(e)}")
    logging.error(f"Database connection error: {str(e)}")
    raise

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")


class Project(Base):
    __tablename__ = 'projects'
    __table_args__ = (
        Index('idx_project_created', 'created_at'),
    )
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text, default="Default Project")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaigns = relationship("Campaign", back_populates="project", cascade="all, delete-orphan")
    knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False, cascade="all, delete-orphan")

class Campaign(Base):
    __tablename__ = 'campaigns'
    __table_args__ = (
        Index('idx_campaign_created', 'created_at'),
        Index('idx_campaign_project', 'project_id'),
    )
    id = Column(BigInteger, primary_key=True)
    campaign_name = Column(Text, default="Default Campaign")
    campaign_type = Column(Text, default="Email")
    project_id = Column(BigInteger, ForeignKey('projects.id', ondelete='CASCADE'), default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    auto_send = Column(Boolean, default=False)
    loop_automation = Column(Boolean, default=False)
    ai_customization = Column(Boolean, default=False)
    max_emails_per_group = Column(BigInteger, default=500)
    loop_interval = Column(BigInteger, default=60)
    project = relationship("Project", back_populates="campaigns")
    email_campaigns = relationship("EmailCampaign", back_populates="campaign", cascade="all, delete-orphan")
    search_terms = relationship("SearchTerm", back_populates="campaign", cascade="all, delete-orphan")
    campaign_leads = relationship("CampaignLead", back_populates="campaign", cascade="all, delete-orphan")


# Replace the existing SearchProcess class with this updated version
class SearchProcess(Base):
    __tablename__ = 'search_processes'
    __table_args__ = (
        Index('idx_search_process_status', 'status'),
        Index('idx_search_process_created', 'created_at'),
    )
    id = Column(BigInteger, primary_key=True)
    search_terms = Column(JSON)  # Store list of search terms
    settings = Column(JSON)      # Store search settings
    status = Column(Text)        # 'running', 'completed', 'failed'
    results = Column(JSON)       # Store search results
    logs = Column(JSON)          # Store process logs
    total_leads_found = Column(BigInteger, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    campaign = relationship("Campaign")
 

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
    __table_args__ = (
        Index('idx_lead_email', 'email'),
        Index('idx_lead_created_at', 'created_at'),
        Index('idx_lead_status', 'status'),
    )
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    phone, first_name, last_name, company, job_title = [Column(Text) for _ in range(5)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_processed = Column(Boolean, default=False)
    last_contacted = Column(DateTime(timezone=True))
    notes = Column(Text)
    status = Column(Text, default='New')
    source = Column(Text)
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
    __table_args__ = (
        Index('idx_campaign_sent_at', 'sent_at'),
        Index('idx_campaign_status', 'status'),
        Index('idx_campaign_tracking', 'tracking_id'),
    )
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
    __table_args__ = (
        Index('idx_group_name', 'name'),
        Index('idx_group_created', 'created_at'),
    )
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    campaign = relationship("Campaign")
    search_terms = relationship("SearchTerm", back_populates="group", cascade="all, delete-orphan")
    email_template_id = Column(BigInteger, ForeignKey('email_templates.id'))
    email_template = relationship("EmailTemplate")

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    __table_args__ = (
        Index('idx_term', 'term'),
        Index('idx_term_group', 'group_id'),
        Index('idx_term_campaign', 'campaign_id'),
        Index('idx_term_created', 'created_at'),
    )
    id = Column(BigInteger, primary_key=True)
    term = Column(Text, nullable=False)
    group_id = Column(BigInteger, ForeignKey('search_term_groups.id', ondelete='SET NULL'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id', ondelete='CASCADE'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    effectiveness_score = Column(Float, default=0.0)
    last_used = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    group = relationship("SearchTermGroup", back_populates="search_terms")
    campaign = relationship("Campaign", back_populates="search_terms")
    lead_sources = relationship("LeadSource", back_populates="search_term")
    optimized_terms = relationship("OptimizedSearchTerm", back_populates="original_term")
    effectiveness = relationship("SearchTermEffectiveness", back_populates="search_term")

class LeadSource(Base):
    __tablename__ = 'lead_sources'
    __table_args__ = (
        Index('idx_source_url', 'url'),
        Index('idx_source_domain', 'domain'),
    )
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    url = Column(Text)
    domain = Column(Text)
    page_title = Column(Text)
    meta_description = Column(Text)
    scrape_duration = Column(Text)
    meta_tags = Column(Text)
    phone_numbers = Column(Text)
    content = Column(Text)
    tags = Column(Text)
    http_status = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="lead_sources")
    search_term = relationship("SearchTerm", back_populates="lead_sources")

class AIRequestLog(Base):
    __tablename__ = 'ai_request_logs'
    
    # Primary key
    id = Column(BigInteger, primary_key=True)
    
    # Text columns
    function_name = Column(Text)
    prompt = Column(Text)
    response = Column(Text)
    model_used = Column(Text)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign keys
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    email_campaign_id = Column(BigInteger, ForeignKey('email_campaigns.id'))
    
    # Relationships
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
    __table_args__ = (
        Index('idx_settings_type', 'setting_type'),
    )
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    setting_type = Column(Text, nullable=False)
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    __table_args__ = (
        Index('idx_email_settings_email', 'email'),
    )
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

def settings_page():
    """Page for managing system settings."""
    st.title("System Settings")
    
    with db_session() as session:
        # Add configuration sections
        tabs = st.tabs([
            "Email Settings",
            "API Configuration",
            "System Preferences",
            "Backup & Restore"
        ])
        
        with tabs[0]:
            with st.form("email_settings"):
                smtp_host = st.text_input("SMTP Host")
                smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535)
                smtp_user = st.text_input("SMTP Username")
                smtp_pass = st.text_input("SMTP Password", type="password")
                
                if st.form_submit_button("Save Email Settings"):
                    try:
                        validate_smtp_settings(smtp_host, smtp_port, smtp_user, smtp_pass)
                        save_email_settings(session, {
                            'host': smtp_host,
                            'port': smtp_port,
                            'username': smtp_user,
                            'password': smtp_pass
                        })
                        st.success("Email settings saved!")
                    except Exception as e:
                        st.error(f"Invalid settings: {str(e)}")
        
        with tabs[3]:
            st.subheader("Backup & Restore")
            
            if st.button("Create Backup"):
                backup_file = create_system_backup(session)
                st.download_button(
                    "Download Backup",
                    backup_file,
                    "system_backup.zip",
                    "application/zip"
                )
            
            uploaded_file = st.file_uploader("Restore from Backup")
            if uploaded_file and st.button("Restore System"):
                try:
                    restore_from_backup(session, uploaded_file)
                    st.success("System restored successfully!")
                except Exception as e:
                    st.error(f"Restore failed: {str(e)}")

@retry(
    # Retry sending email up to 3 times with exponential backoff
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    try:
        email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
        if not email_settings:
            logger.error(f"No email settings found for {from_email}")
            return None, None

        if not all([to_email, subject, body]):
            logger.error("Missing required email parameters: to_email, subject, body")
            raise ValueError("Missing required email parameters")

        tracking_id = str(uuid.uuid4())
        
        # Move tracking pixel addition inside try block
        try:
            tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
            tracked_body = wrap_email_body(body).replace('</body>', f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/></body>')
        except Exception as e:
            logger.warning(f"Failed to add tracking pixel: {str(e)}")
            tracked_body = wrap_email_body(body)

        if email_settings.provider == 'ses':
            if not all([email_settings.aws_access_key_id, email_settings.aws_secret_access_key, email_settings.aws_region]):
                logger.error("Missing AWS credentials")
                return None, None

            try:
                if ses_client is None:
                    aws_session = boto3.Session(
                        aws_access_key_id=email_settings.aws_access_key_id,
                        aws_secret_access_key=email_settings.aws_secret_access_key,
                        region_name=email_settings.aws_region,
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
            except Exception as e:
                logger.error(f"AWS SES error: {str(e)}")
                raise

        elif email_settings.provider == 'smtp':
            if not all([email_settings.smtp_server, email_settings.smtp_port, email_settings.smtp_username, email_settings.smtp_password]):
                logger.error("Missing SMTP credentials")
                return None, None

            try:
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
            except Exception as e:
                logging.error(f"SMTP error: {str(e)}")
                raise
        else:
            logging.error(f"Unknown email provider: {email_settings.provider}")
            return None, None
    except Exception as e:
        logger.error(f"Error in send_email_ses: {str(e)}")
        raise  # Allow retry mechanism to work

# Function to save email campaign data to the database
def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    try:
        lead = session.query(Lead).filter_by(email=lead_email).first()
        if not lead:
            logging.error(f"Lead with email {lead_email} not found.")
            return None

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
        return new_campaign
    except SQLAlchemyError as e:
        logging.error(f"Database error in save_email_campaign: {str(e)}")
        session.rollback()
        return None
    except Exception as e:
        logging.error(f"Error in save_email_campaign: {str(e)}")
        session.rollback()
        return None

# Function to update the log display in Streamlit
def update_log(log_container, message, level='info'):
    icon = {
        'info': '🔵',
        'success': '🟢',
        'warning': '🟠',
        'error': '🔴',
        'email_sent': '🟣'
    }.get(level, '⚪')
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"{icon} [{timestamp}] {message}"
    
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    
    st.session_state.log_entries.append(log_entry)
    
    # Keep only last 1000 entries to prevent memory issues
    if len(st.session_state.log_entries) > 1000:
        st.session_state.log_entries = st.session_state.log_entries[-1000:]
    
    # Create scrollable log container with auto-scroll and improved styling
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
                {'<br>'.join(st.session_state.log_entries)}
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
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def manual_search(session, terms, num_results, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False, language='ES', enable_email_sending=True, log_container=None, from_email=None, reply_to=None, email_template=None, process_id=None):
    domains_processed = set()
    results = []
    total_leads = 0
    
    try:
        ua = UserAgent()
        for original_term in terms:
            # Log the current term being processed
            if process_id:
                update_process_log(session, process_id, f"Processing term: {original_term}", 'info')
            
            # Use thread-local session
            with get_db() as local_session:
                search_term = local_session.query(SearchTerm).filter_by(term=original_term, campaign_id=get_active_campaign_id()).first()
                if not search_term:
                    search_term = SearchTerm(term=original_term, campaign_id=get_active_campaign_id(), created_at=datetime.utcnow())
                    local_session.add(search_term)
                    local_session.commit()
                search_term_id = search_term.id

            # Process in smaller batches
            for url_batch in batch_google_search(search_term, num_results, language):
                for url in url_batch:
                    try:
                        process_url(url, search_term_id, domains_processed, results, total_leads, session, enable_email_sending, from_email, reply_to, email_template, process_id, log_container)
                    except Exception as e:
                        log_error(f"Error processing URL {url}: {str(e)}", process_id, log_container)
                        continue

    except Exception as e:
        error_msg = f"Error in manual_search: {str(e)}"
        log_error(error_msg, process_id, log_container)
    finally:
        cleanup_resources()
    
    return {"total_leads": total_leads, "results": results}

def batch_google_search(search_term, num_results, language):
    """Generator function to yield URLs in batches"""
    batch_size = 10
    for i in range(0, num_results, batch_size):
        yield list(google_search(search_term, batch_size, lang=language))

def process_url(url, search_term_id, domains_processed, results, total_leads, session, enable_email_sending, from_email, reply_to, email_template, process_id, log_container):
    domain = urlparse(url).netloc
    if domain in domains_processed:
        return
    
    with get_db() as local_session:
        # Process URL and save results
        # ... existing URL processing code ...
        pass

def cleanup_resources():
    """Clean up any remaining resources"""
    if hasattr(thread_local, "session"):
        thread_local.session.close()
        if hasattr(thread_local, "session"):
            del thread_local.session

def update_search_term_groups(session, grouped_terms):
    """Update search term groups in the database."""
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
    """Main loop for AI-driven automation."""
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
    """Interact with OpenAI API for chat completion."""
    try:
        with db_session() as session:
            try:
                general_settings = session.query(Settings).filter_by(setting_type='general').first()
                if not general_settings or 'openai_api_key' not in general_settings.value:
                    raise ValueError("OpenAI API key not set. Please configure it in the settings.")

                model_used = general_settings.value.get('openai_model', "Qwen/Qwen2.5-72B-Instruct")  # Define model_used here
                
                client = OpenAI(
                    base_url=general_settings.value.get('openai_api_base', 'https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1/'),
                    api_key=general_settings.value.get('openai_api_key', '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')
                )

                response = client.chat.completions.create(
                    model=model_used,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=500
                )
                if not response.choices:
                    raise ValueError("No response choices returned from API")
                
                result = response.choices[0].message.content
                
                # Log the AI request
                log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id, 
                             general_settings.value.get('openai_model', "Qwen/Qwen2.5-72B-Instruct"))
                
                # Try to parse as JSON if it looks like JSON
                if result and isinstance(result, str) and result.strip().startswith(('{"', '[')):
                    try:
                        return json.loads(result)
                    except json.JSONDecodeError:
                        return result
                return result
                
            except Exception as e:
                error_msg = f"Error in API call: {str(e)}"
                log_ai_request(session, function_name, messages, error_msg, lead_id, email_campaign_id, 
                             general_settings.value.get('openai_model', "Qwen/Qwen2.5-72B-Instruct"))
                raise ValueError(error_msg)
    except Exception as e:
        logger.error(f"Fatal error in openai_chat_completion: {str(e)}")
        raise

def log_ai_request(session, function_name, prompt, response, lead_id=None, email_campaign_id=None, model_used=None):
    """Log AI request details to the database."""
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
    """Save or update a lead in the database."""
    try:
        # Check if lead already exists
        existing_lead = session.query(Lead).filter_by(email=email).first()
        if existing_lead:
            # Update existing lead with new information if provided
            for attr in ['first_name', 'last_name', 'company', 'job_title', 'phone', 'created_at']:
                if locals()[attr]:
                    setattr(existing_lead, attr, locals()[attr])
            lead = existing_lead
        else:
            # Create new lead
            lead = Lead(
                email=email,
                first_name=first_name,
                last_name=last_name,
                company=company,
                job_title=job_title,
                phone=phone,
                created_at=created_at or datetime.utcnow()
            )
            session.add(lead)
        
        session.flush()  # Flush changes to get lead ID
        
        # Save lead source if URL is provided
        if url and search_term_id:
            lead_source = LeadSource(
                lead_id=lead.id,
                url=url,
                search_term_id=search_term_id
            )
            session.add(lead_source)
        
        # Create campaign lead association
        campaign_lead = CampaignLead(
            campaign_id=get_active_campaign_id(),
            lead_id=lead.id,
            status="Not Contacted",
            created_at=datetime.utcnow()
        )
        session.add(campaign_lead)
        
        # Commit all changes
        session.commit()
        logging.info(f"Successfully saved/updated lead: {email}")
        return lead
        
        
    except Exception as e:
        logging.error(f"Error saving lead {email}: {str(e)}")
        session.rollback()
        return None

def save_lead_source(session, lead_id, search_term_id, url, http_status, scrape_duration, page_title=None, meta_description=None, content=None, tags=None, phone_numbers=None):
    """Save lead source information to the database."""
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
    """Extract visible text from HTML content."""
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return ' '.join(chunk for chunk in chunks if chunk)

def log_search_term_effectiveness(session, search_term_id, total_results, valid_leads, blogs_found, directories_found):
    """Log search term effectiveness metrics."""
    session.add(SearchTermEffectiveness(search_term_id=search_term_id, total_results=total_results, valid_leads=valid_leads, irrelevant_leads=total_results - valid_leads, blogs_found=blogs_found, directories_found=directories_found))
    session.commit()

# Lambda functions for accessing and setting active project/campaign IDs
get_active_project_id = lambda: st.session_state.get('active_project_id', 1)
get_active_campaign_id = lambda: st.session_state.get('active_campaign_id', 1)
set_active_project_id = lambda project_id: st.session_state.__setitem__('active_project_id', project_id)
set_active_campaign_id = lambda campaign_id: st.session_state.__setitem__('active_campaign_id', campaign_id)

# Function to add or retrieve a search term from the database
def add_or_get_search_term(session, term, campaign_id, created_at=None):
    """Add a new search term or retrieve an existing one."""
    search_term = session.query(SearchTerm).filter_by(term=term, campaign_id=campaign_id).first()
    if not search_term:
        search_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=created_at or datetime.utcnow())
        session.add(search_term)
        session.commit()
        session.refresh(search_term)
    return search_term.id

def fetch_campaigns(session):
    """Fetch all campaigns from the database."""
    return [f"{camp.id}: {camp.campaign_name}" for camp in session.query(Campaign).all()]

def fetch_projects(session):
    """Fetch all projects from the database."""
    return [f"{project.id}: {project.project_name}" for project in session.query(Project).all()]

def fetch_email_templates(session):
    """Fetch all email templates from the database."""
    return [f"{t.id}: {t.template_name}" for t in session.query(EmailTemplate).all()]

def create_or_update_email_template(session, template_name, subject, body_content, template_id=None, is_ai_customizable=False, created_at=None, language='ES'):
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=get_active_campaign_id(), created_at=created_at or datetime.utcnow())
    if template_id: template.template_name, template.subject, template.body_content, template.is_ai_customizable = template_name, subject, body_content, is_ai_customizable
    template.language = language
    session.add(template)
    session.commit()
    return template.id

safe_datetime_compare = lambda date1, date2: False if date1 is None or date2 is None else date1 > date2
# Function to fetch leads based on various criteria
def fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted):
    try:
        """Fetch leads based on specified criteria."""
        query = session.query(Lead)
        
        if send_option == "Specific Email":
            query = query.filter(Lead.email == specific_email)
        elif send_option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and selected_terms:
            query = (
                query.join(LeadSource)
                .join(SearchTerm)
                .filter(SearchTerm.term.in_(selected_terms))
            )
        
        if exclude_previously_contacted:
            subquery = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None)).subquery()
            query = query.outerjoin(subquery, Lead.id == subquery.c.lead_id).filter(subquery.c.lead_id.is_(None))
        
        return [{"Email": lead.email, "ID": lead.id} for lead in query.all()]
    except Exception as e:
        logging.error(f"Error fetching leads: {str(e)}")
        return []

# Function to update the display of items in a container
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


# Initialize session state for search terms if not exists
if 'search_terms' not in st.session_state:
    st.session_state.search_terms = []

def manual_search_page():
    st.title("Manual Search")
    
    with st.form("search_form"):
        # Add rate limiting warning
        st.info("Note: Searches are rate-limited to prevent API blocks")
        
        search_terms = st_tags(
            label='Enter Search Terms',
            text='Press enter after each term',
            value=st.session_state.get('search_terms', []),
            suggestions=get_recent_search_terms(st.session_state.get('db_session')),
            key='search_terms_input'
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            num_results = st.number_input(
                'Results per term', 
                min_value=1, 
                max_value=100, 
                value=10,
                help="Higher values may increase search time"
            )
        
        with col2:
            enable_email_sending = st.checkbox("Enable email sending", value=True)
            ignore_previously_fetched = st.checkbox("Ignore fetched domains", value=True)
            language = st.selectbox("Language", ["ES", "EN"], index=0)
        
        # Add error handling and validation
        try:
            if st.form_submit_button("Start Search"):
                if not search_terms:
                    st.error("Please enter at least one search term")
                    return
                    
                results = manual_search(
                    st.session_state.get('db_session'),
                    search_terms,
                    num_results,
                    ignore_previously_fetched,
                    language=language,
                    enable_email_sending=enable_email_sending
                )
                
                if results and results.get('results'):
                    st.success(f"Found {len(results['results'])} leads!")
                    display_search_results(results['results'])
                else:
                    st.warning("No results found. Try adjusting your search terms.")
                    
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            logger.error(f"Search error: {str(e)}")

    with processes_tab:
        with db_session() as session:
            # Fetch active processes and display them
            active_processes = fetch_active_processes(session)
            
            if active_processes:
                st.subheader("Active Search Processes")
                for process in active_processes:
                    with st.expander(f"Process {process.id} - {process.status.title()} - Started at {process.created_at.strftime('%Y-%m-%d %H:%M:%S')}", expanded=True):
                        # Display process controls
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Pause/Resume", key=f"toggle_{process.id}"):
                                new_status = "resume" if process.status == "paused" else "pause"
                                toggle_process(session, process.id, new_status)
                                st.rerun()
                        with col2:
                            if st.button("Stop", key=f"stop_{process.id}"):
                                toggle_process(session, process.id, "stop")
                                st.rerun()
                            
                        # Display process logs
                        display_process_logs(process.id)
                        
                        # Display results if available
                        if process.results:
                            st.metric("Total Leads Found", process.results.get('total_leads', 0))
                            if process.results.get('results'):
                                st.dataframe(pd.DataFrame(process.results['results']))
            else:
                st.info("No active search processes found.")

def get_page_description(html_content):
    """Extract page description from HTML meta tags."""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    return meta_desc['content'] if meta_desc else "No description found"

# Function to fetch search terms with their associated lead and email counts
def fetch_search_terms_with_lead_count(session):
    """Fetch search terms with their associated lead and email counts."""
    query = (session.query(SearchTerm.term, 
                          func.count(distinct(Lead.id)).label('lead_count'),
                          func.count(distinct(EmailCampaign.id)).label('email_count'))
             .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
             .join(Lead, LeadSource.lead_id == Lead.id)
             .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
             .group_by(SearchTerm.term))
    df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    return df

# Function to display email campaign logs
def view_campaign_logs():
    st.title("Email Campaign Logs")
    
    with db_session() as session:
        # Add advanced filtering
        col1, col2, col3 = st.columns(3)
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=7), datetime.now())
            )
        with col2:
            status_filter = st.multiselect(
                "Status",
                ["sent", "failed", "bounced", "opened", "clicked"]
            )
        with col3:
            campaign_filter = st.multiselect(
                "Campaigns",
                fetch_campaign_names(session)
            )
        
        # Fetch filtered logs
        logs = fetch_filtered_logs(
            session,
            date_range=date_range,
            status_filter=status_filter,
            campaign_filter=campaign_filter
        )
        
        # Display log analytics
        if logs:
            st.subheader("Log Analytics")
            fig = generate_log_analytics(logs)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add log export
            if st.button("Export Logs"):
                csv = convert_logs_to_csv(logs)
                st.download_button(
                    "Download CSV",
                    csv,
                    "email_logs.csv",
                    "text/csv"
                )
        else:
            st.info("No logs found matching your criteria.")

# Function to fetch filtered logs based on specified criteria
def fetch_filtered_logs(session, date_range, status_filter, campaign_filter):
    # Implement your filtering logic here
    pass

# Function to generate log analytics chart
def generate_log_analytics(logs):
    # Implement your analytics logic here
    pass

# Function to convert logs to CSV format
def convert_logs_to_csv(logs):
    # Implement your conversion logic here
    pass

def update_lead(session, lead_id, updated_data):
    """Update lead information in database"""
    try:
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            for key, value in updated_data.items():
                if hasattr(lead, key):
                    setattr(lead, key, value)
            session.commit()
            return True
        return False
    except SQLAlchemyError as e:
        logging.error(f"Error updating lead {lead_id}: {str(e)}")
        session.rollback()
        return False
    except Exception as e:
        logging.error(f"Unexpected error updating lead {lead_id}: {str(e)}")
        session.rollback() 
        return False

def delete_lead(session, lead_id):
    """Delete lead from database"""
    try:
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            session.delete(lead)
            session.commit()
            return True
        return False
    except SQLAlchemyError as e:
        logging.error(f"Error deleting lead {lead_id}: {str(e)}")
        session.rollback()
        return False
    except Exception as e:
        logging.error(f"Unexpected error deleting lead {lead_id}: {str(e)}")
        session.rollback()
        return False

def is_valid_email(email):
    """Validate email format"""
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False
    except Exception:
        return False


# Function to display the leads management page
def view_leads_page():
    st.title("Lead Management")
    
    with db_session() as session:
        # Add search and filter options
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("Search leads", help="Search by email, name, or company")
        with col2:
            status_filter = st.multiselect(
                "Status",
                ["New", "Contacted", "Converted", "Invalid"],
                default=["New"]
            )
        
        # Add pagination
        page_size = st.select_slider("Leads per page", options=[10, 25, 50, 100], value=25)
        page = st.number_input("Page", min_value=1, value=1)
        
        # Fetch leads with pagination
        leads = fetch_leads_paginated(
            session,
            search=search,
            status_filter=status_filter,
            page=page,
            page_size=page_size
        )
        
        # Display leads table
        if leads:
            edited_df = st.data_editor(
                pd.DataFrame(leads),
                hide_index=True,
                use_container_width=True
            )
            
            # Add export functionality
            if st.button("Export to CSV"):
                csv = convert_df_to_csv(edited_df)
                st.download_button(
                    "Download CSV",
                    csv,
                    "leads_export.csv",
                    "text/csv",
                    key='download-csv'
                )
        else:
            st.info("No leads found matching your criteria.")

# Function to fetch leads with pagination
def fetch_leads_paginated(session, search, status_filter, page, page_size):
    query = session.query(Lead)
    if search:
        query = query.filter(Lead.email.ilike(f"%{search}%") | Lead.first_name.ilike(f"%{search}%") | Lead.last_name.ilike(f"%{search}%") | Lead.company.ilike(f"%{search}%"))
    if status_filter:
        query = query.filter(Lead.status.in_(status_filter))
    return query.offset((page - 1) * page_size).limit(page_size).all()

# Function to convert a Pandas DataFrame to CSV format
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to fetch leads with their associated sources
def fetch_leads_with_sources(session):
    """Fetch leads with their sources"""
    try:
        query = (session.query(
            Lead,
            func.string_agg(LeadSource.url, ', ').label('sources'),
            func.max(EmailCampaign.sent_at).label('last_contact'),
            func.string_agg(EmailCampaign.status, ', ').label('email_statuses')
        )
        .outerjoin(LeadSource)
        .outerjoin(EmailCampaign)
        .group_by(Lead.id))
        
        results = []
        for lead, sources, last_contact, email_statuses in query.all():
            results.append({
                'ID': lead.id,
                'Email': lead.email,
                'First Name': lead.first_name,
                'Last Name': lead.last_name,
                'Company': lead.company,
                'Job Title': lead.job_title,
                'Source': sources,
                'Last Contact': last_contact,
                'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted'
            })
        
        return pd.DataFrame(results)
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_leads_with_sources: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error in fetch_leads_with_sources: {str(e)}")
        return pd.DataFrame()

# Function to delete a lead and its associated sources
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
    """Fetch search terms with their associated lead and email counts."""
    query = (session.query(SearchTerm.term, 
                           func.count(distinct(Lead.id)).label('lead_count'),
                           func.count(distinct(EmailCampaign.id)).label('email_count'))
             .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
             .join(Lead, LeadSource.lead_id == Lead.id)
             .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
             .group_by(SearchTerm.term))
    df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    return df

# Function to add a new search term to the database
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

# Function to get the active campaign ID from session state
def get_active_campaign_id():
    """Get the currently active campaign ID from session state"""
    return st.session_state.get('active_campaign_id', 1)  # Default to 1 if not set

# Function to display the search terms management page
def search_terms_page():
    st.title("Search Terms Management")
    
    with db_session() as session:
        # Add term effectiveness metrics
        st.subheader("Term Performance")
        metrics = fetch_search_term_metrics(session)
        if metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Terms", metrics['total_terms'])
            col2.metric("Active Terms", metrics['active_terms'])
            col3.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
        
        # Add term management
        with st.form("add_term_form"):
            new_term = st.text_input("Add new search term")
            group = st.selectbox(
                "Term Group",
                options=fetch_term_groups(session),
                format_func=lambda x: x.name
            )
            
            if st.form_submit_button("Add Term"):
                if new_term:
                    # Check for duplicates
                    if not is_term_duplicate(session, new_term):
                        add_search_term(session, new_term, group.id if group else None)
                        st.success(f"Added term: {new_term}")
                    else:
                        st.error("This term already exists")
        
        # Display terms with effectiveness
        terms = fetch_terms_with_effectiveness(session)
        if terms:
            st.dataframe(
                pd.DataFrame(terms),
                hide_index=True,
                use_container_width=True
            )

def update_search_term_group(session, group_id, updated_terms):
    """Update search term group membership."""
    try:
        current_term_ids = set(int(term.split(":")[0]) for term in updated_terms)
        existing_terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all()
        
        for term in existing_terms:
            if term.id not in current_term_ids:
                term.group_id = None
        
        for term_str in updated_terms:
            term_id = int(term_str.split(":")[0])
            term = session.query(SearchTerm).get(term_id)
            if term:
                term.group_id = group_id
        
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_search_term_group: {str(e)}")

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    """Add a new search term to the database."""
    try:
        new_search_term = SearchTerm(term=new_term, campaign_id=campaign_id, created_at=datetime.utcnow())
        if group_for_new_term != "None":
            new_search_term.group_id = int(group_for_new_term.split(":")[0])
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
    """Use AI to group ungrouped search terms."""
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
    """Update search term groups in the database."""
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
    """Create a new search term group in the database."""
    try:
        new_group = SearchTermGroup(name=group_name)
        session.add(new_group)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating search term group: {str(e)}")

def delete_search_term_group(session, group_id):
    """Delete a search term group from the database."""
    try:
        group = session.query(SearchTermGroup).get(group_id)
        if group:
            # Set group_id to None for all search terms in this group
            session.query(SearchTerm).filter(SearchTerm.group_id == group_id).update({SearchTerm.group_id: None})
            session.delete(group)
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting search term group: {str(e)}")


def email_templates_page():
    """Page for managing email templates."""
    st.title("Email Templates")
    
    with db_session() as session:
        # Add template versioning
        with st.form("template_form"):
            template_name = st.text_input("Template Name")
            subject = st.text_input("Subject")
            content = st.text_area("Content", height=300)
            
            col1, col2 = st.columns(2)
            with col1:
                version_note = st.text_input("Version Notes")
            with col2:
                is_active = st.checkbox("Set as Active", value=True)
            
            if st.form_submit_button("Save Template"):
                try:
                    save_template_version(
                        session,
                        template_name,
                        subject,
                        content,
                        version_note,
                        is_active
                    )
                    st.success("Template saved!")
                except Exception as e:
                    st.error(f"Failed to save template: {str(e)}")
        
        # Add template testing
        st.subheader("Test Template")
        test_email = st.text_input("Test Email Address")
        if st.button("Send Test") and test_email:
            if is_valid_email(test_email):
                send_test_email(session, template_name, test_email)
                st.success("Test email sent!")
            else:
                st.error("Invalid email address")

def get_email_preview(session, template_id, from_email, reply_to):
    """Get a preview of an email template."""
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if template:
        wrapped_content = wrap_email_body(template.body_content)
        return wrapped_content
    return "<p>Template not found</p>"

# Function to fetch all search terms from the database
def fetch_all_search_terms(session):
    return session.query(SearchTerm).all()

# Function to retrieve knowledge base information
def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

# Function to retrieve an email template by name
def get_email_template_by_name(session, template_name):
    return session.query(EmailTemplate).filter_by(template_name=template_name).first()

def bulk_send_page():
    st.title("Bulk Email Sending")
    
    with db_session() as session:
        # Validate required configurations
        templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)
        
        if not templates or not email_settings:
            st.error("Email templates or settings missing. Please configure them first.")
            if st.button("Go to Settings"):
                st.switch_page("Settings")
            return
            
        with st.form("bulk_send_form"):
            # Template selection with preview
            template_id = st.selectbox(
                "Select Template",
                options=[t.id for t in templates],
                format_func=lambda x: next((t.template_name for t in templates if t.id == x), "")
            )  # Close parenthesis here
            
            if template_id:
                template = next((t for t in templates if t.id == template_id), None)
                if template:
                    st.info(f"Subject: {template.subject}")
                    st.markdown("### Preview")
                    st.markdown(template.body_content)
            else:
                st.warning("Please select a template")  # Add else clause
            
            # Add recipient selection with validation
            recipient_type = st.radio(
                "Send to:",
                ["All Leads", "Specific Email", "Search Term Group"]
            )
            
            if recipient_type == "Specific Email":
                email = st.text_input("Enter email address")
                if email and not is_valid_email(email):
                    st.error("Invalid email format")
            
            # Add send button with confirmation
            if st.form_submit_button("Send Emails"):
                if not template_id:
                    st.error("Please select a template")
                    return
                    
                with st.spinner("Sending emails..."):
                    try:
                        results = send_bulk_emails(
                            session,
                            template_id,
                            recipient_type,
                            email if recipient_type == "Specific Email" else None
                        )
                        st.success(f"Sent {results['sent']} emails successfully!")
                    except Exception as e:
                        st.error(f"Failed to send emails: {str(e)}")

def fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted):
    """Fetch leads based on specified criteria."""
    try:
        query = session.query(Lead)
        if send_option == "Specific Email":
            query = query.filter(Lead.email == specific_email)
        elif send_option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and selected_terms:
            query = (
                query.join(LeadSource)
                .join(SearchTerm)
                .filter(SearchTerm.term.in_(selected_terms))
            )
        
        if exclude_previously_contacted:
            subquery = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None)).subquery()
            query = query.outerjoin(subquery, Lead.id == subquery.c.lead_id).filter(subquery.c.lead_id.is_(None))
        
        return [{"Email": lead.email, "ID": lead.id} for lead in query.all()]
    except Exception as e:
        logging.error(f"Error fetching leads: {str(e)}")
        return []

# Function to fetch email settings from the database
def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        return []
# Function to fetch search terms with their associated lead and email counts
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

# Function to fetch all search term groups
def fetch_search_term_groups(session):
    """Fetch all search term groups"""
    return [f"{group.id}: {group.name}" for group in session.query(SearchTermGroup).all()]

# Function to fetch search terms for given group IDs
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
        res = manual_search(session, [term.term for term in terms], 10, True, False, True, "EN", True, st.empty(), from_email, reply_to, f"{email_template.id}: {email_template.template_name}" if email_template else None)
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
    st.title("Projects & Campaigns")
    
    with db_session() as session:
        # Add project management
        with st.form("project_form"):
            project_name = st.text_input("Project Name")
            description = st.text_area("Description")
            
            if st.form_submit_button("Create Project"):
                if project_name:
                    create_project(session, project_name, description)
                    st.success("Project created!")
        
        # Display projects with metrics
        projects = fetch_projects_with_metrics(session)
        for project in projects:
            with st.expander(f" {project.name}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Active Campaigns", project.active_campaigns)
                col2.metric("Total Leads", project.total_leads)
                col3.metric("Success Rate", f"{project.success_rate:.1f}%")
                
                # Add campaign management
                st.subheader("Campaigns")
                for campaign in project.campaigns:
                    with st.expander(f"📊 {campaign.name}"):
                        display_campaign_metrics(campaign)
                        
                        # Add campaign controls
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Pause Campaign", key=f"pause_{campaign.id}"):
                                toggle_campaign_status(session, campaign.id, False)
                        with col2:
                            if st.button("Archive Campaign", key=f"archive_{campaign.id}"):
                                archive_campaign(session, campaign.id)

def knowledge_base_page():
    st.title("Knowledge Base")
    
    with db_session() as session:
        # Add KB version control
        current_version = get_current_kb_version(session)
        
        with st.form("kb_form"):
            # Add content sections
            sections = ['Company Info', 'Products', 'Target Market', 'Value Proposition']
            kb_content = {}
            
            for section in sections:
                st.subheader(section)
                kb_content[section] = st.text_area(
                    f"{section} Content",
                    value=current_version.get(section, ''),
                    height=200
                )
            
            # Add version tracking
            version_note = st.text_input("Version Notes")
            
            if st.form_submit_button("Save Knowledge Base"):
                try:
                    save_kb_version(session, kb_content, version_note)
                    st.success("Knowledge base updated!")
                except Exception as e:
                    st.error(f"Failed to save: {str(e)}")
        
        # Display version history
        with st.expander("Version History"):
            versions = fetch_kb_versions(session)
            for version in versions:
                st.text(f"Version {version.id}: {version.created_at}")
                if st.button("Restore", key=f"restore_{version.id}"):
                    restore_kb_version(session, version.id)

def autoclient_ai_page():
    st.title("AutoclientAI Control")
    
    with db_session() as session:
        # Add system status monitoring
        system_status = get_system_status(session)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("System Status", system_status['status'])
        col2.metric("Active Tasks", system_status['active_tasks'])
        col3.metric("Queue Size", system_status['queue_size'])
        
        # Add task management
        with st.form("automation_form"):
            task_type = st.selectbox(
                "Task Type",
                ["Lead Generation", "Email Campaign", "Content Generation"]
            )
            
            priority = st.slider("Priority", 1, 5, 3)
            
            if st.form_submit_button("Schedule Task"):
                try:
                    schedule_ai_task(session, task_type, priority)
                    st.success("Task scheduled!")
                except Exception as e:
                    st.error(f"Failed to schedule task: {str(e)}")
        
        # Add monitoring dashboard
        st.subheader("Task Monitor")
        tasks = fetch_active_tasks(session)
        if tasks:
            st.dataframe(
                pd.DataFrame(tasks),
                hide_index=True,
                use_container_width=True
            )

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
        # Add resource monitoring
        resources = get_system_resources()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("CPU Usage", f"{resources['cpu_usage']}%")
        col2.metric("Memory Usage", f"{resources['memory_usage']}%")
        col3.metric("Disk Usage", f"{resources['disk_usage']}%")
        
        # Add process management
        st.subheader("Active Processes")
        processes = fetch_active_processes(session)
        
        for process in processes:
            with st.expander(f"Process {process.id}"):
                st.text(f"Status: {process.status}")
                st.progress(process.progress)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Pause", key=f"pause_{process.id}"):
                        toggle_process(session, process.id, "pause")
                with col2:
                    if st.button("Stop", key=f"stop_{process.id}"):
                        toggle_process(session, process.id, "stop")
        
        # Add alert configuration
        with st.form("alert_config"):
            st.subheader("Alert Configuration")
            cpu_threshold = st.slider("CPU Alert Threshold", 0, 100, 80)
            memory_threshold = st.slider("Memory Alert Threshold", 0, 100, 80)
            
            if st.form_submit_button("Save Alert Config"):
                save_alert_config(session, cpu_threshold, memory_threshold)

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
    BATCH_SIZE = 50
    for batch in [leads[i:i + BATCH_SIZE] for i in range(0, len(leads), BATCH_SIZE)]:
        try:
            with db_session() as session:
                # Process batch
                time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            continue

    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        logging.error(f"Email template with ID {template_id} not found.")
        return [], 0  # No exception handling for failed template lookup

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
                log_message = f" Failed to send email to: {lead['Email']}"
            
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
    """Wrap email body content with HTML template"""
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
            .button {{
                display: inline-block;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }}
            .signature {{
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #eee;
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
        email_campaigns = (
            session.query(EmailCampaign)
            .join(Lead)
            .join(EmailTemplate)
            .options(
                joinedload(EmailCampaign.lead),
                joinedload(EmailCampaign.template)
            )
            .order_by(EmailCampaign.sent_at.desc())
            .all()
        )
        
        return pd.DataFrame({
            'ID': [ec.id for ec in email_campaigns],
            'Sent At': [ec.sent_at.strftime("%Y-%m-%d %H:%M:%S") if ec.sent_at else "" for ec in email_campaigns],
            'Email': [ec.lead.email for ec in email_campaigns],
            'Template': [ec.template.template_name for ec in email_campaigns],
            'Subject': [ec.customized_subject or "No subject" for ec in email_campaigns],
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

def main():
    # Initialize session state
    if 'automation_status' not in st.session_state:
        st.session_state.automation_status = False
    if 'automation_logs' not in st.session_state:
        st.session_state.automation_logs = []
    if 'search_terms' not in st.session_state:
        st.session_state.search_terms = []
    if 'optimized_terms' not in st.session_state:
        st.session_state.optimized_terms = []

    st.set_page_config(
        page_title="Autoclient.ai | Lead Generation AI App",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=""
    )

    st.sidebar.title("AutoclientAI")
    st.sidebar.markdown("Select a page to navigate through the application.")

    pages = {
        "🔍 Manual Search": manual_search_page,
        "📦 Bulk Send": bulk_send_page,
        "👥 View Leads": view_leads_page,
        "🔑 Search Terms": search_terms_page,
        "✉️ Email Templates": email_templates_page,
        "🚀 Projects & Campaigns": projects_campaigns_page,
        "📚 Knowledge Base": knowledge_base_page,
        "🤖 AutoclientAI": autoclient_ai_page,
        "⚙️ Automation Control": automation_control_panel_page,
        "📨 Email Logs": view_campaign_logs,
        "🔄 Settings": settings_page,
        "📨 Sent Campaigns": view_sent_email_campaigns
    }

    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=list(pages.keys()),
            icons=["search", "send", "people", "key", "envelope", "folder", "book", "robot", "gear", "list-check", "gear", "envelope-open"],
            menu_icon="cast",
            default_index=0
        )

    try:
        with db_session() as session:
            pages[selected]()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("An error occurred in the main function")
        st.write("Please try refreshing the page or contact support if the issue persists.")

    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")


def update_process_log(session, process_id, message, level='info'):
    """Update the logs for a search process"""
    try:
        process = session.query(SearchProcess).get(process_id)
        if not process:
            logging.error(f"Process {process_id} not found")
            return False
            
        # Initialize logs array if None
        if process.logs is None:
            process.logs = []
            
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message
        }
        
        # Append new log entry
        process.logs.append(log_entry)
        
        # Update the process
        session.add(process)
        session.commit()
        
        return True
    except Exception as e:
        logging.error(f"Error updating process log: {str(e)}")
        session.rollback()
        return False
def display_process_logs(process_id):
    """Display logs for a search process"""
    with db_session() as session:
        process = session.query(SearchProcess).get(process_id)
        if not process:
            st.info("Process not found")
            return
            
        if not process.logs:
            st.info("No logs available yet. Logs will appear here as the process runs.")
            return
        
        log_container = st.container()
        
        with log_container:
            st.markdown("""
                <style>
                    .process-logs-container {
                        max-height: 400px;
                        overflow-y: auto;
                        border: 1px solid rgba(49, 51, 63, 0.2);
                        border-radius: 0.25rem;
                        padding: 1rem;
                        background-color: rgba(49, 51, 63, 0.1);
                        margin-bottom: 1rem;
                        font-family: monospace;
                    }
                    .process-log-entry {
                        padding: 0.25rem 0;
                        border-bottom: 1px solid rgba(49, 51, 63, 0.1);
                        animation: fadeIn 0.5s ease-in;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(-10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                </style>
            """, unsafe_allow_html=True)
                
            # Format logs with icons
            log_entries = []
            for log in process.logs:
                timestamp = datetime.fromisoformat(log['timestamp']).strftime('%H:%M:%S')
                level = log['level']
                message = log['message']
                icon = {
                    'info': '🔵',
                    'success': '🟢',
                    'warning': '🟠',
                    'error': '🔴',
                    'email_sent': '🟣'
                }.get(level, '⚪')
                log_entries.append(f'<div class="process-log-entry">{icon} [{timestamp}] {message}</div>')
                
            # Display all logs at once in the container
            st.markdown(
                f"""
                <div class="process-logs-container" id="process-logs-{process_id}">
                    {"".join(log_entries)}
                </div>
                <script>
                    function scrollToBottom(containerId) {{
                        const container = document.getElementById(containerId);
                        if (container) {{
                            container.scrollTop = container.scrollHeight;
                        }}
                    }}
                    
                    // Initial scroll
                    scrollToBottom("process-logs-{process_id}");
                    
                    // Set up a mutation observer to watch for changes
                    const observer = new MutationObserver(() => scrollToBottom("process-logs-{process_id}"));
                    const container = document.getElementById("process-logs-{process_id}");
                    if (container) {{
                        observer.observe(container, {{ childList: true, subtree: true }});
                    }}
                </script>
                """,
                unsafe_allow_html=True
            )

def update_process_state(session, process_id, status, error=None):
    try:
        process = session.query(SearchProcess).get(process_id)
        if process:
            process.status = status
            process.error = error
            process.updated_at = datetime.utcnow()
            session.commit()
            return True
    except Exception as e:
        logging.error(f"Error updating process state: {str(e)}")
        session.rollback()
        return False

# Add template versioning and locking
class TemplateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._template_locks = {}
        
    def get_template_lock(self, template_id):
        with self._lock:
            if template_id not in self._template_locks:
                self._template_locks[template_id] = threading.Lock()
            return self._template_locks[template_id]
            
    def update_template(self, session, template_id, updates):
        lock = self.get_template_lock(template_id)
        with lock:
            template = session.query(EmailTemplate).filter_by(id=template_id).with_for_update().first()
            if template:
                for key, value in updates.items():
                    setattr(template, key, value)
                session.commit()
                return True
        return False

template_manager = TemplateManager()

# Update background process management
class ProcessManager:
    def __init__(self):
        self._processes = {}
        self._lock = threading.Lock()
        self._sessions = {}  # Add session tracking
        
    def start_process(self, process_id, target, args):
        with self._lock:
            if process_id in self._processes:
                # Clean up existing process if it's dead
                if not self._processes[process_id].is_alive():
                    del self._processes[process_id]
                else:
                    return False
            process = threading.Thread(target=target, args=args, daemon=True)
            self._processes[process_id] = process
            process.start()
            return True
            
    def stop_process(self, process_id):
        with self._lock:
            if process_id in self._processes:
                # Mark the process for cleanup
                self._processes[process_id] = None
                return True
        return False
        
    def cleanup_finished(self):
        with self._lock:
            dead_processes = [pid for pid, p in self._processes.items() 
                            if p is None or not p.is_alive()]
            for pid in dead_processes:
                if pid in self._sessions:
                    try:
                        self._sessions[pid].close()
                    except Exception:
                        logger.error(f"Error closing session for process {pid}")
                    finally:
                        del self._sessions[pid]
                del self._processes[pid]

process_manager = ProcessManager()

# Update the background manual search function
def background_manual_search(process_id, search_terms, settings):
    try:
        with get_db() as session:
            process = session.query(SearchProcess).get(process_id)
            if not process:
                return
                
            process.status = 'running'
            process.started_at = datetime.utcnow()
            session.commit()
            
            results = manual_search(
                session,
                search_terms,
                settings['num_results'],
                settings['ignore_previously_fetched'],
                settings['optimize_english'],
                settings['optimize_spanish'],
                settings['shuffle_keywords_option'],
                settings['language'],
                settings['enable_email_sending'],
                None,
                settings.get('from_email'),
                settings.get('reply_to'),
                settings.get('email_template'),
                process_id
            )
            
            process.results = results
            process.status = 'completed'
            process.completed_at = datetime.utcnow()
            session.commit()
            
    except Exception as e:
        with get_db() as session:
            process = session.query(SearchProcess).get(process_id)
            if process:
                process.status = 'failed'
                process.error = str(e)
                session.commit()
    finally:
        process_manager.cleanup_finished()

# Update email template handling
def save_email_template(session, template_id, updates):
    return template_manager.update_template(session, template_id, updates)

def get_email_template(session, template_id):
    lock = template_manager.get_template_lock(template_id)
    with lock:
        return session.query(EmailTemplate).get(template_id)

def optimize_search_terms_page():
    st.title("Search Terms Optimization")
    
    with db_session() as session:
        # Get knowledge base info
        kb_info = get_knowledge_base_info(session, get_active_project_id())
        if not kb_info:
            st.error("Please set up your Knowledge Base first")
            return
            
        st.subheader("Current Search Term Groups")
        
        # Display existing groups
        groups = session.query(SearchTermGroup).all()
        if groups:
            for group in groups:
                with st.expander(f"📁 {group.name}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {group.description}")
                        terms = session.query(SearchTerm).filter_by(group_id=group.id).all()
                        if terms:
                            st.markdown("**Terms:**")
                            for term in terms:
                                st.markdown(f"- {term.term}")
                        
                        # Show associated template if exists
                        template = session.query(EmailTemplate).filter_by(campaign_id=get_active_campaign_id()).first()
                        if template:
                            st.markdown("**Email Template:**")
                            st.markdown(f"*{template.template_name}*")
                            with st.expander("Preview Template"):
                                st.text(f"Subject: {template.subject}")
                                st.markdown(template.body_content)
                    
                    with col2:
                        if st.button("🔄 Optimize", key=f"opt_{group.id}"):
                            with st.spinner("Optimizing terms..."):
                                try:
                                    new_terms = generate_optimized_search_terms(
                                        session,
                                        [term.term for term in terms],
                                        kb_info
                                    )
                                    # Update terms
                                    for term in new_terms:
                                        session.add(SearchTerm(
                                            term=term,
                                            group_id=group.id,
                                            campaign_id=get_active_campaign_id()
                                        ))
                                    session.commit()
                                    st.success("Terms optimized!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error optimizing terms: {str(e)}")
        
        # Generate new groups
        st.subheader("Generate New Search Term Groups")
        col1, col2 = st.columns(2)
        
        with col1:
            industry_focus = st.text_input("Industry Focus", 
                help="Specific industry or vertical to target")
            
        with col2:
            target_market = st.text_input("Target Market",
                help="Specific market segment or audience")
            
        if st.button("Generate Groups and Templates", type="primary"):
            with st.spinner("Generating optimized search terms and templates..."):
                try:
                    results = generate_search_term_groups_and_templates(
                        session,
                        kb_info,
                        industry_focus,
                        target_market
                    )
                    
                    st.success("Successfully generated new groups and templates!")
                    
                    # Show results
                    st.subheader("Generated Content")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### New Groups")
                        for group in results["groups"]:
                            st.markdown(f"**{group['name']}**")
                            st.markdown("Terms:")
                            for term in group["terms"]:
                                st.markdown(f"- {term}")
                                
                    with col2:
                        st.markdown("### New Templates")
                        for template in results["templates"]:
                            st.markdown(f"**{template['name']}**")
                            st.markdown(f"Subject: {template['subject']}")
                    
                except Exception as e:
                    st.error(f"Error generating content: {str(e)}")

def generate_or_adjust_email_template(prompt, kb_info=None, current_template=None):
    try:
        context = {
            "prompt": prompt,
            "knowledge_base": kb_info,
            "current_template": current_template
        }
        messages = [
            {"role": "system", "content": "You are an expert email copywriter specializing in B2B communication."},
            {"role": "user", "content": f"Based on this context:\n{json.dumps(context, indent=2)}\n\nGenerate a professional email template with:\n1. Subject line\n2. Body content\n\nRespond with JSON:\n{{\n    \"subject\": \"email subject\",\n    \"body\": \"email body\"\n}}"}
        ]
        response = openai_chat_completion(messages, temperature=0.7)
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return {"subject": "AI Generated Subject", "body": response}  # Potential XSS vulnerability
        return response
    except Exception as e:
        logger.error(f"Error generating email template: {str(e)}")
        return {"subject": "Default Subject", "body": "Default body content"}

def generate_search_term_groups_and_templates(session, kb_info, industry_focus=None, target_market=None):
    try:
        context = {
            "company_info": {
                "description": kb_info.get('company_description', ''),
                "mission": kb_info.get('company_mission', ''),
                "target_market": kb_info.get('company_target_market', ''),
                "product": kb_info.get('product_description', '')
            },
            "industry_focus": industry_focus,
            "target_market": target_market
        }
        prompt = f"Based on this business context:\n{json.dumps(context, indent=2)}\n\nGenerate:\n1. Search term groups with relevant terms for lead generation\n2. Email template variations for each group"
        response = openai_chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert in B2B lead generation and email marketing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        if not response:
            raise ValueError("Failed to generate search terms and templates")
        content = response if isinstance(response, dict) else json.loads(response)
        results = {"groups": [], "templates": []}
        for group_data in content.get("groups", []):
            group = SearchTermGroup(
                name=group_data["name"],
                description=group_data["description"],
                created_at=datetime.utcnow()
            )
            session.add(group)
            session.flush()
            for term in group_data["search_terms"]:
                search_term = SearchTerm(
                    term=term,
                    group_id=group.id,
                    campaign_id=get_active_campaign_id(),
                    created_at=datetime.utcnow()
                )
                session.add(search_term)
            template_data = group_data["email_template"]
            template = EmailTemplate(
                template_name=template_data["name"],
                subject=template_data["subject"],
                body_content=template_data["body"],
                campaign_id=get_active_campaign_id(),
                is_ai_customizable=True,
                created_at=datetime.utcnow()
            )
            session.add(template)
            results["groups"].append({
                "id": group.id,
                "name": group.name,
                "terms": group_data["search_terms"]
            })
            results["templates"].append({
                "name": template_data["name"],
                "subject": template_data["subject"]
            })
        session.commit()
        return results
    except Exception as e:
        session.rollback()
        logger.error(f"Error generating search terms and templates: {str(e)}")
        raise

def fetch_leads_for_search_term_groups(session, groups):
    try:
        logger.info(f"Fetching leads for groups: {groups}")
        query = (
            session.query(Lead)
            .join(CampaignLead)
            .join(SearchTerm)
            .join(SearchTermGroup)
            .filter(SearchTermGroup.id.in_(groups))
            .distinct()
        )
        return query.all()
    except Exception as e:
        logger.error(f"Error fetching leads for groups: {str(e)}")
        return []

def log_error(message, process_id=None, log_container=None):
    if process_id:
        with get_db() as session:
            update_process_log(session, process_id, message, 'error')
    elif log_container:
        update_log(log_container, message, 'error')
    logger.error(message)

# Add connection pooling helper
def get_db_connection():
    """Get a database connection from the pool"""
    try:
        connection = engine.connect()
        return connection
    except Exception as e:
        logging.error(f"Error getting database connection: {str(e)}")
        raise

# Add connection cleanup on app shutdown
def cleanup_connections():
    """Clean up database connections when the app shuts down"""
    try:
        engine.dispose()
    except Exception as e:
        logger.error(f"Error disposing engine connections: {str(e)}")

# Register cleanup handler
import atexit
atexit.register(cleanup_connections)

if __name__ == "__main__":
    main()
