# Import streamlit first
import streamlit as st

# Set page config at the very top
st.set_page_config(
    page_title="Email Lead Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard library imports
import asyncio
import atexit
import html
import json
import logging
import multiprocessing
import os
import random
import re
import smtplib
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from multiprocessing import Process
from typing import List, Optional, TYPE_CHECKING
from urllib.parse import urlparse, urlencode

# Third-party imports
import aiohttp
import boto3
import openai
import pandas as pd
import plotly.express as px
import requests
import urllib3
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from email_validator import validate_email, EmailNotValidError
from fake_useragent import UserAgent
from googlesearch import search as google_search
from openai import OpenAI
from requests.adapters import HTTPAdapter
from sqlalchemy import (
    and_, Boolean, BigInteger, Column, create_engine, DateTime, distinct,
    Float, ForeignKey, func, Index, Integer, JSON, or_, select, String, Table,
    Text, text, TIMESTAMP
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, joinedload, relationship, Session, sessionmaker, configure_mappers
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential
from urllib3.util import Retry
from sqlalchemy import inspect, MetaData

# Initialize SQLAlchemy Base with shared metadata
metadata = MetaData()
Base = declarative_base(metadata=metadata)

@st.cache_resource
def get_base():
    metadata = MetaData()
    Base = declarative_base(metadata=metadata)
    return Base

class Project(Base):
    __tablename__ = 'projects'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    
    # Add caching for relationship loading
    @st.cache_data
    def get_campaigns(self):
        return self.campaigns
    
    campaigns = relationship("Campaign", back_populates="project")
    knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False)

class Campaign(Base):
    __tablename__ = 'campaigns'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    campaign_name = Column(Text, nullable=False)
    campaign_type = Column(Text, nullable=False)
    project_id = Column(BigInteger, ForeignKey('projects.id'), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    auto_send = Column(Boolean, server_default='false')
    loop_automation = Column(Boolean, server_default='false')
    ai_customization = Column(Boolean, server_default='false')
    max_emails_per_group = Column(BigInteger, server_default='50')
    loop_interval = Column(BigInteger, server_default='60')
    
    # Add caching for metrics access
    @st.cache_data
    def get_metrics(self):
        return self.metrics
    
    metrics = Column(JSONB)
    
    # Cache relationship loading
    @st.cache_data
    def get_email_campaigns(self):
        return self.email_campaigns
    
    # Relationships
    project = relationship("Project", back_populates="campaigns")
    email_campaigns = relationship("EmailCampaign", back_populates="campaign", cascade="all, delete-orphan")
    search_terms = relationship("SearchTerm", back_populates="campaign", cascade="all, delete-orphan")
    campaign_leads = relationship("CampaignLead", back_populates="campaign", cascade="all, delete-orphan")
    email_templates = relationship("EmailTemplate", back_populates="campaign", cascade="all, delete-orphan")

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    term = Column(Text)
    category = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    language = Column(Text)
    campaign = relationship("Campaign", back_populates="search_terms")
    lead_sources = relationship("LeadSource", back_populates="search_term")
    effectiveness = relationship("SearchTermEffectiveness", back_populates="search_term", uselist=False)

class Lead(Base):
    __tablename__ = 'leads'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, nullable=False, unique=True)
    phone = Column(Text)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text)
    job_title = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead")
    email_campaigns = relationship("EmailCampaign", back_populates="lead")

class LeadSource(Base):
    __tablename__ = 'lead_sources'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'), nullable=False)
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'), nullable=False)
    url = Column(Text, nullable=False)
    domain = Column(Text)
    page_title = Column(Text)
    meta_description = Column(Text)
    scrape_duration = Column(Text)
    meta_tags = Column(Text)
    phone_numbers = Column(Text)
    content = Column(Text)
    tags = Column(Text)
    http_status = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    lead = relationship("Lead", back_populates="lead_sources")
    search_term = relationship("SearchTerm", back_populates="lead_sources")

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    template_name = Column(Text, nullable=False)
    subject = Column(Text, nullable=False)
    body_content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    is_ai_customizable = Column(Boolean, server_default='false')
    language = Column(Text, server_default='ES')
    campaign = relationship("Campaign", back_populates="email_templates")
    email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    __table_args__ = {'extend_existing': True}
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
    ai_customized = Column(Boolean)
    opened_at = Column(DateTime(timezone=True))
    clicked_at = Column(DateTime(timezone=True))
    open_count = Column(BigInteger)
    click_count = Column(BigInteger)
    campaign = relationship("Campaign", back_populates="email_campaigns")
    lead = relationship("Lead", back_populates="email_campaigns")
    template = relationship("EmailTemplate", back_populates="email_campaigns")

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    __table_args__ = {'extend_existing': True}
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
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    project = relationship("Project", back_populates="knowledge_base")

class CampaignLead(Base):
    __tablename__ = 'campaign_leads'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    status = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    lead = relationship("Lead", back_populates="campaign_leads")
    campaign = relationship("Campaign", back_populates="campaign_leads")

class AIRequestLogs(Base):
    __tablename__ = 'ai_request_logs'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    function_name = Column(Text)
    prompt = Column(Text)
    response = Column(Text)
    model_used = Column(Text)
    created_at = Column(DateTime(timezone=True))
    lead_id = Column(BigInteger)
    email_campaign_id = Column(BigInteger)

class AutomationLogs(Base):
    __tablename__ = 'automation_logs'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger)
    search_term_id = Column(BigInteger)
    leads_gathered = Column(BigInteger)
    emails_sent = Column(BigInteger)
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    status = Column(Text)
    logs = Column(JSON)

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    __table_args__ = {'extend_existing': True}
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
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))

class BackgroundProcessState(Base):
    __tablename__ = 'background_process_state'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    process_id = Column(Text, unique=True)
    status = Column(Text)
    started_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=text('NOW()'))
    completed_at = Column(TIMESTAMP(timezone=True))
    error_message = Column(Text)
    progress = Column(Float)
    total_items = Column(Integer)
    processed_items = Column(Integer)

class OptimizedSearchTerms(Base):
    __tablename__ = 'optimized_search_terms'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    original_term_id = Column(BigInteger)
    term = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True))

class SearchGroups(Base):
    __tablename__ = 'search_groups'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    emails_sent = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True))
    updated_at = Column(TIMESTAMP(timezone=True))

class SearchTermGroup(Base):
    __tablename__ = 'search_term_groups'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    email_template = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(DateTime(timezone=True))

class SearchTermEffectiveness(Base):
    __tablename__ = 'search_term_effectiveness'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    effectiveness_score = Column(Float)
    total_leads = Column(Integer)
    valid_leads = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'))
    search_term = relationship("SearchTerm", back_populates="effectiveness")

class Settings(Base):
    __tablename__ = 'settings'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    setting_type = Column(Text, nullable=False)
    value = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(DateTime(timezone=True))

DB_HOST = os.getenv("SUPABASE_DB_HOST", "aws-0-eu-central-1.pooler.supabase.com")
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres.whwiyccyyfltobvqxiib")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "SamiHalawa1996")
DB_PORT = os.getenv("SUPABASE_DB_PORT", "6543")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
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

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)

@contextmanager
def db_session():
    """Provide a transactional scope around a series of operations"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        session.rollback()
        logging.error(f"Unexpected error in database session: {str(e)}")
        raise
    finally:
        if session:
            try:
                # Ensure all objects are detached before closing
                session.expunge_all()
            except:
                pass
            session.close()

class ProcessManager:
    def __init__(self):
        self._processes = {}
        self._lock = threading.Lock()
        self._cleanup_interval = 3600  # Cleanup every hour
        self._last_cleanup = time.time()
        
    def _cleanup_old_processes(self):
        """Clean up old process records and terminated processes"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
            
        with self._lock:
            # Clean up process dictionary
            terminated_processes = []
            for pid, process in self._processes.items():
                if not process.is_alive():
                    terminated_processes.append(pid)
            
            for pid in terminated_processes:
                del self._processes[pid]
            
            # Clean up database records
            with db_session() as session:
                old_processes = session.query(BackgroundProcessState).filter(
                    or_(
                        BackgroundProcessState.status.in_(['completed', 'failed']),
                        BackgroundProcessState.updated_at < datetime.utcnow() - timedelta(days=7)
                    )
                ).all()
                
                for process in old_processes:
                    session.delete(process)
                session.commit()
            
            self._last_cleanup = current_time
        
    def start_process(self, process_id: str, target: callable, args: tuple) -> bool:
        self._cleanup_old_processes()
        
        with self._lock:
            if process_id in self._processes and self._processes[process_id].is_alive():
                return False
                
            process = Process(target=self._wrapped_target, args=(process_id, target, args))
            process.daemon = True
            process.start()
            self._processes[process_id] = process
            
            with db_session() as session:
                process_state = BackgroundProcessState(
                    process_id=process_id,
                    status='running',
                    started_at=datetime.utcnow(),
                    progress=0.0,
                    total_items=0,
                    processed_items=0
                )
                session.add(process_state)
                session.commit()
            return True
    
    def _wrapped_target(self, process_id: str, target: callable, args: tuple):
        """Wrapper to handle process state updates and error handling"""
        try:
            with db_session() as session:
                process_state = session.query(BackgroundProcessState)\
                    .filter_by(process_id=process_id).first()
                if process_state:
                    process_state.status = 'running'
                    session.commit()
            
            # Run the actual target function
            result = target(*args)
            
            with db_session() as session:
                process_state = session.query(BackgroundProcessState)\
                    .filter_by(process_id=process_id).first()
                if process_state:
                    process_state.status = 'completed'
                    process_state.completed_at = datetime.utcnow()
                    process_state.progress = 100.0
                    session.commit()
            
            return result
            
        except Exception as e:
            logging.error(f"Process {process_id} failed: {str(e)}")
            with db_session() as session:
                process_state = session.query(BackgroundProcessState)\
                    .filter_by(process_id=process_id).first()
                if process_state:
                    process_state.status = 'failed'
                    process_state.error_message = str(e)
                    process_state.completed_at = datetime.utcnow()
                    session.commit()
            raise
    
    def stop_process(self, process_id: str) -> bool:
        with self._lock:
            process = self._processes.get(process_id)
            if process and process.is_alive():
                process.terminate()
                process.join(timeout=5)
                
                with db_session() as session:
                    process_state = session.query(BackgroundProcessState)\
                        .filter_by(process_id=process_id).first()
                    if process_state:
                        process_state.status = 'terminated'
                        process_state.completed_at = datetime.utcnow()
                        session.commit()
                
                del self._processes[process_id]
                return True
            return False
    
    def get_process_state(self, process_id: str) -> Optional[dict]:
        """Get current state of a process"""
        with db_session() as session:
            state = session.query(BackgroundProcessState)\
                .filter_by(process_id=process_id).first()
            if state:
                return {
                    'status': state.status,
                    'progress': state.progress,
                    'error': state.error_message,
                    'started_at': state.started_at,
                    'completed_at': state.completed_at,
                    'processed_items': state.processed_items,
                    'total_items': state.total_items
                }
        return None
    
    def update_process_progress(self, process_id: str, progress: float, 
                              processed_items: int = None, total_items: int = None,
                              status: str = None, error: str = None):
        """Update progress of a running process"""
        with db_session() as session:
            state = session.query(BackgroundProcessState)\
                .filter_by(process_id=process_id).first()
            if state:
                state.progress = progress
                if processed_items is not None:
                    state.processed_items = processed_items
                if total_items is not None:
                    state.total_items = total_items
                if status:
                    state.status = status
                if error:
                    state.error_message = error
                state.updated_at = datetime.utcnow()
                session.commit()

def init_db():
    """Initialize database safely without dropping any data"""
    try:
        # Configure all mappers before creating tables
        configure_mappers()
        
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        # First create any missing tables
        Base.metadata.create_all(engine, checkfirst=True)
        
        # Then check and add any missing columns
        for table_name in existing_tables:
            if table_name not in Base.metadata.tables:
                continue
                
            existing_columns = {col['name'] for col in inspector.get_columns(table_name)}
            model_columns = {col.name for col in Base.metadata.tables[table_name].columns}
            
            missing_columns = model_columns - existing_columns
            
            if missing_columns:
                with engine.begin() as connection:
                    for col_name in missing_columns:
                        col = Base.metadata.tables[table_name].columns[col_name]
                        # Add column with default value if specified
                        default_value = col.server_default.arg if col.server_default else 'NULL'
                        connection.execute(text(
                            f'ALTER TABLE {table_name} ADD COLUMN {col_name} {col.type} DEFAULT {default_value}'
                        ))
                logging.info(f"Added missing columns {missing_columns} to table {table_name}")
                
        logging.info("Database schema synchronized successfully")
        
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise

def delete_email_setting(setting_id):
    with db_session() as session:
        setting = session.query(EmailSettings).get(setting_id)
        if setting:
            session.delete(setting)
            session.commit()
            st.success(f"Deleted {setting.name}")
            st.session_state.edit_id = "New Setting"
        else:
            st.error(f"Setting with id {setting_id} not found")

def update_edit_id(selected_option):
    st.session_state.edit_id = selected_option

def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings: return None, None
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
    if email_settings.provider == 'ses':
        if not ses_client:
            aws_session = boto3.Session(aws_access_key_id=email_settings.aws_access_key_id, aws_secret_access_key=email_settings.aws_secret_access_key, region_name=email_settings.aws_region)
            ses_client = aws_session.client('ses')
        response = ses_client.send_email(Source=from_email, Destination={'ToAddresses': [to_email]}, Message={'Subject': {'Data': subject, 'Charset': charset}, 'Body': {'Html': {'Data': tracked_body, 'Charset': charset}}}, ReplyToAddresses=[reply_to] if reply_to else [])
        return response, tracking_id
    elif email_settings.provider == 'smtp':
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = from_email, to_email, subject
        if reply_to: msg['Reply-To'] = reply_to
        msg.attach(MIMEText(tracked_body, 'html'))
        with smtplib.SMTP(email_settings.smtp_server, email_settings.smtp_port) as server:
            server.starttls()
            server.login(email_settings.smtp_username, email_settings.smtp_password)
            server.send_message(msg)
        return {'MessageId': f'smtp-{uuid.uuid4()}'}, tracking_id
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

def update_log(log_container, message, level='info', process_id=None):
    icon = {
        'info': '🔵',
        'success': '🟢',
        'warning': '🟠',
        'error': '🔴',
        'email_sent': '🟣'
    }.get(level, '⚪')
    
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
    return f'"{search_term}" email OR contact OR "get in touch" site:.com' if language == 'english' else f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es' if language == 'spanish' else search_term

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def is_valid_email(email):
    try:
        validate_email(email)
        return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email))
    except EmailNotValidError:
        return False

def get_domain_from_url(url): return urlparse(url).netloc

def extract_emails_from_html(html_content):
    return re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', html_content)

def extract_info_from_page(soup):
    name = soup.find('meta', {'name': 'author'})
    name = name['content'] if name else ''
    company = soup.find('meta', {'property': 'og:site_name'})
    company = company['content'] if company else ''
    job_title = soup.find('meta', {'name': 'job_title'})
    job_title = job_title['content'] if job_title else ''
    return name, company, job_title

def manual_search(session, terms, max_results=10, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False, language="ES"):
    """Perform manual search with the given parameters"""
    results = {'results': [], 'errors': []}
    
    for term in terms:
        try:
            search_term = term
            if optimize_english and language == "EN":
                search_term = optimize_search_term(term, "english")
            elif optimize_spanish and language == "ES":
                search_term = optimize_search_term(term, "spanish")
            
            if shuffle_keywords_option:
                search_term = shuffle_keywords(search_term)
            
            for url in google_search(search_term, stop=max_results):
                if ignore_previously_fetched and session.query(LeadSource).filter_by(url=url).first():
                    continue
                    
                response = requests.get(url, timeout=10, verify=False)
                soup = BeautifulSoup(response.text, 'html.parser')
                emails = extract_emails_from_html(response.text)
                
                if not emails:
                    continue
                    
                name, company, job_title = extract_info_from_page(soup)
                for email in emails:
                    if not is_valid_email(email):
                        continue
                        
                    lead = save_lead(session, email, name, None, company, job_title, url=url)
                    if lead:
                        results['results'].append({'Email': email, 'URL': url})
                        
        except Exception as e:
            results['errors'].append(str(e))
            
    return results

def generate_or_adjust_email_template(prompt, kb_info=None, current_template=None):
    """Generate or adjust email template using AI with knowledge base context"""
    if not kb_info:
        return current_template or prompt

    # Enhance prompt with knowledge base context
    enhanced_prompt = f"""
    Context:
    - Tone of Voice: {kb_info.get('tone_of_voice', 'Professional')}
    - Communication Style: {kb_info.get('communication_style', 'Direct')}
    - Company Description: {kb_info.get('company_description', '')}
    - Target Market: {kb_info.get('company_target_market', '')}
    
    Original Template or Prompt:
    {prompt}
    
    Please generate an email that:
    1. Matches the specified tone of voice and communication style
    2. Clearly communicates the company's value proposition
    3. Resonates with the target market
    4. Maintains professionalism while being engaging
    """
    
    messages = [
        {"role": "system", "content": "You are an expert email copywriter."},
        {"role": "user", "content": enhanced_prompt}
    ]
    
    response = openai_chat_completion(messages)
    
    if not response:
        return current_template or prompt
        
    return response

def fetch_leads_with_sources(session):
    query = session.query(
        Lead.id, 
        Lead.email,
        Lead.phone,
        Lead.first_name,
        Lead.last_name,
        Lead.company,
        Lead.job_title,
        Lead.created_at,
        func.string_agg(LeadSource.url, ', ').label('sources'),
        func.max(EmailCampaign.sent_at).label('last_contact'),
        func.string_agg(EmailCampaign.status, ', ').label('email_statuses')
    ).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(
        Lead.id,
        Lead.email,
        Lead.phone,
        Lead.first_name,
        Lead.last_name,
        Lead.company,
        Lead.job_title,
        Lead.created_at
    )
    return pd.DataFrame([{
        'id': id,
        'email': email,
        'phone': phone,
        'first_name': first_name,
        'last_name': last_name,
        'company': company,
        'job_title': job_title,
        'created_at': created_at,
        'Source': sources,
        'Last Contact': last_contact,
        'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted',
        'Delete': False
    } for id, email, phone, first_name, last_name, company, job_title, created_at, sources, last_contact, email_statuses in query.all()])

def fetch_email_settings(session):
    """Fetch email settings from the database"""
    settings = session.query(EmailSettings).all()
    return settings

def fetch_search_terms_with_lead_count(session):
    """Fetch search terms with lead count"""
    subquery = session.query(
        LeadSource.search_term_id,
        func.count(distinct(LeadSource.id)).label('lead_count')
    ).group_by(LeadSource.search_term_id).subquery()
    
    return session.query(SearchTerm, subquery.c.lead_count)\
        .outerjoin(subquery, SearchTerm.id == subquery.c.search_term_id).all()

def add_search_term(session, term, campaign_id):
    """Add a new search term to the database"""
    new_term = SearchTerm(term=term, campaign_id=campaign_id)
    session.add(new_term)
    session.commit()

def get_active_project_id(): return 1
def get_active_campaign_id():
    with db_session() as session:
        campaign = session.query(Campaign).filter_by(project_id=1).first()
        return campaign.id if campaign else 1

def set_active_project_id(project_id): st.session_state.__setitem__('active_project_id', project_id)
def set_active_campaign_id(campaign_id): st.session_state.__setitem__('active_campaign_id', campaign_id)

def create_search_term_group(session, name):
    group = SearchTermGroup(name=name)
    session.add(group)
    session.commit()

def update_search_term_group(session, group_id, term_ids):
    group = session.query(SearchTermGroup).get(group_id)
    if group:
        group.search_terms = []
        term_ids = [int(t.split(':')[0]) for t in term_ids]
        group.search_terms.extend(session.query(SearchTerm).filter(SearchTerm.id.in_(term_ids)).all())
        session.commit()

def delete_search_term_group(session, group_id):
    try:
        group = session.query(SearchTermGroup).get(group_id)
        if group: session.delete(group); session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting search term group: {str(e)}")

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

def ai_automation_loop(session, log_container, leads_container):
    automation_logs = []
    total_search_terms = 0
    total_emails_sent = 0
    progress_bar = st.progress(0)
    
    try:
        while st.session_state.get('automation_status', False):
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                automation_logs.append("Knowledge Base not found. Skipping cycle.")
                time.sleep(3600)
                continue

            base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
            if not base_terms:
                automation_logs.append("No search terms found. Skipping cycle.")
                time.sleep(3600)
                continue

            optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            if not optimized_terms:
                automation_logs.append("Failed to generate optimized terms. Skipping cycle.")
                time.sleep(3600)
                continue

            for idx, term in enumerate(optimized_terms):
                try:
                    results = manual_search(session, [term], 10)
                    if results and 'results' in results:
                        new_leads = []
                        for res in results['results']:
                            lead = save_lead(session, res['email'], url=res['url'])
                            if lead:
                                new_leads.append((lead.id, lead.email))
                        
                        if new_leads:
                            template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                            if template:
                                from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                                reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                                logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, 
                                                                  [{'Email': email} for _, email in new_leads])
                                automation_logs.extend(logs)
                                total_emails_sent += sent_count
                        
                        leads_container.text_area("New Leads Found", "\n".join([email for _, email in new_leads]), height=200)
                        progress_bar.progress((idx + 1) / len(optimized_terms))
                except Exception as e:
                    automation_logs.append(f"Error processing term {term}: {str(e)}")
                    continue

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
    existing_lead = session.query(Lead).filter_by(email=email).first()
    if existing_lead:
        for attr in ['first_name', 'last_name', 'company', 'job_title', 'phone', 'created_at']:
            if locals()[attr]: setattr(existing_lead, attr, locals()[attr])
        lead = existing_lead
    else:
        lead = Lead(email=email, first_name=first_name, last_name=last_name, company=company, job_title=job_title, phone=phone, created_at=created_at or datetime.utcnow())
        session.add(lead)
    session.flush()
    session.add(LeadSource(lead_id=lead.id, url=url, search_term_id=search_term_id))
    session.add(CampaignLead(campaign_id=get_active_campaign_id(), lead_id=lead.id, status="Not Contacted", created_at=datetime.utcnow()))
    session.commit()
    return lead

def save_lead_source(session, lead_id, search_term_id, url, http_status, scrape_duration, page_title=None, meta_description=None, content=None, tags=None, phone_numbers=None):
    session.add(LeadSource(lead_id=lead_id, search_term_id=search_term_id, url=url, http_status=http_status, scrape_duration=scrape_duration, page_title=page_title or get_page_title(url), meta_description=meta_description or get_page_description(url), content=content or extract_visible_text(BeautifulSoup(requests.get(url).text, 'html.parser')), tags=tags, phone_numbers=phone_numbers))
    session.commit()

def get_page_title(url):
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else None
        meta_title = soup.find('meta', property='og:title')
        return title.strip() if title else meta_title['content'].strip() if meta_title and meta_title.get('content') else "No title found"
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
    return session.query(EmailTemplate).all()

def create_or_update_email_template(session, template_name, subject, body_content, template_id=None, is_ai_customizable=False, created_at=None, language='ES', project_id=1):
    # Cache template in session state to avoid redundant DB writes
    if 'template_cache' not in st.session_state:
        st.session_state.template_cache = {}
        
    if template_id:
        # Update existing template
        template = session.query(EmailTemplate).get(template_id)
        if template:
            template.template_name = template_name
            template.subject = subject 
            template.body_content = body_content
            template.is_ai_customizable = is_ai_customizable
            template.language = language
    else:
        # Create new template
        template = EmailTemplate(
            template_name=template_name,
            subject=subject,
            body_content=body_content, 
            is_ai_customizable=is_ai_customizable,
            project_id=project_id,
            campaign_id=get_active_campaign_id(),
            created_at=created_at or datetime.utcnow(),
            language=language
        )
        session.add(template)

    session.commit()
    
    # Cache the template after commit
    st.session_state.template_cache[template.id] = {
        'template': template,
        'last_updated': datetime.utcnow()
    }
    return template.id

def safe_datetime_compare(date1, date2):
    """Compare two datetime objects safely with caching"""
    # Cache comparison results
    cache_key = f"{date1}_{date2}"
    if 'datetime_compare_cache' not in st.session_state:
        st.session_state.datetime_compare_cache = {}
    
    if cache_key in st.session_state.datetime_compare_cache:
        return st.session_state.datetime_compare_cache[cache_key]
        
    if date1 is None or date2 is None:
        result = False
    else:
        result = date1 > date2

    st.session_state.datetime_compare_cache[cache_key] = result
    return result

@st.cache_data(ttl=300)  # Cache results for 5 minutes
def fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted):
    """Fetch leads based on criteria with caching"""
    query = session.query(Lead)
    if specific_email:
        query = query.filter(Lead.email == specific_email)
    if selected_terms:
        query = query.join(LeadSource).filter(LeadSource.search_term_id.in_(selected_terms))
        if exclude_previously_contacted:
            contacted_leads = session.query(EmailCampaign.lead_id).filter(EmailCampaign.template_id == template_id)
            query = query.filter(~Lead.id.in_(contacted_leads))
    return query.all()

def update_display(container, items, title, item_key):
    # Initialize session state for caching if not exists
    if 'display_cache' not in st.session_state:
        st.session_state.display_cache = {}
        st.session_state.page_number = 1
        st.session_state.items_per_page = 20

    # Generate cache key from items
    cache_key = f"{title}_{len(items)}"
    
    # Check if we need to update the cache
    if (cache_key not in st.session_state.display_cache or 
        items != st.session_state.display_cache[cache_key]['items']):
        st.session_state.display_cache[cache_key] = {
            'items': items.copy(),
            'last_updated': datetime.now()
        }
    
    # Calculate pagination
    start_idx = (st.session_state.page_number - 1) * st.session_state.items_per_page
    end_idx = start_idx + st.session_state.items_per_page
    visible_items = items[start_idx:end_idx]
    
    # Render the HTML with pagination controls
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
            .pagination {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 1rem;
                padding: 0.5rem;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 0.25rem;
            }}
        </style>
        <div class="container">
            <h4>{title} ({len(items)})</h4>
            {"".join(f'<div class="entry">{item[item_key]}</div>' for item in visible_items)}
            <div class="pagination">
                <span>Page {st.session_state.page_number} of {max(1, (len(items) + st.session_state.items_per_page - 1) // st.session_state.items_per_page)}</span>
                <span>{start_idx + 1}-{min(end_idx, len(items))} of {len(items)} items</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add pagination controls
    col1, col2, col3 = container.columns([1, 3, 1])
    with col1:
        if st.button("Previous", key=f"prev_{title}") and st.session_state.page_number > 1:
            st.session_state.page_number -= 1
            st.rerun()
    with col3:
        max_pages = (len(items) + st.session_state.items_per_page - 1) // st.session_state.items_per_page
        if st.button("Next", key=f"next_{title}") and st.session_state.page_number < max_pages:
            st.session_state.page_number += 1
            st.rerun()

def manual_search_page():
    st.title("Manual Search")
    
    # Initialize containers and variables
    log_container = st.empty()
    results = []

    with db_session() as session:
        # Fetch recent searches within the session
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
            suggestions=['software engineer', 'data scientist', 'product manager'],
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

    email_template = None
    from_email = None
    reply_to = None

    if enable_email_sending:
        if not email_templates:
            st.error("No email templates available. Please create a template first.")
            return
        if not email_settings:
            st.error("No email settings available. Please add email settings first.")
            return

        col3, col4 = st.columns(2)
        with col3:
            email_template = st.selectbox("Email template", options=[f"{t.id}: {t.template_name}" for t in email_templates])
        with col4:
            email_setting = st.selectbox("From Email", options=[f"{s.id}: {s.email}" for s in email_settings])
            if email_setting:
                setting_id = int(email_setting.split(":")[0])
                setting = next((s for s in email_settings if s.id == setting_id), None)
                if setting:
                    from_email = setting.email
                    reply_to = st.text_input("Reply To", value=setting.email)

    # Initialize session state for caching if not exists
    if 'leads_cache' not in st.session_state:
        st.session_state.leads_cache = []
        st.session_state.last_search = None
        st.session_state.page_number = 1
    
    # Auto-refresh with optimized implementation
    auto_refresh = st.checkbox("Auto-refresh logs", value=True)
    if auto_refresh and st.session_state.get('last_refresh', 0) < time.time() - 2:
        st.session_state.last_refresh = time.time()
        st.rerun()

    # Create containers for efficient updates
    leads_container = st.empty()
    status_container = st.empty()
    
    # Cache results in session state
    leads_found = st.session_state.leads_cache
    emails_sent = []

    if st.button("Search"):
        if not search_terms:
            return st.warning("Enter at least one search term.")
            
        # Reset cache for new search
        st.session_state.leads_cache = []

        for i, term in enumerate(search_terms):
            status_container.text(f"Searching: '{term}' ({i+1}/{len(search_terms)})")

            with db_session() as session:
                # Cache results in session state
                term_results = manual_search(session, [term], num_results, 
                                          ignore_previously_fetched, optimize_english, 
                                          optimize_spanish, shuffle_keywords_option, 
                                          language, enable_email_sending, 
                                          log_container, from_email, reply_to, 
                                          email_template)
                
                st.session_state.leads_cache.extend([
                    f"{res['Email']} - {res['Company']}" 
                    for res in term_results['results']
                ])
                results.extend(term_results['results'])

                # Email sending logic with validation
                if enable_email_sending:
                    template = session.query(EmailTemplate).filter_by(
                        id=int(email_template.split(":")[0])).first()
                    for result in term_results['results']:
                        if not result or 'Email' not in result or not is_valid_email(result['Email']):
                            status_container.text(
                                f"Skipping invalid result or email: {result.get('Email') if result else 'None'}"
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
                    if not base_terms:
                        st.session_state.automation_logs.append("No search terms found. Skipping cycle.")
                        time.sleep(3600)
                        continue

                    optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
                    if not optimized_terms:
                        st.session_state.automation_logs.append("Failed to generate optimized terms. Skipping cycle.")
                        time.sleep(3600)
                        continue

                    new_leads_all = []
                    for term in optimized_terms:
                        try:
                            results = manual_search(session, [term], 10)
                            if results and 'results' in results:
                                new_leads = [(res['email'], res['url']) for res in results['results'] 
                                           if res.get('email') and res.get('url') and 
                                           save_lead(session, res['email'], url=res['url'])]
                                new_leads_all.extend(new_leads)
                                st.session_state.automation_logs.append(f"Found {len(new_leads)} leads for term: {term}")
                        except Exception as e:
                            st.session_state.automation_logs.append(f"Error searching term {term}: {str(e)}")
                            continue

                    if new_leads_all:
                        template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                        if template:
                            from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                            reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                            try:
                                logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, 
                                                                  [{'Email': email} for email, _ in new_leads_all])
                                st.session_state.automation_logs.extend(logs)
                            except Exception as e:
                                st.session_state.automation_logs.append(f"Error sending emails: {str(e)}")

                        leads_df = pd.DataFrame(new_leads_all, columns=['Email', 'URL'])
                        leads_container.dataframe(leads_df, hide_index=True)
                    else:
                        leads_container.info("No new leads found in this cycle.")

                    update_display(log_container, st.session_state.get('automation_logs', []), "Latest Logs", "log")
                    time.sleep(3600)

        except Exception as e:
            st.error(f"An error occurred in the automation process: {str(e)}")
            st.session_state.automation_status = False

    # Get the variables from session state
    search_terms = st.session_state.get('search_terms', [])
    submitted = st.session_state.get('submitted', False)
    num_results = st.session_state.get('num_results', 10)
    ignore_fetched = st.session_state.get('ignore_fetched', True)
    optimize_english = st.session_state.get('optimize_english', False)
    optimize_spanish = st.session_state.get('optimize_spanish', False)
    shuffle_keywords_option = st.session_state.get('shuffle_keywords_option', False)

    if submitted and search_terms:
        try:
            with st.spinner("Searching..."):
                process_id = str(uuid.uuid4())
                settings = {
                    'num_results': num_results,
                    'ignore_fetched': ignore_fetched,
                    'optimize_english': optimize_english,
                    'optimize_spanish': optimize_spanish,
                    'shuffle_keywords': shuffle_keywords_option
                }
                
                st.session_state.process_manager.start_process(
                    process_id,
                    background_manual_search,
                    (process_id, search_terms, settings)
                )
                
                st.success("Search process started! Check the Process Monitor for results.")
                
        except Exception as e:
            st.error(f"Error starting search: {str(e)}")
            logging.error(f"Search error: {str(e)}")

# --- Navigation function ---
def navigate_to(page):
    st.session_state.current_page = page

def initialize_session_state():
    """Initialize session state variables."""
    if 'active_page' not in st.session_state:
        st.session_state.active_page = 'Manual Search'
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'current_project_id' not in st.session_state:
        st.session_state.current_project_id = None
    if 'current_campaign_id' not in st.session_state:
        st.session_state.current_campaign_id = None
    if 'edit_template_id' not in st.session_state:
        st.session_state.edit_template_id = None
    if 'search_terms' not in st.session_state:
        st.session_state.search_terms = []
    if 'search_term_groups' not in st.session_state:
        st.session_state.search_term_groups = []
    if 'background_processes' not in st.session_state:
        st.session_state.background_processes = {}
    if 'process_logs' not in st.session_state:
        st.session_state.process_logs = {}
    if 'automation_status' not in st.session_state:
        st.session_state.automation_status = 'stopped'
    if 'process_manager' not in st.session_state:
        st.session_state.process_manager = ProcessManager()
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True

def main():
    initialize_session_state()
    
    # Get page from query params or session state
    query_params = st.experimental_get_query_params()
    current_page = query_params.get('page', ['Manual Search'])[0]
    
    # Navigation menu with query parameter updates
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Manual Search", "Search Terms", "Email Templates", "Bulk Send", 
             "View Leads", "Campaign Logs", "Sent Emails", "Projects & Campaigns",
             "Knowledge Base", "AutoClient AI", "Automation Control", "Settings"],
            icons=['search', 'list-task', 'envelope', 'send', 'person-lines-fill',
                  'journal-text', 'envelope-open', 'folder', 'book',
                  'robot', 'gear', 'gear-fill'],
            menu_icon="cast",
            default_index=0,
            key='nav_menu'
        )
        if selected != current_page:
            st.experimental_set_query_params(page=selected)
            
    # Cache the active page in session state
    if 'active_page' not in st.session_state:
        st.session_state.active_page = current_page
    elif st.session_state.active_page != current_page:
        st.session_state.active_page = current_page

    # Page routing using dictionary for better performance
    page_routes = {
        "Manual Search": manual_search_page,
        "Search Terms": search_terms_page,
        "Email Templates": email_templates_page,
        "Bulk Send": bulk_send_page,
        "View Leads": view_leads_page,
        "Campaign Logs": view_campaign_logs,
        "Sent Emails": view_sent_email_campaigns,
        "Projects & Campaigns": projects_campaigns_page,
        "Knowledge Base": knowledge_base_page,
        "AutoClient AI": autoclient_ai_page,
        "Automation Control": automation_control_panel_page,
        "Settings": settings_page
    }
    
    # Route to the selected page
    if current_page in page_routes:
        page_routes[current_page]()

def search_terms_page():
    st.title("Search Terms Management")
    
    with db_session() as session:
        # Fetch existing search terms and groups
        search_terms = fetch_search_terms_with_lead_count(session)
        groups = session.query(SearchTermGroup).all()
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "View Search Terms", 
            "Add Search Term", 
            "Manage Groups",
            "AI Grouping",
            "Group Settings"
        ])
        
        with tab1:
            st.subheader("Existing Search Terms")
            if search_terms:
                df = pd.DataFrame([{
                    'ID': term.id,
                    'Term': term.term,
                    'Group': term.group.name if term.group else 'None',
                    'Lead Count': term.lead_count if hasattr(term, 'lead_count') else 0,
                    'Created': term.created_at
                } for term in search_terms])
                
                st.dataframe(df)
            else:
                st.info("No search terms found.")
                
        with tab2:
            st.subheader("Add New Search Term")
            new_term = st.text_input("Enter new search term")
            campaign_id = get_active_campaign_id()
            
            group_options = ["None"] + [f"{g.id}: {g.name}" for g in groups]
            group_for_new_term = st.selectbox(
                "Select group for new term",
                options=group_options,
                format_func=lambda x: x.split(":")[1] if ":" in x else x
            )
            
            if st.button("Add Term") and new_term:
                add_new_search_term(session, new_term, campaign_id)
                st.success(f"Added new search term: {new_term}")
                st.rerun()
                
        with tab3:
            st.subheader("Manage Term Groups")
            if groups:
                for group in groups:
                    with st.expander(f"Group: {group.name}"):
                        group_terms = [
                            f"{term.id}: {term.term}" 
                            for term in search_terms 
                            if term.group_id == group.id
                        ]
                        available_terms = [
                            f"{term.id}: {term.term}" 
                            for term in search_terms 
                            if not term.group_id or term.group_id != group.id
                        ]
                        
                        selected_terms = st.multiselect(
                            "Select terms for this group",
                            options=available_terms + group_terms,
                            default=group_terms,
                            key=f"group_{group.id}"
                        )
                        
                        if st.button("Update Group", key=f"update_{group.id}"):
                            update_search_term_group(session, group.id, selected_terms)
                            st.success(f"Updated group: {group.name}")
                            st.rerun()
            else:
                st.info("No groups created yet.")
                
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
                group_to_delete = st.selectbox(
                    "Select Group to Delete",
                    [f"{g.id}: {g.name}" for g in groups],
                    format_func=lambda x: x.split(":")[1] if ":" in x else x
                )
                if st.button("Delete Group") and group_to_delete:
                    group_id = int(group_to_delete.split(":")[0])
                    delete_search_term_group(session, group_id)
                    st.success(f"Deleted group: {group_to_delete.split(':')[1]}")
                    st.rerun()

def background_manual_search(process_id: str, search_terms: list, settings: dict):
    """Background process for manual search with proper state management"""
    process_manager = ProcessManager()
    
    try:
        total_terms = len(search_terms)
        for idx, term in enumerate(search_terms, 1):
            try:
                progress = (idx / total_terms) * 100
                process_manager.update_process_progress(
                    process_id,
                    progress=progress,
                    processed_items=idx,
                    total_items=total_terms,
                    status='searching'
                )
                
                with db_session() as session:
                    results = manual_search(
                        session,
                        [term],
                        settings['num_results'],
                        settings['ignore_fetched']
                    )
                    
                    if results and isinstance(results, dict) and 'results' in results:
                        for result in results['results']:
                            save_lead(session, result['email'], url=result['url'])
                            
            except Exception as e:
                process_manager.update_process_progress(
                    process_id,
                    error=f"Error processing term '{term}': {str(e)}"
                )
                logging.error(f"Search error for term '{term}': {str(e)}")
                continue
        
        process_manager.update_process_progress(
            process_id,
            progress=100.0,
            status='completed'
        )
        
    except Exception as e:
        process_manager.update_process_progress(
            process_id,
            status='failed',
            error=str(e)
        )
        raise

def update_campaign_effectiveness(session, campaign_id):
    """Update campaign effectiveness metrics"""
    try:
        email_campaigns = session.query(EmailCampaign).filter_by(campaign_id=campaign_id).all()
        total_sent = len(email_campaigns)
        if total_sent == 0: return
        opens = sum(1 for ec in email_campaigns if ec.opened_at is not None)
        clicks = sum(1 for ec in email_campaigns if ec.clicked_at is not None)
        campaign = session.query(Campaign).get(campaign_id)
        if campaign:
            campaign.metrics = {'total_sent': total_sent, 'opens': opens, 'clicks': clicks, 'open_rate': (opens/total_sent)*100, 'click_rate': (clicks/total_sent)*100, 'last_updated': datetime.utcnow().isoformat()}
            session.commit()
    except Exception as e:
        logging.error(f"Error updating campaign effectiveness: {str(e)}")
        session.rollback()

def get_campaign_effectiveness(session, campaign_id):
    """Get campaign effectiveness metrics"""
    try:
        campaign = session.query(Campaign).get(campaign_id)
        if campaign and campaign.metrics: return campaign.metrics
        email_campaigns = session.query(EmailCampaign).filter_by(campaign_id=campaign_id).all()
        total_sent = len(email_campaigns)
        if total_sent == 0: return {'total_sent': 0, 'opens': 0, 'clicks': 0, 'open_rate': 0.0, 'click_rate': 0.0}
        opens = sum(1 for ec in email_campaigns if ec.opened_at is not None)
        clicks = sum(1 for ec in email_campaigns if ec.clicked_at is not None)
        return {'total_sent': total_sent, 'opens': opens, 'clicks': clicks, 'open_rate': (opens/total_sent)*100, 'click_rate': (clicks/total_sent)*100}
    except Exception as e:
        logging.error(f"Error getting campaign effectiveness: {str(e)}")
        return {'total_sent': 0, 'opens': 0, 'clicks': 0, 'open_rate': 0.0, 'click_rate': 0.0}

def update_lead_scores(session):
    try:
        leads = session.query(Lead).all()
        for lead in leads:
            email_campaigns = session.query(EmailCampaign).filter_by(lead_id=lead.id).all()
            if not email_campaigns: continue
            score = 50
            for campaign in email_campaigns:
                if campaign.opened_at: score += 10
                if campaign.clicked_at: score += 20
                if campaign.status == 'replied': score += 30
            most_recent = max((c.sent_at for c in email_campaigns if c.sent_at), default=None)
            if most_recent:
                days_since = (datetime.utcnow() - most_recent).days
                if days_since <= 7: score += 15
                elif days_since <= 30: score += 5
            lead.lead_score = min(100, score)
        session.commit()
    except Exception as e:
        logging.error(f"Error updating lead scores: {str(e)}")
        session.rollback()

def add_new_search_term(session, term, campaign_id):
    """Add a new search term"""
    try:
        search_term = SearchTerm(
            term=term,
            campaign_id=campaign_id,
            created_at=datetime.utcnow()
        )
        session.add(search_term)
        session.commit()
        return search_term
    except Exception as e:
        logging.error(f"Error adding search term: {str(e)}")
        session.rollback()
        return None

def get_page_description(html_content):
    """Extract page description from HTML content"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content']
        return ""
    except Exception as e:
        logging.error(f"Error extracting page description: {str(e)}")
        return ""

def email_templates_page():
    st.title("Email Templates")
    with db_session() as session:
        templates = fetch_email_templates(session)
        if templates:
            template_options = {f"{t.template_name} ({t.id})": t.id for t in templates}
            selected_template = st.selectbox("Select Template to Edit", list(template_options.keys()))
            template_id = template_options[selected_template]
            template = session.query(EmailTemplate).get(template_id)
            if template:
                template_name = st.text_input("Template Name", value=template.template_name)
                subject = st.text_input("Subject", value=template.subject)
                body_content = st.text_area("Body Content", value=template.body_content, height=300)
                language = st.selectbox("Language", ["ES", "EN"], index=0 if template.language == "ES" else 1)
                is_ai_customizable = st.checkbox("AI Customizable", value=template.is_ai_customizable)
                if st.button("Update Template"):
                    create_or_update_email_template(session, template_name, subject, body_content, template_id, is_ai_customizable, language=language)
                    st.success("Template updated successfully!")
        st.markdown("---")
        st.subheader("Create New Template")
        new_template_name = st.text_input("New Template Name")
        new_subject = st.text_input("New Subject")
        new_body_content = st.text_area("New Body Content", height=300)
        new_language = st.selectbox("New Language", ["ES", "EN"])
        new_is_ai_customizable = st.checkbox("New AI Customizable")
        if st.button("Create Template"):
            create_or_update_email_template(session, new_template_name, new_subject, new_body_content, is_ai_customizable=new_is_ai_customizable, language=new_language)
            st.success("Template created successfully!")
            st.rerun()

def bulk_send_page():
    st.title("Bulk Send Emails")
    with db_session() as session:
        templates = session.query(EmailTemplate).all()
        if not templates: st.warning("No email templates found"); return
        template_options = {f"{t.template_name} ({t.id})": t.id for t in templates}
        selected_template = st.selectbox("Select Email Template", list(template_options.keys()))
        template_id = template_options[selected_template]
        email_settings = session.query(EmailSettings).all()
        if not email_settings: st.warning("No email settings found"); return
        email_options = {f"{e.name} ({e.email})": e.email for e in email_settings}
        selected_email = st.selectbox("Select From Email", list(email_options.keys()))
        from_email = email_options[selected_email]
        reply_to = st.text_input("Reply-To Email (optional)", value=from_email)
        send_option = st.radio("Send To", ["All Leads", "Specific Email", "Selected Search Terms"])
        specific_email = st.text_input("Enter Email") if send_option == "Specific Email" else None
        selected_terms = st_tags(label="Select Search Terms", value=[], suggestions=[t.term for t in session.query(SearchTerm).all()]) if send_option == "Selected Search Terms" else None
        exclude_previously_contacted = st.checkbox("Exclude Previously Contacted Leads", value=True)
        leads = fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted)
        if not leads.empty:
            st.write(f"Found {len(leads)} leads")
            if st.button("Send Emails"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                log_container = st.empty()
                logs, sent_count = bulk_send_emails(session, template_id, from_email, reply_to, leads.to_dict('records'), progress_bar, status_text, results, log_container)
                st.success(f"Sent {sent_count} emails successfully!")
        else: st.warning("No leads found matching the criteria")

def view_leads_page():
    """View leads page implementation"""
    st.title("View Leads")
    
    with db_session() as session:
        # Fetch leads with their sources
        df = fetch_leads_with_sources(session)
        
        if df.empty:
            st.info("No leads found.")
            return
            
        # Add filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            email_filter = st.text_input("Filter by Email")
        with col2:
            company_filter = st.text_input("Filter by Company")
        with col3:
            status_filter = st.selectbox("Filter by Status", ["All"] + list(df['Last Email Status'].unique()))
            
        # Apply filters
        filtered_df = df.copy()
        if email_filter:
            filtered_df = filtered_df[filtered_df['email'].str.contains(email_filter, case=False, na=False)]
        if company_filter:
            filtered_df = filtered_df[filtered_df['company'].str.contains(company_filter, case=False, na=False)]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['Last Email Status'] == status_filter]
            
        # Display leads
        st.subheader(f"Leads ({len(filtered_df)} found)")
        edited_df = st.data_editor(
            filtered_df,
            hide_index=True,
            column_config={
                "Delete": st.column_config.CheckboxColumn(
                    "Delete",
                    help="Select to delete",
                    default=False,
                )
            }
        )
        
        # Handle deletions
        if st.button("Delete Selected Leads"):
            leads_to_delete = edited_df[edited_df['Delete']]['id'].tolist()
            if leads_to_delete:
                try:
                    for lead_id in leads_to_delete:
                        delete_lead_and_sources(session, lead_id)
                    st.success(f"Successfully deleted {len(leads_to_delete)} leads!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting leads: {str(e)}")
                    
        # Export functionality
        if st.button("Export to CSV"):
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                "leads_export.csv",
                "text/csv",
                key='download-csv'
            )

def view_campaign_logs():
    """View campaign logs page implementation"""
    st.title("Campaign Logs")
    
    with db_session() as session:
        # Get all campaigns
        campaigns = session.query(Campaign).all()
        if not campaigns:
            st.info("No campaigns found.")
            return
            
        # Campaign selection
        campaign_options = [f"{c.id}: {c.campaign_name}" for c in campaigns]
        selected_campaign = st.selectbox("Select Campaign", campaign_options)
        campaign_id = int(selected_campaign.split(":")[0])
        
        # Get campaign logs
        logs = session.query(AutomationLogs).filter_by(campaign_id=campaign_id).order_by(AutomationLogs.start_time.desc()).all()
        
        if not logs:
            st.info("No logs found for this campaign.")
            return
            
        # Display logs
        st.subheader("Campaign Activity")
        
        for log in logs:
            with st.expander(f"Log Entry - {log.start_time.strftime('%Y-%m-%d %H:%M:%S')}"):
                st.write(f"Status: {log.status}")
                st.write(f"Leads Gathered: {log.leads_gathered}")
                st.write(f"Emails Sent: {log.emails_sent}")
                st.write(f"Duration: {(log.end_time - log.start_time).total_seconds() / 60:.1f} minutes")
                
                if log.logs:
                    st.json(log.logs)
                    
        # Display metrics
        st.subheader("Campaign Metrics")
        total_leads = sum(log.leads_gathered for log in logs)
        total_emails = sum(log.emails_sent for log in logs)
        success_rate = sum(1 for log in logs if log.status == 'completed') / len(logs) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Leads Gathered", total_leads)
        col2.metric("Total Emails Sent", total_emails)
        col3.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Display effectiveness metrics
        effectiveness = get_campaign_effectiveness(session, campaign_id)
        if effectiveness:
            st.subheader("Email Effectiveness")
            col1, col2 = st.columns(2)
            col1.metric("Open Rate", f"{effectiveness['open_rate']:.1f}%")
            col2.metric("Click Rate", f"{effectiveness['click_rate']:.1f}%")

# Update variable references
AIRequestLog = AIRequestLogs
AutomationLogHandler = AutomationLogs

def get_email_preview(session, template_id, from_email, reply_to):
    template = session.query(EmailTemplate).get(template_id)
    if not template: return None
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings: return None
    return {'subject': template.subject, 'content': wrap_email_body(template.body_content), 'from_email': from_email, 'reply_to': reply_to}

def delete_lead_and_sources(session, lead_id):
    try:
        session.query(LeadSource).filter_by(lead_id=lead_id).delete()
        session.query(EmailCampaign).filter_by(lead_id=lead_id).delete()
        session.query(CampaignLead).filter_by(lead_id=lead_id).delete()
        session.query(Lead).filter_by(id=lead_id).delete()
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        raise

def wrap_email_body(body):
    return f'<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head><body>{body}</body></html>' if not body.strip().startswith(('<!DOCTYPE html>', '<html>')) else body

def view_sent_email_campaigns():
    """Display sent email campaigns with filtering and sorting options"""
    st.title("Sent Email Campaigns")
    
    with db_session() as session:
        # Fetch all email campaigns
        campaigns = session.query(EmailCampaign).order_by(EmailCampaign.sent_at.desc()).all()
        
        if not campaigns:
            st.info("No email campaigns found.")
            return
            
        # Create DataFrame for display
        df = pd.DataFrame([{
            'ID': c.id,
            'Lead Email': c.lead.email if c.lead else 'Unknown',
            'Subject': c.customized_subject,
            'Status': c.status,
            'Sent At': c.sent_at,
            'Opened': 'Yes' if c.opened_at else 'No',
            'Opened At': c.opened_at or '',
            'Clicked': 'Yes' if c.clicked_at else 'No',
            'Clicked At': c.clicked_at or ''
        } for c in campaigns])
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect(
                'Filter by Status',
                options=df['Status'].unique(),
                default=df['Status'].unique()
            )
        with col2:
            opened_filter = st.multiselect(
                'Filter by Opened',
                options=['Yes', 'No'],
                default=['Yes', 'No']
            )
        with col3:
            clicked_filter = st.multiselect(
                'Filter by Clicked',
                options=['Yes', 'No'],
                default=['Yes', 'No']
            )
        
        # Apply filters
        mask = (
            df['Status'].isin(status_filter) &
            df['Opened'].isin(opened_filter) &
            df['Clicked'].isin(clicked_filter)
        )
        filtered_df = df[mask]
        
        # Display results
        if not filtered_df.empty:
            st.dataframe(filtered_df)
            
            # Display statistics
            st.subheader("Campaign Statistics")
            total = len(filtered_df)
            opened = len(filtered_df[filtered_df['Opened'] == 'Yes'])
            clicked = len(filtered_df[filtered_df['Clicked'] == 'Yes'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Emails", total)
            
            # Calculate rates with proper string formatting
            open_rate = f"{(opened/total*100):.1f}%" if total > 0 else "0%"
            click_rate = f"{(clicked/total*100):.1f}%" if total > 0 else "0%"
            
            col2.metric("Open Rate", open_rate)
            col3.metric("Click Rate", click_rate)
        else:
            st.info("No campaigns match the selected filters.")

def settings_page():
    """Display settings page"""
    st.title("Settings")
    
    with db_session() as session:
        # Email Settings
        st.header("Email Settings")
        settings = session.query(EmailSettings).all()
        
        # Display existing settings
        if settings:
            for setting in settings:
                with st.expander(f"Email Setting: {setting.name}"):
                    st.write(f"Email: {setting.email}")
                    st.write(f"Provider: {setting.provider}")
                    if st.button("Delete", key=f"delete_{setting.id}"):
                        delete_email_setting(setting.id)
                        st.rerun()
        
        # Add new setting form
        with st.form("new_email_setting"):
            st.subheader("Add New Email Setting")
            name = st.text_input("Name")
            email = st.text_input("Email")
            provider = st.selectbox("Provider", ["ses", "smtp"])
            
            # Initialize provider-specific variables
            smtp_server = None
            smtp_port = None
            smtp_username = None
            smtp_password = None
            aws_access_key_id = None
            aws_secret_access_key = None
            aws_region = None
            
            # Provider specific fields
            if provider == "smtp":
                smtp_server = st.text_input("SMTP Server")
                smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535)
                smtp_username = st.text_input("SMTP Username")
                smtp_password = st.text_input("SMTP Password", type="password")
            else:  # ses
                aws_access_key_id = st.text_input("AWS Access Key ID")
                aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password")
                aws_region = st.text_input("AWS Region")
            
            if st.form_submit_button("Save"):
                try:
                    new_setting = EmailSettings(
                        name=name,
                        email=email,
                        provider=provider
                    )
                    
                    if provider == "smtp":
                        new_setting.smtp_server = smtp_server
                        new_setting.smtp_port = smtp_port
                        new_setting.smtp_username = smtp_username
                        new_setting.smtp_password = smtp_password
                    else:
                        new_setting.aws_access_key_id = aws_access_key_id
                        new_setting.aws_secret_access_key = aws_secret_access_key
                        new_setting.aws_region = aws_region
                    
                    session.add(new_setting)
                    session.commit()
                    st.success("Email setting saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving email setting: {str(e)}")

        # General Settings
        st.header("General Settings")
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        
        with st.form("general_settings"):
            current_settings = general_settings.value if general_settings else {}
            
            openai_api_key = st.text_input(
                "OpenAI API Key",
                value=current_settings.get('openai_api_key', ''),
                type="password"
            )
            
            openai_model = st.selectbox(
                "OpenAI Model",
                ["gpt-4", "gpt-3.5-turbo"],
                index=0 if current_settings.get('openai_model') == "gpt-4" else 1
            )
            
            if st.form_submit_button("Save General Settings"):
                try:
                    if not general_settings:
                        general_settings = Settings(
                            name="general_settings",
                            setting_type="general",
                            value={}
                        )
                        session.add(general_settings)
                    
                    general_settings.value = {
                        'openai_api_key': openai_api_key,
                        'openai_model': openai_model
                    }
                    session.commit()
                    st.success("General settings saved successfully!")
                except Exception as e:
                    st.error(f"Error saving general settings: {str(e)}")

def display_logs(log_container, logs):
    # Implement efficient log display with caching and buffering
    if 'cached_logs' not in st.session_state:
        st.session_state.cached_logs = []
        st.session_state.last_log_count = 0
    
    if not logs:
        log_container.markdown(
            '<div class="log-container">No logs to display yet.</div>',
            unsafe_allow_html=True
        )
        return

    # Only update if log count has changed significantly (buffer of 5 entries)
    if (len(logs) - st.session_state.last_log_count >= 5 or 
        logs != st.session_state.cached_logs):
        st.session_state.cached_logs = logs.copy()
        st.session_state.last_log_count = len(logs)
        
    # Static CSS styles
    log_container.markdown(
        """
        <style>
        .log-container {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 8px;
            padding: 1rem;
            background: linear-gradient(to bottom, #ffffff, #f8f9fa);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            font-family: monospace;
        }
        .log-entry {
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            border-radius: 4px;
            background-color: rgba(28, 131, 225, 0.05);
            border-left: 3px solid #1c83e1;
            transition: all 0.2s ease;
        }
        .log-entry:hover {
            background-color: rgba(28, 131, 225, 0.1);
            transform: translateX(2px);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Cache the icons dictionary
    @st.cache_data
    def get_log_icons():
        return {
            'error': '🔴',
            'success': '🟢',
            'completed': '🟢',
            'warning': '🟡',
            'email': '📧',
            'search': '🔍'
        }

    # Implement buffered updates for logs
    if 'log_buffer' not in st.session_state:
        st.session_state.log_buffer = []

    icons = get_log_icons()
    formatted_logs = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Optimize log formatting with batch processing
    MAX_LOGS = 20
    recent_logs = logs[-MAX_LOGS:]
    st.session_state.log_buffer.extend(recent_logs)
    
    # Keep buffer size in check
    if len(st.session_state.log_buffer) > MAX_LOGS:
        st.session_state.log_buffer = st.session_state.log_buffer[-MAX_LOGS:]

    for log in st.session_state.log_buffer:
        icon = next((v for k, v in icons.items() if k in log.lower()), '🔵')
        formatted_logs.append(
            f'<div class="log-entry">{icon} [{timestamp}] {log}</div>'
        )

    # Batch update logs with single markdown call
    log_container.markdown(
        f"""
        <div class="log-container" id="log-container">
            {''.join(formatted_logs)}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Optimize scroll behavior with debounced JavaScript
    st.markdown(
        """
        <script>
            const scrollToBottom = () => {
                requestAnimationFrame(() => {
                    const container = document.getElementById('log-container');
                    if (container) container.scrollTop = container.scrollHeight;
                });
            };
            scrollToBottom();
        </script>
        """,
        unsafe_allow_html=True
    )

# Add helper function to check database state
def check_database_state():
    """Check and log database schema state"""
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        logging.info("Current database schema state:")
        for table in tables:
            columns = inspector.get_columns(table)
            logging.info(f"\nTable: {table}")
            for col in columns:
                logging.info(f"  Column: {col['name']} ({col['type']})")
                
    except Exception as e:
        logging.error(f"Error checking database state: {str(e)}")

def create_default_email_settings():
    """Create default email settings if none exist"""
    with db_session() as session:
        if session.query(EmailSettings).count() == 0:
            # Create a default SES email setting
            default_ses = EmailSettings(
                name="Default SES",
                email="hello@indosy.com",
                provider="ses",
                aws_access_key_id="YOUR_AWS_ACCESS_KEY",
                aws_secret_access_key="YOUR_AWS_SECRET_KEY",
                aws_region="eu-central-1",
                created_at=datetime.utcnow()
            )
            session.add(default_ses)
            
            # Create a default SMTP email setting
            default_smtp = EmailSettings(
                name="Default SMTP",
                email="eugproductions@gmail.com",
                provider="smtp",
                smtp_server="smtp.gmail.com",
                smtp_port=587,
                smtp_username="your_email@gmail.com",
                smtp_password="your_app_password",
                created_at=datetime.utcnow()
            )
            session.add(default_smtp)
            
            try:
                session.commit()
                logging.info("Created default email settings")
            except Exception as e:
                session.rollback()
                logging.error(f"Error creating default email settings: {str(e)}")

def get_knowledge_base_info(session, project_id):
    """Get knowledge base info for a project"""
    return session.query(KnowledgeBase).filter_by(project_id=project_id).first()

def generate_optimized_search_terms(session, base_terms, kb_info):
    """Generate optimized search terms using AI"""
    if not base_terms:
        return []
    prompt = f"Optimize these search terms for lead generation: {', '.join(base_terms)}"
    response = openai_chat_completion([{"role": "system", "content": "You're an AI specializing in optimizing search terms."}, {"role": "user", "content": prompt}])
    return response.get('terms', base_terms) if isinstance(response, dict) else base_terms

def bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_bar=None, status_text=None, results=None, log_container=None):
    """Send bulk emails to leads"""
    template = session.query(EmailTemplate).get(template_id)
    if not template:
        raise ValueError(f"Template {template_id} not found")
    
    sent_count = 0
    logs = []
    
    for idx, lead in enumerate(leads):
        try:
            if progress_bar:
                progress_bar.progress((idx + 1) / len(leads))
            if status_text:
                status_text.text(f"Processing {idx + 1}/{len(leads)}")
                
            response, tracking_id = send_email_ses(session, from_email, lead['Email'], template.subject, template.body_content, reply_to=reply_to)
            if response:
                sent_count += 1
                logs.append(f"✅ Email sent to {lead['Email']}")
            else:
                logs.append(f"❌ Failed to send email to {lead['Email']}")
                
            if log_container:
                log_container.text(logs[-1])
            if results is not None:
                results.append({"Email": lead['Email'], "Status": "sent" if response else "failed"})
                
        except Exception as e:
            logs.append(f"❌ Error sending to {lead['Email']}: {str(e)}")
            if log_container:
                log_container.text(logs[-1])
                
    return logs, sent_count

def projects_campaigns_page():
    """Display projects and campaigns page"""
    st.title("Projects & Campaigns")
    with db_session() as session:
        projects = fetch_projects(session)
        if not projects:
            st.warning("No projects found. Please create a project first.")
            return
        
        selected_project = st.selectbox("Select Project", projects)
        project_id = int(selected_project.split(":")[0])
        
        campaigns = session.query(Campaign).filter_by(project_id=project_id).all()
        if campaigns:
            st.subheader("Campaigns")
            for campaign in campaigns:
                with st.expander(f"Campaign: {campaign.campaign_name}"):
                    st.write(f"Type: {campaign.campaign_type}")
                    st.write(f"Created: {campaign.created_at}")
                    metrics = get_campaign_effectiveness(session, campaign.id)
                    if metrics:
                        st.metric("Emails Sent", metrics["total_sent"])
                        st.metric("Open Rate", f"{metrics['open_rate']:.1f}%")
                        st.metric("Click Rate", f"{metrics['click_rate']:.1f}%")

def knowledge_base_page():
    """Display knowledge base page"""
    st.title("Knowledge Base")
    with db_session() as session:
        project_id = get_active_project_id()
        kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        
        with st.form("knowledge_base_form"):
            kb_name = st.text_input("Knowledge Base Name", value=kb.kb_name if kb else "")
            kb_bio = st.text_area("Bio", value=kb.kb_bio if kb else "")
            kb_values = st.text_area("Values", value=kb.kb_values if kb else "")
            
            if st.form_submit_button("Save"):
                if not kb:
                    kb = KnowledgeBase(project_id=project_id)
                    session.add(kb)
                
                kb.kb_name = kb_name
                kb.kb_bio = kb_bio
                kb.kb_values = kb_values
                session.commit()
                st.success("Knowledge base saved!")

def autoclient_ai_page():
    """Display AutoClient AI page"""
    st.title("AutoClient AI")
    with db_session() as session:
        kb = get_knowledge_base_info(session, get_active_project_id())
        if not kb:
            st.error("Please set up your knowledge base first.")
            return
        
        st.subheader("AI Settings")
        with st.form("ai_settings"):
            model = st.selectbox("AI Model", ["gpt-4", "gpt-3.5-turbo"])
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            
            if st.form_submit_button("Save Settings"):
                settings = session.query(Settings).filter_by(name="ai_settings").first()
                if not settings:
                    settings = Settings(name="ai_settings", setting_type="ai")
                    session.add(settings)
                
                settings.value = {"model": model, "temperature": temperature}
                session.commit()
                st.success("AI settings saved!")

if __name__ == "__main__":
    try:
        # Configure SQLAlchemy mappers
        configure_mappers()
        
        # Initialize database
        init_db()
        
        # Create default settings
        create_default_email_settings()
        
        # Log database state
        check_database_state()
        
        # Start the application
        main()
    except Exception as e:
        st.error(f"Application startup error: {str(e)}")

