import streamlit as st

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="AutoclientAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core imports
import asyncio, atexit, html, json, logging, multiprocessing, os, random, re, smtplib
import threading, time, uuid, traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from multiprocessing import Process
from typing import List, Optional, TYPE_CHECKING
from urllib.parse import urlparse, urlencode
import base64

# External dependencies
import aiohttp, boto3, openai, pandas as pd, plotly.express as px, requests
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from email_validator import validate_email, EmailNotValidError
from fake_useragent import UserAgent
from googlesearch import search as google_search
from openai import OpenAI
from requests.adapters import HTTPAdapter
from sqlalchemy import (and_, Boolean, BigInteger, Column, create_engine, DateTime, distinct,
    Float, ForeignKey, func, Index, Integer, JSON, or_, select, String, Table, Text, text,
    TIMESTAMP, inspect, MetaData)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (declarative_base, joinedload, relationship, Session, 
    sessionmaker, configure_mappers)
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential

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
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    
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
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    auto_send = Column(Boolean, server_default='false')
    loop_automation = Column(Boolean, server_default='false')
    ai_customization = Column(Boolean, server_default='false')
    max_emails_per_group = Column(BigInteger, server_default='50')
    loop_interval = Column(BigInteger, server_default='60')
    status = Column(Text, server_default=text("'active'"))
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
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    language = Column(Text)
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    group_id = Column(BigInteger, ForeignKey('search_term_groups.id'))
    effectiveness = relationship("SearchTermEffectiveness", back_populates="search_term", uselist=False)
    campaign = relationship("Campaign", back_populates="search_terms")
    lead_sources = relationship("LeadSource", back_populates="search_term")
    search_group = relationship("SearchTermGroup", back_populates="search_terms")

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
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
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
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    lead = relationship("Lead", back_populates="lead_sources")
    search_term = relationship("SearchTerm", back_populates="lead_sources")

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    template_name = Column(Text, nullable=False)
    subject = Column(Text, nullable=False)
    body_content = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    is_ai_customizable = Column(Boolean, server_default='false')
    language = Column(Text, server_default='ES')
    template_type = Column(Text)
    variables = Column(JSONB)
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
    engagement_data = Column(JSONB)
    message_id = Column(Text)
    tracking_id = Column(Text, unique=True)
    sent_at = Column(TIMESTAMP(timezone=True))
    ai_customized = Column(Boolean)
    opened_at = Column(TIMESTAMP(timezone=True))
    clicked_at = Column(TIMESTAMP(timezone=True))
    open_count = Column(BigInteger)
    click_count = Column(BigInteger)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
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
    tone_of_voice = Column(Text)
    communication_style = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    project = relationship("Project", back_populates="knowledge_base")

class CampaignLead(Base):
    __tablename__ = 'campaign_leads'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    status = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    lead = relationship("Lead", back_populates="campaign_leads")
    campaign = relationship("Campaign", back_populates="campaign_leads")

class AIRequestLog(Base):
    __tablename__ = 'ai_request_logs'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    function_name = Column(Text)
    prompt = Column(Text)
    response = Column(Text)
    model_used = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    lead_id = Column(BigInteger)
    email_campaign_id = Column(BigInteger)
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    tokens_used = Column(Integer)
    duration_ms = Column(Integer)
    status = Column(Text)
    error = Column(Text)

class AutomationLogs(Base):
    __tablename__ = 'automation_logs'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    leads_gathered = Column(BigInteger)
    emails_sent = Column(BigInteger)
    start_time = Column(TIMESTAMP(timezone=True))
    end_time = Column(TIMESTAMP(timezone=True))
    status = Column(Text)
    logs = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    error = Column(Text)
    duration_seconds = Column(Integer)
    campaign = relationship("Campaign")
    search_term = relationship("SearchTerm")

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
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    is_active = Column(Boolean, server_default='true')
    daily_limit = Column(Integer)
    hourly_limit = Column(Integer)
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    settings = Column(JSONB)

class BackgroundProcessState(Base):
    __tablename__ = 'background_process_state'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    process_id = Column(Text, unique=True)
    status = Column(Text)
    started_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))
    error_message = Column(Text)
    progress = Column(Float)
    total_items = Column(Integer)
    processed_items = Column(Integer)
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    process_type = Column(Text)
    meta_data = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    campaign = relationship("Campaign")
    project = relationship("Project")

class OptimizedSearchTerms(Base):
    __tablename__ = 'optimized_search_terms'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    original_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    term = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    effectiveness_score = Column(Float)
    is_active = Column(Boolean, server_default='true')
    meta_data = Column(JSONB)
    original_term = relationship("SearchTerm")
    campaign = relationship("Campaign")
    project = relationship("Project")

class SearchGroups(Base):
    __tablename__ = 'search_groups'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    emails_sent = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True))
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    is_active = Column(Boolean, server_default='true')
    meta_data = Column(JSONB)
    campaign = relationship("Campaign")
    project = relationship("Project")

class SearchTermGroup(Base):
    __tablename__ = 'search_term_groups'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    description = Column(Text)
    email_template = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(TIMESTAMP(timezone=True))
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    search_terms = relationship("SearchTerm", back_populates="search_group")

class SearchTermEffectiveness(Base):
    __tablename__ = 'search_term_effectiveness'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    effectiveness_score = Column(Float)
    total_leads = Column(Integer)
    valid_leads = Column(Integer)
    irrelevant_leads = Column(Integer)
    blogs_found = Column(Integer)
    directories_found = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(TIMESTAMP(timezone=True))
    search_term = relationship("SearchTerm", back_populates="effectiveness")

class Settings(Base):
    __tablename__ = 'settings'
    __table_args__ = {'extend_existing': True}
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    setting_type = Column(Text, nullable=False)
    value = Column(JSONB, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(TIMESTAMP(timezone=True))

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
    """Enhanced log update function with better formatting"""
    icon = {
        'info': '🔵',
        'success': '🟢',
        'warning': '🟡',
        'error': '🔴',
        'email_sent': '📧',
        'search': '🔍'
    }.get(level, '⚪')
    
    # Console logging for debugging
    print(f"{icon} {message.split('<')[0]}")
    
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    
    # Create formatted log entry
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'icon': icon,
        'message': message,
        'level': level,
        'process_id': process_id
    }
    
    st.session_state.log_entries.append(log_entry)
    
    # Keep log size manageable
    MAX_LOGS = 1000
    if len(st.session_state.log_entries) > MAX_LOGS:
        st.session_state.log_entries = st.session_state.log_entries[-MAX_LOGS:]
    
    # Update display
    display_logs(log_container, [entry['message'] for entry in st.session_state.log_entries])

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

def manual_search(session, terms, max_results=10, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False, language="ES", one_email_per_url=True, one_email_per_domain=True):
    """Perform manual search with the given parameters"""
    results = {'results': [], 'errors': []}
    processed_domains = set()
    processed_urls = set()
    
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
                try:
                    # Skip if URL already processed and one_email_per_url is enabled
                    if one_email_per_url and url in processed_urls:
                        continue
                        
                    # Extract domain and skip if already processed and one_email_per_domain is enabled
                    domain = urlparse(url).netloc
                    if one_email_per_domain and domain in processed_domains:
                        continue
                        
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
                            results['results'].append({'Email': email, 'URL': url, 'Company': company})
                            processed_urls.add(url)
                            processed_domains.add(domain)
                            
                            # Break after first valid email if one_email_per_url is enabled
                            if one_email_per_url:
                                break
                                
                except Exception as e:
                    results['errors'].append(f"Error processing URL {url}: {str(e)}")
                    continue
                    
        except Exception as e:
            results['errors'].append(f"Error processing term {term}: {str(e)}")
            continue
            
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
    """Manual search page implementation"""
    st.title("Manual Search")
    
    # Initialize session state for persistent storage
    if 'search_state' not in st.session_state:
        st.session_state.search_state = {
            'is_searching': False,
            'current_term_index': 0,
            'total_terms': 0,
            'results_cache': [],
            'last_search_params': None,
            'background_processes': {},
            'last_update': time.time()
        }
    
    with db_session() as session:
        # Fetch recent searches within the session
        recent_searches = session.query(SearchTerm).order_by(SearchTerm.created_at.desc()).limit(5).all()
        recent_search_terms = [term.term for term in recent_searches]
        
        email_templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)

    # Create placeholders for dynamic content
    status_container = st.empty()
    progress_container = st.empty()
    metrics_container = st.empty()
    log_container = st.container()

    # Search interface with two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Enhanced search terms input with validation
        search_terms = st_tags(
            label='Enter search terms:',
            text='Press enter to add more',
            value=recent_search_terms,
            suggestions=['software engineer', 'data scientist', 'product manager'],
            maxtags=10,
            key='search_terms_input'
        )
        
        # Results per term with visual slider
        num_results = st.slider(
            "Results per term", 
            min_value=1, 
            max_value=50000, 
            value=10,
            help="Number of results to fetch per search term"
        )

    with col2:
        st.subheader("Search Options")
        
        # Main options with tooltips
        enable_email_sending = st.checkbox(
            "Enable email sending", 
            value=True,
            help="Automatically send emails to found leads"
        )
        
        ignore_previously_fetched = st.checkbox(
            "Ignore fetched domains", 
            value=True,
            help="Skip domains that have been previously searched"
        )
        
        # Advanced options in an expander
        with st.expander("Advanced Options", expanded=False):
            shuffle_keywords_option = st.checkbox(
                "Shuffle Keywords", 
                value=True,
                help="Randomly reorder keywords for better results"
            )
            
            optimize_english = st.checkbox(
                "Optimize (English)", 
                value=False,
                help="Optimize search terms for English results"
            )
            
            optimize_spanish = st.checkbox(
                "Optimize (Spanish)", 
                value=False,
                help="Optimize search terms for Spanish results"
            )
            
            language = st.selectbox(
                "Select Language", 
                options=["ES", "EN"], 
                index=0,
                help="Choose search language"
            )
            
            one_email_per_url = st.checkbox(
                "Only One Email per URL", 
                value=True,
                help="Extract only one email from each URL"
            )
            
            one_email_per_domain = st.checkbox(
                "Only One Email per Domain", 
                value=True,
                help="Extract only one email from each domain"
            )
            
            run_in_background = st.checkbox(
                "Run in background", 
                value=True,
                help="Process search in the background"
            )

    # Email settings section
    if enable_email_sending:
        if not email_templates:
            st.error("No email templates available. Please create a template first.")
            return
        if not email_settings:
            st.error("No email settings available. Please add email settings first.")
            return

        # Email configuration in columns
        col3, col4 = st.columns(2)
        with col3:
            email_template = st.selectbox(
                "Email template", 
                options=[f"{t.id}: {t.template_name}" for t in email_templates],
                help="Select email template to use"
            )
        with col4:
            email_setting = st.selectbox(
                "From Email", 
                options=[f"{s.id}: {s.email}" for s in email_settings],
                help="Select sender email address"
            )
            if email_setting:
                setting_id = int(email_setting.split(":")[0])
                setting = next((s for s in email_settings if s.id == setting_id), None)
                if setting:
                    from_email = setting.email
                    reply_to = st.text_input(
                        "Reply To", 
                        value=setting.email,
                        help="Email address for replies"
                    )

    # Initialize result containers
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'leads_found' not in st.session_state:
        st.session_state.leads_found = []
    if 'emails_sent' not in st.session_state:
        st.session_state.emails_sent = []

    # Search button with loading state
    search_button = st.button(
        "Start Search",
        type="primary",
        disabled=st.session_state.search_state['is_searching'],
        help="Begin the search process"
    )

    if search_button:
        if not search_terms:
            st.warning("Please enter at least one search term")
            return

        # Update search state
        st.session_state.search_state['is_searching'] = True
        st.session_state.search_state['current_term_index'] = 0
        st.session_state.search_state['total_terms'] = len(search_terms)
        st.session_state.search_state['results_cache'] = []
        
        try:
            if run_in_background:
                # Start background process
                process_id = start_background_manual_search(
                    search_terms,
                    num_results,
                    ignore_previously_fetched,
                    optimize_english,
                    optimize_spanish,
                    shuffle_keywords_option,
                    language,
                    one_email_per_url,
                    one_email_per_domain
                )
                
                if process_id:
                    st.success(f"Search started in background. Process ID: {process_id}")
                    st.session_state.current_search_process = process_id
                    
                    # Show process status
                    process_manager = ProcessManager()
                    state = process_manager.get_process_state(process_id)
                    if state:
                        status_container.text(f"Status: {state['status']}")
                        if state['total_items'] > 0:
                            progress = (state['processed_items'] / state['total_items']) * 100
                            progress_container.progress(progress)
                        if state['error']:
                            st.error(f"Error: {state['error']}")
                else:
                    st.error("Failed to start background search process")
            else:
                # Run in foreground with progress tracking
                progress_bar = progress_container.progress(0)
                status_text = status_container.empty()
                
                for i, term in enumerate(search_terms):
                    status_text.text(f"Searching: '{term}' ({i+1}/{len(search_terms)})")
                    progress_bar.progress((i + 1) / len(search_terms))
                    
                    with db_session() as session:
                        results = manual_search(
                            session,
                            [term],
                            num_results,
                            ignore_previously_fetched,
                            optimize_english,
                            optimize_spanish,
                            shuffle_keywords_option,
                            language,
                            one_email_per_url,
                            one_email_per_domain
                        )
                        
                        # Process and display results
                        if results.get('results'):
                            st.session_state.results.extend(results['results'])
                            st.session_state.leads_found.extend(
                                [f"{res['Email']} - {res.get('Company', 'Unknown')}" 
                                 for res in results['results']]
                            )
                            
                            # Update metrics
                            metrics_container.metric(
                                "Leads Found",
                                len(st.session_state.leads_found),
                                delta=len(results['results'])
                            )
                
                # Show final results
                status_text.text("Search completed!")
                
                if st.session_state.leads_found:
                    with st.expander("Search Results", expanded=True):
                        df = pd.DataFrame(st.session_state.results)
                        st.dataframe(
                            df,
                            column_config={
                                "Email": st.column_config.TextColumn(
                                    "Email",
                                    help="Found email addresses",
                                    width="medium"
                                ),
                                "Company": st.column_config.TextColumn(
                                    "Company",
                                    help="Associated company names",
                                    width="medium"
                                ),
                                "URL": st.column_config.LinkColumn(
                                    "Source URL",
                                    help="Where the email was found"
                                )
                            }
                        )
                else:
                    st.info("No leads found for the given search terms")
        
        except Exception as e:
            st.error(f"Search error: {str(e)}")
        finally:
            st.session_state.search_state['is_searching'] = False

    # Show background process status
    if hasattr(st.session_state, 'current_search_process'):
        process_id = st.session_state.current_search_process
        process_manager = ProcessManager()
        state = process_manager.get_process_state(process_id)
        
        if state:
            with st.expander("Background Process Status", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", state['status'].title())
                with col2:
                    if state['total_items'] > 0:
                        progress = (state['processed_items'] / state['total_items']) * 100
                        st.metric("Progress", f"{progress:.1f}%")
                with col3:
                    st.metric("Items Processed", state['processed_items'])
                
                if state['error']:
                    st.error(f"Error: {state['error']}")
                
                # Add stop button for running processes
                if state['status'] == 'running':
                    if st.button("Stop Process"):
                        process_manager.stop_process(process_id)
                        st.success("Process stopped")
                        st.rerun()
                
                # Clear completed processes
                elif state['status'] in ['completed', 'failed']:
                    if st.button("Clear Status"):
                        del st.session_state.current_search_process
                        st.rerun()

def get_page_description(url):
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        meta_description = soup.find('meta', attrs={'name': 'description'})
        return meta_description['content'].strip() if meta_description and meta_description.get('content') else "No description found"
    except Exception as e:
        logging.error(f"Error getting page description for {url}: {str(e)}")
        return "Error fetching description"

def get_campaign_effectiveness(session, campaign_id):
    """Calculate campaign effectiveness metrics"""
    total_sent = session.query(EmailCampaign).filter_by(campaign_id=campaign_id).count()
    open_rate = session.query(EmailCampaign).filter_by(campaign_id=campaign_id, status='Opened').count() / total_sent if total_sent > 0 else 0
    click_rate = session.query(EmailCampaign).filter_by(campaign_id=campaign_id, status='Clicked').count() / total_sent if total_sent > 0 else 0
    return total_sent, open_rate, click_rate

def start_background_manual_search(search_terms, num_results, ignore_previously_fetched=True, 
                                 optimize_english=False, optimize_spanish=False, 
                                 shuffle_keywords_option=False, language='ES',
                                 one_email_per_url=True, one_email_per_domain=True):
    """Start a background search process"""
    process_id = str(uuid.uuid4())
    
    def search_worker():
        try:
            with db_session() as session:
                process_state = session.query(BackgroundProcessState).filter_by(process_id=process_id).first()
                if not process_state:
                    process_state = BackgroundProcessState(
                        process_id=process_id,
                        status='running',
                        total_items=len(search_terms),
                        processed_items=0
                    )
                    session.add(process_state)
                    session.commit()
                
                for i, term in enumerate(search_terms):
                    try:
                        results = manual_search(
                            session,
                            [term],
                            num_results,
                            ignore_previously_fetched,
                            optimize_english,
                            optimize_spanish,
                            shuffle_keywords_option,
                            language,
                            one_email_per_url,
                            one_email_per_domain
                        )
                        
                        process_state.processed_items = i + 1
                        process_state.progress = ((i + 1) / len(search_terms)) * 100
                        session.commit()
                        
                    except Exception as e:
                        process_state.error_message = str(e)
                        process_state.status = 'error'
                        session.commit()
                        return
                
                process_state.status = 'completed'
                process_state.completed_at = datetime.now()
                session.commit()
                
        except Exception as e:
            with db_session() as session:
                process_state = session.query(BackgroundProcessState).filter_by(process_id=process_id).first()
                if process_state:
                    process_state.status = 'error'
                    process_state.error_message = str(e)
                    session.commit()
    
    thread = threading.Thread(target=search_worker)
    thread.daemon = True
    thread.start()
    
    return process_id

def get_email_preview(session, template_id, from_email, reply_to):
    """Generate email preview HTML"""
    template = session.query(EmailTemplate).get(template_id)
    if not template:
        return "Template not found"
    
    subject = template.subject
    body = template.body_content
    
    # Wrap email body in HTML template
    wrapped_body = wrap_email_body(body)
    
    # Generate email preview
    email_preview = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ccc; border-radius: 5px;">
        <h2 style="color: #333;">{subject}</h2>
        <div style="color: #555;">{wrapped_body}</div>
    </div>
    """
    
    return email_preview

def wrap_email_body(body):
    """Wrap email body in HTML template"""
    return f"""
    <div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.5; color: #333;">
        {body}
    </div>
    """

def main():
    """
    Main application entry point that configures and runs the Streamlit application.
    
    This function:
    1. Configures the page layout and logging
    2. Initializes navigation and pages
    3. Handles page routing and error handling
    4. Manages debug information
    """
    # Configure logging first
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize session state for logger if not exists
        if 'logger_initialized' not in st.session_state:
            st.session_state.logger_initialized = True
            st.session_state.log_messages = []

        # Initialize available pages
        pages = {
            "Manual Search": manual_search_page,
            "Email Campaigns": automation_control_panel_page,
            "Projects & Campaigns": projects_campaigns_page,
            "Knowledge Base": knowledge_base_page,
            "AutoclientAI": autoclient_ai_page,
            "Lead Management": view_leads_page,
            "Search Terms": search_terms_page,
            "Email Templates": email_templates_page,
            "Campaign Logs": view_campaign_logs,
            "Sent Campaigns": view_sent_email_campaigns,
            "Settings": settings_page
        }

        # Log debug information
        logger.debug("Current session state keys: %s", list(st.session_state.keys()))

        # Navigation sidebar
        with st.sidebar:
            selected = option_menu(
                menu_title="Navigation",
                options=list(pages.keys()),
                icons=["search", "send", "people", "key", "robot", "person-lines-fill", 
                      "tags", "envelope", "journal-text", "envelope-paper", "gear"],
                menu_icon="cast",
                default_index=0
            )
            logger.info("Selected page: %s", selected)

        # Debug information container
        with st.expander("🔍 Debug Information", expanded=False):
            st.write({
                "Session State": dict(st.session_state),
                "Current Page": selected,
                "Available Pages": list(pages.keys())
            })

        # Execute selected page with performance tracking
        start_time = time.time()
        pages[selected]()
        execution_time = time.time() - start_time

        # Log performance metrics
        logger.debug("Page %s rendered in %.2f seconds", selected, execution_time)

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")

    except Exception as exc:
        logger.exception("Critical application error")
        st.error(f"An error occurred: {str(exc)}")
        st.error("Full error traceback:")
        st.code(traceback.format_exc())

def automation_control_panel_page():
    st.title("Automation Control Panel")

    # Use session state for automation status and logs
    if 'automation_status' not in st.session_state:
        st.session_state.automation_status = False
    if 'automation_logs' not in st.session_state:
        st.session_state.automation_logs = []
    if 'last_metrics_update' not in st.session_state:
        st.session_state.last_metrics_update = 0
    if 'automation_thread' not in st.session_state:
        st.session_state.automation_thread = None

    # Create placeholders for dynamic content
    metrics_container = st.empty()
    status_container = st.empty()
    log_container = st.empty()

    try:
        with db_session() as session:
            # Fetch campaign data - cache for 30 seconds
            current_time = time.time()
            if ('campaign_data' not in st.session_state or 
                current_time - st.session_state.last_metrics_update > 30):
                
                campaign = session.query(Campaign).first()
                if campaign:
                    metrics = {
                        'Total Leads': session.query(Lead).count(),
                        'Active Campaigns': session.query(Campaign).filter_by(auto_send=True).count(),
                        'Emails Sent': session.query(EmailCampaign).filter(EmailCampaign.sent_at.isnot(None)).count(),
                        'Success Rate': f"{(session.query(EmailCampaign).filter_by(status='delivered').count() / max(session.query(EmailCampaign).count(), 1)) * 100:.1f}%"
                    }
                    st.session_state.campaign_data = metrics
                    st.session_state.last_metrics_update = current_time

            # Display metrics using the cached data
            if 'campaign_data' in st.session_state:
                metrics = st.session_state.campaign_data
                cols = metrics_container.columns(len(metrics))
                for col, (label, value) in zip(cols, metrics.items()):
                    col.metric(label, value)

            # Status and control section using HTML for efficiency
            status = "Active" if st.session_state.automation_status else "Inactive"
            status_color = "green" if st.session_state.automation_status else "red"
            
            status_container.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0;">
                    <div style="padding: 0.5rem 1rem; background: {status_color}; color: white; border-radius: 4px;">
                        Status: {status}
                    </div>
                    <div style="flex-grow: 1;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Control buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Start Automation" if not st.session_state.automation_status else "Stop Automation",
                    use_container_width=True,
                    key='automation_toggle'
                ):
                    # Toggle automation status
                    st.session_state.automation_status = not st.session_state.automation_status
                    
                    if st.session_state.automation_status:
                        # Starting automation
                        st.session_state.automation_logs = []
                        # Clear metrics cache to force refresh
                        st.session_state.last_metrics_update = 0
                        if 'campaign_data' in st.session_state:
                            del st.session_state.campaign_data
                        update_log(log_container, "Automation started")
                        
                        # Start automation thread
                        def automation_loop():
                            while st.session_state.automation_status:
                                try:
                                    with db_session() as session:
                                        ai_automation_loop(session, log_container, None)
                                except Exception as e:
                                    update_log(log_container, f"Error in automation loop: {str(e)}", "error")
                                time.sleep(60)

                        thread = threading.Thread(target=automation_loop, daemon=True)
                        thread.start()
                        st.session_state.automation_thread = thread
                    else:
                        # Stopping automation
                        update_log(log_container, "Automation stopped")
                        # Clean up thread
                        if st.session_state.automation_thread and st.session_state.automation_thread.is_alive():
                            st.session_state.automation_thread = None
                        # Clear metrics cache
                        st.session_state.last_metrics_update = 0
                        if 'campaign_data' in st.session_state:
                            del st.session_state.campaign_data

            with col2:
                if st.button("Clear Logs", use_container_width=True):
                    st.session_state.automation_logs = []
                    st.session_state.last_log_count = 0
                    if 'log_buffer' in st.session_state:
                        st.session_state.log_buffer = []
                    update_log(log_container, "Logs cleared")

            # Display logs with optimized updates
            display_logs(log_container, st.session_state.automation_logs)

    except Exception as e:
        st.error(f"An error occurred in the automation control panel: {str(e)}")
        logging.error(f"Automation panel error: {str(e)}")
        # Clean up on error
        st.session_state.automation_status = False
        if st.session_state.automation_thread and st.session_state.automation_thread.is_alive():
            st.session_state.automation_thread = None

def display_logs(log_container, logs):
    """Enhanced log display with animations and filtering"""
    if not logs:
        log_container.markdown(
            '<div class="log-container empty-logs">No logs to display yet.</div>',
            unsafe_allow_html=True
        )
        return

    # Initialize session state for log management
    if 'log_state' not in st.session_state:
        st.session_state.log_state = {
            'last_count': 0,
            'buffer': [],
            'css_injected': False,
            'update_counter': 0,
            'last_update': time.time(),
            'filter': 'all',
            'expanded': True
        }

    # Enhanced CSS with animations and themes
    if not st.session_state.log_state['css_injected']:
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
                position: relative;
            }
            .log-entry {
                margin-bottom: 0.5rem;
                padding: 0.5rem;
                border-radius: 4px;
                background-color: rgba(28, 131, 225, 0.05);
                border-left: 3px solid #1c83e1;
                transition: all 0.2s ease;
                opacity: 0;
                animation: fadeIn 0.3s ease forwards;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .log-entry:hover {
                transform: translateX(5px);
                background-color: rgba(28, 131, 225, 0.1);
            }
            .log-icon {
                font-size: 1.2em;
                min-width: 24px;
                text-align: center;
            }
            .log-timestamp {
                color: #666;
                font-size: 0.8em;
                min-width: 80px;
            }
            .log-message {
                flex-grow: 1;
            }
            .log-controls {
                position: sticky;
                top: 0;
                background: white;
                padding: 0.5rem;
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
                z-index: 100;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .log-error { border-left-color: #dc3545; background-color: rgba(220, 53, 69, 0.05); }
            .log-success { border-left-color: #28a745; background-color: rgba(40, 167, 69, 0.05); }
            .log-warning { border-left-color: #ffc107; background-color: rgba(255, 193, 7, 0.05); }
            .log-email { border-left-color: #6f42c1; background-color: rgba(111, 66, 193, 0.05); }
            .log-search { border-left-color: #17a2b8; background-color: rgba(23, 162, 184, 0.05); }
            .empty-logs {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100px;
                color: #666;
                font-style: italic;
            }
            .copy-button {
                padding: 2px 6px;
                font-size: 0.8em;
                color: #666;
                background: none;
                border: 1px solid #ddd;
                border-radius: 3px;
                cursor: pointer;
                display: none;
            }
            .log-entry:hover .copy-button {
                display: inline-block;
            }
            .copy-button:hover {
                background: #f8f9fa;
                color: #333;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.session_state.log_state['css_injected'] = True

    current_count = len(logs)
    current_time = time.time()
    
    # Add log filtering controls
    with log_container:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            filter_options = {
                'all': 'All Logs',
                'error': 'Errors Only',
                'success': 'Success Only',
                'email': 'Email Logs',
                'search': 'Search Logs'
            }
            selected_filter = st.selectbox(
                "Filter Logs",
                options=list(filter_options.keys()),
                format_func=lambda x: filter_options[x],
                key='log_filter'
            )
        with col2:
            st.checkbox("Auto-scroll", value=True, key='auto_scroll')
        with col3:
            if st.button("Export Logs"):
                # Create downloadable log file
                log_text = "\n".join([f"[{datetime.now().strftime('%H:%M:%S')}] {log}" for log in logs])
                st.download_button(
                    "Download Logs",
                    log_text,
                    file_name=f"search_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

    # Only update if conditions are met
    should_update = (
        current_count - st.session_state.log_state['last_count'] >= 5 or
        current_time - st.session_state.log_state['last_update'] > 2 or
        st.session_state.log_state['update_counter'] < 5
    )

    if should_update:
        # Get new logs
        new_logs = logs[st.session_state.log_state['last_count']:]
        if new_logs:
            # Update buffer with new logs
            st.session_state.log_state['buffer'].extend(new_logs)
            st.session_state.log_state['last_count'] = current_count
            
            # Keep buffer size in check
            MAX_LOGS = 100
            if len(st.session_state.log_state['buffer']) > MAX_LOGS:
                st.session_state.log_state['buffer'] = st.session_state.log_state['buffer'][-MAX_LOGS:]

            # Format logs with icons and classes
            formatted_logs = []
            for log in st.session_state.log_state['buffer']:
                # Skip if filtered
                if selected_filter != 'all':
                    if selected_filter == 'error' and 'error' not in log.lower():
                        continue
                    if selected_filter == 'success' and 'success' not in log.lower():
                        continue
                    if selected_filter == 'email' and 'email' not in log.lower():
                        continue
                    if selected_filter == 'search' and 'search' not in log.lower():
                        continue

                log_class = 'log-entry'
                if 'error' in log.lower():
                    icon, log_class = '🔴', 'log-entry log-error'
                elif 'success' in log.lower() or 'completed' in log.lower():
                    icon, log_class = '🟢', 'log-entry log-success'
                elif 'warning' in log.lower():
                    icon, log_class = '🟡', 'log-entry log-warning'
                elif 'email' in log.lower():
                    icon, log_class = '📧', 'log-entry log-email'
                elif 'search' in log.lower():
                    icon, log_class = '🔍', 'log-entry log-search'
                else:
                    icon = '🔵'
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_logs.append(
                    f"""
                    <div class="{log_class}">
                        <span class="log-icon">{icon}</span>
                        <span class="log-timestamp">[{timestamp}]</span>
                        <span class="log-message">{log}</span>
                        <button class="copy-button" onclick="navigator.clipboard.writeText('{timestamp} {log}')">
                            Copy
                        </button>
                    </div>
                    """
                )

            # Update display with all logs at once
            log_container.markdown(
                f"""
                <div class="log-container" id="log-container">
                    {''.join(formatted_logs)}
                </div>
                <script>
                    (function() {{
                        const container = document.getElementById('log-container');
                        if (container && {str(st.session_state.get('auto_scroll', True)).lower()}) {{
                            container.scrollTop = container.scrollHeight;
                            container.style.scrollBehavior = 'smooth';
                        }}
                    }})();
                </script>
                """,
                unsafe_allow_html=True
            )

            # Update state
            st.session_state.log_state['last_update'] = current_time
            st.session_state.log_state['update_counter'] += 1

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

def get_page_description(url):
    """Extract page description from HTML meta tags"""
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc['content'] if meta_desc else "No description found"
    except Exception as e:
        logging.error(f"Error getting page description for {url}: {str(e)}")
        return "Error fetching description"

def get_campaign_effectiveness(session, campaign_id):
    """Get campaign effectiveness metrics"""
    try:
        campaign = session.query(Campaign).get(campaign_id)
        if not campaign:
            return None
            
        total_sent = session.query(EmailCampaign).filter_by(campaign_id=campaign_id).count()
        if total_sent == 0:
            return {"total_sent": 0, "open_rate": 0.0, "click_rate": 0.0}
            
        opened = session.query(EmailCampaign).filter(
            EmailCampaign.campaign_id == campaign_id,
            EmailCampaign.opened_at.isnot(None)
        ).count()
        
        clicked = session.query(EmailCampaign).filter(
            EmailCampaign.campaign_id == campaign_id,
            EmailCampaign.clicked_at.isnot(None)
        ).count()
        
        return {
            "total_sent": total_sent,
            "open_rate": (opened / total_sent) * 100,
            "click_rate": (clicked / total_sent) * 100
        }
    except Exception as e:
        logging.error(f"Error getting campaign effectiveness: {str(e)}")
        return None

def start_background_manual_search(search_terms, num_results, ignore_previously_fetched=True, 
                                 optimize_english=False, optimize_spanish=False, 
                                 shuffle_keywords_option=False, language='ES',
                                 one_email_per_url=True, one_email_per_domain=True):
    """Start a background search process"""
    process_id = str(uuid.uuid4())
    
    def search_worker():
        try:
            with db_session() as session:
                process_state = session.query(BackgroundProcessState).filter_by(process_id=process_id).first()
                if not process_state:
                    process_state = BackgroundProcessState(
                        process_id=process_id,
                        status='running',
                        total_items=len(search_terms),
                        processed_items=0
                    )
                    session.add(process_state)
                    session.commit()
                
                for i, term in enumerate(search_terms):
                    try:
                        results = manual_search(
                            session,
                            [term],
                            num_results,
                            ignore_previously_fetched,
                            optimize_english,
                            optimize_spanish,
                            shuffle_keywords_option,
                            language,
                            one_email_per_url,
                            one_email_per_domain
                        )
                        
                        process_state.processed_items = i + 1
                        process_state.progress = ((i + 1) / len(search_terms)) * 100
                        session.commit()
                        
                    except Exception as e:
                        process_state.error_message = str(e)
                        process_state.status = 'error'
                        session.commit()
                        return
                
                process_state.status = 'completed'
                process_state.completed_at = datetime.now()
                session.commit()
                
        except Exception as e:
            with db_session() as session:
                process_state = session.query(BackgroundProcessState).filter_by(process_id=process_id).first()
                if process_state:
                    process_state.status = 'error'
                    process_state.error_message = str(e)
                    session.commit()
    
    thread = threading.Thread(target=search_worker)
    thread.daemon = True
    thread.start()
    
    return process_id

def get_email_preview(session, template_id, from_email, reply_to):
    """Get HTML preview of an email template"""
    template = session.query(EmailTemplate).get(template_id)
    if not template:
        return "Template not found"
    
    subject = template.subject
    body = template.body_content
    
    # Wrap email body in HTML template
    wrapped_body = wrap_email_body(body)
    
    # Generate email preview
    email_preview = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ccc; border-radius: 5px;">
        <h2 style="color: #333;">{subject}</h2>
        <div style="color: #555;">{wrapped_body}</div>
    </div>
    """
    
    return email_preview

def wrap_email_body(body_content):
    """Wrap email body content with HTML template"""
    return f"""
    <div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.5; color: #333;">
        {body_content}
    </div>
    """

def bulk_send_page():
    """Bulk email sending page implementation"""
    st.title("Bulk Email Sending")
    
    with db_session() as session:
        # Fetch templates and settings
        templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)
        
        if not templates or not email_settings:
            st.error("No email templates or settings available. Please set them up first.")
            return

        # Template selection
        col1, col2 = st.columns(2)
        with col1:
            template_option = st.selectbox(
                "Email Template", 
                options=[f"{t.id}: {t.template_name}" for t in templates],
                format_func=lambda x: x.split(":")[1].strip()
            )
            template_id = int(template_option.split(":")[0])
            template = session.query(EmailTemplate).filter_by(id=template_id).first()

        # Email settings
        with col2:
            email_setting = st.selectbox(
                "From Email", 
                options=[f"{s.id}: {s.email}" for s in email_settings],
                format_func=lambda x: f"{x.split(':')[1].strip()}"
            )
            if email_setting:
                setting_id = int(email_setting.split(":")[0])
                setting = next((s for s in email_settings if s.id == setting_id), None)
                if setting:
                    from_email = setting.email
                    reply_to = st.text_input("Reply To", value=setting.email)

        # Send options
        send_option = st.radio(
            "Send to:", 
            ["All Leads", "Specific Email", "Leads from Chosen Search Terms", "Leads from Search Term Groups"]
        )
        
        specific_email = None
        selected_terms = None
        
        if send_option == "Specific Email":
            specific_email = st.text_input("Enter email")
            if specific_email and not is_valid_email(specific_email):
                st.error("Please enter a valid email address")
                return
                
        elif send_option == "Leads from Chosen Search Terms":
            search_terms_with_counts = fetch_search_terms_with_lead_count(session)
            selected_terms = st.multiselect(
                "Select Search Terms",
                options=search_terms_with_counts['Term'].tolist()
            )
            selected_terms = [term.split(" (")[0] for term in selected_terms]
            
        elif send_option == "Leads from Search Term Groups":
            groups = session.query(SearchTermGroup).all()
            selected_groups = st.multiselect(
                "Select Search Term Groups",
                options=[f"{g.id}: {g.name}" for g in groups]
            )
            if selected_groups:
                group_ids = [int(group.split(':')[0]) for group in selected_groups]
                selected_terms = [t.term for t in session.query(SearchTerm).filter(SearchTerm.group_id.in_(group_ids)).all()]

        exclude_previously_contacted = st.checkbox("Exclude Previously Contacted Domains", value=True)

        # Email preview
        st.markdown("### Email Preview")
        st.text(f"From: {from_email}\nReply-To: {reply_to}\nSubject: {template.subject}")
        st.components.v1.html(
            get_email_preview(session, template_id, from_email, reply_to),
            height=600,
            scrolling=True
        )

        # Fetch and filter leads
        leads = fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted)
        total_leads = len(leads)
        eligible_leads = [lead for lead in leads if lead.get('language', template.language) == template.language]
        contactable_leads = [lead for lead in eligible_leads if not (exclude_previously_contacted and lead.get('domain_contacted', False))]

        # Display lead statistics
        st.info(
            f"Total leads: {total_leads}\n"
            f"Leads matching template language ({template.language}): {len(eligible_leads)}\n"
            f"Leads to be contacted: {len(contactable_leads)}"
        )

        # Send emails
        if st.button("Send Emails", type="primary"):
            if not contactable_leads:
                st.warning("No leads found matching the selected criteria.")
                return
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            log_container = st.empty()
            
            logs, sent_count = bulk_send_emails(
                session, 
                template_id, 
                from_email, 
                reply_to, 
                contactable_leads,
                progress_bar,
                status_text,
                results,
                log_container
            )
            
            st.success(f"Emails sent successfully to {sent_count} leads.")
            
            # Display results
            st.subheader("Sending Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Calculate and display success rate
            success_rate = (results_df['Status'] == 'sent').mean()
            st.metric("Email Sending Success Rate", f"{success_rate:.2%}")

def view_leads_page():
    """View and manage leads page implementation"""
    st.title("Lead Management")
    
    with db_session() as session:
        # Fetch all leads with their sources
        leads_data = fetch_leads_with_sources(session)
        
        if not leads_data:
            st.warning("No leads found in the database.")
            return
            
        # Dashboard metrics
        total_leads = len(leads_data)
        contacted_leads = sum(1 for lead in leads_data if any(campaign.status == 'sent' for campaign in lead.email_campaigns))
        conversion_rate = (contacted_leads / total_leads) * 100 if total_leads > 0 else 0
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Leads", total_leads)
        with col2:
            st.metric("Contacted Leads", contacted_leads)
        with col3:
            st.metric("Contact Rate", f"{conversion_rate:.1f}%")
            
        # Search and filter
        search_term = st.text_input("Search leads (email, name, company)", "")
        
        # Prepare data for display
        filtered_leads = []
        for lead in leads_data:
            if search_term.lower() in lead.email.lower() or \
               (lead.first_name and search_term.lower() in lead.first_name.lower()) or \
               (lead.company and search_term.lower() in lead.company.lower()):
                filtered_leads.append({
                    'ID': lead.id,
                    'Email': lead.email,
                    'First Name': lead.first_name or '',
                    'Last Name': lead.last_name or '',
                    'Company': lead.company or '',
                    'Job Title': lead.job_title or '',
                    'Source': lead.lead_sources[0].url if lead.lead_sources else 'Unknown',
                    'Last Contact': max([c.sent_at for c in lead.email_campaigns]) if lead.email_campaigns else None,
                    'Status': next((c.status for c in lead.email_campaigns if c.sent_at), 'Not contacted')
                })
        
        # Convert to DataFrame for display
        df = pd.DataFrame(filtered_leads)
        
        # Add edit functionality
        edited_df = st.data_editor(
            df,
            hide_index=True,
            column_config={
                'ID': st.column_config.NumberColumn('ID', disabled=True),
                'Email': st.column_config.TextColumn('Email', disabled=True),
                'First Name': st.column_config.TextColumn('First Name'),
                'Last Name': st.column_config.TextColumn('Last Name'),
                'Company': st.column_config.TextColumn('Company'),
                'Job Title': st.column_config.TextColumn('Job Title'),
                'Source': st.column_config.TextColumn('Source', disabled=True),
                'Last Contact': st.column_config.DatetimeColumn('Last Contact', disabled=True),
                'Status': st.column_config.TextColumn('Status', disabled=True)
            },
            num_rows="dynamic"
        )
        
        # Save changes button
        if st.button("Save Changes", type="primary"):
            try:
                for index, row in edited_df.iterrows():
                    lead_id = row['ID']
                    lead = session.query(Lead).get(lead_id)
                    if lead:
                        lead.first_name = row['First Name']
                        lead.last_name = row['Last Name']
                        lead.company = row['Company']
                        lead.job_title = row['Job Title']
                session.commit()
                st.success("Changes saved successfully!")
            except Exception as e:
                st.error(f"Error saving changes: {str(e)}")
                session.rollback()
        
        # Export functionality
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="leads_export.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        # Lead growth visualization
        st.subheader("Lead Growth Over Time")
        lead_dates = [lead.created_at for lead in leads_data]
        lead_growth_df = pd.DataFrame({'date': lead_dates})
        lead_growth_df['count'] = 1
        lead_growth_df = lead_growth_df.set_index('date').resample('D').sum().cumsum()
        
        fig = px.line(
            lead_growth_df, 
            title='Cumulative Lead Growth',
            labels={'date': 'Date', 'count': 'Total Leads'}
        )
        st.plotly_chart(fig)
        
        # Email campaign performance
        st.subheader("Email Campaign Performance")
        campaign_stats = {
            'sent': sum(1 for lead in leads_data for c in lead.email_campaigns if c.status == 'sent'),
            'opened': sum(1 for lead in leads_data for c in lead.email_campaigns if c.status == 'opened'),
            'clicked': sum(1 for lead in leads_data for c in lead.email_campaigns if c.status == 'clicked')
        }
        
        campaign_df = pd.DataFrame({
            'Status': ['Sent', 'Opened', 'Clicked'],
            'Count': [campaign_stats['sent'], campaign_stats['opened'], campaign_stats['clicked']]
        })
        
        fig = px.bar(
            campaign_df,
            x='Status',
            y='Count',
            title='Email Campaign Performance'
        )
        st.plotly_chart(fig)

def search_terms_page():
    """Search terms management page implementation"""
    st.title("Search Terms Management")
    
    with db_session() as session:
        # Fetch existing search terms with lead counts
        search_terms_df = fetch_search_terms_with_lead_count(session)
        
        # Add new search term
        st.subheader("Add New Search Term")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_term = st.text_input("Enter new search term")
        with col2:
            if st.button("Add Term", type="primary"):
                if new_term:
                    try:
                        add_search_term(session, new_term, get_active_campaign_id())
                        st.success(f"Added search term: {new_term}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding search term: {str(e)}")
        
        # Search term groups
        st.subheader("Search Term Groups")
        
        # Create new group
        new_group_name = st.text_input("New group name")
        if st.button("Create Group"):
            if new_group_name:
                try:
                    create_search_term_group(session, new_group_name)
                    st.success(f"Created group: {new_group_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating group: {str(e)}")
        
        # Manage existing groups
        groups = session.query(SearchTermGroup).all()
        if groups:
            selected_group = st.selectbox(
                "Select group to manage",
                options=[f"{g.id}: {g.name}" for g in groups],
                format_func=lambda x: x.split(":")[1].strip()
            )
            
            if selected_group:
                group_id = int(selected_group.split(":")[0])
                group = session.query(SearchTermGroup).get(group_id)
                
                # Show current terms in group
                st.write("Current terms in group:")
                current_terms = [term.term for term in group.search_terms]
                for term in current_terms:
                    st.text(f"• {term}")
                
                # Add/remove terms
                available_terms = [term for term in search_terms_df['Term'].tolist() 
                                 if term not in current_terms]
                terms_to_add = st.multiselect(
                    "Add terms to group",
                    options=available_terms
                )
                
                if st.button("Update Group"):
                    try:
                        new_terms = current_terms + terms_to_add
                        term_ids = [term.id for term in session.query(SearchTerm)
                                  .filter(SearchTerm.term.in_(new_terms)).all()]
                        update_search_term_group(session, group_id, term_ids)
                        st.success("Group updated successfully")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating group: {str(e)}")
                
                if st.button("Delete Group", type="secondary"):
                    try:
                        delete_search_term_group(session, group_id)
                        st.success("Group deleted successfully")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting group: {str(e)}")
        
        # Display search terms table
        st.subheader("Search Terms Overview")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            search_filter = st.text_input("Filter terms", "")
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                options=["Term", "Lead Count", "Effectiveness", "Created Date"]
            )
        
        # Filter and sort the dataframe
        filtered_df = search_terms_df[
            search_terms_df['Term'].str.contains(search_filter, case=False)
        ]
        
        if sort_by == "Lead Count":
            filtered_df = filtered_df.sort_values("Lead Count", ascending=False)
        elif sort_by == "Effectiveness":
            filtered_df = filtered_df.sort_values("Effectiveness", ascending=False)
        elif sort_by == "Created Date":
            filtered_df = filtered_df.sort_values("Created At", ascending=False)
        else:
            filtered_df = filtered_df.sort_values("Term")
        
        # Display the table
        st.dataframe(
            filtered_df,
            hide_index=True,
            column_config={
                'Term': st.column_config.TextColumn('Term'),
                'Lead Count': st.column_config.NumberColumn('Lead Count'),
                'Effectiveness': st.column_config.ProgressColumn(
                    'Effectiveness',
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),
                'Created At': st.column_config.DatetimeColumn('Created At'),
                'Group': st.column_config.TextColumn('Group')
            }
        )
        
        # Effectiveness visualization
        st.subheader("Search Term Effectiveness")
        effectiveness_df = filtered_df[['Term', 'Effectiveness']].sort_values('Effectiveness', ascending=True)
        
        fig = px.bar(
            effectiveness_df,
            x='Term',
            y='Effectiveness',
            title='Search Term Effectiveness Comparison'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
        
        # AI-powered term optimization
        st.subheader("AI Term Optimization")
        if st.button("Optimize Search Terms"):
            with st.spinner("Analyzing and optimizing search terms..."):
                try:
                    # Get knowledge base info for context
                    kb_info = get_knowledge_base_info(session, get_active_campaign_id())
                    
                    # Generate optimized terms
                    base_terms = filtered_df['Term'].tolist()
                    optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
                    
                    if optimized_terms:
                        st.success("Generated optimized search terms")
                        for term in optimized_terms:
                            st.write(f"• {term}")
                            
                        if st.button("Add Optimized Terms"):
                            for term in optimized_terms:
                                add_search_term(session, term, get_active_campaign_id())
                            st.success("Added optimized terms to database")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error optimizing terms: {str(e)}")

def email_templates_page():
    """Email templates management page implementation"""
    st.title("Email Templates")
    
    with db_session() as session:
        # Fetch existing templates
        templates = session.query(EmailTemplate).all()
        
        # Template creation/editing section
        st.subheader("Create/Edit Template")
        
        # Template selection for editing
        template_action = st.radio("Action", ["Create New", "Edit Existing"])
        
        if template_action == "Edit Existing" and templates:
            selected_template = st.selectbox(
                "Select template to edit",
                options=[f"{t.id}: {t.template_name}" for t in templates],
                format_func=lambda x: x.split(":")[1].strip()
            )
            template_id = int(selected_template.split(":")[0])
            template = session.query(EmailTemplate).get(template_id)
        else:
            template = None
            
        # Template form
        with st.form("template_form"):
            template_name = st.text_input(
                "Template Name",
                value=template.template_name if template else ""
            )
            
            subject = st.text_input(
                "Subject",
                value=template.subject if template else ""
            )
            
            body_content = st.text_area(
                "Body Content",
                value=template.body_content if template else "",
                height=300
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                language = st.selectbox(
                    "Language",
                    options=["ES", "EN"],
                    index=0 if not template or template.language == "ES" else 1
                )
                
            with col2:
                is_ai_customizable = st.checkbox(
                    "Enable AI Customization",
                    value=template.is_ai_customizable if template else False
                )
            
            # Template variables
            st.subheader("Template Variables")
            st.info("Available variables: {first_name}, {company}, {job_title}")
            
            # Preview section
            if body_content:
                st.subheader("Preview")
                preview_html = get_email_preview(
                    session,
                    template.id if template else None,
                    "example@email.com",
                    "reply@email.com",
                    preview_mode=True,
                    template_content={
                        'subject': subject,
                        'body': body_content
                    }
                )
                st.components.v1.html(preview_html, height=400, scrolling=True)
            
            # Submit button
            submitted = st.form_submit_button("Save Template")
            
            if submitted:
                try:
                    if template:
                        # Update existing template
                        template.template_name = template_name
                        template.subject = subject
                        template.body_content = body_content
                        template.language = language
                        template.is_ai_customizable = is_ai_customizable
                        template.updated_at = datetime.now()
                        session.commit()
                        st.success("Template updated successfully!")
                    else:
                        # Create new template
                        new_template = EmailTemplate(
                            template_name=template_name,
                            subject=subject,
                            body_content=body_content,
                            language=language,
                            is_ai_customizable=is_ai_customizable,
                            project_id=get_active_campaign_id()
                        )
                        session.add(new_template)
                        session.commit()
                        st.success("New template created successfully!")
                    
                    # Refresh the page
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving template: {str(e)}")
        
        # Template list and management
        if templates:
            st.subheader("Existing Templates")
            
            # Create a DataFrame for display
            templates_data = []
            for t in templates:
                templates_data.append({
                    'ID': t.id,
                    'Name': t.template_name,
                    'Language': t.language,
                    'AI Customizable': '✓' if t.is_ai_customizable else '✗',
                    'Created': t.created_at,
                    'Last Updated': t.updated_at or t.created_at
                })
            
            df = pd.DataFrame(templates_data)
            st.dataframe(
                df,
                hide_index=True,
                column_config={
                    'ID': st.column_config.NumberColumn('ID'),
                    'Name': st.column_config.TextColumn('Name'),
                    'Language': st.column_config.TextColumn('Language'),
                    'AI Customizable': st.column_config.TextColumn('AI Customizable'),
                    'Created': st.column_config.DatetimeColumn('Created'),
                    'Last Updated': st.column_config.DatetimeColumn('Last Updated')
                }
            )
            
            # Template deletion
            st.subheader("Delete Template")
            template_to_delete = st.selectbox(
                "Select template to delete",
                options=[f"{t.id}: {t.template_name}" for t in templates],
                format_func=lambda x: x.split(":")[1].strip(),
                key="delete_template"
            )
            
            if st.button("Delete Template", type="secondary"):
                try:
                    template_id = int(template_to_delete.split(":")[0])
                    template = session.query(EmailTemplate).get(template_id)
                    if template:
                        session.delete(template)
                        session.commit()
                        st.success("Template deleted successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error deleting template: {str(e)}")
        
        # AI template generation
        st.subheader("AI Template Generation")
        
        with st.form("ai_template_form"):
            prompt = st.text_area(
                "Describe the email template you want to generate",
                placeholder="Example: Generate a professional email template for reaching out to software engineers about job opportunities..."
            )
            
            kb_info = get_knowledge_base_info(session, get_active_campaign_id())
            
            if st.form_submit_button("Generate Template"):
                if prompt:
                    with st.spinner("Generating template..."):
                        try:
                            generated_template = generate_or_adjust_email_template(
                                prompt,
                                kb_info=kb_info
                            )
                            
                            if generated_template:
                                st.success("Template generated successfully!")
                                
                                # Show preview
                                st.subheader("Generated Template Preview")
                                preview_html = get_email_preview(
                                    session,
                                    None,
                                    "example@email.com",
                                    "reply@email.com",
                                    preview_mode=True,
                                    template_content=generated_template
                                )
                                st.components.v1.html(preview_html, height=400, scrolling=True)
                                
                                # Option to save
                                if st.button("Save Generated Template"):
                                    new_template = EmailTemplate(
                                        template_name=f"AI Generated - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                        subject=generated_template['subject'],
                                        body_content=generated_template['body'],
                                        language="EN",
                                        is_ai_customizable=True,
                                        project_id=get_active_campaign_id()
                                    )
                                    session.add(new_template)
                                    session.commit()
                                    st.success("Generated template saved!")
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Error generating template: {str(e)}")
                else:
                    st.warning("Please provide a prompt for template generation")

def view_campaign_logs():
    """Email campaign logs and analytics page implementation"""
    st.title("Email Campaign Logs")
    
    with db_session() as session:
        # Fetch all email campaigns
        campaigns = session.query(EmailCampaign).order_by(EmailCampaign.sent_at.desc()).all()
        
        if not campaigns:
            st.warning("No email campaigns found.")
            return
            
        # Dashboard metrics
        total_sent = len(campaigns)
        total_opened = sum(1 for c in campaigns if c.opened_at is not None)
        total_clicked = sum(1 for c in campaigns if c.clicked_at is not None)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Emails Sent", total_sent)
        with col2:
            st.metric("Open Rate", f"{(total_opened/total_sent)*100:.1f}%" if total_sent > 0 else "0%")
        with col3:
            st.metric("Click Rate", f"{(total_clicked/total_sent)*100:.1f}%" if total_sent > 0 else "0%")
            
        # Filters
        st.subheader("Filter Campaigns")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(
                    datetime.now() - timedelta(days=30),
                    datetime.now()
                )
            )
        
        with col2:
            status_filter = st.multiselect(
                "Status",
                options=["sent", "opened", "clicked", "bounced", "failed"],
                default=["sent", "opened", "clicked"]
            )
            
        with col3:
            template_filter = st.multiselect(
                "Email Template",
                options=list(set(c.template.template_name for c in campaigns if c.template))
            )
            
        # Filter campaigns based on selection
        filtered_campaigns = []
        for campaign in campaigns:
            # Date filter
            if campaign.sent_at:
                campaign_date = campaign.sent_at.date()
                if not (date_range[0] <= campaign_date <= date_range[1]):
                    continue
                    
            # Status filter
            if status_filter and campaign.status not in status_filter:
                continue
                
            # Template filter
            if template_filter and campaign.template and campaign.template.template_name not in template_filter:
                continue
                
            filtered_campaigns.append({
                'ID': campaign.id,
                'Lead': campaign.lead.email if campaign.lead else 'Unknown',
                'Template': campaign.template.template_name if campaign.template else 'Unknown',
                'Status': campaign.status,
                'Sent At': campaign.sent_at,
                'Opened At': campaign.opened_at,
                'Clicked At': campaign.clicked_at,
                'Open Count': campaign.open_count or 0,
                'Click Count': campaign.click_count or 0
            })
            
        # Convert to DataFrame
        df = pd.DataFrame(filtered_campaigns)
        
        # Display campaigns table
        st.subheader("Campaign Logs")
        st.dataframe(
            df,
            hide_index=True,
            column_config={
                'ID': st.column_config.NumberColumn('ID'),
                'Lead': st.column_config.TextColumn('Lead'),
                'Template': st.column_config.TextColumn('Template'),
                'Status': st.column_config.TextColumn('Status'),
                'Sent At': st.column_config.DatetimeColumn('Sent At'),
                'Opened At': st.column_config.DatetimeColumn('Opened At'),
                'Clicked At': st.column_config.DatetimeColumn('Clicked At'),
                'Open Count': st.column_config.NumberColumn('Opens'),
                'Click Count': st.column_config.NumberColumn('Clicks')
            }
        )
        
        # Analytics section
        st.subheader("Campaign Analytics")
        
        # Time series analysis
        st.subheader("Email Activity Over Time")
        activity_df = pd.DataFrame({
            'Date': [c.sent_at.date() for c in campaigns if c.sent_at],
            'Type': ['Sent'] * len([c for c in campaigns if c.sent_at])
        })
        
        # Add opens and clicks
        activity_df = pd.concat([
            activity_df,
            pd.DataFrame({
                'Date': [c.opened_at.date() for c in campaigns if c.opened_at],
                'Type': ['Opened'] * len([c for c in campaigns if c.opened_at])
            }),
            pd.DataFrame({
                'Date': [c.clicked_at.date() for c in campaigns if c.clicked_at],
                'Type': ['Clicked'] * len([c for c in campaigns if c.clicked_at])
            })
        ])
        
        # Group by date and type
        activity_pivot = activity_df.groupby(['Date', 'Type']).size().unstack(fill_value=0)
        
        # Plot
        fig = px.line(
            activity_pivot,
            title='Email Campaign Activity Over Time',
            labels={'value': 'Count', 'Date': 'Date', 'Type': 'Activity Type'}
        )
        st.plotly_chart(fig)
        
        # Template performance comparison
        st.subheader("Template Performance")
        template_stats = {}
        for campaign in campaigns:
            if campaign.template:
                template_name = campaign.template.template_name
                if template_name not in template_stats:
                    template_stats[template_name] = {
                        'sent': 0,
                        'opened': 0,
                        'clicked': 0
                    }
                template_stats[template_name]['sent'] += 1
                if campaign.opened_at:
                    template_stats[template_name]['opened'] += 1
                if campaign.clicked_at:
                    template_stats[template_name]['clicked'] += 1
                    
        # Convert to DataFrame
        template_df = pd.DataFrame([
            {
                'Template': template,
                'Sent': stats['sent'],
                'Open Rate': (stats['opened'] / stats['sent'] * 100) if stats['sent'] > 0 else 0,
                'Click Rate': (stats['clicked'] / stats['sent'] * 100) if stats['sent'] > 0 else 0
            }
            for template, stats in template_stats.items()
        ])
        
        # Plot
        fig = px.bar(
            template_df,
            x='Template',
            y=['Open Rate', 'Click Rate'],
            title='Template Performance Comparison',
            barmode='group'
        )
        st.plotly_chart(fig)
        
        # Export functionality
        if st.button("Export Logs"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="campaign_logs.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

def settings_page():
    """Application settings page implementation"""
    st.title("Settings")
    
    with db_session() as session:
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
                        project_id=get_active_campaign_id()
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
        
        # Delete email setting
        if email_settings:
            st.subheader("Delete Email Setting")
            setting_to_delete = st.selectbox(
                "Select setting to delete",
                options=[f"{s.id}: {s.name}" for s in email_settings],
                format_func=lambda x: x.split(":")[1].strip()
            )
            
            if st.button("Delete Setting", type="secondary"):
                try:
                    setting_id = int(setting_to_delete.split(":")[0])
                    delete_email_setting(setting_id)
                    st.success("Email setting deleted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting email setting: {str(e)}")
        
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
                create_default_email_settings()
                st.success("Default settings created successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error creating default settings: {str(e)}")

def view_sent_email_campaigns():
    """View sent email campaigns page implementation"""
    st.title("Sent Email Campaigns")
    
    with db_session() as session:
        # Fetch all sent campaigns
        campaigns = session.query(EmailCampaign)\
            .filter(EmailCampaign.sent_at.isnot(None))\
            .order_by(EmailCampaign.sent_at.desc())\
            .all()
            
        if not campaigns:
            st.warning("No sent email campaigns found.")
            return
            
        # Campaign metrics
        total_campaigns = len(campaigns)
        total_unique_leads = len(set(c.lead_id for c in campaigns if c.lead_id))
        avg_open_rate = sum(1 for c in campaigns if c.opened_at) / total_campaigns * 100
        avg_click_rate = sum(1 for c in campaigns if c.clicked_at) / total_campaigns * 100
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Campaigns", total_campaigns)
        with col2:
            st.metric("Unique Leads", total_unique_leads)
        with col3:
            st.metric("Avg. Open Rate", f"{avg_open_rate:.1f}%")
        with col4:
            st.metric("Avg. Click Rate", f"{avg_click_rate:.1f}%")
            
        # Filters
        st.subheader("Filter Campaigns")
        
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(
                    datetime.now() - timedelta(days=30),
                    datetime.now()
                )
            )
        with col2:
            template_filter = st.multiselect(
                "Email Template",
                options=list(set(c.template.template_name for c in campaigns if c.template))
            )
            
        # Prepare campaign data
        campaign_data = []
        for campaign in campaigns:
            # Apply date filter
            if campaign.sent_at:
                campaign_date = campaign.sent_at.date()
                if not (date_range[0] <= campaign_date <= date_range[1]):
                    continue
                    
            # Apply template filter
            if template_filter and campaign.template and \
               campaign.template.template_name not in template_filter:
                continue
                
            # Add campaign to filtered data
            campaign_data.append({
                'ID': campaign.id,
                'Template': campaign.template.template_name if campaign.template else 'Unknown',
                'Lead': campaign.lead.email if campaign.lead else 'Unknown',
                'Subject': campaign.customized_subject or campaign.original_subject,
                'Sent At': campaign.sent_at,
                'Opened': '✓' if campaign.opened_at else '✗',
                'Clicked': '✓' if campaign.clicked_at else '✗',
                'Status': campaign.status
            })
            
        # Convert to DataFrame
        df = pd.DataFrame(campaign_data)
        
        # Display campaigns
        st.dataframe(
            df,
            hide_index=True,
            column_config={
                'ID': st.column_config.NumberColumn('ID'),
                'Template': st.column_config.TextColumn('Template'),
                'Lead': st.column_config.TextColumn('Lead'),
                'Subject': st.column_config.TextColumn('Subject'),
                'Sent At': st.column_config.DatetimeColumn('Sent At'),
                'Opened': st.column_config.TextColumn('Opened'),
                'Clicked': st.column_config.TextColumn('Clicked'),
                'Status': st.column_config.TextColumn('Status')
            }
        )
        
        # Campaign analytics
        st.subheader("Campaign Analytics")
        
        # Daily send volume
        daily_sends = pd.DataFrame({
            'Date': [c.sent_at.date() for c in campaigns if c.sent_at],
            'Count': 1
        }).groupby('Date').sum()
        
        fig = px.line(
            daily_sends,
            title='Daily Email Send Volume',
            labels={'Date': 'Date', 'Count': 'Emails Sent'}
        )
        st.plotly_chart(fig)
        
        # Template performance
        template_stats = {}
        for campaign in campaigns:
            if campaign.template:
                template_name = campaign.template.template_name
                if template_name not in template_stats:
                    template_stats[template_name] = {
                        'sent': 0,
                        'opened': 0,
                        'clicked': 0
                    }
                template_stats[template_name]['sent'] += 1
                if campaign.opened_at:
                    template_stats[template_name]['opened'] += 1
                if campaign.clicked_at:
                    template_stats[template_name]['clicked'] += 1
                    
        template_df = pd.DataFrame([
            {
                'Template': template,
                'Sent': stats['sent'],
                'Open Rate': (stats['opened'] / stats['sent'] * 100) if stats['sent'] > 0 else 0,
                'Click Rate': (stats['clicked'] / stats['sent'] * 100) if stats['sent'] > 0 else 0
            }
            for template, stats in template_stats.items()
        ])
        
        fig = px.bar(
            template_df,
            x='Template',
            y=['Open Rate', 'Click Rate'],
            title='Template Performance',
            barmode='group'
        )
        st.plotly_chart(fig)
        
        # Export functionality
        if st.button("Export Campaign Data"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="sent_campaigns.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        # Campaign details
        st.subheader("Campaign Details")
        selected_campaign = st.selectbox(
            "Select campaign to view details",
            options=[f"{c['ID']}: {c['Subject']}" for c in campaign_data],
            format_func=lambda x: f"ID {x.split(':')[0]}: {x.split(':')[1]}"
        )
        
        if selected_campaign:
            campaign_id = int(selected_campaign.split(':')[0])
            campaign = session.query(EmailCampaign).get(campaign_id)
            
            if campaign:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Campaign Information**")
                    st.write(f"Status: {campaign.status}")
                    st.write(f"Sent At: {campaign.sent_at}")
                    st.write(f"Opened At: {campaign.opened_at or 'Not opened'}")
                    st.write(f"Clicked At: {campaign.clicked_at or 'Not clicked'}")
                    st.write(f"Open Count: {campaign.open_count or 0}")
                    st.write(f"Click Count: {campaign.click_count or 0}")
                    
                with col2:
                    st.write("**Lead Information**")
                    if campaign.lead:
                        st.write(f"Email: {campaign.lead.email}")
                        st.write(f"Name: {campaign.lead.first_name} {campaign.lead.last_name}")
                        st.write(f"Company: {campaign.lead.company or 'Unknown'}")
                        st.write(f"Job Title: {campaign.lead.job_title or 'Unknown'}")
                
                st.write("**Email Content**")
                st.write("Subject:", campaign.customized_subject or campaign.original_subject)
                st.components.v1.html(
                    wrap_email_body(campaign.customized_content or campaign.original_content),
                    height=400,
                    scrolling=True
                )

if __name__ == "__main__":
    main()

