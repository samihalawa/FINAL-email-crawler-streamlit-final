#pip install pip install streamlit redis psutil sqlalchemy boto3 openai pandas plotly beautifulsoup4 email-validator tenacity
import os
import json
import re
import logging
import time
import pandas as pd
import streamlit as st
import boto3
import uuid
import random
import smtplib
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
from sqlalchemy import (func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey,
                       Boolean, JSON, select, text, distinct, and_, case)
from sqlalchemy.orm import (declarative_base, sessionmaker, relationship, Session, joinedload)
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI
from urllib.parse import urlparse, urlencode
from contextlib import contextmanager
from email_validator import validate_email, EmailNotValidError
from typing import Optional, List, Dict, Any, Tuple, Union, Generator, Callable, TypeVar, cast
import plotly.express as px
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import asyncio
import aiohttp
import psutil
import gc
import html
from botocore.exceptions import ClientError
from sqlalchemy.exc import SQLAlchemyError
from redis import Redis
from dotenv import load_dotenv
from aiohttp import ClientSession
from functools import wraps
from sqlalchemy.pool import QueuePool
from streamlit_tags import st_tags
from streamlit_option_menu import option_menu

# Initialize database connection
load_dotenv()

DATABASE_CONFIG = {
    'host': os.getenv("SUPABASE_DB_HOST"),
    'name': os.getenv("SUPABASE_DB_NAME"), 
    'user': os.getenv("SUPABASE_DB_USER"),
    'password': os.getenv("SUPABASE_DB_PASSWORD"),
    'port': os.getenv("SUPABASE_DB_PORT")
}

# Validate database configuration
if not all(DATABASE_CONFIG.values()):
    raise ValueError("Database configuration incomplete. Check .env file.")

DATABASE_URL = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['name']}"

# Initialize engine with better configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=0,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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
    """Lead model with improved relationship definitions"""
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    phone = Column(Text)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text)
    job_title = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_processed = Column(Boolean, default=False)
    source_url = Column(Text)  # Add this line to track source URL directly
    
    # Relationships
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary"""
        return {
            'id': self.id,
            'template_name': self.template_name,
            'subject': self.subject,
            'body_content': self.body_content,
            'language': self.language,
            'is_ai_customizable': self.is_ai_customizable
        }

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
    name = Column(Text)
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
try:
    from redis import Redis
    REDIS_ENABLED = True
except ImportError:
    REDIS_ENABLED = False
    print("Redis not available - caching disabled")

class TemplateCache:
    def __init__(self):
        self.redis_client = Redis(connection_pool=redis_pool) if REDIS_ENABLED else None
        
    def get(self, key: str) -> Optional[Any]:
        if not self.redis_client:
            return None
        try:
            return json.loads(self.redis_client.get(f"template:{key}") or 'null')
        except:
            return None
            
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        if not self.redis_client:
            return
        try:
            self.redis_client.setex(f"template:{key}", ttl, json.dumps(value))
        except:
            pass

template_cache = TemplateCache()

@contextmanager 
def safe_db_connection() -> Generator[Session, None, None]:
    session = None
    for attempt in range(3):
        try:
            session = SessionLocal()
            session.execute(text("SELECT 1"))
            yield session
            session.commit()
            break
        except SQLAlchemyError as e:
            if session: session.rollback()
            if attempt == 2: raise
            time.sleep(1)
        finally:
            if session: session.close()

def get_pool_status() -> dict:
    return {
        'pool_size': engine.pool.size(),
        'checkedin': engine.pool.checkedin(), 
        'checkedout': engine.pool.checkedout(),
        'overflow': engine.pool.overflow()
    }
def settings_page():
    st.title("⚙️ Settings")
    
    with safe_db_connection() as session:
        st.header("General Settings")
        general_settings = session.query(Settings).filter_by(setting_type='general').first() or Settings(
            name='General Settings', setting_type='general', value={})
        
        with st.form("general_settings_form"):
            openai_api_key = st.text_input("OpenAI API Key", 
                value=general_settings.value.get('openai_api_key', ''), type="password")
            openai_api_base = st.text_input("OpenAI API Base URL", 
                value=general_settings.value.get('openai_api_base', 'https://api.openai.com/v1'))
            openai_model = st.text_input("OpenAI Model", 
                value=general_settings.value.get('openai_model', 'gpt-4'))
            
            if st.form_submit_button("Save General Settings"):
                general_settings.value = {
                    'openai_api_key': openai_api_key,
                    'openai_api_base': openai_api_base, 
                    'openai_model': openai_model
                }
                session.add(general_settings)
                session.commit()
                st.success("General settings saved!")

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
                if data['password']: setting.smtp_password = data['password']
            else:
                setting.aws_access_key_id = data['key_id']
                if data['secret']: setting.aws_secret_access_key = data['secret']
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
                region = st.text_input("AWS Region", value="us-east-1")
        
        if st.form_submit_button("Add Email Settings", type="primary"):
            try:
                setting = EmailSettings(
                    name=name, email=email, provider=provider,
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

def send_email_ses(session: Session, from_email: str, to_email: str, subject: str, body: str, 
                   charset: str = 'UTF-8', reply_to: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings: return None, None
    tracking_id = str(uuid.uuid4())
    tracked_body = wrap_email_body(body, tracking_id)
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
            msg['From'], msg['To'], msg['Subject'] = from_email, to_email, subject
            if reply_to: msg['Reply-To'] = reply_to
            msg.attach(MIMEText(tracked_body, 'html'))
            with smtplib.SMTP(email_settings.smtp_server, email_settings.smtp_port) as server:
                server.starttls()
                server.login(email_settings.smtp_username, email_settings.smtp_password)
                server.send_message(msg)
            return {'MessageId': f'smtp-{uuid.uuid4()}'}, tracking_id
        return None, None
    except Exception as e:
        logging.error(f"Email sending error: {e}")
        return None, None

# 2. Optimized save_email_campaign  
def save_email_campaign(
    session: Session, 
    email: str,
    template_id: int,
    status: str,
    sent_at: datetime,
    subject: str,
    message_id: str,
    content: str
) -> bool:
    """Save email campaign with validation"""
    try:
        lead = session.query(Lead).filter_by(email=email).first()
        if not lead:
            logging.error(f"Lead not found: {email}")
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
        logging.error(f"Campaign save error: {e}")
        session.rollback()
        return False

# 3. Optimized is_valid_email
def is_valid_email(email: str) -> bool:
    """Comprehensive email validation with domain checks"""
    if not email or len(email) > 254:
        return False
        
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^(test|prueba)@.*",
        r"^email@email\.com$",
        r".*@example\.com$",
        r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$",
        r"^[0-9]+@"
    ]
    
    try:
        validate_email(email)
        domain = email.split('@')[1]
        return not (
            any(re.match(p, email, re.I) for p in invalid_patterns) or
            domain in ['localhost', 'test.com', 'example.com'] or
            len(domain) < 3
        )
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
def save_lead(session: Session, email: str, **kwargs) -> Optional[Lead]:
    """Enhanced lead saving with validation and deduplication"""
    try:
        if not is_valid_email(email):
            logging.warning(f"Invalid email: {email}")
            return None
            
        existing = session.query(Lead).filter_by(email=email).first()
        if existing:
            for k, v in kwargs.items():
                if v and hasattr(existing, k):
                    setattr(existing, k, v)
            lead = existing
        else:
            lead = Lead(email=email, **kwargs)
            session.add(lead)
            
        session.flush()
        return lead
    except Exception as e:
        logging.error(f"Error saving lead: {e}")
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
def manual_search(session: Session, terms: List[str], num_results: int, ignore_previous: bool = True, optimize_en: bool = False, optimize_es: bool = True, shuffle: bool = False) -> Dict[str, Any]:
    """Search for leads based on terms"""
    results = []
    try:
        for term in terms:
            query = session.query(Lead)
            if ignore_previous:
                query = query.filter(Lead.is_processed == False)
            
            matches = query.limit(num_results).all()
            results.extend([{
                'email': lead.email,
                'company': lead.company,
                'name': f"{lead.first_name} {lead.last_name}".strip(),
                'source': term
            } for lead in matches])
            
        return {'results': results}
    except Exception as e:
        logging.error(f"Search error: {e}")
        return {'results': []}

@handle_error
def generate_or_adjust_email_template(prompt: str, kb_info: Optional[Dict] = None) -> Optional[Dict[str, str]]:
    """Generate email template with error handling"""
    try:
        client = OpenAI()
        context = f"Knowledge base info: {kb_info}\n" if kb_info else ""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user", 
                "content": f"{context}Generate email template for: {prompt}"
            }],
            temperature=0.7
        )
        
        if not response.choices:
            st.error("Failed to generate template")
            return None
            
        content = response.choices[0].message.content
        return {
            'subject': content.split('\n')[0],
            'body_content': '\n'.join(content.split('\n')[1:])
        }
    except Exception as e:
        logging.error(f"Template generation error: {e}")
        st.error("Failed to generate template")
        return None

def fetch_leads_with_sources(session):
    try:
        results = session.query(Lead, func.string_agg(LeadSource.url, ', ').label('sources'), 
                              func.max(EmailCampaign.sent_at).label('last_contact'),
                              func.string_agg(EmailCampaign.status, ', ').label('email_statuses'))\
                        .outerjoin(LeadSource).outerjoin(EmailCampaign)\
                        .filter(Lead.id.in_(session.query(CampaignLead.lead_id)\
                        .filter(CampaignLead.campaign_id == get_active_campaign_id())))\
                        .group_by(Lead.id).all()
        
        return pd.DataFrame([{'id': l.id, 'email': l.email, 'first_name': l.first_name, 'last_name': l.last_name,
                            'company': l.company, 'job_title': l.job_title, 'created_at': l.created_at, 
                            'Source': s, 'Last Contact': lc, 'Last Email Status': es.split(', ')[-1] if es else 'Not Contacted',
                            'Delete': False} for l, s, lc, es in results])
    except Exception as e:
        st.error(f"Error fetching leads: {str(e)}")
        return pd.DataFrame()

def update_lead(session, lead_id, data):
    try:
        update_data = {k:v for k,v in data.items() if k in ['email','first_name','last_name','company','job_title'] and v}
        if update_data:session.query(Lead).filter_by(id=lead_id).update(update_data);session.commit();return True
    except Exception as e:session.rollback();st.error(f"Error updating lead: {e}");return False

def view_leads_page():
    with safe_db_connection() as session:
        leads_df = fetch_leads_with_sources(session)
        if leads_df.empty:return st.warning("No leads found")
        
        c1,c2,c3 = st.columns(3)
        with c1:st.metric("Total Leads",len(leads_df))
        with c2:st.metric("Success Rate",f"{(leads_df['Last Email Status']=='sent').mean()*100:.1f}%")
        with c3:st.metric("Unique Companies",leads_df['company'].nunique())
        
        if 'created_at' in leads_df:
            st.subheader("Lead Growth")
            growth = leads_df.groupby(pd.to_datetime(leads_df['created_at']).dt.to_period('M')).size().cumsum()
            st.line_chart(growth)

def bulk_send_page():
    st.title("Bulk Email Campaign")
    with safe_db_connection() as session:
        templates = session.query(EmailTemplate).filter_by(campaign_id=get_active_campaign_id()).all()
        settings = fetch_email_settings(session)
        if not templates or not settings:return st.error("Setup templates/settings first")
        
        c1,c2 = st.columns([2,1])
        with c1:
            template = session.query(EmailTemplate).get(int(st.selectbox("Template",
                [f"{t.id}: {t.template_name}" for t in templates],
                format_func=lambda x:x.split(":")[1].strip()).split(":")[0]))
            st.markdown("### Preview")
            st.text(f"Subject: {template.subject}")
            st.components.v1.html(wrap_email_body(template.body_content),height=300,scrolling=True)
            lead_selection = st.radio("Recipients",["All","Email","Search Terms","Groups"])
        
        with c2:
            setting = st.selectbox("From",settings,format_func=lambda x:f"{x['name']} ({x['email']})")
            if not setting:return st.error("Select email setting")
            from_email = setting['email']
            reply_to = st.text_input("Reply To",setting['email'])
        
        pc,sc = st.columns(2)
        with pc:
            if st.button("Preview",type="secondary",use_container_width=True):
                leads_data = []
                if lead_selection=="All":leads_data=[{'Email':l.email} for l in session.query(Lead).all()]
                elif lead_selection=="Email":
                    email=st.text_input("Email")
                    if email and is_valid_email(email):leads_data=[{'Email':email}]
                    else:st.error("Invalid email")
                elif lead_selection=="Search Terms":
                    terms = st.multiselect("Terms",[t.term for t in session.query(SearchTerm).all()])
                    if terms:leads_data=[{'Email':l.email} for l in session.query(Lead).join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(terms)).all()]
                else:
                    group = st.selectbox("Group",session.query(SearchTermGroup).all())
                    if group:leads_data=[{'Email':l.email} for l in session.query(Lead).join(LeadSource).join(SearchTerm).filter(SearchTerm.group_id==group.id).all()]
        
        with sc:
            if st.button("Send",type="primary",use_container_width=True):
                if not st.session_state.get('confirm_send'):st.session_state.confirm_send=True;st.warning("Confirm send")
                else:
                    with st.spinner("Sending..."):
                        prog=st.progress(0);status=st.empty();sent=failed=0
                        for i,lead in enumerate(leads_data):
                            try:
                                content=wrap_email_body(template.body_content)
                                msg_id = send_email_ses(session, from_email, lead['Email'], template.subject, content, reply_to)
                                if not msg_id or not msg_id.get('MessageId'):
                                    raise Exception("Failed to get message ID")
                                msg_id = msg_id['MessageId']
                                tracking_id = str(uuid.uuid4())
                                campaign = EmailCampaign(
                                    campaign_id=get_active_campaign_id(),
                                    lead_id=lead['ID'],
                                    template_id=template.id,
                                    status='sent',
                                    sent_at=datetime.utcnow(),
                                    original_subject=template.subject,
                                    original_content=content,
                                    message_id=msg_id,
                                    tracking_id=tracking_id
                                )
                                session.add(campaign)
                                sent+=1;status.success(f"Sent to {lead['Email']}")
                            except Exception as e:failed+=1;status.error(f"Failed: {lead['Email']} - {e}")
                            prog.progress((i+1)/len(leads_data))
                        session.commit()
                        if sent:st.balloons();st.success(f"Sent: {sent}, Failed: {failed}")
                        else:st.error("Send failed")
                        st.session_state.confirm_send=False

def view_campaign_logs():
    """Display automation logs"""
    st.title("Campaign Logs")
    with safe_db_connection() as session:
        logs = session.query(AutomationLog).order_by(AutomationLog.start_time.desc()).all()
        if logs:
            log_data = pd.DataFrame([{
                "Campaign": l.campaign.campaign_name,
                "Start": l.start_time,
                "End": l.end_time,
                "Status": l.status,
                "Leads": l.leads_gathered,
                "Emails": l.emails_sent
            } for l in logs])
            st.dataframe(log_data)
        else:
            st.info("No logs found")

def projects_campaigns_page():
    with safe_db_connection() as session:
        st.header("Projects & Campaigns")
        with st.form("add_project"):
            if st.form_submit_button("Add Project") and (pn:=st.text_input("Name").strip()):
                # Add validation before project creation
                if not pn or len(pn) < 3:
                    st.error("Project name must be at least 3 characters")
                    return
                try:session.add(Project(project_name=pn));session.commit();st.success(f"Added {pn}")
                except SQLAlchemyError as e:st.error(f"Error: {e}")
        for p in session.query(Project).all():
            with st.expander(f"Project: {p.project_name}"):
                st.info("Shared resources within project")
                with st.form(f"add_campaign_{p.id}"):
                    if st.form_submit_button("Add Campaign") and (cn:=st.text_input("Name",key=f"cn_{p.id}").strip()):
                        # Add validation before campaign creation
                        if not cn or len(cn) < 3:
                            st.error("Campaign name must be at least 3 characters")
                            return
                        try:session.add(Campaign(campaign_name=cn,project_id=p.id,created_at=datetime.utcnow()));session.commit();st.success(f"Added {cn}")
                        except SQLAlchemyError as e:st.error(f"Error: {e}")
                campaigns=session.query(Campaign).filter_by(project_id=p.id).all()
                st.write("Campaigns:" if campaigns else "No campaigns")
                [st.write(f"- {c.campaign_name}") for c in campaigns]
        if project_options:=[p.project_name for p in session.query(Project).all()]:
            active_project=st.selectbox("Active Project",options=project_options,index=0)
            active_project_id=session.query(Project.id).filter_by(project_name=active_project).scalar()
            set_active_project_id(active_project_id)
            if active_campaigns:=session.query(Campaign).filter_by(project_id=active_project_id).all():
                campaign_options=[c.campaign_name for c in active_campaigns]
                active_campaign=st.selectbox("Active Campaign",options=campaign_options,index=0)
                active_campaign_id=session.query(Campaign.id).filter_by(campaign_name=active_campaign,project_id=active_project_id).scalar()
                set_active_campaign_id(active_campaign_id)
                st.success(f"Active: {active_project} - {active_campaign}")
            else:st.warning(f"No campaigns in {active_project}")
        else:st.warning("No projects")
        
def knowledge_base_page():
    """Knowledge Base management page for projects"""
    st.title("Knowledge Base")
    with safe_db_connection() as session:
        if not (project_options:=fetch_projects(session)): return st.warning("Create project first")
        
        selected_project=st.selectbox("Project",options=project_options)
        project_id=int(selected_project.split(":")[0])
        st.session_state.active_project_id=project_id
        
        kb=session.query(KnowledgeBase).filter_by(project_id=project_id).first() or KnowledgeBase(project_id=project_id)
        
        with st.form("kb"):
            fields={'kb_name':st.text_input,'kb_bio':st.text_area,'kb_values':st.text_area,
                   'contact_name':st.text_input,'contact_role':st.text_input,'contact_email':st.text_input,
                   'company_description':st.text_area,'company_mission':st.text_area,
                   'company_target_market':st.text_area,'company_other':st.text_area,
                   'product_name':st.text_input,'product_description':st.text_area,
                   'product_target_customer':st.text_area,'product_other':st.text_area,
                   'other_context':st.text_area,'example_email':st.text_area}
            form_data={k:f(k.replace('_',' ').title(),value=getattr(kb,k,'')) for k,f in fields.items()}
            
            if st.form_submit_button("Save"):
                try:
                    for k,v in form_data.items():setattr(kb,k,v)
                    if not kb.id:kb.created_at=datetime.utcnow();session.add(kb)
                    session.commit();st.success("Saved!")
                except Exception as e:st.error(f"Error: {e}")

def delete_lead(s:Session,lid:int)->bool:
    try:
        with s.begin_nested():
            [s.query(m).filter_by(lead_id=lid).delete() for m in [CampaignLead,LeadSource,EmailCampaign,AIRequestLog]]
            s.query(Lead).filter_by(id=lid).delete();return True
    except Exception as e:logging.error(f"Error deleting lead {lid}: {e}");return False
º1
def get_active_project_id() -> Optional[int]:
    """Get current active project ID from session state"""
    return st.session_state.get('active_project_id')
def get_active_campaign_id()->Optional[int]:return st.session_state.get('active_campaign_id')
def test_email_settings(s:Session)->List[Dict[str,Any]]:return[{'name':x.name,'success':test_smtp_connection(x.smtp_server,x.smtp_port,x.smtp_username,x.smtp_password)[0]if x.provider=='smtp'else test_aws_connection(x.aws_access_key_id,x.aws_secret_access_key,x.aws_region)[0],'error':None}for x in s.query(EmailSettings).all()]

def test_smtp_connection(s:str,p:int,u:str,pw:str)->Tuple[bool,str]:
    try:
        with smtplib.SMTP(s,p,timeout=10)as m:m.starttls();m.login(u,pw);return True,"Success"
    except Exception as e:return False,str(e)

def test_aws_connection(k:str,s:str,r:str)->Tuple[bool,str]:
    try:return bool(boto3.Session(aws_access_key_id=k,aws_secret_access_key=s,region_name=r).client('ses').get_send_quota()),"Success"
    except Exception as e:return False,str(e)

@retry(stop=stop_after_attempt(3),wait=wait_random_exponential())
def safe_api_call(func:Callable,*args,**kwargs)->Any:
    try:return func(*args,**kwargs)
    except Exception as e:logging.error(f"API call failed: {e}");raise

def handle_database_error(error:Exception,session:Session)->None:
    error_id=str(uuid.uuid4());logging.error(f"Database error {error_id}: {error}")
    session.rollback();st.error(f"Database error (ID: {error_id}). Please try again.")

def initialize_session_state():
    defaults = {
        'authenticated': False,
        'user': None,
        'page': "Dashboard",
        'active_campaign_id': None,
        'active_project_id': None,
        'automation_status': False,
        'automation_logs': [],
        'confirm_send': False,
        'leads': pd.DataFrame(),
        'template_cache': {},
        'last_refresh': datetime.now()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_authentication() -> bool:
    """Check user authentication"""
    if not st.session_state.get('authenticated'):
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if username == os.getenv("ADMIN_USER") and password == os.getenv("ADMIN_PASS"):
                    st.session_state.authenticated = True
                    return True
                st.error("Invalid credentials")
        return False
    return True

def track_email_engagement(tracking_id: str, event_type: str, url: Optional[str] = None) -> None:
    """Track email opens and clicks with enhanced analytics"""
    try:
        with safe_db_connection() as session:
            campaign = session.query(EmailCampaign).filter_by(tracking_id=tracking_id).first()
            if not campaign:
                return
                
            if event_type == 'open':
                campaign.opened_at = datetime.utcnow()
                campaign.open_count += 1
            elif event_type == 'click':
                campaign.clicked_at = datetime.utcnow()
                campaign.click_count += 1
                if url:
                    engagement_data = campaign.engagement_data or {}
                    engagement_data['clicks'] = engagement_data.get('clicks', []) + [url]
                    campaign.engagement_data = engagement_data
            
            session.commit()
    except Exception as e:
        logging.error(f"Error tracking engagement: {e}")

def monitor_performance() -> Dict[str, Any]:
    """Monitor application performance metrics"""
    try:
        process = psutil.Process()
        with safe_db_connection() as session:
            db_stats = session.execute(text("""
                SELECT 
                    (SELECT COUNT(*) FROM leads) as lead_count,
                    (SELECT COUNT(*) FROM email_campaigns) as campaign_count,
                    (SELECT COUNT(*) FROM email_campaigns WHERE status = 'pending') as pending_count
            """)).first()
            
            return {
                'memory': {
                    'used': process.memory_info().rss / 1024 / 1024,
                    'percent': process.memory_percent()
                },
                'cpu': {
                    'percent': process.cpu_percent(),
                    'threads': process.num_threads()
                },
                'db': {
                    'pool_size': engine.pool.size(),
                    'checkedout': engine.pool.checkedout(),
                    'overflow': engine.pool.overflow()
                },
                'stats': {
                    'leads': db_stats.lead_count,
                    'campaigns': db_stats.campaign_count,
                    'pending': db_stats.pending_count
                }
            }
    except Exception as e:
        logging.error(f"Performance monitoring error: {e}")
        return {}

def sanitize_input(value: str) -> str:
    """Sanitize user input"""
    return html.escape(value.strip())

def validate_email_template(template: Dict[str, Any]) -> bool:
    """Validate email template content"""
    required = ['subject', 'body_content', 'template_name']
    if not all(k in template for k in required):
        return False
    if len(template['subject']) > 998:  # RFC 2822
        return False
    return True

def rate_limit_check(key: str, limit: int, window: int = 3600) -> bool:
    """Check rate limiting"""
    now = time.time()
    redis_key = f"rate_limit:{key}"
    with Redis() as redis:
        current = redis.get(redis_key) or 0
        if int(current) >= limit:
            return False
        redis.incr(redis_key)
        redis.expire(redis_key, window)
    return True

def initialize_session_state():
    """Initialize Streamlit session state with default values"""
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'user' not in st.session_state: st.session_state.user = None
    if 'page' not in st.session_state: st.session_state.page = "Dashboard"

def setup_navigation():
    """Setup sidebar navigation and return selected page"""
    return st.sidebar.selectbox("Navigation", ["Dashboard", "Manual Search", "Automation", 
                                             "Leads", "Email Templates", "Projects & Campaigns",
                                             "Settings", "Logs"])

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def show_login_page():
    """Display login page"""
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")

def add_status_bar():
    """Add persistent status bar with key metrics"""
    status_container = st.empty()
    
    def update_status():
        with safe_db_connection() as session:
            metrics = {
                'DB Pool': f"{engine.pool.checkedout()}/{engine.pool.size()}",
                'Memory': f"{psutil.Process().memory_info().rss/1024/1024:.1f}MB",
                'Leads': session.query(Lead).count(),
                'Campaigns': session.query(EmailCampaign).filter(
                    EmailCampaign.sent_at > datetime.now() - timedelta(hours=24)
                ).count()
            }
        
        status_container.markdown(f"""
        <div style='position:fixed;bottom:0;left:0;right:0;background:#262730;padding:4px 8px;font-size:12px'>
            {'  |  '.join(f'{k}: {v}' for k,v in metrics.items())}
        </div>
        """, unsafe_allow_html=True)
    
    return update_status

def add_keyboard_shortcuts():
    """Add keyboard shortcuts for common actions"""
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 's') { // Ctrl+S to save
            document.querySelector('button:contains("Save")').click();
        } else if (e.ctrlKey && e.key === 'r') { // Ctrl+R to refresh
            document.querySelector('button:contains("Refresh")').click();
        } else if (e.ctrlKey && e.key === 'f') { // Ctrl+F to search
            document.querySelector('input[aria-label="Search"]').focus();
        }
    });
    </script>
    """, unsafe_allow_html=True)

def handle_error(func):
    """Decorator for consistent error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.exception(f"Error in {func.__name__}")
            st.write("Please try refreshing the page or contact support.")
    return wrapper

def add_resource_monitor():
    """Add collapsible resource monitoring panel"""
    with st.expander("🔍 System Resources", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
        with col2:
            st.metric("Memory", f"{psutil.virtual_memory().percent}%")
        with col3:
            st.metric("DB Connections", engine.pool.checkedout())
            
        if st.button("Optimize Resources"):
            with st.spinner("Optimizing..."):
                engine.dispose()
                gc.collect()
                st.success("Resources optimized")

def add_quick_actions():
    """Add quick action buttons for common operations"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Quick Scan", use_container_width=True):
            perform_quick_scan(st.session_state.get('session'))
            
    with col2:
        if st.button("📧 Send Test", use_container_width=True):
            send_test_email()
            
    with col3:
        if st.button("🧹 Clean DB", use_container_width=True):
            clean_database()
            
    with col4:
        if st.button("📊 Analytics", use_container_width=True):
            show_analytics()

def add_code_editor(content: str, language: str = "python"):
    """Add enhanced code editor with syntax highlighting"""
    st.markdown(f"""
    <style>
        .code-editor {{
            font-family: 'JetBrains Mono', monospace;
            background: #1e1e1e;
            padding: 1em;
            border-radius: 4px;
            position: relative;
        }}
        .copy-button {{
            position: absolute;
            top: 8px;
            right: 8px;
            padding: 4px 8px;
            background: #333;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
        }}
    </style>
    <div class="code-editor">
        <pre><code class="{language}">{html.escape(content)}</code></pre>
        <button class="copy-button" onclick="navigator.clipboard.writeText(`{content}`)">
            Copy
        </button>
    </div>
    """, unsafe_allow_html=True)

def track_operation_progress(total_steps: int):
    """Track progress of long-running operations"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(step: int, message: str):
        progress = min(step / total_steps, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"{message} ({step}/{total_steps})")
        
        if progress >= 1.0:
            st.success("Operation completed successfully!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
    
    return update_progress

def preserve_context():
    """Preserve user context between sessions"""
    if 'context' not in st.session_state:
        st.session_state.context = {
            'last_search': None,
            'selected_template': None,
            'active_project': None,
            'filters': {},
            'scroll_position': 0
        }
    
    def update_context(**kwargs):
        st.session_state.context.update(kwargs)
    
    return st.session_state.context, update_context

def validate_input(input_type: str, value: Any) -> Tuple[bool, str]:
    """Smart input validation with helpful messages"""
    validators = {
        'email': lambda x: bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', x)),
        'url': lambda x: bool(re.match(r'https?://(?:[\w-]|\.)+', x)),
        'search_term': lambda x: len(x) >= 3 and not x.isspace()
    }
    
    messages = {
        'email': "Please enter a valid email address",
        'url': "Please enter a valid URL starting with http:// or https://",
        'search_term': "Search term must be at least 3 characters"
    }
    
    is_valid = validators.get(input_type, lambda x: True)(value)
    return is_valid, messages.get(input_type, "") if not is_valid else ""

def display_results(results: List[Dict], key: str):
    """Enhanced results display with filtering and sorting"""
    if not results:
        st.info("No results to display")
        return
        
    # Add search/filter
    search = st.text_input("🔍 Filter results", key=f"search_{key}")
    
    # Convert to dataframe for easier manipulation
    df = pd.DataFrame(results)
    
    if search:
        mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
        df = df[mask]
    
    # Add sorting
    sort_col = st.selectbox("Sort by", df.columns, key=f"sort_{key}")
    sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True, key=f"order_{key}")
    
    df = df.sort_values(sort_col, ascending=sort_order=="Ascending")
    
    # Display with pagination
    rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], key=f"rows_{key}")
    page = st.number_input("Page", min_value=1, max_value=max(1, len(df)//rows_per_page + 1), key=f"page_{key}")
    
    start_idx = (page-1) * rows_per_page
    end_idx = start_idx + rows_per_page
    
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)

def search_terms_page():
    """Search terms management with analytics"""
    with safe_db_connection() as session:
        search_terms_df = pd.DataFrame([(t.term, len(t.leads), len([l for l in t.leads if l.email_campaigns])) 
                                      for t in session.query(SearchTerm).all()],
                                     columns=['Term', 'Lead Count', 'Email Count'])
        
        with st.tabs(["Terms List", "Analytics"])[1]:
            chart_type = st.radio("Chart Type", ["Bar", "Pie"], horizontal=True)
            fig = (px.bar(search_terms_df.nlargest(10, 'Lead Count'),
                         x='Term', y=['Lead Count', 'Email Count'], 
                         title='Top 10 Search Terms',
                         labels={'value': 'Count', 'variable': 'Type'},
                         barmode='group') if chart_type == "Bar" 
                   else px.pie(search_terms_df, values='Lead Count',
                             names='Term', title='Lead Distribution'))
            st.plotly_chart(fig, use_container_width=True)

def email_templates_page():
    """Email template management with AI generation"""
    st.title("Email Templates")
    
    with safe_db_connection() as session:
        # Show existing templates
        templates = session.query(EmailTemplate)\
            .filter_by(project_id=get_active_project_id())\
            .order_by(EmailTemplate.created_at.desc()).all()
            
        if templates:
            st.subheader("Existing Templates")
            for template in templates:
                with st.expander(f"📧 {template.template_name}"):
                    st.text_input("Subject", template.subject, key=f"subj_{template.id}")
                    st.text_area("Content", template.body_content, key=f"body_{template.id}")
                    if st.button("Update", key=f"upd_{template.id}"):
                        template.subject = st.session_state[f"subj_{template.id}"]
                        template.body_content = st.session_state[f"body_{template.id}"]
                        session.commit()
                        st.success("Template updated!")
        
        # Create new template form
        with st.expander("Create New Template"):
            name = st.text_input("Template Name")
            if not name:
                st.warning("Template name required")
            else:
                if st.checkbox("Use AI"):
                    prompt = st.text_area("AI Prompt")
                    kb_info = None
                    if st.checkbox("Use Knowledge Base"):
                        try:
                            kb_info = get_knowledge_base_info(session, get_active_project_id())
                        except Exception as e:
                            st.error(f"Failed to get knowledge base: {e}")
                    
                    if st.button("Generate"):
                        with st.spinner("Generating..."):
                            try:
                                template = generate_or_adjust_email_template(prompt, kb_info)
                                if template and template.get('subject') and template.get('body_content'):
                                    if save_template(session, name, template):
                                        st.success("Template generated and saved!")
                                    else:
                                        st.error("Failed to save template")
                                else:
                                    st.error("Invalid template generated")
                            except Exception as e:
                                st.error(f"Template generation failed: {e}")
                                logging.error(f"AI template error: {e}")

def automation_control_panel_page():
    """Automation control with monitoring"""
    st.title("Automation Control Panel")
    status = st.session_state.get('automation_status', False)
    st.columns([2,1])[0].metric("Status", "Active" if status else "Inactive", 
                               "Running" if status else "Stopped",
                               delta_color="normal" if status else "inverse")

    st.subheader("Real-Time Analytics") 
    try:
        with safe_db_connection() as session:
            metrics = {
                'Total Leads': session.query(Lead).count(),
                'Emails Sent': session.query(EmailCampaign).count(),
                'Success Rate': f"{calculate_success_rate(session):.1f}%",
                'Active Campaigns': session.query(Campaign).filter_by(status='active').count()
            }
            cols = st.columns(4)
            for col, (label, val) in zip(cols, metrics.items()):
                col.metric(label, val)
    except Exception as e:
        st.error(f"Analytics error: {e}")

    if status and st.button("Stop Automation", type="secondary"):
        st.session_state.automation_status = False
        st.success("Automation stopped")

def view_sent_email_campaigns():
    st.header("Sent Email Campaigns")
    try:
        with safe_db_connection() as session:
            df = fetch_sent_email_campaigns(session)
            if not df.empty:
                rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100])
                total_pages = len(df) // rows_per_page + (1 if len(df) % rows_per_page else 0)
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                start_idx = (page - 1) * rows_per_page
                st.dataframe(df.iloc[start_idx:start_idx + rows_per_page])
                if selected := st.selectbox("Select campaign for details", df['ID'].tolist()):
                    try:
                        st.text_area("Content", df[df['ID'] == selected]['Content'].iloc[0] or "No content", height=300)
                    except Exception as e:
                        st.error("Failed to load campaign content")
                        logging.error(f"Campaign content error: {e}")
            else:
                st.info("No sent campaigns found")
    except Exception as e:
        st.error(f"Error loading campaigns: {str(e)}")
        logging.error(f"View campaigns error: {e}")

def autoclient_ai_page():
    st.header("AutoclientAI - Automated Lead Generation")
    with st.expander("Knowledge Base Info", expanded=False):
        with safe_db_connection() as session:
            if not (kb_info := get_knowledge_base_info(session, get_active_project_id())):
                return st.error("Knowledge Base not found. Please set up first.")
            st.json(kb_info)
    
    if not (user_input := st.text_area("Enter context for lead generation:", help="Used to generate targeted search terms")):
        if st.button("Generate Optimized Terms", key="gen_terms"):
            st.warning("Please enter context for term generation")
        return

    if st.button("Generate Optimized Terms", key="gen_terms"):
        with st.spinner("Generating terms..."):
            with safe_db_connection() as session:
                try:
                    base_terms = [t.term for t in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
                    if not base_terms:
                        st.warning("No base terms found")
                        return
                    if terms := generate_optimized_search_terms(session, base_terms, kb_info):
                        st.session_state.optimized_terms = terms
                        st.success("Terms optimized!")
                        st.subheader("Optimized Terms")
                        st.write(", ".join(terms))
                    else:
                        st.error("Invalid terms generated")
                except (SQLAlchemyError, Exception) as e:
                    st.error(f"{'Database' if isinstance(e, SQLAlchemyError) else 'Term generation'} error: {e}")

    if st.button("Start Automation"):
        if not st.session_state.get('optimized_terms'):
            st.warning("Please generate terms first")
        else:
            st.session_state.update({'automation_status': True, 'automation_logs': [], 'total_leads_found': 0, 'total_emails_sent': 0, 'start_time': datetime.now()})
            st.success("Automation started!")

    if st.session_state.get('automation_status'):
        st.subheader("Automation Progress")
        progress, log_area, leads_area = st.progress(0), st.empty(), st.empty()
        try:
            with safe_db_connection() as session:
                ai_automation_loop(session, log_area, leads_area)
        except Exception as e:
            st.error(f"Automation error: {e}")
            st.session_state.automation_status = False
            logging.error(f"Automation loop error: {e}")

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = Redis(host=host, port=port, db=db)
        self.default_ttl = 3600

    def get(self, key: str) -> Optional[str]:
        try:
            return self.redis.get(key)
        except Exception as e:
            logging.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        try:
            return self.redis.set(key, value, ex=ttl or self.default_ttl)
        except Exception as e:
            logging.error(f"Redis set error: {e}")
            return False

redis_cache = RedisCache()

def monitor_system_resources() -> Dict[str, Any]:
    try:
        p = psutil.Process()
        with safe_db_connection() as session:
            return {'memory': p.memory_info().rss / 1024**2, 'cpu': p.cpu_percent(), 'threads': p.num_threads(), 'files': len(p.open_files()), 'db': {'active': engine.pool.checkedout(), 'total': engine.pool.size(), 'overflow': engine.pool.overflow()}, 'stats': {'leads': session.query(Lead).count(), 'campaigns': session.query(Campaign).count(), 'pending': session.query(EmailCampaign).filter_by(status='pending').count()}}
    except Exception as e:
        logging.error(f"Monitoring error: {e}")
        return {}

def extract_visible_text(soup: BeautifulSoup) -> str:
    for s in soup(['script', 'style']): s.extract()
    return ' '.join(chunk for line in soup.get_text().splitlines() for chunk in line.strip().split() if chunk)
def log_search_term_effectiveness(session: Session, term: str, total: int,
                                valid: int, blogs: int, dirs: int) -> None:
    try:
        session.add(SearchTermEffectiveness(
            term=term, total_results=total, valid_leads=valid,
            irrelevant_leads=total - valid, blogs_found=blogs,
            directories_found=dirs
        ))
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Search term logging error: {e}")

def main():
    """Main application entry point with improved structure"""
    st.set_page_config(
        page_title="AutoclientAI",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🤖"
    )
    
    router = PageRouter()
    selected_page = router.setup_navigation()
    router.route(selected_page)

def set_active_project_id(pid: int) -> None:
    st.session_state['active_project_id'] = pid
    if 'active_campaign_id' in st.session_state:
        del st.session_state['active_campaign_id']

def set_active_campaign_id(cid: int) -> None:
    st.session_state['active_campaign_id'] = cid

def get_project_campaigns(session: Session, pid: int) -> List[Campaign]:
    return session.query(Campaign).filter_by(project_id=pid)\
        .order_by(Campaign.created_at.desc()).all()

def fetch_sent_email_campaigns(session: Session) -> pd.DataFrame:
    try:
        campaigns = session.query(EmailCampaign)\
            .filter(EmailCampaign.sent_at.isnot(None))\
            .order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame([{
            'ID': c.id,
            'Lead': c.lead.email if c.lead else 'Unknown',
            'Subject': c.customized_subject or c.original_subject,
            'Status': c.status,
            'Sent At': c.sent_at,
            'Opens': c.open_count,
            'Clicks': c.click_count,
            'Content': c.customized_content or c.original_content
        } for c in campaigns])
    except SQLAlchemyError as e:
        logging.error(f"Campaign fetch error: {e}")
        return pd.DataFrame()

def calculate_success_rate(session: Session) -> float:
    try:
        result = session.query(
            func.count().label('total'),
            func.sum(case((EmailCampaign.status == 'delivered', 1), else_=0)).label('success')
        ).filter(EmailCampaign.sent_at.isnot(None)).first()
        return (result.success / result.total * 100) if result.total else 0.0
    except SQLAlchemyError as e:
        logging.error(f"Success rate error: {e}")
        return 0.0

def ai_automation_loop(session: Session, log_area: Any, leads_area: Any) -> None:
    """Handle AI automation workflow"""
    try:
        terms = st.session_state.get('optimized_terms', [])
        for term in terms:
            results = manual_search(session, [term], 10)
            if results.get('results'):
                log_area.info(f"Found {len(results['results'])} leads for: {term}")
                leads_area.dataframe(pd.DataFrame(results['results']))
            time.sleep(2)  # Rate limiting
    except Exception as e:
        logging.error(f"AI automation error: {e}")
        raise

def generate_optimized_search_terms(session: Session, base_terms: List[str], kb_info: Dict[str, Any]) -> List[str]:
    """Generate optimized search terms using AI"""
    try:
        client = OpenAI()
        prompt = f"""Given these base search terms: {base_terms}
        And this business context: {kb_info}
        Generate 5 optimized search terms that would find relevant leads."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        terms = response.choices[0].message.content.split('\n')
        return [t.strip() for t in terms if t.strip()]
    except Exception as e:
        logging.error(f"Error generating terms: {e}")
        return []

def save_template(session: Session, name: str, template: Dict[str, str]) -> bool:
    """Save email template to database with validation"""
    try:
        if not name or not template.get('subject') or not template.get('body_content'):
            st.error("Template name, subject and content are required")
            return False
            
        new_template = EmailTemplate(
            template_name=name,
            subject=template['subject'],
            body_content=template['body_content'],
            campaign_id=get_active_campaign_id(),
            project_id=get_active_project_id()
        )
        session.add(new_template)
        session.commit()
        return True
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error saving template: {e}")
        st.error("Failed to save template")
        return False

@contextmanager
def db_session():
    """Database session context manager with error handling"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Database error: {e}")
        raise
    finally:
        session.close()

def manual_search_page():
    """Manual search page with enhanced functionality"""
    st.title("Manual Lead Search")
    
    with db_session() as session:
        # Fetch recent searches
        recent_searches = session.query(SearchTerm)\
            .order_by(SearchTerm.created_at.desc())\
            .limit(5).all()
        recent_terms = [term.term for term in recent_searches]
        
        # Email settings and templates
        email_templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)

        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_terms = st_tags(
                label='Enter Search Terms:',
                text='Press enter to add more',
                value=recent_terms,
                suggestions=['software engineer', 'data scientist', 'product manager'],
                maxtags=10,
                key='search_terms_input'
            )
            
            num_results = st.slider("Results per term", 1, 50, 10)

        with col2:
            enable_email = st.checkbox("Enable email sending", value=True)
            ignore_previous = st.checkbox("Ignore previously fetched", value=True)
            shuffle_keywords = st.checkbox("Shuffle Keywords", value=False)
            optimize_en = st.checkbox("Optimize for English", value=False)
            optimize_es = st.checkbox("Optimize for Spanish", value=True)
            language = st.selectbox("Language", ["ES", "EN"], index=0)

        if enable_email:
            if not email_templates or not email_settings:
                st.error("Please configure email templates and settings first")
                return
                
            col3, col4 = st.columns(2)
            with col3:
                template = st.selectbox(
                    "Email Template",
                    options=email_templates,
                    format_func=lambda x: x.split(":")[1].strip()
                )
            with col4:
                email_setting = st.selectbox(
                    "From Email",
                    options=email_settings,
                    format_func=lambda x: f"{x['name']} ({x['email']})"
                )
                if email_setting:
                    from_email = email_setting['email']
                    reply_to = st.text_input("Reply To", from_email)

        if st.button("Search", type="primary"):
            if not search_terms:
                st.warning("Please enter at least one search term")
                return

            with st.spinner("Processing search..."):
                progress = st.progress(0)
                status = st.empty()
                
                try:
                    # Perform search and get results
                    results = manual_search(
                        session=session,
                        terms=search_terms,
                        num_results=num_results,
                        ignore_previous=ignore_previous,
                        optimize_en=optimize_en,
                        optimize_es=optimize_es,
                        shuffle=shuffle_keywords
                    )
                    
                    if not results['results']:
                        st.info("No leads found matching your criteria")
                        return
                        
                    # Process and save leads
                    saved_leads = process_leads(
                        session=session,
                        leads_data=results['results'],
                        single_lead_per_url=single_lead_per_url
                    )
                    
                    if not saved_leads:
                        st.warning("No valid leads could be saved")
                        return
                        
                    # Display results
                    df = pd.DataFrame([{
                        'Email': lead.email,
                        'Company': lead.company,
                        'Name': f"{lead.first_name} {lead.last_name}".strip(),
                        'URL': lead.source_url
                    } for lead in saved_leads])
                    
                    st.success(f"Found and saved {len(saved_leads)} leads!")
                    st.dataframe(df)
                    
                    # Send emails if enabled
                    if enable_email:
                        sent, failed = send_campaign_emails(
                            session=session,
                            leads=saved_leads,
                            template_id=int(template.split(':')[0]),
                            from_email=from_email,
                            reply_to=reply_to,
                            send_to_all_leads=send_to_all_leads
                        )
                        st.success(f"Emails sent: {sent}, Failed: {failed}")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logging.error(f"Search process error: {e}")

        # Add new processing options in a clear section
        st.subheader("Lead Processing Options")
        col_process, col_email = st.columns(2)
        
        with col_process:
            single_lead_per_url = st.checkbox(
                "Process Only One Lead per URL",
                value=True,
                help="Save only the first lead found per URL (recommended for quality)"
            )
        
        with col_email:
            if enable_email:  # Only show if email sending is enabled
                send_to_all_leads = st.checkbox(
                    "Send Email to All Saved Leads",
                    value=False,
                    help="Send emails to all saved leads, not just one per URL"
                )

def require_auth(func: Callable) -> Callable:
    """Authentication decorator for protected routes"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not st.session_state.get('authenticated'):
            st.warning("Please log in to access this page")
            show_login_page()
            return None
        return func(*args, **kwargs)
    return wrapper

# Lines 400-450: Add UIComponents class
class UIComponents:
    """Reusable UI components for consistent interface elements"""
    @staticmethod
    def add_resource_monitor():
        """Display system resource monitoring panel"""
        with st.expander("🔍 System Resources", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
            with col2:
                st.metric("Memory", f"{psutil.virtual_memory().percent}%")
            with col3:
                st.metric("DB Connections", engine.pool.checkedout())

    @staticmethod
    def add_quick_actions():
        """Display quick action buttons for common operations"""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🔄 Quick Scan", use_container_width=True):
                perform_quick_scan(st.session_state.get('session'))
        with col2:
            if st.button("📧 Send Test", use_container_width=True):
                send_test_email()
        with col3:
            if st.button("🧹 Clean DB", use_container_width=True):
                clean_database()
        with col4:
            if st.button("📊 Analytics", use_container_width=True):
                show_analytics()

# Lines 500-550: Add PageRouter class
class PageRouter:
    def __init__(self):
        self.pages = {
            "Dashboard": self.dashboard_page,
            "Manual Search": self.manual_search_page,
            "Automation": self.automation_page,
            "Leads": self.leads_page,
            "Email Templates": self.email_templates_page,
            "Settings": self.settings_page
        }
        self.error_handler = ErrorHandler()
        
    @ErrorHandler.handle_error
    def route(self, page_name: str) -> None:
        if not st.session_state.get('authenticated'):
            st.warning("Please log in to continue")
            self.pages['Login']()
            return
        
        if page_name in self.pages:
            if page_name != 'Dashboard':
                if not st.session_state.get('active_project_id'):
                    st.warning("Please select a project first")
                    self.pages['Projects']()
                    return
                
            self.pages[page_name]()
        else:
            st.error("Page not found")
            self.pages['Dashboard']()

    def setup_navigation(self) -> str:
        if not st.session_state.get('authenticated'):
            return 'Login'
        
        return st.sidebar.selectbox(
            "Navigation",
            list(self.pages.keys()),
            key="nav"
        )

    # Add missing page methods
    @ErrorHandler.handle_error
    def dashboard_page(self):
        st.title("Dashboard")
        UIComponents.add_resource_monitor()
        UIComponents.add_quick_actions()
        
    @ErrorHandler.handle_error
    def manual_search_page(self):
        st.title("Manual Search")
        manual_search_page()
        
    @ErrorHandler.handle_error
    def automation_page(self):
        st.title("Automation")
        autoclient_ai_page()
        
    @ErrorHandler.handle_error
    def leads_page(self):
        st.title("Leads")
        view_leads_page()
        
    @ErrorHandler.handle_error
    def email_templates_page(self):
        st.title("Email Templates")
        email_templates_page()
        
    @ErrorHandler.handle_error
    def settings_page(self):
        st.title("Settings")
        settings_page()

# Lines 900-950: Add error handling utilities
def handle_error(func: Callable) -> Callable:
    """Decorator for consistent error handling across the application"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_id = str(uuid.uuid4())
            logging.error(f"Error {error_id}: {str(e)}")
            st.error(f"An error occurred (ID: {error_id}). Please try again or contact support.")
    return wrapper

# ADD: Connection health check
def check_db_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return False

# ADD: Enhanced caching system
class CacheManager:
    def __init__(self):
        self._cache = {}
        self._ttl = {}
        self._default_ttl = 300  # 5 minutes
        self._max_size = 1000
        self._cleanup_threshold = 0.9
        
    def get(self, key: str) -> Optional[Any]:
        self._cleanup_if_needed()
        if key in self._cache:
            if time.time() < self._ttl[key]:
                return self._cache[key]
            else:
                self._remove_key(key)
        return None
        
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        self._cleanup_if_needed()
        if len(self._cache) >= self._max_size:
            self._remove_oldest()
        self._cache[key] = value
        self._ttl[key] = time.time() + (ttl or self._default_ttl)
        
    def _cleanup_if_needed(self) -> None:
        if len(self._cache) > self._max_size * self._cleanup_threshold:
            self._cleanup()
            
    def _cleanup(self) -> None:
        current_time = time.time()
        expired = [k for k, v in self._ttl.items() if current_time > v]
        for k in expired:
            self._remove_key(k)
            
    def _remove_key(self, key: str) -> None:
        self._cache.pop(key, None)
        self._ttl.pop(key, None)
        
    def _remove_oldest(self) -> None:
        if self._ttl:
            oldest = min(self._ttl.items(), key=lambda x: x[1])[0]
            self._remove_key(oldest)

# UPDATE: Enhanced error handling system
class ErrorHandler:
    @staticmethod
    def handle_error(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_id = str(uuid.uuid4())
                error_context = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
                logging.error(f"Error {error_id}: {error_context}", exc_info=True)
                
                # Attempt recovery
                if isinstance(e, SQLAlchemyError):
                    engine.dispose()
                elif isinstance(e, (ConnectionError, TimeoutError)):
                    time.sleep(1)
                    try:
                        return func(*args, **kwargs)
                    except Exception as retry_e:
                        logging.error(f"Retry failed: {retry_e}")
                
                st.error(f"An error occurred (ID: {error_id}). Please try again or contact support.")
                return None
        return wrapper

# ADD: Enhanced resource monitoring
class ResourceMonitor:
    MEMORY_THRESHOLD = 90  # percent
    CPU_THRESHOLD = 80     # percent
    DB_CONN_THRESHOLD = 80 # percent
    
    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        try:
            process = psutil.Process()
            metrics = {
                'memory': {
                    'used': process.memory_info().rss / 1024 / 1024,
                    'percent': process.memory_percent()
                },
                'cpu': {
                    'percent': process.cpu_percent(),
                    'threads': process.num_threads()
                },
                'db': {
                    'pool_size': engine.pool.size(),
                    'checkedout': engine.pool.checkedout(),
                    'overflow': engine.pool.overflow()
                }
            }
            
            # Check thresholds
            alerts = []
            if metrics['memory']['percent'] > ResourceMonitor.MEMORY_THRESHOLD:
                alerts.append("High memory usage")
                gc.collect()
                
            if metrics['cpu']['percent'] > ResourceMonitor.CPU_THRESHOLD:
                alerts.append("High CPU usage")
                
            db_usage = (metrics['db']['checkedout'] / metrics['db']['pool_size']) * 100
            if db_usage > ResourceMonitor.DB_CONN_THRESHOLD:
                alerts.append("High database connections")
                engine.dispose()
                
            metrics['alerts'] = alerts
            return metrics
            
        except Exception as e:
            logging.error(f"Resource monitoring error: {e}")
            return {'error': str(e)}

def perform_quick_scan(session: Optional[Session] = None) -> None:
    """Perform quick system health check"""
    try:
        with safe_db_connection() as db_session:
            # Check database connection
            db_session.execute(text("SELECT 1"))
            
            # Check basic metrics
            metrics = {
                'db_connections': engine.pool.checkedout(),
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_usage': psutil.cpu_percent()
            }
            
            st.success("Quick scan completed successfully")
            st.json(metrics)
            
    except Exception as e:
        st.error(f"Scan failed: {str(e)}")
        logging.error(f"Quick scan error: {e}")

def send_test_email() -> None:
    """Send test email to verify email settings"""
    try:
        with safe_db_connection() as session:
            settings = session.query(EmailSettings).first()
            if not settings:
                st.error("No email settings configured")
                return
                
            if settings.provider == 'smtp':
                success, message = test_smtp_connection(
                    settings.smtp_server,
                    settings.smtp_port,
                    settings.smtp_username,
                    settings.smtp_password
                )
            else:
                success, message = test_aws_connection(
                    settings.aws_access_key_id,
                    settings.aws_secret_access_key,
                    settings.aws_region
                )
                
            if success:
                st.success("Test email configuration successful")
            else:
                st.error(f"Test failed: {message}")
    except Exception as e:
        st.error(f"Test failed: {str(e)}")

def clean_database() -> None:
    """Clean up stale database records"""
    try:
        with safe_db_connection() as session:
            # Remove failed email campaigns older than 30 days
            thirty_days_ago = datetime.now() - timedelta(days=30)
            session.query(EmailCampaign)\
                .filter(EmailCampaign.status == 'failed')\
                .filter(EmailCampaign.created_at < thirty_days_ago)\
                .delete()
            session.commit()
            st.success("Database cleaned successfully")
    except Exception as e:
        st.error(f"Clean failed: {str(e)}")

def show_analytics() -> None:
    """Show basic system analytics"""
    try:
        with safe_db_connection() as session:
            metrics = {}
            for metric in ['Total Leads', 'Emails Sent', 'Success Rate', 'Active Campaigns']:
                try:
                    if metric == 'Total Leads':
                        metrics[metric] = session.query(Lead).count()
                    elif metric == 'Emails Sent':
                        metrics[metric] = session.query(EmailCampaign).count()
                    elif metric == 'Success Rate':
                        metrics[metric] = f"{calculate_success_rate(session):.1f}%"
                    elif metric == 'Active Campaigns':
                        metrics[metric] = session.query(Campaign).filter_by(status='active').count()
                except Exception as e:
                    metrics[metric] = 'Error'
                    logging.error(f"Metric {metric} error: {e}")
            
            cols = st.columns(4)
            for col, (label, val) in zip(cols, metrics.items()):
                col.metric(label, val)
    except Exception as e:
        st.error("Failed to load analytics")
        logging.error(f"Analytics error: {e}")

# Add threshold checking
MEMORY_THRESHOLD = 90  # percent
CPU_THRESHOLD = 80     # percent

metrics = ResourceMonitor.get_metrics()
if metrics.get('memory', {}).get('percent', 0) > MEMORY_THRESHOLD:
    st.warning("High memory usage detected")
if metrics.get('cpu', {}).get('percent', 0) > CPU_THRESHOLD:
    st.warning("High CPU usage detected")

if __name__ == "__main__":
    main()

# Add cleanup after scan
try:
    gc.collect()
    engine.dispose()
    st.success("Resources cleaned up after scan")
except Exception as e:
    logging.error(f"Cleanup error: {e}")

# Add missing Login page handler at line 1880
def login_page():
    st.title("Login")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")

# Add missing authentication at line 1882
def authenticate_user(username: str, password: str) -> bool:
    # Add your authentication logic here
    return username == "admin" and password == "password"  # Replace with real auth
@contextmanager
def safe_db_operation(session: Session):
    """Context manager for safe database operations with error handling"""
    try:
        yield
        session.commit()
    except Exception as e:
        session.rollback()
        error_id = str(uuid.uuid4())
        logging.error(f"Error {error_id}: {str(e)}")
        st.error(f"An error occurred (ID: {error_id}). Please try again.")

def send_email(to: str, subject: str, body: str) -> Tuple[bool, str, Optional[str]]:
    """Send email using configured provider (SMTP or SES) with tracking"""
    tracking_id = str(uuid.uuid4())
    try:
        with safe_db_connection() as session:
            settings = session.query(EmailSettings).first()
            if not settings:
                return False, "No email settings configured", None
                
            tracked_body = wrap_email_body(body, tracking_id)
            msg = MIMEMultipart()
            msg['From'] = settings.email
            msg['To'] = to
            msg['Subject'] = subject
            msg.attach(MIMEText(tracked_body, 'html'))
            
            if settings.provider == 'smtp':
                with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
                    server.starttls()
                    server.login(settings.smtp_username, settings.smtp_password)
                    server.send_message(msg)
                    return True, "Email sent via SMTP", tracking_id
            else:
                ses = boto3.client('ses',
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.aws_region
                )
                response = ses.send_raw_email(RawMessage={'Data': msg.as_string()})
                return True, f"Email sent via SES (ID: {response['MessageId']})", tracking_id
    except Exception as e:
        logging.error(f"Email sending error: {e}")
        return False, str(e), None

def fetch_projects(session: Session) -> List[str]:
    """Fetch all projects with proper formatting"""
    try:
        return [f"{project.id}: {project.project_name}" 
                for project in session.query(Project).order_by(Project.created_at.desc()).all()]
    except SQLAlchemyError as e:
        logging.error(f"Error fetching projects: {e}")
        return []

def create_email_template(session: Session, name: str, subject: str, body: str, 
                         campaign_id: int, is_ai_customizable: bool = False) -> Optional[int]:
    """Create new email template with validation"""
    try:
        template = EmailTemplate(
            template_name=name,
            subject=subject,
            body_content=body,
            campaign_id=campaign_id,
            is_ai_customizable=is_ai_customizable,
            created_at=datetime.utcnow()
        )
        session.add(template)
        session.commit()
        return template.id
    except SQLAlchemyError as e:
        logging.error(f"Error creating template: {e}")
        session.rollback()
        return None

def update_email_template(session: Session, template_id: int, 
                         subject: str, body: str, 
                         is_ai_customizable: bool) -> bool:
    """Update existing email template"""
    try:
        template = session.query(EmailTemplate).get(template_id)
        if template:
            template.subject = subject
            template.body_content = body
            template.is_ai_customizable = is_ai_customizable
            session.commit()
            return True
        return False
    except SQLAlchemyError as e:
        logging.error(f"Error updating template: {e}")
        session.rollback()
        return False

def bulk_send_emails(session: Session, template_id: int, from_email: str, 
                    reply_to: str, leads: List[Dict], 
                    progress_bar: Optional[Any] = None,
                    status_text: Optional[Any] = None) -> Tuple[List[str], int]:
    """Send bulk emails with progress tracking and error handling"""
    logs = []
    sent_count = 0
    total = len(leads)
    
    template = session.query(EmailTemplate).get(template_id)
    if not template:
        return logs, sent_count
        
    for i, lead in enumerate(leads):
        try:
            if not is_valid_email(lead['Email']):
                logs.append(f"Invalid email: {lead['Email']}")
                continue
                
            response, tracking_id = send_email_ses(
                session=session,
                from_email=from_email,
                to_email=lead['Email'],
                subject=template.subject,
                body=template.body_content,
                reply_to=reply_to
            )
            
            if response and response.get('MessageId'):
                sent_count += 1
                logs.append(f"✅ Sent to {lead['Email']}")
                save_email_campaign(
                    session=session,
                    lead_email=lead['Email'],
                    template_id=template_id,
                    status='sent',
                    sent_at=datetime.utcnow(),
                    subject=template.subject,
                    message_id=response['MessageId'],
                    email_body=template.body_content
                )
            else:
                logs.append(f"❌ Failed to send to {lead['Email']}")
                
            if progress_bar:
                progress_bar.progress((i + 1) / total)
            if status_text:
                status_text.text(f"Processed {i + 1}/{total} leads")
                
        except Exception as e:
            logs.append(f"Error sending to {lead['Email']}: {str(e)}")
            logging.error(f"Email sending error: {e}")
            
    return logs, sent_count

def calculate_analytics(session: Session) -> Dict[str, Any]:
    """Calculate key analytics metrics"""
    try:
        total_leads = session.query(Lead).count()
        total_emails = session.query(EmailCampaign).count()
        success_rate = calculate_success_rate(session)
        
        recent_leads = session.query(Lead)\
            .order_by(Lead.created_at.desc())\
            .limit(10)\
            .all()
            
        return {
            'total_leads': total_leads,
            'total_emails': total_emails,
            'success_rate': success_rate,
            'recent_leads': [
                {
                    'email': lead.email,
                    'company': lead.company,
                    'created_at': lead.created_at
                }
                for lead in recent_leads
            ]
        }
    except SQLAlchemyError as e:
        logging.error(f"Analytics error: {e}")
        return {}

def process_leads(session: Session, leads_data: List[Dict], single_lead_per_url: bool) -> List[Lead]:
    """Process and save leads with URL tracking"""
    processed_urls = set()
    saved_leads = []
    
    try:
        for lead_data in leads_data:
            url = lead_data.get('url')
            email = lead_data.get('email')
            
            # Skip if URL already processed and single lead per URL is enabled
            if single_lead_per_url and url in processed_urls:
                continue
                
            if not is_valid_email(email):
                continue
                
            # Check for existing lead
            lead = session.query(Lead).filter_by(email=email).first()
            if not lead:
                lead = Lead(
                    email=email,
                    first_name=lead_data.get('first_name'),
                    last_name=lead_data.get('last_name'),
                    company=lead_data.get('company'),
                    source_url=url
                )
                session.add(lead)
            
            processed_urls.add(url)
            saved_leads.append(lead)
            
        session.commit()
        return saved_leads
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error while processing leads: {e}")
        return []

def send_campaign_emails(
    session: Session,
    leads: List[Lead],
    template_id: int,
    from_email: str,
    reply_to: Optional[str] = None,
    send_to_all_leads: bool = False
) -> Tuple[int, int]:
    """Send emails with improved URL tracking and lead selection"""
    sent = failed = 0
    processed_urls = set()
    
    try:
        template = session.query(EmailTemplate).get(template_id)
        if not template:
            raise ValueError("Email template not found")
        
        for lead in leads:
            # Skip if we've already sent to this URL and not sending to all
            if not send_to_all_leads and lead.source_url in processed_urls:
                continue
            
            try:
                response, tracking_id = send_email_ses(
                    session=session,
                    from_email=from_email,
                    to_email=lead.email,
                    subject=template.subject,
                    body=template.body_content,
                    reply_to=reply_to
                )
                
                if response and tracking_id:
                    campaign = EmailCampaign(
                        campaign_id=get_active_campaign_id(),
                        lead_id=lead.id,
                        template_id=template_id,
                        status='sent',
                        sent_at=datetime.utcnow(),
                        original_subject=template.subject,
                        original_content=template.body_content,
                        message_id=response.get('MessageId'),
                        tracking_id=tracking_id
                    )
                    session.add(campaign)
                    processed_urls.add(lead.source_url)
                    sent += 1
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                logging.error(f"Failed to send email to {lead.email}: {e}")
                
        session.commit()
        return sent, failed
        
    except Exception as e:
        session.rollback()
        logging.error(f"Email campaign error: {e}")
        return sent, failed
