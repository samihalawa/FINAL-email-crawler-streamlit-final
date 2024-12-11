import os, re, logging, time, requests, pandas as pd, boto3, uuid, urllib3, smtplib, gradio as gr, asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import (func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, Integer, Float, or_, Index)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.exc import SQLAlchemyError
from email_validator import validate_email, EmailNotValidError
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import plotly.express as px
from queue import Queue
import random
import streamlit as st
import json
import openai
import threading
import sys

# Database configuration
load_dotenv()
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME") 
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Database setup
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Database Models
class Project(Base):
    __tablename__ = 'projects'
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text, default="Default Project")
    description = Column(Text)
    project_type = Column(Text)
    status = Column(Text, default="Active")
    priority = Column(Integer, default=3)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaigns = relationship("Campaign", back_populates="project")
    knowledge_bases = relationship("KnowledgeBase", back_populates="project")

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
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    progress = Column(Integer, default=0)
    total_tasks = Column(Integer, default=0)
    completed_tasks = Column(Integer, default=0)
    email_settings_id = Column(BigInteger, ForeignKey('email_settings.id'))

    # Relationships
    project = relationship("Project", back_populates="campaigns")
    tasks = relationship("AutomationTask", back_populates="campaign")
    email_templates = relationship("EmailTemplate", back_populates="campaign")
    search_terms = relationship("SearchTerm", back_populates="campaign")
    search_term_groups = relationship("SearchTermGroup", back_populates="campaign")
    campaign_leads = relationship("CampaignLead", back_populates="campaign")
    automation_logs = relationship("AutomationLog", back_populates="campaign")
    email_settings = relationship("EmailSettings", back_populates="campaigns")
    automation_status = relationship("AutomationStatus", back_populates="campaign")
    email_campaigns = relationship("EmailCampaign", back_populates="campaign")

    __table_args__ = (
        Index('idx_campaign_created_at', 'created_at'),
        Index('idx_campaign_status', 'status'),
    )

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    id = Column(BigInteger, primary_key=True)
    term = Column(Text)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    group_id = Column(BigInteger, ForeignKey('search_term_groups.id'), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    leads = relationship("Lead", back_populates="search_term")
    lead_sources = relationship("LeadSource", back_populates="search_term")
    automation_logs = relationship("AutomationLog", back_populates="search_term")

class SearchTermGroup(Base):
    __tablename__ = 'search_term_groups'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text)
    job_title = Column(Text)
    phone = Column(Text)
    url = Column(Text)
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaign_leads = relationship("CampaignLead", back_populates="lead")
    ai_requests = relationship("AIRequest", back_populates="lead")

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class LeadSource(Base):
    __tablename__ = 'lead_sources'
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    url = Column(Text)
    http_status = Column(Integer)
    scrape_duration = Column(Float)
    page_title = Column(Text)
    meta_description = Column(Text)
    content = Column(Text)
    tags = Column(JSON)
    phone_numbers = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    template_name = Column(Text)
    subject = Column(Text)
    body_content = Column(Text)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False)
    language = Column(Text, default='ES')
    version = Column(Integer, default=1)
    parent_version_id = Column(BigInteger, ForeignKey('email_templates.id'), nullable=True)

    # Relationships
    campaign = relationship("Campaign", back_populates="email_templates")

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    template_id = Column(BigInteger, ForeignKey('email_templates.id'))
    status = Column(Text)
    sent_at = Column(DateTime(timezone=True))
    subject = Column(Text)
    message_id = Column(Text)
    email_body = Column(Text)
    ai_customized = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    open_count = Column(BigInteger, default=0)
    click_count = Column(BigInteger, default=0)

    # Relationships
    campaign = relationship("Campaign", back_populates="email_campaigns")
    lead = relationship("Lead", back_populates="email_campaigns")
    template = relationship("EmailTemplate")
    ai_requests = relationship("AIRequest", back_populates="email_campaign")

    __table_args__ = (
        Index('idx_email_campaign_sent_at', 'sent_at'),
        Index('idx_email_campaign_status', 'status'),
    )

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    email = Column(Text)
    smtp_host = Column(Text)
    smtp_port = Column(Integer)
    smtp_username = Column(Text)
    smtp_password = Column(Text)
    provider = Column(Text, default='smtp')
    aws_access_key_id = Column(Text, nullable=True)
    aws_secret_access_key = Column(Text, nullable=True)
    aws_region = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    campaigns = relationship("Campaign", back_populates="email_settings")

    __table_args__ = (
        Index('idx_email_settings_email', 'email'),
    )

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    id = Column(BigInteger, primary_key=True)
    project_id = Column(BigInteger, ForeignKey('projects.id'))
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    language = Column(Text, default='ES')
    project = relationship("Project", back_populates="knowledge_bases")

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class SearchProcess(Base):
    __tablename__ = 'search_processes'
    id = Column(BigInteger, primary_key=True)
    status = Column(Text)
    total_leads_found = Column(BigInteger, default=0)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    logs = Column(JSON, default=list)

class AIRequest(Base):
    __tablename__ = 'ai_requests'
    id = Column(BigInteger, primary_key=True)
    function_name = Column(Text)
    prompt = Column(Text)
    response = Column(Text)
    lead_id = Column(BigInteger, ForeignKey('leads.id'), nullable=True)
    email_campaign_id = Column(BigInteger, ForeignKey('email_campaigns.id'), nullable=True)
    model_used = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    logs = Column(JSON, default=list)

class CampaignLead(Base):
    __tablename__ = 'campaign_leads'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    status = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class OptimizedSearchTerm(Base):
    __tablename__ = 'optimized_search_terms'
    id = Column(BigInteger, primary_key=True)
    original_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    term = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class SearchTermEffectiveness(Base):
    __tablename__ = 'search_term_effectiveness'
    id = Column(BigInteger, primary_key=True)
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    total_results = Column(BigInteger)
    valid_leads = Column(BigInteger)
    irrelevant_leads = Column(BigInteger)
    blogs_found = Column(BigInteger)
    directories_found = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AutomationLog(Base):
    __tablename__ = 'automation_logs'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    leads_gathered = Column(BigInteger)
    emails_sent = Column(BigInteger)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    status = Column(Text)
    logs = Column(JSON)

class Settings(Base):
    __tablename__ = 'settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    setting_type = Column(Text, nullable=False)
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class AutomationStatus(Base):
    __tablename__ = 'automation_status'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    status = Column(Text)
    started_at = Column(DateTime(timezone=True))
    stopped_at = Column(DateTime(timezone=True))
    paused_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    campaign = relationship("Campaign", back_populates="automation_status")

class AutomationTask(Base):
    __tablename__ = 'automation_tasks'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    task_type = Column(Text)
    status = Column(Text)
    progress = Column(Integer, default=0)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    eta = Column(DateTime(timezone=True))
    logs = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    retry_count = Column(Integer, default=0)

    # Relationships
    campaign = relationship("Campaign", back_populates="tasks")

    __table_args__ = (
        Index('idx_automation_task_status', 'status'),
        Index('idx_automation_task_created_at', 'created_at'),
    )

class AutomationSchedule(Base):
    __tablename__ = 'automation_schedules'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    task_type = Column(Text)
    frequency = Column(Text)
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    time_of_day = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AutomationRule(Base):
    __tablename__ = 'automation_rules'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    rule_type = Column(Text)
    condition = Column(Text)
    threshold = Column(Float)
    action = Column(Text)
    notification_email = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AutomationError(Base):
    __tablename__ = 'automation_errors'
    id = Column(BigInteger, primary_key=True)
    task_type = Column(Text)
    error_type = Column(Text)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Add thread-local storage for database sessions
thread_local = threading.local()

@contextmanager
def get_db():
    """Thread-safe database session context manager with better error handling"""
    if not hasattr(thread_local, "session"):
        try:
            thread_local.session = SessionLocal()
        except SQLAlchemyError as e:
            logging.error(f"Database connection error: {str(e)}")
            raise
    session = thread_local.session
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        session.rollback()
        logging.error(f"Unexpected error: {str(e)}")
        raise
    finally:
        if hasattr(thread_local, "session"):
            session.close()
            del thread_local.session

# Utility functions
def get_active_project_id() -> int:
    return 1  # Default project ID for Gradio app

def get_active_campaign_id() -> int:
    return 1  # Default campaign ID for Gradio app

def manual_search(session: Session, terms: List[str], num_results: int, 
                 ignore_previously_fetched: bool = True, 
                 optimize_english: bool = False,
                 optimize_spanish: bool = False,
                 shuffle_keywords_option: bool = False,
                 language: str = 'ES',
                 enable_email_sending: bool = True,
                 log_container = None,
                 from_email: Optional[str] = None,
                 reply_to: Optional[str] = None,
                 email_template_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Execute manual search with optional email sending
    """
    ua = UserAgent()
    results = []
    total_leads = 0
    domains_processed = set()
    processed_emails_per_domain = {}

    try:
        for original_term in terms:
            try:
                search_term_id = add_or_get_search_term(session, original_term, get_active_campaign_id())
                search_term = shuffle_keywords(original_term) if shuffle_keywords_option else original_term
                search_term = optimize_search_term(search_term, 'english' if optimize_english else 'spanish') if optimize_english or optimize_spanish else search_term
                
                if log_container:
                    log_container.update(value=f"Searching for '{original_term}' (Used '{search_term}')")
                
                for url in google_search(search_term, num_results, lang=language):
                    try:
                        domain = get_domain_from_url(url)
                        if ignore_previously_fetched and domain in domains_processed:
                            if log_container:
                                log_container.update(value=f"Skipping Previously Fetched: {domain}")
                            continue
                        
                        if log_container:
                            log_container.update(value=f"Fetching: {url}")
                        
                        if not url.startswith(('http://', 'https://')):
                            url = 'http://' + url
                        
                        response = requests.get(url, timeout=10, verify=False, headers={'User-Agent': ua.random})
                        response.raise_for_status()
                        html_content = response.text
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        emails = extract_emails_from_html(html_content)
                        valid_emails = [email for email in emails if is_valid_email(email)]
                        
                        if log_container:
                            log_container.update(value=f"Found {len(valid_emails)} valid email(s) on {url}")
                        
                        if not valid_emails:
                            continue
                        
                        name, company, job_title = extract_info_from_page(soup)
                        page_title = get_page_title(html_content)
                        page_description = get_page_description(html_content)
                        
                        if domain not in processed_emails_per_domain:
                            processed_emails_per_domain[domain] = set()
                        
                        for email in valid_emails:
                            if email in processed_emails_per_domain[domain]:
                                continue
                            
                            processed_emails_per_domain[domain].add(email)
                            
                            lead = save_lead(session, email=email, first_name=name,
                                           company=company, job_title=job_title,
                                           url=url, search_term_id=search_term_id,
                                           created_at=datetime.utcnow())
                            
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
                                
                                if log_container:
                                    log_container.update(value=f"Saved lead: {email}")

                                if enable_email_sending and from_email and email_template_id:
                                    template = session.query(EmailTemplate).filter_by(id=email_template_id).first()
                                    if template:
                                        wrapped_content = wrap_email_body(template.body_content)
                                        response, tracking_id = send_email_ses(
                                            session, from_email, email, template.subject,
                                            wrapped_content, reply_to=reply_to
                                        )
                                        
                                        if response:
                                            if log_container:
                                                log_container.update(value=f"Sent email to: {email}")
                                            save_email_campaign(
                                                session, email, template.id, 'Sent',
                                                datetime.utcnow(), template.subject,
                                                response['MessageId'], wrapped_content
                                            )
                                        else:
                                            if log_container:
                                                log_container.update(value=f"Failed to send email to: {email}")
                                            save_email_campaign(
                                                session, email, template.id, 'Failed',
                                                datetime.utcnow(), template.subject,
                                                None, wrapped_content
                                            )
                            else:
                                if log_container:
                                    log_container.update(value=f"Template not found for ID: {email_template_id}")

                    except requests.RequestException as e:
                        if log_container:
                            log_container.update(value=f"Error processing URL {url}: {str(e)}")
                        continue
                    except Exception as e:
                        if log_container:
                            log_container.update(value=f"Unexpected error processing URL {url}: {str(e)}")
                        continue

            except Exception as e:
                if log_container:
                    log_container.update(value=f"Error processing term '{original_term}': {str(e)}")
                continue

    except Exception as e:
        logging.error(f"Fatal error in search process: {str(e)}")
        if log_container:
            log_container.update(value=f"Error: {str(e)}")
        return {"total_leads": 0, "results": [], "error": str(e)}
    
    if log_container:
        log_container.update(value=f"Total leads found: {total_leads}")
    
    return {"total_leads": total_leads, "results": results}

def wrap_email_body(content: str) -> str:
    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .button {{ display: inline-block; padding: 10px 20px; background-color: #007bff;
                      color: white; text-decoration: none; border-radius: 5px; }}
            .signature {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <div class="container">
            {content}
        </div>
    </body>
    </html>
    """

def send_email_ses(session: Session, from_email: str, to_email: str, subject: str, 
                  body: str, charset: str = 'UTF-8', reply_to: Optional[str] = None,
                  ses_client = None) -> tuple:
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

def save_email_campaign(session: Session, lead_email: str, template_id: int, 
                       status: str, sent_at: datetime, subject: str,
                       message_id: Optional[str], email_body: str) -> Optional[EmailCampaign]:
    """Save email campaign to database"""
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
        logging.error(f"Database error saving email campaign: {str(e)}")
        session.rollback()
        return None
    except Exception as e:
        logging.error(f"Error saving email campaign: {str(e)}")
        session.rollback()
        return None

def get_page_description(html_content):
    """Extract meta description from HTML content"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        meta = soup.find('meta', {'name': 'description'}) or soup.find('meta', {'property': 'og:description'})
        return meta['content'] if meta else "No description available"
    except Exception as e:
        logging.error(f"Error getting page description: {str(e)}")
        return "Error fetching description"

def extract_visible_text(soup):
    """Extract visible text from BeautifulSoup object"""
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return ' '.join(chunk for chunk in chunks if chunk)

def is_valid_email(email):
    """Validate email address"""
    if email is None: 
        return False
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^email@email\.com$",
        r".*@example\.com$",
        r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"
    ]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    if any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns): 
        return False
    if any(email.lower().endswith(f"@{domain}") for domain in typo_domains): 
        return False
    try: 
        validate_email(email)
        return True
    except EmailNotValidError: 
        return False

def get_domain_from_url(url: str) -> str:
    """Extract domain from URL"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except Exception as e:
        logging.error(f"Error extracting domain from {url}: {str(e)}")
        return url

def extract_emails_from_html(html_content: str) -> List[str]:
    """Extract email addresses from HTML content"""
    try:
        # Email regex pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, html_content)
        return list(set(emails))  # Remove duplicates
    except Exception as e:
        logging.error(f"Error extracting emails: {str(e)}")
        return []

def extract_info_from_page(soup: BeautifulSoup) -> tuple:
    """Extract name, company and job title from page"""
    try:
        # Default values
        name = company = job_title = ""
        
        # Look for common patterns in meta tags
        meta_tags = {
            'author': ['author', 'article:author'],
            'company': ['organization', 'og:site_name'],
            'job_title': ['job_title', 'position']
        }
        
        for info_type, meta_names in meta_tags.items():
            for meta_name in meta_names:
                meta = soup.find('meta', {'name': meta_name}) or soup.find('meta', {'property': meta_name})
                if meta and meta.get('content'):
                    if info_type == 'author':
                        name = meta['content']
                    elif info_type == 'company':
                        company = meta['content']
                    elif info_type == 'job_title':
                        job_title = meta['content']
        
        return name, company, job_title
    except Exception as e:
        logging.error(f"Error extracting page info: {str(e)}")
        return "", "", ""

def get_page_title(html_content: str) -> str:
    """Extract page title from HTML content"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.find('title')
        return title.string if title else "No title available"
    except Exception as e:
        logging.error(f"Error getting page title: {str(e)}")
        return "Error fetching title"

def add_or_get_search_term(session: Session, term: str, campaign_id: int) -> int:
    """Add new search term or get existing one"""
    try:
        search_term = session.query(SearchTerm).filter_by(
            term=term,
            campaign_id=campaign_id
        ).first()
        
        if not search_term:
            search_term = SearchTerm(
                term=term,
                campaign_id=campaign_id
            )
            session.add(search_term)
            session.commit()
        
        return search_term.id
    except SQLAlchemyError as e:
        logging.error(f"Database error adding search term: {str(e)}")
        session.rollback()
        raise
    except Exception as e:
        logging.error(f"Error adding search term: {str(e)}")
        raise

def save_lead(session: Session, **lead_data) -> Optional[Lead]:
    """Save lead to database"""
    try:
        existing_lead = session.query(Lead).filter_by(email=lead_data['email']).first()
        if existing_lead:
            return existing_lead
            
        new_lead = Lead(**lead_data)
        session.add(new_lead)
        session.commit()
        return new_lead
    except SQLAlchemyError as e:
        logging.error(f"Database error saving lead: {str(e)}")
        session.rollback()
        return None
    except Exception as e:
        logging.error(f"Error saving lead: {str(e)}")
        return None

def remove_invalid_leads(session):
    """Remove invalid leads from database"""
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

def get_knowledge_base_info(session, project_id):
    """Get knowledge base information for a project"""
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

def shuffle_search_term(term: str) -> str:
    """Shuffle words in search term"""
    words = term.split()
    random.shuffle(words)
    return " ".join(words)

def optimize_search_term(term: str, language: str) -> str:
    """Optimize search term for given language"""
    if language == 'english':
        return f'"{term}" email OR contact OR "get in touch" site:.com'
    elif language == 'spanish':
        return f'"{term}" correo OR contacto OR "ponte en contacto" site:.es'
    return term

def is_contacted_lead(lead: Lead) -> bool:
    """Check if lead has been contacted"""
    return bool(lead.email_campaigns)

def is_successful_lead(lead: Lead) -> bool:
    """Check if lead was successfully processed"""
    return bool(lead and lead.email and is_valid_email(lead.email))

def generate_or_adjust_email_template(prompt: str, kb_info: Optional[Dict[str, Any]] = None, current_template: Optional[str] = None) -> Dict[str, str]:
    """Generate or adjust email template using AI"""
    messages = []
    
    if kb_info:
        messages.append({
            "role": "system",
            "content": f"Use this knowledge base info to inform the email: {kb_info}"
        })
    
    if current_template:
        messages.append({
            "role": "system", 
            "content": f"Adjust this template: {current_template}"
        })
        
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    response = openai_chat_completion(messages, "generate_email_template")
    
    return {
        "subject": response.split("\nBody:")[0].replace("Subject:", "").strip(),
        "body": response.split("\nBody:")[1].strip() if "\nBody:" in response else response
    }

def openai_chat_completion(messages: List[Dict[str, str]], function_name: Optional[str] = None, temperature: float = 0.7) -> Any:
    """Execute OpenAI chat completion"""
    try:
        with get_db() as session:
            settings = session.query(Settings).filter_by(setting_type='general').first()
            if not settings or 'openai_api_key' not in settings.value:
                raise ValueError("OpenAI API key not configured")
            
            openai.api_key = settings.value['openai_api_key']
            openai.api_base = settings.value.get('openai_api_base', 'https://api.openai.com/v1')
            
            response = openai.ChatCompletion.create(
                model=settings.value.get('openai_model', 'gpt-3.5-turbo'),
                messages=messages,
                temperature=temperature
            )
            
            # Log the request
            log_entry = AIRequest(
                function_name=function_name,
                prompt=str(messages),
                response=str(response.choices[0].message.content),
                model_used=settings.value.get('openai_model', 'gpt-3.5-turbo')
            )
            session.add(log_entry)
            session.commit()
            
            return response.choices[0].message.content
            
    except Exception as e:
        logging.error(f"Error in openai_chat_completion: {str(e)}")
        return None

class GradioAutoclientApp:
    def __init__(self):
        self.automation_status = False
        self.automation_logs = []
        self.total_leads_found = 0
        self.total_emails_sent = 0
        self.stop_sending_flag = False
        
    def fetch_template_names(self) -> List[str]:
        """Fetch email template names"""
        with get_db() as session:
            templates = session.query(EmailTemplate).all()
            return [f"{t.id}: {t.template_name}" for t in templates]

    def fetch_email_settings_names(self) -> List[str]:
        """Fetch email settings names"""
        with get_db() as session:
            settings = session.query(EmailSettings).all()
            return [f"{s.id}: {s.name} ({s.email})" for s in settings]

    def fetch_sent_campaigns(self) -> List[str]:
        """Fetch sent campaign names"""
        with get_db() as session:
            campaigns = session.query(EmailCampaign).filter(EmailCampaign.sent_at.isnot(None)).all()
            return [f"{c.id}: {c.name}" for c in campaigns]

    def fetch_projects(self) -> List[str]:
        """Fetch project names"""
        with get_db() as session:
            projects = session.query(Project).all()
            return [f"{p.id}: {p.project_name}" for p in projects]

    def fetch_search_terms(self) -> List[str]:
        """Fetch search terms"""
        with get_db() as session:
            terms = session.query(SearchTerm).all()
            return [t.term for t in terms]

    async def perform_search(self, search_terms: str, num_results: int, ignore_fetched: bool,
                           shuffle_keywords: bool, optimize_english: bool, optimize_spanish: bool,
                           language: str, enable_email: bool = False, template_id: Optional[str] = None,
                           email_setting_id: Optional[str] = None, reply_to: Optional[str] = None):
        """Perform search with progress tracking"""
        try:
            # Extract IDs from dropdown values
            template_id_num = int(template_id.split(':')[0]) if template_id else None
            email_setting_id_num = int(email_setting_id.split(':')[0]) if email_setting_id else None

            with get_db() as session:
                terms = [term.strip() for term in search_terms.split('\n') if term.strip()]
                total_terms = len(terms)
                results = {"results": [], "logs": []}
                
                from_email = None
                if enable_email and email_setting_id_num:
                    email_settings = session.query(EmailSettings).get(email_setting_id_num)
                    from_email = email_settings.email if email_settings else None
                
                for i, term in enumerate(terms):
                    term_results = await asyncio.to_thread(
                        manual_search,
                        session=session,
                        terms=[term],
                        num_results=num_results,
                        ignore_previously_fetched=ignore_fetched,
                        optimize_english=optimize_english,
                        optimize_spanish=optimize_spanish,
                        shuffle_keywords_option=shuffle_keywords,
                        language=language,
                        enable_email_sending=enable_email,
                        from_email=from_email,
                        reply_to=reply_to,
                        email_template_id=template_id_num
                    )
                    
                    results["results"].extend(term_results.get("results", []))
                    results["logs"].append(f"Completed search for '{term}': Found {len(term_results.get('results', []))} results")

                return "Search completed", "\n".join(results["logs"]), pd.DataFrame(results["results"])

        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            logging.error(error_msg)
            return f"Error: {str(e)}", error_msg, None

    def create_ui(self):
        """Create the Gradio UI"""
        with gr.Blocks(title="AutoclientAI - Lead Generation Platform") as demo:
            gr.Markdown("""
                # AutoclientAI Lead Generation Platform
                Transform your sales pipeline with AI-powered lead generation
            """)
            
            with gr.Tab("🔍 Manual Search"):
                with gr.Row():
                    with gr.Column(scale=2):
                        search_terms = gr.Textbox(
                            label="Search Terms",
                            placeholder="Enter search terms (one per line)",
                            lines=5
                        )
                        num_results = gr.Slider(
                            label="Results per term",
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1
                        )
                    
                    with gr.Column(scale=1):
                        enable_email = gr.Checkbox(
                            label="Enable email sending",
                            value=True
                        )
                        ignore_fetched = gr.Checkbox(
                            label="Ignore fetched domains",
                            value=True
                        )
                        shuffle_keywords = gr.Checkbox(
                            label="Shuffle Keywords",
                            value=True
                        )
                        optimize_english = gr.Checkbox(
                            label="Optimize (English)",
                            value=False
                        )
                        optimize_spanish = gr.Checkbox(
                            label="Optimize (Spanish)",
                            value=False
                        )
                        language = gr.Dropdown(
                            label="Language",
                            choices=["ES", "EN"],
                            value="ES"
                        )

                email_group = gr.Group(visible=False)
                with email_group:
                    with gr.Row():
                        template_dropdown = gr.Dropdown(
                            label="Email Template",
                            choices=self.fetch_template_names()
                        )
                        email_settings_dropdown = gr.Dropdown(
                            label="From Email",
                            choices=self.fetch_email_settings_names()
                        )
                    reply_to = gr.Textbox(
                        label="Reply To",
                        placeholder="Enter reply-to email address"
                    )

                search_btn = gr.Button("Search", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
                logs = gr.Textbox(label="Logs", interactive=False)
                results = gr.DataFrame(
                    headers=["Email", "URL", "Lead Source", "Title", "Description", "Company", "Name"],
                    label="Search Results"
                )

                # Event handlers
                enable_email.change(
                    fn=lambda x: gr.Group(visible=x),
                    inputs=[enable_email],
                    outputs=[email_group]
                )

                search_btn.click(
                    fn=self.perform_search,
                    inputs=[
                        search_terms, num_results, ignore_fetched,
                        shuffle_keywords, optimize_english, optimize_spanish,
                        language, enable_email, template_dropdown,
                        email_settings_dropdown, reply_to
                    ],
                    outputs=[status, logs, results]
                )

        return demo

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_env_vars():
    """Validate required environment variables"""
    required_vars = ['SUPABASE_DB_HOST', 'SUPABASE_DB_NAME', 'SUPABASE_DB_USER', 
                    'SUPABASE_DB_PASSWORD', 'SUPABASE_DB_PORT']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

def create_db_engine():
    """Create database engine"""
    return create_engine(DATABASE_URL, pool_size=20, max_overflow=0)

def check_system_health():
    """Check system health"""
    try:
        with get_db() as session:
            session.execute("SELECT 1")
        return True, "System healthy"
    except Exception as e:
        return False, f"System health check failed: {str(e)}"

if __name__ == "__main__":
    setup_logging()
    validate_env_vars()
    engine = create_db_engine()
    Base.metadata.create_all(bind=engine)
    
    health_status, health_message = check_system_health()
    if not health_status:
        logging.error(health_message)
        sys.exit(1)
    
    app = GradioAutoclientApp()
    demo = app.create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)