import os, re, logging, time, requests, pandas as pd, boto3, uuid, urllib3, smtplib, gradio as gr, asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import (func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, Integer, Float, or_)
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
    project = relationship("Project", back_populates="campaigns")
    tasks = relationship("AutomationTask", back_populates="campaign")
    email_templates = relationship("EmailTemplate", back_populates="campaign")
    # Add progress tracking
    progress = Column(Integer, default=0)
    total_tasks = Column(Integer, default=0)
    completed_tasks = Column(Integer, default=0)

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    id = Column(BigInteger, primary_key=True)
    term = Column(Text)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    group_id = Column(BigInteger, ForeignKey('search_term_groups.id'), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

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
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False)
    language = Column(Text, default='ES')
    campaign = relationship("Campaign", back_populates="email_templates")
    # Add version tracking
    version = Column(Integer, default=1)
    parent_version_id = Column(BigInteger, ForeignKey('email_templates.id'), nullable=True)

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    id = Column(BigInteger, primary_key=True)
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
    status = Column(Text)
    started_at = Column(DateTime(timezone=True))
    stopped_at = Column(DateTime(timezone=True))
    paused_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AutomationTask(Base):
    __tablename__ = 'automation_tasks'
    id = Column(BigInteger, primary_key=True)
    task_type = Column(Text)
    status = Column(Text)
    progress = Column(Integer, default=0)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    eta = Column(DateTime(timezone=True))
    logs = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

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
    """Thread-safe database session context manager"""
    if not hasattr(thread_local, "session"):
        thread_local.session = SessionLocal()
    session = thread_local.session
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
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

def create_theme():
    return gr.Theme.from_hub("freddyaboulton/dracula_revamped").set(
        body_background_fill="*neutral-50",
        button_primary_background_fill="*blue-600",
        button_primary_background_fill_hover="*blue-700",
        input_background_fill="white",
        input_border_color="*neutral-200",
        input_shadow="*shadow-sm",
        spacing_sm="2",
        spacing_md="4", 
        radius_sm="0.375rem",
        radius_md="0.5rem",
        radius_lg="0.75rem",
        text_md="0.875rem",
        font=["Inter", "sans-serif"]
    )

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
        self.theme = create_theme()
        self.automation_status = False
        self.automation_logs = []
        self.total_leads_found = 0
        self.total_emails_sent = 0
        
        # Initialize database
        Base.metadata.create_all(bind=engine)

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

    async def perform_search(self, search_terms: str, num_results: int, ignore_fetched: bool, 
                           shuffle_keywords: bool, optimize_english: bool, optimize_spanish: bool, 
                           language: str, enable_email: bool = False, template_id: Optional[int] = None, 
                           email_setting_id: Optional[int] = None, reply_to: Optional[str] = None):
        """Perform search with progress tracking"""
        progress = gr.Progress()
        
        try:
            with get_db() as session:
                terms = [term.strip() for term in search_terms.split('\n') if term.strip()]
                total_terms = len(terms)
                results = {"results": [], "logs": []}
                
                from_email = None
                if enable_email and email_setting_id:
                    email_settings = session.query(EmailSettings).get(email_setting_id)
                    from_email = email_settings.email if email_settings else None
                
                for i, term in enumerate(terms):
                    progress(i/total_terms, desc=f"Searching term {i+1}/{total_terms}: {term}")
                    
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
                        email_template_id=template_id
                    )
                    
                    results["results"].extend(term_results.get("results", []))
                    results["logs"].append(f"Completed search for '{term}': Found {len(term_results.get('results', []))} results")
                
                progress(1.0, desc="Search completed")
                return "Search completed", "\n".join(results["logs"]), pd.DataFrame(results["results"])
                
        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            logging.error(error_msg)
            return f"Error: {str(e)}", error_msg, None

    def create_ui(self):
        with gr.Blocks(theme=self.theme, title="AutoclientAI - Lead Generation Platform") as demo:
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

                with gr.Group(visible=False) as email_group:
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
                search_btn.click(
                    fn=self.perform_search,
                    inputs=[
                        search_terms, num_results, ignore_fetched,
                        shuffle_keywords, optimize_english, optimize_spanish,
                        language, enable_email, template_id,
                        email_setting_id, reply_to
                    ],
                    outputs=[status, logs, results],
                    api_name="search"
                )

                enable_email.change(
                    fn=lambda x: gr.Group(visible=x),
                    inputs=[enable_email],
                    outputs=[email_group]
                )

            # Add other tabs here...

        return demo.queue()

    def create_manual_search_tab(self):
        """Create the Manual Search tab UI"""
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

        with gr.Group(visible=False) as email_group:
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

    def export_results(self, results):
        """Export results to CSV"""
        if not results:
            return None
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False)
        return csv

    def format_log(self, message, level='info'):
        """Format log message with HTML styling"""
        colors = {
            'info': '#2196F3',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'error': '#F44336',
            'email_sent': '#9C27B0'
        }
        return f'<div style="color: {colors.get(level, "#000")}; margin: 5px 0;">{message}</div>'

    def create_email_templates_tab(self):
        def save_template_callback(template_name, subject, body, is_ai_customizable, language):
            with get_db() as session:
                try:
                    template_id = create_or_update_email_template(
                        session, template_name, subject, body,
                        is_ai_customizable=is_ai_customizable,
                        language=language
                    )
                    return gr.update(value=f"Template saved successfully: {template_id}")
                except Exception as e:
                    return gr.update(value=f"Error saving template: {str(e)}")
        
        # Add proper event handlers with session management
        template_name = gr.Textbox(label="Template Name")
        subject = gr.Textbox(label="Subject")
        body = gr.TextArea(label="Body")
        save_btn = gr.Button("Save")
        status = gr.Textbox(label="Status")
        
        save_btn.click(
            fn=save_template_callback,
            inputs=[template_name, subject, body],
            outputs=[status]
        )

    def get_active_project_id(self):
        return st.session_state.get('active_project_id', 1)
    
    def generate_or_adjust_email_template(prompt, use_kb, current_template=None):
        """Generate or adjust email template using AI"""
        with get_db() as session:
            kb_info = get_knowledge_base_info(session, self.get_active_project_id()) if use_kb else None
            result = generate_or_adjust_email_template(prompt, kb_info, current_template)
    
    def generate_or_modify_template(self, prompt, use_kb, current_template=None):
        """Generate or modify template using AI"""
        with get_db() as session:
            kb_info = get_knowledge_base_info(session, self.get_active_project_id()) if use_kb else None
            result = generate_or_adjust_email_template(prompt, kb_info, current_template)
            
            if result:
                subject = result.get('subject', '')
                body = result.get('body', '')
                preview = self.wrap_email_body(body)
                return subject, body, preview
            
            return "", "", ""

    def wrap_email_body(self, body):
        """Wrap email body with styling"""
        return f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
            {body}
        </div>
        """

    def create_bulk_email_tab(self):
        """Create the Bulk Email tab UI"""
        with gr.Row():
            with gr.Column(scale=2):
                template_select = gr.Dropdown(
                    label="Email Template",
                    choices=self.fetch_template_names()
                )
                email_setting = gr.Dropdown(
                    label="From Email",
                    choices=self.fetch_email_settings_names()
                )
                reply_to = gr.Textbox(label="Reply To")
                
                send_option = gr.Radio(
                    label="Send to",
                    choices=[
                        "All Leads",
                        "Specific Email",
                        "Leads from Chosen Search Terms",
                        "Leads from Search Term Groups",
                        "Advanced Targeting"
                    ]
                )
                
                with gr.Group(visible=False) as specific_email_group:
                    specific_email = gr.Textbox(label="Enter email")
                
                with gr.Group(visible=False) as search_terms_group:
                    search_terms = gr.Dropdown(
                        label="Select Search Terms",
                        choices=self.fetch_search_terms(),
                        multiselect=True
                    )
                
                with gr.Group(visible=False) as advanced_group:
                    with gr.Row():
                        min_date = gr.Date(label="Lead Added After")
                        max_date = gr.Date(label="Lead Added Before")
                    with gr.Row():
                        company_filter = gr.Textbox(label="Company Contains")
                        job_title_filter = gr.Textbox(label="Job Title Contains")
                    with gr.Row():
                        domain_filter = gr.Textbox(label="Domain Contains")
                        source_filter = gr.Dropdown(
                            label="Lead Source",
                            choices=self.fetch_search_terms(),
                            multiselect=True
                        )
                
                exclude_contacted = gr.Checkbox(
                    label="Exclude Previously Contacted",
                    value=True
                )
                
                with gr.Row():
                    estimate_btn = gr.Button("Estimate Recipients")
                    send_btn = gr.Button("Send Emails", variant="primary")
            
            with gr.Column(scale=2):
                preview = gr.HTML(label="Email Preview")
                with gr.Row():
                    recipient_count = gr.Number(label="Estimated Recipients", value=0)
                    success_rate = gr.Number(label="Success Rate", value=0)
                
                progress = gr.Progress()
                results = gr.DataFrame(
                    headers=[
                        "Email",
                        "Status",
                        "Sent At",
                        "Error Message"
                    ],
                    label="Sending Results"
                )
                
                with gr.Accordion("Detailed Logs", open=False):
                    logs = gr.HTML()
                
                with gr.Row():
                    export_btn = gr.Button("Export Results")
                    stop_btn = gr.Button("Stop Sending")

        # Event handlers
        send_option.change(
            fn=self.update_send_options,
            inputs=[send_option],
            outputs=[specific_email_group, search_terms_group, advanced_group]
        )
        
        template_select.change(
            fn=self.preview_template,
            inputs=[template_select],
            outputs=[preview]
        )
        
        estimate_btn.click(
            fn=self.estimate_recipients,
            inputs=[
                send_option, specific_email, search_terms,
                min_date, max_date, company_filter,
                job_title_filter, domain_filter, source_filter,
                exclude_contacted
            ],
            outputs=[recipient_count]
        )
        
        send_btn.click(
            fn=self.send_bulk_emails,
            inputs=[
                template_select, email_setting, reply_to,
                send_option, specific_email, search_terms,
                min_date, max_date, company_filter,
                job_title_filter, domain_filter, source_filter,
                exclude_contacted
            ],
            outputs=[results, logs, success_rate]
        )
        
        export_btn.click(
            fn=self.export_email_results,
            inputs=[results],
            outputs=[gr.File(label="Download Results")]
        )
        
        stop_btn.click(
            fn=self.stop_sending,
            inputs=[],
            outputs=[logs]
        )

    def estimate_recipients(self, send_option, specific_email=None, search_terms=None,
                          min_date=None, max_date=None, company=None, job_title=None,
                          domain=None, sources=None, exclude_contacted=True):
        """Estimate number of recipients based on selected criteria"""
        with get_db() as session:
            query = session.query(Lead)
            
            if send_option == "Specific Email":
                if specific_email:
                    query = query.filter(Lead.email == specific_email)
            elif send_option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"]:
                if search_terms:
                    query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(search_terms))
            elif send_option == "Advanced Targeting":
                if min_date:
                    query = query.filter(Lead.created_at >= min_date)
                if max_date:
                    query = query.filter(Lead.created_at <= max_date)
                if company:
                    query = query.filter(Lead.company.ilike(f"%{company}%"))
                if job_title:
                    query = query.filter(Lead.job_title.ilike(f"%{job_title}%"))
                if domain:
                    query = query.filter(Lead.email.ilike(f"%@{domain}%"))
                if sources:
                    query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(sources))
            
            if exclude_contacted:
                subquery = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None))
                query = query.outerjoin(subquery, Lead.id == subquery.c.lead_id).filter(subquery.c.lead_id.is_(None))
            
            return query.count()

    def export_email_results(self, results):
        """Export email sending results to CSV"""
        if not results:
            return None
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False)
        return csv

    def stop_sending(self):
        """Stop the email sending process"""
        self.stop_sending_flag = True
        return self.format_log("Stopping email sending process...", "warning")

    def create_campaign_logs_tab(self):
        """Create the Campaign Logs tab UI"""
        with gr.Row():
            with gr.Column():
                date_range = gr.DateRange(label="Date Range")
                search = gr.Textbox(
                    label="Search",
                    placeholder="Search by email or subject"
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    export_btn = gr.Button("Export Logs")
            
            with gr.Column():
                with gr.Row():
                    total_sent = gr.Number(label="Total Emails Sent", value=0)
                    success_rate = gr.Number(label="Success Rate", value=0)
                    open_rate = gr.Number(label="Open Rate", value=0)
                    click_rate = gr.Number(label="Click Rate", value=0)
                
                metrics_chart = gr.Plot(label="Email Performance Over Time")
        
        logs_table = gr.DataFrame(
            headers=[
                "Date",
                "Email",
                "Template",
                "Subject",
                "Status",
                "Opens",
                "Clicks",
                "Last Open",
                "Last Click"
            ],
            label="Email Logs"
        )
        
        with gr.Row():
            with gr.Column():
                engagement_chart = gr.Plot(label="Engagement Metrics")
            with gr.Column():
                hourly_chart = gr.Plot(label="Sending Distribution")
        
        with gr.Accordion("Email Content Preview", open=False):
            preview = gr.HTML()
        
        # Event handlers
        refresh_btn.click(
            fn=self.fetch_campaign_logs,
            inputs=[date_range, search],
            outputs=[
                logs_table, total_sent, success_rate,
                open_rate, click_rate, metrics_chart,
                engagement_chart, hourly_chart
            ]
        )
        
        export_btn.click(
            fn=self.export_campaign_logs,
            inputs=[date_range, search],
            outputs=[gr.File(label="Download Logs")]
        )
        
        logs_table.select(
            fn=self.preview_email_content,
            inputs=[logs_table],
            outputs=[preview]
        )

    def fetch_campaign_logs(self, date_range, search_term=None):
        """Fetch and process campaign logs"""
        with get_db() as session:
            query = session.query(EmailCampaign).join(Lead).join(EmailTemplate)
            
            if date_range:
                start_date, end_date = date_range
                if start_date:
                    query = query.filter(EmailCampaign.sent_at >= start_date)
                if end_date:
                    query = query.filter(EmailCampaign.sent_at <= end_date)
            
            if search_term:
                query = query.filter(
                    or_(
                        Lead.email.ilike(f"%{search_term}%"),
                        EmailCampaign.subject.ilike(f"%{search_term}%")
                    )
                )
            
            campaigns = query.all()
            
            # Prepare table data
            table_data = []
            for campaign in campaigns:
                table_data.append({
                    "Date": campaign.sent_at,
                    "Email": campaign.lead.email,
                    "Template": campaign.template.template_name,
                    "Subject": campaign.subject,
                    "Status": campaign.status,
                    "Opens": campaign.open_count,
                    "Clicks": campaign.click_count,
                    "Last Open": campaign.opened_at,
                    "Last Click": campaign.clicked_at
                })
            
            # Calculate metrics
            total = len(campaigns)
            successful = sum(1 for c in campaigns if c.status == 'sent')
            opened = sum(1 for c in campaigns if c.open_count > 0)
            clicked = sum(1 for c in campaigns if c.click_count > 0)
            
            success_rate_val = (successful / total) if total > 0 else 0
            open_rate_val = (opened / successful) if successful > 0 else 0
            click_rate_val = (clicked / opened) if opened > 0 else 0
            
            # Create performance chart
            df = pd.DataFrame(table_data)
            df['Date'] = pd.to_datetime(df['Date'])
            daily_metrics = df.groupby(df['Date'].dt.date).agg({
                'Status': lambda x: (x == 'sent').mean(),
                'Opens': 'mean',
                'Clicks': 'mean'
            }).reset_index()
            
            metrics_fig = px.line(
                daily_metrics,
                x='Date',
                y=['Status', 'Opens', 'Clicks'],
                title='Daily Email Performance'
            )
            
            # Create engagement chart
            engagement_data = {
                'Metric': ['Sent', 'Opened', 'Clicked'],
                'Count': [successful, opened, clicked]
            }
            engagement_fig = px.bar(
                engagement_data,
                x='Metric',
                y='Count',
                title='Email Engagement Funnel'
            )
            
            # Create hourly distribution chart
            df['Hour'] = df['Date'].dt.hour
            hourly_counts = df['Hour'].value_counts().sort_index()
            hourly_fig = px.bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title='Sending Time Distribution',
                labels={'x': 'Hour of Day', 'y': 'Number of Emails'}
            )
            
            return (
                table_data,
                total,
                success_rate_val,
                open_rate_val,
                click_rate_val,
                metrics_fig,
                engagement_fig,
                hourly_fig
            )

    def export_campaign_logs(self, date_range, search_term=None):
        """Export campaign logs to CSV"""
        table_data, *_ = self.fetch_campaign_logs(date_range, search_term)
        if not table_data:
            return None
        df = pd.DataFrame(table_data)
        csv = df.to_csv(index=False)
        return csv

    def preview_email_content(self, selected_rows):
        """Preview email content for selected log entry"""
        if not selected_rows or len(selected_rows) == 0:
            return ""
        
        row = selected_rows[0]
        with get_db() as session:
            campaign = session.query(EmailCampaign).join(Lead).filter(
                Lead.email == row['Email'],
                EmailCampaign.sent_at == row['Date']
            ).first()
            
            if campaign:
                return self.wrap_email_body(campaign.email_body)
            return ""

    def create_lead_management_tab(self):
        """Create the Lead Management tab UI"""
        with gr.Row():
            with gr.Column():
                search = gr.Textbox(
                    label="Search Leads",
                    placeholder="Search by email, name, company..."
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    export_btn = gr.Button("Export Leads")
                    quick_scan_btn = gr.Button("Quick Scan")
            
            with gr.Column():
                with gr.Row():
                    total_leads = gr.Number(label="Total Leads", value=0)
                    contacted_leads = gr.Number(label="Contacted Leads", value=0)
                    conversion_rate = gr.Number(label="Conversion Rate", value=0)
                    avg_response_rate = gr.Number(label="Avg Response Rate", value=0)
                
                growth_chart = gr.Plot(label="Lead Growth Over Time")
        
        with gr.Tabs():
            with gr.Tab("All Leads"):
                leads_table = gr.DataFrame(
                    headers=[
                        "Email",
                        "Name",
                        "Company",
                        "Job Title",
                        "Source",
                        "Last Contact",
                        "Status",
                        "Actions"
                    ],
                    label="Leads",
                    interactive=True
                )
                
                with gr.Row():
                    bulk_action = gr.Dropdown(
                        label="Bulk Action",
                        choices=[
                            "Delete Selected",
                            "Mark as Contacted",
                            "Export Selected",
                            "Add to Campaign"
                        ]
                    )
                    apply_btn = gr.Button("Apply")
            
            with gr.Tab("Lead Sources"):
                sources_table = gr.DataFrame(
                    headers=[
                        "Source",
                        "Total Leads",
                        "Valid Leads",
                        "Success Rate",
                        "Last Used"
                    ],
                    label="Lead Sources"
                )
                source_chart = gr.Plot(label="Lead Sources Distribution")
            
            with gr.Tab("Lead Quality"):
                quality_metrics = gr.DataFrame(
                    headers=[
                        "Metric",
                        "Value",
                        "Change"
                    ],
                    label="Quality Metrics"
                )
                quality_chart = gr.Plot(label="Lead Quality Trends")
        
        with gr.Accordion("Lead Details", open=False):
            with gr.Row():
                lead_info = gr.JSON(label="Lead Information")
                lead_history = gr.DataFrame(
                    headers=[
                        "Date",
                        "Action",
                        "Details"
                    ],
                    label="Lead History"
                )
        
        # Event handlers
        refresh_btn.click(
            fn=self.fetch_leads_data,
            inputs=[search],
            outputs=[
                leads_table, sources_table, quality_metrics,
                total_leads, contacted_leads, conversion_rate,
                avg_response_rate, growth_chart, source_chart,
                quality_chart
            ]
        )
        
        export_btn.click(
            fn=self.export_leads,
            inputs=[search],
            outputs=[gr.File(label="Download Leads")]
        )
        
        quick_scan_btn.click(
            fn=self.perform_quick_scan,
            inputs=[],
            outputs=[leads_table, total_leads, growth_chart]
        )
        
        leads_table.select(
            fn=self.fetch_lead_details,
            inputs=[leads_table],
            outputs=[lead_info, lead_history]
        )
        
        apply_btn.click(
            fn=self.apply_bulk_action,
            inputs=[bulk_action, leads_table],
            outputs=[leads_table, total_leads, contacted_leads]
        )

    def fetch_leads_data(self, search_term=None):
        """Fetch and process leads data"""
        with get_db() as session:
            # Base query
            query = session.query(Lead)
            
            if search_term:
                query = query.filter(
                    or_(
                        Lead.email.ilike(f"%{search_term}%"),
                        Lead.first_name.ilike(f"%{search_term}%"),
                        Lead.last_name.ilike(f"%{search_term}%"),
                        Lead.company.ilike(f"%{search_term}%")
                    )
                )
            
            leads = query.all()
            
            # Prepare leads table data
            leads_data = []
            for lead in leads:
                leads_data.append({
                    "Email": lead.email,
                    "Name": f"{lead.first_name or ''} {lead.last_name or ''}".strip() or "Unknown",
                    "Company": lead.company or "Unknown",
                    "Job Title": lead.job_title or "Unknown",
                    "Source": self.get_lead_sources(lead),
                    "Last Contact": self.get_last_contact(lead),
                    "Status": self.get_lead_status(lead),
                    "Actions": "Edit|Delete"
                })
            
            # Prepare sources data
            sources_data = self.analyze_lead_sources(session)
            
            # Prepare quality metrics
            quality_data = self.analyze_lead_quality(session)
            
            # Calculate metrics
            total = len(leads)
            contacted = sum(1 for lead in leads if self.get_last_contact(lead))
            responses = sum(1 for lead in leads if self.has_response(lead))
            
            conversion = (contacted / total) if total > 0 else 0
            response_rate = (responses / contacted) if contacted > 0 else 0
            
            # Create charts
            growth_fig = self.create_growth_chart(leads)
            source_fig = self.create_source_chart(sources_data)
            quality_fig = self.create_quality_chart(quality_data)
            
            return (
                leads_data,
                sources_data,
                quality_data,
                total,
                contacted,
                conversion,
                response_rate,
                growth_fig,
                source_fig,
                quality_fig
            )

    def get_lead_sources(self, lead):
        """Get formatted lead sources"""
        return ", ".join(source.url for source in lead.lead_sources)

    def get_last_contact(self, lead):
        """Get last contact date"""
        if lead.email_campaigns:
            return max(campaign.sent_at for campaign in lead.email_campaigns)
        return None

    def get_lead_status(self, lead):
        """Get lead status"""
        if not lead.email_campaigns:
            return "Not Contacted"
        latest = max(lead.email_campaigns, key=lambda x: x.sent_at)
        return latest.status

    def has_response(self, lead):
        """Check if lead has responded"""
        return any(campaign.open_count > 0 or campaign.click_count > 0 
                  for campaign in lead.email_campaigns)

    def analyze_lead_sources(self, session):
        """Analyze lead sources"""
        sources = session.query(SearchTerm).all()
        return [{
            "Source": source.term,
            "Total Leads": len(source.lead_sources),
            "Valid Leads": sum(1 for ls in source.lead_sources if self.is_valid_lead(ls.lead)),
            "Success Rate": sum(1 for ls in source.lead_sources if self.is_successful_lead(ls.lead)) / len(source.lead_sources) if source.lead_sources else 0,
            "Last Used": max((ls.created_at for ls in source.lead_sources), default=None)
        } for source in sources]

    def analyze_lead_quality(self, session):
        """Analyze lead quality"""
        leads = session.query(Lead).all()
        total = len(leads)
        
        metrics = [
            {
                "Metric": "Email Validity",
                "Value": sum(1 for lead in leads if self.is_valid_email(lead.email)) / total if total > 0 else 0,
                "Change": "+2.5%"  # You would calculate this based on historical data
            },
            {
                "Metric": "Company Info",
                "Value": sum(1 for lead in leads if lead.company) / total if total > 0 else 0,
                "Change": "+1.8%"
            },
            {
                "Metric": "Job Title",
                "Value": sum(1 for lead in leads if lead.job_title) / total if total > 0 else 0,
                "Change": "+3.2%"
            }
        ]
        return metrics

    def create_growth_chart(self, leads):
        """Create lead growth chart"""
        df = pd.DataFrame([(lead.created_at.date(), 1) for lead in leads], columns=['Date', 'Count'])
        df = df.groupby('Date').sum().cumsum().reset_index()
        return px.line(df, x='Date', y='Count', title='Lead Growth Over Time')

    def create_source_chart(self, sources_data):
        """Create lead sources chart"""
        df = pd.DataFrame(sources_data)
        return px.pie(df, values='Total Leads', names='Source', title='Lead Distribution by Source')

    def create_quality_chart(self, quality_data):
        """Create lead quality chart"""
        df = pd.DataFrame(quality_data)
        return px.bar(df, x='Metric', y='Value', title='Lead Quality Metrics')

    def fetch_lead_details(self, selected_rows):
        """Fetch detailed information for selected lead"""
        if not selected_rows or len(selected_rows) == 0:
            return None, None
        
        row = selected_rows[0]
        with get_db() as session:
            lead = session.query(Lead).filter(Lead.email == row['Email']).first()
            
            if not lead:
                return None, None
            
            # Lead information
            info = lead.to_dict()
            info['Sources'] = [source.url for source in lead.lead_sources]
            
            # Lead history
            history = []
            for campaign in lead.email_campaigns:
                history.append({
                    "Date": campaign.sent_at,
                    "Action": "Email Sent",
                    "Details": f"Template: {campaign.template.template_name}, Status: {campaign.status}"
                })
                if campaign.opened_at:
                    history.append({
                        "Date": campaign.opened_at,
                        "Action": "Email Opened",
                        "Details": f"Opens: {campaign.open_count}"
                    })
                if campaign.clicked_at:
                    history.append({
                        "Date": campaign.clicked_at,
                        "Action": "Link Clicked",
                        "Details": f"Clicks: {campaign.click_count}"
                    })
            
            return info, sorted(history, key=lambda x: x['Date'], reverse=True)

    def apply_bulk_action(self, action, selected_rows):
        """Apply bulk action to selected leads"""
        if not selected_rows or len(selected_rows) == 0:
            return None, None, None
        
        with get_db() as session:
            for row in selected_rows:
                lead = session.query(Lead).filter(Lead.email == row['Email']).first()
                if not lead:
                    continue
                
                if action == "Delete Selected":
                    session.delete(lead)
                elif action == "Mark as Contacted":
                    campaign = EmailCampaign(
                        lead_id=lead.id,
                        status="manual_contact",
                        sent_at=datetime.utcnow()
                    )
                    session.add(campaign)
            
            session.commit()
            
            # Refresh leads data
            leads_data, *rest = self.fetch_leads_data()
            return leads_data, rest[3], rest[4]  # leads_table, total_leads, contacted_leads

    def export_leads(self, search_term=None):
        """Export leads to CSV"""
        leads_data, *_ = self.fetch_leads_data(search_term)
        if not leads_data:
            return None
        df = pd.DataFrame(leads_data)
        csv = df.to_csv(index=False)
        return csv

    def create_search_terms_tab(self):
        """Create the Search Terms tab UI"""
        with gr.Tabs():
            # Groups Tab
            with gr.Tab("Groups"):
                with gr.Row():
                    with gr.Column():
                        group_name = gr.Textbox(label="Group Name")
                        create_group_btn = gr.Button("Create Group")
                    
                    with gr.Column():
                        group_select = gr.Dropdown(
                            label="Select Group",
                            choices=self.fetch_search_term_groups()
                        )
                        delete_group_btn = gr.Button("Delete Group")
                
                terms_list = gr.Dropdown(
                    label="Search Terms",
                    choices=self.fetch_search_terms(),
                    multiselect=True
                )
                update_group_btn = gr.Button("Update Group")
                
                with gr.Row():
                    effectiveness_chart = gr.Plot(label="Group Effectiveness")
                    performance_metrics = gr.DataFrame(
                        headers=["Metric", "Value"],
                        label="Group Performance"
                    )
            
            # Terms Tab
            with gr.Tab("Terms"):
                with gr.Row():
                    with gr.Column():
                        new_term = gr.Textbox(label="New Search Term")
                        group_for_term = gr.Dropdown(
                            label="Assign to Group",
                            choices=["None"] + self.fetch_search_term_groups()
                        )
                        with gr.Row():
                            add_term_btn = gr.Button("Add Term")
                            optimize_btn = gr.Button("Optimize Term")
                    
                    with gr.Column():
                        language = gr.Dropdown(
                            label="Language",
                            choices=["ES", "EN"],
                            value="ES"
                        )
                        category = gr.Dropdown(
                            label="Category",
                            choices=["Industry", "Role", "Technology", "Location", "Custom"],
                            value="Custom"
                        )
                
                terms_table = gr.DataFrame(
                    headers=[
                        "Term",
                        "Group",
                        "Lead Count",
                        "Email Count",
                        "Success Rate",
                        "Last Used"
                    ],
                    label="Search Terms"
                )
                
                with gr.Row():
                    term_effectiveness = gr.Plot(label="Term Effectiveness")
                    lead_quality = gr.Plot(label="Lead Quality by Term")
            
            # AI Grouping Tab
            with gr.Tab("AI Grouping"):
                with gr.Row():
                    with gr.Column():
                        ai_prompt = gr.TextArea(
                            label="AI Instructions",
                            placeholder="Enter instructions for AI grouping..."
                        )
                        use_kb = gr.Checkbox(label="Use Knowledge Base")
                        ai_group_btn = gr.Button("Group Terms with AI")
                    
                    with gr.Column():
                        ai_results = gr.DataFrame(
                            headers=["Group", "Terms", "Confidence"],
                            label="AI Grouping Results"
                        )
                        apply_grouping_btn = gr.Button("Apply Grouping")
            
            # Analytics Tab
            with gr.Tab("Analytics"):
                with gr.Row():
                    date_range = gr.DateRange(label="Date Range")
                    refresh_analytics_btn = gr.Button("Refresh")
                
                with gr.Row():
                    total_terms = gr.Number(label="Total Terms", value=0)
                    active_terms = gr.Number(label="Active Terms", value=0)
                    avg_leads = gr.Number(label="Avg Leads/Term", value=0)
                    success_rate = gr.Number(label="Success Rate", value=0)
                
                with gr.Row():
                    term_trends = gr.Plot(label="Term Performance Trends")
                    category_dist = gr.Plot(label="Category Distribution")
                    language_dist = gr.Plot(label="Language Distribution")
        
        # Event handlers
        create_group_btn.click(
            fn=self.create_search_term_group,
            inputs=[group_name],
            outputs=[group_select, terms_list]
        )
        
        delete_group_btn.click(
            fn=self.delete_search_term_group,
            inputs=[group_select],
            outputs=[group_select, terms_list]
        )
        
        update_group_btn.click(
            fn=self.update_search_term_group,
            inputs=[group_select, terms_list],
            outputs=[terms_list, effectiveness_chart, performance_metrics]
        )
        
        add_term_btn.click(
            fn=self.add_search_term,
            inputs=[new_term, group_for_term, language, category],
            outputs=[terms_table, term_effectiveness, lead_quality]
        )
        
        optimize_btn.click(
            fn=self.optimize_search_term,
            inputs=[new_term, language],
            outputs=[new_term]
        )
        
        ai_group_btn.click(
            fn=self.ai_group_search_terms,
            inputs=[ai_prompt, use_kb],
            outputs=[ai_results]
        )
        
        apply_grouping_btn.click(
            fn=self.apply_ai_grouping,
            inputs=[ai_results],
            outputs=[terms_table, group_select, terms_list]
        )
        
        refresh_analytics_btn.click(
            fn=self.refresh_term_analytics,
            inputs=[date_range],
            outputs=[
                total_terms, active_terms, avg_leads, success_rate,
                term_trends, category_dist, language_dist
            ]
        )

    def create_search_term_group(self, group_name):
        """Create a new search term group"""
        with get_db() as session:
            group = SearchTermGroup(
                name=group_name,
                campaign_id=self.get_active_campaign_id()
            )
            session.add(group)
            session.commit()
            
            return self.fetch_search_term_groups(), self.fetch_search_terms()

    def delete_search_term_group(self, group_id):
        """Delete a search term group"""
        if not group_id:
            return None, None
        
        with get_db() as session:
            group = session.query(SearchTermGroup).get(int(group_id.split(':')[0]))
            if group:
                # Update terms to remove group association
                session.query(SearchTerm).filter_by(group_id=group.id).update({SearchTerm.group_id: None})
                session.delete(group)
                session.commit()
            
            return self.fetch_search_term_groups(), self.fetch_search_terms()

    def update_search_term_group(self, group_id, term_ids):
        """Update terms in a group"""
        if not group_id or not term_ids:
            return None, None, None
        
        with get_db() as session:
            group = session.query(SearchTermGroup).get(int(group_id.split(':')[0]))
            if not group:
                return None, None, None
            
            # Update term associations
            term_ids = [int(tid.split(':')[0]) for tid in term_ids]
            session.query(SearchTerm).filter(SearchTerm.id.in_(term_ids)).update(
                {SearchTerm.group_id: group.id},
                synchronize_session=False
            )
            session.commit()
            
            # Calculate effectiveness metrics
            effectiveness = self.calculate_group_effectiveness(session, group)
            performance = self.calculate_group_performance(session, group)
            
            return (
                self.fetch_search_terms(),
                self.create_effectiveness_chart(effectiveness),
                performance
            )

    def calculate_group_effectiveness(self, session, group):
        """Calculate effectiveness metrics for a group"""
        metrics = {
            'total_leads': 0,
            'valid_leads': 0,
            'contacted_leads': 0,
            'responded_leads': 0,
            'daily_leads': []
        }
        
        for term in group.search_terms:
            for source in term.lead_sources:
                metrics['total_leads'] += 1
                if self.is_valid_lead(source.lead):
                    metrics['valid_leads'] += 1
                if self.is_contacted_lead(source.lead):
                    metrics['contacted_leads'] += 1
                if self.has_response(source.lead):
                    metrics['responded_leads'] += 1
                
                metrics['daily_leads'].append({
                    'date': source.created_at.date(),
                    'count': 1
                })
        
        return metrics

    def calculate_group_performance(self, session, group):
        """Calculate performance metrics for a group"""
        metrics = []
        total_leads = sum(len(term.lead_sources) for term in group.search_terms)
        
        if total_leads > 0:
            metrics.extend([
                {
                    "Metric": "Total Leads",
                    "Value": total_leads
                },
                {
                    "Metric": "Valid Lead Rate",
                    "Value": f"{sum(1 for term in group.search_terms for source in term.lead_sources if self.is_valid_lead(source.lead)) / total_leads:.1%}"
                },
                {
                    "Metric": "Response Rate",
                    "Value": f"{sum(1 for term in group.search_terms for source in term.lead_sources if self.has_response(source.lead)) / total_leads:.1%}"
                },
                {
                    "Metric": "Average Leads/Term",
                    "Value": f"{total_leads / len(group.search_terms):.1f}"
                }
            ])
        
        return metrics

    def create_effectiveness_chart(self, metrics):
        """Create effectiveness chart from metrics"""
        df = pd.DataFrame(metrics['daily_leads'])
        df = df.groupby('date')['count'].sum().reset_index()
        return px.line(df, x='date', y='count', title='Daily Leads Generated')

    def refresh_term_analytics(self, date_range):
        """Refresh search term analytics"""
        with get_db() as session:
            query = session.query(SearchTerm)
            
            if date_range:
                start_date, end_date = date_range
                if start_date:
                    query = query.filter(SearchTerm.created_at >= start_date)
                if end_date:
                    query = query.filter(SearchTerm.created_at <= end_date)
            
            terms = query.all()
            
            # Calculate metrics
            total = len(terms)
            active = sum(1 for term in terms if term.lead_sources)
            total_leads = sum(len(term.lead_sources) for term in terms)
            avg_leads = total_leads / total if total > 0 else 0
            success_rate = sum(1 for term in terms if self.is_successful_term(term)) / total if total > 0 else 0
            
            # Create trend chart
            trend_data = []
            for term in terms:
                for source in term.lead_sources:
                    trend_data.append({
                        'date': source.created_at.date(),
                        'term': term.term,
                        'count': 1
                    })
            
            trend_df = pd.DataFrame(trend_data)
            trend_fig = px.line(
                trend_df.groupby(['date', 'term'])['count'].sum().reset_index(),
                x='date',
                y='count',
                color='term',
                title='Term Performance Over Time'
            )
            
            # Create category distribution
            category_data = pd.DataFrame([{
                'category': term.category, 'count': len(term.lead_sources)}
                for term in terms if term.category
            ])
            category_fig = px.pie(
                category_data,
                values='count',
                names='category',
                title='Leads by Term Category'
            )
            
            # Create language distribution
            language_data = pd.DataFrame([{
                'language': term.language, 'count': len(term.lead_sources)}
                for term in terms if term.language
            ])
            language_fig = px.pie(
                language_data,
                values='count',
                names='language',
                title='Leads by Term Language'
            )
            
            return (
                total,
                active,
                avg_leads,
                success_rate,
                trend_fig,
                category_fig,
                language_fig
            )

    def is_successful_term(self, term):
        """Check if a search term is successful"""
        if not term.lead_sources:
            return False
        
        valid_leads = sum(1 for source in term.lead_sources if self.is_valid_lead(source.lead))
        return valid_leads / len(term.lead_sources) >= 0.5  # 50% success rate threshold

    def create_automation_control_tab(self):
        """Create the Automation Control tab UI"""
        with gr.Tabs():
            # Status Tab
            with gr.Tab("Status"):
                with gr.Row():
                    with gr.Column():
                        status_indicator = gr.Label(
                            label="Automation Status",
                            value="Stopped"
                        )
                        with gr.Row():
                            start_btn = gr.Button("Start Automation")
                            stop_btn = gr.Button("Stop Automation")
                            pause_btn = gr.Button("Pause Automation")
                    
                    with gr.Column():
                        active_tasks = gr.DataFrame(
                            headers=[
                                "Task",
                                "Status",
                                "Progress",
                                "Started At",
                                "ETA"
                            ],
                            label="Active Tasks"
                        )
                        refresh_tasks_btn = gr.Button("Refresh Tasks")
                
                with gr.Row():
                    with gr.Column():
                        daily_stats = gr.DataFrame(
                            headers=[
                                "Metric",
                                "Today",
                                "This Week",
                                "This Month"
                            ],
                            label="Automation Stats"
                        )
                    
                    with gr.Column():
                        error_log = gr.DataFrame(
                            headers=[
                                "Time",
                                "Task",
                                "Error",
                                "Details"
                            ],
                            label="Error Log"
                        )
            
            # Schedule Tab
            with gr.Tab("Schedule"):
                with gr.Row():
                    with gr.Column():
                        schedule_name = gr.Textbox(label="Schedule Name")
                        task_type = gr.Dropdown(
                            label="Task Type",
                            choices=[
                                "Email Campaign",
                                "Lead Search",
                                "Data Update",
                                "Report Generation"
                            ]
                        )
                        frequency = gr.Radio(
                            label="Frequency",
                            choices=[
                                "Once",
                                "Daily",
                                "Weekly",
                                "Monthly"
                            ],
                            value="Once"
                        )
                    
                    with gr.Column():
                        start_date = gr.Datetime(label="Start Date")
                        end_date = gr.Datetime(label="End Date")
                        time_of_day = gr.Textbox(
                            label="Time of Day",
                            placeholder="HH:MM"
                        )
                
                with gr.Row():
                    save_schedule_btn = gr.Button("Save Schedule")
                    delete_schedule_btn = gr.Button("Delete Schedule")
                
                schedules_list = gr.DataFrame(
                    headers=[
                        "Name",
                        "Task",
                        "Frequency",
                        "Next Run",
                        "Status"
                    ],
                    label="Scheduled Tasks"
                )
            
            # Rules Tab
            with gr.Tab("Rules"):
                with gr.Row():
                    with gr.Column():
                        rule_name = gr.Textbox(label="Rule Name")
                        rule_type = gr.Dropdown(
                            label="Rule Type",
                            choices=[
                                "Email Limit",
                                "Lead Quality",
                                "Response Rate",
                                "Error Rate"
                            ]
                        )
                        condition = gr.Dropdown(
                            label="Condition",
                            choices=[
                                "Greater Than",
                                "Less Than",
                                "Equals",
                                "Not Equals"
                            ]
                        )
                        threshold = gr.Number(label="Threshold")
                    
                    with gr.Column():
                        action = gr.Dropdown(
                            label="Action",
                            choices=[
                                "Stop Automation",
                                "Pause Automation",
                                "Send Alert",
                                "Adjust Settings"
                            ]
                        )
                        notification_email = gr.Textbox(label="Notification Email")
                        is_active = gr.Checkbox(
                            label="Rule Active",
                            value=True
                        )
                
                with gr.Row():
                    save_rule_btn = gr.Button("Save Rule")
                    delete_rule_btn = gr.Button("Delete Rule")
                
                rules_list = gr.DataFrame(
                    headers=[
                        "Name",
                        "Type",
                        "Condition",
                        "Action",
                        "Status"
                    ],
                    label="Automation Rules"
                )
            
            # Monitoring Tab
            with gr.Tab("Monitoring"):
                with gr.Row():
                    with gr.Column():
                        monitor_date = gr.DateRange(label="Date Range")
                        refresh_monitor_btn = gr.Button("Refresh")
                    
                    with gr.Column():
                        monitor_metrics = gr.DataFrame(
                            headers=[
                                "Metric",
                                "Value",
                                "Status",
                                "Trend"
                            ],
                            label="Performance Metrics"
                        )
                
                with gr.Row():
                    task_distribution = gr.Plot(label="Task Distribution")
                    performance_trends = gr.Plot(label="Performance Trends")
        
        # Event handlers
        start_btn.click(
            fn=self.start_automation,
            inputs=[],
            outputs=[status_indicator, active_tasks]
        )
        
        stop_btn.click(
            fn=self.stop_automation,
            inputs=[],
            outputs=[status_indicator, active_tasks]
        )
        
        pause_btn.click(
            fn=self.pause_automation,
            inputs=[],
            outputs=[status_indicator, active_tasks]
        )
        
        refresh_tasks_btn.click(
            fn=self.refresh_tasks,
            inputs=[],
            outputs=[active_tasks, daily_stats, error_log]
        )
        
        save_schedule_btn.click(
            fn=self.save_schedule,
            inputs=[
                schedule_name, task_type, frequency,
                start_date, end_date, time_of_day
            ],
            outputs=[schedules_list]
        )
        
        delete_schedule_btn.click(
            fn=self.delete_schedule,
            inputs=[schedules_list],
            outputs=[schedules_list]
        )
        
        save_rule_btn.click(
            fn=self.save_rule,
            inputs=[
                rule_name, rule_type, condition,
                threshold, action, notification_email,
                is_active
            ],
            outputs=[rules_list]
        )
        
        delete_rule_btn.click(
            fn=self.delete_rule,
            inputs=[rules_list],
            outputs=[rules_list]
        )
        
        refresh_monitor_btn.click(
            fn=self.refresh_monitoring,
            inputs=[monitor_date],
            outputs=[
                monitor_metrics,
                task_distribution,
                performance_trends
            ]
        )

    def start_automation(self):
        """Start automation process"""
        with get_db() as session:
            automation = session.query(AutomationStatus).first()
            if not automation:
                automation = AutomationStatus(status="running")
                session.add(automation)
            else:
                automation.status = "running"
                automation.started_at = datetime.now()
            
            session.commit()
            
            return "Running", self.get_active_tasks()

    def stop_automation(self):
        """Stop automation process"""
        with get_db() as session:
            automation = session.query(AutomationStatus).first()
            if automation:
                automation.status = "stopped"
                automation.stopped_at = datetime.now()
                session.commit()
            
            return "Stopped", []

    def pause_automation(self):
        """Pause automation process"""
        with get_db() as session:
            automation = session.query(AutomationStatus).first()
            if automation:
                automation.status = "paused"
                automation.paused_at = datetime.now()
                session.commit()
            
            return "Paused", self.get_active_tasks()

    def refresh_tasks(self):
        """Refresh automation tasks and stats"""
        return (
            self.get_active_tasks(),
            self.get_daily_stats(),
            self.get_error_log()
        )

    def save_schedule(self, name, task_type, frequency, start_date, end_date, time_of_day):
        """Save automation schedule"""
        with get_db() as session:
            schedule = AutomationSchedule(
                name=name,
                task_type=task_type,
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
                time_of_day=time_of_day
            )
            session.add(schedule)
            session.commit()
            
            return self.get_schedules()

    def delete_schedule(self, selected_rows):
        """Delete automation schedule"""
        if not selected_rows:
            return self.get_schedules()
        
        with get_db() as session:
            for row in selected_rows:
                schedule = session.query(AutomationSchedule).filter_by(name=row['Name']).first()
                if schedule:
                    session.delete(schedule)
            session.commit()
            
            return self.get_schedules()

    def save_rule(self, name, rule_type, condition, threshold, action, notification_email, is_active):
        """Save automation rule"""
        with get_db() as session:
            rule = AutomationRule(
                name=name,
                rule_type=rule_type,
                condition=condition,
                threshold=threshold,
                action=action,
                notification_email=notification_email,
                is_active=is_active
            )
            session.add(rule)
            session.commit()
            
            return self.get_rules()

    def delete_rule(self, selected_rows):
        """Delete automation rule"""
        if not selected_rows:
            return self.get_rules()
        
        with get_db() as session:
            for row in selected_rows:
                rule = session.query(AutomationRule).filter_by(name=row['Name']).first()
                if rule:
                    session.delete(rule)
            session.commit()
            
            return self.get_rules()

    def refresh_monitoring(self, date_range):
        """Refresh automation monitoring"""
        with get_db() as session:
            # Query automation tasks
            query = session.query(AutomationTask)
            
            if date_range:
                start_date, end_date = date_range
                if start_date:
                    query = query.filter(AutomationTask.started_at >= start_date)
                if end_date:
                    query = query.filter(AutomationTask.started_at <= end_date)
            
            tasks = query.all()
            
            # Calculate metrics
            total_tasks = len(tasks)
            if total_tasks == 0:
                return [], None, None
            
            success_rate = sum(1 for t in tasks if t.status == 'completed') / total_tasks
            error_rate = sum(1 for t in tasks if t.status == 'error') / total_tasks
            avg_duration = sum((t.completed_at - t.started_at).total_seconds() for t in tasks if t.completed_at) / total_tasks
            
            metrics = [
                {
                    "Metric": "Success Rate",
                    "Value": f"{success_rate*100:.1f}%",
                    "Status": "Good" if success_rate >= 0.9 else "Warning",
                    "Trend": "↑"
                },
                {
                    "Metric": "Error Rate",
                    "Value": f"{error_rate*100:.1f}%",
                    "Status": "Good" if error_rate <= 0.1 else "Warning",
                    "Trend": "↓"
                },
                {
                    "Metric": "Avg Duration",
                    "Value": f"{avg_duration:.1f}s",
                    "Status": "Good" if avg_duration <= 60 else "Warning",
                    "Trend": "→"
                }
            ]
            
            # Create distribution chart
            dist_data = pd.DataFrame([{
                'Task Type': t.task_type,
                'Count': 1,
                'Status': t.status
            } for t in tasks])
            
            dist_fig = px.bar(
                dist_data.groupby(['Task Type', 'Status'])['Count'].sum().reset_index(),
                x='Task Type',
                y='Count',
                color='Status',
                title='Task Distribution by Type and Status'
            )
            
            # Create trends chart
            trends_data = pd.DataFrame([{
                'Date': t.started_at.date(),
                'Success Rate': 1 if t.status == 'completed' else 0,
                'Error Rate': 1 if t.status == 'error' else 0
            } for t in tasks])
            
            trends_fig = px.line(
                trends_data.groupby('Date').mean().reset_index(),
                x='Date',
                y=['Success Rate', 'Error Rate'],
                title='Performance Trends Over Time'
            )
            
            return metrics, dist_fig, trends_fig

    def get_active_tasks(self):
        """Get list of active automation tasks"""
        with get_db() as session:
            tasks = session.query(AutomationTask)\
                .filter(AutomationTask.status.in_(['running', 'pending']))\
                .all()
            
            return [{
                "Task": task.task_type,
                "Status": task.status,
                "Progress": f"{task.progress}%",
                "Started At": task.started_at,
                "ETA": task.eta
            } for task in tasks]

    def get_daily_stats(self):
        """Get daily automation statistics"""
        with get_db() as session:
            today = datetime.now().date()
            week_ago = today - timedelta(days=7)
            month_ago = today - timedelta(days=30)
            
            def get_stats(start_date):
                tasks = session.query(AutomationTask)\
                    .filter(AutomationTask.started_at >= start_date)\
                    .all()
                
                total = len(tasks)
                if total == 0:
                    return 0, 0, 0
                
                success = sum(1 for t in tasks if t.status == 'completed')
                errors = sum(1 for t in tasks if t.status == 'error')
                
                return total, success, errors
            
            today_stats = get_stats(today)
            week_stats = get_stats(week_ago)
            month_stats = get_stats(month_ago)
            
            return [{
                "Metric": "Total Tasks",
                "Today": today_stats[0],
                "This Week": week_stats[0],
                "This Month": month_stats[0]
            }]  # Close both the dictionary and list

    def get_error_log(self):
        """Get automation error log"""
        with get_db() as session:
            errors = session.query(AutomationError)\
                .order_by(AutomationError.created_at.desc())\
                .limit(100)\
                .all()
            
            return [{
                "Time": error.created_at,
                "Task": error.task_type,
                "Error": error.error_type,
                "Details": error.error_message
            } for error in errors]

    def get_schedules(self):
        """Get list of automation schedules"""
        with get_db() as session:
            schedules = session.query(AutomationSchedule).all()
            
            return [{
                "Name": schedule.name,
                "Task": schedule.task_type,
                "Frequency": schedule.frequency,
                "Next Run": self.calculate_next_run(schedule),
                "Status": "Active" if self.is_schedule_active(schedule) else "Inactive"
            } for schedule in schedules]

    def get_rules(self):
        """Get list of automation rules"""
        with get_db() as session:
            rules = session.query(AutomationRule).all()
            
            return [{
                "Name": rule.name,
                "Type": rule.rule_type,
                "Condition": f"{rule.condition} {rule.threshold}",
                "Action": rule.action,
                "Status": "Active" if rule.is_active else "Inactive"
            } for rule in rules]

    def calculate_next_run(self, schedule):
        """Calculate next run time for a schedule"""
        now = datetime.now()
        
        if schedule.end_date and now > schedule.end_date:
            return None
        
        if now < schedule.start_date:
            return schedule.start_date
        
        time_parts = schedule.time_of_day.split(':')
        run_time = time(int(time_parts[0]), int(time_parts[1]))
        
        if schedule.frequency == 'Once':
            return schedule.start_date
        
        next_run = datetime.combine(now.date(), run_time)
        
        if schedule.frequency == 'Daily':
            if now > next_run:
                next_run += timedelta(days=1)
        elif schedule.frequency == 'Weekly':
            while now > next_run:
                next_run += timedelta(days=7)
        elif schedule.frequency == 'Monthly':
            while now > next_run:
                if next_run.month == 12:
                    next_run = next_run.replace(year=next_run.year + 1, month=1)
                else:
                    next_run = next_run.replace(month=next_run.month + 1)
        
        return next_run

    def is_schedule_active(self, schedule):
        """Check if a schedule is active"""
        now = datetime.now()
        
        if schedule.end_date and now > schedule.end_date:
            return False
        
        if now < schedule.start_date:
            return False
        
        return True

    def run(self):
        demo = self.create_ui()
        demo.launch(share=True)

    def create_projects_campaigns_tab(self):
        """Create the Projects & Campaigns tab UI"""
        with gr.Tabs():
            # Projects Tab
            with gr.Tab("Projects"):
                with gr.Row():
                    with gr.Column():
                        project_name = gr.Textbox(label="Project Name")
                        project_description = gr.TextArea(
                            label="Description",
                            lines=3
                        )
                        project_type = gr.Dropdown(
                            label="Project Type",
                            choices=[
                                "Lead Generation",
                                "Email Marketing",
                                "Market Research",
                                "Custom"
                            ]
                        )
                    
                    with gr.Column():
                        project_status = gr.Radio(
                            label="Status",
                            choices=[
                                "Active",
                                "Paused",
                                "Completed",
                                "Archived"
                            ],
                            value="Active"
                        )
                        project_priority = gr.Slider(
                            label="Priority",
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1
                        )
                
                with gr.Row():
                    save_project_btn = gr.Button("Save Project")
                    delete_project_btn = gr.Button("Delete Project")
                
                projects_list = gr.DataFrame(
                    headers=[
                        "Name",
                        "Type",
                        "Status",
                        "Campaigns",
                        "Last Activity"
                    ],
                    label="Projects"
                )
            
            # Campaigns Tab
            with gr.Tab("Campaigns"):
                with gr.Row():
                    with gr.Column():
                        campaign_name = gr.Textbox(label="Campaign Name")
                        campaign_project = gr.Dropdown(
                            label="Project",
                            choices=self.fetch_projects()
                        )
                        campaign_type = gr.Dropdown(
                            label="Campaign Type",
                            choices=[
                                "Email Outreach",
                                "Lead Search",
                                "Follow-up",
                                "Custom"
                            ]
                        )
                    
                    with gr.Column():
                        campaign_status = gr.Radio(
                            label="Status",
                            choices=[
                                "Draft",
                                "Active",
                                "Paused",
                                "Completed"
                            ],
                            value="Draft"
                        )
                        campaign_priority = gr.Slider(
                            label="Priority",
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1
                        )
                
                with gr.Row():
                    save_campaign_btn = gr.Button("Save Campaign")
                    delete_campaign_btn = gr.Button("Delete Campaign")
                    duplicate_campaign_btn = gr.Button("Duplicate Campaign")
                
                campaigns_list = gr.DataFrame(
                    headers=[
                        "Name",
                        "Project",
                        "Type",
                        "Status",
                        "Progress"
                    ],
                    label="Campaigns"
                )
            
            # Analytics Tab
            with gr.Tab("Analytics"):
                with gr.Row():
                    with gr.Column():
                        analytics_date = gr.DateRange(label="Date Range")
                        analytics_project = gr.Dropdown(
                            label="Project",
                            choices=["All"] + self.fetch_projects()
                        )
                        refresh_analytics_btn = gr.Button("Refresh")
                    
                    with gr.Column():
                        analytics_metrics = gr.DataFrame(
                            headers=[
                                "Metric",
                                "Value",
                                "Change"
                            ],
                            label="Project Metrics"
                        )
                
                with gr.Row():
                    project_progress = gr.Plot(label="Project Progress")
                    campaign_distribution = gr.Plot(label="Campaign Distribution")
        
        # Event handlers
        save_project_btn.click(
            fn=self.save_project,
            inputs=[
                project_name, project_description,
                project_type, project_status,
                project_priority
            ],
            outputs=[projects_list]
        )
        
        delete_project_btn.click(
            fn=self.delete_project,
            inputs=[projects_list],
            outputs=[projects_list]
        )
        
        save_campaign_btn.click(
            fn=self.save_campaign,
            inputs=[
                campaign_name, campaign_project,
                campaign_type, campaign_status,
                campaign_priority
            ],
            outputs=[campaigns_list]
        )
        
        delete_campaign_btn.click(
            fn=self.delete_campaign,
            inputs=[campaigns_list],
            outputs=[campaigns_list]
        )
        
        duplicate_campaign_btn.click(
            fn=self.duplicate_campaign,
            inputs=[campaigns_list],
            outputs=[campaigns_list]
        )
        
        refresh_analytics_btn.click(
            fn=self.refresh_project_analytics,
            inputs=[analytics_date, analytics_project],
            outputs=[
                analytics_metrics,
                project_progress,
                campaign_distribution
            ]
        )

    def save_project(self, name, description, project_type, status, priority):
        """Save project information"""
        with get_db() as session:
            project = session.query(Project).filter_by(name=name).first()
            if not project:
                project = Project(name=name)
                session.add(project)
            
            project.description = description
            project.project_type = project_type
            project.status = status
            project.priority = priority
            
            session.commit()
            
            return self.get_projects_list()

    def delete_project(self, selected_rows):
        """Delete project"""
        if not selected_rows:
            return self.get_projects_list()
        
        with get_db() as session:
            for row in selected_rows:
                project = session.query(Project).filter_by(name=row['Name']).first()
                if project:
                    session.delete(project)
            session.commit()
            
            return self.get_projects_list()

    def save_campaign(self, name, project_id, campaign_type, status, priority):
        """Save campaign information"""
        with get_db() as session:
            campaign = session.query(Campaign).filter_by(name=name).first()
            if not campaign:
                campaign = Campaign(name=name)
                session.add(campaign)
            
            campaign.project_id = int(project_id.split(':')[0])
            campaign.campaign_type = campaign_type
            campaign.status = status
            campaign.priority = priority
            
            session.commit()
            
            return self.get_campaigns_list()

    def delete_campaign(self, selected_rows):
        """Delete campaign"""
        if not selected_rows:
            return self.get_campaigns_list()
        
        with get_db() as session:
            for row in selected_rows:
                campaign = session.query(Campaign).filter_by(name=row['Name']).first()
                if campaign:
                    session.delete(campaign)
            session.commit()
            
            return self.get_campaigns_list()

    def duplicate_campaign(self, selected_rows):
        """Duplicate campaign"""
        if not selected_rows:
            return self.get_campaigns_list()
        
        with get_db() as session:
            for row in selected_rows:
                original = session.query(Campaign).filter_by(name=row['Name']).first()
                if original:
                    duplicate = Campaign(
                        name=f"{original.name} (Copy)",
                        project_id=original.project_id,
                        campaign_type=original.campaign_type,
                        status="Draft",
                        priority=original.priority
                    )
                    session.add(duplicate)
            session.commit()
            
            return self.get_campaigns_list()

    def refresh_project_analytics(self, date_range, project_id):
        """Refresh project analytics"""
        with get_db() as session:
            # Query campaigns
            query = session.query(Campaign)
            
            if project_id != "All":
                query = query.filter_by(project_id=int(project_id.split(':')[0]))
            
            if date_range:
                start_date, end_date = date_range
                if start_date:
                    query = query.filter(Campaign.created_at >= start_date)
                if end_date:
                    query = query.filter(Campaign.created_at <= end_date)
            
            campaigns = query.all()
            
            # Calculate metrics
            total_campaigns = len(campaigns)
            if total_campaigns == 0:
                return [], None, None
            
            active = sum(1 for c in campaigns if c.status == 'Active')
            completed = sum(1 for c in campaigns if c.status == 'Completed')
            
            metrics = [
                {
                    "Metric": "Total Campaigns",
                    "Value": total_campaigns,
                    "Change": "+0%"
                },
                {
                    "Metric": "Active Campaigns",
                    "Value": active,
                    "Change": "+2.5%"
                },
                {
                    "Metric": "Completion Rate",
                    "Value": f"{(completed/total_campaigns)*100:.1f}%",
                    "Change": "+1.8%"
                }
            ]
            
            # Create progress chart
            progress_data = pd.DataFrame([{
                'Campaign': c.name,
                'Progress': self.calculate_campaign_progress(c)
            } for c in campaigns])
            
            progress_fig = px.bar(
                progress_data,
                x='Campaign',
                y='Progress',
                title='Campaign Progress'
            )
            
            # Create distribution chart
            dist_data = pd.DataFrame([{
                'Type': c.campaign_type,
                'Status': c.status,
                'Count': 1
            } for c in campaigns])
            
            dist_fig = px.bar(
                dist_data.groupby(['Type', 'Status'])['Count'].sum().reset_index(),
                x='Type',
                y='Count',
                color='Status',
                title='Campaign Distribution by Type and Status'
            )
            
            return metrics, progress_fig, dist_fig

    def get_projects_list(self):
        """Get list of projects"""
        with get_db() as session:
            projects = session.query(Project).all()
            
            return [{
                "Name": project.name,
                "Type": project.project_type,
                "Status": project.status,
                "Campaigns": len(project.campaigns),
                "Last Activity": self.get_project_last_activity(project)
            } for project in projects]

    def get_campaigns_list(self):
        """Get list of campaigns"""
        with get_db() as session:
            campaigns = session.query(Campaign).all()
            
            return [{
                "Name": campaign.name,
                "Project": campaign.project.name,
                "Type": campaign.campaign_type,
                "Status": campaign.status,
                "Progress": f"{self.calculate_campaign_progress(campaign)}%"
            } for campaign in campaigns]

    def get_project_last_activity(self, project):
        """Get project's last activity date"""
        activities = [
            campaign.updated_at for campaign in project.campaigns
            if campaign.updated_at
        ]
        return max(activities) if activities else None

    def calculate_campaign_progress(self, campaign):
        """Calculate campaign progress percentage"""
        if campaign.status == 'Completed':
            return 100
        elif campaign.status == 'Draft':
            return 0
        
        # Calculate based on completed tasks
        total_tasks = len(campaign.tasks)
        if total_tasks == 0:
            return 0
        
        completed_tasks = sum(1 for task in campaign.tasks if task.status == 'completed')
        return int((completed_tasks / total_tasks) * 100)

    def create_sent_campaigns_tab(self):
        """Create the Sent Campaigns tab UI"""
        with gr.Tabs():
            # Overview Tab
            with gr.Tab("Overview"):
                with gr.Row():
                    with gr.Column():
                        date_range = gr.DateRange(label="Date Range")
                        campaign_type = gr.Dropdown(
                            label="Campaign Type",
                            choices=[
                                "All",
                                "Email Outreach",
                                "Lead Search",
                                "Follow-up"
                            ],
                            value="All"
                        )
                        refresh_btn = gr.Button("Refresh")
                    
                    with gr.Column():
                        campaign_metrics = gr.DataFrame(
                            headers=[
                                "Metric",
                                "Value",
                                "Change"
                            ],
                            label="Campaign Metrics"
                        )
                
                with gr.Row():
                    sent_trends = gr.Plot(label="Sending Trends")
                    response_rates = gr.Plot(label="Response Rates")
            
            # Campaign List Tab
            with gr.Tab("Campaign List"):
                with gr.Row():
                    with gr.Column():
                        search_term = gr.Textbox(label="Search Campaigns")
                        status_filter = gr.Dropdown(
                            label="Status",
                            choices=[
                                "All",
                                "Completed",
                                "Failed",
                                "Partial"
                            ],
                            value="All"
                        )
                    
                    with gr.Column():
                        sort_by = gr.Dropdown(
                            label="Sort By",
                            choices=[
                                "Date",
                                "Name",
                                "Success Rate",
                                "Response Rate"
                            ],
                            value="Date"
                        )
                        sort_order = gr.Radio(
                            label="Order",
                            choices=["Ascending", "Descending"],
                            value="Descending"
                        )
                
                campaigns_table = gr.DataFrame(
                    headers=[
                        "Campaign",
                        "Type",
                        "Sent At",
                        "Total Sent",
                        "Success Rate",
                        "Response Rate",
                        "Status"
                    ],
                    label="Sent Campaigns"
                )
                
                with gr.Row():
                    export_btn = gr.Button("Export Data")
                    archive_btn = gr.Button("Archive Selected")
            
            # Campaign Details Tab
            with gr.Tab("Campaign Details"):
                with gr.Row():
                    with gr.Column():
                        campaign_select = gr.Dropdown(
                            label="Select Campaign",
                            choices=self.fetch_sent_campaigns()
                        )
                        refresh_details_btn = gr.Button("Refresh Details")
                    
                    with gr.Column():
                        campaign_info = gr.DataFrame(
                            headers=[
                                "Field",
                                "Value"
                            ],
                            label="Campaign Information"
                        )
                
                with gr.Row():
                    with gr.Column():
                        email_stats = gr.DataFrame(
                            headers=[
                                "Email",
                                "Status",
                                "Sent At",
                                "Opened",
                                "Clicked",
                                "Responded"
                            ],
                            label="Email Statistics"
                        )
                    
                    with gr.Column():
                        response_timeline = gr.Plot(label="Response Timeline")
            
            # Analytics Tab
            with gr.Tab("Analytics"):
                with gr.Row():
                    with gr.Column():
                        analytics_date = gr.DateRange(label="Date Range")
                        analytics_type = gr.Dropdown(
                            label="Analysis Type",
                            choices=[
                                "Performance",
                                "Response",
                                "Content",
                                "Timing"
                            ],
                            value="Performance"
                        )
                        refresh_analytics_btn = gr.Button("Refresh Analytics")
                    
                    with gr.Column():
                        analytics_metrics = gr.DataFrame(
                            headers=[
                                "Metric",
                                "Value",
                                "Benchmark",
                                "Status"
                            ],
                            label="Analytics Metrics"
                        )
                
                with gr.Row():
                    performance_chart = gr.Plot(label="Performance Analysis")
                    insights_list = gr.DataFrame(
                        headers=[
                            "Category",
                            "Insight",
                            "Impact",
                            "Recommendation"
                        ],
                        label="Campaign Insights"
                    )
        
        # Event handlers
        refresh_btn.click(
            fn=self.refresh_campaign_overview,
            inputs=[date_range, campaign_type],
            outputs=[
                campaign_metrics,
                sent_trends,
                response_rates
            ]
        )
        
        search_term.change(
            fn=self.filter_campaigns,
            inputs=[
                search_term,
                status_filter,
                sort_by,
                sort_order
            ],
            outputs=[campaigns_table]
        )
        
        status_filter.change(
            fn=self.filter_campaigns,
            inputs=[
                search_term,
                status_filter,
                sort_by,
                sort_order
            ],
            outputs=[campaigns_table]
        )
        
        sort_by.change(
            fn=self.filter_campaigns,
            inputs=[
                search_term,
                status_filter,
                sort_by,
                sort_order
            ],
            outputs=[campaigns_table]
        )
        
        sort_order.change(
            fn=self.filter_campaigns,
            inputs=[
                search_term,
                status_filter,
                sort_by,
                sort_order
            ],
            outputs=[campaigns_table]
        )
        
        export_btn.click(
            fn=self.export_campaign_data,
            inputs=[campaigns_table],
            outputs=[gr.File(label="Download Data")]
        )
        
        archive_btn.click(
            fn=self.archive_campaigns,
            inputs=[campaigns_table],
            outputs=[campaigns_table]
        )
        
        campaign_select.change(
            fn=self.load_campaign_details,
            inputs=[campaign_select],
            outputs=[
                campaign_info,
                email_stats,
                response_timeline
            ]
        )
        
        refresh_details_btn.click(
            fn=self.load_campaign_details,
            inputs=[campaign_select],
            outputs=[
                campaign_info,
                email_stats,
                response_timeline
            ]
        )
        
        refresh_analytics_btn.click(
            fn=self.refresh_campaign_analytics,
            inputs=[
                analytics_date,
                analytics_type
            ],
            outputs=[
                analytics_metrics,
                performance_chart,
                insights_list
            ]
        )

    def refresh_campaign_overview(self, date_range, campaign_type):
        """Refresh campaign overview"""
        with get_db() as session:
            # Query sent campaigns
            query = session.query(EmailCampaign)
            
            if date_range:
                start_date, end_date = date_range
                if start_date:
                    query = query.filter(EmailCampaign.sent_at >= start_date)
                if end_date:
                    query = query.filter(EmailCampaign.sent_at <= end_date)
            
            if campaign_type != "All":
                query = query.filter_by(campaign_type=campaign_type)
            
            campaigns = query.all()
            
            # Calculate metrics
            total = len(campaigns)
            if total == 0:
                return [], None, None
            
            success = sum(1 for c in campaigns if c.status == 'completed')
            responses = sum(1 for c in campaigns if c.response_count > 0)
            
            metrics = [
                {
                    "Metric": "Total Campaigns",
                    "Value": total,
                    "Change": "+0%"
                },
                {
                    "Metric": "Success Rate",
                    "Value": f"{(success/total)*100:.1f}%",
                    "Change": "+2.5%"
                },
                {
                    "Metric": "Response Rate",
                    "Value": f"{(responses/total)*100:.1f}%",
                    "Change": "+1.8%"
                }
            ]
            
            # Create trends chart
            trends_data = pd.DataFrame([{
                'Date': c.sent_at.date(),
                'Count': 1,
                'Type': c.campaign_type
            } for c in campaigns])
            
            trends_fig = px.line(
                trends_data.groupby(['Date', 'Type'])['Count'].sum().reset_index(),
                x='Date',
                y='Count',
                color='Type',
                title='Campaign Sending Trends'
            )
            
            # Create response rates chart
            rates_data = pd.DataFrame([{
                'Campaign': c.name,
                'Response Rate': (c.response_count / c.sent_count * 100) if c.sent_count > 0 else 0
            } for c in campaigns])
            
            rates_fig = px.bar(
                rates_data.sort_values('Response Rate', ascending=False).head(10),
                x='Campaign',
                y='Response Rate',
                title='Top 10 Campaigns by Response Rate'
            )
            
            return metrics, trends_fig, rates_fig

    def filter_campaigns(self, search, status, sort_by, sort_order):
        """Filter and sort campaigns list"""
        with get_db() as session:
            query = session.query(EmailCampaign)
            
            if search:
                query = query.filter(EmailCampaign.name.ilike(f"%{search}%"))
            
            if status != "All":
                query = query.filter_by(status=status.lower())
            
            # Apply sorting
            if sort_by == "Date":
                order_by = EmailCampaign.sent_at
            elif sort_by == "Name":
                order_by = EmailCampaign.name
            elif sort_by == "Success Rate":
                order_by = (EmailCampaign.success_count / EmailCampaign.sent_count)
            else:  # Response Rate
                order_by = (EmailCampaign.response_count / EmailCampaign.sent_count)
            
            if sort_order == "Descending":
                order_by = order_by.desc()
            
            campaigns = query.order_by(order_by).all()
            
            return [{
                "Campaign": c.name,
                "Type": c.campaign_type,
                "Sent At": c.sent_at,
                "Total Sent": c.sent_count,
                "Success Rate": f"{(c.success_count/c.sent_count*100):.1f}%" if c.sent_count > 0 else "0%",
                "Response Rate": f"{(c.response_count/c.sent_count*100):.1f}%" if c.sent_count > 0 else "0%",
                "Status": c.status.title()
            } for c in campaigns]

    def export_campaign_data(self, selected_rows):
        """Export campaign data to CSV"""
        if not selected_rows:
            return None
        
        df = pd.DataFrame(selected_rows)
        
        # Save to temporary file
        temp_file = "campaign_export.csv"
        df.to_csv(temp_file, index=False)
        
        # Read file content
        with open(temp_file, 'rb') as f:
            data = f.read()
        
        # Clean up
        os.remove(temp_file)
        
        return data

    def archive_campaigns(self, selected_rows):
        """Archive selected campaigns"""
        if not selected_rows:
            return self.filter_campaigns("", "All", "Date", "Descending")
        
        with get_db() as session:
            for row in selected_rows:
                campaign = session.query(EmailCampaign).filter_by(name=row['Campaign']).first()
                if campaign:
                    campaign.status = 'archived'
            session.commit()
            
            return self.filter_campaigns("", "All", "Date", "Descending")

    def load_campaign_details(self, campaign_id):
        """Load detailed campaign information"""
        with get_db() as session:
            campaign = session.query(EmailCampaign).get(int(campaign_id.split(':')[0]))
            if not campaign:
                return [], [], None
            
            # Campaign info
            info = [
                {"Field": "Name", "Value": campaign.name},
                {"Field": "Type", "Value": campaign.campaign_type},
                {"Field": "Status", "Value": campaign.status.title()},
                {"Field": "Sent At", "Value": campaign.sent_at},
                {"Field": "Total Sent", "Value": campaign.sent_count},
                {"Field": "Success Rate", "Value": f"{(campaign.success_count/campaign.sent_count*100):.1f}%" if campaign.sent_count > 0 else "0%"},
                {"Field": "Response Rate", "Value": f"{(campaign.response_count/campaign.sent_count*100):.1f}%" if campaign.sent_count > 0 else "0%"}
            ]
            
            # Email stats
            stats = [{
                "Email": email.recipient,
                "Status": email.status.title(),
                "Sent At": email.sent_at,
                "Opened": "Yes" if email.opened_at else "No",
                "Clicked": "Yes" if email.clicked_at else "No",
                "Responded": "Yes" if email.response_received_at else "No"
            } for email in campaign.emails]
            
            # Response timeline
            timeline_data = pd.DataFrame([{
                'Date': email.response_received_at.date(),
                'Count': 1
            } for email in campaign.emails if email.response_received_at])
            
            timeline_fig = px.line(
                timeline_data.groupby('Date')['Count'].sum().reset_index(),
                x='Date',
                y='Count',
                title='Response Timeline'
            ) if not timeline_data.empty else None
            
            return info, stats, timeline_fig

    def refresh_campaign_analytics(self, date_range, analysis_type):
        """Refresh campaign analytics"""
        with get_db() as session:
            # Query campaigns
            query = session.query(EmailCampaign)
            
            if date_range:
                start_date, end_date = date_range
                if start_date:
                    query = query.filter(EmailCampaign.sent_at >= start_date)
                if end_date:
                    query = query.filter(EmailCampaign.sent_at <= end_date)
            
            campaigns = query.all()
            
            if not campaigns:
                return [], None, []
            
            # Calculate metrics based on analysis type
            if analysis_type == "Performance":
                metrics = self.calculate_performance_metrics(campaigns)
                chart = self.create_performance_chart(campaigns)
                insights = self.generate_performance_insights(campaigns)
            elif analysis_type == "Response":
                metrics = self.calculate_response_metrics(campaigns)
                chart = self.create_response_chart(campaigns)
                insights = self.generate_response_insights(campaigns)
            elif analysis_type == "Content":
                metrics = self.calculate_content_metrics(campaigns)
                chart = self.create_content_chart(campaigns)
                insights = self.generate_content_insights(campaigns)
            else:  # Timing
                metrics = self.calculate_timing_metrics(campaigns)
                chart = self.create_timing_chart(campaigns)
                insights = self.generate_timing_insights(campaigns)
            
            return metrics, chart, insights

    def calculate_performance_metrics(self, campaigns):
        """Calculate performance metrics"""
        total = len(campaigns)
        if total == 0:
            return []
        
        total_sent = sum(c.sent_count for c in campaigns)
        total_success = sum(c.success_count for c in campaigns)
        total_responses = sum(c.response_count for c in campaigns)
        
        return [
            {
                "Metric": "Success Rate",
                "Value": f"{(total_success/total_sent*100):.1f}%",
                "Benchmark": "95%",
                "Status": "Good" if total_success/total_sent >= 0.95 else "Warning"
            },
            {
                "Metric": "Response Rate",
                "Value": f"{(total_responses/total_sent*100):.1f}%",
                "Benchmark": "10%",
                "Status": "Good" if total_responses/total_sent >= 0.1 else "Warning"
            },
            {
                "Metric": "Average Sent",
                "Value": f"{total_sent/total:.1f}",
                "Benchmark": "100",
                "Status": "Good" if total_sent/total >= 100 else "Warning"
            }
        ]

    def create_performance_chart(self, campaigns):
        """Create performance analysis chart"""
        data = pd.DataFrame([{
            'Campaign': c.name,
            'Success Rate': (c.success_count/c.sent_count*100) if c.sent_count > 0 else 0,
            'Response Rate': (c.response_count/c.sent_count*100) if c.sent_count > 0 else 0
        } for c in campaigns])
        
        return px.scatter(
            data,
            x='Success Rate',
            y='Response Rate',
            hover_data=['Campaign'],
            title='Campaign Performance Analysis'
        )

    def generate_performance_insights(self, campaigns):
        """Generate performance insights"""
        insights = []
        
        # Success rate analysis
        success_rates = [(c, c.success_count/c.sent_count if c.sent_count > 0 else 0) for c in campaigns]
        top_success = sorted(success_rates, key=lambda x: x[1], reverse=True)[:3]
        low_success = sorted(success_rates, key=lambda x: x[1])[:3]
        
        insights.extend([{
            "Category": "Success Rate",
            "Insight": f"Top performer: {c.name} ({rate*100:.1f}%)",
            "Impact": "High",
            "Recommendation": "Analyze and replicate success factors"
        } for c, rate in top_success])
        
        insights.extend([{
            "Category": "Success Rate",
            "Insight": f"Low performer: {c.name} ({rate*100:.1f}%)",
            "Impact": "High",
            "Recommendation": "Review and optimize campaign settings"
        } for c, rate in low_success])
        
        # Response rate analysis
        response_rates = [(c, c.response_count/c.sent_count if c.sent_count > 0 else 0) for c in campaigns]
        top_response = sorted(response_rates, key=lambda x: x[1], reverse=True)[:3]
        
        insights.extend([{
            "Category": "Response Rate",
            "Insight": f"Best response: {c.name} ({rate*100:.1f}%)",
            "Impact": "High",
            "Recommendation": "Analyze content and timing factors"
        } for c, rate in top_response])
        
        return insights

    def get_setting(self, session: Session, name: str, default: Any = None) -> Any:
        """Get a setting value from the database"""
        setting = session.query(Settings).filter_by(name=name).first()
        if setting:
            return setting.value
        return default

    def set_setting(self, session: Session, name: str, value: Any, setting_type: str = 'general') -> None:
        """Save or update a setting in the database"""
        setting = session.query(Settings).filter_by(name=name).first()
        if setting:
            setting.value = value
            setting.updated_at = datetime.utcnow()
        else:
            setting = Settings(name=name, value=value, setting_type=setting_type)
            session.add(setting)
        session.commit()

def create_or_update_email_template(session, template_name, subject, body_content, template_id=None, is_ai_customizable=False, created_at=None, language='ES'):
    # Check for duplicate template name
    existing = session.query(EmailTemplate).filter_by(template_name=template_name).first()
    if existing and (not template_id or existing.id != template_id):
        raise ValueError("Template name already exists")
        
    # Validate campaign_id
    campaign_id = get_active_campaign_id()
    if not session.query(Campaign).filter_by(id=campaign_id).first():
        raise ValueError("Invalid campaign ID")
        
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(
        template_name=template_name,
        subject=subject,
        body_content=body_content,
        is_ai_customizable=is_ai_customizable,
        campaign_id=campaign_id,
        created_at=created_at or datetime.utcnow(),
        language=language
    )
    
    if template_id:
        template.template_name = template_name
        template.subject = subject
        template.body_content = body_content
        template.is_ai_customizable = is_ai_customizable
        template.language = language
        
    try:
        session.add(template)
        session.commit()
        return template.id
    except SQLAlchemyError as e:
        session.rollback()
        raise ValueError(f"Database error: {str(e)}")

def extract_info_from_page(soup):
    """Extract name, company and job title from page content"""
    try:
        # Initialize variables
        name = company = job_title = None
        
        # Look for common name patterns
        name_patterns = [
            'h1.author', 'span.author', 'div.author',
            'h1.name', 'span.name', 'div.name',
            'h1.profile-name', 'div.profile-name'
        ]
        
        # Look for common company patterns
        company_patterns = [
            'div.company', 'span.company', 
            'div.organization', 'span.organization',
            'div.employer', 'span.employer'
        ]
        
        # Look for common job title patterns
        title_patterns = [
            'div.job-title', 'span.job-title',
            'div.position', 'span.position',
            'div.role', 'span.role'
        ]
        
        # Try to find name
        for pattern in name_patterns:
            element = soup.select_one(pattern)
            if element:
                name = element.get_text().strip()
                break
        
        # Try to find company
        for pattern in company_patterns:
            element = soup.select_one(pattern)
            if element:
                company = element.get_text().strip()
                break
        
        # Try to find job title
        for pattern in title_patterns:
            element = soup.select_one(pattern)
            if element:
                job_title = element.get_text().strip()
                break
        
        # If structured data not found, try meta tags
        if not all([name, company, job_title]):
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                content = tag.get('content', '')
                if tag.get('name') == 'author' and not name:
                    name = content
                elif tag.get('property') == 'og:site_name' and not company:
                    company = content
        
        # Clean up extracted data
        if name:
            name = re.sub(r'\s+', ' ', name).strip()
        if company:
            company = re.sub(r'\s+', ' ', company).strip()
        if job_title:
            job_title = re.sub(r'\s+', ' ', job_title).strip()
        
        return name, company, job_title
    
    except Exception as e:
        logging.error(f"Error extracting info from page: {str(e)}")
        return None, None, None

# Add these utility functions at the top of the file, after the imports

def add_or_get_search_term(session: Session, term: str, campaign_id: int) -> int:
    """Add a new search term or get existing one"""
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

def shuffle_keywords(term: str) -> str:
    """Shuffle keywords in search term"""
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

def get_domain_from_url(url: str) -> str:
    """Extract domain from URL"""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.netloc

def extract_emails_from_html(html_content: str) -> List[str]:
    """Extract email addresses from HTML content"""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return list(set(re.findall(email_pattern, html_content)))

def get_page_title(html_content: str) -> str:
    """Extract page title from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.find('title')
    return title.get_text() if title else "No title"

def save_lead(session, email, **kwargs):
    # Add deduplication check
    existing_lead = session.query(Lead).filter_by(email=email).first()
    if existing_lead:
        logging.info(f"Lead already exists: {email}")
        return existing_lead
    # Rest of the function...

def is_valid_lead(lead: Lead) -> bool:
    """Check if lead is valid"""
    return bool(lead and lead.email and is_valid_email(lead.email))

def is_contacted_lead(lead: Lead) -> bool:
    """Check if lead has been contacted"""
    return bool(lead.email_campaigns)

def is_successful_lead(lead: Lead) -> bool:
    """Check if lead was successfully processed"""
    return bool(lead and lead.email and is_valid_email(lead.email))

def generate_or_adjust_email_template(prompt: str, kb_info: Optional[Dict[str, Any]] = None, current_template: Optional[str] = None) -> Dict[str, str]:
    """Generate or adjust email template using AI"""
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in creating and refining email templates for marketing campaigns."},
        {"role": "user", "content": f"""{'Adjust the following email template based on the given instructions:' if current_template else 'Create an email template based on the following prompt:'} {prompt}

        {'Current Template:' if current_template else 'Guidelines:'}
        {current_template if current_template else '1. Focus on benefits to the reader\n2. Address potential customer doubts\n3. Include clear CTAs\n4. Use a natural tone\n5. Be concise'}

        Knowledge Base Context:
        {json.dumps(kb_info) if kb_info else 'No knowledge base information provided'}
        """}
    ]

    try:
        response = openai_chat_completion(messages, function_name="generate_or_adjust_email_template")
        
        if isinstance(response, dict):
            return response
        elif isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "subject": "AI Generated Subject",
                    "body": f"<p>{response}</p>"
                }
        else:
            return {
                "subject": "",
                "body": "<p>Failed to generate email content.</p>"
            }
    except Exception as e:
        logging.error(f"Error generating email template: {str(e)}")
        return {
            "subject": "",
            "body": f"<p>Error generating template: {str(e)}</p>"
        }

class RateLimiter:
    def __init__(self, max_requests=100, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = threading.Lock()

    def is_allowed(self):
        now = time.time()
        with self._lock:
            self.requests = [req for req in self.requests if now - req < self.time_window]
            if len(self.requests) >= self.max_requests:
                return False
            self.requests.append(now)
            return True

rate_limiter = RateLimiter()

class EmailQueue:
    def __init__(self):
        self.queue = Queue()
        self._lock = threading.Lock()
        
    def add_email(self, email_data):
        with self._lock:
            self.queue.put(email_data)
            
    def get_email(self):
        with self._lock:
            return self.queue.get() if not self.queue.empty() else None

email_queue = EmailQueue()

def calculate_campaign_metrics(session, campaign_id):
    """Calculate comprehensive campaign metrics"""
    metrics = {
        'total_leads': 0,
        'valid_leads': 0,
        'emails_sent': 0,
        'responses': 0,
        'success_rate': 0
    }
    # Add calculation logic...
    return metrics

class BackgroundTaskManager:
    def __init__(self):
        self.tasks = {}
        self._lock = threading.Lock()

    def add_task(self, task_id, task):
        with self._lock:
            self.tasks[task_id] = task

    def get_task(self, task_id):
        with self._lock:
            return self.tasks.get(task_id)

task_manager = BackgroundTaskManager()

def recover_failed_tasks(session):
    """Recover failed tasks and campaigns"""
    failed_tasks = session.query(AutomationTask).filter_by(status='failed').all()
    for task in failed_tasks:
        try:
            # Recovery logic
            task.status = 'pending'
            task.retry_count = (task.retry_count or 0) + 1
            session.commit()
        except Exception as e:
            logging.error(f"Failed to recover task {task.id}: {str(e)}")

class CacheManager:
    def __init__(self):
        self.cache = {}
        self._lock = threading.Lock()
        self.max_size = 1000

    def get(self, key):
        with self._lock:
            return self.cache.get(key)

    def set(self, key, value):
        with self._lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest = min(self.cache.keys())
                del self.cache[oldest]
            self.cache[key] = value

cache_manager = CacheManager()

if __name__ == "__main__":
    app = GradioAutoclientApp()
    demo = app.create_ui()
    demo.launch(share=True)