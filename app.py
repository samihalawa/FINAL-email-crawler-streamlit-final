import os
import json
import re
import logging
import random
import time
import traceback
import uuid
import requests
import asyncio
import urllib3
import smtplib
import sys
from googlesearch import search
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import List, Optional, Dict
from urllib.parse import urlparse

# FastAPI / Jinja2
from fastapi import (
    FastAPI, Request, Form, Depends, HTTPException, BackgroundTasks, Response, status
)
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# SQLAlchemy
from sqlalchemy import (
    create_engine,
    Column,
    BigInteger,
    Text,
    DateTime,
    ForeignKey,
    Boolean,
    JSON,
    func,
    distinct,
    text,
    event
)
from sqlalchemy.orm import (
    sessionmaker,
    declarative_base,
    relationship,
    joinedload,
    Session,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

# Additional libs
import boto3
from botocore.exceptions import ClientError
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email_validator import validate_email, EmailNotValidError
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from pydantic import BaseModel

"""
FASTAPI SINGLE-FILE APPLICATION
Replicates all pages, routes, and logic from a Streamlit code,
but using FastAPI + Jinja2 for the user interface, and a single file approach.
To run locally:
   python main.py
Then visit http://localhost:8000 in your browser.
"""

# ------------- ENV / DATABASE SETUP -------------
load_dotenv()

DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure SQLAlchemy engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    echo=False  # Set to True for debugging SQL queries
)

@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    connection_record.info['pid'] = os.getpid()

@event.listens_for(engine, "checkout")
def checkout(dbapi_connection, connection_record, connection_proxy):
    pid = os.getpid()
    if connection_record.info['pid'] != pid:
        connection_record.connection = None
        raise SQLAlchemyError(
            f"Connection record belongs to pid {connection_record.info['pid']}, "
            f"attempting to check out in pid {pid}"
        )

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ------------- MODELS (replicated from the original code) -------------
class Project(Base):
    __tablename__ = "projects"
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text, default="Default Project")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaigns = relationship("Campaign", back_populates="project")
    knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False)


class Campaign(Base):
    __tablename__ = "campaigns"
    id = Column(BigInteger, primary_key=True)
    campaign_name = Column(Text, default="Default Campaign")
    campaign_type = Column(Text, default="Email")
    project_id = Column(BigInteger, ForeignKey("projects.id"), default=1)
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
    __tablename__ = "campaign_leads"
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey("campaigns.id"))
    lead_id = Column(BigInteger, ForeignKey("leads.id"))
    status = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="campaign_leads")
    campaign = relationship("Campaign", back_populates="campaign_leads")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    id = Column(BigInteger, primary_key=True)
    project_id = Column(BigInteger, ForeignKey("projects.id"), nullable=False)
    kb_name = Column(Text)
    kb_bio = Column(Text)
    kb_values = Column(JSON)
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
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), index=True)
    project = relationship("Project", back_populates="knowledge_base")

    def to_dict(self):
        fields = [
            "kb_name",
            "kb_bio",
            "kb_values",
            "contact_name",
            "contact_role",
            "contact_email",
            "company_description",
            "company_mission",
            "company_target_market",
            "company_other",
            "product_name",
            "product_description",
            "product_target_customer",
            "product_other",
            "other_context",
            "example_email",
        ]
        return {attr: getattr(self, attr) for attr in fields}


class Lead(Base):
    __tablename__ = "leads"
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True, index=True)
    phone = Column(Text, index=True)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text, index=True)
    job_title = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead")
    email_campaigns = relationship("EmailCampaign", back_populates="lead")


class EmailTemplate(Base):
    __tablename__ = "email_templates"
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey("campaigns.id"))
    template_name = Column(Text)
    subject = Column(Text)
    body_content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=True)
    language = Column(Text, default="ES")
    campaign = relationship("Campaign")
    email_campaigns = relationship("EmailCampaign", back_populates="template")


class EmailCampaign(Base):
    __tablename__ = "email_campaigns"
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey("campaigns.id"), index=True)
    lead_id = Column(BigInteger, ForeignKey("leads.id"), index=True)
    template_id = Column(BigInteger, ForeignKey("email_templates.id"), index=True)
    customized_subject = Column(Text)
    customized_content = Column(Text)
    original_subject = Column(Text)
    original_content = Column(Text)
    status = Column(Text, index=True)
    engagement_data = Column(JSON)
    message_id = Column(Text, unique=True, index=True)
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
    __tablename__ = "optimized_search_terms"
    id = Column(BigInteger, primary_key=True)
    original_term_id = Column(BigInteger, ForeignKey("search_terms.id"))
    term = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    original_term = relationship("SearchTerm", back_populates="optimized_terms")


class SearchTermEffectiveness(Base):
    __tablename__ = "search_term_effectiveness"
    id = Column(BigInteger, primary_key=True)
    search_term_id = Column(BigInteger, ForeignKey("search_terms.id"))
    total_results = Column(BigInteger)
    valid_leads = Column(BigInteger)
    irrelevant_leads = Column(BigInteger)
    blogs_found = Column(BigInteger)
    directories_found = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_term = relationship("SearchTerm", back_populates="effectiveness")


class SearchTermGroup(Base):
    __tablename__ = "search_term_groups"
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    email_template = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_terms = relationship("SearchTerm", back_populates="group")


class SearchTerm(Base):
    __tablename__ = "search_terms"
    id = Column(BigInteger, primary_key=True)
    group_id = Column(BigInteger, ForeignKey("search_term_groups.id"))
    campaign_id = Column(BigInteger, ForeignKey("campaigns.id"))
    term = Column(Text)
    category = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    language = Column(Text, default="ES")
    group = relationship("SearchTermGroup", back_populates="search_terms")
    campaign = relationship("Campaign", back_populates="search_terms")
    optimized_terms = relationship("OptimizedSearchTerm", back_populates="original_term")
    lead_sources = relationship("LeadSource", back_populates="search_term")
    effectiveness = relationship("SearchTermEffectiveness", back_populates="search_term", uselist=False)


class LeadSource(Base):
    __tablename__ = "lead_sources"
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey("leads.id"))
    search_term_id = Column(BigInteger, ForeignKey("search_terms.id"))
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
    __tablename__ = "ai_request_logs"
    id = Column(BigInteger, primary_key=True)
    function_name = Column(Text)
    prompt = Column(Text)
    response = Column(Text)
    model_used = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead_id = Column(BigInteger, ForeignKey("leads.id"))
    email_campaign_id = Column(BigInteger, ForeignKey("email_campaigns.id"))
    lead = relationship("Lead")
    email_campaign = relationship("EmailCampaign")


class AutomationLog(Base):
    __tablename__ = "automation_logs"
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey("campaigns.id"))
    search_term_id = Column(BigInteger, ForeignKey("search_terms.id"))
    leads_gathered = Column(BigInteger)
    emails_sent = Column(BigInteger)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    status = Column(Text)
    logs = Column(JSON)
    campaign = relationship("Campaign")
    search_term = relationship("SearchTerm")


class Settings(Base):
    __tablename__ = "settings"
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    setting_type = Column(Text, nullable=False)  # 'general', 'email', etc.
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class EmailSettings(Base):
    __tablename__ = "email_settings"
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

# ------------- DATABASE SESSION HELPER -------------
@contextmanager
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Replace with proper FastAPI dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------- UTILITY / LOGIC FUNCTIONS -------------
def get_active_project_id() -> int:
    """In the original app, we used st.session_state. Here, default to 1 or expand with a real approach."""
    return 1

def get_active_campaign_id() -> int:
    return 1

def set_active_project_id(project_id: int):
    pass  # In a real scenario, store in a user session/cookie

def set_active_campaign_id(campaign_id: int):
    pass  # In a real scenario, store in a user session/cookie

def check_required_settings(session: Session):
    try:
        project_id = get_active_project_id()
        campaign_id = get_active_campaign_id()
        if not project_id or not campaign_id:
            return False, "No active project or campaign selected"

        email_settings = session.query(EmailSettings).first()
        if not email_settings:
            return False, "Email settings not configured"

        templates = session.query(EmailTemplate).filter_by(campaign_id=campaign_id).first()
        if not templates:
            return False, "No email templates found"

        return True, None
    except Exception as e:
        return False, str(e)

def is_valid_email(email: str):
    if not email:
        return False
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59",
    ]
    return random.choice(user_agents)

def should_skip_domain(domain: str):
    skip_domains = {
        "www.airbnb.es",
        "www.airbnb.com",
        "www.linkedin.com",
        "es.linkedin.com",
        "www.idealista.com",
        "www.facebook.com",
        "www.instagram.com",
        "www.youtube.com",
        "youtu.be",
    }
    return domain in skip_domains

def get_domain_from_url(url: str):
    return urlparse(url).netloc

def extract_emails_from_html(html_content: str):
    """Extract email addresses from HTML content using multiple patterns."""
    emails = set()
    
    # Basic email pattern
    basic_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    emails.update(re.findall(basic_pattern, html_content))
    
    # Look for obfuscated emails
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Check text content
    text_content = soup.get_text()
    emails.update(re.findall(basic_pattern, text_content))
    
    # Check mailto links
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('mailto:'):
            email = href.replace('mailto:', '').split('?')[0]
            if re.match(basic_pattern, email):
                emails.add(email)
    
    # Check for common email patterns in text
    contact_texts = soup.find_all(text=re.compile(r'contacto|contact|email|correo|e-mail'))
    for text in contact_texts:
        parent = text.parent
        if parent:
            emails.update(re.findall(basic_pattern, str(parent)))
    
    return list(emails)

def is_valid_contact_email(email: str):
    email_low = email.lower()
    invalid_patterns = [
        "sentry", "noreply", "no-reply", "donotreply", "do-not-reply", "automated",
        "notification", "alert", "system", "admin@", "postmaster", "mailer-daemon",
        "webmaster", "hostmaster", "support@", "error@", "report@", "test@",
        "office@", "mail@", "email@"
    ]
    if any(p in email_low for p in invalid_patterns):
        return False
    invalid_domains = ["example.com", "test.com", "sample.com", "mail.com", "website.com"]
    domain = email_low.split("@")[-1]
    if domain in invalid_domains:
        return False
    return True

def extract_info_from_page(soup: BeautifulSoup):
    name_tag = soup.find("meta", {"name": "author"})
    if name_tag and name_tag.has_attr("content"):
        name = name_tag["content"]
    else:
        name = ""

    og_site = soup.find("meta", {"property": "og:site_name"})
    if og_site and og_site.has_attr("content"):
        company = og_site["content"]
    else:
        company = ""

    job_title_tag = soup.find("meta", {"name": "job_title"})
    if job_title_tag and job_title_tag.has_attr("content"):
        job_title = job_title_tag["content"]
    else:
        job_title = ""
    return name, company, job_title

def extract_company_name(soup: BeautifulSoup, url: str) -> str:
    og_site = soup.find("meta", {"property": "og:site_name"})
    if og_site and og_site.has_attr("content"):
        return og_site["content"]
    domain = get_domain_from_url(url)
    if domain.startswith("www."):
        domain = domain.replace("www.", "")
    return domain.split(".")[0].title()

def save_lead_source(
    session: Session,
    lead_id: int,
    search_term_id,
    url,
    http_status,
    scrape_duration,
    page_title=None,
    meta_description=None,
    content=None,
    tags=None,
    phone_numbers=None,
):
    lead_source = LeadSource(
        lead_id=lead_id,
        search_term_id=search_term_id,
        url=url,
        http_status=http_status,
        scrape_duration=scrape_duration,
        page_title=page_title,
        meta_description=meta_description,
        content=content,
        tags=tags,
        phone_numbers=phone_numbers,
    )
    session.add(lead_source)
    session.commit()

def save_lead(
    session: Session,
    email: str,
    first_name=None,
    last_name=None,
    company=None,
    job_title=None,
    phone=None,
    url=None,
    search_term_id=None,
    created_at=None,
):
    """Upsert lead with enhanced logic. If lead is new, add it. Then add LeadSource, CampaignLead."""
    if not email:
        return None
    try:
        lead = session.query(Lead).filter_by(email=email).first()
        if not lead:
            lead = Lead(
                email=email,
                first_name=first_name,
                last_name=last_name,
                company=company,
                job_title=job_title,
                phone=phone,
                created_at=created_at or datetime.utcnow(),
            )
            session.add(lead)
            session.flush()

        lead_source = LeadSource(
            lead_id=lead.id, url=url, search_term_id=search_term_id, domain=get_domain_from_url(url), http_status=200
        )
        session.add(lead_source)

        campaign_id = get_active_campaign_id()
        c_lead = session.query(CampaignLead).filter_by(campaign_id=campaign_id, lead_id=lead.id).first()
        if not c_lead:
            c_lead = CampaignLead(
                campaign_id=campaign_id,
                lead_id=lead.id,
                status="Not Contacted",
                created_at=datetime.utcnow(),
            )
            session.add(c_lead)

        session.commit()
        return lead
    except Exception as e:
        logging.error(f"Error saving lead: {e}")
        session.rollback()
        return None

def wrap_email_body(body_content: str) -> str:
    """Wrap email content in a minimal HTML template for sending."""
    try:
        soup = BeautifulSoup(body_content, "html.parser")
        for tag in soup.find_all(True):
            if tag.name in ["script", "iframe", "object", "embed"]:
                tag.decompose()
        sanitized = str(soup)
        template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8" />
            <title>Email Body</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            {sanitized}
        </body>
        </html>
        """
        return template
    except Exception as e:
        logging.error(f"Error wrapping body: {e}")
        return body_content

def send_email_ses(
    session: Session,
    from_email: str,
    to_email: str,
    subject: str,
    body: str,
    reply_to: Optional[str] = None,
):
    if not all([from_email, to_email, subject, body]):
        logging.error("Missing required email fields")
        return None, None
        
    try:
        s = session.query(EmailSettings).filter_by(provider="AWS SES").first()
        if not s:
            logging.error("No AWS SES settings found")
            return None, None
            
        if not all([s.aws_access_key_id, s.aws_secret_access_key]):
            logging.error("Missing AWS credentials")
            return None, None
            
        ses_client = boto3.client(
            "ses",
            aws_access_key_id=s.aws_access_key_id,
            aws_secret_access_key=s.aws_secret_access_key,
            region_name=s.aws_region or "us-east-1",
        )
        
        email_data = {
            "Source": from_email,
            "Destination": {"ToAddresses": [to_email]},
            "Message": {
                "Subject": {"Data": subject},
                "Body": {"Html": {"Data": body}},
            },
        }
        
        if reply_to:
            email_data["ReplyToAddresses"] = [reply_to]
            
        resp = ses_client.send_email(**email_data)
        return resp, resp.get("MessageId")
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        logging.error(f"AWS SES error ({error_code}): {e}")
        return None, None
    except Exception as ex:
        logging.error(f"Unexpected error sending email: {ex}")
        return None, None

def save_email_campaign(
    session: Session,
    lead_email: str,
    template_id: int,
    status: str,
    sent_at: datetime,
    subject: str,
    message_id: Optional[str],
    email_body: str,
):
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

        ec = EmailCampaign(
            campaign_id=campaign.id,
            lead_id=lead.id,
            template_id=template_id,
            status=status,
            sent_at=sent_at,
            original_subject=subject,
            original_content=email_body,
            message_id=message_id,
            tracking_id=str(uuid.uuid4()),
        )
        session.add(ec)
        return ec
    except Exception as e:
        logging.error(f"Error saving email campaign: {e}")
        return None

def manual_search(
    session: Session,
    terms: List[str],
    num_results: int,
    language: str = "ES",
    ignore_previously_fetched: bool = True,
    optimize_english: bool = False,
    optimize_spanish: bool = False,
    shuffle_keywords_option: bool = False,
    enable_email_sending: bool = False,
    from_email: Optional[str] = None,
    reply_to: Optional[str] = None,
    email_template: Optional[str] = None,
):
    """Performs manual search for leads using provided search terms"""
    from googlesearch import search
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse
    import time
    
    results = {'total_leads': 0, 'results': [], 'email_logs': [], 'search_logs': []}
    processed_domains = set()
    processed_urls = set()
    
    for term in terms:
        try:
            results['search_logs'].append(f'Searching for term: {term}')
            
            # Perform Google search
            print(f"Searching for: {term}")
            search_results = search(term, num_results=num_results, lang=language.lower())
            print(f"Got search results")
            urls = list(search_results)
            print(f"Converted to list: {urls}")
            results['search_logs'].append(f'Found {len(urls)} URLs for term: {term}')
            
            if not urls:
                results['search_logs'].append(f'No results found for term: {term}')
                continue
            
            for url in urls:
                try:
                    # Skip if URL already processed
                    if url in processed_urls:
                        continue
                        
                    # Extract domain and skip if already processed
                    domain = get_domain_from_url(url)
                    if domain in processed_domains:
                        continue
                        
                    # Skip if domain should be skipped
                    if should_skip_domain(domain):
                        continue
                        
                    # Skip if previously fetched
                    if ignore_previously_fetched and session.query(LeadSource).filter_by(url=url).first():
                        continue
                    
                    # Fetch and parse webpage
                    headers = {'User-Agent': get_random_user_agent()}
                    response = requests.get(url, headers=headers, timeout=10, verify=False)
                    if response.status_code != 200:
                        continue
                        
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract emails
                    emails = extract_emails_from_html(response.text)
                    if not emails:
                        continue
                        
                    # Get page info
                    page_info = extract_info_from_page(soup)
                    company_name = extract_company_name(soup, url)
                    
                    # Process each email
                    for email in emails:
                        if not is_valid_contact_email(email):
                            continue
                            
                        # Save lead
                        lead = save_lead(
                            session=session,
                            email=email,
                            company=company_name,
                            url=url
                        )
                        
                        if lead:
                            # Save lead source
                            save_lead_source(
                                session=session,
                                lead_id=lead.id,
                                search_term_id=None,
                                url=url,
                                http_status=response.status_code,
                                scrape_duration=str(time.time()),
                                page_title=page_info.get('title'),
                                meta_description=page_info.get('description'),
                                content=page_info.get('content'),
                                tags=page_info.get('tags'),
                                phone_numbers=page_info.get('phones')
                            )
                            
                            results['results'].append({
                                'Email': email,
                                'Company': company_name,
                                'URL': url
                            })
                            results['total_leads'] += 1
                            
                            # Send email if enabled
                            if enable_email_sending and email_template and from_email:
                                template = session.query(EmailTemplate).get(int(email_template))
                                if template:
                                    wrapped_content = wrap_email_body(template.body_content)
                                    response = send_email_ses(
                                        session=session,
                                        from_email=from_email,
                                        to_email=email,
                                        subject=template.subject,
                                        body=wrapped_content,
                                        reply_to=reply_to
                                    )
                                    
                                    if response:
                                        save_email_campaign(
                                            session=session,
                                            lead_email=email,
                                            template_id=template.id,
                                            status='sent',
                                            sent_at=datetime.utcnow(),
                                            subject=template.subject,
                                            message_id=response.get('MessageId'),
                                            email_body=wrapped_content
                                        )
                                        results['email_logs'].append(f'Email sent to {email}')
                    
                    processed_urls.add(url)
                    processed_domains.add(domain)
                    
                except Exception as e:
                    results['search_logs'].append(f'Error processing URL {url}: {str(e)}')
                    continue
                    
        except Exception as e:
            results['search_logs'].append(f'Error searching term {term}: {str(e)}')
            continue
            
    return results

def fetch_projects(session: Session):
    return session.query(Project).all()

def fetch_campaigns(session: Session, project_id: int):
    return session.query(Campaign).filter_by(project_id=project_id).all()

def fetch_email_templates(session: Session):
    return session.query(EmailTemplate).all()

def fetch_email_settings(session: Session):
    return session.query(EmailSettings).all()

def fetch_leads_with_sources(session: Session):
    results = []
    leads = session.query(Lead).order_by(Lead.created_at.desc()).all()
    for lead in leads:
        # Filter out None values and convert all URLs to strings
        source_urls = [ls.url for ls in lead.lead_sources if ls and ls.url]
        last_campaign = (
            session.query(EmailCampaign)
            .filter_by(lead_id=lead.id)
            .order_by(EmailCampaign.sent_at.desc())
            .first()
        )
        last_contact = last_campaign.sent_at if (last_campaign and last_campaign.sent_at) else None
        last_status = last_campaign.status if last_campaign else "Not Contacted"
        results.append(
            {
                "id": lead.id,
                "email": lead.email,
                "first_name": lead.first_name or "",
                "last_name": lead.last_name or "",
                "company": lead.company or "",
                "job_title": lead.job_title or "",
                "created_at": lead.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "source": ", ".join(source_urls) if source_urls else "",
                "last_contact": last_contact.strftime("%Y-%m-%d %H:%M:%S") if last_contact else "",
                "last_status": last_status,
            }
        )
    return results

def fetch_all_email_logs(session: Session):
    # Return a pandas-like structure or dict for email logs
    logs = session.query(EmailCampaign).order_by(EmailCampaign.sent_at.desc()).all()
    return [{'ID': log.id, 'SentAt': log.sent_at, 'Email': log.lead.email if log.lead else '',
            'Template': log.template.template_name if log.template else '',
            'Subject': log.original_subject, 'Status': log.status,
            'MessageID': log.message_id} for log in logs]

def bulk_send_emails(session, template_id, from_email, reply_to, lead_list):
    template = session.query(EmailTemplate).get(template_id)
    sent_count = 0
    logs = []

    for lead in lead_list:
        try:
            # Send email
            wrapped_body = wrap_email_body(template.body_content)
            response, msg_id = send_email_ses(
                session,
                from_email=from_email,
                to_email=lead["Email"],
                subject=template.subject,
                body=wrapped_body,
                reply_to=reply_to
            )

            if response:
                sent_count += 1
                # Save email campaign
                save_email_campaign(
                    session,
                    lead_email=lead["Email"],
                    template_id=template_id,
                    status="sent",
                    sent_at=datetime.utcnow(),
                    subject=template.subject,
                    message_id=msg_id,
                    email_body=wrapped_body,
                )
                logs.append(f"Email sent successfully to {lead['Email']}")
            else:
                logs.append(f"Failed to send email to {lead['Email']}")

        except Exception as e:
            logs.append(f"Error sending to {lead['Email']}: {str(e)}")

    return logs, sent_count

def delete_lead_and_sources(session: Session, lead_id: int):
    try:
        session.query(LeadSource).filter(LeadSource.lead_id == lead_id).delete()
        session.query(CampaignLead).filter(CampaignLead.lead_id == lead_id).delete()
        session.query(EmailCampaign).filter(EmailCampaign.lead_id == lead_id).delete()
        lead = session.query(Lead).filter_by(id=lead_id).first()
        if lead:
            session.delete(lead)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting lead {lead_id} and its sources: {e}")
        return False

def update_lead(session: Session, lead_id: int, updated_data: dict):
    try:
        lead = session.query(Lead).filter_by(id=lead_id).first()
        if lead:
            for k, v in updated_data.items():
                # Map "First Name" -> "first_name" if needed
                field_name = k.lower().replace(" ", "_")
                setattr(lead, field_name, v)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        logging.error(f"Error updating lead: {e}")
        return False

def initialize_database():
    retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(retries):
        try:
            Base.metadata.create_all(bind=engine)
            logging.info("Database initialized successfully")
            
            # Create default project and campaign if they don't exist
            with db_session() as session:
                if not session.query(Project).first():
                    default_project = Project(project_name="Default Project")
                    session.add(default_project)
                    session.commit()
                    
                    default_campaign = Campaign(
                        campaign_name="Default Campaign",
                        project_id=default_project.id
                    )
                    session.add(default_campaign)
                    
                    # Create default search term group
                    default_group = SearchTermGroup(
                        name="Default Group",
                        description="Default search term group"
                    )
                    session.add(default_group)
                    
                    # Create default email settings if none exist
                    if not session.query(EmailSettings).first():
                        default_settings = EmailSettings(
                            name="Default Settings",
                            email="default@example.com",
                            provider="AWS SES"
                        )
                        session.add(default_settings)
                    
                    session.commit()
                    
            return
        except Exception as e:
            if attempt == retries - 1:
                logging.error(f"Failed to initialize database after {retries} attempts: {str(e)}")
                raise
            logging.warning(f"Database initialization attempt {attempt + 1} failed: {str(e)}")
            time.sleep(retry_delay)

def cleanup_database():
    try:
        engine.dispose()
        logging.info("Database connections cleaned up")
    except Exception as e:
        logging.error(f"Error cleaning up database connections: {str(e)}")

# ------------- DATABASE SETUP -------------
# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------- FASTAPI APP -------------
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test endpoint with DB connection check
@app.get("/test")
def test(db: Session = Depends(get_db)):
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        return {"status": "ok", "message": "FastAPI app is running, database connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# ------------- HOME PAGE -------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8"/>
        <title>AutoclientAI - FastAPI Enhanced</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
    </head>
    <body>
      <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
          <a class="navbar-brand" href="#">AutoclientAI</a>
          <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
              aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
              <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav me-auto">
              <li class="nav-item"><a class="nav-link" href="/manual_search">Manual Search</a></li>
              <li class="nav-item"><a class="nav-link" href="/bulk_send">Bulk Send</a></li>
              <li class="nav-item"><a class="nav-link" href="/view_leads">View Leads</a></li>
              <li class="nav-item"><a class="nav-link" href="/search_terms">Search Terms</a></li>
              <li class="nav-item"><a class="nav-link" href="/email_templates">Email Templates</a></li>
              <li class="nav-item"><a class="nav-link" href="/knowledge_base">Knowledge Base</a></li>
              <li class="nav-item"><a class="nav-link" href="/autoclient_ai">AutoclientAI</a></li>
              <li class="nav-item"><a class="nav-link" href="/automation_control">Automation Control</a></li>
              <li class="nav-item"><a class="nav-link" href="/manual_search_worker">Manual Search Worker</a></li>
              <li class="nav-item"><a class="nav-link" href="/email_logs">Email Logs</a></li>
              <li class="nav-item"><a class="nav-link" href="/sent_campaigns">Sent Campaigns</a></li>
              <li class="nav-item"><a class="nav-link" href="/settings">Settings</a></li>
              <li class="nav-item"><a class="nav-link" href="/projects_campaigns">Projects &amp; Campaigns</a></li>
            </ul>
          </div>
        </div>
      </nav>
      <div class="container mt-4">
        <h1>AutoclientAI - FastAPI</h1>
        <p>Welcome! This is the FastAPI version replicating the entire Streamlit app's logic.</p>
      </div>
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

# ------------- MANUAL SEARCH PAGE (GET FORM) -------------
@app.get("/manual_search", response_class=HTMLResponse)
def manual_search_form():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8"/>
        <title>Manual Search</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
    </head>
    <body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">AutoclientAI</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
            aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
          <ul class="navbar-nav me-auto">
            <li class="nav-item"><a class="nav-link active" href="/">Home</a></li>
            <li class="nav-item"><a class="nav-link" href="/bulk_send">Bulk Send</a></li>
            <li class="nav-item"><a class="nav-link" href="/view_leads">View Leads</a></li>
          </ul>
        </div>
      </div>
    </nav>

    <div class="container mt-4">
        <h2>Manual Search</h2>
        <form id="searchForm">
          <div class="mb-3">
            <label class="form-label">Search Terms (comma-separated):</label>
            <input type="text" class="form-control" name="search_terms" placeholder="software engineer, data scientist">
          </div>

          <div class="mb-3">
            <label class="form-label">Number of results per term:</label>
            <input type="number" class="form-control" name="num_results" value="10" min="1" max="100">
          </div>

          <div class="mb-3">
            <label class="form-label">Language:</label>
            <select class="form-select" name="language">
              <option value="ES">Spanish</option>
              <option value="EN">English</option>
            </select>
          </div>

          <div class="form-check mb-3">
            <input class="form-check-input" type="checkbox" name="ignore_previously_fetched" checked>
            <label class="form-check-label">Ignore previously fetched domains</label>
          </div>

          <div class="form-check mb-3">
            <input class="form-check-input" type="checkbox" name="optimize_english">
            <label class="form-check-label">Optimize for English</label>
          </div>

          <div class="form-check mb-3">
            <input class="form-check-input" type="checkbox" name="optimize_spanish">
            <label class="form-check-label">Optimize for Spanish</label>
          </div>

          <div class="form-check mb-3">
            <input class="form-check-input" type="checkbox" name="shuffle_keywords">
            <label class="form-check-label">Shuffle Keywords</label>
          </div>

          <hr>
          <div class="card mb-3">
            <div class="card-header">
              <div class="form-check">
                <input class="form-check-input" type="checkbox" name="enable_email_sending" id="enableEmail">
                <label class="form-check-label" for="enableEmail">Enable Email Sending</label>
              </div>
            </div>
            <div class="card-body" id="emailSettings">
              <div class="mb-3">
                <label class="form-label">From Email:</label>
                <input type="email" class="form-control" name="from_email">
              </div>

              <div class="mb-3">
                <label class="form-label">Reply To:</label>
                <input type="email" class="form-control" name="reply_to">
              </div>

              <div class="mb-3">
                <label class="form-label">Email Template (ID:Name):</label>
                <input type="text" class="form-control" name="email_template">
              </div>
            </div>
          </div>

          <button type="submit" class="btn btn-primary">Search</button>
        </form>

        <div id="results-container" class="mt-4"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
    document.getElementById('enableEmail').addEventListener('change', function() {
        const emailSettings = document.getElementById('emailSettings');
        emailSettings.style.display = this.checked ? 'block' : 'none';
    });

    document.getElementById('searchForm').addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        document.getElementById('results-container').innerHTML = '<div class="alert alert-info">Searching...</div>';
        
        fetch('/manual_search', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(html => {
            document.getElementById('results-container').innerHTML = html;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('results-container').innerHTML = 
                '<div class="alert alert-danger">An error occurred while searching.</div>';
        });
    });

    // Initialize email settings visibility
    document.getElementById('emailSettings').style.display = 
        document.getElementById('enableEmail').checked ? 'block' : 'none';
    </script>
    </body>
    </html>
    """

# ------------- MANUAL SEARCH (POST) -------------
@app.post("/manual_search")
async def do_manual_search(
    request: Request,
    search_terms: str = Form(...),
    num_results: int = Form(10),
    language: str = Form("ES"),
    ignore_previously_fetched: bool = Form(True),
    optimize_english: bool = Form(False),
    optimize_spanish: bool = Form(False),
    shuffle_keywords: bool = Form(False),
    enable_email_sending: bool = Form(False),
    from_email: Optional[str] = Form(None),
    reply_to: Optional[str] = Form(None),
    email_template: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        terms = [t.strip() for t in search_terms.split(",") if t.strip()]
        
        results = manual_search(
            session=db,
            terms=terms,
            num_results=num_results,
            language=language,
            ignore_previously_fetched=ignore_previously_fetched,
            optimize_english=optimize_english,
            optimize_spanish=optimize_spanish,
            shuffle_keywords_option=shuffle_keywords,
            enable_email_sending=enable_email_sending,
            from_email=from_email,
            reply_to=reply_to,
            email_template=email_template
        )
        
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------- BULK SEND -------------
@app.get("/bulk_send", response_class=HTMLResponse)
def bulk_send_form(session: Session = Depends(db_session)):
    try:
        templates = session.query(EmailTemplate).all()
        template_options = "".join([f'<option value="{t.id}">{t.template_name}</option>' for t in templates])
        
        return f"""
        <html>
        <head>
            <title>Bulk Send</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
        </head>
        <body class="bg-light">
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
                <div class="container-fluid">
                  <a class="navbar-brand" href="/">AutoclientAI</a>
                  <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
                      aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                      <span class="navbar-toggler-icon"></span>
                  </button>
                  <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav me-auto">
                      <li class="nav-item"><a class="nav-link" href="/manual_search">Manual Search</a></li>
                      <li class="nav-item"><a class="nav-link active" href="/bulk_send">Bulk Send</a></li>
                      <li class="nav-item"><a class="nav-link" href="/view_leads">View Leads</a></li>
                    </ul>
                  </div>
                </div>
            </nav>
            
            <div class="container mt-4">
                <h1>Bulk Send</h1>
                <form id="bulkSendForm" method="post">
                    <div class="mb-3">
                        <label class="form-label">Email Template</label>
                        <select name="template_id" class="form-select" required>
                            {template_options}
                        </select>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">From Email</label>
                        <input type="email" name="from_email" class="form-control" required>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">Reply To</label>
                        <input type="email" name="reply_to" class="form-control" required>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">Leads (one email per line)</label>
                        <textarea name="leads" rows="10" class="form-control" required></textarea>
                    </div>

                    <button type="submit" class="btn btn-primary">Send Emails</button>
                </form>
                <div id="results" class="mt-4"></div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                document.getElementById('bulkSendForm').onsubmit = async (e) => {{
                    e.preventDefault();
                    const form = e.target;
                    const results = document.getElementById('results');
                    results.innerHTML = '<div class="alert alert-info">Sending emails...</div>';
                    
                    try {{
                        const response = await fetch('/bulk_send', {{
                            method: 'POST',
                            body: new FormData(form)
                        }});
                        
                        const data = await response.text();
                        results.innerHTML = data;
                    }} catch (error) {{
                        results.innerHTML = '<div class="alert alert-danger">Error: ' + error.message + '</div>';
                    }}
                }};
            </script>
        </body>
        </html>
        """
    except Exception as e:
        logging.error(f"Error in bulk_send_form: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk_send")
async def bulk_send(
    request: Request,
    template_id: int = Form(...),
    from_email: str = Form(...),
    reply_to: str = Form(...),
    leads: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        lead_list = [email.strip() for email in leads.split(',')]
        logs, sent_count = bulk_send_emails(db, template_id, from_email, reply_to, lead_list)
        return JSONResponse(content={"success": True, "sent_count": sent_count, "logs": logs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------- VIEW LEADS -------------
@app.get("/view_leads", response_class=HTMLResponse)
def view_leads_page(db: Session = Depends(get_db)):
    leads = fetch_leads_with_sources(db)
    
    # Build HTML table rows
    rows = ""
    for lead in leads:
        rows += f"""
        <tr>
            <td>{lead['id']}</td>
            <td>{lead['email']}</td>
            <td>{lead['first_name']}</td>
            <td>{lead['last_name']}</td>
            <td>{lead['company']}</td>
            <td>{lead['job_title']}</td>
            <td>{lead['created_at']}</td>
            <td>{lead['source']}</td>
            <td>{lead['last_contact']}</td>
            <td>{lead['last_status']}</td>
        </tr>
        """
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>View Leads - AutoclientAI</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
    </head>
    <body class="bg-light">
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="/">AutoclientAI</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
                    aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav me-auto">
                        <li class="nav-item"><a class="nav-link" href="/manual_search">Manual Search</a></li>
                        <li class="nav-item"><a class="nav-link" href="/bulk_send">Bulk Send</a></li>
                        <li class="nav-item"><a class="nav-link active" href="/view_leads">View Leads</a></li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <h2>View Leads</h2>
            <table class="table table-striped table-bordered">
                <thead class="table-dark">
                    <tr>
                        <th>ID</th>
                        <th>Email</th>
                        <th>First Name</th>
                        <th>Last Name</th>
                        <th>Company</th>
                        <th>Job Title</th>
                        <th>Created At</th>
                        <th>Source</th>
                        <th>Last Contact</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

# ------------- SEARCH TERMS PAGE -------------
@app.get("/search_terms", response_class=HTMLResponse)
def search_terms_page(session: Session = Depends(db_session)):
    campaign_id = get_active_campaign_id()
    groups = session.query(SearchTermGroup).all()
    search_terms = session.query(SearchTerm).filter_by(campaign_id=campaign_id).all()
    
    # Build the groups list HTML
    groups_html = "".join([
        f'<li class="list-group-item">{g.name} (ID: {g.id})</li>'
        for g in groups
    ])
    
    # Build the search terms table HTML
    terms_html = "".join([
        f'''<tr>
            <td>{st.id}</td>
            <td>{st.term}</td>
            <td>{st.category or ''}</td>
            <td>{st.language}</td>
            <td>{st.group.name if st.group else ''}</td>
        </tr>'''
        for st in search_terms
    ])
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Search Terms Management</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
    </head>
    <body class="bg-light">
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="/">AutoclientAI</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
                    aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav me-auto">
                        <li class="nav-item"><a class="nav-link" href="/manual_search">Manual Search</a></li>
                        <li class="nav-item"><a class="nav-link" href="/bulk_send">Bulk Send</a></li>
                        <li class="nav-item"><a class="nav-link" href="/view_leads">View Leads</a></li>
                        <li class="nav-item"><a class="nav-link active" href="/search_terms">Search Terms</a></li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <h1>Search Terms Management</h1>
            
            <h2>Search Term Groups</h2>
            <ul class="list-group mb-4">
                {groups_html}
            </ul>
            
            <h2>Search Terms for Campaign {campaign_id}</h2>
            <table class="table table-bordered table-striped">
                <thead class="table-dark">
                    <tr>
                        <th>ID</th>
                        <th>Term</th>
                        <th>Category</th>
                        <th>Language</th>
                        <th>Group</th>
                    </tr>
                </thead>
                <tbody>
                {terms_html}
                </tbody>
            </table>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

# ------------- EMAIL TEMPLATES PAGE -------------
@app.get("/email_templates", response_class=HTMLResponse)
def email_templates_page(session: Session = Depends(db_session)):
    campaign_id = get_active_campaign_id()
    tpls = session.query(EmailTemplate).filter_by(campaign_id=campaign_id).all()
    
    # Build the table rows HTML
    rows_html = "".join([
        f'''<tr>
            <td>{t.id}</td>
            <td>{t.template_name}</td>
            <td>{t.subject}</td>
            <td>{t.body_content[:100]}...</td>
            <td>{t.language}</td>
            <td>{"Yes" if t.is_ai_customizable else "No"}</td>
            <td>{t.created_at}</td>
        </tr>'''
        for t in tpls
    ])
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Email Templates - AutoclientAI</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
    </head>
    <body class="bg-light">
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="/">AutoclientAI</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
                    aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav me-auto">
                        <li class="nav-item"><a class="nav-link" href="/manual_search">Manual Search</a></li>
                        <li class="nav-item"><a class="nav-link" href="/bulk_send">Bulk Send</a></li>
                        <li class="nav-item"><a class="nav-link" href="/view_leads">View Leads</a></li>
                        <li class="nav-item"><a class="nav-link" href="/search_terms">Search Terms</a></li>
                        <li class="nav-item"><a class="nav-link active" href="/email_templates">Email Templates</a></li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <h2>Email Templates (Campaign: {campaign_id})</h2>
            <table class="table table-bordered table-striped">
                <thead class="table-dark">
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Subject</th>
                        <th>Content</th>
                        <th>Language</th>
                        <th>AI Custom?</th>
                        <th>CreatedAt</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

# ------------- PROJECTS & CAMPAIGNS -------------
@app.get("/projects_campaigns", response_class=HTMLResponse)
def projects_campaigns_page():
    with db_session() as session:
        projs = fetch_projects(session)
        data = []
        for p in projs:
            c = fetch_campaigns(session, p.id)
            data.append((p, c))

    # Build the projects HTML
    html_projs = "".join([
        f'''<div class="card mb-3">
            <div class="card-header">Project: {p.project_name} (ID: {p.id})</div>
            <div class="card-body">
                <ul>
                {"".join([f"<li>{c.campaign_name} (ID: {c.id})</li>" for c in c_list])}
                </ul>
            </div>
        </div>'''
        for p, c_list in data
    ])
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Projects & Campaigns - AutoclientAI</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
    </head>
    <body class="bg-light">
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="/">AutoclientAI</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
                    aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav me-auto">
                        <li class="nav-item"><a class="nav-link" href="/manual_search">Manual Search</a></li>
                        <li class="nav-item"><a class="nav-link" href="/bulk_send">Bulk Send</a></li>
                        <li class="nav-item"><a class="nav-link" href="/view_leads">View Leads</a></li>
                        <li class="nav-item"><a class="nav-link" href="/search_terms">Search Terms</a></li>
                        <li class="nav-item"><a class="nav-link" href="/email_templates">Email Templates</a></li>
                        <li class="nav-item"><a class="nav-link" href="/knowledge_base">Knowledge Base</a></li>
                        <li class="nav-item"><a class="nav-link" href="/autoclient_ai">AutoclientAI</a></li>
                        <li class="nav-item"><a class="nav-link" href="/automation_control">Automation Control</a></li>
                        <li class="nav-item"><a class="nav-link" href="/manual_search_worker">Manual Search Worker</a></li>
                        <li class="nav-item"><a class="nav-link" href="/email_logs">Email Logs</a></li>
                        <li class="nav-item"><a class="nav-link" href="/sent_campaigns">Sent Campaigns</a></li>
                        <li class="nav-item"><a class="nav-link" href="/settings">Settings</a></li>
                        <li class="nav-item"><a class="nav-link active" href="/projects_campaigns">Projects &amp; Campaigns</a></li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <h2>Projects & Campaigns</h2>
            {html_projs}
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

# ------------- KNOWLEDGE BASE -------------
@app.get("/knowledge_base", response_class=HTMLResponse)
def knowledge_base_page(session: Session = Depends(db_session)):
    pid = get_active_project_id()
    kb = session.query(KnowledgeBase).filter_by(project_id=pid).first()
    
    style = """
        body { font-family: Arial, sans-serif; }
        .container { max-width: 800px; margin: 20px auto; }
    """
    
    if not kb:
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Knowledge Base - New</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
            <style>{style}</style>
        </head>
        <body class="bg-light">
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
                <div class="container-fluid">
                    <a class="navbar-brand" href="/">AutoclientAI</a>
                </div>
            </nav>
            <div class="container mt-4">
                <h2>Knowledge Base (Project ID: {pid})</h2>
                <p>No Knowledge Base found. Create one:</p>
                <form method="post" action="/knowledge_base">
                    <div class="mb-3">
                        <label class="form-label">KB Name:</label>
                        <input type="text" class="form-control" name="kb_name" value="">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">KB Bio:</label>
                        <textarea class="form-control" name="kb_bio" rows="5"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">KB Values:</label>
                        <textarea class="form-control" name="kb_values" rows="3"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Contact Name:</label>
                        <input type="text" class="form-control" name="contact_name" value="">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Contact Role:</label>
                        <input type="text" class="form-control" name="contact_role" value="">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Contact Email:</label>
                        <input type="email" class="form-control" name="contact_email" value="">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Company Description:</label>
                        <textarea class="form-control" name="company_description" rows="3"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Company Mission:</label>
                        <textarea class="form-control" name="company_mission" rows="5"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Company Target Market:</label>
                        <textarea class="form-control" name="company_target_market" rows="5"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Company Other:</label>
                        <textarea class="form-control" name="company_other" rows="5"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Product Name:</label>
                        <input type="text" class="form-control" name="product_name" value="">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Product Description:</label>
                        <textarea class="form-control" name="product_description" rows="5"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Product Target Customer:</label>
                        <textarea class="form-control" name="product_target_customer" rows="5"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Product Other:</label>
                        <textarea class="form-control" name="product_other" rows="5"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Other Context:</label>
                        <textarea class="form-control" name="other_context" rows="5"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Example Email:</label>
                        <textarea class="form-control" name="example_email" rows="5"></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary">Save</button>
                </form>
            </div>
        </body>
        </html>
        """
    else:
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Knowledge Base - Edit</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
            <style>{style}</style>
        </head>
        <body class="bg-light">
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
                <div class="container-fluid">
                    <a class="navbar-brand" href="/">AutoclientAI</a>
                </div>
            </nav>
            <div class="container mt-4">
            <h2>Knowledge Base (Project ID: {pid})</h2>
            <form method="post" action="/knowledge_base">
                <div class="mb-3">
                    <label class="form-label">KB Name:</label>
                    <input type="text" class="form-control" name="kb_name" value="{kb.kb_name or ''}">
                </div>
                <div class="mb-3">
                    <label class="form-label">KB Bio:</label>
                    <textarea class="form-control" name="kb_bio" rows="5">{kb.kb_bio or ''}</textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">KB Values:</label>
                    <textarea class="form-control" name="kb_values" rows="3">{kb.kb_values or ''}</textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">Contact Name:</label>
                    <input type="text" class="form-control" name="contact_name" value="{kb.contact_name or ''}">
                </div>
                <div class="mb-3">
                    <label class="form-label">Contact Role:</label>
                    <input type="text" class="form-control" name="contact_role" value="{kb.contact_role or ''}">
                </div>
                <div class="mb-3">
                    <label class="form-label">Contact Email:</label>
                    <input type="email" class="form-control" name="contact_email" value="{kb.contact_email or ''}">
                </div>
                <div class="mb-3">
                    <label class="form-label">Company Description:</label>
                    <textarea class="form-control" name="company_description" rows="3">{kb.company_description or ''}</textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">Company Mission:</label>
                    <textarea class="form-control" name="company_mission" rows="5">{kb.company_mission or ''}</textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">Company Target Market:</label>
                    <textarea class="form-control" name="company_target_market" rows="5">{kb.company_target_market or ''}</textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">Company Other:</label>
                    <textarea class="form-control" name="company_other" rows="5">{kb.company_other or ''}</textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">Product Name:</label>
                    <input type="text" class="form-control" name="product_name" value="{kb.product_name or ''}">
                </div>
                <div class="mb-3">
                    <label class="form-label">Product Description:</label>
                    <textarea class="form-control" name="product_description" rows="5">{kb.product_description or ''}</textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">Product Target Customer:</label>
                    <textarea class="form-control" name="product_target_customer" rows="5">{kb.product_target_customer or ''}</textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">Product Other:</label>
                    <textarea class="form-control" name="product_other" rows="5">{kb.product_other or ''}</textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">Other Context:</label>
                    <textarea class="form-control" name="other_context" rows="5">{kb.other_context or ''}</textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">Example Email:</label>
                    <textarea class="form-control" name="example_email" rows="5">{kb.example_email or ''}</textarea>
                </div>
                <button type="submit" class="btn btn-primary">Save</button>
            </form>
            </div>
        </body>
        </html>
        """

@app.post("/knowledge_base", response_class=HTMLResponse)
def knowledge_base_update(
    kb_name: str = Form(""),
    kb_bio: str = Form(""),
    kb_values: str = Form(""),
    contact_name: str = Form(""),
    contact_role: str = Form(""),
    contact_email: str = Form(""),
    company_description: str = Form(""),
    company_mission: str = Form(""),
    company_target_market: str = Form(""),
    company_other: str = Form(""),
    product_name: str = Form(""),
    product_description: str = Form(""),
    product_target_customer: str = Form(""),
    product_other: str = Form(""),
    other_context: str = Form(""),
    example_email: str = Form(""),
):
    pid = get_active_project_id()
    with db_session() as session:
        kb_entry = session.query(KnowledgeBase).filter_by(project_id=pid).first()
        if not kb_entry:
            kb_entry = KnowledgeBase(project_id=pid)
            session.add(kb_entry)
        kb_entry.kb_name = kb_name
        kb_entry.kb_bio = kb_bio
        kb_entry.kb_values = kb_values
        kb_entry.contact_name = contact_name
        kb_entry.contact_role = contact_role
        kb_entry.contact_email = contact_email
        kb_entry.company_description = company_description
        kb_entry.company_mission = company_mission
        kb_entry.company_target_market = company_target_market
        kb_entry.company_other = company_other
        kb_entry.product_name = product_name
        kb_entry.product_description = product_description
        kb_entry.product_target_customer = product_target_customer
        kb_entry.product_other = product_other
        kb_entry.other_context = other_context
        kb_entry.example_email = example_email
        session.commit()
    return RedirectResponse("/knowledge_base", status_code=302)

# ------------- AI AUTOMATION -------------
@app.get("/ai_automation", response_class=HTMLResponse)
def ai_automation(session: Session = Depends(db_session)):
    campaign_id = get_active_campaign_id()
    campaign = session.query(Campaign).get(campaign_id)

    if not campaign:
        return """
        <html><body style="font-family:Arial;margin:20px;">
        <h2>AI Automation</h2>
        <p>No active campaign found.</p>
        <hr/>
        <a href="/">Home</a>
        </body></html>
        """

    automation_logs = session.query(AutomationLog).filter_by(campaign_id=campaign_id).order_by(AutomationLog.start_time.desc()).all()
    logs_html = ""
    for log in automation_logs:
        logs_html += f"""
        <tr>
          <td>{log.id}</td>
          <td>{log.search_term.term if log.search_term else ''}</td>
          <td>{log.leads_gathered}</td>
          <td>{log.emails_sent}</td>
          <td>{log.start_time}</td>
          <td>{log.end_time}</td>
          <td>{log.status}</td>
        </tr>
        
        """

    return f"""
    <html><body style="font-family:Arial;margin:20px;">
    <h2>AI Automation (Campaign ID: {campaign_id})</h2>
    <p>Current Campaign: {campaign.campaign_name}</p>
    <p>Loop Automation: {'Enabled' if campaign.loop_automation else 'Disabled'}</p>
    <p>Auto Send: {'Enabled' if campaign.auto_send else 'Disabled'}</p>
    <h3>Automation Logs</h3>
    <table border="1" cellpadding="6" cellspacing="0">
      <tr>
        <th>Log ID</th>
        <th>Search Term</th>
        <th>Leads Gathered</th>
        <th>Emails Sent</th>
        <th>Start Time</th>
        <th>End Time</th>
        <th>Status</th>
      </tr>
      {logs_html}
    </table>
    <hr/>
    <a href="/">Home</a>
    </body></html>
    """

# ------------- SETTINGS PAGE -------------
@app.get("/settings", response_class=HTMLResponse)
def settings_page(db: Session = Depends(get_db)):
    try:
        email_sets = fetch_email_settings(db)
        ai_settings = db.query(Settings).filter_by(setting_type="ai").first()
        auto_settings = db.query(Settings).filter_by(setting_type="automation").first()
        
        return JSONResponse(content={
            "email_settings": [es.__dict__ for es in email_sets],
            "ai_settings": ai_settings.value if ai_settings else {},
            "auto_settings": auto_settings.value if auto_settings else {}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional route to demonstrate DB is connected
@app.get("/ping_db")
def ping_db():
    try:
        with db_session() as session:
            session.execute(text("SELECT 1"))
        return {"message": "DB connection OK"}
    except:
        raise HTTPException(status_code=500, detail="DB connection failed")

def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# ------------- EMAIL LOGS PAGE -------------
@app.get("/email_logs", response_class=HTMLResponse)
def email_logs_page(session: Session = Depends(db_session)):
    df = fetch_all_email_logs(session)
    rows = ""
    for _, row in df.iterrows():
        rows += f"""
        <tr>
            <td>{row['ID']}</td>
            <td>{row['SentAt']}</td>
            <td>{row['Email']}</td>
            <td>{row['Template']}</td>
            <td>{row['Subject']}</td>
            <td>{row['Status']}</td>
            <td>{row['MessageID']}</td>
        </tr>
        """
    return f"""
    <html><body style="font-family:Arial;margin:20px;">
    <h2>Email Logs</h2>
    <table border="1" cellpadding="6" cellspacing="0">
        <tr>
            <th>ID</th>
            <th>Sent At</th>
            <th>Email</th>
            <th>Template</th>
            <th>Subject</th>
            <th>Status</th>
            <th>Message ID</th>
        </tr>
        {rows}
    </table>
    <hr/>
    <a href="/">Home</a>
    </body></html>
    """

# ------------- SENT CAMPAIGNS PAGE -------------
@app.get("/sent_campaigns", response_class=HTMLResponse)
def sent_campaigns_page(session: Session = Depends(db_session)):
    campaign_id = get_active_campaign_id()
    sent = (
        session.query(EmailCampaign)
        .filter(EmailCampaign.campaign_id == campaign_id)
        .filter(EmailCampaign.status == "Sent")
        .order_by(EmailCampaign.sent_at.desc())
        .all()
    )
    
    rows = ""
    for ec in sent:
        rows += f"""
        <tr>
            <td>{ec.id}</td>
            <td>{ec.sent_at}</td>
            <td>{ec.lead.email if ec.lead else ''}</td>
            <td>{ec.template.template_name if ec.template else ''}</td>
            <td>{ec.status}</td>
            <td>{ec.open_count}</td>
            <td>{ec.click_count}</td>
        </tr>
        """
    
    return f"""
    <html><body style="font-family:Arial;margin:20px;">
    <h2>Sent Campaigns (Campaign ID: {campaign_id})</h2>
    <table border="1" cellpadding="6" cellspacing="0">
        <tr>
            <th>ID</th>
            <th>Sent At</th>
            <th>Email</th>
            <th>Template</th>
            <th>Status</th>
            <th>Opens</th>
            <th>Clicks</th>
        </tr>
        {rows}
    </table>
    <hr/>
    <a href="/">Home</a>
    </body></html>
    """

# ------------- AUTOMATION CONTROL PANEL -------------
@app.get("/automation_control", response_class=HTMLResponse)
def automation_control_panel(session: Session = Depends(db_session)):
    campaign_id = get_active_campaign_id()
    campaign = session.query(Campaign).get(campaign_id)
    
    if not campaign:
        return """
        <html><body style="font-family:Arial;margin:20px;">
        <h2>Automation Control Panel</h2>
        <p>No active campaign selected.</p>
        <hr/>
        <a href="/">Home</a>
        </body></html>
        """
        
    return f"""
    <html><body style="font-family:Arial;margin:20px;">
    <h2>Automation Control Panel</h2>
    <h3>Campaign: {campaign.campaign_name}</h3>
    
    <form method="post" action="/automation_control">
        <h4>Automation Settings</h4>
        <label>
            <input type="checkbox" name="loop_automation" {'checked' if campaign.loop_automation else ''}>
            Enable Loop Automation
        </label><br/>
        <label>
            <input type="checkbox" name="auto_send" {'checked' if campaign.auto_send else ''}>
            Enable Auto Send
        </label><br/>
        <label>
            <input type="checkbox" name="ai_customization" {'checked' if campaign.ai_customization else ''}>
            Enable AI Customization
        </label><br/><br/>
        
        <label>Max Emails Per Group:</label><br/>
        <input type="number" name="max_emails_per_group" value="{campaign.max_emails_per_group}"><br/><br/>
        
        <label>Loop Interval (minutes):</label><br/>
        <input type="number" name="loop_interval" value="{campaign.loop_interval}"><br/><br/>
        
        <button type="submit">Update Settings</button>
    </form>
    
    <h4>Automation Status</h4>
    <p>Loop Automation: {'Running' if campaign.loop_automation else 'Stopped'}</p>
    <p>Auto Send: {'Enabled' if campaign.auto_send else 'Disabled'}</p>
    <p>AI Customization: {'Enabled' if campaign.ai_customization else 'Disabled'}</p>
    
    <hr/>
    <a href="/">Home</a>
    </body></html>
    """

@app.post("/automation_control", response_class=HTMLResponse)
def update_automation_control(
    loop_automation: Optional[str] = Form(None),
    auto_send: Optional[str] = Form(None),
    ai_customization: Optional[str] = Form(None),
    max_emails_per_group: int = Form(...),
    loop_interval: int = Form(...),
):
    with db_session() as session:
        campaign_id = get_active_campaign_id()
        campaign = session.query(Campaign).get(campaign_id)
        if campaign:
            campaign.loop_automation = bool(loop_automation)
            campaign.auto_send = bool(auto_send)
            campaign.ai_customization = bool(ai_customization)
            campaign.max_emails_per_group = max_emails_per_group
            campaign.loop_interval = loop_interval
            session.commit()
    return RedirectResponse("/automation_control", status_code=302)

# ------------- MANUAL SEARCH WORKER -------------
@app.get("/manual_search_worker", response_class=HTMLResponse)
def manual_search_worker_page():
    return """
    <html><body style="font-family:Arial;margin:20px;">
    <h2>Manual Search Worker</h2>
    <form method="post" action="/manual_search_worker">
        <label>Search Terms (one per line):</label><br/>
        <textarea name="search_terms" rows="10" cols="60"></textarea><br/><br/>
        
        <label>Results per term:</label><br/>
        <input type="number" name="results_per_term" value="10"><br/><br/>
        
        <label>Language:</label><br/>
        <select name="language">
            <option value="ES">Spanish</option>
            <option value="EN">English</option>
        </select><br/><br/>
        
        <button type="submit">Start Search</button>
    </form>
    <div id="status"></div>
    <hr/>
    <a href="/">Home</a>
    
    <script>
    document.querySelector('form').addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        document.getElementById('status').innerHTML = '<p>Search started...</p>';
        
        fetch('/manual_search_worker', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('status').innerHTML = 
                '<h3>Search Results</h3>' +
                '<p>Total Leads Found: ' + data.total_leads + '</p>' +
                '<p>Processed Terms: ' + data.processed_terms + '</p>' +
                '<p>Status: ' + data.status + '</p>';
        })
        .catch(error => {
            document.getElementById('status').innerHTML = '<p>Error occurred during search</p>';
        });
    });
    </script>
    </body></html>
    """

@app.post("/manual_search_worker")
async def manual_search_worker_process(
    search_terms: str = Form(...),
    results_per_term: int = Form(...),
    language: str = Form(...),
):
    terms = [t.strip() for t in search_terms.splitlines() if t.strip()]
    total_leads = 0
    all_logs = []
    
    with db_session() as session:
        for term in terms:
            results = manual_search(
                session=session,
                terms=[term],
                num_results=results_per_term,
                language=language,
            )
            total_leads += results.get("total_leads", 0)
            all_logs.extend(results.get("search_logs", []))
    
    return {
        "status": "completed",
        "total_leads": total_leads,
        "processed_terms": len(terms),
        "logs": all_logs
    }

# ------------- AUTOCLIENT AI PAGE -------------
@app.get("/autoclient_ai", response_class=HTMLResponse)
def autoclient_ai_page(session: Session = Depends(db_session)):
    campaign_id = get_active_campaign_id()
    campaign = session.query(Campaign).get(campaign_id)
    
    if not campaign:
        return """
        <html><body style="font-family:Arial;margin:20px;">
        <h2>AutoclientAI</h2>
        <p>No active campaign selected.</p>
        <hr/>
        <a href="/">Home</a>
        </body></html>
        """
        
    ai_logs = (
        session.query(AIRequestLog)
        .order_by(AIRequestLog.created_at.desc())
        .limit(50)
        .all()
    )
    
    logs_html = ""
    for log in ai_logs:
        logs_html += f"""
        <tr>
            <td>{log.created_at}</td>
            <td>{log.function_name}</td>
            <td>{log.model_used}</td>
            <td>{log.prompt[:100]}...</td>
            <td>{log.response[:100]}...</td>
        </tr>
        """
    
    return f"""
    <html><body style="font-family:Arial;margin:20px;">
    <h2>AutoclientAI</h2>
    <h3>Campaign: {campaign.campaign_name}</h3>
    
    <h4>AI Settings</h4>
    <form method="post" action="/autoclient_ai">
        <label>
            <input type="checkbox" name="ai_customization" {'checked' if campaign.ai_customization else ''}>
            Enable AI Email Customization
        </label><br/><br/>
        
        <button type="submit">Update Settings</button>
    </form>
    
    <h4>Recent AI Interactions</h4>
    <table border="1" cellpadding="6" cellspacing="0">
        <tr>
            <th>Timestamp</th>
            <th>Function</th>
            <th>Model</th>
            <th>Prompt</th>
            <th>Response</th>
        </tr>
        {logs_html}
    </table>
    
    <hr/>
    <a href="/">Home</a>
    </body></html>
    """

@app.post("/autoclient_ai", response_class=HTMLResponse)
def update_autoclient_ai(
    ai_customization: Optional[str] = Form(None),
):
    with db_session() as session:
        campaign_id = get_active_campaign_id()
        campaign = session.query(Campaign).get(campaign_id)
        if campaign:
            campaign.ai_customization = bool(ai_customization)
            session.commit()
    return RedirectResponse("/autoclient_ai", status_code=302)

# ------------- BACKGROUND TASKS -------------
async def send_email_task(email_details):
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            with db_session() as session:
                await send_email_ses(
                    session=session,
                    from_email=email_details['from_email'],
                    to_email=email_details['to_email'],
                    subject=email_details['subject'],
                    body=email_details['body'],
                    reply_to=email_details.get('reply_to')
                )
                logging.info(f"Email sent successfully: {email_details}")
                return
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to send email after {max_retries} attempts: {str(e)}")
                raise
            logging.warning(f"Email send attempt {attempt + 1} failed: {str(e)}")
            time.sleep(retry_delay)  # Use time.sleep instead of asyncio.sleep since we're in a background task

@app.post("/send_email")
async def send_email(email_details: dict):
    try:
        asyncio.create_task(send_email_task(email_details))
        return {"message": "Email sending initiated in the background"}
    except Exception as e:
        logging.error(f"Failed to initiate email task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------- MAIN EXECUTION -------------
if __name__ == "__main__":
    import uvicorn
    import time
    import sys
    import asyncio
    import os
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.critical(f"Server failed to start: {str(e)}")
        sys.exit(1)