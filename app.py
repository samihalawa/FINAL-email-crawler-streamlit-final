from fastapi import FastAPI, HTTPException, Depends, Request, Form, Response, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import json
import re
import logging
import asyncio
import time
import requests
import pandas as pd
from openai import OpenAI
import boto3
import uuid
import aiohttp
import urllib3
import random
import html
import smtplib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select, text, distinct, and_, or_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from email_validator import validate_email, EmailNotValidError
from urllib.parse import urlparse, urlencode
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import signal
import subprocess

load_dotenv()

# Database Configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# SQLAlchemy Setup
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal, Base = sessionmaker(bind=engine), declarative_base()

# --- SQLAlchemy Models ---
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
    template_name, subject, body_content = Column(Text), Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False)
    language = Column(Text, default='ES')
    campaign = relationship("Campaign")
    email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
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
    setting_type = Column(Text, nullable=False)
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

# --- Database Functions ---

@contextmanager
def db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# --- FastAPI Setup ---
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Pydantic Models ---

class SettingsUpdate(BaseModel):
    openai_api_key: str
    openai_api_base: str
    openai_model: str

class EmailSettingCreate(BaseModel):
    name: str
    email: str
    provider: str
    smtp_server: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None

class EmailSettingUpdate(BaseModel):
    id: int
    name: str
    email: str
    provider: str
    smtp_server: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None

class SearchTermsInput(BaseModel):
    terms: List[str]
    num_results: int
    optimize_english: bool
    optimize_spanish: bool
    shuffle_keywords: bool
    language: str
    enable_email_sending: bool
    email_template: Optional[str] = None
    email_setting_option: Optional[str] = None
    reply_to: Optional[str] = None
    ignore_previously_fetched: Optional[bool] = None

class EmailTemplateCreate(BaseModel):
    template_name: str
    subject: str
    body_content: str
    is_ai_customizable: Optional[bool] = False
    language: Optional[str] = 'ES'

class EmailTemplateUpdate(BaseModel):
    id: int
    template_name: str
    subject: str
    body_content: str
    is_ai_customizable: Optional[bool] = False
    language: Optional[str] = 'ES'

class LeadUpdate(BaseModel):
    id: int
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None

class BulkSendInput(BaseModel):
    template_id: int
    from_email: str
    reply_to: str
    send_option: str
    specific_email: Optional[str] = None
    selected_terms: Optional[List[str]] = None
    exclude_previously_contacted: Optional[bool] = None

class ProjectCreate(BaseModel):
    project_name: str

class CampaignCreate(BaseModel):
    campaign_name: str
    project_id: int

class KnowledgeBaseCreate(BaseModel):
    project_id: int
    kb_name: Optional[str] = None
    kb_bio: Optional[str] = None
    kb_values: Optional[str] = None
    contact_name: Optional[str] = None
    contact_role: Optional[str] = None
    contact_email: Optional[str] = None
    company_description: Optional[str] = None
    company_mission: Optional[str] = None
    company_target_market: Optional[str] = None
    company_other: Optional[str] = None
    product_name: Optional[str] = None
    product_description: Optional[str] = None
    product_target_customer: Optional[str] = None
    product_other: Optional[str] = None
    other_context: Optional[str] = None
    example_email: Optional[str] = None

class SearchTermGroupCreate(BaseModel):
    name: str

class SearchTermGroupUpdate(BaseModel):
    group_id: int
    updated_terms: List[str]

class SearchTermCreate(BaseModel):
    term: str
    campaign_id: int
    group_for_new_term: Optional[str] = None

class GroupedSearchTerm(BaseModel):
    group_name: str
    terms: List[str]

class SearchTermsGrouping(BaseModel):
    grouped_terms: List[GroupedSearchTerm]
    ungrouped_terms: List[str]

# --- Helper Functions ---

def get_domain_from_url(url): 
    return urlparse(url).netloc

def is_valid_email(email):
    if email is None:
        return False
        
    invalid_patterns = [
        r".*(\.png|\.jpg|\.jpeg|\.gif|\.css|\.js)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^(email|info|contact|support|hello|hola|hi|salutations|greetings|inquiries|questions)@.*",
        r"email@email\.com",
        r".*@example\.com$",
        r".*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"
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
    chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
    return ' '.join(chunk for chunk in chunks if chunk)

def get_page_description(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    return meta_desc['content'] if meta_desc else "No description found"

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def safe_google_search(query, num_results=10, lang='es'):
    try:
        return list(google_search(query, num_results=num_results, lang=lang, stop=num_results))
    except Exception as e:
        logging.error(f"Google search error for '{query}': {str(e)}")
        return []

def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"Optimize and expand these search terms for lead generation:\n{', '.join(base_terms)}\n\nConsider:\n1. Relevance to business and target market\n2. Potential for high-quality leads\n3. Variations and related terms\n4. Industry-specific jargon\n\nRespond with a JSON array of optimized terms."
    response = openai_chat_completion([{"role": "system", "content": "You're an AI specializing in optimizing search terms for lead generation. Be concise and effective."}, {"role": "user", "content": prompt}], function_name="generate_optimized_search_terms")
    return response.get('optimized_terms', base_terms) if isinstance(response, dict) else base_terms

def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
    session.add(SearchTermEffectiveness(term=term, total_results=total_results, valid_leads=valid_leads, irrelevant_leads=total_results - valid_leads, blogs_found=blogs_found, directories_found=directories_found))
    session.commit()

def save_lead_source(session, lead_id, search_term_id, url, http_status, scrape_duration, page_title=None, meta_description=None, content=None, tags=None, phone_numbers=None):
    session.add(LeadSource(
        lead_id=lead_id,
        search_term_id=search_term_id,
        url=url,
        http_status=http_status,
        scrape_duration=scrape_duration,
        page_title=page_title or get_page_title(url),
        meta_description=meta_description or get_page_description(url),
        content=content or extract_visible_text(BeautifulSoup(requests.get(url).text, 'html.parser')),
        tags=tags,
        phone_numbers=phone_numbers
    ))
    session.commit()

def save_lead(session, email, first_name=None, last_name=None, company=None, job_title=None, phone=None, url=None, search_term_id=None, created_at=None):
    try:
        existing_lead = session.query(Lead).filter_by(email=email).first()
        if existing_lead:
            for attr in ['first_name', 'last_name', 'company', 'job_title', 'phone', 'created_at']:
                if locals()[attr]:
                    setattr(existing_lead, attr, locals()[attr])
            lead = existing_lead
        else:
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

        session.flush()

        lead_source = LeadSource(
            lead_id=lead.id,
            url=url,
            search_term_id=search_term_id
        )
        session.add(lead_source)

        campaign_lead = CampaignLead(
            campaign_id=get_active_campaign_id(),
            lead_id=lead.id,
            status="Not Contacted",
            created_at=datetime.utcnow()
        )
        session.add(campaign_lead)

        session.commit()
        return lead

    except Exception as e:
        logging.error(f"Error saving lead: {str(e)}")
        session.rollback()
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

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:
            raise HTTPException(status_code=400, detail="OpenAI API key not set. Please configure it in the settings.")

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
        with db_session() as session:
            log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id, model)
        raise HTTPException(status_code=500, detail=f"Error in OpenAI API call: {str(e)}")

def add_or_get_search_term(session, term, campaign_id, created_at=None):
    search_term = session.query(SearchTerm).filter_by(term=term, campaign_id=campaign_id).first()
    if not search_term:
        search_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=created_at or datetime.utcnow())
        session.add(search_term)
        session.commit()
        session.refresh(search_term)
        return search_term.id

def create_or_update_email_template(session, template_name, subject, body_content, template_id=None, is_ai_customizable=False, created_at=None, language='ES'):
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=get_active_campaign_id(), created_at=created_at or datetime.utcnow())
    if template_id: 
        template.template_name, template.subject, template.body_content, template.is_ai_customizable = template_name, subject, body_content, is_ai_customizable
    template.language = language
    session.add(template)
    session.commit()
    return template.id

def fetch_email_templates(session):
    return [f"{t.id}: {t.template_name}" for t in session.query(EmailTemplate).all()]

def fetch_campaigns(session):
    return [f"{camp.id}: {camp.campaign_name}" for camp in session.query(Campaign).all()]

def fetch_projects(session):
    return [f"{project.id}: {project.project_name}" for project in session.query(Project).all()]

def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        return []

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

def fetch_leads_with_sources(session):
    try:
        leads = session.query(Lead).options(joinedload(Lead.lead_sources)).all()
        leads_data = []
        for lead in leads:
            lead_info = {
                "id": lead.id,
                "email": lead.email,
                "first_name": lead.first_name,
                "last_name": lead.last_name,
                "company": lead.company,
                "job_title": lead.job_title,
                "created_at": lead.created_at.strftime('%Y-%m-%d %H:%M:%S') if lead.created_at else None,
                "sources": [{"url": source.url, "search_term": source.search_term.term if source.search_term else "N/A"} for source in lead.lead_sources]
            }
            leads_data.append(lead_info)
        return pd.DataFrame(leads_data)
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_leads_with_sources: {str(e)}")
        return pd.DataFrame()

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

def get_latest_logs(automation_log_id):
    with db_session() as session:
        log = session.query(AutomationLog).get(automation_log_id)
        return log.logs if log else []

def is_process_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def get_automation_status(automation_log_id):
    try:
        with db_session() as session:
            log = session.query(AutomationLog).get(automation_log_id)
            if not log:
                return {'status': 'unknown', 'leads_gathered': 0, 'emails_sent': 0, 'latest_logs': []}
            return {
                'status': log.status,
                'leads_gathered': log.leads_gathered or 0,
                'emails_sent': log.emails_sent or 0,
                'latest_logs': log.logs[-10:] if log.logs else []
            }
    except Exception as e:
        logging.error(f"Error getting automation status: {e}")
        return {'status': 'error', 'leads_gathered': 0, 'emails_sent': 0, 'latest_logs': []}

def get_ai_response(prompt):
    return openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100).choices[0].text.strip()

# --- FastAPI Endpoints ---

# --- /settings ---
@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request, db: Session = Depends(db_session)):
    email_settings = db.query(EmailSettings).all()
    general_settings = db.query(Settings).filter_by(setting_type='general').first()
    general_settings_dict = general_settings.value if general_settings else {}

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "email_settings": email_settings,
        "general_settings": general_settings_dict
    })

@app.post("/settings/update/email")
async def update_email_settings(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    email_setting_id = form_data.get("email_setting_id")
    email_setting = db.query(EmailSettings).filter_by(id=email_setting_id).first()

    if email_setting:
        email_setting.name = form_data.get("name")
        email_setting.email = form_data.get("email")
        email_setting.provider = form_data.get("provider")
        email_setting.smtp_server = form_data.get("smtp_server")
        email_setting.smtp_port = form_data.get("smtp_port")
        email_setting.smtp_username = form_data.get("smtp_username")
        email_setting.smtp_password = form_data.get("smtp_password")
        email_setting.aws_access_key_id = form_data.get("aws_access_key_id")
        email_setting.aws_secret_access_key = form_data.get("aws_secret_access_key")
        email_setting.aws_region = form_data.get("aws_region")
        db.commit()
        return {"message": "Email settings updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Email settings not found")

@app.post("/settings/update/general")
async def update_general_settings(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    general_settings = db.query(Settings).filter_by(setting_type='general').first()

    settings_dict = general_settings.value if general_settings else {}
    for key, value in form_data.items():
        settings_dict[key] = value

    if general_settings:
        general_settings.value = settings_dict
    else:
        general_settings = Settings(setting_type='general', value=settings_dict)
        db.add(general_settings)

    db.commit()
    return {"message": "General settings updated successfully"}

# --- /email-templates ---
@app.get("/email-templates", response_class=HTMLResponse)
async def email_templates(request: Request, db: Session = Depends(db_session)):
    email_templates = db.query(EmailTemplate).all()
    return templates.TemplateResponse("email_templates.html", {
        "request": request,
        "email_templates": email_templates
    })

@app.post("/email-templates/create")
async def create_email_template(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    template_name = form_data.get("template_name")
    subject = form_data.get("subject")
    body_content = form_data.get("body_content")
    is_ai_customizable = form_data.get("is_ai_customizable") == "on"
    language = form_data.get("language")

    new_template = EmailTemplate(
        template_name=template_name,
        subject=subject,
        body_content=body_content,
        is_ai_customizable=is_ai_customizable,
        language=language
    )
    db.add(new_template)
    db.commit()
    return {"message": "Email template created successfully"}

@app.post("/email-templates/update")
async def update_email_template(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    template_id = form_data.get("template_id")
    template = db.query(EmailTemplate).filter_by(id=template_id).first()

    if template:
        template.template_name = form_data.get("template_name")
        template.subject = form_data.get("subject")
        template.body_content = form_data.get("body_content")
        template.is_ai_customizable = form_data.get("is_ai_customizable") == "on"
        template.language = form_data.get("language")
        db.commit()
        return {"message": "Email template updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Email template not found")

@app.post("/email-templates/delete")
async def delete_email_template(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    template_id = form_data.get("template_id")
    template = db.query(EmailTemplate).filter_by(id=template_id).first()

    if template:
        db.delete(template)
        db.commit()
        return {"message": "Email template deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Email template not found")

@app.post("/email-templates/ai-adjust", status_code=200)
async def adjust_email_template_with_ai(template_id: int = Form(...), ai_adjustment_prompt: str = Form(...), use_kb: bool = Form(...), session: Session = Depends(db_session)):
    template = session.query(EmailTemplate).filter(EmailTemplate.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Email template not found")

    if use_kb and not active_project_id:
        raise HTTPException(status_code=400, detail="Active project ID not set for using Knowledge Base")

    kb_info = get_knowledge_base_info(session, active_project_id) if use_kb else None
    prompt = f"Original Email:\nSubject: {template.subject}\nContent: {template.body_content}\n\n"
    prompt += f"AI Adjustment Prompt: {ai_adjustment_prompt}\n\n"
    
    if kb_info:
        prompt += f"Knowledge Base Info:\n{json.dumps(kb_info)}\n\n"

    prompt += "Please provide the adjusted email subject and content in the following JSON format:\n"
    prompt += '{"subject": "Adjusted Subject", "body_content": "Adjusted Body Content"}'

    try:
        response = openai_chat_completion([{"role": "system", "content": "You are a helpful assistant that adjusts email templates."}, {"role": "user", "content": prompt}], function_name="adjust_email_template_with_ai")
        if isinstance(response, dict) and "subject" in response and "body_content" in response:
            return {"updated_subject": response["subject"], "updated_body_content": response["body_content"]}
        else:
            logging.error(f"Unexpected response format from AI: {response}")
            raise HTTPException(status_code=500, detail="Unexpected response format from AI")
    except Exception as e:
        logging.error(f"Error adjusting email template with AI: {e}")
        raise HTTPException(status_code=500, detail=f"Error adjusting email template with AI: {e}")

# --- /projects-campaigns ---
@app.get("/projects-campaigns", response_class=HTMLResponse)
async def projects_campaigns(request: Request, db: Session = Depends(db_session)):
    projects = db.query(Project).all()
    campaigns = db.query(Campaign).all()
    return templates.TemplateResponse("projects_campaigns.html", {
        "request": request,
        "projects": projects,
        "campaigns": campaigns
    })

@app.post("/projects/create")
async def create_project(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    project_name = form_data.get("project_name")
    new_project = Project(project_name=project_name)
    db.add(new_project)
    db.commit()
    return {"message": "Project created successfully"}

@app.post("/campaigns/create")
async def create_campaign(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    campaign_name = form_data.get("campaign_name")
    project_id = form_data.get("project_id")
    new_campaign = Campaign(campaign_name=campaign_name, project_id=project_id)
    db.add(new_campaign)
    db.commit()
    return {"message": "Campaign created successfully"}

@app.post("/projects/update")
async def update_project(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    project_id = form_data.get("project_id")
    project = db.query(Project).filter_by(id=project_id).first()
    if project:
        project.project_name = form_data.get("project_name")
        db.commit()
        return {"message": "Project updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Project not found")

@app.post("/campaigns/update")
async def update_campaign(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    campaign_id = form_data.get("campaign_id")
    campaign = db.query(Campaign).filter_by(id=campaign_id).first()
    if campaign:
        campaign.campaign_name = form_data.get("campaign_name")
        campaign.project_id = form_data.get("project_id")
        db.commit()
        return {"message": "Campaign updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Campaign not found")

@app.post("/projects/delete")
async def delete_project(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    project_id = form_data.get("project_id")
    project = db.query(Project).filter_by(id=project_id).first()
    if project:
        db.delete(project)
        db.commit()
        return {"message": "Project deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Project not found")

@app.post("/campaigns/delete")
async def delete_campaign(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    campaign_id = form_data.get("campaign_id")
    campaign = db.query(Campaign).filter_by(id=campaign_id).first()
    if campaign:
        db.delete(campaign)
        db.commit()
        return {"message": "Campaign deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Campaign not found")

# --- /knowledge-base ---
@app.get("/knowledge-base", response_class=HTMLResponse)
async def knowledge_base(request: Request, db: Session = Depends(db_session)):
    knowledge_bases = db.query(KnowledgeBase).all()
    return templates.TemplateResponse("knowledge_base.html", {
        "request": request,
        "knowledge_bases": knowledge_bases
    })

@app.post("/knowledge-base/create")
async def create_knowledge_base(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    project_id = form_data.get("project_id")
    new_kb = KnowledgeBase(
        project_id=project_id,
        kb_name=form_data.get("kb_name"),
        kb_bio=form_data.get("kb_bio"),
        kb_values=form_data.get("kb_values"),
        contact_name=form_data.get("contact_name"),
        contact_role=form_data.get("contact_role"),
        contact_email=form_data.get("contact_email"),
        company_description=form_data.get("company_description"),
        company_mission=form_data.get("company_mission"),
        company_target_market=form_data.get("company_target_market"),
        company_other=form_data.get("company_other"),
        product_name=form_data.get("product_name"),
        product_description=form_data.get("product_description"),
        product_target_customer=form_data.get("product_target_customer"),
        product_other=form_data.get("product_other"),
        other_context=form_data.get("other_context"),
        example_email=form_data.get("example_email")
    )
    db.add(new_kb)
    db.commit()
    return {"message": "Knowledge base created successfully"}

@app.post("/knowledge-base/update")
async def update_knowledge_base(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    kb_id = form_data.get("kb_id")
    kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
    if kb:
        kb.project_id = form_data.get("project_id")
        kb.kb_name = form_data.get("kb_name")
        kb.kb_bio = form_data.get("kb_bio")
        kb.kb_values = form_data.get("kb_values")
        kb.contact_name = form_data.get("contact_name")
        kb.contact_role = form_data.get("contact_role")
        kb.contact_email = form_data.get("contact_email")
        kb.company_description = form_data.get("company_description")
        kb.company_mission = form_data.get("company_mission")
        kb.company_target_market = form_data.get("company_target_market")
        kb.company_other = form_data.get("company_other")
        kb.product_name = form_data.get("product_name")
        kb.product_description = form_data.get("product_description")
        kb.product_target_customer = form_data.get("product_target_customer")
        kb.product_other = form_data.get("product_other")
        kb.other_context = form_data.get("other_context")
        kb.example_email = form_data.get("example_email")
        db.commit()
        return {"message": "Knowledge base updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

@app.post("/knowledge-base/delete")
async def delete_knowledge_base(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    kb_id = form_data.get("kb_id")
    kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
    if kb:
        db.delete(kb)
        db.commit()
        return {"message": "Knowledge base deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

# --- /search-terms ---
@app.get("/search-terms", response_class=HTMLResponse)
async def search_terms(request: Request, db: Session = Depends(db_session)):
    search_terms = db.query(SearchTerm).all()
    return templates.TemplateResponse("search_terms.html", {
        "request": request,
        "search_terms": search_terms
    })

@app.post("/search-terms/create")
async def create_search_term(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    term = form_data.get("term")
    campaign_id = form_data.get("campaign_id")
    new_term = SearchTerm(term=term, campaign_id=campaign_id)
    db.add(new_term)
    db.commit()
    return {"message": "Search term created successfully"}

@app.post("/search-terms/update")
async def update_search_term(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    term_id = form_data.get("term_id")
    term = db.query(SearchTerm).filter_by(id=term_id).first()
    if term:
        term.term = form_data.get("term")
        term.campaign_id = form_data.get("campaign_id")
        db.commit()
        return {"message": "Search term updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Search term not found")

@app.post("/search-terms/delete")
async def delete_search_term(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    term_id = form_data.get("term_id")
    term = db.query(SearchTerm).filter_by(id=term_id).first()
    if term:
        db.delete(term)
        db.commit()
        return {"message": "Search term deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Search term not found")

# --- /leads ---
@app.get("/leads", response_class=HTMLResponse)
async def leads(request: Request, db: Session = Depends(db_session)):
    leads = db.query(Lead).all()
    return templates.TemplateResponse("leads.html", {
        "request": request,
        "leads": leads
    })

@app.post("/leads/create")
async def create_lead(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    email = form_data.get("email")
    new_lead = Lead(email=email)
    db.add(new_lead)
    db.commit()
    return {"message": "Lead created successfully"}

@app.post("/leads/update")
async def update_lead(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    lead_id = form_data.get("lead_id")
    lead = db.query(Lead).filter_by(id=lead_id).first()
    if lead:
        lead.email = form_data.get("email")
        db.commit()
        return {"message": "Lead updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Lead not found")

@app.post("/leads/delete")
async def delete_lead(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    lead_id = form_data.get("lead_id")
    lead = db.query(Lead).filter_by(id=lead_id).first()
    if lead:
        db.delete(lead)
        db.commit()
        return {"message": "Lead deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Lead not found")

# --- /campaign-logs ---
@app.get("/campaign-logs", response_class=HTMLResponse)
async def campaign_logs(request: Request, db: Session = Depends(db_session)):
    campaign_logs = db.query(EmailCampaign).all()
    return templates.TemplateResponse("campaign_logs.html", {
        "request": request,
        "campaign_logs": campaign_logs
    })

# --- /sent-campaigns ---
@app.get("/sent-campaigns", response_class=HTMLResponse)
async def sent_campaigns(request: Request, db: Session = Depends(db_session)):
    sent_campaigns = fetch_sent_email_campaigns(db)
    return templates.TemplateResponse("sent_campaigns.html", {
        "request": request,
        "sent_campaigns": sent_campaigns.to_dict(orient="records")
    })

# --- /automation-control ---
@app.get("/automation-control", response_class=HTMLResponse)
async def automation_control(request: Request, db: Session = Depends(db_session)):
    campaigns = db.query(Campaign).all()
    return templates.TemplateResponse("automation_control.html", {
        "request": request,
        "campaigns": campaigns
    })

@app.post("/automation/start")
async def start_automation(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    campaign_id = form_data.get("campaign_id")
    search_terms = form_data.get("search_terms")
    num_results = int(form_data.get("num_results"))
    optimize_english = form_data.get("optimize_english") == "on"
    optimize_spanish = form_data.get("optimize_spanish") == "on"
    shuffle_keywords = form_data.get("shuffle_keywords") == "on"
    language = form_data.get("language")

    try:
        automation_log = AutomationLog(
            campaign_id=campaign_id,
            status='running',
            start_time=datetime.utcnow(),
            logs=[],
            search_term_id=None,
            leads_gathered=0,
            emails_sent=0
        )
        db.add(automation_log)
        db.commit()

        process = subprocess.Popen([
            'python', 'automated_search.py',
            str(automation_log.id)
        ])

        return JSONResponse(status_code=200, content={
            "message": "Automation started successfully",
            "automation_log_id": automation_log.id,
            "pid": process.pid
        })
    except Exception as e:
        logging.error(f"Failed to start automation: {e}")
        if 'automation_log' in locals():
            db.delete(automation_log)
            db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to start automation: {e}")

@app.get("/automation/status/{automation_log_id}")
async def automation_status(automation_log_id: int, db: Session = Depends(db_session)):
    status = get_automation_status(automation_log_id)
    return status

@app.post("/automation/stop/{automation_log_id}")
async def stop_automation(automation_log_id: int, db: Session = Depends(db_session)):
    try:
        with db_session() as session:
            log = session.query(AutomationLog).get(automation_log_id)
            if log and log.status == 'running':
                log.status = 'stopped'
                log.end_time = datetime.utcnow()
                session.commit()
                return {"message": "Automation stopped successfully"}
            else:
                return {"message": "Automation is not running or does not exist"}
    except Exception as e:
        logging.error(f"Failed to stop automation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop automation: {e}")

# --- /bulk-send ---
@app.get("/bulk-send", response_class=HTMLResponse)
async def bulk_send(request: Request, db: Session = Depends(db_session)):
    email_templates = db.query(EmailTemplate).all()
    return templates.TemplateResponse("bulk_send.html", {
        "request": request,
        "email_templates": email_templates
    })

@app.post("/bulk-send/send")
async def send_bulk_emails(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    template_id = form_data.get("template_id")
    from_email = form_data.get("from_email")
    reply_to = form_data.get("reply_to")
    send_option = form_data.get("send_option")
    specific_email = form_data.get("specific_email")
    selected_terms = form_data.getlist("selected_terms")
    exclude_previously_contacted = form_data.get("exclude_previously_contacted") == "on"

    leads = fetch_leads(db, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted)
    logs, sent_count = bulk_send_emails(db, template_id, from_email, reply_to, leads)

    return {
        "message": f"Bulk send completed. {sent_count} emails sent.",
        "logs": logs
    }

# --- /manual-search ---
@app.get("/manual-search", response_class=HTMLResponse)
async def manual_search(request: Request, db: Session = Depends(db_session)):
    campaigns = db.query(Campaign).all()
    return templates.TemplateResponse("manual_search.html", {
        "request": request,
        "campaigns": campaigns
    })

@app.post("/manual-search/run")
async def run_manual_search(request: Request, db: Session = Depends(db_session)):
    form_data = await request.form()
    campaign_id = form_data.get("campaign_id")
    search_terms = form_data.getlist("search_terms")
    num_results = int(form_data.get("num_results"))
    optimize_english = form_data.get("optimize_english") == "on"
    optimize_spanish = form_data.get("optimize_spanish") == "on"
    shuffle_keywords = form_data.get("shuffle_keywords") == "on"
    language = form_data.get("language")

    email_setting = fetch_email_settings(db)[0] if fetch_email_settings(db) else None
    from_email = email_setting['email'] if email_setting else None
    reply_to = from_email
    email_template = db.query(EmailTemplate).first()

    res = manual_search(db, search_terms, num_results, optimize_english, optimize_spanish, shuffle_keywords, True, language, True, None, from_email, reply_to, f"{email_template.id}: {email_template.template_name}" if email_template else None)
    return {"new_leads": len(res['results']), "terms_used": search_terms}

# --- / (Root) ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Helper Functions ---
def get_active_campaign_id():
    # Implement logic to determine the active campaign ID
    return 1  # Placeholder
def manual_search(session, terms, num_results, optimize_english, optimize_spanish, shuffle_keywords, enable_email_sending, language, ignore_previously_fetched, email_template, from_email, reply_to, email_setting_option):
    """
    Performs a manual search using the provided terms, fetches and stores lead data, and sends emails if enabled.

    Args:
        session: Database session.
        terms: List of search terms.
        num_results: Number of search results to fetch per term.
        optimize_english: Flag to optimize search terms for English.
        optimize_spanish: Flag to optimize search terms for Spanish.
        shuffle_keywords: Flag to shuffle keywords in optimized terms.
        enable_email_sending: Flag to enable sending emails to leads.
        language: Language for search optimization.
        ignore_previously_fetched: Flag to ignore previously fetched leads.
        email_template: Email template ID or name.
        from_email: Sender's email address.
        reply_to: Reply-to email address.
        email_setting_option: Option for email settings.

    Returns:
        A dictionary containing the search results and the terms used.
    """
    results = []
    domains_processed = set()
    terms_used = set()

    def add_or_get_search_term(session, term, campaign_id):
        """Adds a search term to the database if it doesn't exist, or retrieves it if it does."""
        search_term = session.query(SearchTerm).filter_by(term=term, campaign_id=campaign_id).first()
        if not search_term:
            search_term = SearchTerm(term=term, campaign_id=campaign_id)
            session.add(search_term)
            session.commit()
        return search_term.id

    def generate_optimized_search_terms(session, base_terms, kb_info):
        """Generates optimized search terms using AI."""
        prompt = f"Optimize and expand these search terms for lead generation:\n{', '.join(base_terms)}\n\nConsider:\n1. Relevance to business and target market\n2. Potential for high-quality leads\n3. Variations and related terms\n4. Industry-specific jargon\n\nRespond with a JSON array of optimized terms."
        response = openai_chat_completion([{"role": "system", "content": "You're an AI specializing in optimizing search terms for lead generation. Be concise and effective."}, {"role": "user", "content": prompt}], function_name="generate_optimized_search_terms")
        return response.get('optimized_terms', base_terms) if isinstance(response, dict) else base_terms

    def fetch_and_store_lead_data(session, url, search_term_id):
        """Fetches lead data from a URL and stores it in the database."""
        try:
            # Placeholder for actual lead data fetching logic
            # This would involve making a request to the URL and parsing the response
            # For now, we'll simulate fetching data
            lead_data = {
                "website": url,
                "email": "test@example.com",  # Replace with actual email extraction
                "first_name": "John",  # Replace with actual name extraction
                "last_name": "Doe",  # Replace with actual name extraction
                "company": "Example Corp",  # Replace with actual company extraction
                "job_title": "CEO",  # Replace with actual job title extraction
                "source": "Manual Search",
                "search_term_id": search_term_id
            }

            lead = Lead(**lead_data)
            session.add(lead)
            session.commit()
            return lead
        except Exception as e:
            logging.error(f"Error fetching or storing lead data for {url}: {e}")
            return None

    def send_email_for_lead(session, lead_data, email_template, from_email, reply_to):
        """Sends an email to a lead using AWS SES."""
        try:
            logging.info(f"Sending email to {lead_data.email} from {from_email} with template {email_template.subject}")
            message_id = send_email_ses(
                subject=email_template.subject,
                body_text=email_template.body,
                sender_email=from_email,
                recipient_email=lead_data.email,
                reply_to_email=reply_to
            )
            if message_id:
                email_status = 'sent' if message_id else 'failed'
                log_message = f"Email sent to {lead_data.email}" if message_id else f"Failed to send email to {lead_data.email}"
                logging.info(log_message) if message_id else logging.error(log_message)
                save_email_campaign(session, lead_data.id, email_template.id, email_status, datetime.utcnow(), email_template.subject, message_id, email_template.body)
        except Exception as e:
            logging.error(f"Error sending email to {lead_data.email}: {e}")
            save_email_campaign(session, lead_data.id, email_template.id, 'error', datetime.utcnow(), email_template.subject, None, email_template.body)

    for term in terms:
        campaign_id = get_active_campaign_id()
        search_term_id = add_or_get_search_term(session, term, campaign_id)
        terms_used.add(term)

        if optimize_english or optimize_spanish:
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            optimized_terms = generate_optimized_search_terms(session, [term], kb_info)
            
            if shuffle_keywords:
                # Placeholder for keyword shuffling logic
                optimized_terms = [t + " (shuffled)" for t in optimized_terms]
        else:
            optimized_terms = [term]

        for search_term in optimized_terms:
            try:
                urls = list(google_search(search_term, num_results=num_results))
                for url in urls:
                    domain = urlparse(url).netloc
                    if domain not in domains_processed:
                        if ignore_previously_fetched:
                            existing_lead = session.query(Lead).filter_by(website=url).first()
                            if existing_lead:
                                continue
                        
                        results.append(url)
                        domains_processed.add(domain)

                        if enable_email_sending:
                            lead_data = fetch_and_store_lead_data(session, url, search_term_id)
                            if lead_data and email_setting_option:
                                send_email_for_lead(session, lead_data, email_template, from_email, reply_to)
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching URL {url}: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")

    return {'results': results, 'terms_used': list(terms_used)}

def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"Optimize and expand these search terms for lead generation:\n{', '.join(base_terms)}\n\nConsider:\n1. Relevance to business and target market\n2. Potential for high-quality leads\n3. Variations and related terms\n4. Industry-specific jargon\n\nRespond with a JSON array of optimized terms."
    response = openai_chat_completion([{"role": "system", "content": "You're an AI specializing in optimizing search terms for lead generation. Be concise and effective."}, {"role": "user", "content": prompt}], function_name="generate_optimized_search_terms")
    return response.get('optimized_terms', base_terms) if isinstance(response, dict) else base_terms

def bulk_send_emails(session, template_id, from_email, reply_to, leads):
    logs = []
    sent_count = 0
    email_template = session.query(EmailTemplate).filter_by(id=template_id).first()

    if not email_template:
        logs.append("Error: Email template not found.")
        return logs, sent_count

    for lead in leads:
        try:
            email = lead["Email"]
            lead_id = lead["ID"]
            
            if email_template.is_ai_customizable:
                customized_subject, customized_content = customize_email_with_ai(session, email_template, lead_id)
            else:
                customized_subject = email_template.subject
                customized_content = email_template.body_content

            send_email_ses(session, from_email, email, customized_subject, customized_content, reply_to=reply_to)
            save_email_campaign(session, lead_id, template_id, 'sent', datetime.utcnow(), customized_subject, None, customized_content)
            sent_count += 1
            logs.append(f"Email sent to {email} at {datetime.utcnow()}")
        except Exception as e:
            logs.append(f"Error sending email to {email}: {str(e)}")

    return logs, sent_count

def customize_email_with_ai(session, email_template, lead_id):
    lead = session.query(Lead).filter_by(id=lead_id).first()
    if not lead:
        raise ValueError(f"Lead with ID {lead_id} not found.")

    kb_info = get_knowledge_base_info(session, active_project_id)
    prompt = f"Original Email:\nSubject: {email_template.subject}\nContent: {email_template.body_content}\n\n"
    prompt += f"Lead Info:\nEmail: {lead.email}\n"
    if lead.first_name:
        prompt += f"First Name: {lead.first_name}\n"
    if lead.last_name:
        prompt += f"Last Name: {lead.last_name}\n"
    if lead.company:
        prompt += f"Company: {lead.company}\n"
    if lead.job_title:
        prompt += f"Job Title: {lead.job_title}\n"

    if kb_info:
        prompt += f"Knowledge Base Info:\n{json.dumps(kb_info)}\n\n"

    prompt += "Please provide the AI-customized email subject and content in the following JSON format:\n"
    prompt += '{"subject": "Customized Subject", "body_content": "Customized Body Content"}'

    try:
        response = openai_chat_completion([
            {"role": "system", "content": "You are a helpful assistant that customizes email templates based on lead information and knowledge base."},
            {"role": "user", "content": prompt}
        ], function_name="customize_email_with_ai", lead_id=lead_id, email_campaign_id=None)

        if isinstance(response, dict) and "subject" in response and "body_content" in response:
            return response["subject"], response["body_content"]
        else:
            logging.error(f"Unexpected response format from AI: {response}")
            return email_template.subject, email_template.body_content
    except Exception as e:
        logging.error(f"Error customizing email with AI: {e}")
        return email_template.subject, email_template.body_content
def wrap_email_body(body_content):
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
</style>
</head>
<body>
    {body_content}
</body>
</html>
"""

# --- Main Function ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)