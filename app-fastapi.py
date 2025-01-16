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
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, urlencode
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import signal
import subprocess
from fastapi import FastAPI, HTTPException, Depends, Request, Form, Response, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

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

engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(bind=engine)
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    project = relationship("Project", back_populates="knowledge_base")

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']}

class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    phone = Column(Text)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text)
    job_title = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead")
    email_campaigns = relationship("EmailCampaign", back_populates="lead")

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    template_name = Column(Text)
    subject = Column(Text)
    body_content = Column(Text)
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
    total_results = Column(BigInteger)
    valid_leads = Column(BigInteger)
    irrelevant_leads = Column(BigInteger)
    blogs_found = Column(BigInteger)
    directories_found = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_term = relationship("SearchTerm", back_populates="effectiveness")

class SearchTermGroup(Base):
    __tablename__ = 'search_term_groups'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    email_template = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_terms = relationship("SearchTerm", back_populates="group")

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    id = Column(BigInteger, primary_key=True)
    group_id = Column(BigInteger, ForeignKey('search_term_groups.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    term = Column(Text)
    category = Column(Text)
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
    id = Column(BigInteger, primary_key=True)
    function_name = Column(Text)
    prompt = Column(Text)
    response = Column(Text)
    model_used = Column(Text)
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
    leads_gathered = Column(BigInteger)
    emails_sent = Column(BigInteger)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    status = Column(Text)
    logs = Column(JSON)
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

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/settings", response_model=Dict[str, Any])
async def get_settings(session: Session = Depends(db_session)):
    general_settings = session.query(Settings).filter_by(setting_type='general').first()
    email_settings = session.query(EmailSettings).all()

    return {
        "general_settings": general_settings.value if general_settings else {},
        "email_settings": [{"id": setting.id, "name": setting.name, "email": setting.email, "provider": setting.provider, "smtp_server": setting.smtp_server, "smtp_port": setting.smtp_port, "aws_region": setting.aws_region} for setting in email_settings] if email_settings else [],
    }

@app.post("/settings", status_code=200)
async def update_settings(settings_update: SettingsUpdate, session: Session = Depends(db_session)):
    general_settings = session.query(Settings).filter_by(setting_type='general').first() or Settings(name='General Settings', setting_type='general', value={})
    general_settings.value = settings_update.dict()
    session.add(general_settings)
    session.commit()
    return {"message": "General settings saved successfully!"}

@app.post("/email-settings", status_code=201)
async def create_email_setting(email_setting: EmailSettingCreate, session: Session = Depends(db_session)):
    try:
        new_setting = EmailSettings(**email_setting.dict())
        session.add(new_setting)
        session.commit()
        return {"message": "Email setting saved successfully!"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/email-settings", status_code=200)
async def update_email_setting(email_setting_update: EmailSettingUpdate, session: Session = Depends(db_session)):
    try:
        setting = session.get(EmailSettings, email_setting_update.id)
        if not setting:
            raise HTTPException(status_code=404, detail="Email setting not found")
        for key, value in email_setting_update.dict(exclude={'id'}).items():
            setattr(setting, key, value)
        session.commit()
        return {"message": "Email setting updated successfully!"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/email-settings/{setting_id}", status_code=200)
async def delete_email_setting(setting_id: int, session: Session = Depends(db_session)):
    setting = session.get(EmailSettings, setting_id)
    if setting:
        session.delete(setting)
        session.commit()
        return {"message": f"Deleted {setting.name}"}
    else:
        raise HTTPException(status_code=404, detail="Email setting not found")

@app.post("/test-email", status_code=200)
async def test_email(test_email: str = Form(...), email_id: int = Form(...), session: Session = Depends(db_session)):
    try:
        setting = session.get(EmailSettings, email_id)
        if not setting:
            raise HTTPException(status_code=404, detail="Email setting not found")
        response, _ = send_email_ses(
            session,
            setting.email,
            test_email,
            "Test Email from AutoclientAI",
            "<p>This is a test email from your AutoclientAI email settings.</p>",
            reply_to=setting.email
        )
        if response:
            return {"message": f"Test email sent successfully to {test_email}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send test email")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending test email: {str(e)}")

@app.post("/manual-search", status_code=200)
async def perform_manual_search(search_input: SearchTermsInput, session: Session = Depends(db_session)):
    if not search_input.terms:
        raise HTTPException(status_code=400, detail="Enter at least one search term.")

    if search_input.enable_email_sending:
        if not search_input.email_template:
            raise HTTPException(status_code=400, detail="No email templates available. Please create a template first.")
        if not search_input.email_setting_option:
            raise HTTPException(status_code=400, detail="No email settings available. Please add email settings first.")

        email_setting = session.query(EmailSettings).filter(EmailSettings.name == search_input.email_setting_option.split(" (")[0].strip()).first()
        if not email_setting:
            raise HTTPException(status_code=404, detail="Email setting not found.")
        from_email = email_setting.email
    else:
        from_email = None

    email_template = None
    if search_input.email_template:
        template = session.query(EmailTemplate).filter(EmailTemplate.template_name == search_input.email_template.split(":")[1].strip()).first()
        if template:
            email_template = f"{template.id}: {template.template_name}"
        else:
            raise HTTPException(status_code=404, detail="Email template not found.")

    results = manual_search(
        session, 
        search_input.terms, 
        search_input.num_results,
        search_input.ignore_previously_fetched if search_input.ignore_previously_fetched is not None else True,
        search_input.optimize_english, 
        search_input.optimize_spanish,
        search_input.shuffle_keywords,
        search_input.language, 
        search_input.enable_email_sending,
        None,
        from_email,
        search_input.reply_to,
        email_template
    )
    return {"total_leads": results['total_leads'], "results": results['results']}

@app.get("/leads", response_model=List[Dict])
async def list_leads(session: Session = Depends(db_session)):
    leads_df = fetch_leads_with_sources(session)
    return leads_df.to_dict(orient='records') if not leads_df.empty else []

@app.put("/leads/{lead_id}", status_code=200)
async def update_lead_endpoint(lead_id: int, lead_data: LeadUpdate, session: Session = Depends(db_session)):
    updated_data = {k: v for k, v in lead_data.dict().items() if k != 'id' and v is not None}
    if update_lead(session, lead_id, updated_data):
        return {"message": f"Updated lead: {lead_data.email}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to update lead with id {lead_id}")

@app.delete("/leads/{lead_id}", status_code=200)
async def delete_lead_endpoint(lead_id: int, session: Session = Depends(db_session)):
    if delete_lead_and_sources(session, lead_id):
        return {"message": f"Deleted lead with id {lead_id}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to delete lead with id {lead_id}")

@app.get("/search-terms", response_model=List[Dict])
async def get_search_terms(session: Session = Depends(db_session)):
    search_terms_df = fetch_search_terms_with_lead_count(session)
    return search_terms_df.to_dict(orient='records') if not search_terms_df.empty else []

@app.post("/search-terms", status_code=201)
async def add_new_search_term_endpoint(search_term: SearchTermCreate, session: Session = Depends(db_session)):
    try:
        add_new_search_term(session, search_term.term, search_term.campaign_id, search_term.group_for_new_term)
        return {"message": f"Added: {search_term.term}"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error adding search term: {str(e)}")

@app.get("/search-term-groups", response_model=List[Dict])
async def get_search_term_groups(session: Session = Depends(db_session)):
    groups = session.query(SearchTermGroup).all()
    return [{"id": group.id, "name": group.name} for group in groups]

@app.post("/search-term-groups", status_code=201)
async def create_search_term_group_endpoint(search_term_group: SearchTermGroupCreate, session: Session = Depends(db_session)):
    try:
        create_search_term_group(session, search_term_group.name)
        return {"message": f"Created new group: {search_term_group.name}"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating search term group: {str(e)}")

@app.delete("/search-term-groups/{group_id}", status_code=200)
async def delete_search_term_group_endpoint(group_id: int, session: Session = Depends(db_session)):
    try:
        delete_search_term_group(session, group_id)
        return {"message": f"Deleted search term group with id {group_id}"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting search term group: {str(e)}")

@app.put("/search-term-groups", status_code=200)
async def update_search_term_group_endpoint(update_data: SearchTermGroupUpdate, session: Session = Depends(db_session)):
    try:
        update_search_term_group(session, update_data.group_id, update_data.updated_terms)
        return {"message": "Group updated successfully"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating search term group: {str(e)}")

@app.post("/search-terms/ai-group", response_model=Dict)
async def ai_grouping_search_terms(search_terms: SearchTermsGrouping, session: Session = Depends(db_session)):
    try:
        ungrouped_terms = session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
        if search_terms.grouped_terms:
            update_search_term_groups(session, {group.group_name: group.terms for group in search_terms.grouped_terms})
        if search_terms.ungrouped_terms:
            ungrouped_terms_from_request = [term.term for term in session.query(SearchTerm).filter(SearchTerm.term.in_(search_terms.ungrouped_terms))]
            if ungrouped_terms_from_request:
                grouped_terms = ai_group_search_terms(session, ungrouped_terms_from_request)
                update_search_term_groups(session, grouped_terms)
        return {"message": "Search terms have been grouped successfully!"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error grouping search terms: {str(e)}")

@app.get("/email-templates", response_model=List[Dict])
async def get_email_templates(session: Session = Depends(db_session)):
    templates = session.query(EmailTemplate).all()
    return [{"id": t.id, "template_name": t.template_name, "subject": t.subject, "body_content": t.body_content, "is_ai_customizable": t.is_ai_customizable, "language": t.language} for t in templates]

@app.post("/email-templates", status_code=201)
async def create_email_template(template_data: EmailTemplateCreate, session: Session = Depends(db_session)):
    try:
        new_template = EmailTemplate(**template_data.dict(), campaign_id=get_active_campaign_id())
        session.add(new_template)
        session.commit()
        return {"message": "Email template saved successfully!"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/email-templates", status_code=200)
async def update_email_template(template_data: EmailTemplateUpdate, session: Session = Depends(db_session)):
    try:
        template = session.get(EmailTemplate, template_data.id)
        if not template:
            raise HTTPException(status_code=404, detail="Email template not found")
        for key, value in template_data.dict(exclude={'id'}).items():
            setattr(template, key, value)
        session.commit()
        return {"message": "Email template updated successfully!"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/email-templates/{template_id}", status_code=200)
async def delete_email_template(template_id: int, session: Session = Depends(db_session)):
    try:
        template = session.get(EmailTemplate, template_id)
        if template:
            session.delete(template)
            session.commit()
            return {"message": "Email template deleted successfully!"}
        else:
            raise HTTPException(status_code=404, detail="Email template not found")
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk-send", status_code=200)
async def bulk_send_emails_endpoint(bulk_send_input: BulkSendInput, session: Session = Depends(db_session)):
    try:
        leads = fetch_leads(
            session,
            bulk_send_input.template_id,
            bulk_send_input.send_option,
            bulk_send_input.specific_email,
            bulk_send_input.selected_terms,
            bulk_send_input.exclude_previously_contacted
        )
        if not leads:
            raise HTTPException(status_code=404, detail="No leads found for the specified criteria")

        template = session.query(EmailTemplate).filter_by(id=bulk_send_input.template_id).first()
        if not template:
            raise HTTPException(status_code=404, detail="Email template not found")

        from_email = bulk_send_input.from_email
        reply_to = bulk_send_input.reply_to

        logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, leads)
        return {"message": f"Emails sent: {sent_count}", "logs": logs}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error in bulk email sending: {str(e)}")

@app.get("/email-logs", response_model=List[Dict])
async def get_email_logs(session: Session = Depends(db_session)):
    try:
        email_logs_df = fetch_all_email_logs(session)
        return email_logs_df.to_dict(orient='records') if not email_logs_df.empty else []
    except Exception as e:
        logging.error(f"Error fetching email logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching email logs")

@app.get("/automation-status/{automation_log_id}", response_model=Dict)
async def get_automation_status_endpoint(automation_log_id: int):
    try:
        status = get_automation_status(automation_log_id)
        return status
    except Exception as e:
        logging.error(f"Error getting automation status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting automation status")

@app.post("/quick-scan", status_code=200)
async def perform_quick_scan_endpoint(session: Session = Depends(db_session)):
    try:
        result = perform_quick_scan(session)
        return result
    except Exception as e:
        logging.error(f"Error performing quick scan: {str(e)}")
        raise HTTPException(status_code=500, detail="Error performing quick scan")

@app.post("/ai-response", response_model=Dict)
async def get_ai_response_endpoint(prompt: str):
    try:
        response = get_ai_response(prompt)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error getting AI response: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting AI response")

def send_email_ses(session, from_email, to_email, subject, body, reply_to=None):
    # Placeholder for sending email via SES
    # Implement actual email sending logic here
    return {"MessageId": str(uuid.uuid4())}, None

def manual_search(session, terms, num_results, ignore_previously_fetched, optimize_english, optimize_spanish, shuffle_keywords, language, enable_email_sending, email_setting, from_email, reply_to, email_template):
    # Placeholder for manual search logic
    # Implement actual search logic here
    return {"total_leads": 0, "results": []}

def fetch_leads_with_sources(session):
    # Placeholder for fetching leads with sources
    # Implement actual fetching logic here
    return pd.DataFrame()

def fetch_search_terms_with_lead_count(session):
    # Placeholder for fetching search terms with lead count
    # Implement actual fetching logic here
    return pd.DataFrame()

def update_search_term_group(session, group_id, updated_terms):
    # Placeholder for updating search term group
    # Implement actual update logic here
    pass

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    # Placeholder for adding a new search term
    # Implement actual addition logic here
    pass

def create_search_term_group(session, group_name):
    # Placeholder for creating a search term group
    # Implement actual creation logic here
    pass

def delete_search_term_group(session, group_id):
    # Placeholder for deleting a search term group
    # Implement actual deletion logic here
    pass

def ai_group_search_terms(session, ungrouped_terms):
    # Placeholder for AI grouping of search terms
    # Implement actual AI logic here
    return {}

def update_search_term_groups(session, grouped_terms):
    # Placeholder for updating search term groups
    # Implement actual update logic here
    pass

def fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted):
    # Placeholder for fetching leads
    # Implement actual fetching logic here
    return []

def bulk_send_emails(session, template_id, from_email, reply_to, leads):
    # Placeholder for bulk sending emails
    # Implement actual sending logic here
    return [], 0

def fetch_all_email_logs(session):
    # Placeholder for fetching all email logs
    # Implement actual fetching logic here
    return pd.DataFrame()

def get_automation_status(automation_log_id):
    # Placeholder for getting automation status
    # Implement actual status retrieval logic here
    return {}

def perform_quick_scan(session):
    # Placeholder for performing a quick scan
    # Implement actual scan logic here
    return {}

def get_ai_response(prompt):
    # Placeholder for getting AI response
    # Implement actual AI response logic here
    return "AI response"

def save_email_campaign(session, lead_id, template_id, status, sent_at, subject, message_id, content):
    # Placeholder for saving email campaign details
    # Implement actual save logic here
    pass

def get_knowledge_base_info(session, project_id):
    # Placeholder for fetching knowledge base information
    # Implement actual fetching logic here
    return {}

def openai_chat_completion(messages, function_name=None, lead_id=None, email_campaign_id=None):
    # Placeholder for OpenAI chat completion
    # Implement actual OpenAI API call here
    return {"subject": "Sample Subject", "body_content": "Sample Body Content"}

def get_active_campaign_id():
    # Placeholder for getting active campaign ID
    # Implement actual logic here
    return 1

def get_active_project_id():
    # Placeholder for getting active project ID
    # Implement actual logic here
    return 1

def log_ai_request(session, function_name, prompt, response, lead_id=None, email_campaign_id=None, model_used=None):
    # Placeholder for logging AI requests
    # Implement actual logging logic here
    pass

def update_log(log_container, log_message):
    # Placeholder for updating logs
    # Implement actual log update logic here
    pass

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

def is_valid_email(email):
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def remove_invalid_leads(session):
    # Placeholder for removing invalid leads
    # Implement actual removal logic here
    return 0

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

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
