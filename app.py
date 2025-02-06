from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import sys
import random
from bs4 import BeautifulSoup
import re
import asyncio
import time
import requests
import json
from datetime import datetime
from googlesearch import search as google_search
from fake_useragent import UserAgent
from email_validator import validate_email, EmailNotValidError
from urllib.parse import urlparse
import aiohttp
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, func, BigInteger, Text, JSON, Boolean, case, text, distinct, cast, Float, or_, and_, select, join
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import pandas as pd
import plotly.express as px
from contextlib import contextmanager
import boto3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
import uuid
from threading import Thread
from queue import Queue
from urllib.parse import urlencode
from sqlalchemy import Float

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI Setup
app = FastAPI(title="Email Crawler", 
             description="Search and extract emails from web pages",
             version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()



# Add new Task Queue model
class TaskQueue(Base):
    __tablename__ = 'task_queue'
    id = Column(BigInteger, primary_key=True)
    task_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    search_term = Column(Text)
    parameters = Column(JSON)
    status = Column(Text, default='pending')  # pending, running, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(Text)  # Could tie to user sessions if needed

# Background worker setup
task_queue = Queue()

def update_progress(processed, total):
    """Update progress for the current task"""
    if total > 0:
        progress = (processed / total) * 100
        logger.info(f"Progress: {progress:.2f}% ({processed}/{total})")

async def background_worker():
    """Asynchronous background worker for processing tasks"""
    while True:
        try:
            with db_session() as session:
                task = session.query(TaskQueue).filter(
                    TaskQueue.status == 'pending'
                ).order_by(TaskQueue.created_at.asc()).first()
                
                if task:
                    logger.info(f"Processing task {task.task_id}")
                    task.status = 'running'
                    session.commit()
                    
                    try:
                        await process_search_task(task)
                        task.status = 'completed'
                        logger.info(f"Task {task.task_id} completed successfully")
                    except Exception as e:
                        task.status = 'failed'
                        logger.error(f"Task {task.task_id} failed: {str(e)}")
                    finally:
                        session.commit()
                else:
                    await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Background worker error: {str(e)}")
            await asyncio.sleep(10)

# Start background worker on startup
@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        # Start background worker
        asyncio.create_task(background_worker())
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

async def process_search_task(task):
    """Process search task and store results in Supabase"""
    with db_session() as session:
        try:
            # Create automation log entry
            automation_log = AutomationLog(
                campaign_id=1,  # Default campaign
                search_term_id=None,
                status='running',
                logs=[]
            )
            session.add(automation_log)
            session.commit()

            # Get search settings
            search_settings = SettingsManager.get_search_settings(session)
            
            # Process the search
            search_term = task.search_term
            params = task.parameters
            
            # Initialize Google search with parameters
            search_query = search_term
            if params.get('optimize_spanish'):
                search_query += ' site:.es OR site:.mx OR site:.ar OR site:.co'
            if params.get('optimize_english'):
                search_query += ' site:.com OR site:.org OR site:.net'
                
            # Perform Google search
            search_results = list(google_search(
                search_query, 
                num_results=params.get('num_results', 10),
                lang=params.get('language', 'es')
            ))
            
            total_results = len(search_results)
            processed = 0
            emails_found = 0
            valid_emails = []
            
            # Process each result
            async with aiohttp.ClientSession() as client:
                for url in search_results:
                    try:
                        # Fetch and parse webpage
                        headers = {'User-Agent': UserAgent().random}
                        html_content = await fetch_url(client, url, headers)
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Extract emails
                        emails = extract_emails(soup.get_text())
                        current_valid_emails = [email for email in emails if is_valid_email(email)]
                        valid_emails.extend(current_valid_emails)
                        
                        if current_valid_emails:
                            # Store results in database
                            for email in current_valid_emails:
                                lead = Lead(email=email)
                                session.add(lead)
                                session.flush()  # Get the lead ID
                                
                                lead_source = LeadSource(
                                    lead_id=lead.id,
                                    url=url,
                                    domain=urlparse(url).netloc,
                                    page_title=soup.title.string if soup.title else None
                                )
                                session.add(lead_source)
                                
                            emails_found += len(valid_emails)
                            
                        processed += 1
                        # Update progress
                        update_progress(processed, total_results)
                        
                    except Exception as e:
                        logger.error(f"Error processing URL {url}: {str(e)}")
                        automation_log.logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "message": f"Error processing URL {url}: {str(e)}",
                            "type": "error"
                        })
                        session.commit()
            
            # Send emails if enabled
            if params.get('enable_email_sending') and params.get('email_template_id'):
                for lead in session.query(Lead).filter(Lead.email.in_(valid_emails)).all():
                    await send_email(
                        session,
                        lead.email,
                        params['email_template_id'],
                        params.get('from_email')
                    )
            
            automation_log.status = 'completed'
            automation_log.leads_gathered = emails_found
            session.commit()

        except Exception as e:
            automation_log.status = 'failed'
            automation_log.logs.append({
                "timestamp": datetime.now().isoformat(),
                "message": f"Error: {str(e)}",
                "type": "error"
            })
            session.commit()
            raise

def extract_emails(text):
    """Extract email addresses from text using regex with improved patterns"""
    # Basic email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    # Additional patterns to find obfuscated emails
    obfuscated_patterns = [
        r'[a-zA-Z0-9._%+-]+\s*[\[\(]at\[\)\]\s*[a-zA-Z0-9.-]+\s*[\[\(]dot[\)\]\s*[a-zA-Z]{2,}',  # name(at)domain(dot)com
        r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\s*\.\s*[a-zA-Z]{2,}',  # Spaces around @ and .
        r'[a-zA-Z0-9._%+-]+\s*\[at\]\s*[a-zA-Z0-9.-]+\s*\[dot\]\s*[a-zA-Z]{2,}'  # [at] and [dot]
    ]
    
    emails = set()
    
    # Find standard emails
    found_emails = re.findall(email_pattern, text, re.IGNORECASE)
    emails.update(found_emails)
    
    # Find obfuscated emails
    for pattern in obfuscated_patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        for email in found:
            # Clean up obfuscated emails
            clean_email = email.replace(' ', '').replace('[at]', '@').replace('(at)', '@')
            clean_email = clean_email.replace('[dot]', '.').replace('(dot)', '.')
            if re.match(email_pattern, clean_email, re.IGNORECASE):
                emails.add(clean_email)
    
    return list(emails)

@app.get("/stream-search")
async def stream_search(
    term: str,
    num_results: int = 10,
    optimize_english: bool = False,
    optimize_spanish: bool = False,
    language: str = 'ES'
):
    """Stream search results for email extraction"""
    if not term:
        raise HTTPException(status_code=400, detail="Search term is required")
    
    if language not in ['ES', 'EN']:
        raise HTTPException(status_code=400, detail="Language must be either 'ES' or 'EN'")
        
    if num_results > 100:
        raise HTTPException(status_code=400, detail="Maximum number of results is 100")

    async def event_generator():
        try:
            # Build optimized search query
            contact_terms = [
                'contact', 'email', 'about', 'team', 'staff', 'people',
                'contacto', 'correo', 'equipo', 'nosotros', 'personal'
            ]
            
            base_query = term
            if optimize_spanish:
                base_query = f'"{term}" ({" OR ".join(contact_terms)})'
                base_query += ' site:.es OR site:.mx OR site:.ar OR site:.co OR site:.pe OR site:.cl'
            if optimize_english:
                base_query = f'"{term}" ({" OR ".join(contact_terms)})'
                base_query += ' site:.com OR site:.org OR site:.net OR site:.io OR site:.dev'
                
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting search...', 'progress': 0})}\n\n"
                
            # Perform Google search
            search_results = list(google_search(
                base_query, 
                num_results=num_results,
                lang=language
            ))
            
            total_results = len(search_results)
            processed = 0
            emails_found = 0
            
            # Process each result
            async with aiohttp.ClientSession() as client:
                for url in search_results:
                    try:
                        # Send progress update
                        progress = int((processed / total_results) * 100)
                        yield f"data: {json.dumps({'type': 'progress', 'value': progress})}\n\n"
                        
                        # Fetch and process URL
                        headers = {'User-Agent': UserAgent().random}
                        async with client.get(url, headers=headers, timeout=10) as response:
                            if response.status == 200:
                                html = await response.text()
                                # Process HTML and extract emails
                                # ... rest of the processing code ...
                            else:
                                yield f"data: {json.dumps({'type': 'error', 'url': url, 'status': response.status})}\n\n"
                        
                        processed += 1
                        await asyncio.sleep(0.1)

    except Exception as e:
                        logger.error(f"Error processing URL {url}: {str(e)}")
                        yield f"data: {json.dumps({'type': 'error', 'url': url, 'error': str(e)})}\n\n"
                        continue
            
            # Send completion event
                yield f"data: {json.dumps({'type': 'complete', 'total_processed': processed, 'total_emails': emails_found})}\n\n"
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# New endpoint to get task logs
@app.get("/task-logs/{task_id}")
async def get_task_logs(task_id: str):
    """Stream task logs with proper timeout handling"""
    if not task_id:
        raise HTTPException(status_code=400, detail="Task ID is required")
        
    async def event_stream():
        last_id = 0
        timeout = time.time() + 300  # 5 minute timeout
        
        while time.time() < timeout:
            with db_session() as session:
                # Get task status
                task = session.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()
                if not task:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Task not found'})}\n\n"
                    break
                    
                # Check if task is complete
                if task.status in ['completed', 'failed']:
                    yield f"data: {json.dumps({'type': 'complete', 'status': task.status})}\n\n"
                    break
                
                # Get new logs since last_id
                logs = session.query(AutomationLog).filter(
                    AutomationLog.id > last_id,
                    AutomationLog.task_id == task_id
                ).order_by(AutomationLog.id.asc()).all()
                
                for log in logs:
                    last_id = log.id
                    yield f"data: {json.dumps({'type': 'log', 'message': log.message})}\n\n"
            
            await asyncio.sleep(1)
        
        if time.time() >= timeout:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Stream timeout'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Database Models
class SearchTerm(Base):
    __tablename__ = "search_terms"
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

class Lead(Base):
    __tablename__ = "leads"
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

class EmailCampaign(Base):
    __tablename__ = "email_campaigns"
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

class SearchTermGroup(Base):
    __tablename__ = "search_term_groups"
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    email_template = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_terms = relationship("SearchTerm", back_populates="group")

class EmailTemplate(Base):
    __tablename__ = "email_templates"
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
        """Convert the knowledge base to a dictionary"""
        return {
            'kb_name': self.kb_name,
            'kb_bio': self.kb_bio,
            'kb_values': self.kb_values,
            'contact_name': self.contact_name,
            'contact_role': self.contact_role,
            'contact_email': self.contact_email,
            'company_description': self.company_description,
            'company_mission': self.company_mission,
            'company_target_market': self.company_target_market,
            'company_other': self.company_other,
            'product_name': self.product_name,
            'product_description': self.product_description,
            'product_target_customer': self.product_target_customer,
            'product_other': self.product_other,
            'other_context': self.other_context,
            'example_email': self.example_email
        }

class CampaignLead(Base):
    __tablename__ = 'campaign_leads'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    status = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="campaign_leads")
    campaign = relationship("Campaign", back_populates="campaign_leads")

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
    setting_type = Column(Text, nullable=False)  # 'general', 'email', etc.
    value = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    email = Column(Text, nullable=False)
    provider = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    smtp_server = Column(Text)
    smtp_port = Column(BigInteger)
    smtp_username = Column(Text)
    smtp_password = Column(Text)
    aws_access_key_id = Column(Text)
    aws_secret_access_key = Column(Text)
    aws_region = Column(Text)

class AutomationStatus(Base):
    __tablename__ = 'automation_status'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    is_running = Column(Boolean, default=False)
    current_search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    emails_sent_in_current_group = Column(BigInteger, default=0)
    last_run_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    campaign = relationship("Campaign")
    current_search_term = relationship("SearchTerm")

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

# Add connection retry logic with improved error handling
def get_db():
    db = None
    try:
        db = SessionLocal()
        # Test the connection with a proper SQLAlchemy query
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        return db
    except Exception as e:
        if db:
            db.close()
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Database connection error. Please try again later."
        )

@contextmanager
def db_session():
    db = None
    try:
        db = get_db()
        yield db
        db.commit()  # Automatically commit if no exceptions
    except Exception as e:
        if db:
            db.rollback()  # Rollback on error
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        if db:
            db.close()

# HTML template as a constant (not written to file)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html data-theme="light">
<head>
    <title>AutoclientAI | Lead Generation</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/daisyui@3.9.3/dist/full.css" rel="stylesheet" type="text/css" />
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tom-select/dist/css/tom-select.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/tom-select/dist/js/tom-select.complete.min.js"></script>
    <style>
        .nav-active {
            background-color: rgba(37, 99, 235, 0.1);
            border-left: 4px solid #2563eb;
        }
        .search-chip {
            transition: all 0.3s ease;
        }
        .search-chip:hover {
            transform: translateY(-2px);
        }
        .results-card {
            transition: all 0.3s ease;
        }
        .results-card:hover {
            transform: translateX(5px);
        }
    </style>
</head>
<body class="bg-gray-50">
    <!-- Top Navigation -->
    <div class="navbar bg-base-100 shadow-lg px-4 sm:px-8">
        <div class="flex-1">
            <a class="btn btn-ghost normal-case text-xl">
                <i class="fas fa-robot text-primary mr-2"></i>
                AutoclientAI
            </a>
        </div>
        <div class="flex-none gap-2">
            <div class="tabs tabs-boxed">
                <a class="tab tab-active" onclick="switchPage('search')">
                    <i class="fas fa-search mr-2"></i>Search
                </a>
                <a class="tab" onclick="switchPage('templates')">
                    <i class="fas fa-envelope mr-2"></i>Templates
                </a>
                <a class="tab" onclick="switchPage('leads')">
                    <i class="fas fa-users mr-2"></i>Leads
                </a>
                <a class="tab" onclick="switchPage('settings')">
                    <i class="fas fa-cog mr-2"></i>Settings
                </a>
                <a class="tab" onclick="switchPage('analytics')">
                    <i class="fas fa-chart-line mr-2"></i>Analytics
                </a>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="flex min-h-screen">
        <!-- Left Sidebar -->
        <div class="w-1/5 bg-base-100 p-4 shadow-lg">
            <div class="space-y-4">
                <div class="card bg-base-200">
                    <div class="card-body">
                        <h3 class="card-title text-sm">
                            <i class="fas fa-history mr-2"></i>Recent Searches
                        </h3>
                        <div id="recent-terms-list" class="space-y-2">
                                        <!-- Recent terms will be added here -->
                                    </div>
                                </div>
                            </div>

                <div class="card bg-base-200">
                    <div class="card-body">
                        <h3 class="card-title text-sm">
                            <i class="fas fa-save mr-2"></i>Saved Searches
                                </h3>
                        <div id="saved-searches" class="space-y-2">
                            <!-- Saved searches will be added here -->
                        </div>
                    </div>
                </div>

                <div class="card bg-base-200">
                    <div class="card-body">
                        <h3 class="card-title text-sm">
                            <i class="fas fa-filter mr-2"></i>Quick Filters
                        </h3>
                        <div class="space-y-2">
                                <label class="label cursor-pointer justify-start gap-2">
                                <input type="checkbox" class="checkbox checkbox-primary" id="filter-companies">
                                <span>Companies Only</span>
                                </label>
                                <label class="label cursor-pointer justify-start gap-2">
                                <input type="checkbox" class="checkbox checkbox-primary" id="filter-decision-makers">
                                <span>Decision Makers</span>
                                </label>
                                <label class="label cursor-pointer justify-start gap-2">
                                <input type="checkbox" class="checkbox checkbox-primary" id="filter-verified">
                                <span>Verified Emails</span>
                                </label>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Content Area -->
        <div class="w-3/5 p-6">
            <!-- Search Interface -->
            <div class="card bg-base-100 shadow-xl mb-6">
                <div class="card-body">
                    <div class="flex items-center gap-4 mb-4">
                        <div class="flex-1">
                            <select id="search-term" class="w-full" multiple placeholder="Enter search terms...">
                                <!-- Search terms will be added here -->
                            </select>
                        </div>
                        <button class="btn btn-primary" onclick="saveSearch()">
                            <i class="fas fa-save mr-2"></i>Save
                        </button>
                                </div>

                    <div class="grid grid-cols-2 gap-4 mb-4">
                        <div class="form-control">
                            <label class="label">
                                <span class="label-text">Language</span>
                            </label>
                            <select class="select select-bordered w-full" id="language">
                                <option value="ES">
                                    <i class="flag-icon flag-icon-es"></i>Spanish
                                </option>
                                <option value="EN">
                                    <i class="flag-icon flag-icon-gb"></i>English
                                </option>
                                    </select>
                                </div>

                        <div class="form-control">
                            <label class="label">
                                <span class="label-text">Results per term</span>
                                </label>
                            <input type="number" id="num-results" class="input input-bordered" value="10" min="1" max="100">
                        </div>
                            </div>

                    <div class="divider">Advanced Options</div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="label cursor-pointer justify-start gap-2">
                                <input type="checkbox" id="optimize-english" class="checkbox checkbox-primary">
                                <span>Optimize for English</span>
                            </label>
                            <label class="label cursor-pointer justify-start gap-2">
                                <input type="checkbox" id="optimize-spanish" class="checkbox checkbox-primary">
                                <span>Optimize for Spanish</span>
                            </label>
                        </div>
                        <div>
                            <label class="label cursor-pointer justify-start gap-2">
                                <input type="checkbox" id="enable-email-sending" class="checkbox checkbox-secondary">
                                <span>Enable Email Sending</span>
                            </label>
                            <div id="email-options" class="mt-2" style="display: none;">
                                <select id="email-template" class="select select-bordered w-full mb-2">
                                    <option value="">Select Email Template...</option>
                                </select>
                                <select id="from-email" class="select select-bordered w-full">
                                    <option value="">Select From Email...</option>
                                </select>
                            </div>
                                </div>
                            </div>

                    <div class="mt-4">
                        <button class="btn btn-primary w-full" onclick="startSearch()">
                            <i class="fas fa-search mr-2"></i>Search
                                </button>
                            </div>
                </div>
            </div>

            <!-- Search Results -->
            <div id="results-container">
                        <div id="loading" class="text-center p-8" style="display: none;">
                            <span class="loading loading-spinner loading-lg text-primary"></span>
                            <p class="mt-4 text-gray-600">Searching...</p>
                        </div>
                <div id="results" class="space-y-4">
                    <!-- Results will be added here -->
            </div>
        </div>
    </div>

        <!-- Right Sidebar -->
        <div class="w-1/5 bg-base-100 p-4 shadow-lg">
            <!-- Stats -->
            <div class="stats stats-vertical shadow w-full mb-4">
            <div class="stat">
                    <div class="stat-title">Pages Scanned</div>
                    <div class="stat-value text-primary" id="pages-scanned">0</div>
            </div>
            <div class="stat">
                    <div class="stat-title">Emails Found</div>
                    <div class="stat-value text-secondary" id="emails-found">0</div>
            </div>
            <div class="stat">
                    <div class="stat-title">Success Rate</div>
                    <div class="stat-value text-accent" id="success-rate">0%</</div>
            </div>
        </div>

            <!-- Quick Actions -->
            <div class="card bg-base-200 mb-4">
                    <div class="card-body">
                    <h3 class="card-title text-sm">
                        <i class="fas fa-bolt mr-2"></i>Quick Actions
                    </h3>
                    <div class="space-y-2">
                        <button class="btn btn-sm btn-block" onclick="exportResults()">
                            <i class="fas fa-download mr-2"></i>Export Results
                        </button>
                        <button class="btn btn-sm btn-block" onclick="sendEmails()">
                            <i class="fas fa-envelope mr-2"></i>Send Emails
                        </button>
                        <button class="btn btn-sm btn-block" onclick="optimizeTerms()">
                            <i class="fas fa-magic mr-2"></i>Optimize Terms
                        </button>
                </div>
            </div>
        </div>

            <!-- Progress -->
            <div class="card bg-base-200">
                    <div class="card-body">
                    <h3 class="card-title text-sm">
                        <i class="fas fa-tasks mr-2"></i>Progress
                    </h3>
                    <div id="progress-container" style="display: none;">
                        <div class="w-full bg-gray-200 rounded-full h-2.5 mb-2">
                            <div id="progress-bar-fill" class="bg-primary h-2.5 rounded-full" style="width: 0%"></div>
                    </div>
                        <div id="progress-text" class="text-center text-sm text-gray-600"></div>
                </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize TomSelect for search terms
        let searchSelect = new TomSelect('#search-term', {
            create: true,
            createOnBlur: true,
            maxItems: null,
            persist: false,
            plugins: ['remove_button'],
            render: {
                item: function(data, escape) {
                    return `<div class="search-chip">${escape(data.text)}</div>`;
                }
            }
        });

        // Load recent terms
        async function loadRecentTerms() {
            try {
                const response = await fetch('/recent-search-terms');
                const terms = await response.json();
                
                const recentTermsList = document.getElementById('recent-terms-list');
                recentTermsList.innerHTML = terms.map(term => `
                    <div class="search-chip flex items-center justify-between p-2 bg-base-300 rounded hover:bg-base-200 cursor-pointer"
                         onclick="searchSelect.addItem('${term.term}')">
                        <span class="text-sm">${term.term}</span>
                        <span class="badge badge-sm">${term.lead_count}</span>
                    </div>
                `).join('');
                
                        terms.forEach(term => {
                    searchSelect.addOption({value: term.term, text: term.term});
                });
            } catch (error) {
                console.error('Error loading recent terms:', error);
            }
        }
        
        // Initialize email settings
        document.getElementById('enable-email-sending').addEventListener('change', function() {
            document.getElementById('email-options').style.display = this.checked ? 'block' : 'none';
        });

        // Load email settings
        async function loadEmailSettings() {
            try {
                const [templatesResponse, settingsResponse] = await Promise.all([
                    fetch('/email-templates'),
                    fetch('/email-settings')
                ]);
                
                const templates = await templatesResponse.json();
                const settings = await settingsResponse.json();
                
                const templateSelect = document.getElementById('email-template');
                const fromEmailSelect = document.getElementById('from-email');
                
                templates.forEach(template => {
                    const option = document.createElement('option');
                    option.value = template.id;
                    option.textContent = template.template_name;
                    templateSelect.appendChild(option);
                });
                
                settings.forEach(setting => {
                    const option = document.createElement('option');
                    option.value = setting.email;
                    option.textContent = `${setting.name} (${setting.email})`;
                    fromEmailSelect.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading email settings:', error);
            }
        }

        // Start search
        async function startSearch() {
            const terms = searchSelect.items;
            if (!terms.length) {
                showError('Please enter at least one search term');
                return;
            }

            const resultsDiv = document.getElementById('results');
            const loadingDiv = document.getElementById('loading');
            
            resultsDiv.innerHTML = '';
            loadingDiv.style.display = 'block';
            document.getElementById('progress-container').style.display = 'block';
            
            const searchParams = new URLSearchParams({
                terms: JSON.stringify(terms),
                num_results: document.getElementById('num-results').value,
                optimize_english: document.getElementById('optimize-english').checked,
                optimize_spanish: document.getElementById('optimize-spanish').checked,
                language: document.getElementById('language').value,
                enable_email_sending: document.getElementById('enable-email-sending').checked,
                email_template_id: document.getElementById('email-template').value,
                from_email: document.getElementById('from-email').value
            });

            try {
                const response = await fetch(`/bulk-search?${searchParams.toString()}`);
                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const {value, done} = await reader.read();
                    if (done) break;
                    
                    const events = decoder.decode(value).split('\n\n');
                    for (const event of events) {
                        if (!event.trim()) continue;
                        
                        const data = JSON.parse(event.replace('data: ', ''));
                        handleSearchEvent(data);
                    }
                }
            } catch (error) {
                console.error('Search error:', error);
                showError('An error occurred during search');
            } finally {
                loadingDiv.style.display = 'none';
                document.getElementById('progress-container').style.display = 'none';
            }
        }

        // Handle search events
        function handleSearchEvent(data) {
            const resultsDiv = document.getElementById('results');
            
            switch (data.type) {
                case 'result':
                    const resultDiv = document.createElement('div');
                    resultDiv.className = 'results-card card bg-base-200 hover:bg-base-300';
                    resultDiv.innerHTML = `
                        <div class="card-body">
                            <h3 class="card-title text-lg">
                                <i class="fas fa-globe mr-2"></i>
                                ${data.title || 'No title'}
                            </h3>
                            <p class="text-sm text-gray-600">
                                <i class="fas fa-link mr-1"></i>
                                <a href="${data.url}" target="_blank" class="link link-primary">${data.url}</a>
                            </p>
                            <p class="text-sm text-gray-600">
                                <i class="fas fa-server mr-1"></i>
                                ${data.domain}
                            </p>
                            <div class="mt-2">
                                ${data.emails.length ? `
                                <div class="font-semibold text-secondary">
                                    <i class="fas fa-envelope mr-1"></i>
                                        Emails Found:
                                </div>
                                    <div class="mt-1 flex flex-wrap gap-2">
                                        ${data.emails.map(email => `
                                            <div class="badge badge-secondary gap-1">
                                                <i class="fas fa-envelope-open text-xs"></i>
                                                ${email}
                                            </div>
                                        `).join('')}
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                    resultsDiv.appendChild(resultDiv);
                    
                    // Update stats
                    document.getElementById('pages-scanned').textContent = data.total_processed;
                    document.getElementById('emails-found').textContent = data.total_emails_found;
                    document.getElementById('success-rate').textContent = 
                        `${((data.total_emails_found / data.total_processed) * 100).toFixed(1)}%`;
                    
                    // Update progress
                    const progress = (data.total_processed / data.total) * 100;
                    document.getElementById('progress-bar-fill').style.width = `${progress}%`;
                    document.getElementById('progress-text').textContent = 
                        `Processing: ${data.total_processed}/${data.total}`;
                    break;
                    
                case 'error':
                    console.error('Search error:', data.message);
                    showError(data.message);
                    break;
                    
                case 'complete':
                    console.log('Search completed:', data);
                    break;
            }
        }

        // Show error message
        function showError(message) {
            // Implementation here
        }

        // Export results
        function exportResults() {
            const resultsDiv = document.getElementById('results');
            const results = [];

            // Extract data from result cards
            resultsDiv.querySelectorAll('.results-card').forEach(card => {
                const title = card.querySelector('.card-title').textContent.trim();
                const url = card.querySelector('.link-primary').href;
                const domain = card.querySelector('.text-sm:nth-child(3)').textContent.replace('<i class="fas fa-server mr-1"></i>', '').trim();
                const emails = Array.from(card.querySelectorAll('.badge-secondary')).map(badge => badge.textContent.trim());

                results.push({
                    title: title,
                    url: url,
                    domain: domain,
                    emails: emails
                });
            });

            if (results.length === 0) {
                showError('No results to export');
                return;
            }

            // Convert results to JSON
            const json = JSON.stringify(results, null, 2);

            // Create a Blob and trigger download
            const blob = new Blob([json], {
                type: 'application/json'
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'search_results.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        // send emails
        async function sendEmails() {
            const templateId = document.getElementById('email-template').value;
            const fromEmail = document.getElementById('from-email').value;
            const resultsDiv = document.getElementById('results');
            const emails = new Set();

            // Collect emails from results
            resultsDiv.querySelectorAll('.results-card').forEach(card => {
                card.querySelectorAll('.badge-secondary').forEach(badge => {
                    emails.add(badge.textContent.trim());
                });
            });

            if (emails.size === 0) {
                showError('No emails found to send');
                return;
            }

            if (!templateId || !fromEmail) {
                showError('Please select an email template and a "from" email');
                return;
            }

            try {
                const response = await fetch('/send-emails', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        emails: Array.from(emails),
                        template_id: templateId,
                        from_email: fromEmail
                    })
                });

                if (response.ok) {
                    alert('Emails are being sent!');
                } else {
                    showError('Failed to send emails');
                }
            } catch (error) {
                console.error('Error sending emails:', error);
                showError('An error occurred while sending emails');
            }
        }

        // switch page
        function switchPage(page) {
            alert(`Navigating to ${page} (Not fully implemented)`);
            // TODO: Implement page switching logic here
        }
    </script>
</body>
</html>'''

# Add SettingsManager class before the routes
class SettingsManager:
    @staticmethod
    def get_search_settings(session):
        """Get search settings with defaults"""
        settings = session.query(Settings).filter(
            Settings.setting_type == 'search',
            Settings.is_active == True
        ).all()
        
        # Default settings
        defaults = {
            'max_results_per_term': 100,
            'rate_limit_delay': 2,
            'max_retries': 3,
            'timeout': 30,
            'ignore_domains': ['example.com', 'test.com']
        }
        
        # Override defaults with database settings
        for setting in settings:
            if isinstance(setting.value, dict):
                defaults.update(setting.value)
        
        return defaults

    @staticmethod
    def get_email_settings(session):
        """Get email settings with defaults"""
        settings = session.query(Settings).filter(
            Settings.setting_type == 'email',
            Settings.is_active == True
        ).all()
        
        # Default settings
        defaults = {
            'retry_attempts': 3,
            'retry_delay': 60,
            'batch_size': 50,
            'daily_limit': 1000
        }
        
        # Override defaults with database settings
        for setting in settings:
            if isinstance(setting.value, dict):
                defaults.update(setting.value)
        
        return defaults

    @staticmethod
    def get_ai_settings(session):
        """Get AI settings with defaults"""
        settings = session.query(Settings).filter(
            Settings.setting_type == 'ai',
            Settings.is_active == True
        ).all()
        
        # Default settings
        defaults = {
            'model_name': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 2000,
            'openai_api_key': os.getenv('OPENAI_API_KEY', '')
        }
        
        # Override defaults with database settings
        for setting in settings:
            if isinstance(setting.value, dict):
                defaults.update(setting.value)
        
        return defaults

class ActiveState:
    def __init__(self):
        self._project_id = 1
        self._campaign_id = 1
        self._lock = asyncio.Lock()
    
    async def get_project_id(self):
        async with self._lock:
            return self._project_id
    
    async def get_campaign_id(self):
        async with self._lock:
            return self._campaign_id
    
    async def set_project_id(self, project_id: int):
        async with self._lock:
            self._project_id = project_id
    
    async def set_campaign_id(self, campaign_id: int):
        async with self._lock:
            self._campaign_id = campaign_id

active_state = ActiveState()

@app.post("/set-active-project")
async def set_active_project(request: Request):
    """Set the active project with proper state management"""
    try:
        data = await request.json()
        project_id = data.get('project_id')
        if not project_id:
            raise HTTPException(status_code=400, detail="Project ID is required")
        
        with db_session() as session:
            project = session.query(Project).filter_by(id=project_id).first()
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            await active_state.set_project_id(project_id)
            
            return {"status": "success", "message": f"Active project set to: {project.project_name}"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set-active-campaign")
async def set_active_campaign(request: Request):
    """Set the active campaign with proper state management"""
    try:
        data = await request.json()
        campaign_id = data.get('campaign_id')
        if not campaign_id:
            raise HTTPException(status_code=400, detail="Campaign ID is required")
        
        with db_session() as session:
            campaign = session.query(Campaign).filter_by(id=campaign_id).first()
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            await active_state.set_campaign_id(campaign_id)
            
            return {"status": "success", "message": f"Active campaign set to: {campaign.campaign_name}"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Routes
@app.get("/")
async def root():
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)

@app.get("/recent-search-terms")
async def get_recent_search_terms():
    try:
        with db_session() as session:
            recent_terms = (
                session.query(
                    SearchTerm.term, 
                    func.count(LeadSource.id).label('lead_count'),
                    SearchTerm.created_at.label('last_used')
                )
                .outerjoin(LeadSource, SearchTerm.id == LeadSource.search_term_id)
                .group_by(SearchTerm.term, SearchTerm.created_at)
                .order_by(SearchTerm.created_at.desc().nullslast())
                .limit(20)
                .all()
            )
            
            terms_data = [{
                "term": term.term,
                "lead_count": int(term.lead_count or 0),
                "last_used": term.last_used.isoformat() if term.last_used else None
            } for term in recent_terms]
            
            return JSONResponse(content=terms_data)
    except Exception as e:
        logger.error(f"Error fetching recent search terms: {str(e)}")
        return JSONResponse(
            content={"terms": [], "message": "No search terms found"},
            status_code=200
        )

@app.get("/email-templates")
async def get_email_templates():
    """Get all email templates with proper error handling"""
    try:
        with db_session() as session:
            templates = session.query(EmailTemplate).all()
            return {
                "templates": [
                    {
                "id": template.id,
                "name": template.template_name,
                "subject": template.subject,
                        "body": template.body_content,
                        "language": template.language,
                        "is_ai_customizable": template.is_ai_customizable,
                        "created_at": template.created_at.isoformat() if template.created_at else None
                    } for template in templates
                ]
            }
    except Exception as e:
        logger.error(f"Error fetching email templates: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch email templates")

@app.get("/email-settings")
async def get_email_settings():
    """Get all email settings with proper error handling"""
    try:
        with db_session() as session:
            settings = session.query(EmailSettings).filter_by(is_active=True).all()
            return {
                "settings": [
                    {
                "id": setting.id,
                "name": setting.name,
                "email": setting.email,
                        "provider": setting.provider,
                        "is_active": setting.is_active,
                        "smtp_server": setting.smtp_server if setting.provider == 'smtp' else None,
                        "smtp_port": setting.smtp_port if setting.provider == 'smtp' else None,
                        "aws_region": setting.aws_region if setting.provider == 'aws' else None
                    } for setting in settings
                ]
            }
    except Exception as e:
        logger.error(f"Error fetching email settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch email settings")

@app.post("/email-templates")
async def create_email_template(request: Request):
    """Create a new email template with validation"""
    try:
        data = await request.json()
        
        # Validate required fields
        required_fields = ['template_name', 'subject', 'body_content', 'language']
        for field in required_fields:
            if not data.get(field):
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        with db_session() as session:
            template = EmailTemplate(
                template_name=data['template_name'],
                subject=data['subject'],
                body_content=data['body_content'],
                language=data['language'],
                is_ai_customizable=data.get('is_ai_customizable', False),
                campaign_id=await active_state.get_campaign_id()
            )
            session.add(template)
                session.commit()
                
            return {
                "status": "success",
                "message": "Template created successfully",
                "template_id": template.id
            }
            
    except HTTPException:
        raise
            except Exception as e:
        logger.error(f"Error creating email template: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create email template")

@app.put("/email-templates/{template_id}")
async def update_email_template(template_id: int, request: Request):
    """Update an existing email template with validation"""
    try:
        data = await request.json()
        
        with db_session() as session:
            template = session.query(EmailTemplate).filter_by(id=template_id).first()
            if not template:
                raise HTTPException(status_code=404, detail="Template not found")
            
            # Update fields if provided
            if 'template_name' in data:
                template.template_name = data['template_name']
            if 'subject' in data:
                template.subject = data['subject']
            if 'body_content' in data:
                template.body_content = data['body_content']
            if 'language' in data:
                template.language = data['language']
            if 'is_ai_customizable' in data:
                template.is_ai_customizable = data['is_ai_customizable']
                
            session.commit()
            
            return {
                "status": "success",
                "message": "Template updated successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating email template: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update email template")

@app.delete("/email-templates/{template_id}")
async def delete_email_template(template_id: int):
    """Delete an email template with proper cleanup"""
    try:
        with db_session() as session:
            template = session.query(EmailTemplate).filter_by(id=template_id).first()
            if not template:
                raise HTTPException(status_code=404, detail="Template not found")
            
            # Check if template is in use
            campaign_count = session.query(EmailCampaign).filter_by(template_id=template_id).count()
            if campaign_count > 0:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot delete template that is in use by email campaigns"
                )
            
            session.delete(template)
        session.commit()
            
            return {
                "status": "success",
                "message": "Template deleted successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting email template: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete email template")

@app.get("/analytics/overview")
async def get_analytics_overview():
    """Get overview analytics with proper error handling"""
    try:
        with db_session() as session:
            # Get active campaign
            campaign_id = await active_state.get_campaign_id()
            campaign = session.query(Campaign).filter_by(id=campaign_id).first()
            if not campaign:
                raise HTTPException(status_code=404, detail="Active campaign not found")
            
            # Get analytics data
            total_leads = session.query(Lead).join(CampaignLead).filter(
                CampaignLead.campaign_id == campaign_id
            ).count()
            
            total_emails_sent = session.query(EmailCampaign).filter(
                EmailCampaign.campaign_id == campaign_id,
                EmailCampaign.status == 'sent'
            ).count()
            
            total_opens = session.query(func.sum(EmailCampaign.open_count)).filter(
                EmailCampaign.campaign_id == campaign_id
            ).scalar() or 0
            
            total_clicks = session.query(func.sum(EmailCampaign.click_count)).filter(
                EmailCampaign.campaign_id == campaign_id
            ).scalar() or 0
            
            # Calculate rates
            open_rate = (total_opens / total_emails_sent * 100) if total_emails_sent > 0 else 0
            click_rate = (total_clicks / total_emails_sent * 100) if total_emails_sent > 0 else 0
            
            return {
                "campaign_name": campaign.campaign_name,
                    "total_leads": total_leads,
                    "total_emails_sent": total_emails_sent,
                "total_opens": total_opens,
                "total_clicks": total_clicks,
                "open_rate": round(open_rate, 2),
                "click_rate": round(click_rate, 2)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics overview: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get analytics overview")

@app.get("/analytics/search-terms")
async def get_search_term_analytics():
    """Get search term performance analytics with proper error handling"""
    try:
        with db_session() as session:
            campaign_id = await active_state.get_campaign_id()
            
            # Get all search terms for the campaign with their effectiveness data
            search_terms = session.query(
                SearchTerm,
                SearchTermEffectiveness,
                func.count(Lead.id).label('total_leads')
            ).outerjoin(
                SearchTermEffectiveness,
                SearchTerm.id == SearchTermEffectiveness.search_term_id
            ).outerjoin(
                LeadSource,
                SearchTerm.id == LeadSource.search_term_id
            ).outerjoin(
                Lead,
                LeadSource.lead_id == Lead.id
            ).filter(
                SearchTerm.campaign_id == campaign_id
            ).group_by(
                SearchTerm.id,
                SearchTermEffectiveness.id
            ).all()
            
            results = []
            for term, effectiveness, total_leads in search_terms:
                result = {
                    "id": term.id,
                    "term": term.term,
                    "total_leads": total_leads,
                    "total_results": effectiveness.total_results if effectiveness else 0,
                    "valid_leads": effectiveness.valid_leads if effectiveness else 0,
                    "irrelevant_leads": effectiveness.irrelevant_leads if effectiveness else 0,
                    "success_rate": round(
                        (effectiveness.valid_leads / effectiveness.total_results * 100)
                        if effectiveness and effectiveness.total_results > 0
                        else 0,
                        2
                    )
                }
                results.append(result)
            
            return {"search_terms": results}
            
    except Exception as e:
        logger.error(f"Error getting search term analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get search term analytics")

@app.get("/analytics/email-campaigns")
async def get_email_campaign_analytics():
    """Get email campaign performance analytics with proper error handling"""
    try:
        with db_session() as session:
            campaign_id = await active_state.get_campaign_id()
            
            # Get all email campaigns with their performance data
            email_campaigns = session.query(
                EmailCampaign,
                Lead,
                EmailTemplate
            ).join(
                Lead,
                EmailCampaign.lead_id == Lead.id
            ).join(
                EmailTemplate,
                EmailCampaign.template_id == EmailTemplate.id
            ).filter(
                EmailCampaign.campaign_id == campaign_id
            ).all()
            
            results = []
            for campaign, lead, template in email_campaigns:
                                result = {
                    "id": campaign.id,
                    "lead_email": lead.email,
                    "template_name": template.template_name,
                    "sent_at": campaign.sent_at.isoformat() if campaign.sent_at else None,
                    "status": campaign.status,
                    "opens": campaign.open_count,
                    "clicks": campaign.click_count,
                    "is_ai_customized": campaign.ai_customized
                }
                results.append(result)
            
            return {
                "total_campaigns": len(results),
                "campaigns": results
            }
                    
                except Exception as e:
        logger.error(f"Error getting email campaign analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get email campaign analytics")

@app.post("/automation/start")
async def start_automation():
    """Start the automation process for the active campaign with proper error handling"""
    try:
        with db_session() as session:
            campaign_id = await active_state.get_campaign_id()
            if not campaign_id:
                raise HTTPException(status_code=400, detail="No active campaign selected")
            
            # Get campaign
            campaign = session.query(Campaign).filter_by(id=campaign_id).first()
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            # Check if automation is already running
            automation_status = session.query(AutomationStatus).filter_by(
                campaign_id=campaign_id
            ).first()
            
            if automation_status and automation_status.is_running:
                raise HTTPException(status_code=400, detail="Automation is already running")
            
            # Create or update automation status
            if not automation_status:
                automation_status = AutomationStatus(campaign_id=campaign_id)
                session.add(automation_status)
            
            automation_status.is_running = True
            automation_status.last_run_at = func.now()
            automation_status.emails_sent_in_current_group = 0
            session.commit()

            # Start background task
            asyncio.create_task(run_automation(campaign_id))
            
            return {
                "status": "success",
                "message": "Automation started successfully",
                "campaign_id": campaign_id,
                "campaign_name": campaign.campaign_name
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting automation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start automation")

@app.post("/automation/stop")
async def stop_automation():
    """Stop the automation process for the active campaign with proper error handling"""
    try:
        with db_session() as session:
            campaign_id = await active_state.get_campaign_id()
            if not campaign_id:
                raise HTTPException(status_code=400, detail="No active campaign selected")
            
            # Get automation status
            automation_status = session.query(AutomationStatus).filter_by(
                campaign_id=campaign_id
            ).first()
            
            if not automation_status:
                raise HTTPException(status_code=404, detail="No automation status found")
            
            if not automation_status.is_running:
                raise HTTPException(status_code=400, detail="Automation is not running")
            
            # Stop automation
                automation_status.is_running = False
            automation_status.last_run_at = func.now()
                session.commit()
            
            return {
                "status": "success",
                "message": "Automation stopped successfully",
                "campaign_id": campaign_id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping automation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to stop automation")

@app.get("/automation/status")
async def get_automation_status():
    """Get the current automation status with proper error handling"""
    try:
        with db_session() as session:
            campaign_id = await active_state.get_campaign_id()
            if not campaign_id:
                raise HTTPException(status_code=400, detail="No active campaign selected")
            
            # Get automation status
            automation_status = session.query(AutomationStatus).filter_by(
                campaign_id=campaign_id
            ).first()
            
            if not automation_status:
                return {
                    "status": "not_found",
                    "is_running": False,
                    "campaign_id": campaign_id,
                    "last_run_at": None,
                    "current_search_term": None,
                    "emails_sent_in_current_group": 0
                }
            
            # Get current search term info if exists
            current_term = None
            if automation_status.current_search_term_id:
                search_term = session.query(SearchTerm).filter_by(
                    id=automation_status.current_search_term_id
                ).first()
                if search_term:
                current_term = {
                        "id": search_term.id,
                        "term": search_term.term,
                        "language": search_term.language
                }
            
            return {
                "status": "success",
                "is_running": automation_status.is_running,
                "campaign_id": campaign_id,
                "last_run_at": automation_status.last_run_at.isoformat() if automation_status.last_run_at else None,
                "current_search_term": current_term,
                "emails_sent_in_current_group": automation_status.emails_sent_in_current_group
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting automation status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get automation status")

@app.get("/automation/logs")
async def get_automation_logs():
    """Get automation logs with proper error handling"""
    try:
        with db_session() as session:
            campaign_id = await active_state.get_campaign_id()
            if not campaign_id:
                raise HTTPException(status_code=400, detail="No active campaign selected")
            
            # Get logs for the campaign
            logs = session.query(AutomationLog).filter_by(
                campaign_id=campaign_id
            ).order_by(
                AutomationLog.start_time.desc()
            ).limit(100).all()
            
            return {
                "status": "success",
                "logs": [{
                    "id": log.id,
                    "start_time": log.start_time.isoformat() if log.start_time else None,
                    "end_time": log.end_time.isoformat() if log.end_time else None,
                    "leads_gathered": log.leads_gathered,
                    "emails_sent": log.emails_sent,
                    "status": log.status,
                    "details": log.logs
                } for log in logs]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting automation logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get automation logs")

async def run_automation(campaign_id: int):
    """Background task to run the automation process"""
    try:
        while True:
            with db_session() as session:
                # Check if automation should continue
                automation_status = session.query(AutomationStatus).filter_by(
                    campaign_id=campaign_id
                ).first()
                
                if not automation_status or not automation_status.is_running:
                    break
                
                # Get campaign settings
                campaign = session.query(Campaign).get(campaign_id)
                if not campaign:
                    logger.error(f"Campaign {campaign_id} not found")
                    break
                
                # Get next search term
                if automation_status.current_search_term_id is None:
                    search_term = session.query(SearchTerm).filter(
                        SearchTerm.campaign_id == campaign_id
                    ).order_by(SearchTerm.id).first()
                else:
                    search_term = session.query(SearchTerm).filter(
                        SearchTerm.campaign_id == campaign_id,
                        SearchTerm.id > automation_status.current_search_term_id
                    ).order_by(SearchTerm.id).first()
                
                if not search_term:
                    if campaign.loop_automation:
                        automation_status.current_search_term_id = None
                        session.commit()
                        continue
                    else:
                        automation_status.is_running = False
                        session.commit()
                        break
                
                # Update current search term
                automation_status.current_search_term_id = search_term.id
                session.commit()
                
                # Create log entry
                log_entry = AutomationLog(
                    campaign_id=campaign_id,
                    search_term_id=search_term.id,
                    start_time=func.now(),
                    status="running",
                    leads_gathered=0,
                    emails_sent=0,
                    logs={"messages": []}
                )
                session.add(log_entry)
                session.commit()
                
                try:
                    # Process search term
                    await process_search_term(session, search_term, campaign, automation_status, log_entry)
                    
                    # Update log entry
                    log_entry.status = "completed"
                    log_entry.end_time = func.now()
                    session.commit()
                    
                except Exception as e:
                    logger.error(f"Error processing search term {search_term.id}: {str(e)}")
                    log_entry.status = "failed"
                    log_entry.end_time = func.now()
                    log_entry.logs["error"] = str(e)
                    session.commit()
                
                # Wait for loop interval
                await asyncio.sleep(campaign.loop_interval)
                
    except Exception as e:
        logger.error(f"Error in automation process: {str(e)}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_google_search(query: str, num_results: int = 10, lang: str = 'ES') -> list:
    """Safely perform Google search with retries and error handling"""
    try:
        results = list(google_search(query, num_results=num_results, lang=lang))
        return results
    except Exception as e:
        logger.error(f"Error performing Google search: {str(e)}")
        raise

# Update the process_search_term function to use safe_google_search
async def process_search_term(session, search_term, campaign, automation_status, log_entry):
    """Process a single search term and send emails to found leads"""
    try:
        # Build search query
        contact_terms = [
            'contact', 'email', 'about', 'team', 'staff', 'people',
            'contacto', 'correo', 'equipo', 'nosotros', 'personal'
        ]
        
        base_query = search_term.term
        if search_term.language == 'ES':
            base_query = f'"{search_term.term}" ({" OR ".join(contact_terms)})'
            base_query += ' site:.es OR site:.mx OR site:.ar OR site:.co OR site:.pe OR site:.cl'
        else:
            base_query = f'"{search_term.term}" ({" OR ".join(contact_terms)})'
            base_query += ' site:.com OR site:.org OR site:.net OR site:.io OR site:.dev'
        
        # Perform Google search with retries
        try:
            search_results = safe_google_search(
                base_query,
                num_results=10,
                lang=search_term.language
            )
            log_entry.logs["messages"].append({
                "type": "info",
                "message": f"Found {len(search_results)} results for search term: {search_term.term}"
            })
        except Exception as e:
            log_entry.logs["messages"].append({
                "type": "error",
                "message": f"Failed to perform search for term {search_term.term}: {str(e)}"
            })
            raise
        
        total_results = len(search_results)
        processed = 0
        emails_found = 0
        valid_emails = []
        
        # Process each result
        async with aiohttp.ClientSession() as client:
            for url in search_results:
                try:
                    # Fetch and parse webpage
                    headers = {'User-Agent': UserAgent().random}
                    async with client.get(url, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Extract emails from text content
                            text_content = soup.get_text()
                            emails = extract_emails(text_content)
                            
                            # Also look for mailto links
                            mailto_links = soup.find_all('a', href=re.compile(r'^mailto:'))
                            for link in mailto_links:
                                email = link['href'].replace('mailto:', '').split('?')[0].strip()
                                if is_valid_email(email):
                                    emails.append(email)
                            
                            # Remove duplicates and validate
                            valid_emails = list(set([email for email in emails if is_valid_email(email)]))
                            
                            # Store results in database
                            for email in valid_emails:
                                # Create or get lead
                                lead = session.query(Lead).filter_by(email=email).first()
                        if not lead:
                                    lead = Lead(email=email)
                            session.add(lead)
                            session.flush()
                        
                        # Create lead source
                        lead_source = LeadSource(
                            lead_id=lead.id,
                            search_term_id=search_term.id,
                                    url=url,
                                    domain=urlparse(url).netloc,
                                    page_title=soup.title.string if soup.title else None
                        )
                        session.add(lead_source)
                        
                        # Create campaign lead
                        campaign_lead = CampaignLead(
                            campaign_id=campaign.id,
                            lead_id=lead.id,
                                    status="pending"
                        )
                        session.add(campaign_lead)
                        
                                # Update log entry
                                log_entry.leads_gathered += 1
                                
                                # Send email if campaign has email template
                                if campaign.auto_send:
                                    email_template = session.query(EmailTemplate).filter_by(
                                        campaign_id=campaign.id
                                    ).first()
                                    
                                    if email_template:
                                        # Get email settings
                                        email_settings = SettingsManager.get_email_settings(session)
                                        if email_settings:
                                            try:
                                                # Customize email content if AI customization is enabled
                        subject = email_template.subject
                        body = email_template.body_content
                        
                        if campaign.ai_customization:
                                                    # Get AI settings
                                                    ai_settings = SettingsManager.get_ai_settings(session)
                                                    if ai_settings.get('openai_api_key'):
                                                        # Get company info from webpage
                                                        company_info = {
                                                            'domain': urlparse(url).netloc,
                                                            'title': soup.title.string if soup.title else None,
                                                            'description': soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else None,
                                                            'content': text_content[:1000]  # First 1000 chars
                                                        }
                                                        
                                                        # Use AI to customize email
                                                        client = OpenAI(api_key=ai_settings['openai_api_key'])
                                                        response = client.chat.completions.create(
                                                            model=ai_settings.get('model_name', 'gpt-3.5-turbo'),
                                                            messages=[{
                                                                "role": "system",
                                                                "content": "You are an AI that customizes email content based on company information."
                                                            }, {
                                                                "role": "user",
                                                                "content": f"Customize this email for {company_info['domain']}:\nSubject: {subject}\nBody: {body}\n\nCompany info: {json.dumps(company_info)}"
                                                            }]
                                                        )
                                                        
                                                        customized_content = response.choices[0].message.content
                                                        if '\n' in customized_content:
                                                            subject, body = customized_content.split('\n', 1)
                                                            subject = subject.replace('Subject:', '').strip()
                                                            body = body.replace('Body:', '').strip()
                                                
                                                # Send email
                                                message_id, tracking_id = await send_email_ses(
                            session,
                                                    email_settings['email'],
                            lead.email,
                            subject,
                            wrap_email_body(body)
                        )
                        
                        if message_id:
                                                    # Create email campaign
                            email_campaign = EmailCampaign(
                                campaign_id=campaign.id,
                                lead_id=lead.id,
                                template_id=email_template.id,
                                status="sent",
                                sent_at=func.now(),
                                message_id=message_id,
                                                        tracking_id=tracking_id,
                                                        customized_subject=subject,
                                                        customized_content=body,
                                ai_customized=campaign.ai_customization
                            )
                            session.add(email_campaign)
                            
                                                    # Update counters
                            automation_status.emails_sent_in_current_group += 1
                                                    log_entry.emails_sent += 1
                            
                            # Check if we've reached the max emails per group
                            if automation_status.emails_sent_in_current_group >= campaign.max_emails_per_group:
                                                        break
                            
                                                    # Wait a bit between emails
                            await asyncio.sleep(2)
                    
                    except Exception as e:
                                                logger.error(f"Error sending email to {lead.email}: {str(e)}")
                                                log_entry.logs["messages"].append({
                                                    "type": "error",
                                                    "message": f"Failed to send email to {lead.email}: {str(e)}"
                                                })
                            
                            session.commit()
                            
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {str(e)}")
                    log_entry.logs["messages"].append({
                        "type": "error",
                        "message": f"Failed to process URL {url}: {str(e)}"
                    })
                        continue
                
                # Break if we've reached max emails
                if automation_status.emails_sent_in_current_group >= campaign.max_emails_per_group:
                    break
    
    except Exception as e:
        logger.error(f"Error processing search term: {str(e)}")
        raise

def is_valid_email(email: str) -> bool:
    """Validate email address format and domain"""
    try:
        # Basic validation using email_validator library
        validate_email(email)
        
        # Additional validation rules
        if len(email) > 254:  # RFC 5321
            return False
            
        # Check for common disposable email domains
        domain = email.split('@')[1].lower()
        disposable_domains = {
            'tempmail.com', 'throwawaymail.com', 'mailinator.com',
            'guerrillamail.com', 'sharklasers.com', 'spam4.me',
            'yopmail.com', 'trashmail.com', 'temp-mail.org'
        }
        if domain in disposable_domains:
            return False
            
        return True
        
    except EmailNotValidError:
        return False

async def send_email_ses(session, from_email: str, to_email: str, subject: str, body: str) -> tuple[str, str]:
    """Send email using AWS SES with proper error handling"""
    try:
        # Get AWS credentials from settings
        email_settings = session.query(EmailSettings).filter_by(
            email=from_email,
            provider='aws',
            is_active=True
        ).first()
        
        if not email_settings:
            raise ValueError(f"No active AWS email settings found for {from_email}")
        
        # Create SES client
        ses = boto3.client(
            'ses',
            aws_access_key_id=email_settings.aws_access_key_id,
            aws_secret_access_key=email_settings.aws_secret_access_key,
            region_name=email_settings.aws_region
        )
        
        # Generate tracking ID
        tracking_id = str(uuid.uuid4())
        
        # Add tracking pixel and click tracking
        tracked_body = add_tracking(body, tracking_id)
        
        # Create email message
        message = {
            'Subject': {'Data': subject},
            'Body': {'Html': {'Data': tracked_body}}
        }
        
        # Send email
        response = ses.send_email(
            Source=from_email,
            Destination={'ToAddresses': [to_email]},
            Message=message,
            ConfigurationSetName='EmailTracking'  # Must be configured in AWS SES
        )
        
        return response['MessageId'], tracking_id
            
    except Exception as e:
        logger.error(f"Error sending email via SES: {str(e)}")
        raise

def add_tracking(body: str, tracking_id: str) -> str:
    """Add tracking pixel and click tracking to email body"""
    # Add tracking pixel
    tracking_pixel = f'<img src="https://your-tracking-domain.com/pixel/{tracking_id}" width="1" height="1" />'
    
    # Add click tracking to links
    soup = BeautifulSoup(body, 'html.parser')
    for link in soup.find_all('a'):
        if link.get('href'):
            original_url = link['href']
            tracked_url = f"https://your-tracking-domain.com/click/{tracking_id}?url={urlencode({'url': original_url})}"
            link['href'] = tracked_url
    
    # Add tracking pixel at the end of the body
    tracked_body = str(soup) + tracking_pixel
    return tracked_body

def wrap_email_body(body: str) -> str:
    """Wrap email body in proper HTML structure"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 20px; font-family: Arial, sans-serif;">
        {body}
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)