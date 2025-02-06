from fastapi import FastAPI, HTTPException, Request, Depends
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
                        # Fetch and parse webpage
                        headers = {'User-Agent': UserAgent().random}
                        html_content = await fetch_url(client, url, headers)
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
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
                        try:
                            with db_session() as session:
                                for email in valid_emails:
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
                                    session.commit()
                                    
                                total_emails_found += len(valid_emails)
                        except Exception as e:
                            logger.error(f"Error storing results in database: {str(e)}")
                        
                        # Send result event
                        result = {
                            "type": "result",
                            "url": url,
                            "title": soup.title.string if soup.title else "No title",
                            "domain": urlparse(url).netloc,
                            "emails": valid_emails,
                            "total_processed": processed + 1,
                            "total": total_results,
                            "total_emails_found": emails_found
                        }
                        yield f"data: {json.dumps(result)}\n\n"
                        
                        processed += 1
                        await asyncio.sleep(0.1)  # Small delay to prevent rate limiting

    except Exception as e:
                        logger.error(f"Error processing URL {url}: {str(e)}")
                        error_result = {
                            "type": "error",
                            "url": url,
                            "message": str(e)
                        }
                        yield f"data: {json.dumps(error_result)}\n\n"
            
            # Send completion event
            completion = {
                "type": "complete",
                "total_processed": processed,
                "total_emails_found": emails_found
            }
            yield f"data: {json.dumps(completion)}\n\n"
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            error = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# New endpoint to get task logs
@app.get("/task-logs/{task_id}")
async def get_task_logs(task_id: str):
    def event_stream():
        last_id = 0
        while True:
            with db_session() as session:
                # Get existing logs
                logs = session.query(AutomationLog).filter(
                    AutomationLog.task_id == task_id
                ).order_by(AutomationLog.created_at.asc()).all()
                
                # Send new logs
                for log in logs[last_id:]:
                    yield f"data: {json.dumps(log.logs)}\n\n"
                    last_id = len(logs)
                
                # Check if task is completed
                task = session.query(TaskQueue).filter(
                    TaskQueue.task_id == task_id
                ).first()
                
                if task and task.status in ['completed', 'failed']:
                    yield "event: complete\ndata: {}\n\n"
                    break
                
                time.sleep(1)

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
            // Implementation here
        }

        // Send emails
        function sendEmails() {
            // Implementation here
        }

        // Optimize terms
        function optimizeTerms() {
            // Implementation here
        }

        // Initialize
        window.addEventListener('load', async function() {
            await loadRecentTerms();
            await loadEmailSettings();
        });
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

# Global state management
_ACTIVE_PROJECT_ID = 1
_ACTIVE_CAMPAIGN_ID = 1

def get_active_project_id():
    """Get the currently active project ID"""
    return _ACTIVE_PROJECT_ID

def get_active_campaign_id():
    """Get the currently active campaign ID"""
    return _ACTIVE_CAMPAIGN_ID

def set_active_project_id(project_id: int):
    """Set the active project ID"""
    global _ACTIVE_PROJECT_ID
    _ACTIVE_PROJECT_ID = project_id

def set_active_campaign_id(campaign_id: int):
    """Set the active campaign ID"""
    global _ACTIVE_CAMPAIGN_ID
    _ACTIVE_CAMPAIGN_ID = campaign_id

@app.post("/set-active-project")
async def set_active_project(request: Request):
    """Set the active project"""
    try:
        data = await request.json()
        project_id = data.get('project_id')
        if not project_id:
            raise HTTPException(status_code=400, detail="Project ID is required")
        
        with db_session() as session:
            project = session.query(Project).filter_by(id=project_id).first()
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            set_active_project_id(project_id)
            return JSONResponse(content={"message": f"Active project set to: {project.project_name}"})
    except Exception as e:
        logger.error(f"Error setting active project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set-active-campaign")
async def set_active_campaign(request: Request):
    """Set the active campaign"""
    try:
        data = await request.json()
        campaign_id = data.get('campaign_id')
        if not campaign_id:
            raise HTTPException(status_code=400, detail="Campaign ID is required")
        
        with db_session() as session:
            campaign = session.query(Campaign).filter_by(id=campaign_id).first()
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            set_active_campaign_id(campaign_id)
            return JSONResponse(content={"message": f"Active campaign set to: {campaign.campaign_name}"})
    except Exception as e:
        logger.error(f"Error setting active campaign: {str(e)}")
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
    try:
        with db_session() as session:
            templates = session.query(EmailTemplate).all()
            return JSONResponse(content=[{
                "id": template.id,
                "name": template.template_name,
                "subject": template.subject,
                "body": template.body_content
            } for template in templates])
    except Exception as e:
        logger.error(f"Error fetching email templates: {str(e)}")
        return JSONResponse(
            content={"templates": [], "message": "No templates found"},
            status_code=200
        )

@app.get("/email-settings")
async def get_email_settings():
    try:
        with db_session() as session:
            settings = session.query(EmailSettings).filter(EmailSettings.is_active == True).all()
            return JSONResponse(content=[{
                "id": setting.id,
                "name": setting.name,
                "email": setting.email,
                "provider": setting.provider
            } for setting in settings])
    except Exception as e:
        logger.error(f"Error fetching email settings: {str(e)}")
        return JSONResponse(
            content={"settings": [], "message": "No settings found"},
            status_code=200
        )

# Rate limiting and retry decorators
@backoff.on_exception(backoff.expo, Exception, max_tries=3)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_url(session, url, headers):
    """Fetch URL with retry logic and rate limiting."""
    async with session.get(url, headers=headers, ssl=False, timeout=10) as response:
        if response.status == 429:  # Too Many Requests
            retry_after = int(response.headers.get('Retry-After', 60))
            await asyncio.sleep(retry_after)
            raise Exception("Rate limited")
        response.raise_for_status()
        return await response.text()

def is_valid_email(email: str) -> bool:
    """Enhanced email validation."""
    if not email:
        return False
        
    # Common patterns to exclude
    invalid_patterns = [
        r".*(\.png|\.jpg|\.jpeg|\.gif|\.css|\.js)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply|no-reply|donotreply)@.*",
        r"^(email|info|contact|support|hello|hola|hi|salutations|greetings|inquiries|questions)@.*",
        r"email@email\.com",
        r".*@example\.com$",
        r".*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$",
        r".*@.*\.(local|test|example|invalid)$"
    ]
    
    typo_domains = [
        "gmil.com", "gmal.com", "gmaill.com", "gnail.com",
        "hotmai.com", "hotmal.com", "hotmial.com",
        "yaho.com", "yahooo.com"
    ]
    
    if any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns):
        return False
        
    if any(email.lower().endswith(f"@{domain}") for domain in typo_domains):
        return False
    
    try:
        # Strict email validation
        validate_email(email, check_deliverability=False)
        
        # Additional checks
        local_part, domain = email.split('@')
        if len(local_part) > 64 or len(domain) > 255:
            return False
            
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return False
            
        return True
    except EmailNotValidError:
        return False

@app.post("/ai-group-terms")
async def ai_group_search_terms():
    try:
        with db_session() as session:
            # Get AI settings from database
            ai_settings = SettingsManager.get_ai_settings(session)
            if not ai_settings.get('openai_api_key'):
                raise ValueError("OpenAI API key not found in settings")

            model = ai_settings.get('model_name', 'gpt-3.5-turbo')  # Default fallback
            
            ungrouped_terms = session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
            
            # Use settings from database
            openai_client = OpenAI(api_key=ai_settings['openai_api_key'])
            
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "system",
                    "content": "Group these marketing search terms semantically: " + ", ".join([t.term for t in ungrouped_terms])
                }]
            )
            
            # Parse AI response
            groups = {}
            for line in response.choices[0].message.content.split('\n'):
                if ':' in line:
                    group_name, terms = line.split(':', 1)
                    groups[group_name.strip()] = [t.strip() for t in terms.split(',')]
            
            # Create groups and assign terms
            created_groups = []
            for group_name, terms in groups.items():
                group = SearchTermGroup(name=group_name)
                session.add(group)
                session.commit()
                
                for term in terms:
                    st = session.query(SearchTerm).filter(SearchTerm.term.ilike(f"%{term}%")).first()
                    if st:
                        st.group_id = group.id
                created_groups.append(group_name)
            
            return JSONResponse(content={
                "message": f"Created {len(created_groups)} groups: {', '.join(created_groups)}"
            })
            
    except Exception as e:
        logger.error(f"AI grouping failed: {str(e)}")
        raise HTTPException(status_code=500, detail="AI grouping failed")

def wrap_email_body(body_content):
    """Wrap email content in a responsive HTML template"""
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
                margin: 10px 0;
            }}
            .footer {{
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        {body_content}
        <div class="footer">
            <p>This email was sent by AutoclientAI</p>
        </div>
    </body>
    </html>
    """

async def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    """Enhanced email sending function supporting both SES and SMTP"""
    if not all([from_email, to_email, subject, body]):
        logging.error("Missing required email parameters")
        return None, None

    email_settings = session.query(EmailSettings).filter_by(email=from_email, is_active=True).first()
    if not email_settings:
        logging.error(f"No active email settings found for {from_email}")
        return None, None

    # Validate email format
    if not is_valid_email(to_email) or not is_valid_email(from_email):
        logging.error(f"Invalid email format: to={to_email}, from={from_email}")
        return None, None

    tracking_id = str(uuid.uuid4())
    tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
    wrapped_body = wrap_email_body(body)
    tracked_body = wrapped_body.replace('</body>', f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/></body>')

    try:
        if email_settings.provider == 'ses':
            if not ses_client:
                if not all([email_settings.aws_access_key_id, email_settings.aws_secret_access_key, email_settings.aws_region]):
                    raise ValueError("Incomplete AWS credentials")
                
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
            if not all([email_settings.smtp_server, email_settings.smtp_port, 
                       email_settings.smtp_username, email_settings.smtp_password]):
                raise ValueError("Incomplete SMTP settings")

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
            raise ValueError(f"Unknown email provider: {email_settings.provider}")
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        raise

def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    """Save email campaign with enhanced tracking"""
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
            tracking_id=str(uuid.uuid4()),
            ai_customized=False,
            open_count=0,
            click_count=0
        )
        session.add(new_campaign)
        session.commit()
        return new_campaign
            except Exception as e:
        logging.error(f"Error saving email campaign: {str(e)}")
        session.rollback()
        return None

# Add a new endpoint to get search settings
@app.get("/search-settings")
async def get_search_settings():
    try:
        with db_session() as session:
            settings = SettingsManager.get_search_settings(session)
            return JSONResponse(content=settings)
    except Exception as e:
        logger.error(f"Error fetching search settings: {str(e)}")
        return JSONResponse(
            content={"settings": {}, "message": "No settings found"},
            status_code=200
        )

@app.get("/test-settings")
async def test_settings():
    """Test endpoint to verify all settings are loading correctly."""
    try:
        with db_session() as session:
            # Get raw settings from database
            db_settings = {
                'search': session.query(Settings).filter(
                    Settings.setting_type == 'search'
                ).all(),
                'email': session.query(Settings).filter(
                    Settings.setting_type == 'email'
                ).all(),
                'ai': session.query(Settings).filter(
                    Settings.setting_type == 'ai'
                ).all()
            }
            
            # Get settings through SettingsManager (includes defaults)
            search_settings = SettingsManager.get_search_settings(session)
            email_settings = SettingsManager.get_email_settings(session)
            ai_settings = SettingsManager.get_ai_settings(session)
            
            return JSONResponse(content={
                "database_settings": {
                    "search": [{"name": s.name, "value": s.value} for s in db_settings['search']],
                    "email": [{"name": s.name, "value": s.value} for s in db_settings['email']],
                    "ai": [{"name": s.name, "value": s.value} for s in db_settings['ai']]
                },
                "effective_settings": {
                    "search": {
                        "settings": search_settings,
                        "using_defaults": not bool(db_settings['search'])
                    },
                    "email": {
                        "settings": email_settings,
                        "using_defaults": not bool(db_settings['email'])
                    },
                    "ai": {
                        "settings": ai_settings,
                        "using_defaults": not bool(db_settings['ai'])
                    }
                }
            })
            
    except Exception as e:
        logger.error(f"Settings test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Settings test failed: {str(e)}")

@app.get("/settings")
async def get_settings():
    """Get all settings"""
    try:
        with db_session() as session:
            general_settings = session.query(Settings).filter(
                Settings.setting_type == 'general',
                Settings.is_active == True
            ).first()
            
            email_settings = session.query(EmailSettings).filter(
                EmailSettings.is_active == True
            ).all()
            
            return JSONResponse(content={
                "general_settings": general_settings.value if general_settings else {},
                "email_settings": [{
                    "id": setting.id,
                    "name": setting.name,
                    "email": setting.email,
                    "provider": setting.provider,
                    "smtp_server": setting.smtp_server if setting.provider == 'smtp' else None,
                    "smtp_port": setting.smtp_port if setting.provider == 'smtp' else None,
                    "aws_region": setting.aws_region if setting.provider == 'ses' else None
                } for setting in email_settings]
            })
    except Exception as e:
        logger.error(f"Error fetching settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching settings")

@app.post("/settings/general")
async def update_general_settings(request: Request):
    """Update general settings"""
    try:
        data = await request.json()
        with db_session() as session:
            settings = session.query(Settings).filter(
                Settings.setting_type == 'general',
                Settings.is_active == True
            ).first()
            
            if not settings:
                settings = Settings(
                    name="General Settings",
                    setting_type='general',
                    value=data
                )
                session.add(settings)
            else:
                settings.value = data
            
            session.commit()
            return JSONResponse(content={"message": "Settings updated successfully"})
    except Exception as e:
        logger.error(f"Error updating general settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating settings")

@app.post("/settings/email")
async def create_or_update_email_settings(request: Request):
    """Create or update email settings"""
    try:
        data = await request.json()
        with db_session() as session:
            if 'id' in data:
                # Update existing
                settings = session.query(EmailSettings).filter_by(id=data['id']).first()
                if not settings:
                    raise HTTPException(status_code=404, detail="Email settings not found")
            else:
                # Create new
                settings = EmailSettings()
            
            # Update fields
            for key in ['name', 'email', 'provider', 'smtp_server', 'smtp_port', 
                       'smtp_username', 'smtp_password', 'aws_access_key_id', 
                       'aws_secret_access_key', 'aws_region']:
                if key in data:
                    setattr(settings, key, data[key])
            
            session.add(settings)
            session.commit()
            
            return JSONResponse(content={"message": "Email settings saved successfully", "id": settings.id})
    except Exception as e:
        logger.error(f"Error saving email settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving email settings")

@app.delete("/settings/email/{setting_id}")
async def delete_email_settings(setting_id: int):
    """Delete email settings"""
    try:
        with db_session() as session:
            settings = session.query(EmailSettings).filter_by(id=setting_id).first()
            if not settings:
                raise HTTPException(status_code=404, detail="Email settings not found")
            
            session.delete(settings)
            session.commit()
            return JSONResponse(content={"message": "Email settings deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting email settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting email settings")

@app.post("/settings/email/test")
async def test_email_settings(request: Request):
    """Test email settings by sending a test email"""
    try:
        data = await request.json()
        setting_id = data.get('setting_id')
        test_email = data.get('test_email')
        
        if not all([setting_id, test_email]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        with db_session() as session:
            settings = session.query(EmailSettings).filter_by(id=setting_id).first()
            if not settings:
                raise HTTPException(status_code=404, detail="Email settings not found")
            
            response, tracking_id = await send_email_ses(
                session,
                settings.email,
                test_email,
                "Test Email from AutoclientAI",
                "<p>This is a test email from your AutoclientAI email settings.</p>",
                reply_to=settings.email
            )
            
            if response:
                return JSONResponse(content={"message": "Test email sent successfully"})
            else:
                raise HTTPException(status_code=500, detail="Failed to send test email")
    except Exception as e:
        logger.error(f"Error testing email settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_optimized_search_terms(session, base_terms, kb_info):
    """Generate optimized search terms using AI based on knowledge base context"""
    try:
        ai_settings = SettingsManager.get_ai_settings(session)
        if not ai_settings.get('openai_api_key'):
            raise ValueError("OpenAI API key not found in settings")

        client = OpenAI(api_key=ai_settings['openai_api_key'])
        model = ai_settings.get('model_name', 'gpt-3.5-turbo')

        # Create a detailed prompt using knowledge base info
        prompt = f"""Generate optimized search terms for lead generation based on:
        Base terms: {', '.join(base_terms)}
        
        Company Context:
        - Description: {kb_info.get('company_description', '')}
        - Target Market: {kb_info.get('company_target_market', '')}
        - Product/Service: {kb_info.get('product_description', '')}
        - Target Customer: {kb_info.get('product_target_customer', '')}
        
        Guidelines:
        1. Generate variations that are likely to find decision-makers
        2. Include industry-specific terminology
        3. Consider regional variations if applicable
        4. Focus on high-intent search terms
        5. Include role-based variations
        
        Respond with a JSON array of optimized terms."""

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": "You are an AI specializing in B2B lead generation and search term optimization."
            }, {
                "role": "user",
                "content": prompt
            }]
        )

        # Parse the response
        try:
            optimized_terms = json.loads(response.choices[0].message.content)
            if isinstance(optimized_terms, list):
                # Log the optimization for analytics
                log_search_term_optimization(session, base_terms, optimized_terms)
                return optimized_terms
            else:
                raise ValueError("Invalid response format from AI")
        except json.JSONDecodeError:
            # Fallback to simple parsing if JSON parsing fails
            terms = response.choices[0].message.content.split('\n')
            return [term.strip() for term in terms if term.strip()]

    except Exception as e:
        logger.error(f"Error generating optimized search terms: {str(e)}")
        return base_terms

def log_search_term_optimization(session, original_terms, optimized_terms):
    """Log search term optimization for analytics"""
    try:
        for original_term in original_terms:
            term = session.query(SearchTerm).filter_by(term=original_term).first()
            if term:
                for opt_term in optimized_terms:
                    session.add(OptimizedSearchTerm(
                        original_term_id=term.id,
                        term=opt_term,
                        created_at=datetime.utcnow()
                    ))
        session.commit()
    except Exception as e:
        logger.error(f"Error logging search term optimization: {str(e)}")
        session.rollback()

@app.post("/search-terms/optimize")
async def optimize_search_terms(request: Request):
    """Endpoint to optimize search terms using AI"""
    try:
        data = await request.json()
        terms = data.get('terms', [])
        if not terms:
            raise HTTPException(status_code=400, detail="No search terms provided")

        with db_session() as session:
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                raise HTTPException(status_code=404, detail="Knowledge base not found")

            optimized_terms = generate_optimized_search_terms(session, terms, kb_info)
            return JSONResponse(content={"optimized_terms": optimized_terms})

    except Exception as e:
        logger.error(f"Error optimizing search terms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_knowledge_base_info(session, project_id):
    """Get knowledge base information for a project"""
    kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb.to_dict() if kb else None

@app.get("/analytics/overview")
async def get_analytics_overview():
    """Get overview analytics for the dashboard using existing schema"""
    try:
        with db_session() as session:
            # Get total counts from existing tables
            total_leads = session.query(func.count(Lead.id)).scalar() or 0
            total_emails_sent = session.query(func.count(EmailCampaign.id)).scalar() or 0
            total_search_terms = session.query(func.count(SearchTerm.id)).scalar() or 0
            
            # Calculate email success metrics from EmailCampaign table
            email_metrics = (
                session.query(
                    func.count(EmailCampaign.id).label('total'),
                    func.sum(case((EmailCampaign.opened_at.isnot(None), 1), else_=0)).label('opened'),
                    func.sum(case((EmailCampaign.clicked_at.isnot(None), 1), else_=0)).label('clicked')
                )
            ).first()
            
            # Get recent leads with their sources
            recent_leads = (
                session.query(Lead, LeadSource)
                .outerjoin(LeadSource)
                .order_by(Lead.created_at.desc())
                .limit(5)
                .all()
            )
            
            # Get recent campaigns with engagement data
            recent_campaigns = (
                session.query(EmailCampaign)
                .order_by(EmailCampaign.sent_at.desc())
                .limit(5)
                .all()
            )
            
            # Calculate lead growth over time
            lead_growth = (
                session.query(
                    func.date_trunc('day', Lead.created_at).label('date'),
                    func.count(Lead.id).label('count')
                )
                .group_by(text('date'))
                .order_by(text('date'))
                .all()
            )
            
            return JSONResponse(content={
                "overview": {
                    "total_leads": total_leads,
                    "total_emails_sent": total_emails_sent,
                    "total_search_terms": total_search_terms,
                    "email_metrics": {
                        "total": email_metrics.total or 0,
                        "opened": email_metrics.opened or 0,
                        "clicked": email_metrics.clicked or 0,
                        "open_rate": (email_metrics.opened / email_metrics.total * 100) if email_metrics.total else 0,
                        "click_rate": (email_metrics.clicked / email_metrics.total * 100) if email_metrics.total else 0
                    }
                },
                "recent_activity": {
                    "leads": [{
                        "id": lead.id,
                        "email": lead.email,
                        "source": source.url if source else None,
                        "created_at": lead.created_at.isoformat()
                    } for lead, source in recent_leads],
                    "campaigns": [{
                        "id": campaign.id,
                        "status": campaign.status,
                        "sent_at": campaign.sent_at.isoformat() if campaign.sent_at else None,
                        "opened": campaign.opened_at is not None,
                        "clicked": campaign.clicked_at is not None
                    } for campaign in recent_campaigns]
                },
                "lead_growth": [{
                    "date": date.isoformat(),
                    "count": count
                } for date, count in lead_growth]
            })
    except Exception as e:
        logger.error(f"Error fetching analytics overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/search-terms")
async def get_search_term_analytics():
    """Get detailed analytics for search terms using existing schema"""
    try:
        with db_session() as session:
            # Get search term performance using existing tables
            search_term_stats = (
                session.query(
                    SearchTerm.term,
                    SearchTerm.category,
                    SearchTerm.language,
                    func.count(distinct(LeadSource.id)).label('sources_found'),
                    func.count(distinct(Lead.id)).label('leads_generated'),
                    func.count(distinct(EmailCampaign.id)).label('emails_sent'),
                    func.count(distinct(case((EmailCampaign.opened_at.isnot(None), EmailCampaign.id), else_=None))).label('emails_opened')
                )
                .outerjoin(LeadSource, SearchTerm.id == LeadSource.search_term_id)
                .outerjoin(Lead, LeadSource.lead_id == Lead.id)
                .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
                .group_by(SearchTerm.term, SearchTerm.category, SearchTerm.language)
                .order_by(func.count(distinct(Lead.id)).desc())
                .all()
            )
            
            # Get optimization effectiveness using OptimizedSearchTerm
            optimization_stats = (
                session.query(
                    SearchTerm.term.label('original_term'),
                    func.count(distinct(OptimizedSearchTerm.id)).label('optimized_variations'),
                    func.count(distinct(LeadSource.id)).label('sources_from_optimized'),
                    func.count(distinct(Lead.id)).label('leads_from_optimized')
                )
                .join(OptimizedSearchTerm, SearchTerm.id == OptimizedSearchTerm.original_term_id)
                .outerjoin(LeadSource, OptimizedSearchTerm.id == LeadSource.search_term_id)
                .outerjoin(Lead, LeadSource.lead_id == Lead.id)
                .group_by(SearchTerm.term)
                .order_by(func.count(distinct(Lead.id)).desc())
                .all()
            )
            
            return JSONResponse(content={
                "search_term_performance": [{
                    "term": term,
                    "category": category,
                    "language": language,
                    "sources_found": int(sources_found),
                    "leads_generated": int(leads_generated),
                    "emails_sent": int(emails_sent),
                    "emails_opened": int(emails_opened),
                    "conversion_rate": (leads_generated / sources_found * 100) if sources_found > 0 else 0,
                    "email_open_rate": (emails_opened / emails_sent * 100) if emails_sent > 0 else 0
                } for term, category, language, sources_found, leads_generated, emails_sent, emails_opened in search_term_stats],
                
                "optimization_effectiveness": [{
                    "original_term": term,
                    "optimized_variations": int(variations),
                    "sources_found": int(sources),
                    "leads_generated": int(leads),
                    "optimization_impact": (leads / variations) if variations > 0 else 0
                } for term, variations, sources, leads in optimization_stats]
            })
    except Exception as e:
        logger.error(f"Error fetching search term analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/email-campaigns")
async def get_email_campaign_analytics():
    """Get detailed analytics for email campaigns using existing schema"""
    try:
        with db_session() as session:
            # Get campaign performance metrics
            campaign_stats = (
                session.query(
                    EmailCampaign.template_id,
                    EmailTemplate.template_name.label('template_name'),
                    func.count(EmailCampaign.id).label('total_sent'),
                    func.count(case((EmailCampaign.opened_at.isnot(None), 1), else_=None)).label('total_opened'),
                    func.count(case((EmailCampaign.clicked_at.isnot(None), 1), else_=None)).label('total_clicked'),
                    func.avg(cast(cast(case((EmailCampaign.opened_at.isnot(None), 1), else_=0), Integer), Float)).label('open_rate'),
                    func.avg(cast(cast(case((EmailCampaign.clicked_at.isnot(None), 1), else_=0), Integer), Float)).label('click_rate'),
                    func.avg(cast(cast(EmailCampaign.ai_customized, Integer), Float)).label('ai_usage_rate')
                )
                .join(EmailTemplate)
                .group_by(EmailCampaign.template_id, EmailTemplate.template_name)
                .order_by(func.count(EmailCampaign.id).desc())
                .all()
            )

            # Get AI customization effectiveness
            ai_effectiveness = (
                session.query(
                    cast(EmailCampaign.ai_customized, Boolean).label('is_ai_customized'),
                    func.count(EmailCampaign.id).label('total_sent'),
                    func.count(case((EmailCampaign.opened_at.isnot(None), 1), else_=None)).label('total_opened'),
                    func.count(case((EmailCampaign.clicked_at.isnot(None), 1), else_=None)).label('total_clicked'),
                    func.avg(cast(cast(case((EmailCampaign.opened_at.isnot(None), 1), else_=0), Integer), Float)).label('open_rate'),
                    func.avg(cast(cast(case((EmailCampaign.clicked_at.isnot(None), 1), else_=0), Integer), Float)).label('click_rate')
                )
                .group_by(EmailCampaign.ai_customized)
                .all()
            )

            return JSONResponse(content={
                "campaign_performance": [{
                    "template_id": template_id,
                    "template_name": template_name,
                    "total_sent": int(total_sent),
                    "total_opened": int(total_opened),
                    "total_clicked": int(total_clicked),
                    "open_rate": float(open_rate or 0),
                    "click_rate": float(click_rate or 0),
                    "ai_usage_rate": float(ai_usage_rate or 0)
                } for template_id, template_name, total_sent, total_opened, total_clicked, open_rate, click_rate, ai_usage_rate in campaign_stats],
                
                "ai_effectiveness": [{
                    "is_ai_customized": bool(is_ai_customized),
                    "total_sent": int(total_sent),
                    "total_opened": int(total_opened),
                    "total_clicked": int(total_clicked),
                    "open_rate": float(open_rate or 0),
                    "click_rate": float(click_rate or 0)
                } for is_ai_customized, total_sent, total_opened, total_clicked, open_rate, click_rate in ai_effectiveness]
            })
    except Exception as e:
        logger.error(f"Error fetching email campaign analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk-search")
async def bulk_search(request: Request):
    """Search for multiple terms at once"""
    try:
        data = await request.json()
        terms = data.get('terms', [])
        num_results = data.get('num_results', 10)
        optimize_english = data.get('optimize_english', False)
        optimize_spanish = data.get('optimize_spanish', False)
        language = data.get('language', 'ES')

        if not terms:
            raise HTTPException(status_code=400, detail="No search terms provided")

        async def event_generator():
            total_processed = 0
            total_emails_found = 0

            for term in terms:
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
                        
                    # Perform Google search
                    search_results = list(google_search(
                        base_query, 
                        num_results=num_results,
                        lang=language
                    ))
                    
                    # Send term start event
                    yield f"data: {json.dumps({'type': 'term_start', 'term': term})}\n\n"
                    
                    # Process each result
                    async with aiohttp.ClientSession() as client:
                        for url in search_results:
                            try:
                                # Fetch and parse webpage
                                headers = {'User-Agent': UserAgent().random}
                                html_content = await fetch_url(client, url, headers)
                                soup = BeautifulSoup(html_content, 'html.parser')
                                
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
                                try:
                                    with db_session() as session:
                                        for email in valid_emails:
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
                                            session.commit()
                                            
                                        total_emails_found += len(valid_emails)
                                except Exception as e:
                                    logger.error(f"Error storing results in database: {str(e)}")
                                
                                # Send result event
                                result = {
                                    "type": "result",
                                    "term": term,
                                    "url": url,
                                    "title": soup.title.string if soup.title else "No title",
                                    "domain": urlparse(url).netloc,
                                    "emails": valid_emails
                                }
                                yield f"data: {json.dumps(result)}\n\n"
                                
                                total_processed += 1
                                await asyncio.sleep(0.1)  # Small delay to prevent rate limiting
                                
                            except Exception as e:
                                logger.error(f"Error processing URL {url}: {str(e)}")
                                error_result = {
                                    "type": "error",
                                    "term": term,
                                    "url": url,
                                    "message": str(e)
                                }
                                yield f"data: {json.dumps(error_result)}\n\n"
                    
                    # Send term completion event
                    yield f"data: {json.dumps({'type': 'term_complete', 'term': term})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Error processing term {term}: {str(e)}")
                    error = {
                        "type": "error",
                        "term": term,
                        "message": str(e)
                    }
                    yield f"data: {json.dumps(error)}\n\n"
            
            # Send final completion event
            completion = {
                "type": "complete",
                "total_processed": total_processed,
                "total_emails_found": total_emails_found,
                "terms_processed": len(terms)
            }
            yield f"data: {json.dumps(completion)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Bulk search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/automation/start")
async def start_automation():
    """Start the automation process for the active campaign"""
    try:
        with db_session() as session:
            campaign_id = get_active_campaign_id()
            if not campaign_id:
                raise HTTPException(status_code=400, detail="No active campaign selected")
            
            # Get or create automation status
            automation_status = (
                session.query(AutomationStatus)
                .filter(AutomationStatus.campaign_id == campaign_id)
                .first()
            )
            
            if not automation_status:
                automation_status = AutomationStatus(campaign_id=campaign_id)
                session.add(automation_status)
            
            automation_status.is_running = True
            automation_status.last_run_at = func.now()
            session.commit()

            # Start background task
            asyncio.create_task(run_automation(campaign_id))
            
            return {"message": "Automation started successfully"}
    except Exception as e:
        logger.error(f"Error starting automation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/automation/stop")
async def stop_automation():
    """Stop the automation process for the active campaign"""
    try:
        with db_session() as session:
            campaign_id = get_active_campaign_id()
            if not campaign_id:
                raise HTTPException(status_code=400, detail="No active campaign selected")
            
            automation_status = (
                session.query(AutomationStatus)
                .filter(AutomationStatus.campaign_id == campaign_id)
                .first()
            )
            
            if automation_status:
                automation_status.is_running = False
                session.commit()
            
            return {"message": "Automation stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping automation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/automation/status")
async def get_automation_status():
    """Get the current automation status for the active campaign"""
    try:
        with db_session() as session:
            campaign_id = get_active_campaign_id()
            if not campaign_id:
                raise HTTPException(status_code=400, detail="No active campaign selected")
            
            automation_status = (
                session.query(AutomationStatus)
                .filter(AutomationStatus.campaign_id == campaign_id)
                .first()
            )
            
            if not automation_status:
                return {
                    "is_running": False,
                    "current_search_term": None,
                    "emails_sent_in_current_group": 0,
                    "last_run_at": None
                }
            
            current_term = None
            if automation_status.current_search_term:
                current_term = {
                    "id": automation_status.current_search_term.id,
                    "term": automation_status.current_search_term.term
                }
            
            return {
                "is_running": automation_status.is_running,
                "current_search_term": current_term,
                "emails_sent_in_current_group": automation_status.emails_sent_in_current_group,
                "last_run_at": automation_status.last_run_at
            }
    except Exception as e:
        logger.error(f"Error getting automation status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_automation(campaign_id: int):
    """Background task to run the automation process"""
    try:
        while True:
            with db_session() as session:
                # Check if automation should continue
                automation_status = (
                    session.query(AutomationStatus)
                    .filter(AutomationStatus.campaign_id == campaign_id)
                    .first()
                )
                
                if not automation_status or not automation_status.is_running:
                    break
                
                # Get campaign settings
                campaign = session.query(Campaign).get(campaign_id)
                if not campaign:
                    break
                
                # Get next search term to process
                search_term = (
                    session.query(SearchTerm)
                    .filter(SearchTerm.campaign_id == campaign_id)
                    .filter(or_(
                        SearchTerm.id > automation_status.current_search_term_id,
                        automation_status.current_search_term_id.is_(None)
                    ))
                    .order_by(SearchTerm.id)
                    .first()
                )
                
                if not search_term:
                    # If we've processed all terms, start over if loop_automation is enabled
                    if campaign.loop_automation:
                        automation_status.current_search_term_id = None
                        continue
                    else:
                        automation_status.is_running = False
                        session.commit()
                        break
                
                # Update current search term
                automation_status.current_search_term_id = search_term.id
                session.commit()
                
                # Perform search and email sending
                await process_search_term(session, search_term, campaign, automation_status)
                
                # Wait for loop interval
                await asyncio.sleep(campaign.loop_interval)
    except Exception as e:
        logger.error(f"Error in automation process: {str(e)}")

async def process_search_term(session, search_term, campaign, automation_status):
    """Process a single search term in the automation"""
    try:
        # Use existing bulk search functionality
        search_params = {
            "terms": [search_term.term],
            "num_results": 10,
            "optimize_english": False,
            "optimize_spanish": search_term.language == 'ES',
            "language": search_term.language
        }
        
        # Get email template for the campaign
        email_template = (
            session.query(EmailTemplate)
            .filter(EmailTemplate.campaign_id == campaign.id)
            .first()
        )
        
        if not email_template:
            logger.error(f"No email template found for campaign {campaign.id}")
            return
            
        # Get email settings
        email_settings = SettingsManager.get_email_settings(session)
        if not email_settings:
            logger.error("No email settings configured")
            return
            
        async for event in bulk_search_generator(**search_params):
            if isinstance(event, dict) and event.get("type") == "lead":
                # Process lead and send email
                lead_data = event.get("data", {})
                if lead_data.get("email") and is_valid_email(lead_data["email"]):
                    try:
                        # Create or get lead
                        lead = (
                            session.query(Lead)
                            .filter(Lead.email == lead_data["email"])
                            .first()
                        )
                        
                        if not lead:
                            lead = Lead(
                                email=lead_data["email"],
                                company=lead_data.get("company"),
                                created_at=func.now()
                            )
                            session.add(lead)
                            session.flush()
                        
                        # Create lead source
                        lead_source = LeadSource(
                            lead_id=lead.id,
                            search_term_id=search_term.id,
                            url=lead_data.get("url"),
                            domain=lead_data.get("domain"),
                            created_at=func.now()
                        )
                        session.add(lead_source)
                        
                        # Create campaign lead
                        campaign_lead = CampaignLead(
                            campaign_id=campaign.id,
                            lead_id=lead.id,
                            status="pending",
                            created_at=func.now()
                        )
                        session.add(campaign_lead)
                        
                        # Prepare and send email
                        subject = email_template.subject
                        body = email_template.body_content
                        
                        if campaign.ai_customization:
                            # TODO: Implement AI customization logic here
                            pass
                        
                        # Send email using AWS SES
                        message_id = await send_email_ses(
                            session,
                            email_settings["email"],
                            lead.email,
                            subject,
                            wrap_email_body(body)
                        )
                        
                        if message_id:
                            # Save email campaign
                            email_campaign = EmailCampaign(
                                campaign_id=campaign.id,
                                lead_id=lead.id,
                                template_id=email_template.id,
                                status="sent",
                                sent_at=func.now(),
                                message_id=message_id,
                                ai_customized=campaign.ai_customization
                            )
                            session.add(email_campaign)
                            
                            # Update automation status
                            automation_status.emails_sent_in_current_group += 1
                            session.commit()
                            
                            # Check if we've reached the max emails per group
                            if automation_status.emails_sent_in_current_group >= campaign.max_emails_per_group:
                                automation_status.emails_sent_in_current_group = 0
                                session.commit()
                                return
                            
                            # Wait a bit between emails to avoid rate limits
                            await asyncio.sleep(2)
                    
                    except Exception as e:
                        logger.error(f"Error processing lead {lead_data.get('email')}: {str(e)}")
                        continue
    
    except Exception as e:
        logger.error(f"Error processing search term: {str(e)}")

@app.get("/automation/logs")
async def get_automation_logs():
    try:
        with db_session() as session:
            # Query automation logs with relationships
            logs = (
                session.query(AutomationLog)
                .outerjoin(Campaign)
                .outerjoin(SearchTerm)
                .order_by(AutomationLog.start_time.desc())
                .all()
            )
            
            # Format the results
            formatted_logs = []
            for log in logs:
                try:
                    log_entry = {
                        "id": log.id,
                        "campaign_name": log.campaign.campaign_name if log.campaign else None,
                        "search_term": log.search_term.term if log.search_term else None,
                        "leads_gathered": log.leads_gathered,
                        "emails_sent": log.emails_sent,
                        "start_time": log.start_time,
                        "end_time": log.end_time,
                        "status": log.status,
                        "logs": log.logs
                    }
                    formatted_logs.append(log_entry)
                except Exception as e:
                    logger.warning(f"Error formatting log entry {log.id}: {str(e)}")
                    # Add a simplified version of the log entry if there's an error
                    formatted_logs.append({
                        "id": log.id,
                        "leads_gathered": log.leads_gathered,
                        "emails_sent": log.emails_sent,
                        "start_time": log.start_time,
                        "end_time": log.end_time,
                        "status": log.status,
                        "logs": log.logs,
                        "error_formatting": str(e)
                    })
                
            return {"logs": formatted_logs}
            
    except Exception as e:
        logger.error(f"Error fetching automation logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)