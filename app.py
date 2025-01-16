from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, func, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import func
from dotenv import load_dotenv
import os
import logging
import sys
import random
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import requests
from fake_useragent import UserAgent
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

load_dotenv()

app = FastAPI(title="AutoclientAI API", version="1.0.0")

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
    raise ValueError("Database environment variables not set")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(bind=engine)

# SQLAlchemy Models
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

class SearchTermGroup(Base):
    __tablename__ = 'search_term_groups'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    email_template = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_terms = relationship("SearchTerm", back_populates="group")

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

# Initialize database
Base.metadata.create_all(bind=engine)

# Dependency for database sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility functions
def get_domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc
    except:
        return ""

def is_valid_email(email: str) -> bool:
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def extract_emails_from_html(html_content: str) -> List[str]:
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(email_pattern, text)))
    except Exception as e:
        logging.error(f"Error extracting emails: {e}")
        return []

def extract_info_from_page(soup: BeautifulSoup) -> tuple:
    try:
        name = ""
        name_candidates = soup.find_all(['h1', 'h2', 'h3', 'p'], 
            string=re.compile(r'(?i)(name|nombre|contact|contacto)'))
        if name_candidates:
            for candidate in name_candidates:
                text = candidate.get_text()
                if len(text) < 50:
                    name = text
                    break

        company = ""
        company_meta = soup.find('meta', {'property': 'og:site_name'})
        if company_meta:
            company = company_meta['content']
        else:
            company_candidates = soup.find_all(['title', 'h1'], limit=1)
            if company_candidates:
                company = company_candidates[0].get_text()

        job_title = ""
        title_candidates = soup.find_all(['p', 'div', 'span'], 
            string=re.compile(r'(?i)(manager|director|ceo|founder|owner|presidente|gerente|director)'))
        if title_candidates:
            job_title = title_candidates[0].get_text()

        return name.strip(), company.strip(), job_title.strip()
    except Exception as e:
        logging.error(f"Error extracting page info: {e}")
        return "", "", ""

def safe_google_search(query: str, num_results: int = 10, lang: str = 'es') -> List[str]:
    try:
        logging.info(f"Starting search for query: {query}")
        headers = {
            'User-Agent': UserAgent().random
        }
        search_url = f"https://www.google.com/search?q={query}&num={num_results}&hl={lang}"
        response = requests.get(search_url, headers=headers, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for g in soup.find_all('div', class_='g'):
            anchors = g.find_all('a')
            if anchors:
                link = anchors[0]['href']
                if link.startswith('http'):
                    results.append(link)
        logging.info(f"Found {len(results)} results")
        return results
    except Exception as e:
        logging.error(f"Google search error for '{query}': {str(e)}")
        return []

def save_lead(db: Session, email: str, first_name: str = None, company: str = None, 
              job_title: str = None, url: str = None) -> Optional[Lead]:
    try:
        lead = db.query(Lead).filter(Lead.email == email).first()
        if not lead:
            lead = Lead(
                email=email,
                first_name=first_name,
                company=company,
                job_title=job_title
            )
            db.add(lead)
            db.commit()
            db.refresh(lead)

        if url:
            lead_source = LeadSource(
                lead_id=lead.id,
                url=url,
                domain=get_domain_from_url(url)
            )
            db.add(lead_source)
            db.commit()

        return lead
    except Exception as e:
        logging.error(f"Database error saving lead: {str(e)}")
        db.rollback()
        return None

# API endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/projects/")
async def create_project(project_name: str, db: Session = Depends(get_db)):
    try:
        db_project = Project(project_name=project_name)
        db.add(db_project)
        db.commit()
        db.refresh(db_project)
        return db_project
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/")
async def get_projects(db: Session = Depends(get_db)):
    return db.query(Project).all()

@app.post("/campaigns/")
async def create_campaign(campaign_name: str, project_id: int, db: Session = Depends(get_db)):
    try:
        db_campaign = Campaign(campaign_name=campaign_name, project_id=project_id)
        db.add(db_campaign)
        db.commit()
        db.refresh(db_campaign)
        return db_campaign
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/campaigns/")
async def get_campaigns(db: Session = Depends(get_db)):
    return db.query(Campaign).all()

@app.post("/search")
async def manual_search(search_terms: List[str], num_results: int = 10, language: str = 'en', optimize_english: bool = False, optimize_spanish: bool = False, shuffle_keywords: bool = False, ignore_previously_fetched: bool = True, db: Session = Depends(get_db)):
    logging.info(f"Received search request: {search_terms}")
    results = []
    domains_processed = set()
    
    for term in search_terms:
        search_term = term
        if shuffle_keywords:
            words = search_term.split()
            random.shuffle(words)
            search_term = ' '.join(words)
            logging.info(f"Shuffled search term: {search_term}")

        if optimize_english or optimize_spanish:
            lang = 'english' if optimize_english else 'spanish'
            search_term = f'"{search_term}" email OR contact OR "get in touch" site:.com' if lang == 'english' else f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es'
            logging.info(f"Optimized search term: {search_term}")

        try:
            urls = safe_google_search(search_term, num_results, lang=language)
            logging.info(f"Found {len(urls)} URLs for term: {search_term}")

            for url in urls:
                logging.info(f"Processing URL: {url}")
                domain = get_domain_from_url(url)
                if ignore_previously_fetched and domain in domains_processed:
                    logging.info(f"Skipping already processed domain: {domain}")
                    continue

                try:
                    if not url.startswith(('http://', 'https://')):
                        url = 'http://' + url

                    response = requests.get(url, timeout=10, verify=False, 
                                         headers={'User-Agent': UserAgent().random})
                    response.raise_for_status()
                    html_content = response.text
                    soup = BeautifulSoup(html_content, 'html.parser')
                    emails = extract_emails_from_html(html_content)
                    logging.info(f"Found {len(emails)} emails in {url}")

                    for email in filter(is_valid_email, emails):
                        logging.info(f"Processing valid email: {email}")
                        if domain not in domains_processed:
                            name, company, job_title = extract_info_from_page(soup)
                            logging.info(f"Extracted info - Name: {name}, Company: {company}, Job: {job_title}")
                            
                            lead = save_lead(
                                db, 
                                email=email,
                                first_name=name,
                                company=company,
                                job_title=job_title,
                                url=url
                            )

                            if lead:
                                result = {
                                    "email": email,
                                    "url": url,
                                    "company": company or "",
                                    "name": name or "",
                                    "job_title": job_title or "",
                                    "source": term,
                                }
                                results.append(result)
                                domains_processed.add(domain)
                                logging.info(f"Added new lead: {email}")

                except requests.RequestException as e:
                    logging.error(f"Error processing URL {url}: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"Error in search term {term}: {str(e)}")
            continue

    logging.info(f"Search completed. Found {len(results)} total leads")
    return {
        "total_leads": len(results),
        "results": results
    }

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    total_leads = db.query(func.count(Lead.id)).scalar()
    active_campaigns = db.query(func.count(Campaign.id)).filter(Campaign.auto_send == True).scalar()
    total_emails = db.query(func.count(EmailCampaign.id)).filter(EmailCampaign.status == 'sent').scalar()
    
    return {
        "total_leads": total_leads,
        "active_campaigns": active_campaigns,
        "total_emails": total_emails
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 