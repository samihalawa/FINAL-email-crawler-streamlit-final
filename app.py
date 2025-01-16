from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv
import os
import logging
import sys
import random
from models import *
from schemas import *
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
        from googlesearch import search
        results = []
        for url in search(query, stop=num_results, lang=lang, pause=2):
            results.append(url)
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

@app.post("/projects/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    try:
        db_project = Project(**project.dict())
        db.add(db_project)
        db.commit()
        db.refresh(db_project)
        return db_project
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/", response_model=List[ProjectResponse])
async def get_projects(db: Session = Depends(get_db)):
    return db.query(Project).all()
@app.post("/campaigns/", response_model=CampaignResponse)
async def create_campaign(campaign: CampaignCreate, db: Session = Depends(get_db)):
    try:
        db_campaign = Campaign(**campaign.dict())
        db.add(db_campaign)
        db.commit()
        db.refresh(db_campaign)
        return db_campaign
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/campaigns/", response_model=List[CampaignResponse])
async def get_campaigns(db: Session = Depends(get_db)):
    return db.query(Campaign).all()

@app.post("/search", response_model=SearchResponse)
async def manual_search(request: SearchRequest, db: Session = Depends(get_db)):
    logging.info(f"Received search request: {request}")
    results = []
    domains_processed = set()
    
    for term in request.search_terms:
        search_term = term
        if request.shuffle_keywords:
            words = search_term.split()
            random.shuffle(words)
            search_term = ' '.join(words)
            logging.info(f"Shuffled search term: {search_term}")

        if request.optimize_english or request.optimize_spanish:
            lang = 'english' if request.optimize_english else 'spanish'
            search_term = f'"{search_term}" email OR contact OR "get in touch" site:.com' if lang == 'english' else f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es'
            logging.info(f"Optimized search term: {search_term}")

        try:
            urls = safe_google_search(search_term, num_results=request.num_results, lang=request.language)
            logging.info(f"Found {len(urls)} URLs for term: {search_term}")

            for url in urls:
                logging.info(f"Processing URL: {url}")
                domain = get_domain_from_url(url)
                if request.ignore_previously_fetched and domain in domains_processed:
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