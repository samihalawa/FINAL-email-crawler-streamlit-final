from fastapi import FastAPI, HTTPException, Depends, Request, Form, Response, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os, json, re, logging, asyncio, time, requests, pandas as pd
from openai import OpenAI
import boto3, uuid, aiohttp, urllib3, random, html, smtplib
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
import signal, subprocess

load_dotenv()

DB_HOST = os.getenv("SUPABASE_DB_HOST");DB_NAME = os.getenv("SUPABASE_DB_NAME");DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD");DB_PORT = os.getenv("SUPABASE_DB_PORT")
if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):raise ValueError("Missing DB env vars")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0);SessionLocal, Base = sessionmaker(bind=engine), declarative_base()

class Project(Base):
    __tablename__ = 'projects';id = Column(BigInteger, primary_key=True);project_name = Column(Text, default="Default Project")
    created_at = Column(DateTime(timezone=True), server_default=func.now());campaigns = relationship("Campaign", back_populates="project")
    knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False)
class Campaign(Base):
    __tablename__ = 'campaigns';id = Column(BigInteger, primary_key=True);campaign_name = Column(Text, default="Default Campaign")
    campaign_type = Column(Text, default="Email");project_id = Column(BigInteger, ForeignKey('projects.id'), default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now());auto_send = Column(Boolean, default=False)
    loop_automation = Column(Boolean, default=False);ai_customization = Column(Boolean, default=False)
    max_emails_per_group = Column(BigInteger, default=40);loop_interval = Column(BigInteger, default=60)
    project = relationship("Project", back_populates="campaigns");email_campaigns = relationship("EmailCampaign", back_populates="campaign")
    search_terms = relationship("SearchTerm", back_populates="campaign");campaign_leads = relationship("CampaignLead", back_populates="campaign")
class CampaignLead(Base):
    __tablename__ = 'campaign_leads';id = Column(BigInteger, primary_key=True);campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'));status = Column(Text);created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="campaign_leads");campaign = relationship("Campaign", back_populates="campaign_leads")
class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base';id = Column(BigInteger, primary_key=True);project_id = Column(BigInteger, ForeignKey('projects.id'), nullable=False)
    kb_name, kb_bio, kb_values, contact_name, contact_role, contact_email = [Column(Text) for _ in range(6)]
    company_description, company_mission, company_target_market, company_other = [Column(Text) for _ in range(4)]
    product_name, product_description, product_target_customer, product_other = [Column(Text) for _ in range(4)]
    other_context, example_email = Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now());updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    project = relationship("Project", back_populates="knowledge_base")
    def to_dict(self):return {attr: getattr(self, attr) for attr in ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']}
class Lead(Base):
    __tablename__ = 'leads';id = Column(BigInteger, primary_key=True);email = Column(Text, unique=True)
    phone, first_name, last_name, company, job_title = [Column(Text) for _ in range(5)]
    created_at = Column(DateTime(timezone=True), server_default=func.now());campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead");email_campaigns = relationship("EmailCampaign", back_populates="lead")
class EmailTemplate(Base):
    __tablename__ = 'email_templates';id = Column(BigInteger, primary_key=True);campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    template_name, subject, body_content = Column(Text), Column(Text), Column(Text);created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False);language = Column(Text, default='ES')
    campaign = relationship("Campaign");email_campaigns = relationship("EmailCampaign", back_populates="template")
class EmailCampaign(Base):
    __tablename__ = 'email_campaigns';id = Column(BigInteger, primary_key=True);campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'));template_id = Column(BigInteger, ForeignKey('email_templates.id'))
    customized_subject = Column(Text);customized_content = Column(Text);original_subject = Column(Text);original_content = Column(Text)
    status = Column(Text);engagement_data = Column(JSON);message_id = Column(Text);tracking_id = Column(Text, unique=True)
    sent_at = Column(DateTime(timezone=True));ai_customized = Column(Boolean, default=False);opened_at = Column(DateTime(timezone=True))
    clicked_at = Column(DateTime(timezone=True));open_count = Column(BigInteger, default=0);click_count = Column(BigInteger, default=0)
    campaign = relationship("Campaign", back_populates="email_campaigns");lead = relationship("Lead", back_populates="email_campaigns")
    template = relationship("EmailTemplate", back_populates="email_campaigns")
class OptimizedSearchTerm(Base):
    __tablename__ = 'optimized_search_terms';id = Column(BigInteger, primary_key=True);original_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    term = Column(Text);created_at = Column(DateTime(timezone=True), server_default=func.now())
    original_term = relationship("SearchTerm", back_populates="optimized_terms")
class SearchTermEffectiveness(Base):
    __tablename__ = 'search_term_effectiveness';id = Column(BigInteger, primary_key=True);search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    total_results, valid_leads, irrelevant_leads, blogs_found, directories_found = [Column(BigInteger) for _ in range(5)]
    created_at = Column(DateTime(timezone=True), server_default=func.now());search_term = relationship("SearchTerm", back_populates="effectiveness", uselist=False)
class SearchTermGroup(Base):
    __tablename__ = 'search_term_groups';id = Column(BigInteger, primary_key=True);name, email_template, description = Column(Text), Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now());search_terms = relationship("SearchTerm", back_populates="group")
class SearchTerm(Base):
    __tablename__ = 'search_terms';id = Column(BigInteger, primary_key=True);group_id = Column(BigInteger, ForeignKey('search_term_groups.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'));term, category = Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now());language = Column(Text, default='ES')
    group = relationship("SearchTermGroup", back_populates="search_terms");campaign = relationship("Campaign", back_populates="search_terms")
    optimized_terms = relationship("OptimizedSearchTerm", back_populates="original_term");lead_sources = relationship("LeadSource", back_populates="search_term")
    effectiveness = relationship("SearchTermEffectiveness", back_populates="search_term", uselist=False)
class LeadSource(Base):
    __tablename__ = 'lead_sources';id = Column(BigInteger, primary_key=True);lead_id = Column(BigInteger, ForeignKey('leads.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'));url, domain, page_title, meta_description, scrape_duration = [Column(Text) for _ in range(5)]
    meta_tags, phone_numbers, content, tags = [Column(Text) for _ in range(4)];http_status = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now());lead = relationship("Lead", back_populates="lead_sources")
    search_term = relationship("SearchTerm", back_populates="lead_sources")
class AIRequestLog(Base):
    __tablename__ = 'ai_request_logs';id = Column(BigInteger, primary_key=True)
    function_name, prompt, response, model_used = [Column(Text) for _ in range(4)]
    created_at = Column(DateTime(timezone=True), server_default=func.now());lead_id = Column(BigInteger, ForeignKey('leads.id'))
    email_campaign_id = Column(BigInteger, ForeignKey('email_campaigns.id'));lead = relationship("Lead");email_campaign = relationship("EmailCampaign")
class AutomationLog(Base):
    __tablename__ = 'automation_logs';id = Column(BigInteger, primary_key=True);campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'));leads_gathered, emails_sent = Column(BigInteger), Column(BigInteger)
    start_time = Column(DateTime(timezone=True), server_default=func.now());end_time = Column(DateTime(timezone=True));status, logs = Column(Text), Column(JSON)
    campaign = relationship("Campaign");search_term = relationship("SearchTerm")
class Settings(Base):
    __tablename__ = 'settings';id = Column(BigInteger, primary_key=True);name = Column(Text, nullable=False);setting_type = Column(Text, nullable=False)
    value = Column(JSON, nullable=False);created_at = Column(DateTime(timezone=True), server_default=func.now());updated_at = Column(DateTime(timezone=True), onupdate=func.now())
class EmailSettings(Base):
    __tablename__ = 'email_settings';id = Column(BigInteger, primary_key=True);name = Column(Text, nullable=False);email = Column(Text, nullable=False)
    provider = Column(Text, nullable=False);smtp_server = Column(Text);smtp_port = Column(BigInteger)
    smtp_username = Column(Text);smtp_password = Column(Text);aws_access_key_id = Column(Text);aws_secret_access_key = Column(Text);aws_region = Column(Text)

@contextmanager
def db_session():session = SessionLocal();try:yield session;session.commit()
except Exception:session.rollback();raise;finally:session.close()
app = FastAPI();origins = ["*"];app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates");app.mount("/static", StaticFiles(directory="static"), name="static")
class SettingsUpdate(BaseModel):openai_api_key: str;openai_api_base: str;openai_model: str
class EmailSettingCreate(BaseModel):name: str;email: str;provider: str;smtp_server: Optional[str] = None;smtp_port: Optional[int] = None;smtp_username: Optional[str] = None
smtp_password: Optional[str] = None;aws_access_key_id: Optional[str] = None;aws_secret_access_key: Optional[str] = None;aws_region: Optional[str] = None
class EmailSettingUpdate(BaseModel):id: int;name: str;email: str;provider: str;smtp_server: Optional[str] = None;smtp_port: Optional[int] = None
smtp_username: Optional[str] = None;smtp_password: Optional[str] = None;aws_access_key_id: Optional[str] = None;aws_secret_access_key: Optional[str] = None
aws_region: Optional[str] = None
class SearchTermsInput(BaseModel):terms: List[str];num_results: int;optimize_english: bool;optimize_spanish: bool;shuffle_keywords: bool
language: str;enable_email_sending: bool;email_template: Optional[str] = None;email_setting_option: Optional[str] = None;reply_to: Optional[str] = None
ignore_previously_fetched: Optional[bool] = None
class EmailTemplateCreate(BaseModel):template_name: str;subject: str;body_content: str;is_ai_customizable: Optional[bool] = False;language: Optional[str] = 'ES'
class EmailTemplateUpdate(BaseModel):id: int;template_name: str;subject: str;body_content: str;is_ai_customizable: Optional[bool] = False;language: Optional[str] = 'ES'
class LeadUpdate(BaseModel):id: int;email: str;first_name: Optional[str] = None;last_name: Optional[str] = None;company: Optional[str] = None;job_title: Optional[str] = None
class BulkSendInput(BaseModel):template_id: int;from_email: str;reply_to: str;send_option: str;specific_email: Optional[str] = None
selected_terms: Optional[List[str]] = None;exclude_previously_contacted: Optional[bool] = None
class ProjectCreate(BaseModel):project_name: str
class CampaignCreate(BaseModel):campaign_name: str;project_id: int
class KnowledgeBaseCreate(BaseModel):project_id: int;kb_name: Optional[str] = None;kb_bio: Optional[str] = None;kb_values: Optional[str] = None
contact_name: Optional[str] = None;contact_role: Optional[str] = None;contact_email: Optional[str] = None;company_description: Optional[str] = None
company_mission: Optional[str] = None;company_target_market: Optional[str] = None;company_other: Optional[str] = None;product_name: Optional[str] = None
product_description: Optional[str] = None;product_target_customer: Optional[str] = None;product_other: Optional[str] = None
other_context: Optional[str] = None;example_email: Optional[str] = None
class SearchTermGroupCreate(BaseModel):name: str
class SearchTermGroupUpdate(BaseModel):group_id: int;updated_terms: List[str]
class SearchTermCreate(BaseModel):term: str;campaign_id: int;group_for_new_term: Optional[str] = None
class GroupedSearchTerm(BaseModel):group_name: str;terms: List[str]
class SearchTermsGrouping(BaseModel):grouped_terms: List[GroupedSearchTerm];ungrouped_terms: List[str]

def get_domain_from_url(url):return urlparse(url).netloc
def is_valid_email(email):
    if email is None:return False
    invalid_patterns = [r".*(\.png|\.jpg|\.jpeg|\.gif|\.css|\.js)$", r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^(email|info|contact|support|hello|hola|hi|salutations|greetings|inquiries|questions)@.*", r"email@email\.com",
        r".*@example\.com$", r".*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    if any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns):return False
    if any(email.lower().endswith(f"@{domain}") for domain in typo_domains):return False
    try:validate_email(email);return True
    except EmailNotValidError:return False
def get_page_title(url):
    try:response = requests.get(url, timeout=10);soup = BeautifulSoup(response.text, 'html.parser');title = soup.title.string if soup.title else "No title found"
    return title.strip()
    except Exception as e:logging.error(f"Error getting page title for {url}: {str(e)}");return "Error fetching title"
def extract_visible_text(soup):
    for script in soup(["script", "style"]):script.extract()
    text = soup.get_text();lines = (line.strip() for line in text.splitlines());chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
    return ' '.join(chunk for chunk in chunks if chunk)
def get_page_description(html_content):
    soup = BeautifulSoup(html_content, 'html.parser');meta_desc = soup.find('meta', attrs={'name': 'description'})
    return meta_desc['content'] if meta_desc else "No description found"
def shuffle_keywords(term):words = term.split();random.shuffle(words);return ' '.join(words)
def safe_google_search(query: str, num_results: int = 10, lang: str = 'es') -> List[str]:
    try:
        logging.info(f"Starting search for query: {query}");headers = {'User-Agent': UserAgent().random}
        search_url = f"https://www.google.com/search?q={query}&num={num_results}&hl={lang}";response = requests.get(search_url, headers=headers, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser');results = []
        for g in soup.find_all('div', class_='g'):
            anchors = g.find_all('a');if anchors:link = anchors[0]['href'];if link.startswith('http'):results.append(link)
        logging.info(f"Found {len(results)} results");return results
    except Exception as e:logging.error(f"Google search error for '{query}': {str(e)}");return []
def get_knowledge_base_info(session, project_id):kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
return kb_info.to_dict() if kb_info else None
def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"Optimize and expand these search terms for lead generation:\n{', '.join(base_terms)}\n\nConsider:\n1. Relevance to business and target market\n2. Potential for high-quality leads\n3. Variations and related terms\n4. Industry-specific jargon\n\nRespond with a JSON array of optimized terms."
    response = openai_chat_completion([{"role": "system", "content": "You're an AI specializing in optimizing search terms for lead generation. Be concise and effective."}, {"role": "user", "content": prompt}], function_name="generate_optimized_search_terms")
    return response.get('optimized_terms', base_terms) if isinstance(response, dict) else base_terms
def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
    session.add(SearchTermEffectiveness(term=term, total_results=total_results, valid_leads=valid_leads, irrelevant_leads=total_results - valid_leads, blogs_found=blogs_found, directories_found=directories_found))
    session.commit()
def save_lead_source(session, lead_id, search_term_id, url, http_status, scrape_duration, page_title=None, meta_description=None, content=None, tags=None, phone_numbers=None):
    session.add(LeadSource(lead_id=lead_id, search_term_id=search_term_id, url=url, http_status=http_status, scrape_duration=scrape_duration,
    page_title=page_title or get_page_title(url), meta_description=meta_description or get_page_description(url), content=content or extract_visible_text(BeautifulSoup(requests.get(url).text, 'html.parser')), tags=tags, phone_numbers=phone_numbers))
    session.commit()
def save_lead(session, email, first_name=None, last_name=None, company=None, job_title=None, phone=None, url=None, search_term_id=None, created_at=None):
    try:
        existing_lead = session.query(Lead).filter_by(email=email).first()
        if existing_lead:
            for attr in ['first_name', 'last_name', 'company', 'job_title', 'phone', 'created_at']:
                if locals()[attr]:setattr(existing_lead, attr, locals()[attr])
            lead = existing_lead
        else:
            lead = Lead(email=email, first_name=first_name, last_name=last_name, company=company, job_title=job_title, phone=phone, created_at=created_at or datetime.utcnow())
            session.add(lead)
        session.flush()
        lead_source = LeadSource(lead_id=lead.id, url=url, search_term_id=search_term_id);session.add(lead_source)
        campaign_lead = CampaignLead(campaign_id=get_active_campaign_id(), lead_id=lead.id, status="Not Contacted", created_at=datetime.utcnow());session.add(campaign_lead)
        session.commit();return lead
    except Exception as e:logging.error(f"Error saving lead: {str(e)}");session.rollback();return None
def log_ai_request(session, function_name, prompt, response, lead_id=None, email_campaign_id=None, model_used=None):
    session.add(AIRequestLog(function_name=function_name, prompt=json.dumps(prompt), response=json.dumps(response) if response else None,
        lead_id=lead_id, email_campaign_id=email_campaign_id, model_used=model_used))
    session.commit()
def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:raise HTTPException(status_code=400, detail="OpenAI API key not set. Please configure it in the settings.")
        client = OpenAI(api_key=general_settings.value['openai_api_key']);model = general_settings.value.get('openai_model', "gpt-4o-mini")
    try:
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        result = response.choices[0].message.content
        with db_session() as session:log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id, model)
        try:return json.loads(result)
        except json.JSONDecodeError:return result
    except Exception as e:
        with db_session() as session:log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id, model)
        raise HTTPException(status_code=500, detail=f"Error in OpenAI API call: {str(e)}")
@app.get("/projects/")
async def get_projects(db: Session = Depends(db_session)):return db.query(Project).all()
@app.post("/campaigns/")
async def create_campaign(campaign_name: str, project_id: int, db: Session = Depends(db_session)):
    try:db_campaign = Campaign(campaign_name=campaign_name, project_id=project_id);db.add(db_campaign);db.commit();db.refresh(db_campaign);return db_campaign
    except Exception as e:logging.error(f"Error creating campaign: {e}");return []
@app.get("/campaigns/")
async def get_campaigns(db: Session = Depends(db_session)):return db.query(Campaign).all()
@app.post("/search")
async def manual_search(search_terms: List[str], num_results: int = 10, language: str = 'en', optimize_english: bool = False, optimize_spanish: bool = False, shuffle_keywords: bool = False, ignore_previously_fetched: bool = True, db: Session = Depends(db_session)):
    logging.info(f"Received search request: {search_terms}");results = [];domains_processed = set()
    for term in search_terms:
        search_term = term
        if shuffle_keywords:words = search_term.split();random.shuffle(words);search_term = ' '.join(words);logging.info(f"Shuffled search term: {search_term}")
        if optimize_english or optimize_spanish:
            lang = 'english' if optimize_english else 'spanish'
            search_term = f'"{search_term}" email OR contact OR "get in touch" site:.com' if lang == 'english' else f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es'
            logging.info(f"Optimized search term: {search_term}")
        try:
            urls = safe_google_search(search_term, num_results, lang=language);logging.info(f"Found {len(urls)} URLs for term: {search_term}")
            for url in urls:domain = get_domain_from_url(url);if domain not in domains_processed:results.append(url);domains_processed.add(domain);logging.info(f"Added URL: {url}")
        except Exception as e:logging.error(f"Error searching term {search_term}: {str(e)}");continue
    return {"results": results}
def fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted):
    try:
        query = session.query(Lead)
        if send_option == "Specific Email":query = query.filter(Lead.email == specific_email)
        elif send_option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and selected_terms:
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))
        if exclude_previously_contacted:
            subquery = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None)).subquery()
            query = query.outerjoin(subquery, Lead.id == subquery.c.lead_id).filter(subquery.c.lead_id.is_(None))
        return [{"Email": lead.email, "ID": lead.id} for lead in query.all()]
    except Exception as e:logging.error(f"Error fetching leads: {str(e)}");return []
def fetch_leads_with_sources(session):
    try:
        leads = session.query(Lead).options(joinedload(Lead.lead_sources)).all();leads_data = []
        for lead in leads:
            lead_info = {"id": lead.id, "email": lead.email, "first_name": lead.first_name, "last_name": lead.last_name,
                "company": lead.company, "job_title": lead.job_title, "created_at": lead.created_at.strftime('%Y-%m-%d %H:%M:%S') if lead.created_at else None,
                "sources": [{"url": source.url, "search_term": source.search_term.term if source.search_term else "N/A"} for source in lead.lead_sources]}
            leads_data.append(lead_info)
        return pd.DataFrame(leads_data)
    except SQLAlchemyError as e:logging.error(f"Database error in fetch_leads_with_sources: {str(e)}");return pd.DataFrame()
def fetch_all_email_logs(session):
    try:
        email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({'ID': [ec.id for ec in email_campaigns], 'Sent At': [ec.sent_at for ec in email_campaigns], 'Email': [ec.lead.email for ec in email_campaigns],
            'Template': [ec.template.template_name for ec in email_campaigns], 'Subject': [ec.customized_subject or "No subject" for ec in email_campaigns],
            'Content': [ec.customized_content or "No content" for ec in email_campaigns], 'Status': [ec.status for ec in email_campaigns],
            'Message ID': [ec.message_id or "No message ID" for ec in email_campaigns], 'Campaign ID': [ec.campaign_id for ec in email_campaigns],
            'Lead Name': [f"{ec.lead.first_name or ''} {ec.lead.last_name or ''}".strip() or "Unknown" for ec in email_campaigns],
            'Lead Company': [ec.lead.company or "Unknown" for ec in email_campaigns]})
    except SQLAlchemyError as e:logging.error(f"Database error in fetch_all_email_logs: {str(e)}");return pd.DataFrame()
def fetch_sent_email_campaigns(session):
    try:
        email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({'ID': [ec.id for ec in email_campaigns],
            'Sent At': [ec.sent_at.strftime("%Y-%m-%d %H:%M:%S") if ec.sent_at else "" for ec in email_campaigns], 'Email': [ec.lead.email for ec in email_campaigns],
            'Template': [ec.template.template_name for ec in email_campaigns], 'Subject': [ec.customized_subject or "No subject" for ec in email_campaigns],
            'Content': [ec.customized_content or "No content" for ec in email_campaigns], 'Status': [ec.status for ec in email_campaigns],
            'Message ID': [ec.message_id or "No message ID" for ec in email_campaigns], 'Campaign ID': [ec.campaign_id for ec in email_campaigns],
            'Lead Name': [f"{ec.lead.first_name or ''} {ec.lead.last_name or ''}".strip() or "Unknown" for ec in email_campaigns],
            'Lead Company': [ec.lead.company or "Unknown" for ec in email_campaigns]})
    except SQLAlchemyError as e:logging.error(f"Database error in fetch_sent_email_campaigns: {str(e)}");return pd.DataFrame()
def get_latest_logs(automation_log_id):
    with db_session() as session:log = session.query(AutomationLog).get(automation_log_id);return log.logs if log else []
def is_process_running(pid):
    try:os.kill(pid, 0);return True
from fastapi import FastAPI, HTTPException, Depends, Request, Form, Response, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os, json, re, logging, asyncio, time, requests, pandas as pd
from openai import OpenAI
import boto3, uuid, aiohttp, urllib3, random, html, smtplib
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
import signal, subprocess

load_dotenv()

DB_HOST = os.getenv("SUPABASE_DB_HOST");DB_NAME = os.getenv("SUPABASE_DB_NAME");DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD");DB_PORT = os.getenv("SUPABASE_DB_PORT")
if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):raise ValueError("Missing DB env vars")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0);SessionLocal, Base = sessionmaker(bind=engine), declarative_base()

class Project(Base):
    __tablename__ = 'projects';id = Column(BigInteger, primary_key=True);project_name = Column(Text, default="Default Project")
    created_at = Column(DateTime(timezone=True), server_default=func.now());campaigns = relationship("Campaign", back_populates="project")
    knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False)
class Campaign(Base):
    __tablename__ = 'campaigns';id = Column(BigInteger, primary_key=True);campaign_name = Column(Text, default="Default Campaign")
    campaign_type = Column(Text, default="Email");project_id = Column(BigInteger, ForeignKey('projects.id'), default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now());auto_send = Column(Boolean, default=False)
    loop_automation = Column(Boolean, default=False);ai_customization = Column(Boolean, default=False)
    max_emails_per_group = Column(BigInteger, default=40);loop_interval = Column(BigInteger, default=60)
    project = relationship("Project", back_populates="campaigns");email_campaigns = relationship("EmailCampaign", back_populates="campaign")
    search_terms = relationship("SearchTerm", back_populates="campaign");campaign_leads = relationship("CampaignLead", back_populates="campaign")
class CampaignLead(Base):
    __tablename__ = 'campaign_leads';id = Column(BigInteger, primary_key=True);campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'));status = Column(Text);created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="campaign_leads");campaign = relationship("Campaign", back_populates="campaign_leads")
class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base';id = Column(BigInteger, primary_key=True);project_id = Column(BigInteger, ForeignKey('projects.id'), nullable=False)
    kb_name, kb_bio, kb_values, contact_name, contact_role, contact_email = [Column(Text) for _ in range(6)]
    company_description, company_mission, company_target_market, company_other = [Column(Text) for _ in range(4)]
    product_name, product_description, product_target_customer, product_other = [Column(Text) for _ in range(4)]
    other_context, example_email = Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now());updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    project = relationship("Project", back_populates="knowledge_base")
    def to_dict(self):return {attr: getattr(self, attr) for attr in ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']}
class Lead(Base):
    __tablename__ = 'leads';id = Column(BigInteger, primary_key=True);email = Column(Text, unique=True)
    phone, first_name, last_name, company, job_title = [Column(Text) for _ in range(5)]
    created_at = Column(DateTime(timezone=True), server_default=func.now());campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead");email_campaigns = relationship("EmailCampaign", back_populates="lead")
class EmailTemplate(Base):
    __tablename__ = 'email_templates';id = Column(BigInteger, primary_key=True);campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    template_name, subject, body_content = Column(Text), Column(Text), Column(Text);created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False);language = Column(Text, default='ES')
    campaign = relationship("Campaign");email_campaigns = relationship("EmailCampaign", back_populates="template")
class EmailCampaign(Base):
    __tablename__ = 'email_campaigns';id = Column(BigInteger, primary_key=True);campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'));template_id = Column(BigInteger, ForeignKey('email_templates.id'))
    customized_subject = Column(Text);customized_content = Column(Text);original_subject = Column(Text);original_content = Column(Text)
    status = Column(Text);engagement_data = Column(JSON);message_id = Column(Text);tracking_id = Column(Text, unique=True)
    sent_at = Column(DateTime(timezone=True));ai_customized = Column(Boolean, default=False);opened_at = Column(DateTime(timezone=True))
    clicked_at = Column(DateTime(timezone=True));open_count = Column(BigInteger, default=0);click_count = Column(BigInteger, default=0)
    campaign = relationship("Campaign", back_populates="email_campaigns");lead = relationship("Lead", back_populates="email_campaigns")
    template = relationship("EmailTemplate", back_populates="email_campaigns")
class OptimizedSearchTerm(Base):
    __tablename__ = 'optimized_search_terms';id = Column(BigInteger, primary_key=True);original_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    term = Column(Text);created_at = Column(DateTime(timezone=True), server_default=func.now())
    original_term = relationship("SearchTerm", back_populates="optimized_terms")
class SearchTermEffectiveness(Base):
    __tablename__ = 'search_term_effectiveness';id = Column(BigInteger, primary_key=True);search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    total_results, valid_leads, irrelevant_leads, blogs_found, directories_found = [Column(BigInteger) for _ in range(5)]
    created_at = Column(DateTime(timezone=True), server_default=func.now());search_term = relationship("SearchTerm", back_populates="effectiveness", uselist=False)
class SearchTermGroup(Base):
    __tablename__ = 'search_term_groups';id = Column(BigInteger, primary_key=True);name, email_template, description = Column(Text), Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now());search_terms = relationship("SearchTerm", back_populates="group")
class SearchTerm(Base):
    __tablename__ = 'search_terms';id = Column(BigInteger, primary_key=True);group_id = Column(BigInteger, ForeignKey('search_term_groups.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'));term, category = Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now());language = Column(Text, default='ES')
    group = relationship("SearchTermGroup", back_populates="search_terms");campaign = relationship("Campaign", back_populates="search_terms")
    optimized_terms = relationship("OptimizedSearchTerm", back_populates="original_term");lead_sources = relationship("LeadSource", back_populates="search_term")
    effectiveness = relationship("SearchTermEffectiveness", back_populates="search_term", uselist=False)
class LeadSource(Base):
    __tablename__ = 'lead_sources';id = Column(BigInteger, primary_key=True);lead_id = Column(BigInteger, ForeignKey('leads.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'));url, domain, page_title, meta_description, scrape_duration = [Column(Text) for _ in range(5)]
    meta_tags, phone_numbers, content, tags = [Column(Text) for _ in range(4)];http_status = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now());lead = relationship("Lead", back_populates="lead_sources")
    search_term = relationship("SearchTerm", back_populates="lead_sources")
class AIRequestLog(Base):
    __tablename__ = 'ai_request_logs';id = Column(BigInteger, primary_key=True)
    function_name, prompt, response, model_used = [Column(Text) for _ in range(4)]
    created_at = Column(DateTime(timezone=True), server_default=func.now());lead_id = Column(BigInteger, ForeignKey('leads.id'))
    email_campaign_id = Column(BigInteger, ForeignKey('email_campaigns.id'));lead = relationship("Lead");email_campaign = relationship("EmailCampaign")
class AutomationLog(Base):
    __tablename__ = 'automation_logs';id = Column(BigInteger, primary_key=True);campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'));leads_gathered, emails_sent = Column(BigInteger), Column(BigInteger)
    start_time = Column(DateTime(timezone=True), server_default=func.now());end_time = Column(DateTime(timezone=True));status, logs = Column(Text), Column(JSON)
    campaign = relationship("Campaign");search_term = relationship("SearchTerm")
class Settings(Base):
    __tablename__ = 'settings';id = Column(BigInteger, primary_key=True);name = Column(Text, nullable=False);setting_type = Column(Text, nullable=False)
    value = Column(JSON, nullable=False);created_at = Column(DateTime(timezone=True), server_default=func.now());updated_at = Column(DateTime(timezone=True), onupdate=func.now())
class EmailSettings(Base):
    __tablename__ = 'email_settings';id = Column(BigInteger, primary_key=True);name = Column(Text, nullable=False);email = Column(Text, nullable=False)
    provider = Column(Text, nullable=False);smtp_server = Column(Text);smtp_port = Column(BigInteger)
    smtp_username = Column(Text);smtp_password = Column(Text);aws_access_key_id = Column(Text);aws_secret_access_key = Column(Text);aws_region = Column(Text)

@contextmanager
def db_session():session = SessionLocal();try:yield session;session.commit()
except Exception:session.rollback();raise;finally:session.close()
app = FastAPI();origins = ["*"];app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates");app.mount("/static", StaticFiles(directory="static"), name="static")
class SettingsUpdate(BaseModel):openai_api_key: str;openai_api_base: str;openai_model: str
class EmailSettingCreate(BaseModel):name: str;email: str;provider: str;smtp_server: Optional[str] = None;smtp_port: Optional[int] = None;smtp_username: Optional[str] = None
smtp_password: Optional[str] = None;aws_access_key_id: Optional[str] = None;aws_secret_access_key: Optional[str] = None;aws_region: Optional[str] = None
class EmailSettingUpdate(BaseModel):id: int;name: str;email: str;provider: str;smtp_server: Optional[str] = None;smtp_port: Optional[int] = None
smtp_username: Optional[str] = None;smtp_password: Optional[str] = None;aws_access_key_id: Optional[str] = None;aws_secret_access_key: Optional[str] = None
aws_region: Optional[str] = None
class SearchTermsInput(BaseModel):terms: List[str];num_results: int;optimize_english: bool;optimize_spanish: bool;shuffle_keywords: bool
language: str;enable_email_sending: bool;email_template: Optional[str] = None;email_setting_option: Optional[str] = None;reply_to: Optional[str] = None
ignore_previously_fetched: Optional[bool] = None
class EmailTemplateCreate(BaseModel):template_name: str;subject: str;body_content: str;is_ai_customizable: Optional[bool] = False;language: Optional[str] = 'ES'
class EmailTemplateUpdate(BaseModel):id: int;template_name: str;subject: str;body_content: str;is_ai_customizable: Optional[bool] = False;language: Optional[str] = 'ES'
class LeadUpdate(BaseModel):id: int;email: str;first_name: Optional[str] = None;last_name: Optional[str] = None;company: Optional[str] = None;job_title: Optional[str] = None
class BulkSendInput(BaseModel):template_id: int;from_email: str;reply_to: str;send_option: str;specific_email: Optional[str] = None
selected_terms: Optional[List[str]] = None;exclude_previously_contacted: Optional[bool] = None
class ProjectCreate(BaseModel):project_name: str
class CampaignCreate(BaseModel):campaign_name: str;project_id: int
class KnowledgeBaseCreate(BaseModel):project_id: int;kb_name: Optional[str] = None;kb_bio: Optional[str] = None;kb_values: Optional[str] = None
contact_name: Optional[str] = None;contact_role: Optional[str] = None;contact_email: Optional[str] = None;company_description: Optional[str] = None
company_mission: Optional[str] = None;company_target_market: Optional[str] = None;company_other: Optional[str] = None;product_name: Optional[str] = None
product_description: Optional[str] = None;product_target_customer: Optional[str] = None;product_other: Optional[str] = None
other_context: Optional[str] = None;example_email: Optional[str] = None
class SearchTermGroupCreate(BaseModel):name: str
class SearchTermGroupUpdate(BaseModel):group_id: int;updated_terms: List[str]
class SearchTermCreate(BaseModel):term: str;campaign_id: int;group_for_new_term: Optional[str] = None
class GroupedSearchTerm(BaseModel):group_name: str;terms: List[str]
class SearchTermsGrouping(BaseModel):grouped_terms: List[GroupedSearchTerm];ungrouped_terms: List[str]

def get_domain_from_url(url):return urlparse(url).netloc
def is_valid_email(email):
    if email is None:return False
    invalid_patterns = [r".*(\.png|\.jpg|\.jpeg|\.gif|\.css|\.js)$", r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^(email|info|contact|support|hello|hola|hi|salutations|greetings|inquiries|questions)@.*", r"email@email\.com",
        r".*@example\.com$", r".*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    if any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns):return False
    if any(email.lower().endswith(f"@{domain}") for domain in typo_domains):return False
    try:validate_email(email);return True
    except EmailNotValidError:return False
def get_page_title(url):
    try:response = requests.get(url, timeout=10);soup = BeautifulSoup(response.text, 'html.parser');title = soup.title.string if soup.title else "No title found"
    return title.strip()
    except Exception as e:logging.error(f"Error getting page title for {url}: {str(e)}");return "Error fetching title"
def extract_visible_text(soup):
    for script in soup(["script", "style"]):script.extract()
    text = soup.get_text();lines = (line.strip() for line in text.splitlines());chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
    return ' '.join(chunk for chunk in chunks if chunk)
def get_page_description(html_content):
    soup = BeautifulSoup(html_content, 'html.parser');meta_desc = soup.find('meta', attrs={'name': 'description'})
    return meta_desc['content'] if meta_desc else "No description found"
def shuffle_keywords(term):words = term.split();random.shuffle(words);return ' '.join(words)
def safe_google_search(query: str, num_results: int = 10, lang: str = 'es') -> List[str]:
    try:
        logging.info(f"Starting search for query: {query}");headers = {'User-Agent': UserAgent().random}
        search_url = f"https://www.google.com/search?q={query}&num={num_results}&hl={lang}";response = requests.get(search_url, headers=headers, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser');results = []
        for g in soup.find_all('div', class_='g'):
            anchors = g.find_all('a');if anchors:link = anchors[0]['href'];if link.startswith('http'):results.append(link)
        logging.info(f"Found {len(results)} results");return results
    except Exception as e:logging.error(f"Google search error for '{query}': {str(e)}");return []
def get_knowledge_base_info(session, project_id):kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
return kb_info.to_dict() if kb_info else None
def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"Optimize and expand these search terms for lead generation:\n{', '.join(base_terms)}\n\nConsider:\n1. Relevance to business and target market\n2. Potential for high-quality leads\n3. Variations and related terms\n4. Industry-specific jargon\n\nRespond with a JSON array of optimized terms."
    response = openai_chat_completion([{"role": "system", "content": "You're an AI specializing in optimizing search terms for lead generation. Be concise and effective."}, {"role": "user", "content": prompt}], function_name="generate_optimized_search_terms")
    return response.get('optimized_terms', base_terms) if isinstance(response, dict) else base_terms
def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
    session.add(SearchTermEffectiveness(term=term, total_results=total_results, valid_leads=valid_leads, irrelevant_leads=total_results - valid_leads, blogs_found=blogs_found, directories_found=directories_found))
    session.commit()
def save_lead_source(session, lead_id, search_term_id, url, http_status, scrape_duration, page_title=None, meta_description=None, content=None, tags=None, phone_numbers=None):
    session.add(LeadSource(lead_id=lead_id, search_term_id=search_term_id, url=url, http_status=http_status, scrape_duration=scrape_duration,
    page_title=page_title or get_page_title(url), meta_description=meta_description or get_page_description(url), content=content or extract_visible_text(BeautifulSoup(requests.get(url).text, 'html.parser')), tags=tags, phone_numbers=phone_numbers))
    session.commit()
def save_lead(session, email, first_name=None, last_name=None, company=None, job_title=None, phone=None, url=None, search_term_id=None, created_at=None):
    try:
        existing_lead = session.query(Lead).filter_by(email=email).first()
        if existing_lead:
            for attr in ['first_name', 'last_name', 'company', 'job_title', 'phone', 'created_at']:
                if locals()[attr]:setattr(existing_lead, attr, locals()[attr])
            lead = existing_lead
        else:
            lead = Lead(email=email, first_name=first_name, last_name=last_name, company=company, job_title=job_title, phone=phone, created_at=created_at or datetime.utcnow())
            session.add(lead)
        session.flush()
        lead_source = LeadSource(lead_id=lead.id, url=url, search_term_id=search_term_id);session.add(lead_source)
        campaign_lead = CampaignLead(campaign_id=get_active_campaign_id(), lead_id=lead.id, status="Not Contacted", created_at=datetime.utcnow());session.add(campaign_lead)
        session.commit();return lead
    except Exception as e:logging.error(f"Error saving lead: {str(e)}");session.rollback();return None
def log_ai_request(session, function_name, prompt, response, lead_id=None, email_campaign_id=None, model_used=None):
    session.add(AIRequestLog(function_name=function_name, prompt=json.dumps(prompt), response=json.dumps(response) if response else None,
        lead_id=lead_id, email_campaign_id=email_campaign_id, model_used=model_used))
    session.commit()
def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:raise HTTPException(status_code=400, detail="OpenAI API key not set. Please configure it in the settings.")
        client = OpenAI(api_key=general_settings.value['openai_api_key']);model = general_settings.value.get('openai_model', "gpt-4o-mini")
    try:
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        result = response.choices[0].message.content
        with db_session() as session:log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id, model)
        try:return json.loads(result)
        except json.JSONDecodeError:return result
    except Exception as e:
        with db_session() as session:log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id, model)
        raise HTTPException(status_code=500, detail=f"Error in OpenAI API call: {str(e)}")
@app.get("/projects/")
async def get_projects(db: Session = Depends(db_session)):return db.query(Project).all()
@app.post("/campaigns/")
async def create_campaign(campaign_name: str, project_id: int, db: Session = Depends(db_session)):
    try:db_campaign = Campaign(campaign_name=campaign_name, project_id=project_id);db.add(db_campaign);db.commit();db.refresh(db_campaign);return db_campaign
    except Exception as e:logging.error(f"Error creating campaign: {e}");return []
@app.get("/campaigns/")
async def get_campaigns(db: Session = Depends(db_session)):return db.query(Campaign).all()
@app.post("/search")
async def manual_search(search_terms: List[str], num_results: int = 10, language: str = 'en', optimize_english: bool = False, optimize_spanish: bool = False, shuffle_keywords: bool = False, ignore_previously_fetched: bool = True, db: Session = Depends(db_session)):
    logging.info(f"Received search request: {search_terms}");results = [];domains_processed = set()
    for term in search_terms:
        search_term = term
        if shuffle_keywords:words = search_term.split();random.shuffle(words);search_term = ' '.join(words);logging.info(f"Shuffled search term: {search_term}")
        if optimize_english or optimize_spanish:
            lang = 'english' if optimize_english else 'spanish'
            search_term = f'"{search_term}" email OR contact OR "get in touch" site:.com' if lang == 'english' else f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es'
            logging.info(f"Optimized search term: {search_term}")
        try:
            urls = safe_google_search(search_term, num_results, lang=language);logging.info(f"Found {len(urls)} URLs for term: {search_term}")
            for url in urls:domain = get_domain_from_url(url);if domain not in domains_processed:results.append(url);domains_processed.add(domain);logging.info(f"Added URL: {url}")
        except Exception as e:logging.error(f"Error searching term {search_term}: {str(e)}");continue
    return {"results": results}
def fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted):
    try:
        query = session.query(Lead)
        if send_option == "Specific Email":query = query.filter(Lead.email == specific_email)
        elif send_option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and selected_terms:
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))
        if exclude_previously_contacted:
            subquery = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None)).subquery()
            query = query.outerjoin(subquery, Lead.id == subquery.c.lead_id).filter(subquery.c.lead_id.is_(None))
        return [{"Email": lead.email, "ID": lead.id} for lead in query.all()]
    except Exception as e:logging.error(f"Error fetching leads: {str(e)}");return []
def fetch_leads_with_sources(session):
    try:
        leads = session.query(Lead).options(joinedload(Lead.lead_sources)).all();leads_data = []
        for lead in leads:
            lead_info = {"id": lead.id, "email": lead.email, "first_name": lead.first_name, "last_name": lead.last_name,
                "company": lead.company, "job_title": lead.job_title, "created_at": lead.created_at.strftime('%Y-%m-%d %H:%M:%S') if lead.created_at else None,
                "sources": [{"url": source.url, "search_term": source.search_term.term if source.search_term else "N/A"} for source in lead.lead_sources]}
            leads_data.append(lead_info)
        return pd.DataFrame(leads_data)
    except SQLAlchemyError as e:logging.error(f"Database error in fetch_leads_with_sources: {str(e)}");return pd.DataFrame()
def fetch_all_email_logs(session):
    try:
        email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({'ID': [ec.id for ec in email_campaigns], 'Sent At': [ec.sent_at for ec in email_campaigns], 'Email': [ec.lead.email for ec in email_campaigns],
            'Template': [ec.template.template_name for ec in email_campaigns], 'Subject': [ec.customized_subject or "No subject" for ec in email_campaigns],
            'Content': [ec.customized_content or "No content" for ec in email_campaigns], 'Status': [ec.status for ec in email_campaigns],
            'Message ID': [ec.message_id or "No message ID" for ec in email_campaigns], 'Campaign ID': [ec.campaign_id for ec in email_campaigns],
            'Lead Name': [f"{ec.lead.first_name or ''} {ec.lead.last_name or ''}".strip() or "Unknown" for ec in email_campaigns],
            'Lead Company': [ec.lead.company or "Unknown" for ec in email_campaigns]})
    except SQLAlchemyError as e:logging.error(f"Database error in fetch_all_email_logs: {str(e)}");return pd.DataFrame()
def fetch_sent_email_campaigns(session):
    try:
        email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({'ID': [ec.id for ec in email_campaigns],
            'Sent At': [ec.sent_at.strftime("%Y-%m-%d %H:%M:%S") if ec.sent_at else "" for ec in email_campaigns], 'Email': [ec.lead.email for ec in email_campaigns],
            'Template': [ec.template.template_name for ec in email_campaigns], 'Subject': [ec.customized_subject or "No subject" for ec in email_campaigns],
            'Content': [ec.customized_content or "No content" for ec in email_campaigns], 'Status': [ec.status for ec in email_campaigns],
            'Message ID': [ec.message_id or "No message ID" for ec in email_campaigns], 'Campaign ID': [ec.campaign_id for ec in email_campaigns],
            'Lead Name': [f"{ec.lead.first_name or ''} {ec.lead.last_name or ''}".strip() or "Unknown" for ec in email_campaigns],
            'Lead Company': [ec.lead.company or "Unknown" for ec in email_campaigns]})
    except SQLAlchemyError as e:logging.error(f"Database error in fetch_sent_email_campaigns: {str(e)}");return pd.DataFrame()
def get_latest_logs(automation_log_id):
    with db_session() as session:log = session.query(AutomationLog).get(automation_log_id);return log.logs if log else []
def is_process_running(pid):
    try:os.kill(pid, 0);return True
 base_terms) if isinstance(response, dict) else base_terms
def bulk_send_emails(session, template_id, from_email, reply_to, leads):
    logs = [];sent_count = 0;email_template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not email_template:logs.append("Error: Email template not found.");return logs, sent_count
    for lead in leads:
        try:
            email = lead["Email"];lead_id = lead["ID"]
            if email_template.is_ai_customizable:customized_subject, customized_content = customize_email_with_ai(session, email_template, lead_id)
            else:customized_subject = email_template.subject;customized_content = email_template.body_content
            send_email_ses(session, from_email, email, customized_subject, customized_content, reply_to=reply_to)
            save_email_campaign(session, lead_id, template_id, 'sent', datetime.utcnow(), customized_subject, None, customized_content)
            sent_count += 1;logs.append(f"Email sent to {email} at {datetime.utcnow()}")
        except Exception as e:logs.append(f"Error sending email to {email}: {str(e)}")
    return logs, sent_count
def customize_email_with_ai(session, email_template, lead_id):
    lead = session.query(Lead).filter_by(id=lead_id).first()
    if not lead:raise ValueError(f"Lead with ID {lead_id} not found.")
    kb_info = get_knowledge_base_info(session, active_project_id)
    prompt = f"Original Email:\nSubject: {email_template.subject}\nContent: {email_template.body_content}\n\n"
    prompt += f"Lead Info:\nEmail: {lead.email}\n"
    if lead.first_name:prompt += f"First Name: {lead.first_name}\n"
    if lead.last_name:prompt += f"Last Name: {lead.last_name}\n"
    if lead.company:prompt += f"Company: {lead.company}\n"
    if lead.job_title:prompt += f"Job Title: {lead.job_title}\n"
    if kb_info:prompt += f"Knowledge Base Info:\n{json.dumps(kb_info)}\n\n"
    prompt += "Please provide the AI-customized email subject and content in the following JSON format:\n"
    prompt += '{"subject": "Customized Subject", "body_content": "Customized Body Content"}'
    try:
        response = openai_chat_completion([
            {"role": "system", "content": "You are a helpful assistant that customizes email templates based on lead information and knowledge base."},
            {"role": "user", "content": prompt}
        ], function_name="customize_email_with_ai", lead_id=lead_id, email_campaign_id=None)
        if isinstance(response, dict) and "subject" in response and "body_content" in response:return response["subject"], response["body_content"]
        else:logging.error(f"Unexpected response format from AI: {response}");return email_template.subject, email_template.body_content
    except Exception as e:logging.error(f"Error customizing email with AI: {e}");return email_template.subject, email_template.body_content
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
def extract_emails_from_html(html_content):
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", html_content)
    return [email for email in emails if is_valid_email(email)]
def extract_info_from_page(soup):
    name_elements = soup.find_all(class_=re.compile(r'name|author|user', re.IGNORECASE))
    company_elements = soup.find_all(class_=re.compile(r'company|business|org', re.IGNORECASE))
    job_elements = soup.find_all(class_=re.compile(r'job|title|role', re.IGNORECASE))
    name = ' '.join([el.get_text(strip=True) for el in name_elements]) if name_elements else None
    company = ' '.join([el.get_text(strip=True) for el in company_elements]) if company_elements else None
    job_title = ' '.join([el.get_text(strip=True) for el in job_elements]) if job_elements else None
    return name, company, job_title
def fetch_email_settings(db):
    email_settings = db.query(EmailSettings).all()
    return [{"id": setting.id, "name": setting.name, "email": setting.email, "provider": setting.provider,
             "smtp_server": setting.smtp_server, "smtp_port": setting.smtp_port, "smtp_username": setting.smtp_username,
             "smtp_password": setting.smtp_password, "aws_access_key_id": setting.aws_access_key_id,
             "aws_secret_access_key": setting.aws_secret_access_key, "aws_region": setting.aws_region} for setting in email_settings]
def save_email_campaign(session, lead_id, template_id, status, sent_at, subject, message_id, content):
    tracking_id = str(uuid.uuid4())
    email_campaign = EmailCampaign(lead_id=lead_id, template_id=template_id, status=status, sent_at=sent_at, customized_subject=subject,
        customized_content=content, message_id=message_id, tracking_id=tracking_id, campaign_id=get_active_campaign_id())
    session.add(email_campaign);session.commit()
def send_email_ses(session, sender_email, recipient_email, subject, body_text, reply_to=None):
    email_settings = session.query(EmailSettings).first()
    if not email_settings or not email_settings.aws_access_key_id or not email_settings.aws_secret_access_key or not email_settings.aws_region:
        logging.error("Email settings not configured for AWS SES.");return None
    try:
        client = boto3.client('ses', region_name=email_settings.aws_region,
            aws_access_key_id=email_settings.aws_access_key_id, aws_secret_access_key=email_settings.aws_secret_access_key)
        message = MIMEMultipart('alternative')
        message['Subject'] = subject
        message['From'] = sender_email
        message['To'] = recipient_email
        if reply_to:message['Reply-To'] = reply_to
        part1 = MIMEText(body_text, 'plain')
        part2 = MIMEText(wrap_email_body(body_text), 'html')
        message.attach(part1);message.attach(part2)
        response = client.send_raw_email(
            Source=sender_email,
            Destinations=[recipient_email],
            RawMessage={'Data': message.as_bytes()}
        )
        logging.info(f"Email sent to {recipient_email}, MessageId: {response['MessageId']}")
        return response['MessageId']
    except ClientError as e:
        logging.error(f"AWS SES Error sending email to {recipient_email}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error sending email with SES: {e}")
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)