import os, json, re, logging, asyncio, time, requests, pandas as pd, streamlit as st, openai, boto3
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select, text, distinct, and_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from openai import OpenAIError
from typing import List, Optional
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
import plotly.express as px, uuid, aiohttp, urllib3, random, html
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT = map(os.getenv, ["SUPABASE_DB_HOST", "SUPABASE_DB_NAME", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD", "SUPABASE_DB_PORT"])
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal, Base = sessionmaker(bind=engine), declarative_base()

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
    campaign = relationship("Campaign")
    email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    template_id = Column(BigInteger, ForeignKey('email_templates.id'))
    customized_subject, customized_content, original_subject, original_content = [Column(Text) for _ in range(4)]
    status = Column(Text)
    engagement_data = Column(JSON)
    message_id, tracking_id = Column(Text), Column(Text, unique=True)
    sent_at = Column(DateTime(timezone=True))
    ai_customized = Column(Boolean, default=False)
    opened_at, clicked_at = Column(DateTime(timezone=True)), Column(DateTime(timezone=True))
    open_count, click_count = Column(BigInteger, default=0), Column(BigInteger, default=0)
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

def save_email_campaign(session, lead_email, template_id, status, sent_at=None, subject=None, message_id=None, customized_content=None, campaign_id=None, tracking_id=None):
    try:
        lead = session.query(Lead).filter_by(email=lead_email).first()
        if not lead: return None
        campaign = EmailCampaign(campaign_id=campaign_id or get_active_campaign_id(), lead_id=lead.id, template_id=template_id, status=status, sent_at=sent_at or datetime.utcnow(), customized_subject=subject, customized_content=customized_content, message_id=message_id, tracking_id=tracking_id)
        session.add(campaign)
        session.commit()
        return campaign
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error saving email campaign for lead_email: {lead_email}: {str(e)}")
        raise

os.environ['PGGSSENCMODE'] = 'disable'
load_dotenv()
openai.api_key, openai.api_base = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_API_BASE")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

Base.metadata.create_all(bind=engine)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        with get_db_connection() as session:
            log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id)
        return json.loads(result)
    except Exception as e:
        st.error(f"Error in OpenAI API call: {str(e)}")
        with get_db_connection() as session:
            log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id)
        return None

def generate_or_adjust_email_template(session, terms=None, current_template=None, adjustment_prompt=None, kb_info=None):
    if current_template:
        prompt = f"""Adjust the following email template based on the given instructions:

Current Template:
{current_template}

Adjustment Instructions:
{adjustment_prompt}

Knowledge Base Info:
{json.dumps(kb_info)}

Guidelines:
1. Maintain focus on conversion and avoiding spam filters.
2. Preserve the natural, conversational tone.
3. Ensure benefits to the reader remain highlighted.
4. Continue addressing potential customer doubts and fears.
5. Keep clear CTAs at the beginning and end.
6. Remain concise and impactful.
7. Maintain minimal formatting suitable for an email.

Provide the adjusted email body content in HTML format, excluding <body> tags.
"""
    else:
        prompt = f"""Create an email template for the following search terms: {', '.join(terms)}.

Knowledge Base Info: 
{json.dumps(kb_info)}

Guidelines: Focus on benefits to the reader, address potential customer doubts, 
include clear CTAs, use a natural tone, and be concise. 
"""

    prompt += """
Respond with a JSON object in the following format:
{
    "subject": "Your email subject here",
    "body": "Your HTML email body here"
}"""

    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in creating and refining high-converting email templates for targeted marketing campaigns."},
        {"role": "user", "content": prompt}
    ]
    return openai_chat_completion(messages, function_name="generate_or_adjust_email_template") or {"subject": "", "body": ""}

def parse_email_address(email):
    if not email: return None, None
    match = re.match(r'^(.*?)\s*<(.+@.+)>$', email.strip())
    return (match.groups()[0].strip(), match.groups()[1].strip()) if match else (None, email.strip()) if '@' in email else (None, None)

def send_email_ses(from_email, to_email, subject, body, charset='UTF-8'):
    from_name, from_address = parse_email_address(from_email)
    to_name, to_address = parse_email_address(to_email)
    if not from_address or not to_address:
        logging.error(f"Invalid email address: From: {from_email}, To: {to_email}")
        return None, None
    try:
        client = initialize_aws_session().client('ses', region_name=os.getenv('AWS_REGION'))
        tracking_id = str(uuid.uuid4())
        tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
        tracked_body = body + f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/>'
        soup = BeautifulSoup(tracked_body, 'html.parser')
        for a in soup.find_all('a', href=True):
            original_url = a['href']
            tracked_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'click', 'url': original_url})}"
            a['href'] = tracked_url
        tracked_body = str(soup)
        response = client.send_email(Source=from_address, Destination={'ToAddresses': [to_address]}, Message={'Subject': {'Data': subject, 'Charset': charset}, 'Body': {'Html': {'Data': tracked_body, 'Charset': charset}}}, ReturnPath=from_address)
        logging.info(f"Email sent successfully. Message ID: {response['MessageId']}")
        return response, tracking_id
    except ClientError as e:
        logging.error(f"AWS SES ClientError: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error sending email: {str(e)}")
        return None, None
def initialize_aws_session():
    return boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

get_db_connection = SessionLocal

def is_valid_email(email):
    if email is None:
        return False
    # First, check against custom patterns
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

    # If it passes custom checks, use email_validator
    try:
        v = validate_email(email)
        return True
    except EmailNotValidError:
        return False

def remove_invalid_leads(session):
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
        # Remove associated lead sources
        session.query(LeadSource).filter(LeadSource.lead_id == lead.id).delete()
        # Remove the lead
        session.delete(lead)

    session.commit()
    return len(invalid_leads)

find_emails = lambda text: set(re.findall(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', text))
extract_phone_numbers = lambda text: [f"+{m[0]} ({m[1]}) {m[2]}-{m[3]}".strip() for m in re.findall(r'\b(?:\+?(\d{1,3}))?[\s.-]?(?:\(?(\d{2,4})\)?[\s.-]?)?(\d{3})[\s.-]?(\d{4})\b', text) if any(m)]

def extract_visible_text(soup):
    for element in soup(['style', 'script', 'head', 'title', 'meta', '[document]']): element.extract()
    return ' '.join(soup.stripped_strings)

@st.cache_data
def get_domain_from_url(url):
    return urlparse(url).netloc

def save_lead(session, email, first_name=None, last_name=None, company=None, job_title=None, phone=None, url=None, domain=None, search_term_id=None, created_at=None):
    try:
        existing_lead = session.query(Lead).filter_by(email=email).first()
        if existing_lead:
            for attr in ['first_name', 'last_name', 'company', 'job_title', 'phone', 'created_at']:
                if locals()[attr]: setattr(existing_lead, attr, locals()[attr])
            lead = existing_lead
        else:
            lead = Lead(email=email, first_name=first_name, last_name=last_name, company=company, job_title=job_title, phone=phone, created_at=created_at or datetime.utcnow())
            session.add(lead)
        session.flush()
        lead_source = LeadSource(lead_id=lead.id, url=url, domain=domain or get_domain_from_url(url), search_term_id=search_term_id)
        session.add(lead_source)
        session.add(CampaignLead(campaign_id=get_active_campaign_id(), lead_id=lead.id, status="Not Contacted", created_at=datetime.utcnow()))
        session.commit()
        return lead
    except Exception as e:
        logging.error(f"Error saving lead: {str(e)}")
        session.rollback()
        return None

def save_lead_source(session, lead_id, search_term_id, url, http_status, scrape_duration, domain=None, page_title=None, meta_description=None, content=None, tags=None, phone_numbers=None):
    session.add(LeadSource(lead_id=lead_id, search_term_id=search_term_id, url=url, domain=domain or get_domain_from_url(url), http_status=http_status, scrape_duration=scrape_duration, page_title=page_title or get_page_title(url), meta_description=meta_description or get_page_description(url), content=content or extract_visible_text(BeautifulSoup(requests.get(url).text, 'html.parser')), tags=tags, phone_numbers=phone_numbers))
    session.commit()

def log_ai_request(session, function_name, prompt, response, lead_id=None, email_campaign_id=None):
    session.add(AIRequestLog(function_name=function_name, prompt=json.dumps(prompt), response=json.dumps(response) if response else None, lead_id=lead_id, email_campaign_id=email_campaign_id, model_used=openai.model))
    session.commit()

def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
    session.add(SearchTermEffectiveness(term=term, total_results=total_results, valid_leads=valid_leads, irrelevant_leads=total_results - valid_leads, blogs_found=blogs_found, directories_found=directories_found))
    session.commit()

get_active_project_id = lambda: st.session_state.get('active_project_id', 1)
get_active_campaign_id = lambda: st.session_state.get('active_campaign_id', 1)
set_active_project_id = lambda project_id: st.session_state.__setitem__('active_project_id', project_id)
set_active_campaign_id = lambda campaign_id: st.session_state.__setitem__('active_campaign_id', campaign_id)

def add_or_get_search_term(session, term, campaign_id, created_at=None):
    search_term = session.query(SearchTerm).filter_by(term=term, campaign_id=campaign_id).first()
    if not search_term:
        search_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=created_at or datetime.utcnow())
        session.add(search_term)
        session.commit()
        session.refresh(search_term)
    return search_term.id

fetch_campaigns = lambda session: [f"{camp.id}: {camp.campaign_name}" for camp in session.query(Campaign).all()]
fetch_projects = lambda session: [f"{project.id}: {project.project_name}" for project in session.query(Project).all()]
fetch_email_templates = lambda session: [f"{t.id}: {t.template_name}" for t in session.execute(select(EmailTemplate)).scalars()]

def create_or_update_email_template(session, template_name, subject, body_content, template_id=None, is_ai_customizable=False, created_at=None):
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=get_active_campaign_id(), created_at=created_at or datetime.utcnow())
    if template_id: template.template_name, template.subject, template.body_content, template.is_ai_customizable = template_name, subject, body_content, is_ai_customizable
    session.add(template)
    session.commit()
    return template.id

safe_datetime_compare = lambda date1, date2: False if date1 is None or date2 is None else date1 > date2

def fetch_search_terms_with_lead_count(session):
    try:
        query = (session.query(SearchTerm.id, SearchTerm.term, func.count(distinct(LeadSource.lead_id)).label('lead_count'), func.count(distinct(EmailCampaign.id)).label('email_count'), SearchTerm.created_at, SearchTerm.campaign_id, Campaign.campaign_name)
                 .outerjoin(LeadSource, SearchTerm.id == LeadSource.search_term_id)
                 .outerjoin(EmailCampaign, LeadSource.lead_id == EmailCampaign.lead_id)
                 .outerjoin(Campaign, SearchTerm.campaign_id == Campaign.id)
                 .group_by(SearchTerm.id, Campaign.id)
                 .order_by(func.count(distinct(LeadSource.lead_id)).desc()))
        return pd.DataFrame([{'ID': r.id, 'Term': f"{r.term} ({r.lead_count})", 'Lead Count': r.lead_count, 'Email Count': r.email_count, 'Created At': r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "", 'Campaign ID': r.campaign_id, 'Campaign Name': r.campaign_name or ""} for r in query.all()])
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_search_terms_with_lead_count: {str(e)}")
        return pd.DataFrame(columns=['ID', 'Term', 'Lead Count', 'Email Count', 'Created At', 'Campaign ID', 'Campaign Name'])

def fetch_leads(session):
    try:
        leads = session.query(Lead).options(joinedload(Lead.email_campaigns)).all()
        return pd.DataFrame({
            "ID": [l.id for l in leads],
            "Email": [l.email for l in leads],
            "Phone": [l.phone for l in leads],
            "First Name": [l.first_name for l in leads],
            "Last Name": [l.last_name for l in leads],
            "Company": [l.company for l in leads],
            "Job Title": [l.job_title for l in leads],
            "Created At": [l.created_at.strftime("%Y-%m-%d %H:%M:%S") if l.created_at else "" for l in leads],
            "Last Contact": [max((c.sent_at for c in l.email_campaigns if c.sent_at), default=None) for l in leads],
            "Total Emails Sent": [len(l.email_campaigns) for l in leads],
            "Last Email Status": [l.email_campaigns[-1].status if l.email_campaigns else "" for l in leads]
        })
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_leads: {str(e)}")
        return pd.DataFrame()

def fetch_sent_email_campaigns(session):
    try:
        email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({
            'ID': [ec.id for ec in email_campaigns],
            'Sent At': [ec.sent_at.strftime("%Y-%m-%d %H:%M:%S") if ec.sent_at else "" for ec in email_campaigns],
            'Email': [ec.lead.email for ec in email_campaigns],
            'Template': [ec.template.template_name for ec in email_campaigns],
            'Subject': [ec.customized_subject for ec in email_campaigns],
            'Content': [ec.customized_content for ec in email_campaigns],
            'Status': [ec.status for ec in email_campaigns],
            'Message ID': [ec.message_id for ec in email_campaigns],
            'Campaign ID': [ec.campaign_id for ec in email_campaigns],
            'Lead Name': [f"{ec.lead.first_name} {ec.lead.last_name}".strip() for ec in email_campaigns],
            'Lead Company': [ec.lead.company for ec in email_campaigns]
        })
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_sent_email_campaigns: {str(e)}")
        return pd.DataFrame()

def view_sent_email_campaigns():
    st.header("Sent Email Campaigns")
    with get_db_connection() as session:
        email_campaigns = fetch_sent_email_campaigns(session)
    if not email_campaigns.empty:
        st.dataframe(email_campaigns)
        st.subheader("Detailed Content")
        st.text_area("Content", "\n".join(email_campaigns['Content']), height=300)
    else:
        st.info("No sent email campaigns found.")

categorize_page_content = lambda soup, url: 'blog' if 'blog' in (url + (soup.title.string or '').lower() + (soup.find('meta', attrs={'name': 'description'})['content'].lower() if soup.find('meta', attrs={'name': 'description'}) else '')).lower() else 'directory' if 'directory' in (url + (soup.title.string or '').lower() + (soup.find('meta', attrs={'name': 'description'})['content'].lower() if soup.find('meta', attrs={'name': 'description'}) else '')).lower() else 'company'

def extract_emails_from_html(html_content):
    emails = set()
    if html_content:
        emails.update(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', html_content))
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup.find_all(['a', 'p', 'span', 'div']):
            if tag.name == 'a' and tag.get('href', '').startswith('mailto:'): emails.add(tag['href'][7:])
            else: emails.update(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', tag.get_text()))
    return list(emails)

update_search_term_status = lambda session, term_id, status, leads_found: setattr(session.query(SearchTerm).filter(SearchTerm.id == int(term_id)).first(), 'status', status) or setattr(session.query(SearchTerm).filter(SearchTerm.id == int(term_id)).first(), 'leads_found', leads_found)

def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

create_log_container = lambda: st.empty()

def update_log(log_container, message, level='info'):
    icon = {'info': '🔵', 'success': '🟢', 'warning': '🟠', 'error': '🔴'}.get(level, '⚪')
    log_entry = f"{icon} {html.escape(message)}"
    if 'log_entries' not in st.session_state: st.session_state.log_entries = []
    st.session_state.log_entries.append(log_entry)
    log_html = f"<div style='height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.8em; line-height: 1.2;'>{'<br>'.join(st.session_state.log_entries)}</div>"
    log_container.markdown(log_html, unsafe_allow_html=True)

def manual_search(session, terms, num_results, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False):
    ua, results, total_leads, domains_processed, log_container = UserAgent(), [], 0, set(), create_log_container()
    for original_term in terms:
        try:
            search_term_id = add_or_get_search_term(session, original_term, get_active_campaign_id())
            search_term = shuffle_keywords(original_term) if shuffle_keywords_option else original_term
            search_term = optimize_search_term(search_term, 'english' if optimize_english else 'spanish') if optimize_english or optimize_spanish else search_term
            update_log(log_container, f"Searching for '{original_term}' (Used '{search_term}')")
            for url in google_search(search_term, num_results):
                domain = get_domain_from_url(url)
                if ignore_previously_fetched and domain in domains_processed:
                    update_log(log_container, f"Skipping Previously Fetched: {domain}", 'warning')
                    continue
                update_log(log_container, f"Fetching: {url}")
                try:
                    response = requests.get(url, timeout=10, verify=False, headers={'User-Agent': ua.random})
                    response.raise_for_status()
                    html_content, soup = response.text, BeautifulSoup(response.text, 'html.parser')
                    emails = extract_emails_from_html(html_content)
                    update_log(log_container, f"Found {len(emails)} email(s) on {url}", 'success')
                    for email in filter(is_valid_email, emails):
                        if domain not in domains_processed:
                            name, company, job_title = extract_info_from_page(soup)
                            lead = save_lead(session, email=email, first_name=name, company=company, job_title=job_title, url=url, domain=domain, search_term_id=search_term_id, created_at=datetime.utcnow())
                            if lead:
                                total_leads += 1
                                results.append({'Email': email, 'URL': url, 'Lead Source': original_term, 'Title': get_page_title(html_content), 'Description': get_page_description(html_content), 'Tags': [], 'Name': name, 'Company': company, 'Job Title': job_title, 'Search Term ID': search_term_id})
                                update_log(log_container, f"Saved lead: {email}", 'success')
                                domains_processed.add(domain)
                                break
                except requests.RequestException as e:
                    update_log(log_container, f"Error processing URL {url} for term '{original_term}': {str(e)}", 'error')
        except Exception as e:
            update_log(log_container, f"Error processing term '{original_term}': {str(e)}", 'error')
    update_log(log_container, f"Total leads found: {total_leads}", 'info')
    return {"total_leads": total_leads, "results": results}

optimize_search_term = lambda term, language: f'"{term}" AND ("{random.choice(["contact", "email", "get in touch", "reach out", "contact us", "contact information"] if language == "english" else ["contacto", "correo electrónico", "ponte en contacto", "contáctanos", "contáctenos", "información de contacto"])}" OR "{random.choice(["contact", "email", "get in touch", "reach out", "contact us", "contact information"] if language == "english" else ["contacto", "correo electrónico", "ponte en contacto", "contáctanos", "contáctenos", "información de contacto"])}") AND "@"'

extract_info_from_page = lambda soup: (soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else None, soup.find('meta', property='og:site_name')['content'] if soup.find('meta', property='og:site_name') else None, soup.find('meta', property='og:description')['content'] if soup.find('meta', property='og:description') else None)

fetch_search_terms = lambda session: pd.DataFrame([{'Term': term.term, 'Lead Count': session.query(func.count(distinct(LeadSource.id))).filter(LeadSource.search_term_id == term.id).scalar(), 'Email Count': session.query(func.count(distinct(EmailCampaign.id))).join(Lead).join(LeadSource).filter(LeadSource.search_term_id == term.id).scalar()} for term in session.query(SearchTerm).all()])

def bulk_send_page():
    st.title("Bulk Email Sending")
    with get_db_connection() as session:
        templates = fetch_email_templates(session)
        if not templates:
            st.error("No email templates available. Please create a template first.")
            return
        template_option = st.selectbox("Email Template", options=templates, format_func=lambda x: x.split(":")[1].strip(), index=next((i for i, t in enumerate(templates) if str(st.session_state.get('last_template_id', '')) in t), 0))
        template_id = int(template_option.split(":")[0])
        template = session.query(EmailTemplate).filter_by(id=template_id).first()
        st.session_state['last_template_id'] = template_id
        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input("Subject", value=template.subject if template else "")
            from_email = st.text_input("From Email", "Sami Halawa <hello@indosy.com>")
            reply_to = st.text_input("Reply To", "sami@samihalawa.com")
        with col2:
            send_option = st.radio("Send to:", ["All Leads", "Not Previously Contacted with This Template", "Not Previously Contacted", "Specific Email", "Leads from Chosen Search Terms"])
            specific_email = st.text_input("Enter email", "samihalawaster@gmail.com") if send_option == "Specific Email" else None
            selected_terms = []
            if send_option == "Leads from Chosen Search Terms":
                search_terms_with_counts = fetch_search_terms_with_lead_count(session).sort_values('Lead Count', ascending=False)
                selected_terms = st.multiselect("Select Search Terms", options=search_terms_with_counts['Term'].tolist(), format_func=lambda x: f"{x.split('(')[0].strip()} ({x.split('(')[1].split(')')[0]} leads)")
                if selected_terms:
                    total_leads = sum(int(term.split('(')[1].split(')')[0]) for term in selected_terms)
                    st.write(f"Total leads for selected terms: {total_leads}")
                    st.write("Lead counts for selected terms:")
                    for term in selected_terms:
                        count = int(term.split('(')[1].split(')')[0])
                        st.write(f"- {term.split('(')[0].strip()}: {count} leads")
        st.markdown("### Email Preview")
        st.text(f"From: {from_email}")
        st.text(f"Reply-To: {reply_to}")
        st.text(f"Subject: {subject}")
        preview_html = get_email_preview(session, template_id, from_email, reply_to)
        st.components.v1.html(preview_html, height=600, scrolling=True)
        leads = fetch_leads_for_bulk_send(session, template_id, send_option, specific_email, selected_terms)
        lead_count = len(leads)
        st.info(f"This action will send emails to {lead_count} lead{'s' if lead_count != 1 else ''}.")
        if st.button("Send Emails", type="primary"):
            if not leads:
                st.warning("No leads found matching the selected criteria.")
                return
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            for index, lead in enumerate(leads):
                if not is_valid_email(lead.email):
                    status_text.text(f"Skipping invalid email: {lead.email}")
                    continue
                try:
                    status_text.text(f"Sending email to {lead.email}...")
                    email_body = template.body_content.format(first_name=lead.first_name, last_name=lead.last_name, company=lead.company)
                    response, tracking_id = send_email_ses(from_email, lead.email, subject, email_body)
                    status = 'sent' if response and 'MessageId' in response else 'failed'
                    message = f"Email {status} for {lead.email}"
                    results.append({'Email': lead.email, 'Status': status, 'Message': message})
                    save_email_campaign(session, lead.email, template_id, status, tracking_id=tracking_id)
                except Exception as e:
                    message = f"Error sending email to {lead.email}: {str(e)}"
                    results.append({'Email': lead.email, 'Status': 'failed', 'Message': message})
                finally:
                    session.commit()
                progress_bar.progress((index + 1) / len(leads))
            results_df = pd.DataFrame(results)
            success_rate = (results_df['Status'] == 'sent').mean() * 100
            st.success(f"Bulk email sending completed. Success rate: {success_rate:.2f}%")
            st.dataframe(results_df)
            st.download_button("Download Results CSV", results_df.to_csv(index=False).encode('utf-8'), "bulk_send_results.csv", "text/csv", key='download-csv')
            st.plotly_chart(px.pie(names=['Sent', 'Failed'], values=[results_df['Status'].value_counts().get('sent', 0), results_df['Status'].value_counts().get('failed', 0)], title="Email Sending Status"), use_container_width=True)

def send_email_ses(from_email, to_email, subject, body, charset='UTF-8'):
    from_name, from_address = parse_email_address(from_email)
    to_name, to_address = parse_email_address(to_email)
    if not from_address or not to_address:
        logging.error(f"Invalid email address: From: {from_email}, To: {to_email}")
        return None, None
    try:
        client = initialize_aws_session().client('ses', region_name=os.getenv('AWS_REGION'))
        tracking_id = str(uuid.uuid4())
        tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
        tracked_body = body + f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/>'
        soup = BeautifulSoup(tracked_body, 'html.parser')
        for a in soup.find_all('a', href=True):
            original_url = a['href']
            tracked_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'click', 'url': original_url})}"
            a['href'] = tracked_url
        tracked_body = str(soup)
        response = client.send_email(Source=from_address, Destination={'ToAddresses': [to_address]}, Message={'Subject': {'Data': subject, 'Charset': charset}, 'Body': {'Html': {'Data': tracked_body, 'Charset': charset}}}, ReturnPath=from_address)
        logging.info(f"Email sent successfully. Message ID: {response['MessageId']}")
        return response, tracking_id
    except ClientError as e:
        error_code, error_message = e.response['Error']['Code'], e.response['Error']['Message']
        logging.error(f"AWS SES ClientError: {error_code} - {error_message}")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error sending email: {str(e)}")
        return None, None

fetch_lead_counts_for_terms = lambda session, terms: {term: session.query(func.count(Lead.id)).join(LeadSource).filter(LeadSource.search_term == term).scalar() for term in terms}

get_email_preview = lambda session, template_id, from_email, reply_to: f"<html><head><style>body {{ font-family: Arial, sans-serif; }}</style></head><body>{session.query(EmailTemplate).filter_by(id=template_id).first().body_content if session.query(EmailTemplate).filter_by(id=template_id).first() else '<p>Template not found</p>'}</body></html>"

def fetch_leads_for_bulk_send(session, template_id, send_option, specific_email=None, selected_terms=None):
    query = session.query(Lead)
    if send_option == "Not Previously Contacted with This Template":
        query = query.outerjoin(EmailCampaign, and_(EmailCampaign.lead_id == Lead.id, EmailCampaign.template_id == template_id)).filter(EmailCampaign.id == None)
    elif send_option == "Not Previously Contacted":
        query = query.outerjoin(EmailCampaign).filter(EmailCampaign.id == None)
    elif send_option == "Specific Email":
        query = query.filter(Lead.email == specific_email)
    elif send_option == "Leads from Chosen Search Terms" and selected_terms:
        term_ids = [int(term.split('(')[0].strip().split(':')[0]) for term in selected_terms]
        query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.id.in_(term_ids))
    return query.all()

def bulk_send_emails(session: Session, template_id: int, from_email: str, reply_to: str, leads: List[Lead]):
    logs, all_results = [], []
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        logging.error("Template not found.")
        return ["Template not found."], []

    removed_count = remove_invalid_leads(session)
    if removed_count > 0:
        logs.append(f"Removed {removed_count} invalid leads before sending emails.")

    progress, success_container = st.progress(0), st.empty()
    for index, lead in enumerate(leads, start=1):
        if not is_valid_email(lead.email):
            logs.append(f"Skipping invalid email: {lead.email}")
            continue

        logs.append(f"Processing lead {lead.id} with email {lead.email}...")
        try:
            sent_at, message_id, status = datetime.utcnow(), f"msg-{lead.id}-{int(time.time())}", 'pending'
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            customized_template = generate_or_adjust_email_template(session, current_template=template.body_content, adjustment_prompt=f"Customize for {lead.first_name} from {lead.company}", kb_info=kb_info)
            customized_subject, customized_content = customized_template['subject'], customized_template['body']
            ses_response = send_email_ses(from_email, lead.email, customized_subject, customized_content)
            if ses_response:
                message_id = ses_response['MessageId']
                status = 'sent'
                logs.append(f"Email sent to {lead.email} with Message ID: {message_id}")
            else:
                status = 'failed'
                logs.append(f"Failed to send email to {lead.email}")

            save_email_campaign(session, lead.email, template_id, status, sent_at, customized_subject, message_id, customized_content)
            
            if status == 'sent':
                success_container.markdown(f"<h3 style='color: green;'>✅ Email sent to {lead.email}</h3>", unsafe_allow_html=True)
            else:
                success_container.markdown(f"<h3 style='color: red;'>❌ Failed to send email to {lead.email}</h3>", unsafe_allow_html=True)

            all_results.append({
                "Email": lead.email,
                "Status": status,
                "Message ID": message_id,
                "Subject": customized_subject,
                "Content": customized_content[:100] + "..." if len(customized_content) > 100 else customized_content
            })
            time.sleep(0.1)
            progress.progress(index / len(leads))
        except Exception as e:
            logs.append(f"Error processing {lead.email}: {str(e)}")
            save_email_campaign(session, lead.email, template_id, 'failed', datetime.utcnow(), template.subject, None, str(e))
            all_results.append({"Email": lead.email, "Status": 'failed', "Message ID": None, "Subject": template.subject})

    return logs, all_results

def generate_email_template(session, terms, kb_info):
    prompt = f"""Create an email template for the following search terms: {', '.join(terms)}.

Knowledge Base Info: 
{json.dumps(kb_info)}

Guidelines: Focus on benefits to the reader, address potential customer doubts, 
include clear CTAs, use a natural tone, and be concise. 

Respond with a JSON object in the format: 
{{
    "subject": "Subject line here",
    "body": "HTML body content here"
}}"""
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in creating high-converting email templates for targeted marketing campaigns."},
        {"role": "user", "content": prompt}
    ]
    return openai_chat_completion(messages, function_name="generate_email_template") or {"subject": "", "body": ""}

def adjust_email_template(session, current_template, adjustment_prompt, kb_info):
    prompt = f"""Adjust the following email template based on the given instructions:

Current Template:
{current_template}

Adjustment Instructions:
{adjustment_prompt}

Knowledge Base Info:
{json.dumps(kb_info)}

Guidelines:
1. Maintain focus on conversion and avoiding spam filters.
2. Preserve the natural, conversational tone.
3. Ensure benefits to the reader remain highlighted.
4. Continue addressing potential customer doubts and fears.
5. Keep clear CTAs at the beginning and end.
6. Remain concise and impactful.
7. Maintain minimal formatting suitable for an email.

Provide the adjusted email body content in HTML format, excluding <body> tags.

Respond with a JSON object in the following format:
{{
    "subject": "Your adjusted email subject here",
    "body": "Your adjusted HTML email body here"
}}"""
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in refining high-converting email templates for targeted marketing campaigns."},
        {"role": "user", "content": prompt}
    ]
    return openai_chat_completion(messages, function_name="adjust_email_template") or {"subject": "", "body": ""}

def update_log_display(log_container, logs):
    log_container.markdown(
        f"""
        <style>
        .log-container {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
            background-color: rgba(49, 51, 63, 0.1);
        }}
        .log-entry {{
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 0.25rem;
        }}
        </style>
        <div class="log-container">
            <h4>Latest Logs</h4>
            {"".join(f'<div class="log-entry">{log}</div>' for log in logs[-20:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def update_results_display(results_container, results):
    results_container.markdown(
        f"""
        <style>
        .results-container {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
            background-color: rgba(49, 51, 63, 0.1);
        }}
        .result-entry {{
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 0.25rem;
        }}
        </style>
        <div class="results-container">
            <h4>Found Leads ({len(results)})</h4>
            {"".join(f'<div class="result-entry"><strong>{res["Email"]}</strong><br>{res["Company"]}</div>' for res in results[-10:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def get_domain_from_url(url): return urlparse(url).netloc

def manual_search_page():
    st.title("Manual Search")
    
    with get_db_connection() as session:
        recent_searches = session.query(SearchTerm).order_by(SearchTerm.created_at.desc()).limit(5).all()
        email_templates = fetch_email_templates(session)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_terms = st_tags(
            label='Enter search terms:',
            text='Press enter to add more',
            value=[term.term for term in recent_searches],
            suggestions=['software engineer', 'data scientist', 'product manager'],
            maxtags=10,
            key='search_terms_input'
        )
        num_results = st.slider("Results per term", 1, 50, 10)
    
    with col2:
        enable_email_sending = st.checkbox("Enable email sending")
        ignore_previously_fetched = st.checkbox("Ignore fetched domains", value=True)
        shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=True)
        optimize_english = st.checkbox("Optimize (English)", value=False)
        optimize_spanish = st.checkbox("Optimize (Spanish)", value=False)
    
    if enable_email_sending:
        col3, col4 = st.columns(2)
        with col3:
            email_template = st.selectbox("Email template", options=email_templates, format_func=lambda x: x.split(":")[1].strip())
        with col4:
            from_email = st.text_input("From Email", "Sami Halawa <hello@indosy.com>")
            reply_to = st.text_input("Reply To", "sami@samihalawa.com")

    if st.button("Search"):
        if not search_terms:
            return st.warning("Enter at least one search term.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        email_status = st.empty()
        results = []
        
        leads_container = st.empty()
        leads_found, emails_sent = [], []
        
        for i, term in enumerate(search_terms):
            status_text.text(f"Searching: '{term}' ({i+1}/{len(search_terms)})")
            
            term_results = manual_search(session, [term], num_results, ignore_previously_fetched, optimize_english, optimize_spanish, shuffle_keywords_option)
            results.extend(term_results['results'])
            
            leads_found.extend([f"{res['Email']} - {res['Company']}" for res in term_results['results']])
            
            if enable_email_sending:
                template = session.query(EmailTemplate).filter_by(id=int(email_template.split(":")[0])).first()
                for result in term_results['results']:
                    response, tracking_id = send_email_ses(from_email, result['Email'], template.subject, template.body_content)
                    if response:
                        save_email_campaign(session, result['Email'], template.id, 'sent', datetime.utcnow(), template.subject, response.get('MessageId', 'Unknown'), template.body_content, tracking_id=tracking_id)
                        emails_sent.append(result['Email'])
                    else:
                        save_email_campaign(session, result['Email'], template.id, 'failed', datetime.utcnow(), template.subject, None, template.body_content)
            
            leads_container.dataframe(pd.DataFrame({"Leads Found": leads_found, "Emails Sent": emails_sent + [""] * (len(leads_found) - len(emails_sent))}))
            progress_bar.progress((i + 1) / len(search_terms))
        
        st.subheader("Search Results")
        st.dataframe(pd.DataFrame(results))
        
        st.download_button(
            label="Download CSV",
            data=pd.DataFrame(results).to_csv(index=False).encode('utf-8'),
            file_name="search_results.csv",
            mime="text/csv",
        )

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def get_page_title(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.title.string if soup.title else "No title found"

def get_page_description(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    return meta_desc['content'] if meta_desc else "No description found"

def is_valid_email(email):
    if email is None:
        return False
    # First, check against custom patterns
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^(email|info|contact|support|hello|hola|hi|salutations|greetings|inquiries|questions)@.*",
        r"^email@email\.com$",
        r".*@example\.com$",
        r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"
    ]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    
    if any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns):
        return False
    if any(email.lower().endswith(f"@{domain}") for domain in typo_domains):
        return False

    # If it passes custom checks, use email_validator
    try:
        v = validate_email(email)
        return True
    except EmailNotValidError:
        return False

def perform_quick_scan(session):
    with st.spinner("Performing quick scan..."):
        terms = session.query(SearchTerm).order_by(func.random()).limit(3).all()
        res = manual_search(session, [term.term for term in terms], 10)
    st.success(f"Quick scan completed! Found {len(res)} new leads.")
    return {"new_leads": len(res), "terms_used": [term.term for term in terms]}

def bulk_send_emails(session: Session, template_id: int, from_email: str, reply_to: str, leads: List[Lead]):
    logs, all_results = [], []
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        logging.error("Template not found.")
        return ["Template not found."], []

    removed_count = remove_invalid_leads(session)
    if removed_count > 0:
        logs.append(f"Removed {removed_count} invalid leads before sending emails.")

    progress, success_container = st.progress(0), st.empty()
    for index, lead in enumerate(leads, start=1):
        if not is_valid_email(lead.email):
            logs.append(f"Skipping invalid email: {lead.email}")
            continue

        logs.append(f"Processing lead {lead.id} with email {lead.email}...")
        try:
            sent_at, message_id, status = datetime.utcnow(), f"msg-{lead.id}-{int(time.time())}", 'pending'
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            customized_template = generate_or_adjust_email_template(session, current_template=template.body_content, adjustment_prompt=f"Customize for {lead.first_name} from {lead.company}", kb_info=kb_info)
            customized_subject, customized_content = customized_template['subject'], customized_template['body']
            ses_response = send_email_ses(from_email, lead.email, customized_subject, customized_content)
            if ses_response:
                message_id = ses_response['MessageId']
                status = 'sent'
                logs.append(f"Email sent to {lead.email} with Message ID: {message_id}")
            else:
                status = 'failed'
                logs.append(f"Failed to send email to {lead.email}")

            save_email_campaign(session, lead.email, template_id, status, sent_at, customized_subject, message_id, customized_content)
            
            if status == 'sent':
                success_container.markdown(f"<h3 style='color: green;'>✅ Email sent to {lead.email}</h3>", unsafe_allow_html=True)
            else:
                success_container.markdown(f"<h3 style='color: red;'>❌ Failed to send email to {lead.email}</h3>", unsafe_allow_html=True)

            all_results.append({
                "Email": lead.email,
                "Status": status,
                "Message ID": message_id,
                "Subject": customized_subject,
                "Content": customized_content[:100] + "..." if len(customized_content) > 100 else customized_content
            })
            time.sleep(0.1)
            progress.progress(index / len(leads))
        except Exception as e:
            logs.append(f"Error processing {lead.email}: {str(e)}")
            save_email_campaign(session, lead.email, template_id, 'failed', datetime.utcnow(), template.subject, None, str(e))
            all_results.append({"Email": lead.email, "Status": 'failed', "Message ID": None, "Subject": template.subject})

    return logs, all_results

def view_campaign_logs():
    st.header("Email Logs")
    with get_db_connection() as session:
        campaigns = fetch_campaigns(session)
        selected_campaign = st.selectbox("Select Campaign", options=campaigns, format_func=lambda x: x.split(":")[1].strip())
        campaign_id = int(selected_campaign.split(":")[0])

        logs = fetch_campaign_logs(session, campaign_id)
        if logs.empty:
            st.info("No logs found for this campaign.")
        else:
            st.write(f"Total emails sent: {len(logs)}")
            st.write(f"Success rate: {(logs['Status'] == 'sent').mean():.2%}")

            # Add date range filter
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=logs['Sent At'].min().date())
            with col2:
                end_date = st.date_input("End Date", value=logs['Sent At'].max().date())
            
            filtered_logs = logs[(logs['Sent At'].dt.date >= start_date) & (logs['Sent At'].dt.date <= end_date)]

            # Add search functionality
            search_term = st.text_input("Search by email or subject")
            if search_term:
                filtered_logs = filtered_logs[filtered_logs['Email'].str.contains(search_term, case=False) | 
                                              filtered_logs['Subject'].str.contains(search_term, case=False)]

            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Emails Sent", len(filtered_logs))
            with col2:
                st.metric("Unique Recipients", filtered_logs['Email'].nunique())
            with col3:
                st.metric("Success Rate", f"{(filtered_logs['Status'] == 'sent').mean():.2%}")

            # Create a bar chart of emails sent per day
            daily_counts = filtered_logs.resample('D', on='Sent At')['Email'].count()
            st.bar_chart(daily_counts)

            # Display logs in an expandable table
            st.subheader("Detailed Email Logs")
            for _, log in filtered_logs.iterrows():
                with st.expander(f"{log['Sent At'].strftime('%Y-%m-%d %H:%M:%S')} - {log['Email']} - {log['Status']}"):
                    st.write(f"**Subject:** {log['Subject']}")
                    st.write(f"**Content Preview:** {log['Content'][:100]}...")
                    if st.button("View Full Email", key=f"view_email_{log['ID']}"):
                        st.text_area("Full Email Content", log['Content'], height=300)
                    if log['Status'] != 'sent':
                        st.error(f"Status: {log['Status']}")

            # Add pagination
            logs_per_page = 20
            total_pages = (len(filtered_logs) - 1) // logs_per_page + 1
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            start_idx = (page - 1) * logs_per_page
            end_idx = start_idx + logs_per_page
            
            # Display paginated logs
            st.table(filtered_logs.iloc[start_idx:end_idx][['Sent At', 'Email', 'Subject', 'Status']])

            # Add export functionality
            if st.button("Export Logs to CSV"):
                csv = filtered_logs.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="email_logs.csv",
                    mime="text/csv"
                )

def fetch_campaign_logs(session, campaign_id):
    return pd.DataFrame([{'ID': log.id, 'Sent At': log.sent_at, 'Email': log.lead.email, 'Status': log.status, 'Subject': log.customized_subject, 'Content': log.customized_content} for log in session.query(EmailCampaign).filter_by(campaign_id=campaign_id).order_by(EmailCampaign.sent_at.desc())])

def fetch_leads(session):
    try:
        leads = session.query(Lead).options(joinedload(Lead.email_campaigns)).all()
        return pd.DataFrame({
            "ID": [l.id for l in leads], "Email": [l.email for l in leads], "Phone": [l.phone for l in leads],
            "First Name": [l.first_name for l in leads], "Last Name": [l.last_name for l in leads],
            "Company": [l.company for l in leads], "Job Title": [l.job_title for l in leads],
            "Created At": [l.created_at.strftime("%Y-%m-%d %H:%M:%S") if l.created_at else "" for l in leads],
            "Last Contact": [max((c.sent_at for c in l.email_campaigns if c.sent_at), default=None) for l in leads],
            "Total Emails Sent": [len(l.email_campaigns) for l in leads],
            "Last Email Status": [l.email_campaigns[-1].status if l.email_campaigns else "" for l in leads]
        })
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_leads: {str(e)}")
        return pd.DataFrame()

def update_lead(session, lead_id, updated_data):
    try:
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            for key, value in updated_data.items(): setattr(lead, key, value)
            session.commit()
            return True
    except SQLAlchemyError as e:
        logging.error(f"Error updating lead {lead_id}: {str(e)}")
        session.rollback()
    return False

def delete_lead(session, lead_id):
    try:
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            session.delete(lead)
            session.commit()
            return True
    except SQLAlchemyError as e:
        logging.error(f"Error deleting lead {lead_id}: {str(e)}")
        session.rollback()
    return False

def is_valid_email(email):
    try: validate_email(email); return True
    except EmailNotValidError: return False

def view_leads_page():
    st.title("Lead Management Dashboard")
    with get_db_connection() as session:
        if 'leads' not in st.session_state or st.button("Refresh Leads"):
            st.session_state.leads = fetch_leads_with_sources(session)
        if not st.session_state.leads.empty:
            total_leads = len(st.session_state.leads)
            contacted_leads = len(st.session_state.leads[st.session_state.leads['Last Contact'].notna()])
            conversion_rate = (st.session_state.leads['Last Email Status'] == 'sent').mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Leads", f"{total_leads:,}")
            col2.metric("Contacted Leads", f"{contacted_leads:,}")
            col3.metric("Conversion Rate", f"{conversion_rate:.2%}")
            
            st.subheader("Leads Table")
            search_term = st.text_input("Search leads by email, name, company, or source")
            filtered_leads = st.session_state.leads[st.session_state.leads.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]
            
            leads_per_page, page_number = 20, st.number_input("Page", min_value=1, value=1)
            start_idx, end_idx = (page_number - 1) * leads_per_page, page_number * leads_per_page
            
            edited_df = st.data_editor(
                filtered_leads.iloc[start_idx:end_idx],
                column_config={
                    "ID": st.column_config.NumberColumn("ID", disabled=True),
                    "Email": st.column_config.TextColumn("Email"),
                    "First Name": st.column_config.TextColumn("First Name"),
                    "Last Name": st.column_config.TextColumn("Last Name"),
                    "Company": st.column_config.TextColumn("Company"),
                    "Job Title": st.column_config.TextColumn("Job Title"),
                    "Source": st.column_config.TextColumn("Source", disabled=True),
                    "Last Contact": st.column_config.DatetimeColumn("Last Contact", disabled=True),
                    "Last Email Status": st.column_config.TextColumn("Last Email Status", disabled=True),
                    "Delete": st.column_config.CheckboxColumn("Delete")
                },
                disabled=["ID", "Source", "Last Contact", "Last Email Status"],
                hide_index=True,
                num_rows="dynamic"
            )
            
            if st.button("Save Changes", type="primary"):
                for index, row in edited_df.iterrows():
                    if row['Delete']:
                        if delete_lead_and_sources(session, row['ID']):
                            st.success(f"Deleted lead: {row['Email']}")
                    else:
                        updated_data = {k: row[k] for k in ['Email', 'First Name', 'Last Name', 'Company', 'Job Title']}
                        if update_lead(session, row['ID'], updated_data):
                            st.success(f"Updated lead: {row['Email']}")
                st.rerun()
            
            st.download_button(
                "Export Filtered Leads to CSV",
                filtered_leads.to_csv(index=False).encode('utf-8'),
                "exported_leads.csv",
                "text/csv"
            )
            
            st.subheader("Lead Growth")
            if 'Created At' in st.session_state.leads.columns:
                lead_growth = st.session_state.leads.groupby(pd.to_datetime(st.session_state.leads['Created At']).dt.to_period('M')).size().cumsum()
                st.line_chart(lead_growth)
            else:
                st.warning("Created At data is not available for lead growth chart.")
            
            st.subheader("Email Campaign Performance")
            email_status_counts = st.session_state.leads['Last Email Status'].value_counts()
            st.plotly_chart(px.pie(
                values=email_status_counts.values,
                names=email_status_counts.index,
                title="Distribution of Email Statuses"
            ), use_container_width=True)
        else:
            st.info("No leads available. Start by adding some leads to your campaigns.")

def fetch_leads_with_sources(session):
    try:
        query = session.query(Lead, func.string_agg(LeadSource.url, ', ').label('sources'), func.max(EmailCampaign.sent_at).label('last_contact'), func.string_agg(EmailCampaign.status, ', ').label('email_statuses')).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(Lead.id)
        return pd.DataFrame([{**{k: getattr(lead, k) for k in ['id', 'email', 'first_name', 'last_name', 'company', 'job_title', 'created_at']}, 'Source': sources, 'Last Contact': last_contact, 'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted', 'Delete': False} for lead, sources, last_contact, email_statuses in query.all()])
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_leads_with_sources: {str(e)}")
        return pd.DataFrame()

def delete_lead_and_sources(session, lead_id):
    try:
        session.query(LeadSource).filter(LeadSource.lead_id == lead_id).delete()
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            session.delete(lead)
            session.commit()
            return True
    except SQLAlchemyError as e:
        logging.error(f"Error deleting lead {lead_id} and its sources: {str(e)}")
        session.rollback()
    return False

def search_terms_page():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Search Terms Dashboard</h1>", unsafe_allow_html=True)
    with get_db_connection() as session:
        search_terms_df = fetch_search_terms_with_lead_count(session)
        if not search_terms_df.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Search Terms", len(search_terms_df))
            col2.metric("Total Leads", search_terms_df['Lead Count'].sum())
            col3.metric("Total Emails Sent", search_terms_df['Email Count'].sum())
            st.subheader("Top Performing Search Terms")
            st.table(search_terms_df.nlargest(5, 'Lead Count')[['Term', 'Lead Count', 'Email Count']])
            st.subheader("Search Term Performance")
            chart_type = st.radio("Select Chart Type", ["Bar", "Pie"])
            fig = px.bar(search_terms_df.nlargest(10, 'Lead Count'), x='Term', y=['Lead Count', 'Email Count'], title='Top 10 Search Terms by Lead Count', labels={'value': 'Count', 'variable': 'Type'}, barmode='group') if chart_type == "Bar" else px.pie(search_terms_df, values='Lead Count', names='Term', title='Distribution of Leads Across Search Terms')
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Search and Filter Terms")
            search_query, sort_by, ascending = st.text_input("Search Terms", ""), st.selectbox("Sort by", ['Lead Count', 'Email Count', 'Created At']), st.checkbox("Ascending Order", value=False)
            filtered_df = search_terms_df[search_terms_df['Term'].str.contains(search_query, case=False, na=False)].sort_values(by=sort_by, ascending=ascending)
            st.dataframe(filtered_df[['Term', 'Lead Count', 'Email Count', 'Created At', 'Campaign Name']].style.background_gradient(subset=['Lead Count', 'Email Count'], cmap='YlOrRd'), use_container_width=True)
            st.download_button("Export to CSV", filtered_df.to_csv(index=False).encode('utf-8'), "search_terms_export.csv", "text/csv")
        else: st.info("No search terms available. Start by adding some search terms to your campaigns.")
        st.subheader("Add New Search Term")
        with st.form("add_search_term"):
            new_term, campaign_id = st.text_input("New Search Term"), get_active_campaign_id()
            if st.form_submit_button("Add Term") and new_term:
                try:
                    add_search_term(session, new_term, campaign_id)
                    st.success(f"Added new search term: {new_term}")
                    st.rerun()
                except Exception as e: st.error(f"Error adding search term: {str(e)}")

def fetch_search_term_groups(session):
    try: return [group[0] for group in session.query(func.coalesce(SearchTerm.group, 'Ungrouped')).distinct()]
    except Exception as e:
        logging.error(f"Error in fetch_search_term_groups: {str(e)}")
        return []

def fetch_leads_for_search_term_groups(session, groups):
    try:
        leads = session.query(Lead).join(CampaignLead).join(SearchTerm).filter(func.coalesce(SearchTerm.group, 'Ungrouped').in_(groups)).all()
        logging.info(f"Number of leads fetched: {len(leads)}")
        return leads
    except Exception as e:
        logging.error(f"Error in fetch_leads_for_search_term_groups: {str(e)}")
        return []

def create_email_template(session, template_name, subject, body_content):
    new_template = EmailTemplate(template_name=template_name, subject=subject, body_content=body_content)
    session.add(new_template)
    session.commit()
    return new_template.id
def email_templates_page():
    st.header("Email Templates")
    with get_db_connection() as session:
        templates = session.query(EmailTemplate).all()
        with st.expander("Create New Template", expanded=False):
            new_template_name = st.text_input("Template Name", key="new_template_name")
            new_template_subject = st.text_input("Subject", key="new_template_subject")
            new_template_body = st.text_area("Body Content", height=200, key="new_template_body")
            if st.button("Create Template", key="create_template_button") and all([new_template_name, new_template_subject, new_template_body]):
                session.add(EmailTemplate(template_name=new_template_name, subject=new_template_subject, body_content=new_template_body, campaign_id=get_active_campaign_id()))
                session.commit()
                st.success("New template created successfully!")
                st.rerun()
            elif st.button("Clear Fields", key="clear_fields_button"):
                st.warning("Please fill in all fields to create a new template.")
        if templates:
            for template in templates:
                with st.expander(f"Template: {template.template_name}", expanded=False):
                    col1, col2 = st.columns(2)
                    edited_subject = col1.text_input("Subject", value=template.subject, key=f"subject_{template.id}")
                    is_ai_customizable = col2.checkbox("AI Customizable", value=template.is_ai_customizable, key=f"ai_{template.id}")
                    edited_body = st.text_area("Body Content", value=template.body_content, height=200, key=f"body_{template.id}")
                    if st.button("Save Changes", key=f"save_{template.id}"):
                        template.subject, template.body_content, template.is_ai_customizable = edited_subject, edited_body, is_ai_customizable
                        session.commit()
                        st.success("Template updated successfully!")
                    st.markdown("### Preview")
                    st.text(f"Subject: {edited_subject}")
                    st.components.v1.html(get_email_preview(session, template.id, "Sami Halawa <hello@indosy.com>", "sami@samihalawa.com"), height=400, scrolling=True)
                    if st.button("Delete Template", key=f"delete_{template.id}"):
                        session.delete(template)
                        session.commit()
                        st.success("Template deleted successfully!")
                        st.rerun()
        else: st.info("No email templates found. Create a new template to get started.")

def get_email_preview(session, template_id, from_email, reply_to):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    return f"<html><head><style>body {{ font-family: Arial, sans-serif; }}</style></head><body>{template.body_content}</body></html>" if template else "<p>Template not found</p>"

def fetch_all_search_terms(session): return session.query(SearchTerm).all()

def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return {attr: getattr(kb_info, attr) for attr in ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']} if kb_info else None

def get_email_template_by_name(session, template_name):
    return session.query(EmailTemplate).filter_by(template_name=template_name).first()

def continuous_automation_process(session, log_container, leads_container):
    emails_found = []
    while st.session_state.get('automation_status', False):
        try:
            search_terms, kb_info = fetch_all_search_terms(session), get_knowledge_base_info(session, get_active_project_id())
            if not search_terms or not kb_info:
                log_container.warning("Missing search terms or knowledge base. Skipping cycle.")
                time.sleep(3600)
                continue
            classified_terms = classify_search_terms(session, [term.term for term in search_terms], kb_info)
            update_log_display(log_container, st.session_state.automation_logs)
            log_container.info(f"Classified search terms into {len(classified_terms)} groups.")
            reference_template = get_email_template_by_name(session, "REFERENCE")
            if not reference_template:
                log_container.warning("REFERENCE email template not found. Skipping cycle.")
                time.sleep(3600)
                continue
            for group_name, terms in classified_terms.items():
                if group_name == "low_quality_search_terms": continue
                adjusted_template_data = adjust_email_template(session, reference_template.body_content, terms, kb_info)
                if not adjusted_template_data:
                    log_container.warning(f"Failed to adjust template for '{group_name}'. Skipping group.")
                    continue
                new_template_id = create_email_template(session, f"{reference_template.template_name}_{group_name}", adjusted_template_data['subject'], adjusted_template_data['body'])
                update_log_display(log_container, st.session_state.automation_logs)
                log_container.info(f"Created template '{reference_template.template_name}_{group_name}' for '{group_name}'.")
                results = manual_search(session, terms, 10)
                new_emails = [res['Email'] for res in results['results'] if res['Email'] not in emails_found]
                emails_found.extend(new_emails)
                leads_container.text_area("Emails Found", "\n".join(emails_found[-50:]), height=200)
                leads_to_send = [lead for lead in fetch_leads_for_search_terms(session, terms) if lead.email in new_emails]
                from_email, reply_to = kb_info.get('contact_email', 'Sami Halawa <hello@indosy.com>'), kb_info.get('contact_email', 'sami@samihalawa.com')
                logs, _ = bulk_send_emails(session, new_template_id, from_email, reply_to, leads_to_send)
                st.session_state.automation_logs.extend(logs)
                update_log_display(log_container, st.session_state.automation_logs)
            time.sleep(3600)
        except Exception as e:
            log_container.error(f"Critical error in automation cycle: {str(e)}")
            logging.exception("Error in continuous_automation_process")
            time.sleep(300)
    log_container.info("Automation stopped")
    
def adjust_email_template(body_content, terms, kb_info):
    return {"body": body_content.replace("{{terms}}", ", ".join(terms)), "subject": f"Adjusted for {', '.join(terms)}"}

def display_search_results(results, key_suffix):
    if not results: return st.warning("No results to display.")
    with st.expander("Search Results", expanded=True):
        st.markdown(f"### Total Leads Found: **{len(results)}**")
    for i, res in enumerate(results):
        with st.expander(f"Lead: {res['Email']}", key=f"lead_expander_{key_suffix}_{i}"):
            st.markdown(f"**URL:** [{res['URL']}]({res['URL']})  \n**Title:** {res['Title']}  \n**Description:** {res['Description']}  \n**Tags:** {', '.join(res['Tags'])}  \n**Lead Source:** {res['Lead Source']}  \n**Lead Email:** {res['Email']}")

def perform_quick_scan(session):
    terms = fetch_search_terms_with_lead_count(session)[:3]
    results = manual_search(session, terms, 10, st.empty(), st.empty())
    return {"new_leads": len(results)}

def fetch_search_terms_with_lead_count(session):
    try:
        query = (session.query(SearchTerm.id, SearchTerm.term, func.count(distinct(LeadSource.lead_id)).label('lead_count'), func.count(distinct(EmailCampaign.id)).label('email_count'), SearchTerm.created_at, SearchTerm.campaign_id, Campaign.campaign_name)
                 .outerjoin(LeadSource, SearchTerm.id == LeadSource.search_term_id)
                 .outerjoin(EmailCampaign, LeadSource.lead_id == EmailCampaign.lead_id)
                 .outerjoin(Campaign, SearchTerm.campaign_id == Campaign.id)
                 .group_by(SearchTerm.id, Campaign.id)
                 .order_by(func.count(distinct(LeadSource.lead_id)).desc()))
        results = query.all()
        return pd.DataFrame([{'ID': r.id, 'Term': f"{r.term} ({r.lead_count})", 'Lead Count': r.lead_count, 'Email Count': r.email_count, 'Created At': r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "", 'Campaign ID': r.campaign_id, 'Campaign Name': r.campaign_name or ""} for r in results])
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_search_terms_with_lead_count: {str(e)}")
        return pd.DataFrame(columns=['ID', 'Term', 'Lead Count', 'Email Count', 'Created At', 'Campaign ID', 'Campaign Name'])

def fetch_leads_for_search_terms(session, search_term_ids) -> List[Lead]:
    return session.query(Lead).distinct().join(LeadSource).filter(LeadSource.search_term_id.in_(search_term_ids)).all()

def projects_campaigns_page():
    with get_db_connection() as session:
        st.header("Projects and Campaigns")
        st.subheader("Add New Project")
        with st.form("add_project_form"):
            project_name = st.text_input("Project Name")
            if st.form_submit_button("Add Project"):
                if project_name.strip():
                    try:
                        session.add(Project(project_name=project_name, created_at=datetime.utcnow()))
                        session.commit()
                        st.success(f"Project '{project_name}' added successfully.")
                    except SQLAlchemyError as e: st.error(f"Error adding project: {str(e)}")
                else: st.warning("Please enter a project name.")
        st.subheader("Existing Projects")
        projects = session.query(Project).all()
        for project in projects:
            with st.expander(f"Project: {project.project_name}"):
                with st.form(f"add_campaign_form_{project.id}"):
                    campaign_name = st.text_input("Campaign Name", key=f"campaign_name_{project.id}")
                    if st.form_submit_button("Add Campaign"):
                        if campaign_name.strip():
                            try:
                                session.add(Campaign(campaign_name=campaign_name, project_id=project.id, created_at=datetime.utcnow()))
                                session.commit()
                                st.success(f"Campaign '{campaign_name}' added to '{project.project_name}'.")
                            except SQLAlchemyError as e: st.error(f"Error adding campaign: {str(e)}")
                        else: st.warning("Please enter a campaign name.")
                campaigns = session.query(Campaign).filter_by(project_id=project.id).all()
                if campaigns:
                    st.write("Campaigns:")
                    for campaign in campaigns: st.write(f"- {campaign.campaign_name}")
                else: st.info(f"No campaigns for {project.project_name} yet.")
        st.subheader("Set Active Project and Campaign")
        project_options = [p.project_name for p in projects]
        active_project = st.selectbox("Select Active Project", options=project_options, index=0)
        active_project_id = session.query(Project.id).filter_by(project_name=active_project).scalar()
        set_active_project_id(active_project_id)
        active_project_campaigns = session.query(Campaign).filter_by(project_id=active_project_id).all()
        if active_project_campaigns:
            campaign_options = [c.campaign_name for c in active_project_campaigns]
            active_campaign = st.selectbox("Select Active Campaign", options=campaign_options, index=0)
            active_campaign_id = session.query(Campaign.id).filter_by(campaign_name=active_campaign, project_id=active_project_id).scalar()
            set_active_campaign_id(active_campaign_id)
            st.success(f"Active Project: {active_project}, Active Campaign: {active_campaign}")
        else: st.warning(f"No campaigns available for {active_project}. Please add a campaign.")

def knowledge_base_page():
    with get_db_connection() as session:
        project_options = fetch_projects(session)
        if not project_options: return st.warning("No projects found. Please create a project first.")
        selected_project = st.selectbox("Select Project", options=project_options)
        project_id = int(selected_project.split(":")[0])
        set_active_project_id(project_id)
        kb_entry = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        with st.form("knowledge_base_form"):
            fields = ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']
            form_data = {field: st.text_input(field.replace('_', ' ').title(), value=getattr(kb_entry, field, '')) if field in ['kb_name', 'contact_name', 'contact_role', 'contact_email', 'product_name'] else st.text_area(field.replace('_', ' ').title(), value=getattr(kb_entry, field, '')) for field in fields}
            if st.form_submit_button("Save Knowledge Base"):
                try:
                    form_data['project_id'], form_data['created_at'] = project_id, datetime.utcnow()
                    if kb_entry:
                        for k, v in form_data.items(): setattr(kb_entry, k, v)
                    else: session.add(KnowledgeBase(**form_data))
                    session.commit()
                    st.success("Knowledge Base saved successfully!", icon="✅")
                except Exception as e: st.error(f"An error occurred while saving the Knowledge Base: {str(e)}")

def autoclient_ai_page():
    st.header("AutoclientAI - Automated Lead Generation")
    with st.expander("Knowledge Base Information", expanded=False):
        with get_db_connection() as session:
            kb_info = get_knowledge_base_info(session, get_active_project_id())
        if not kb_info: return st.error("Knowledge Base not found for the active project. Please set it up first.")
        st.json(kb_info)
    user_input = st.text_area("Enter additional context or specific goals for lead generation:", help="This information will be used to generate more targeted search terms.")
    if st.button("Generate Search Term Groups and Email Templates", key="generate_groups_and_templates"):
        with st.spinner("Generating search term groups and email templates..."):
            with get_db_connection() as session:
                search_term_groups, email_templates = generate_search_term_groups_and_templates(kb_info, user_input)
            if not search_term_groups: return st.error("Failed to generate search term groups and email templates. Please try again.")
            st.session_state.search_term_groups, st.session_state.email_templates = search_term_groups, email_templates
        st.success("Search term groups and email templates generated successfully!")
        st.subheader("Generated Search Term Groups and Email Templates")
        for group_name, terms in search_term_groups.items():
            with st.expander(f"Group: {group_name}"):
                st.write(f"**Search Terms:**\n{', '.join(terms)}")
                st.write("**Email Template:**")
                st.write(f"Subject: {email_templates[group_name]['subject']}")
                st.write(f"Body: {email_templates[group_name]['body']}")
    if st.button("Start Automation", key="start_automation"):
        st.session_state.automation_status, st.session_state.automation_logs, st.session_state.total_leads_found, st.session_state.total_emails_sent = True, [], 0, 0
        st.success("Automation started!")
    if st.session_state.get('automation_status', False):
        st.subheader("Automation in Progress")
        progress_bar, log_container, leads_container, analytics_container = st.progress(0), st.empty(), st.empty(), st.empty()
        try:
            with get_db_connection() as session: ai_automation_loop(session, log_container, leads_container)
        except Exception as e:
            st.error(f"An error occurred in the automation process: {str(e)}")
            st.session_state.automation_status = False
    if not st.session_state.get('automation_status', False) and st.session_state.get('automation_logs'):
        st.subheader("Automation Results")
        st.metric("Total Leads Found", st.session_state.total_leads_found)
        st.metric("Total Emails Sent", st.session_state.total_emails_sent)
        st.subheader("Automation Logs")
        st.text_area("Logs", "\n".join(st.session_state.automation_logs), height=300)
    if 'email_logs' in st.session_state:
        st.subheader("Email Sending Logs")
        df_logs = pd.DataFrame(st.session_state.email_logs)
        st.dataframe(df_logs)
        success_rate = (df_logs['Status'] == 'sent').mean() * 100
        st.metric("Email Sending Success Rate", f"{success_rate:.2f}%")
    st.subheader("Debug Information")
    st.json(st.session_state)
    st.write("Current function:", autoclient_ai_page.__name__)
    st.write("Session state keys:", list(st.session_state.keys()))

def optimize_existing_groups(kb_info):
    try:
        with get_db_connection() as session:
            search_terms = session.query(SearchTerm).all()
            terms = [term.term for term in search_terms]
            if not terms: return st.info("No search terms available to optimize.")
            classification = classify_search_terms(session, terms, kb_info)
            st.write("**Optimized Search Term Groups:**")
            st.json(classification)
    except Exception as e: st.error(f"An error occurred: {str(e)}")

def ai_automation_loop(session, log_container, leads_container):
    st.session_state.automation_logs, total_search_terms, total_emails_sent = [], 0, 0
    while st.session_state.get('automation_status', False):
        try:
            log_container.info("Starting automation cycle")
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                log_container.warning("Knowledge Base not found. Skipping cycle.")
                time.sleep(3600)
                continue
            base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
            optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            st.subheader("Optimized Search Terms")
            st.write(", ".join(optimized_terms))
            if st.button("Confirm Search Terms", key="confirm_search_terms"):
                total_search_terms = len(optimized_terms)
                classified_terms = classify_search_terms(session, optimized_terms, kb_info)
                progress_bar = st.progress(0)
                for idx, (group_name, terms) in enumerate(classified_terms.items()):
                    if group_name == "low_quality_search_terms": continue
                    reference_template = session.query(EmailTemplate).filter_by(template_name="REFERENCE").first()
                    if not reference_template: continue
                    adjusted_template = adjust_email_template_api(session, reference_template.body_content, f"Adjust for: {', '.join(terms[:5])}", kb_info)
                    if not adjusted_template or 'body' not in adjusted_template: continue
                    new_template = EmailTemplate(template_name=f"{reference_template.template_name}_{group_name}", subject=adjusted_template['subject'], body_content=adjusted_template['body'], project_id=get_active_project_id())
                    session.add(new_template)
                    session.commit()
                    results = manual_search(session, terms, 10)
                    new_leads = [(res['Lead ID'], res['Email']) for res in results['results'] if save_lead(session, res['Email'], domain=get_domain_from_url(res['URL']))]
                    if new_leads:
                        from_email, reply_to = kb_info['contact_email'] or 'hello@indosy.com', kb_info['contact_email'] or 'eugproductions@gmail.com'
                        logs, sent_count = bulk_send_emails(session, new_template.id, from_email, reply_to, new_leads)
                        st.session_state.automation_logs.extend(logs)
                        total_emails_sent += sent_count
                    leads_container.text_area("New Leads Found", "\n".join([email for _, email in new_leads]), height=200)
                    progress_bar.progress((idx + 1) / len(classified_terms))
                st.success(f"Automation cycle completed. Total search terms: {total_search_terms}, Total emails sent: {total_emails_sent}")
                time.sleep(3600)
            else:
                log_container.warning("Please confirm the search terms to proceed.")
                continue
        except Exception as e:
            log_container.error(f"Critical error in automation cycle: {str(e)}")
            time.sleep(300)
    log_container.info("Automation stopped")

def update_log_display(log_container, logs):
    log_container.markdown(
        f"""
        <style>
        .log-container {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
            background-color: rgba(49, 51, 63, 0.1);
        }}
        .log-entry {{
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 0.25rem;
        }}
        </style>
        <div class="log-container">
            <h4>Latest Logs</h4>
            {"".join(f'<div class="log-entry">{log}</div>' for log in logs[-20:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def update_results_display(results_container, results):
    results_container.markdown(
        f"""
        <style>
        .results-container {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
            background-color: rgba(49, 51, 63, 0.1);
        }}
        .result-entry {{
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 0.25rem;
        }}
        </style>
        <div class="results-container">
            <h4>Found Leads ({len(results)})</h4>
            {"".join(f'<div class="result-entry"><strong>{res["Email"]}</strong><br>{res["URL"]}</div>' for res in results[-10:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def automation_control_panel_page():
    st.title("Automation Control Panel")

    # Status and Control
    col1, col2 = st.columns([2, 1])
    with col1:
        status = "Active" if st.session_state.get('automation_status', False) else "Inactive"
        st.metric("Automation Status", status)
    with col2:
        button_text = "Stop Automation" if st.session_state.get('automation_status', False) else "Start Automation"
        if st.button(button_text, use_container_width=True):
            st.session_state.automation_status = not st.session_state.get('automation_status', False)
            if st.session_state.automation_status:
                st.session_state.automation_logs = []
            st.rerun()

    # Quick Scan
    if st.button("Perform Quick Scan", use_container_width=True):
        with st.spinner("Performing quick scan..."):
            try:
                with get_db_connection() as session:
                    new_leads = session.query(Lead).filter(Lead.is_processed == False).count()
                    session.query(Lead).filter(Lead.is_processed == False).update({Lead.is_processed: True})
                    session.commit()
                st.success(f"Quick scan completed! Found {new_leads} new leads.")
            except Exception as e:
                st.error(f"An error occurred during quick scan: {str(e)}")

    # Real-Time Analytics
    st.subheader("Real-Time Analytics")
    try:
        with get_db_connection() as session:
            total_leads = session.query(Lead).count()
            emails_sent = session.query(EmailCampaign).count()
            col1, col2 = st.columns(2)
            col1.metric("Total Leads", total_leads)
            col2.metric("Emails Sent", emails_sent)
    except Exception as e:
        st.error(f"An error occurred while displaying analytics: {str(e)}")

    # Automation Logs
    st.subheader("Automation Logs")
    log_container = st.empty()
    update_log_display(log_container, st.session_state.get('automation_logs', []))

    # New Leads Found
    st.subheader("Recently Found Leads")
    leads_container = st.empty()

    if st.session_state.get('automation_status', False):
        st.info("Automation is currently running in the background.")
        try:
            with get_db_connection() as session:
                while st.session_state.get('automation_status', False):
                    kb_info = get_knowledge_base_info(session, get_active_project_id())
                    if not kb_info:
                        st.session_state.automation_logs.append("Knowledge Base not found. Skipping cycle.")
                        time.sleep(3600)
                        continue
                    
                    base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
                    optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
                    
                    new_leads_all = []
                    for term in optimized_terms:
                        results = manual_search(session, [term], 10)
                        new_leads = [(res['Email'], res['URL']) for res in results['results'] if save_lead(session, res['Email'], domain=get_domain_from_url(res['URL']))]
                        new_leads_all.extend(new_leads)
                        
                        if new_leads:
                            template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                            if template:
                                from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                                reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                                logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [lead[0] for lead in new_leads])
                                st.session_state.automation_logs.extend(logs)
                    
                    if new_leads_all:
                        leads_df = pd.DataFrame(new_leads_all, columns=['Email', 'URL'])
                        leads_container.dataframe(leads_df, hide_index=True)
                    else:
                        leads_container.info("No new leads found in this cycle.")
                    
                    update_log_display(log_container, st.session_state.get('automation_logs', []))
                    time.sleep(3600)  # Wait for an hour before next cycle
        except Exception as e:
            st.error(f"An error occurred in the automation process: {str(e)}")

def update_log_display(log_container, logs):
    if not logs:
        log_container.info("No logs to display yet.")
        return

    log_container.markdown(
        """
        <style>
        .log-container {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
        }
        .log-entry {
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            border-radius: 0.25rem;
            background-color: rgba(28, 131, 225, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    log_entries = "".join(f'<div class="log-entry">{log}</div>' for log in logs[-20:])
    log_container.markdown(f'<div class="log-container">{log_entries}</div>', unsafe_allow_html=True)

def display_email_tracking_statistics():
    st.title("Email Tracking Statistics")
    
    with get_db_connection() as session:
        # Fetch email campaigns
        email_campaigns = session.query(EmailCampaign).all()
        
        if not email_campaigns:
            st.info("No email campaigns found.")
            return
        
        # Create DataFrame
        df = pd.DataFrame([{
            'sent_at': ec.sent_at,
            'opened_at': ec.opened_at,
            'clicked_at': ec.clicked_at,
            'open_count': ec.open_count,
            'click_count': ec.click_count,
            'status': ec.status
        } for ec in email_campaigns])
        
        # Ensure 'sent_at' column exists and is not null
        if 'sent_at' not in df.columns or df['sent_at'].isnull().all():
            st.warning("No valid sent dates found in the data.")
            return
        
        # Convert 'sent_at' to datetime and extract date
        df['date'] = pd.to_datetime(df['sent_at']).dt.date
        
        # Group by date and calculate metrics
        daily_stats = df.groupby('date').agg({
            'sent_at': 'count',
            'opened_at': lambda x: x.notnull().sum(),
            'clicked_at': lambda x: x.notnull().sum(),
            'open_count': 'sum',
            'click_count': 'sum'
        }).reset_index()
        
        daily_stats.columns = ['date', 'emails_sent', 'unique_opens', 'unique_clicks', 'total_opens', 'total_clicks']
        daily_stats['open_rate'] = daily_stats['unique_opens'] / daily_stats['emails_sent']
        daily_stats['click_rate'] = daily_stats['unique_clicks'] / daily_stats['emails_sent']
        
        # Convert date to string for display
        daily_stats['date'] = daily_stats['date'].astype(str)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Emails Sent", daily_stats['emails_sent'].sum())
        col2.metric("Average Open Rate", f"{daily_stats['open_rate'].mean():.2%}")
        col3.metric("Average Click Rate", f"{daily_stats['click_rate'].mean():.2%}")
        
        # Plot daily statistics
        st.subheader("Daily Email Statistics")
        fig = px.line(daily_stats, x='date', y=['emails_sent', 'unique_opens', 'unique_clicks'], 
                      title="Daily Email Activity")
        st.plotly_chart(fig)
        
        # Plot open and click rates
        st.subheader("Open and Click Rates")
        fig = px.line(daily_stats, x='date', y=['open_rate', 'click_rate'], 
                      title="Daily Open and Click Rates")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig)
        
        # Display raw data
        st.subheader("Raw Data")
        st.dataframe(daily_stats)

        # Display detailed email tracking data
        st.subheader("Detailed Email Tracking Data")
        detailed_data = session.query(EmailCampaign).all()
        detailed_df = pd.DataFrame([{
            'Sent At': email.sent_at,
            'Email': email.lead.email,
            'Opened': 'Yes' if email.opened_at else 'No',
            'Clicked': 'Yes' if email.clicked_at else 'No',
            'Open Count': email.open_count,
            'Click Count': email.click_count
        } for email in detailed_data])

        # Allow sorting and filtering
        sort_by = st.selectbox("Sort by", detailed_df.columns)
        ascending = st.checkbox("Ascending", value=False)
        filtered_df = detailed_df.sort_values(by=sort_by, ascending=ascending)
        st.dataframe(filtered_df)

        # Add export functionality
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Detailed Data CSV",
            data=csv,
            file_name="email_tracking_data.csv",
            mime="text/csv"
        )

def main():
    st.set_page_config(
        page_title="Lead Generation Tool",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🎯"
    )

    st.sidebar.title("AutoclientAI")
    st.sidebar.markdown("Select a page to navigate through the application.")

    pages = {
        "🔍 Manual Search": manual_search_page,
        "📊 Bulk Send": bulk_send_page,
        "👥 View Leads": view_leads_page,
        "🔑 Search Terms": search_terms_page,
        "✉️ Email Templates": email_templates_page,
        "🚀 Projects & Campaigns": projects_campaigns_page,
        "📚 Knowledge Base": knowledge_base_page,
        "🤖 AutoclientAI": autoclient_ai_page,
        "⚙️ Automation Control": automation_control_panel_page,
        "📧 Email Logs": view_campaign_logs,
        "📊 Email Tracking": display_email_tracking_statistics
    }

    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=list(pages.keys()),
            icons=["search", "send", "people", "key", "envelope", "folder", "book", "robot", "gear", "list-check", "chart-line"],
            menu_icon="cast",
            default_index=0
        )

    # st.markdown("---")

    try:
        pages[selected]()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("An error occurred in the main function")

    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")

if __name__ == "__main__":
    main()

