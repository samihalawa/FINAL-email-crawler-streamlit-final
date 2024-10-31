import os, json, re, logging, asyncio, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid, aiohttp, urllib3, random, html, smtplib
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
from openai import OpenAI 
from typing import List, Optional
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
import plotly.express as px
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
from collections import defaultdict

DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT = map(os.getenv, ["SUPABASE_DB_HOST", "SUPABASE_DB_NAME", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD", "SUPABASE_DB_PORT"])
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal, Base = sessionmaker(bind=engine), declarative_base()

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

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
    campaign_name, campaign_type = Column(Text, default="Default Campaign"), Column(Text, default="Email")
    project_id = Column(BigInteger, ForeignKey('projects.id'), default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    auto_send, loop_automation, ai_customization = [Column(Boolean, default=False) for _ in range(3)]
    max_emails_per_group, loop_interval = [Column(BigInteger, default=40) for _ in range(1)] + [Column(BigInteger, default=60)]
    project = relationship("Project", back_populates="campaigns")
    email_campaigns = relationship("EmailCampaign", back_populates="campaign")
    search_terms = relationship("SearchTerm", back_populates="campaign")
    campaign_leads = relationship("CampaignLead", back_populates="campaign")

class CampaignLead(Base):
    __tablename__ = 'campaign_leads'
    id = Column(BigInteger, primary_key=True)
    campaign_id, lead_id = [Column(BigInteger, ForeignKey('campaigns.id')) for _ in range(1)] + [Column(BigInteger, ForeignKey('leads.id'))]
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
        return {attr: getattr(self, attr) for attr in vars(self) if isinstance(getattr(self, attr), (str, list))}

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
    template_name, subject, body_content = [Column(Text) for _ in range(3)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False)
    language = Column(Text, default='ES')
    campaign = relationship("Campaign")
    email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_id, lead_id, template_id = [Column(BigInteger, ForeignKey('campaigns.id')) for _ in range(1)] + [Column(BigInteger, ForeignKey('leads.id')) for _ in range(1)] + [Column(BigInteger, ForeignKey('email_templates.id'))]
    customized_subject, customized_content, original_subject, original_content, status = [Column(Text) for _ in range(5)]
    engagement_data, message_id, tracking_id = [Column(JSON) for _ in range(3)]
    sent_at, opened_at, clicked_at = [Column(DateTime(timezone=True)) for _ in range(3)]
    open_count, click_count = [Column(BigInteger, default=0) for _ in range(2)]
    ai_customized = Column(Boolean, default=False)
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
    name, email_template, description = [Column(Text) for _ in range(3)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_terms = relationship("SearchTerm", back_populates="group")

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    id = Column(BigInteger, primary_key=True)
    group_id = Column(BigInteger, ForeignKey('search_term_groups.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    term, category = [Column(Text) for _ in range(2)]
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
    campaign_id, search_term_id = [Column(BigInteger, ForeignKey('campaigns.id')) for _ in range(1)] + [Column(BigInteger, ForeignKey('search_terms.id'))]
    leads_gathered, emails_sent = [Column(BigInteger) for _ in range(2)]
    start_time, end_time = [Column(DateTime(timezone=True), server_default=func.now()) for _ in range(1)] + [Column(DateTime(timezone=True))]
    status, logs = Column(Text), Column(JSON)
    campaign = relationship("Campaign")
    search_term = relationship("SearchTerm")

class Settings(Base):
    __tablename__ = 'settings'
    id = Column(BigInteger, primary_key=True)
    name, setting_type = [Column(Text, nullable=False) for _ in range(2)]
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    id = Column(BigInteger, primary_key=True)
    name, email, provider = [Column(Text, nullable=False) for _ in range(3)]
    smtp_server, smtp_port, smtp_username, smtp_password = [Column(Text) for _ in range(4)]
    aws_access_key_id, aws_secret_access_key, aws_region = [Column(Text) for _ in range(3)]

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

def settings_page():
    st.title("Settings")
    
    with db_session() as session:
        # General Settings
        st.header("General Settings")
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        
        with st.form("general_settings"):
            api_key = st.text_input("OpenAI API Key", 
                                  value=general_settings.value.get('api_key', '') if general_settings else '',
                                  type="password")
            
            if st.form_submit_button("Save General Settings"):
                if not general_settings:
                    general_settings = Settings(
                        name="General Settings",
                        setting_type="general",
                        value={'api_key': api_key}
                    )
                    session.add(general_settings)
                else:
                    general_settings.value['api_key'] = api_key
                session.commit()
                st.success("Settings saved successfully!")
        
        # Email Settings
        st.header("Email Settings")
        handle_email_settings(session)

def check_required_settings():
    with db_session() as s: return bool(s.query(Settings).filter_by(setting_type='general').first() and s.query(EmailSettings).first())

def send_email_ses(s, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    if not (email_settings := s.query(EmailSettings).filter_by(email=from_email).first()): return None, None
    tracking_id = str(uuid.uuid4())
    tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
    tracked_body = add_tracking(body, tracking_id, tracking_pixel_url)
    try: return send_email_provider(email_settings, from_email, to_email, subject, tracked_body, charset, reply_to, ses_client), tracking_id
    except Exception as e: logging.error(f"Email error: {str(e)}"); return None, None

def save_email_campaign(s, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    if not (lead := s.query(Lead).filter_by(email=lead_email).first()): logging.error(f"Lead not found: {lead_email}"); return
    s.add(EmailCampaign(lead_id=lead.id, template_id=template_id, status=status, sent_at=sent_at, customized_subject=subject or "No subject", 
        message_id=message_id or f"unknown-{uuid.uuid4()}", customized_content=email_body or "No content", campaign_id=get_active_campaign_id(), tracking_id=str(uuid.uuid4())))
    s.commit()

def update_log(log_container, message, level='info'):
    icon = {'info': '🔵', 'success': '🟢', 'warning': '🟠', 'error': '🔴', 'email_sent': '🟣'}.get(level, '⚪')
    if 'log_entries' not in st.session_state: st.session_state.log_entries = []
    st.session_state.log_entries.append(f"{icon} {message}")
    log_container.markdown(f"<div style='height:300px;overflow-y:auto;font-family:monospace;font-size:0.8em;line-height:1.2'>{'<br>'.join(st.session_state.log_entries)}</div>", unsafe_allow_html=True)

def optimize_search_term(term, lang): 
    return f'"{term}" email OR contact OR "get in touch" site:.com' if lang == 'english' else f'"{term}" correo OR contacto OR "ponte en contacto" site:.es' if lang == 'spanish' else term

def shuffle_keywords(term): return ' '.join(random.sample(term.split(), len(term.split())))

def get_domain_from_url(url): return urlparse(url).netloc

def is_valid_email(email):
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email))

def extract_emails_from_html(html):
    valid_emails = []
    for email in re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', html):
        try: validate_email(email); valid_emails.append((email, email.split('@')[1].lower()))
        except EmailNotValidError: continue
    return valid_emails

def extract_info_from_page(soup):
    return (soup.find('meta', {'name': 'author'})['content'] if (name := soup.find('meta', {'name': 'author'})) else '',
            soup.find('meta', {'property': 'og:site_name'})['content'] if (company := soup.find('meta', {'property': 'og:site_name'})) else '',
            soup.find('meta', {'name': 'job_title'})['content'] if (job_title := soup.find('meta', {'name': 'job_title'})) else '')

def manual_search(s, terms, num_results, ignore_prev=True, opt_en=False, opt_es=False, shuffle=False, lang='ES', enable_email=True, log_container=None, from_email=None, reply_to=None, email_template=None):
    # Fix: Add error handling and validation
    if not terms:
        update_log(log_container, "No search terms provided", 'error')
        return {"total_leads": 0, "results": []}
        
    ua, results, domains = UserAgent(), [], defaultdict(int)
    
    try:
        campaign = s.query(Campaign).get(get_active_campaign_id())
        if not campaign:
            update_log(log_container, "No active campaign", 'error')
            return {"total_leads": 0, "results": []}
            
        max_per_domain = campaign.max_emails_per_group
        return process_search_terms(s, terms, num_results, ua, ignore_prev, opt_en, opt_es, 
                                  shuffle, lang, enable_email, log_container, from_email, 
                                  reply_to, email_template, max_per_domain)
    except Exception as e:
        logging.error(f"Manual search error: {str(e)}")
        update_log(log_container, f"Search error: {str(e)}", 'error')
        return {"total_leads": 0, "results": []}

def process_search_terms(s, terms, num_results, ua, ignore_prev, opt_en, opt_es, shuffle, lang, 
                        enable_email, log_container, from_email, reply_to, email_template, max_per_domain):
    results = []
    domains_processed = defaultdict(int)
    
    for term in terms:
        try:
            # Fix: Add term optimization based on language
            optimized_term = optimize_search_term(term, lang) if (opt_en and lang == 'EN') or (opt_es and lang == 'ES') else term
            if shuffle:
                optimized_term = shuffle_keywords(optimized_term)
                
            update_log(log_container, f"Searching: {optimized_term}", 'info')
            
            # Process search results
            for url in google_search(optimized_term, num_results=num_results, lang=lang):
                domain = get_domain_from_url(url)
                
                # Skip if domain limit reached
                if domains_processed[domain] >= max_per_domain:
                    continue
                    
                if process_url(s, url, ua, domain, ignore_prev, enable_email, from_email, 
                             reply_to, email_template, log_container):
                    domains_processed[domain] += 1
                    
        except Exception as e:
            logging.error(f"Error processing term {term}: {str(e)}")
            update_log(log_container, f"Term error: {str(e)}", 'error')
            continue
            
    return {"total_leads": len(results), "results": results}

def process_url(s, url, ua, domain, ignore_prev, enable_email, from_email, reply_to, email_template, log_container):
    try:
        # Fix: Add request timeout and retries
        response = safe_request(url, ua)
        if not response or response.status_code != 200:
            return False
            
        soup = BeautifulSoup(response.text, 'html.parser')
        emails = extract_emails_from_html(response.text)
        
        for email, email_domain in emails:
            if save_lead_with_source(s, email, url, domain, soup, ignore_prev):
                update_log(log_container, f"Found lead: {email}", 'success')
                
                if enable_email and email_template:
                    send_email_to_lead(s, email, from_email, reply_to, email_template, log_container)
                    
                return True
                
        return False
        
    except Exception as e:
        logging.error(f"URL processing error: {str(e)}")
        return False

def extract_page_info(soup):
    return {'name': extract_name(soup), 'company': extract_company(soup), 'job_title': extract_job_title(soup),
            'title': get_page_title(soup), 'description': get_page_description(soup), 'phones': extract_phone_numbers(soup),
            'social_links': extract_social_links(soup)}

def safe_request(url, ua):
    if not url.startswith(('http://', 'https://')): url = 'http://' + url
    try: return requests.get(url, timeout=10, verify=False, headers={'User-Agent': ua.random}, allow_redirects=True)
    except requests.RequestException as e: logging.error(f"Request failed for {url}: {str(e)}"); return None

def send_email_to_lead(s, lead, from_email, reply_to, template_id, log_container):
    try:
        if not (template := s.query(EmailTemplate).filter_by(id=int(template_id.split(":")[0])).first()): update_log(log_container, "Template not found", 'error'); return False
        wrapped_content = wrap_email_body(template.body_content)
        if (response := send_email_ses(s, from_email, lead.email, template.subject, wrapped_content, reply_to=reply_to))[0]:
            save_email_campaign(s, lead.email, template.id, 'sent', datetime.utcnow(), template.subject, response.get('MessageId'), wrapped_content)
            update_log(log_container, f"📧 Sent to: {lead.email}", 'email_sent'); return True
        save_email_campaign(s, lead.email, template.id, 'failed', datetime.utcnow(), template.subject, None, wrapped_content)
        update_log(log_container, f"❌ Failed: {lead.email}", 'error'); return False
    except Exception as e: update_log(log_container, f"Error sending to {lead.email}: {str(e)}", 'error'); return False

def generate_or_adjust_email_template(prompt, kb_info=None, current_template=None):
    messages = [{"role": "system", "content": "You are an AI assistant for email templates. Respond with JSON containing 'subject' and 'body' keys."},
                {"role": "user", "content": f"{'Adjust template:' if current_template else 'Create template:'} {prompt}\n{'Current:' if current_template else 'Guidelines:'}\n{current_template if current_template else 'Focus on benefits, address doubts, include CTAs'}"}]
    if kb_info: messages.append({"role": "user", "content": f"KB info: {json.dumps(kb_info)}"})
    response = openai_chat_completion(messages, function_name="generate_or_adjust_email_template")
    return json.loads(response) if isinstance(response, str) else response if isinstance(response, dict) else {"subject": "", "body": "<p>Failed to generate content.</p>"}

def fetch_leads_with_sources(s):
    try:
        query = s.query(Lead, func.string_agg(LeadSource.url, ', ').label('sources'), 
                       func.max(EmailCampaign.sent_at).label('last_contact'),
                       func.string_agg(EmailCampaign.status, ', ').label('email_statuses')).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(Lead.id)
        return pd.DataFrame([{**{k: getattr(lead, k) for k in ['id', 'email', 'first_name', 'last_name', 'company', 'job_title', 'created_at']},
                            'Source': sources, 'Last Contact': last_contact, 'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted',
                            'Delete': False} for lead, sources, last_contact, email_statuses in query.all()])
    except SQLAlchemyError as e: logging.error(f"DB error: {str(e)}"); return pd.DataFrame()

def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term, 
             func.count(distinct(Lead.id)).label('lead_count'),
                       func.count(distinct(EmailCampaign.id)).label('email_count'))
            .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
            .join(Lead, LeadSource.lead_id == Lead.id)
            .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
            .group_by(SearchTerm.term))
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

def add_search_term(s, term, campaign_id):
    try: s.add(SearchTerm(term=term, campaign_id=campaign_id, created_at=datetime.utcnow())); s.commit()
    except SQLAlchemyError as e: s.rollback(); logging.error(f"Error adding term: {str(e)}"); raise

def update_search_term_group(s, group_id, updated_terms):
    try:
        current_ids = {int(term.split(":")[0]) for term in updated_terms}
        for term in s.query(SearchTerm).filter(SearchTerm.group_id == group_id):
            term.group_id = group_id if term.id in current_ids else None
        for term_str in updated_terms:
            if term := s.query(SearchTerm).get(int(term_str.split(":")[0])):
                term.group_id = group_id
        s.commit()
    except Exception as e:
        s.rollback()
        logging.error(f"Group update error: {str(e)}")

def add_new_search_term(s, new_term, campaign_id, group_for_new_term):
    try: s.add(SearchTerm(term=new_term, campaign_id=campaign_id, created_at=datetime.utcnow(),
                         group_id=int(group_for_new_term.split(":")[0]) if group_for_new_term != "None" else None)); s.commit()
    except Exception as e: s.rollback(); logging.error(f"Term add error: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
    existing = session.query(SearchTermGroup).all()
    prompt = f"Group these terms:\n{', '.join([term.term for term in ungrouped_terms])}\nExisting groups: {', '.join([group.name for group in existing])}"  # Fix brackets and parentheses
    return openai_chat_completion([{"role": "system", "content": "You categorize search terms."}, {"role": "user", "content": prompt}],
                                function_name="ai_group_search_terms") or {}

def update_search_term_groups(s, grouped_terms):
    for group_name, terms in grouped_terms.items():
        if not (group := s.query(SearchTermGroup).filter_by(name=group_name).first()): 
            group = SearchTermGroup(name=group_name); s.add(group); s.flush()
        for term in terms:
            if search_term := s.query(SearchTerm).filter_by(term=term).first(): search_term.group_id = group.id
    s.commit()

def create_search_term_group(s, group_name):
    try: s.add(SearchTermGroup(name=group_name)); s.commit()
    except Exception as e: s.rollback(); logging.error(f"Group create error: {str(e)}")

def delete_search_term_group(s, group_id):
    try:
        group = s.query(SearchTermGroup).get(group_id)
        if group:
            s.query(SearchTerm).filter(SearchTerm.group_id == group_id).update({SearchTerm.group_id: None})
            s.delete(group)
            s.commit()
    except Exception as e:
        s.rollback()
        logging.error(f"Error deleting search term group: {str(e)}")

def ai_automation_loop(session, log_container, leads_container):
    """Enhanced automation loop with better control and monitoring"""
    with managed_automation_session():
        # Initialize tracking
        progress_update = track_automation_progress(total_steps=4)
        config = load_automation_config()
        
        while st.session_state.get('automation_status', False):
            try:
                # Step 1: Load Knowledge Base
                progress_update(1, "Loading Knowledge Base")
                kb_info = get_knowledge_base_info(session, get_active_project_id())
                if not kb_info:
                    log_container.warning("Knowledge Base not found")
                    time.sleep(config['retry_delay'])
                    continue
                
                # Step 2: Generate Search Terms
                progress_update(2, "Generating Search Terms")
                base_terms = fetch_search_terms(session)
                optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
                
                # Step 3: Process Search Terms
                progress_update(3, "Processing Search Terms")
                for term in optimized_terms:
                    process_search_term(session, term, config, leads_container)
                    
                # Step 4: Send Emails
                progress_update(4, "Sending Emails")
                process_email_queue(session, kb_info, config)
                
                # Update metrics
                update_automation_metrics(session)
                
                # Wait for next cycle
                time.sleep(config['cycle_delay'])
                
            except Exception as e:
                handle_automation_error(e, "automation_loop")
                
    return {
        'status': 'completed',
        'metrics': get_automation_metrics()
    }

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    with db_session() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first()
        if not general_settings or 'openai_api_key' not in general_settings.value:
            st.error("OpenAI API key not set. Please configure it in the settings.")
            return None

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
        st.error(f"Error in OpenAI API call: {str(e)}")
        with db_session() as session:
            log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id, model)
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

def save_lead(session, email, first_name=None, last_name=None, company=None, job_title=None, phone=None, url=None, search_term_id=None, created_at=None):
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
        lead_source = LeadSource(lead_id=lead.id, url=url, search_term_id=search_term_id)
        session.add(lead_source)
        campaign_lead = CampaignLead(campaign_id=get_active_campaign_id(), lead_id=lead.id, status="Not Contacted", created_at=datetime.utcnow())
        session.add(campaign_lead)
        session.commit()
        return lead
    except Exception as e:
        logging.error(f"Error saving lead: {str(e)}")
        session.rollback()
        return None

def save_lead_source(session, lead_id: int, search_term_id: int, url: str, 
                    email_domain: str, **kwargs) -> LeadSource:
    """Enhanced lead source saving with email domain tracking"""
    source = LeadSource(
        lead_id=lead_id,
        search_term_id=search_term_id,
        url=url,
        domain=get_domain_from_url(url),
        email_domain=email_domain,
        http_status=kwargs.get('http_status'),
        scrape_duration=kwargs.get('scrape_duration'),
        page_title=kwargs.get('page_title'),
        meta_description=kwargs.get('meta_description')
    )  # Add closing parenthesis
    session.add(source)
    return source

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
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  # Remove extra )
    return ' '.join(chunk for chunk in chunks if chunk)

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

def fetch_campaigns(session):
    return [f"{camp.id}: {camp.campaign_name}" for camp in session.query(Campaign).all()]

def fetch_projects(session):
    return [f"{project.id}: {project.project_name}" for project in session.query(Project).all()]

def fetch_email_templates(session):
    return [f"{t.id}: {t.template_name}" for t in session.query(EmailTemplate).all()]

def create_or_update_email_template(session, template_name, subject, body_content, template_id=None, is_ai_customizable=False, created_at=None, language='ES'):
    template = session.query(EmailTemplate).filter_by(id=template_id).first() if template_id else EmailTemplate(template_name=template_name, subject=subject, body_content=body_content, is_ai_customizable=is_ai_customizable, campaign_id=get_active_campaign_id(), created_at=created_at or datetime.utcnow())
    if template_id: template.template_name, template.subject, template.body_content, template.is_ai_customizable = template_name, subject, body_content, is_ai_customizable
    template.language = language
    session.add(template)
    session.commit()
    return template.id

safe_datetime_compare = lambda date1, date2: False if date1 is None or date2 is None else date1 > date2

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

def update_display(container, items: List[dict], title: str, item_key: str) -> None:
    container.markdown(
        f"""<style>.container{{max-height:400px;overflow-y:auto;border:1px solid rgba(49,51,63,0.2);
        border-radius:0.25rem;padding:1rem;background-color:rgba(49,51,63,0.1)}}
        .entry{{margin-bottom:0.5rem;padding:0.5rem;background-color:rgba(255,255,255,0.1);
        border-radius:0.25rem}}</style>
        <div class="container"><h4>{title} ({len(items)})</h4>
        {"".join(f'<div class="entry">{item[item_key]}</div>' for item in items[-20:])}</div>""",
        unsafe_allow_html=True
    )

def get_domain_from_url(url): return urlparse(url).netloc

def manual_search_page():
    # Add form key for proper rerendering
    form_key = "search_form_" + str(hash(frozenset(st.session_state.items())))
    
    with st.form(form_key):
        # Add form validation
        required_fields = {
            'search_terms': search_terms,
            'num_results': num_results
        }
        
        if enable_email_sending:
            required_fields.update({
                'template': template,
                'email_setting': email_setting
            })
        
        if not all(required_fields.values()):
            missing = [k for k, v in required_fields.items() if not v]
            st.error(f"Missing required fields: {', '.join(missing)}")
            return
    
    # Fix indentation and form structure
    st.title("Manual Search")
    
    if not check_required_settings():
        st.error("⚠️ Please configure settings first")
        st.button("Go to Settings", on_click=lambda: st.switch_page("settings"))
        return
    
    with st.form("search_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_terms = st_tags(
                label='Enter Search Terms',
                text='Press enter to add more terms',
                value=[],
                suggestions=['software engineer', 'data scientist', 'product manager'],
                maxtags=10,
                key='search_terms_input'
            )
            
            num_results = st.slider(
                "Results per term", 
                min_value=1, 
                max_value=100, 
                value=10,
                help="Number of results to fetch per search term"
            )

        with col2:
            enable_email_sending = st.checkbox(
                "Enable email sending",
                value=True,
                help="Automatically send emails to found leads"
            )
            
            ignore_previously_fetched = st.checkbox(
                "Skip processed domains",
                value=True,
                help="Skip domains that have been processed before"
            )
            
            language = st.selectbox(
                "Search Language",
                options=["ES", "EN"],
                help="Select search language"
            )
            
            optimize_terms = st.checkbox(
                f"Optimize for {language}",
                help="Add language-specific search operators"
            )

        # Email settings section
        if enable_email_sending:
            with st.expander("Email Settings"):
                with db_session() as session:
                    templates = fetch_email_templates(session)
                    email_settings = fetch_email_settings(session)
                
                if not templates:
                    st.error("No email templates available")
                    return
                    
                if not email_settings:
                    st.error("No email settings configured")
                    return

                template = st.selectbox(
                    "Email Template",
                    options=templates,
                    format_func=lambda x: x.split(":")[1].strip()
                )
                
                email_setting = st.selectbox(
                    "From Email",
                    options=email_settings,
                    format_func=lambda x: f"{x['name']} ({x['email']})"
                )
                
                if email_setting:
                    from_email = email_setting['email']
                    reply_to = st.text_input(
                        "Reply-To Email",
                        value=email_setting['email']
                    )

        # Form submission and search execution
        if st.form_submit_button("Start Search", use_container_width=True):
            if not search_terms:
                st.warning("Please enter at least one search term")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            results_container = st.empty()

            try:
                with db_session() as session:
                    results = manual_search(
                        session=session,
                        terms=search_terms,
                        num_results=num_results,
                        ignore_previously_fetched=ignore_previously_fetched,
                        optimize_english=optimize_terms and language == "EN",
                        optimize_spanish=optimize_terms and language == "ES",
                        language=language,
                        enable_email_sending=enable_email_sending,
                        log_container=log_container,
                        from_email=from_email if enable_email_sending else None,
                        reply_to=reply_to if enable_email_sending else None,
                        email_template=template if enable_email_sending else None
                    )
                    
                    if results['results']:
                        display_search_results(results, log_container)
                    else:
                        st.warning("No leads found matching the criteria")
                        
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")
                logging.exception("Search error")

def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term, 
             func.count(distinct(Lead.id)).label('lead_count'),
                       func.count(distinct(EmailCampaign.id)).label('email_count'))
            .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
            .join(Lead, LeadSource.lead_id == Lead.id)
            .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
            .group_by(SearchTerm.term))
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

def ai_automation_loop(session, log_container, leads_container):
    """Enhanced automation loop with better control and monitoring"""
    with managed_automation_session():
        # Initialize tracking
        progress_update = track_automation_progress(total_steps=4)
        config = load_automation_config()
        
        while st.session_state.get('automation_status', False):
            try:
                # Step 1: Load Knowledge Base
                progress_update(1, "Loading Knowledge Base")
                kb_info = get_knowledge_base_info(session, get_active_project_id())
                if not kb_info:
                    log_container.warning("Knowledge Base not found")
                    time.sleep(config['retry_delay'])
                    continue
                
                # Step 2: Generate Search Terms
                progress_update(2, "Generating Search Terms")
                base_terms = fetch_search_terms(session)
                optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
                
                # Step 3: Process Search Terms
                progress_update(3, "Processing Search Terms")
                for term in optimized_terms:
                    process_search_term(session, term, config, leads_container)
                    
                # Step 4: Send Emails
                progress_update(4, "Sending Emails")
                process_email_queue(session, kb_info, config)
                
                # Update metrics
                update_automation_metrics(session)
                
                # Wait for next cycle
                time.sleep(config['cycle_delay'])
                
            except Exception as e:
                handle_automation_error(e, "automation_loop")
                
    return {
        'status': 'completed',
        'metrics': get_automation_metrics()
    }

def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def get_page_description(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    return meta_desc['content'] if meta_desc else "No description found"

def is_valid_email(email):
    if not email: return False
    invalid_patterns = [r".*\.(png|jpg|jpeg|gif|css|js)$", r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*", r"^(test|prueba)@.*", r"^email@email\.com$", r".*@example\.com$", r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    if any(re.match(p, email, re.I) for p in invalid_patterns) or any(email.lower().endswith(f"@{d}") for d in typo_domains): return False
    try: validate_email(email); return True
    except EmailNotValidError: return False

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
        session.query(LeadSource).filter(LeadSource.lead_id == lead.id).delete()
        session.delete(lead)

    session.commit()
    return len(invalid_leads)

def perform_quick_scan(session):
    with st.spinner("Performing quick scan..."):
        terms = session.query(SearchTerm).order_by(func.random()).limit(3).all()
        email_setting = fetch_email_settings(session)[0] if fetch_email_settings(session) else None
        from_email = email_setting['email'] if email_setting else None
        reply_to = from_email
        email_template = session.query(EmailTemplate).first()
        res = manual_search(session, [term.term for term in terms], 10, True, False, True, "EN", True, st.empty(), from_email, reply_to, f"{email_template.id}: {email_template.template_name}" if email_template else None)
    st.success(f"Quick scan completed! Found {len(res['results'])} new leads.")
    return {"new_leads": len(res['results']), "terms_used": [term.term for term in terms]}

def bulk_send_emails(session,template_id,from_email,reply_to,leads,progress_bar=None,status_text=None,results=None,log_container=None):
    template=session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:return[],0
    logs,sent_count=[],0
    total_leads=len(leads)
    for idx,lead in enumerate(leads):
        try:
            validate_email(lead['Email'])
            response,tracking_id=send_email_ses(session,from_email,lead['Email'],template.subject,template.body_content,reply_to=reply_to)
            status='sent'if response else'failed'
            message_id=response.get('MessageId',f"{'sent'if response else'failed'}-{uuid.uuid4()}")
            if response:sent_count+=1
            log_message=f"{'✅'if response else'❌'}{'Email sent to'if response else'Failed to send email to'}: {lead['Email']}"
            save_email_campaign(session,lead['Email'],template_id,status,datetime.utcnow(),template.subject,message_id,template.body_content)
            logs.append(log_message)
            if progress_bar:progress_bar.progress((idx+1)/total_leads)
            if status_text:status_text.text(f"Processed {idx+1}/{total_leads} leads")
            if results is not None:results.append({"Email":lead['Email'],"Status":status})
            if log_container:log_container.text(log_message)
        except EmailNotValidError:logs.append(f"❌ Invalid email address: {lead['Email']}")
        except Exception as e:
            error_message=f"Error sending email to {lead['Email']}: {str(e)}"
            logging.error(error_message)
            save_email_campaign(session,lead['Email'],template_id,'failed',datetime.utcnow(),template.subject,f"error-{uuid.uuid4()}",template.body_content)
            logs.append(f"❌ Error sending email to: {lead['Email']} (Error: {str(e)})")
    return logs,sent_count

def view_campaign_logs():
    st.header("Email Logs")
    with db_session()as session:
        logs=fetch_all_email_logs(session)
        if logs.empty:st.info("No email logs found.");return
        st.write(f"Total emails sent: {len(logs)}")
        st.write(f"Success rate: {(logs['Status']=='sent').mean():.2%}")
        col1,col2=st.columns(2)
        start_date=col1.date_input("Start Date",value=logs['Sent At'].min().date())
        end_date=col2.date_input("End Date",value=logs['Sent At'].max().date())
        filtered_logs=logs[(logs['Sent At'].dt.date>=start_date)&(logs['Sent At'].dt.date<=end_date)]
        search_term=st.text_input("Search by email or subject")
        if search_term:filtered_logs=filtered_logs[filtered_logs['Email'].str.contains(search_term,case=False)|filtered_logs['Subject'].str.contains(search_term,case=False)]
        col1,col2,col3=st.columns(3)
        col1.metric("Emails Sent",len(filtered_logs))
        col2.metric("Unique Recipients",filtered_logs['Email'].nunique())
        col3.metric("Success Rate",f"{(filtered_logs['Status']=='sent').mean():.2%}")
        st.bar_chart(filtered_logs.resample('D',on='Sent At')['Email'].count())
        st.subheader("Detailed Email Logs")
        for _,log in filtered_logs.iterrows():
            with st.expander(f"{log['Sent At'].strftime('%Y-%m-%d %H:%M:%S')} - {log['Email']} - {log['Status']}"):
                st.write(f"**Subject:** {log['Subject']}")
                st.write(f"**Content Preview:** {log['Content'][:100]}...")
                if st.button("View Full Email",key=f"view_email_{log['ID']}"):st.components.v1.html(wrap_email_body(log['Content']),height=400,scrolling=True)
                if log['Status']!='sent':st.error(f"Status: {log['Status']}")
        logs_per_page=20
        page=st.number_input("Page",min_value=1,max_value=(len(filtered_logs)-1)//logs_per_page+1,value=1)
        start_idx=(page-1)*logs_per_page
        st.table(filtered_logs.iloc[start_idx:start_idx+logs_per_page][['Sent At','Email','Subject','Status']])
        if st.button("Export Logs to CSV"):st.download_button("Download CSV",filtered_logs.to_csv(index=False),"email_logs.csv","text/csv")

def fetch_all_email_logs(session):
    try:
        email_campaigns=session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead),joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({'ID':[ec.id for ec in email_campaigns],'Sent At':[ec.sent_at for ec in email_campaigns],'Email':[ec.lead.email for ec in email_campaigns],'Template':[ec.template.template_name for ec in email_campaigns],'Subject':[ec.customized_subject or"No subject"for ec in email_campaigns],'Content':[ec.customized_content or"No content"for ec in email_campaigns],'Status':[ec.status for ec in email_campaigns],'Message ID':[ec.message_id or"No message ID"for ec in email_campaigns],'Campaign ID':[ec.campaign_id for ec in email_campaigns],'Lead Name':[f"{ec.lead.first_name or''}{ec.lead.last_name or''}".strip()or"Unknown"for ec in email_campaigns],'Lead Company':[ec.lead.company or"Unknown"for ec in email_campaigns]})
    except SQLAlchemyError as e:logging.error(f"Database error in fetch_all_email_logs: {str(e)}");return pd.DataFrame()

def update_lead(session,lead_id,updated_data):
    try:
        lead=session.query(Lead).filter(Lead.id==lead_id).first()
        if lead:
            for k,v in updated_data.items():setattr(lead,k,v)
            return True
    except SQLAlchemyError as e:logging.error(f"Error updating lead {lead_id}: {str(e)}");session.rollback()
    return False

def delete_lead(session,lead_id):
    try:
        lead=session.query(Lead).filter(Lead.id==lead_id).first()
        if lead:session.delete(lead);return True
    except SQLAlchemyError as e:logging.error(f"Error deleting lead {lead_id}: {str(e)}");session.rollback()
    return False

def is_valid_email(email):
    try:validate_email(email);return True
    except EmailNotValidError:return False

def view_leads_page():
    st.title("Lead Management Dashboard")
    with db_session()as session:
        if'leads'not in st.session_state or st.button("Refresh Leads"):st.session_state.leads=fetch_leads_with_sources(session)
        if not st.session_state.leads.empty:
            total_leads=len(st.session_state.leads)
            contacted_leads=len(st.session_state.leads[st.session_state.leads['Last Contact'].notna()])
            conversion_rate=(st.session_state.leads['Last Email Status']=='sent').mean()
            st.columns(3)[0].metric("Total Leads",f"{total_leads:,}")
            st.columns(3)[1].metric("Contacted Leads",f"{contacted_leads:,}")
            st.columns(3)[2].metric("Conversion Rate",f"{conversion_rate:.2%}")
            st.subheader("Leads Table")
            search_term=st.text_input("Search leads by email, name, company, or source")
            filtered_leads=st.session_state.leads[st.session_state.leads.apply(lambda r:search_term.lower()in str(r).lower(),axis=1)]
            leads_per_page,page_number=20,st.number_input("Page",min_value=1,value=1)
            start_idx,end_idx=(page_number-1)*leads_per_page,page_number*leads_per_page
            edited_df=st.data_editor(filtered_leads.iloc[start_idx:end_idx],column_config={"ID":st.column_config.NumberColumn("ID",disabled=True),"Email":st.column_config.TextColumn("Email"),"First Name":st.column_config.TextColumn("First Name"),"Last Name":st.column_config.TextColumn("Last Name"),"Company":st.column_config.TextColumn("Company"),"Job Title":st.column_config.TextColumn("Job Title"),"Source":st.column_config.TextColumn("Source",disabled=True),"Last Contact":st.column_config.DatetimeColumn("Last Contact",disabled=True),"Last Email Status":st.column_config.TextColumn("Last Email Status",disabled=True),"Delete":st.column_config.CheckboxColumn("Delete")},disabled=["ID","Source","Last Contact","Last Email Status"],hide_index=True,num_rows="dynamic")
            if st.button("Save Changes",type="primary"):
                for i,r in edited_df.iterrows():
                    if r['Delete']:
                        if delete_lead_and_sources(session,r['ID']):st.success(f"Deleted lead: {r['Email']}")
                    else:
                        if update_lead(session,r['ID'],{k:r[k]for k in['Email','First Name','Last Name','Company','Job Title']}):st.success(f"Updated lead: {r['Email']}")
                st.rerun()
            st.download_button("Export Filtered Leads to CSV",filtered_leads.to_csv(index=False).encode('utf-8'),"exported_leads.csv","text/csv")
            st.subheader("Lead Growth")
            if'Created At'in st.session_state.leads.columns:st.line_chart(st.session_state.leads.groupby(pd.to_datetime(st.session_state.leads['Created At']).dt.to_period('M')).size().cumsum())
            else:st.warning("Created At data is not available for lead growth chart.")
            st.subheader("Email Campaign Performance")
            email_status_counts=st.session_state.leads['Last Email Status'].value_counts()
            st.plotly_chart(px.pie(values=email_status_counts.values,names=email_status_counts.index,title="Distribution of Email Statuses"),use_container_width=True)
        else:st.info("No leads available. Start by adding some leads to your campaigns.")

def fetch_leads_with_sources(session):
    try:
        query=session.query(Lead,func.string_agg(LeadSource.url,', ').label('sources'),func.max(EmailCampaign.sent_at).label('last_contact'),func.string_agg(EmailCampaign.status,', ').label('email_statuses')).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(Lead.id)
        return pd.DataFrame([{**{k:getattr(lead,k)for k in['id','email','first_name','last_name','company','job_title','created_at']},'Source':sources,'Last Contact':last_contact,'Last Email Status':email_statuses.split(', ')[-1]if email_statuses else'Not Contacted','Delete':False}for lead,sources,last_contact,email_statuses in query.all()])
    except SQLAlchemyError as e:logging.error(f"Database error in fetch_leads_with_sources: {str(e)}");return pd.DataFrame()

def delete_lead_and_sources(session, lead_id):
    try:
        session.query(LeadSource).filter(LeadSource.lead_id == lead_id).delete()
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            session.delete(lead)
            return True
    except SQLAlchemyError as e:
        logging.error(f"Error deleting lead {lead_id} and its sources: {str(e)}")
        session.rollback()
    return False

def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term, 
                           func.count(distinct(Lead.id)).label('lead_count'),
                           func.count(distinct(EmailCampaign.id)).label('email_count'))
             .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
             .join(Lead, LeadSource.lead_id == Lead.id)
             .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
             .group_by(SearchTerm.term))
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

def add_search_term(session, term, campaign_id):
    try:
        new_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=datetime.utcnow())
        session.add(new_term)
        session.commit()
        return new_term.id
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")
        raise

def get_active_campaign_id():
    return st.session_state.get('active_campaign_id', 1)

def search_terms_page():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Search Terms Dashboard</h1>", unsafe_allow_html=True)
    with db_session() as session:
        search_terms_df = fetch_search_terms_with_lead_count(session)
        if not search_terms_df.empty:
            st.columns(3)[0].metric("Total Search Terms", len(search_terms_df))
            st.columns(3)[1].metric("Total Leads", search_terms_df['Lead Count'].sum())
            st.columns(3)[2].metric("Total Emails Sent", search_terms_df['Email Count'].sum())
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Search Term Groups", "Performance", "Add New Term", "AI Grouping", "Manage Groups"])
            
            with tab1:
                groups = session.query(SearchTermGroup).all()
                groups.append("Ungrouped")
                for group in groups:
                    with st.expander(group.name if isinstance(group, SearchTermGroup) else group, expanded=True):
                        group_id = group.id if isinstance(group, SearchTermGroup) else None
                        terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all() if group_id else session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
                        updated_terms = st_tags(
                            label="",
                            text="Add or remove terms",
                            value=[f"{term.id}: {term.term}" for term in terms],
                            suggestions=[term for term in search_terms_df['Term'] if term not in [f"{t.id}: {t.term}" for t in terms]],
                            key=f"group_{group_id}"
                        )
                        if st.button("Update", key=f"update_{group_id}"):
                            update_search_term_group(session, group_id, updated_terms)
                            st.success("Group updated successfully")
                            st.rerun()
            
            with tab2:
                col1, col2 = st.columns([3, 1])
                with col1:
                    chart_type = st.radio("Chart Type", ["Bar", "Pie"], horizontal=True)
                    fig = px.bar(search_terms_df.nlargest(10, 'Lead Count'), x='Term', y=['Lead Count', 'Email Count'], title='Top 10 Search Terms', labels={'value': 'Count', 'variable': 'Type'}, barmode='group') if chart_type == "Bar" else px.pie(search_terms_df, values='Lead Count', names='Term', title='Lead Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(search_terms_df.nlargest(5, 'Lead Count')[['Term', 'Lead Count', 'Email Count']], use_container_width=True)
            
            with tab3:
                col1, col2, col3 = st.columns([2,1,1])
                new_term = col1.text_input("New Search Term")
                campaign_id = get_active_campaign_id()
                group_for_new_term = col2.selectbox("Assign to Group", ["None"] + [f"{g.id}: {g.name}" for g in groups if isinstance(g, SearchTermGroup)], format_func=lambda x: x.split(":")[1] if ":" in x else x)
                if col3.button("Add Term", use_container_width=True) and new_term:
                    add_new_search_term(session, new_term, campaign_id, group_for_new_term)
                    st.success(f"Added: {new_term}")
                    st.rerun()

            with tab4:
                st.subheader("AI-Powered Search Term Grouping")
                ungrouped_terms = session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
                if ungrouped_terms:
                    st.write(f"Found {len(ungrouped_terms)} ungrouped search terms.")
                    if st.button("Group Ungrouped Terms with AI"):
                        with st.spinner("AI is grouping terms..."):
                            grouped_terms = ai_group_search_terms(session, ungrouped_terms)
                            update_search_term_groups(session, grouped_terms)
                            st.success("Search terms have been grouped successfully!")
                            st.rerun()
                else:
                    st.info("No ungrouped search terms found.")

            with tab5:
                st.subheader("Manage Search Term Groups")
                col1, col2 = st.columns(2)
                with col1:
                    new_group_name = st.text_input("New Group Name")
                    if st.button("Create New Group") and new_group_name:
                        create_search_term_group(session, new_group_name)
                        st.success(f"Created new group: {new_group_name}")
                        st.rerun()
                with col2:
                    group_to_delete = st.selectbox("Select Group to Delete", 
                                                   [f"{g.id}: {g.name}" for g in groups if isinstance(g, SearchTermGroup)],
                                                   format_func=lambda x: x.split(":")[1])
                    if st.button("Delete Group") and group_to_delete:
                        group_id = int(group_to_delete.split(":")[0])
                        delete_search_term_group(session, group_id)
                        st.success(f"Deleted group: {group_to_delete.split(':')[1]}")
                        st.rerun()

        else:
            st.info("No search terms available. Add some to your campaigns.")

def update_search_term_group(session, group_id, updated_terms):
    try:
        current_ids = {int(term.split(":")[0]) for term in updated_terms}
        for term in session.query(SearchTerm).filter(SearchTerm.group_id == group_id):
            term.group_id = group_id if term.id in current_ids else None
        for term_str in updated_terms:
            if term := session.query(SearchTerm).get(int(term_str.split(":")[0])):
                term.group_id = group_id
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Group update error: {str(e)}")

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        new_search_term = SearchTerm(term=new_term, campaign_id=campaign_id, created_at=datetime.utcnow())
        if group_for_new_term != "None":
            new_search_term.group_id = int(group_for_new_term.split(":")[0])
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
    existing = session.query(SearchTermGroup).all()
    prompt = f"Group these terms:\n{', '.join([term.term for term in ungrouped_terms])}\nExisting groups: {', '.join([group.name for group in existing])}"  # Fix brackets and parentheses
    return openai_chat_completion([{"role": "system", "content": "You categorize search terms."}, {"role": "user", "content": prompt}],
                                function_name="ai_group_search_terms") or {}

def update_search_term_groups(s, grouped_terms):
    for group_name, terms in grouped_terms.items():
        if not (group := s.query(SearchTermGroup).filter_by(name=group_name).first()): 
            group = SearchTermGroup(name=group_name); s.add(group); s.flush()
        for term in terms:
            if search_term := s.query(SearchTerm).filter_by(term=term).first(): search_term.group_id = group.id
    s.commit()

def create_search_term_group(s, group_name):
    try: s.add(SearchTermGroup(name=group_name)); s.commit()
    except Exception as e: s.rollback(); logging.error(f"Group create error: {str(e)}")

def delete_search_term_group(s, group_id):
    try:
        group = s.query(SearchTermGroup).get(group_id)
        if group:
            s.query(SearchTerm).filter(SearchTerm.group_id == group_id).update({SearchTerm.group_id: None})
            s.delete(group)
            s.commit()
    except Exception as e:
        s.rollback()
        logging.error(f"Error deleting search term group: {str(e)}")

def email_templates_page():
    st.title("Email Templates")
    
    with st.form("template_form"):
        template_name = st.text_input("Template Name")
        subject = st.text_input("Subject")
        body = st.text_area("Body Content", height=300)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(TEMPLATE_HELP)
        
        with col2:
            if st.button("Preview"):
                template = EmailTemplate(
                    template_name=template_name,
                    subject=subject, 
                    body_content=body
                )
                preview_subject, preview_body = preview_template(template)
                st.subheader("Preview")
                st.text(f"Subject: {preview_subject}")
                st.markdown(preview_body, unsafe_allow_html=True)
        
        if st.form_submit_button("Save Template"):
            with db_session() as session:
                template_id = create_or_update_email_template(
                    session, template_name, subject, body
                )
                st.success(f"Template saved successfully! ID: {template_id}")

def get_email_preview(session, template_id, from_email, reply_to):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if template:
        wrapped_content = wrap_email_body(template.body_content)
        return wrapped_content
    return "<p>Template not found</p>"

def fetch_all_search_terms(session):
    return session.query(SearchTerm).all()

def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

def get_email_template_by_name(session, template_name):
    return session.query(EmailTemplate).filter_by(template_name=template_name).first()

def bulk_send_page():
    st.title("Bulk Email Sending")
    with db_session() as session:
        templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)
        if not templates or not email_settings:
            st.error("No email templates or settings available. Please set them up first.")
            return

        template_option = st.selectbox("Email Template", options=templates, format_func=lambda x: x.split(":")[1].strip())
        template_id = int(template_option.split(":")[0])
        template = session.query(EmailTemplate).filter_by(id=template_id).first()
        
        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input("Subject", value=template.subject if template else "")
            email_setting_option = st.selectbox(
                "From Email", 
                options=email_settings, 
                format_func=lambda x: f"{x['name']} ({x['email']})"
            )
            if email_setting_option:
                from_email = email_setting_option['email']
                reply_to = st.text_input("Reply To", email_setting_option['email'])
            else:
                st.error("Selected email setting not found. Please choose a valid email setting.")
                return
        
        with col2:
            send_option = st.radio("Send to:", ["All Leads", "Specific Email", "Leads from Chosen Search Terms", "Leads from Search Term Groups"])
            specific_email = None
            selected_terms = None
            if send_option == "Specific Email":
                specific_email = st.text_input("Enter email")
            elif send_option == "Leads from Chosen Search Terms":
                search_terms_with_counts = fetch_search_terms_with_lead_count(session)
                selected_terms = st.multiselect("Select Search Terms", options=search_terms_with_counts['Term'].tolist())
                selected_terms = [term.split(" (")[0] for term in selected_terms]
            elif send_option == "Leads from Search Term Groups":
                groups = fetch_search_term_groups(session)
                selected_groups = st.multiselect("Select Search Term Groups", options=groups)
                if selected_groups:
                    group_ids = [int(group.split(':')[0]) for group in selected_groups]
                    selected_terms = fetch_search_terms_for_groups(session, group_ids)

        exclude_previously_contacted = st.checkbox("Exclude Previously Contacted Domains", value=True)

        st.markdown("### Email Preview")
        st.text(f"From: {from_email}\nReply-To: {reply_to}\nSubject: {subject}")
        st.components.v1.html(get_email_preview(session, template_id, from_email, reply_to), height=600, scrolling=True)

        leads = fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted)
        total_leads = len(leads)
        eligible_leads = [lead for lead in leads if lead.get('language', template.language) == template.language]
        contactable_leads = [lead for lead in eligible_leads if not (exclude_previously_contacted and lead.get('domain_contacted', False))]

        st.info(f"Total leads: {total_leads}\n"
                f"Leads matching template language ({template.language}): {len(eligible_leads)}\n"
                f"Leads to be contacted: {len(contactable_leads)}")

        if st.button("Send Emails", type="primary"):
            if not contactable_leads:
                st.warning("No leads found matching the selected criteria.")
                return
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            log_container = st.empty()
            logs, sent_count = bulk_send_emails(session, template_id, from_email, reply_to, contactable_leads, progress_bar, status_text, results, log_container)
            st.success(f"Emails sent successfully to {sent_count} leads.")
            st.subheader("Sending Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            success_rate = (results_df['Status'] == 'sent').mean()
            st.metric("Email Sending Success Rate", f"{success_rate:.2%}")

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

def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        return []

def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term, 
                           func.count(distinct(Lead.id)).label('lead_count'),
                           func.count(distinct(EmailCampaign.id)).label('email_count'))
             .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
             .join(Lead, LeadSource.lead_id == Lead.id)
             .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
             .group_by(SearchTerm.term))
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

def fetch_search_term_groups(session):
    return [f"{group.id}: {group.name}" for group in session.query(SearchTermGroup).all()]

def fetch_search_terms_for_groups(session, group_ids):
    terms = session.query(SearchTerm).filter(SearchTerm.group_id.in_(group_ids)).all()
    return [term.term for term in terms]

def ai_automation_loop(session, log_container, leads_container):
    """Enhanced automation loop with better control and monitoring"""
    with managed_automation_session():
        # Initialize tracking
        progress_update = track_automation_progress(total_steps=4)
        config = load_automation_config()
        
        while st.session_state.get('automation_status', False):
            try:
                # Step 1: Load Knowledge Base
                progress_update(1, "Loading Knowledge Base")
                kb_info = get_knowledge_base_info(session, get_active_project_id())
                if not kb_info:
                    log_container.warning("Knowledge Base not found")
                    time.sleep(config['retry_delay'])
                    continue
                
                # Step 2: Generate Search Terms
                progress_update(2, "Generating Search Terms")
                base_terms = fetch_search_terms(session)
                optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
                
                # Step 3: Process Search Terms
                progress_update(3, "Processing Search Terms")
                for term in optimized_terms:
                    process_search_term(session, term, config, leads_container)
                    
                # Step 4: Send Emails
                progress_update(4, "Sending Emails")
                process_email_queue(session, kb_info, config)
                
                # Update metrics
                update_automation_metrics(session)
                
                # Wait for next cycle
                time.sleep(config['cycle_delay'])
                
            except Exception as e:
                handle_automation_error(e, "automation_loop")
                
    return {
        'status': 'completed',
        'metrics': get_automation_metrics()
    }

def display_search_results(results, key_suffix):
    if not results: return st.warning("No results to display.")
    with st.expander("Search Results", expanded=True):
        st.markdown(f"### Total Leads Found: **{len(results)}**")
        for i, res in enumerate(results):
            with st.expander(f"Lead: {res['Email']}", key=f"lead_expander_{key_suffix}_{i}"):
                st.markdown(f"**URL:** [{res['URL']}]({res['URL']})  \n**Title:** {res['Title']}  \n**Description:** {res['Description']}  \n**Tags:** {', '.join(res['Tags'])}  \n**Lead Source:** {res['Lead Source']}  \n**Lead Email:** {res['Email']}")

def perform_quick_scan(session):
    with st.spinner("Performing quick scan..."):
        terms = session.query(SearchTerm).order_by(func.random()).limit(3).all()
        email_setting = fetch_email_settings(session)[0] if fetch_email_settings(session) else None
        from_email = email_setting['email'] if email_setting else None
        reply_to = from_email
        email_template = session.query(EmailTemplate).first()
        res = manual_search(session, [term.term for term in terms], 10, True, False, False, True, "EN", True, st.empty(), from_email, reply_to, f"{email_template.id}: {email_template.template_name}" if email_template else None)
    st.success(f"Quick scan completed! Found {len(res['results'])} new leads.")
    return {"new_leads": len(res['results']), "terms_used": [term.term for term in terms]}

def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"Optimize and expand these search terms for lead generation:\n{', '.join(base_terms)}\n\nConsider:\n1. Relevance to business and target market\n2. Potential for high-quality leads\n3. Variations and related terms\n4. Industry-specific jargon\n\nRespond with a JSON array of optimized terms."
    response = openai_chat_completion([{"role": "system", "content": "You're an AI specializing in optimizing search terms for lead generation. Be concise and effective."}, {"role": "user", "content": prompt}], function_name="generate_optimized_search_terms")
    return response.get('optimized_terms', base_terms) if isinstance(response, dict) else base_terms

def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term, 
                           func.count(distinct(Lead.id)).label('lead_count'),
                           func.count(distinct(EmailCampaign.id)).label('email_count'))
            .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
            .join(Lead, LeadSource.lead_id == Lead.id)
            .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
            .group_by(SearchTerm.term))
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

def fetch_leads_for_search_terms(session, search_term_ids) -> List[Lead]:
    return session.query(Lead).distinct().join(LeadSource).filter(LeadSource.search_term_id.in_(search_term_ids)).all()

def projects_campaigns_page():
    with db_session() as session:  # Fixed: session was undefined
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
                    except SQLAlchemyError as e:
                        st.error(f"Error adding project: {str(e)}")
                else:
                    st.warning("Please enter a project name.")
        st.subheader("Existing Projects and Campaigns")
        projects = session.query(Project).all()
        for project in projects:
            with st.expander(f"Project: {project.project_name}"):
                st.info("Campaigns share resources and settings within a project.")
                with st.form(f"add_campaign_form_{project.id}"):
                    campaign_name = st.text_input("Campaign Name", key=f"campaign_name_{project.id}")
                    if st.form_submit_button("Add Campaign"):
                        if campaign_name.strip():
                            try:
                                session.add(Campaign(campaign_name=campaign_name, project_id=project.id, created_at=datetime.utcnow()))
                                session.commit()
                                st.success(f"Campaign '{campaign_name}' added to '{project.project_name}'.")
                            except SQLAlchemyError as e:
                                st.error(f"Error adding campaign: {str(e)}")
                        else:
                            st.warning("Please enter a campaign name.")
                campaigns = session.query(Campaign).filter_by(project_id=project.id).all()
                st.write("Campaigns:" if campaigns else f"No campaigns for {project.project_name} yet.")
                for campaign in campaigns:
                    st.write(f"- {campaign.campaign_name}")
        st.subheader("Set Active Project and Campaign")
        project_options = [p.project_name for p in projects]
        if project_options:
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
            else:
                st.warning(f"No campaigns available for {active_project}. Please add a campaign.")
        else:
            st.warning("No projects found. Please add a project first.")

def knowledge_base_page():
    st.title("Knowledge Base")
    with db_session() as session:
        project_options = fetch_projects(session)
        if not project_options:
            return st.warning("No projects found. Please create a project first.")
        
        selected_project = st.selectbox("Select Project", options=project_options)
        project_id = int(selected_project.split(":")[0])
        set_active_project_id(project_id)
        kb_entry = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        
        with st.form("knowledge_base_form"):
            fields = [
                'kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 
                'contact_email', 'company_description', 'company_mission', 
                'company_target_market', 'company_other', 'product_name', 
                'product_description', 'product_target_customer', 'product_other', 
                'other_context', 'example_email'
            ]
            
            form_data = {}
            for field in fields:
                if field in ['kb_name', 'contact_name', 'contact_role', 'contact_email', 'product_name']:
                    form_data[field] = st.text_input(
                        field.replace('_', ' ').title(),
                        value=getattr(kb_entry, field, ''))  # Fixed: Added closing parenthesis
                else:
                    form_data[field] = st.text_area(
                        field.replace('_', ' ').title(),
                        value=getattr(kb_entry, field, ''))  # Fixed: Added closing parenthesis

            if st.form_submit_button("Save Knowledge Base"):
                try:
                    form_data.update({
                        'project_id': project_id,
                        'created_at': datetime.utcnow()
                    })
                    
                    if kb_entry:
                        for k, v in form_data.items():
                            setattr(kb_entry, k, v)
                    else:
                        session.add(KnowledgeBase(**form_data))
                    
                    session.commit()
                    st.success("Knowledge Base saved successfully!", icon="✅")
                except Exception as e:
                    st.error(f"An error occurred while saving the Knowledge Base: {str(e)}")

def autoclient_ai_page():
    st.header("AutoclientAI - Automated Lead Generation")
    with st.expander("Knowledge Base Information", expanded=False):
        with db_session() as session:
            kb_info = get_knowledge_base_info(session, get_active_project_id())
        if not kb_info: return st.error("Knowledge Base not found for the active project. Please set it up first.")
        st.json(kb_info)
    
    user_input = st.text_area("Enter additional context or specific goals for lead generation:", help="This information will be used to generate more targeted search terms.")
    
    if st.button("Generate Optimized Search Terms", key="generate_optimized_terms"):
        with st.spinner("Generating optimized search terms..."):
            with db_session() as session:
                base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
                optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            if optimized_terms:
                st.session_state.optimized_terms = optimized_terms
                st.success("Search terms optimized successfully!")
                st.subheader("Optimized Search Terms")
                st.write(", ".join(optimized_terms))
            else: st.error("Failed to generate optimized search terms. Please try again.")

    if st.button("Start Automation", key="start_automation"):
        st.session_state.update({"automation_status": True, "automation_logs": [], "total_leads_found": 0, "total_emails_sent": 0})
        st.success("Automation started!")

    if st.session_state.get('automation_status', False):
        st.subheader("Automation in Progress")
        progress_bar, log_container, leads_container, analytics_container = st.progress(0), st.empty(), st.empty(), st.empty()
        try:
            with db_session() as session:
                ai_automation_loop(session, log_container, leads_container)
        except Exception as e:
            st.error(f"Automation error: {str(e)}")
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

def update_search_terms(session, classified_terms):
    for group, terms in classified_terms.items():
        for term in terms:
            existing = session.query(SearchTerm).filter_by(term=term, project_id=get_active_project_id()).first()
            if existing: existing.group = group
            else: session.add(SearchTerm(term=term, group=group, project_id=get_active_project_id()))
    session.commit()

def update_results_display(results_container, results):
    results_container.markdown(f"""<style>.results-container{{max-height:400px;overflow-y:auto;border:1px solid rgba(49,51,63,0.2);border-radius:0.25rem;padding:1rem;background-color:rgba(49,51,63,0.1)}}.result-entry{{margin-bottom:0.5rem;padding:0.5rem;background-color:rgba(255,255,255,0.1);border-radius:0.25rem}}</style><div class="results-container"><h4>Found Leads ({len(results)})</h4>{"".join(f'<div class="result-entry"><strong>{res["Email"]}</strong><br>{res["URL"]}</div>' for res in results[-10:])}</div>""", unsafe_allow_html=True)

def automation_control_panel_page():
    st.title("Automation Control Panel")
    col1, col2 = st.columns([2, 1])
    with col1: st.metric("Automation Status", "Active" if st.session_state.get('automation_status', False) else "Inactive")
    with col2:
        button_text = "Stop Automation" if st.session_state.get('automation_status', False) else "Start Automation"
        if st.button(button_text, use_container_width=True):
            st.session_state.automation_status = not st.session_state.get('automation_status', False)
            if st.session_state.automation_status: st.session_state.automation_logs = []
            st.rerun()

    if st.button("Perform Quick Scan", use_container_width=True):
        with st.spinner("Performing quick scan..."):
            try:
                with db_session() as session:
                    new_leads = session.query(Lead).filter(Lead.is_processed == False).count()
                    session.query(Lead).filter(Lead.is_processed == False).update({Lead.is_processed: True})
                    session.commit()
                    st.success(f"Quick scan completed! Found {new_leads} new leads.")
            except Exception as e: st.error(f"Quick scan error: {str(e)}")

    st.subheader("Real-Time Analytics")
    try:
        with db_session() as session:
            total_leads = session.query(Lead).count()
            emails_sent = session.query(EmailCampaign).count()
            col1, col2 = st.columns(2)
            col1.metric("Total Leads", total_leads)
            col2.metric("Emails Sent", emails_sent)
    except Exception as e: st.error(f"Analytics error: {str(e)}")

    st.subheader("Automation Logs")
    log_container = st.empty()
    update_display(log_container, st.session_state.get('automation_logs', []), "Latest Logs", "log")

    st.subheader("Recently Found Leads")
    leads_container = st.empty()

    if st.session_state.get('automation_status', False):
        st.info("Automation is currently running in the background.")
        try:
            with db_session() as session:
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
                        new_leads = [(res['Email'], res['URL']) for res in results['results'] if save_lead(session, res['Email'], url=res['URL'])]
                        new_leads_all.extend(new_leads)

                        if new_leads:
                            template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                            if template:
                                from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                                reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                                logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': email} for email, _ in new_leads])
                                st.session_state.automation_logs.extend(logs)

                    if new_leads_all:
                        leads_df = pd.DataFrame(new_leads_all, columns=['Email', 'URL'])
                        leads_container.dataframe(leads_df, hide_index=True)
                    else: leads_container.info("No new leads found in this cycle.")

                    update_display(log_container, st.session_state.get('automation_logs', []), "Latest Logs", "log")
                    time.sleep(3600)
        except Exception as e:
            st.error(f"Automation error: {str(e)}")

def get_knowledge_base_info(session, project_id):
    kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb.to_dict() if kb else None

def generate_optimized_search_terms(session, base_terms, kb_info):
    return get_ai_response(f"Generate 5 optimized search terms based on: {', '.join(base_terms)}. Context: {kb_info}").split('\n')

def update_display(container, items, title, item_type):
    container.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True)
    for item in items[-10:]: container.text(item)

def get_search_terms(session):
    return [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]

def get_ai_response(prompt):
    return openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100).choices[0].text.strip()

def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": s.id, "name": s.name, "email": s.email} for s in settings]
    except Exception as e:
        logging.error(f"Email settings error: {e}")
        return []

def bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_bar=None, status_text=None, results=None, log_container=None):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template: return [], 0
    logs, sent_count = [], 0
    total_leads = len(leads)
    for i, lead in enumerate(leads):
        try:
            validate_email(lead['Email'])
            response, tracking_id = send_email_ses(session, from_email, lead['Email'], template.subject, template.body_content, reply_to=reply_to)
            status = 'sent' if response else 'failed'
            message_id = response.get('MessageId', f"{'sent' if response else 'failed'}-{uuid.uuid4()}")
            if response: sent_count += 1
            save_email_campaign(session, lead['Email'], template_id, status, datetime.utcnow(), template.subject, message_id, template.body_content)
            log_msg = f"{'✅' if response else '❌'} Email {'sent to' if response else 'failed for'}: {lead['Email']}"
            logs.append(log_msg)
            if progress_bar: progress_bar.progress((i + 1) / total_leads)
            if status_text: status_text.text(f"Processed {i + 1}/{total_leads} leads")
            if results is not None: results.append({"Email": lead['Email'], "Status": status})
            if log_container: log_container.text(log_msg)
        except EmailNotValidError:
            logs.append(f"❌ Invalid email: {lead['Email']}")
        except Exception as e:
            error_msg = f"Error sending to {lead['Email']}: {str(e)}"
            logging.error(error_msg)
            save_email_campaign(session, lead['Email'], template_id, 'failed', datetime.utcnow(), template.subject, f"error-{uuid.uuid4()}", template.body_content)
            logs.append(f"❌ Error: {lead['Email']} ({str(e)})")
    return logs, sent_count

def wrap_email_body(body_content):
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Email</title><style>body{{font-family:Arial,sans-serif;line-height:1.6;color:#333;max-width:600px;margin:0 auto;padding:20px}}</style></head><body>{body_content}</body></html>"""

def fetch_sent_email_campaigns(session):
    try:
        campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({
            'ID': [c.id for c in campaigns],
            'Sent At': [c.sent_at.strftime("%Y-%m-%d %H:%M:%S") if c.sent_at else "" for c in campaigns],
            'Email': [c.lead.email for c in campaigns],
            'Template': [c.template.template_name for c in campaigns],
            'Subject': [c.customized_subject or "No subject" for c in campaigns],
            'Content': [c.customized_content or "No content" for c in campaigns],
            'Status': [c.status for c in campaigns],
            'Message ID': [c.message_id or "No message ID" for c in campaigns],
            'Campaign ID': [c.campaign_id for c in campaigns],
            'Lead Name': [f"{c.lead.first_name or ''} {c.lead.last_name or ''}".strip() or "Unknown" for c in campaigns],
            'Lead Company': [c.lead.company or "Unknown" for c in campaigns]
        })
    except SQLAlchemyError as e:
        logging.error(f"Database error: {str(e)}")
        return pd.DataFrame()

def display_logs(log_container, logs):
    if not logs: return log_container.info("No logs to display yet.")
    log_container.markdown("""<style>.log-container{max-height:300px;overflow-y:auto;border:1px solid rgba(49,51,63,0.2);border-radius:0.25rem;padding:1rem}.log-entry{margin-bottom:0.5rem;padding:0.5rem;border-radius:0.25rem;background-color:rgba(28,131,225,0.1)}</style>""", unsafe_allow_html=True)
    log_entries = "".join(f'<div class="log-entry">{log}</div>' for log in logs[-20:])
    log_container.markdown(f'<div class="log-container">{log_entries}</div>', unsafe_allow_html=True)

def view_sent_email_campaigns():
    st.header("Sent Email Campaigns")
    try:
        with db_session() as session:
            campaigns = fetch_sent_email_campaigns(session)
        if not campaigns.empty:
            st.dataframe(campaigns)
            st.subheader("Detailed Content")
            selected = st.selectbox("Select campaign to view details", campaigns['ID'].tolist())
            if selected:
                content = campaigns[campaigns['ID'] == selected]['Content'].iloc[0]
                st.text_area("Content", content if content else "No content available", height=300)
        else: st.info("No sent email campaigns found.")
    except Exception as e:
        st.error(f"Error fetching campaigns: {str(e)}")
        logging.error(f"Error in view_sent_email_campaigns: {str(e)}")

def handle_automation_error(e, context):
    error_msg = f"Error in {context}: {str(e)}"
    st.error(error_msg)
    if st.button("Retry"): st.experimental_rerun()

def main():
    st.set_page_config(page_title="Autoclient.ai | Lead Generation AI App", layout="wide", initial_sidebar_state="expanded", page_icon="")
    st.sidebar.title("AutoclientAI")
    st.sidebar.markdown("Select a page to navigate through the application.")
    pages = {
        "🔍 Manual Search": manual_search_page,
        "📦 Bulk Send": bulk_send_page, 
        "👥 View Leads": view_leads_page,
        "🔑 Search Terms": search_terms_page,
        "✉️ Email Templates": email_templates_page,
        "🚀 Projects & Campaigns": projects_campaigns_page,
        "📚 Knowledge Base": knowledge_base_page,
        "🤖 AutoclientAI": autoclient_ai_page,
        "⚙️ Automation Control": automation_control_panel_page,
        "📨 Email Logs": view_campaign_logs,
        "⚙️ Settings": settings_page,
        "📨 Sent Campaigns": view_sent_email_campaigns
    }
    with st.sidebar:
        selected = option_menu("Navigation", list(pages.keys()), icons=["search", "send", "people", "key", "envelope", "folder", "book", "robot", "gear", "list-check", "gear", "envelope-open"], menu_icon="cast", default_index=0)
    try: pages[selected]()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logging.exception("Main function error")
        st.write("Please refresh or contact support.")
    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")

@st.cache_resource
def get_db_session(): return SessionLocal()

@st.cache_data(ttl=300)
def fetch_cached_email_templates(session): return fetch_email_templates(session)

@st.cache_data(ttl=300)
def fetch_cached_email_settings(session): return fetch_email_settings(session)

TEMPLATE_HELP = """Available Variables:
{{name}} - Full name
{{first_name}} - First name
{{last_name}} - Last name
{{company}} - Company name
{{job_title}} - Job title"""

def process_template_variables(content: str, lead: Lead) -> str:
    variables = {
        'name': f"{lead.first_name or ''} {lead.last_name or ''}".strip() or "there",
        'first_name': lead.first_name or "there", 
        'last_name': lead.last_name or "",
        'company': lead.company or "your company",
        'job_title': lead.job_title or "professional"
    }
    return ''.join(content.replace(f"{{{{{var}}}}}", value) for var, value in variables.items())

def extract_name_from_url(url: str) -> tuple[str, str]:
    if 'linkedin.com/in/' not in url: return '', ''
    path = urlparse(url).path.strip('/').split('/')[-1].replace('-', ' ').split()
    return (path[0].title(), ' '.join(path[1:]).title()) if len(path) >= 2 else ('', '')

def save_lead_with_name(session: Session, email: str, url: str = None, **kwargs) -> Lead:
    lead = session.query(Lead).filter_by(email=email).first()
    if url and not (kwargs.get('first_name') and kwargs.get('last_name')):
        first_name, last_name = extract_name_from_url(url)
        if first_name and last_name: kwargs.update({'first_name': first_name, 'last_name': last_name})
    if lead:
        for key, value in kwargs.items():
            if value and not getattr(lead, key): setattr(lead, key, value)
    else:
        lead = Lead(email=email, created_at=datetime.utcnow(), **kwargs)
        session.add(lead)
    session.commit()
    return lead

def preview_template(template: EmailTemplate) -> tuple[str, str]:
    sample_lead = Lead(first_name="John", last_name="Doe", company="ACME Corp", job_title="Manager")
    return process_template_variables(template.subject, sample_lead), process_template_variables(template.body_content, sample_lead)

def handle_email_settings(s):
    email_settings = s.query(EmailSettings).all()
    if email_settings:
        st.subheader("Existing Email Settings")
        for setting in email_settings:
            with st.expander(f"{setting.name} ({setting.email})"):
                st.text(f"Provider: {setting.provider}")
                st.text(f"SMTP Server: {setting.smtp_server}")
    with st.form("email_settings_form"):
        name, email = st.text_input("Name"), st.text_input("Email")
        provider = st.selectbox("Provider", ["SMTP", "AWS SES"])
        smtp_config, aws_config = {}, {}
        if provider == "SMTP":
            smtp_config.update({
                "smtp_server": st.text_input("SMTP Server"),
                "smtp_port": st.text_input("SMTP Port"),
                "smtp_username": st.text_input("SMTP Username"),
                "smtp_password": st.text_input("SMTP Password", type="password")
            })
        else:
            aws_config.update({
                "aws_access_key_id": st.text_input("AWS Access Key ID"),
                "aws_secret_access_key": st.text_input("AWS Secret Access Key", type="password"),
                "aws_region": st.text_input("AWS Region")
            })
        if st.form_submit_button("Save Email Settings"):
            try:
                s.add(EmailSettings(name=name, email=email, provider=provider, **smtp_config, **aws_config))
                s.commit()
                st.success("✅ Email settings saved successfully!")
            except Exception as e:
                st.error(f"❌ Error saving email settings: {str(e)}")

def send_email_provider(email_settings, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    try:
        return send_smtp_email(email_settings, from_email, to_email, subject, body, charset, reply_to) if email_settings.provider == "SMTP" else send_ses_email(email_settings, from_email, to_email, subject, body, charset, reply_to, ses_client)
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        return None

def send_smtp_email(settings, from_email, to_email, subject, body, charset, reply_to):
    msg = MIMEMultipart()
    msg.update({'From': from_email, 'To': to_email, 'Subject': subject})
    if reply_to: msg.add_header('Reply-To', reply_to)
    msg.attach(MIMEText(body, 'html', charset))
    with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
        server.starttls()
        server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(msg)
    return {'MessageId': f'smtp-{uuid.uuid4()}'}

def send_ses_email(settings, from_email, to_email, subject, body, charset, reply_to, ses_client=None):
    if not ses_client:
        ses_client = boto3.client('ses', aws_access_key_id=settings.aws_access_key_id, aws_secret_access_key=settings.aws_secret_access_key, region_name=settings.aws_region)
    email_args = {
        'Source': from_email,
        'Destination': {'ToAddresses': [to_email]},
        'Message': {
            'Subject': {'Data': subject, 'Charset': charset},
            'Body': {'Html': {'Data': body, 'Charset': charset}}
        }
    }
    if reply_to: email_args['ReplyToAddresses'] = [reply_to]
    return ses_client.send_email(**email_args)

def add_tracking(body: str, tracking_id: str, tracking_pixel_url: str) -> str:
    tracking_pixel = f'<img src="{tracking_pixel_url}" width="1" height="1" />'
    soup = BeautifulSoup(body, 'html.parser')
    for link in soup.find_all('a'):
        if original_url := link.get('href'):
            link['href'] = f"https://autoclient-email-analytics.trigox.workers.dev/click?id={tracking_id}&url={urlencode({'url': original_url})}"
    modified_body = str(soup)
    return modified_body.replace('</body>', f'{tracking_pixel}</body>') if '</body>' in modified_body else modified_body + tracking_pixel

@contextmanager 
def managed_automation_session():
    try:
        st.session_state.automation_active = True
        yield
    finally:
        st.session_state.automation_active = False

def track_automation_progress(total_steps: int):
    progress = st.progress(0)
    status = st.empty()
    def update(step: int, message: str):
        progress.progress(step / total_steps)
        status.text(f"Step {step}/{total_steps}: {message}")
    return update

def load_automation_config():
    return {
        'batch_size': st.sidebar.slider("Batch Size", 5, 50, 20),
        'delay': st.sidebar.number_input("Delay (sec)", 1, 60, 5),
        'retries': st.sidebar.number_input("Retries", 1, 10, 3)
    }

def fetch_search_terms(session):
    return [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).filter(SearchTerm.is_active == True).all()]

def process_search_term(session, term: str, config: dict, leads_container):
    results = manual_search(session, [term], config['batch_size'])
    new_leads = [(res['Email'], res['URL']) for res in results['results'] if save_lead(session, res['Email'], url=res['URL'])]
    if new_leads:
        leads_df = pd.DataFrame(new_leads, columns=['Email', 'URL'])
        leads_container.dataframe(leads_df, hide_index=True)
    return new_leads

def process_email_queue(session, kb_info: dict, config: dict):
    template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
    if template:
        from_email = kb_info.get('contact_email') or 'hello@indosy.com'
        reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
        for _ in range(config['retries']):
            try:
                bulk_send_emails(session, template.id, from_email, reply_to)
                break
            except Exception as e:
                logger.error(f"Email error: {str(e)}")
                time.sleep(config['delay'])

def update_automation_metrics(session):
    try:
        metrics = {
            'total_leads': session.query(Lead).count(),
            'emails_sent': session.query(EmailCampaign).filter(EmailCampaign.sent_at.isnot(None)).count(),
            'success_rate': session.query(EmailCampaign).filter(EmailCampaign.status == 'sent').count() / session.query(EmailCampaign).filter(EmailCampaign.sent_at.isnot(None)).count()
        }
        st.session_state.automation_metrics = metrics
        return metrics
    except Exception as e:
        logging.error(f"Error updating metrics: {str(e)}")
        return {'total_leads': 0, 'emails_sent': 0, 'success_rate': 0}

def extract_name(soup):
    name_tags = soup.find_all(['h1', 'h2', 'h3'], class_=lambda x: x and 'name' in x.lower())
    return name_tags[0].text.strip() if name_tags else ""

def extract_company(soup):
    company_meta = soup.find('meta', {'property': 'og:site_name'})
    if company_meta: return company_meta['content']
    company_tags = soup.find_all(['span', 'div', 'p'], class_=lambda x: x and 'company' in x.lower())
    return company_tags[0].text.strip() if company_tags else ""

def extract_job_title(soup):
    title_tags = soup.find_all(['span', 'div', 'p'], class_=lambda x: x and ('title' in x.lower() or 'role' in x.lower()))
    return title_tags[0].text.strip() if title_tags else ""

def extract_phone_numbers(soup):
    phone_pattern = r'(\+?[\d\s-]{10,})'
    text = soup.get_text()
    return list(set(re.findall(phone_pattern, text)))

def extract_social_links(soup):
    social_patterns = ['linkedin.com', 'twitter.com', 'facebook.com', 'instagram.com']
    social_links = []
    for link in soup.find_all('a', href=True):
        if any(pattern in link['href'].lower() for pattern in social_patterns):
            social_links.append(link['href'])
    return social_links

if __name__ == "__main__": main()
