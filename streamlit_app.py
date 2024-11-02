import os, json, re, logging, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid, random, smtplib
from datetime import datetime
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from streamlit_option_menu import option_menu
from openai import OpenAI
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
from contextlib import contextmanager
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import distinct
import urllib3
from dotenv import load_dotenv
from email_validator import validate_email, EmailNotValidError
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import plotly.express as px
import asyncio, aiohttp

# Initialize database connection once
load_dotenv()  # Move this to the top before accessing env vars

DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("Database configuration incomplete. Check .env file.")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Initialize engine with proper configuration
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0, pool_pre_ping=True)
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
    kb_name, kb_bio, kb_values, contact_name, contact_role, contact_email = [Column(Text) for _ in range(6)]
    company_description, company_mission, company_target_market, company_other = [Column(Text) for _ in range(4)]
    product_name, product_description, product_target_customer, product_other = [Column(Text) for _ in range(4)]
    other_context, example_email = Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    project = relationship("Project", back_populates="knowledge_base")

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']}

# Update the Lead model to remove the domain attribute
class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    phone, first_name, last_name, company, job_title = [Column(Text) for _ in range(5)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_processed = Column(Boolean, default=False)  # Add this line
    campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead")
    email_campaigns = relationship("EmailCampaign", back_populates="lead")

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    project_id = Column(BigInteger, ForeignKey('projects.id'))  # Add project relationship
    template_name, subject, body_content = Column(Text), Column(Text), Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False)
    language = Column(Text, default='ES')
    campaign = relationship("Campaign")
    project = relationship("Project")  # Add project relationship
    email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    template_id = Column(BigInteger, ForeignKey('email_templates.id'))
    customized_subject = Column(Text)
    customized_content = Column(Text)  # Make sure this column exists
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
    language = Column(Text, default='ES')  # Add language column
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

# Replace the existing EmailSettings class with this unified Settings class
class Settings(Base):
    __tablename__ = 'settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    setting_type = Column(Text, nullable=False)  # 'general', 'email', etc.
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

@contextmanager
def safe_db_connection():
    """Enhanced database connection with retry logic"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            session = SessionLocal()
            yield session
            session.commit()
            break
        except SQLAlchemyError as e:
            if session:
                session.rollback()
            if attempt == max_retries - 1:
                logging.error(f"Database connection failed after {max_retries} attempts: {e}")
                raise
            time.sleep(retry_delay)
        finally:
            if session:
                session.close()

@contextmanager
def db_session():
    """Enhanced database session management with retry logic"""
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
    with safe_db_connection() as session:
        general_settings = session.query(Settings).filter_by(setting_type='general').first() or Settings(name='General Settings', setting_type='general', value={})
        st.header("General Settings")
        with st.form("general_settings_form"):
            openai_api_key = st.text_input("OpenAI API Key", value=general_settings.value.get('openai_api_key', ''), type="password")
            openai_api_base = st.text_input("OpenAI API Base URL", value=general_settings.value.get('openai_api_base', 'https://api.openai.com/v1'))
            openai_model = st.text_input("OpenAI Model", value=general_settings.value.get('openai_model', 'gpt-4o-mini'))
            if st.form_submit_button("Save General Settings"):
                general_settings.value = {'openai_api_key': openai_api_key, 'openai_api_base': openai_api_base, 'openai_model': openai_model}
                session.add(general_settings)
                session.commit()
                st.success("General settings saved successfully!")

        st.header("Email Settings")
        email_settings = session.query(EmailSettings).all()
        for setting in email_settings:
            with st.expander(f"{setting.name} ({setting.email})"):
                st.write(f"Provider: {setting.provider}")
                st.write(f"{'SMTP Server: ' + setting.smtp_server if setting.provider == 'smtp' else 'AWS Region: ' + setting.aws_region}")
                if st.button(f"Delete {setting.name}", key=f"delete_{setting.id}"):
                    session.delete(setting)
                    session.commit()
                    st.success(f"Deleted {setting.name}")
                    st.rerun()

        edit_id = st.selectbox("Edit existing setting", ["New Setting"] + [f"{s.id}: {s.name}" for s in email_settings])
        edit_setting = session.query(EmailSettings).get(int(edit_id.split(":")[0])) if edit_id != "New Setting" else None
        with st.form("email_setting_form"):
            name = st.text_input("Name", value=edit_setting.name if edit_setting else "", placeholder="e.g., Company Gmail")
            email = st.text_input("Email", value=edit_setting.email if edit_setting else "", placeholder="your.email@example.com")
            provider = st.selectbox("Provider", ["smtp", "ses"], index=0 if edit_setting and edit_setting.provider == "smtp" else 1)
            if provider == "smtp":
                smtp_server = st.text_input("SMTP Server", value=edit_setting.smtp_server if edit_setting else "", placeholder="smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535, value=edit_setting.smtp_port if edit_setting else 587)
                smtp_username = st.text_input("SMTP Username", value=edit_setting.smtp_username if edit_setting else "", placeholder="your.email@gmail.com")
                smtp_password = st.text_input("SMTP Password", type="password", value=edit_setting.smtp_password if edit_setting else "", placeholder="Your SMTP password")
            else:
                aws_access_key_id = st.text_input("AWS Access Key ID", value=edit_setting.aws_access_key_id if edit_setting else "", placeholder="AKIAIOSFODNN7EXAMPLE")
                aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password", value=edit_setting.aws_secret_access_key if edit_setting else "", placeholder="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
                aws_region = st.text_input("AWS Region", value=edit_setting.aws_region if edit_setting else "", placeholder="us-west-2")
            if st.form_submit_button("Save Email Setting"):
                setting_data = {k: v for k, v in locals().items() if k in ['name', 'email', 'provider', 'smtp_server', 'smtp_port', 'smtp_username', 'smtp_password', 'aws_access_key_id', 'aws_secret_access_key', 'aws_region'] and v is not None}
                try:
                    if edit_setting:
                        for k, v in setting_data.items():
                            setattr(edit_setting, k, v)
                    else:
                        new_setting = EmailSettings(**setting_data)
                        session.add(new_setting)
                    session.commit()
                    st.success("Email setting saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving email setting: {str(e)}")
                    session.rollback()

        # Add Database Settings section
        st.header("Database Settings")
        db_settings = session.query(Settings).filter_by(setting_type='database').first() or Settings(
            name='Database Settings',
            setting_type='database',
            value={}
        )
        
        with st.form("database_settings_form"):
            db_host = st.text_input("Database Host", value=db_settings.value.get('db_host', ''))
            db_name = st.text_input("Database Name", value=db_settings.value.get('db_name', ''))
            db_user = st.text_input("Database User", value=db_settings.value.get('db_user', ''))
            db_password = st.text_input("Database Password", value=db_settings.value.get('db_password', ''), type="password")
            db_port = st.text_input("Database Port", value=db_settings.value.get('db_port', '5432'))
            
            if st.form_submit_button("Save Database Settings"):
                db_settings.value = {
                    'db_host': db_host,
                    'db_name': db_name,
                    'db_user': db_user,
                    'db_password': db_password,
                    'db_port': db_port
                }
                session.add(db_settings)
                session.commit()
                st.success("Database settings saved successfully!")
                st.info("Restart the application for changes to take effect")

# 1. Optimized send_email_ses
def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    """Enhanced email sending with SES and SMTP support"""
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings:
        logging.error(f"No email settings found for {from_email}")
        return None, None

    tracking_id = str(uuid.uuid4())
    tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
    wrapped_body = wrap_email_body(body)
    tracked_body = wrapped_body.replace('</body>', f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/></body>')

    soup = BeautifulSoup(tracked_body, 'html.parser')
    for a in soup.find_all('a', href=True):
        original_url = a['href']
        tracked_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'click', 'url': original_url})}"
        a['href'] = tracked_url
    tracked_body = str(soup)

    try:
        test_response = requests.get(f"https://autoclient-email-analytics.trigox.workers.dev/test", timeout=5)
        if test_response.status_code != 200:
            logging.warning("Analytics worker is down. Using original URLs.")
            tracked_body = wrapped_body
    except requests.RequestException:
        logging.warning("Failed to reach analytics worker. Using original URLs.")
        tracked_body = wrapped_body

    try:
        if email_settings.provider == 'ses':
            if ses_client is None:
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
            logging.error(f"Unknown email provider: {email_settings.provider}")
            return None, None
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        return None, None

# 2. Optimized save_email_campaign  
def save_email_campaign(session: Session, email: str, template_id: int, status: str, sent_at: datetime, subject: str, message_id: str, body: str) -> bool:
    try:
        lead = session.query(Lead).filter_by(email=email).first()
        if not lead: return False
        template = session.query(EmailTemplate).get(template_id)
        if not template or template.project_id != get_active_project_id(): return False
        ec = EmailCampaign(
            campaign_id=get_active_campaign_id(),
            lead_id=lead.id,
            template_id=template_id,
            status=status,
            sent_at=sent_at,
            original_subject=subject,
            message_id=message_id,
            original_content=body,
            tracking_id=str(uuid.uuid4())
        )
        session.add(ec)
        session.commit()
        return True
    except Exception as e:
        logging.error(f"Save campaign error: {e}")
        session.rollback()
        return False

# 3. Optimized is_valid_email
def is_valid_email(email):
    """Single consolidated email validation function"""
    if email is None:
        return False
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^(test|prueba)@.*",
        r"^email@email\.com$",
        r".*@example\.com$", 
        r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"
    ]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    try:
        validate_email(email)
        return not (any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns) or 
                   any(email.lower().endswith(f"@{domain}") for domain in typo_domains))
    except EmailNotValidError:
        return False

# 4. Optimized extract_info_from_page
def extract_info_from_page(s):
    try: return {k:s.find('meta',{'name':k})['content'] if s.find('meta',{'name':k}) else '' 
                for k in ['author','og:site_name','job_title','description']}
    except Exception as e: logging.error(f"Extract error: {e}"); return {}

# 5. Optimized get_knowledge_base_info
def get_knowledge_base_info(session: Session, project_id: int) -> Optional[Dict[str,str]]:
    """Get knowledge base info for project"""
    try:
        kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        return kb.to_dict() if kb else None
    except Exception as e:
        logging.error(f"KB error: {e}")
        return None

# 6. Optimized update_search_terms
def update_search_terms(session: Session, terms_dict: Dict[str, List[str]]) -> None:
    """Update search terms with proper error handling"""
    try:
        for group, terms in terms_dict.items():
            for term in terms:
                existing = session.query(SearchTerm).filter_by(
                    term=term,
                    project_id=get_active_project_id()
                ).first()
                if existing:
                    existing.group = group
                else:
                    session.add(SearchTerm(
                        term=term,
                        group=group,
                        project_id=get_active_project_id()
                    ))
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error updating search terms: {e}")

# 7. Optimized fetch_email_settings
def fetch_email_settings(session: Session) -> List[Dict[str, Any]]:
    """Fetch email settings with error handling"""
    try:
        return [
            {"id": s.id, "name": s.name, "email": s.email}
            for s in session.query(EmailSettings).all()
        ]
    except Exception as e:
        logging.error(f"Error fetching settings: {e}")
        return []

# 8. Optimized save_lead
def save_lead(session, email, first_name=None, last_name=None, company=None, job_title=None, phone=None, url=None, search_term_id=None, created_at=None):
    """Enhanced lead saving with proper error handling"""
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
        
        if url and search_term_id:
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

# 9. Optimized wrap_email_body
def wrap_email_body(body: str, tracking_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
    """Enhanced email body wrapper with tracking and metadata support"""
    metadata_html = ""
    if metadata:
        metadata_html = f"""
        <div style='margin-bottom:20px;color:#666'>
            {chr(10).join(f'<strong>{k}:</strong> {v}<br>' for k,v in metadata.items())}
        </div>
        """
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <style>
        body {{
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #ffffff;
        }}
        a {{
            color: #007bff;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
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
        .button:hover {{
            background-color: #0056b3;
        }}
    </style>
</head>
<body>
    {metadata_html}
    {body}
    {f"""<img src="https://autoclient-email-analytics.trigox.workers.dev/track?id={tracking_id}&type=open" 
         width="1" height="1" style="display:none"/>""" if tracking_id else ""}
</body>
</html>'''

# 10. Optimized fetch_leads
def fetch_leads(
    session: Session,
    template_id: int,
    option: str,
    email: str,
    search_terms: List[str],
    exclude_contacted: bool
) -> List[Dict[str, Any]]:
    try:
        query = session.query(Lead)
        if option == "Specific Email":
            query = query.filter(Lead.email == email)
        elif option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and search_terms:
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(search_terms))
        
        if exclude_contacted:
            contacted_leads = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None)).subquery()
            query = query.outerjoin(contacted_leads, Lead.id == contacted_leads.c.lead_id).filter(contacted_leads.c.lead_id.is_(None))
        
        return [{"Email": lead.email, "ID": lead.id} for lead in query.all()]
    except Exception as e:
        logging.error(f"Error fetching leads: {str(e)}")
        return []

def update_log(c, msg, lvl='info'): 
    if 'log_entries' not in st.session_state: st.session_state.log_entries = []
    st.session_state.log_entries.append(f"{{'info':'🔵','success':'🟢','warning':'🟠','error':'🔴','email_sent':'🟣'}}.get(lvl,'⚪') {msg}")
    c.markdown(f"<div style='height:300px;overflow-y:auto;font-family:monospace'><br>".join(st.session_state.log_entries), unsafe_allow_html=True)

def optimize_search_term(term, lang): return f'"{term}" {{"en":"email OR contact site:.com","es":"correo OR contacto site:.es"}}.get(lang,term)'

def shuffle_keywords(term): return ' '.join(random.sample(term.split(),len(term.split())))

def get_domain_from_url(url): return urlparse(url).netloc

def manual_search(
    session: Session, 
    terms: List[str],
    num_results: int,
    ignore_previously_fetched: bool = True,
    optimize_english: bool = False,
    optimize_spanish: bool = False,
    shuffle_keywords_option: bool = False,
    language: str = 'ES',
    enable_email_sending: bool = True,
    log_container: Any = None,
    from_email: Optional[str] = None,
    reply_to: Optional[str] = None, 
    email_template: Optional[str] = None
) -> Dict[str, Any]:
    """Perform manual search for leads"""
    results = []
    try:
        for term in terms:
            # Apply optimizations
            if optimize_english and language == 'EN':
                term = f"{term} email OR contact site:.com"
            elif optimize_spanish and language == 'ES':
                term = f"{term} correo OR contacto site:.es"
            
            if shuffle_keywords_option:
                term = shuffle_keywords(term)

            # Perform search
            for url in google_search(term, num_results=num_results, lang=language):
                if ignore_previously_fetched:
                    domain = get_domain_from_url(url)
                    if session.query(LeadSource).filter(LeadSource.domain == domain).first():
                        continue
                
                # Extract info and save lead
                try:
                    page_info = extract_info_from_page(url)
                    if email := page_info.get('email'):
                        if is_valid_email(email):
                            lead = save_lead(session, email, url=url)
                            if lead:
                                results.append({
                                    'Email': email,
                                    'URL': url,
                                    'Company': page_info.get('company', '')
                                })
                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")
                    continue

        return {'results': results}
    except Exception as e:
        logging.error(f"Manual search error: {e}")
        return {'results': []}

def generate_or_adjust_email_template(
    prompt: str,
    kb_info: Optional[Dict[str, Any]] = None,
    current_template: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Generate or adjust email template using AI"""
    try:
        messages = [
            {"role": "system", "content": "You are an AI email template generator."},
            {"role": "user", "content": prompt}
        ]
        
        if kb_info:
            messages.append({
                "role": "user",
                "content": f"Context: {json.dumps(kb_info)}"
            })
            
        if current_template:
            messages.append({
                "role": "user", 
                "content": f"Adjust: {current_template}"
            })

        response = openai_chat_completion(
            messages, 
            function_name="generate_or_adjust_email_template"
        )
        
        if response and isinstance(response, str):
            template_data = json.loads(response)
            if all(k in template_data for k in ['subject', 'body']):
                return template_data
            
        return {"subject": "", "body": ""}
        
    except Exception as e:
        logging.error(f"Failed to generate template: {str(e)}")
        st.error(f"Failed to generate template: {str(e)}")
        return {"subject": "", "body": ""}

def fetch_leads_with_sources(session):
    try:
        return pd.DataFrame([{**{k:getattr(l,k) for k in ['id','email','first_name','last_name','company','job_title','created_at']},'Source':s,'Last Contact':lc,'Last Email Status':es.split(', ')[-1] if es else 'Not Contacted','Delete':False} for l,s,lc,es in session.query(Lead,func.string_agg(LeadSource.url,', ').label('sources'),func.max(EmailCampaign.sent_at).label('last_contact'),func.string_agg(EmailCampaign.status,', ').label('email_statuses')).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(Lead.id).all()])
    except SQLAlchemyError as e: return pd.DataFrame() if logging.error(f"Database error: {str(e)}") else None

def get_knowledge_base_info(s: Session, p: int) -> Optional[Dict[str,str]]:
    try: return (k:=s.query(KnowledgeBase).filter_by(project_id=p).first()) and k.to_dict()
    except Exception as e: logging.error(f"KB error: {e}"); return None

def update_search_terms(s,ct): [[setattr(e,'group',g) if (e:=s.query(SearchTerm).filter_by(term=t,project_id=get_active_project_id()).first()) else s.add(SearchTerm(term=t,group=g,project_id=get_active_project_id())) for t in ts] for g,ts in ct.items()]; s.commit()

def update_display(container, items, title, style_class="container"):
    """Unified display function for all display needs"""
    container.markdown(
        f"""
        <style>
            .{style_class} {{
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.25rem;                padding: 1rem;
                background-color: rgba(49, 51, 63, 0.1);
            }}
            .entry {{
                margin-bottom: 0.5rem;
                padding: 0.5rem;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 0.25rem;
            }}
        </style>
        <div class="{style_class}">
            <h4>{title} ({len(items)})</h4>
            {"".join(f'<div class="entry">{item}</div>' for item in items[-20:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def fetch_email_settings(s): return [{"id":x.id,"name":x.name,"email":x.email} for x in s.query(EmailSettings).all()] if True else logging.error("Error fetching settings")

def get_search_terms(s): return [t.term for t in s.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]

def generate_optimized_search_terms(s,bt,ki): return get_ai_response(f"Generate 5 optimized search terms based on: {', '.join(bt)}. Context: {ki}").split('\n')

def update_search_term_group(s, gid, ut):
    try:
        # Remove group_id for terms no longer in group
        [setattr(t, 'group_id', None) for t in s.query(SearchTerm).filter(SearchTerm.group_id==gid).all() 
         if t.id not in {int(x.split(":")[0]) for x in ut if ":" in x}]
        
        # Set group_id for terms in group
        [setattr(t, 'group_id', gid) for x in ut if ":" in x 
         if (t := s.query(SearchTerm).get(int(x.split(":")[0])))]
        
        s.commit()
        return True
    except Exception as e:
        s.rollback()
        logging.error(f"Error updating group: {str(e)}")
        return False

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try: session.add(SearchTerm(term=new_term, campaign_id=campaign_id, created_at=datetime.utcnow(), 
                    group_id=int(group_for_new_term.split(":")[0]) if group_for_new_term != "None" else None)) and session.commit()
    except Exception as e: session.rollback(); logging.error(f"Error adding search term: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
    try:
        existing_groups = session.query(SearchTermGroup).all()
        prompt = f"""Categorize these search terms into existing groups or suggest new ones:
{', '.join([term.term for term in ungrouped_terms])}
Existing groups: {', '.join([group.name for group in existing_groups])}
Respond with a JSON object: {{group_name: [term1, term2, ...]}}"""
        messages = [
            {"role": "system", "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."},
            {"role": "user", "content": prompt}
        ]
        return openai_chat_completion(messages, function_name="ai_group_search_terms") or {}
    except Exception as e:
        logging.error(f"Error in AI grouping: {str(e)}")
        return {}

def update_search_term_groups(session, grouped_terms):
    try:
        for group_name, terms in grouped_terms.items():
            group = session.query(SearchTermGroup).filter_by(name=group_name).first()
            if not group:
                group = SearchTermGroup(name=group_name)
                session.add(group)
                session.flush()
            for term in terms:
                if search_term := session.query(SearchTerm).filter_by(term=term).first():
                    search_term.group_id = group.id
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error updating groups: {str(e)}")

def create_search_term_group(session, group_name):
    try:
        session.add(SearchTermGroup(name=group_name))
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating group: {str(e)}")
        return False

def delete_search_term_group(session, group_id):
    try:
        if group := session.query(SearchTermGroup).get(group_id):
            session.query(SearchTerm).filter(SearchTerm.group_id == group_id).update({SearchTerm.group_id: None})
            session.delete(group)
            session.commit()
            return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting group: {str(e)}")
        return False

def get_domain_from_url(url): return urlparse(url).netloc

def get_page_title(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.title.string.strip() if soup.title else "No title found"
    except Exception as e:
        logging.error(f"Error getting title: {str(e)}")
        return "No title found"

def get_page_description(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc['content'] if meta_desc else "No description found"
    except Exception as e:
        logging.error(f"Error getting description: {str(e)}")
        return "No description found"

def extract_visible_text(soup):
    [s.extract() for s in soup(["script", "style"])]
    return ' '.join(p.strip() for l in soup.get_text().splitlines() if l.strip() for p in l.split() if p.strip())

def safe_datetime_compare(date1, date2): return False if None in (date1, date2) else date1 > date2

def fetch_campaigns(session): return [f"{c.id}: {c.campaign_name}" for c in session.query(Campaign).all()]

def fetch_projects(session): return [f"{p.id}: {p.project_name}" for p in session.query(Project).all()]

def fetch_email_templates(session): return [f"{t.id}: {t.template_name}" for t in session.query(EmailTemplate).all()]

def fetch_all_email_logs(session):
    try:
        logs = session.query(EmailCampaign).join(Lead).join(EmailTemplate)\
            .options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template))\
            .order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({
            'ID': [l.id for l in logs],
            'Sent At': [l.sent_at for l in logs],
            'Email': [l.email for l in logs],
            'Template': [l.template.template_name for l in logs],
            'Subject': [l.subject for l in logs],
            'Content': [l.content for l in logs],
            'Status': [l.status for l in logs],
            'Message ID': [l.message_id for l in logs],
            'Campaign ID': [l.campaign_id for l in logs],
            'Lead Name': [l.lead.name for l in logs],
            'Lead Company': [l.lead.company for l in logs]
        })
    except SQLAlchemyError as e:
        logging.error(f"Database error in email logs: {str(e)}")
        return pd.DataFrame()

def update_lead(session, lead_id, data):
    try:
        session.query(Lead).filter(Lead.id == lead_id).update(data)
        session.commit()
        return True
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error updating lead {lead_id}: {str(e)}")
        return False

def delete_lead(session, lead_id):
    try:
        session.query(Lead).filter(Lead.id == lead_id).delete()
        session.commit()
        return True
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error deleting lead {lead_id}: {str(e)}")
        return False

def perform_quick_scan(session):
    with st.spinner("Performing quick scan..."):
        terms = session.query(SearchTerm).order_by(func.random()).limit(3).all()
        email_cfg = fetch_email_settings(session)[0] if fetch_email_settings(session) else {}
        template = session.query(EmailTemplate).first()
        res = manual_search(
            session=session,
            search_terms=[t.term for t in terms],
            max_results=10,
            check_duplicates=True,
            use_proxy=False,
            use_cache=False,
            validate_emails=True,
            language="EN",
            save_results=True,
            progress_bar=st.empty(),
            from_email=email_cfg.get('email'),
            reply_to=email_cfg.get('email'),
            template_id=f"{template.id}: {template.template_name}" if template else None
        )
    st.success(f"Quick scan completed! Found {len(res['results'])} new leads.")
    return {"new_leads": len(res['results']), "terms_used": [t.term for t in terms]}

def email_templates_page():
    def validate_template(name, subject, body):
        return (True, "") if all([name and len(name) >= 3, subject, body and len(body) >= 10]) else (False, "Invalid")

    @st.cache_data(ttl=300)
    def get_cached_templates(session): return session.query(EmailTemplate).all()

    @st.cache_data(ttl=300)
    def get_cached_preview(template_id, body): return wrap_email_body(body)

    with safe_db_connection() as session:
        if 'email_template_cache' not in st.session_state.app_state:
            st.session_state.app_state['email_template_cache'] = get_cached_templates(session)
        
        templates = st.session_state.app_state['email_template_cache']
        st.header("Email Templates")
        
        with st.expander("Create New Template", False):
            name = st.text_input("Template Name", key="new_template_name")
            if st.checkbox("Use AI", key="use_ai"):
                prompt = st.text_area("AI Prompt", key="ai_prompt")
                kb = get_knowledge_base_info(session, get_active_project_id()) if st.checkbox("Use KB") else None
                if st.button("Generate", key="gen_ai"):
                    with st.spinner("Generating..."):
                        generated = generate_or_adjust_email_template(prompt, kb)
                        if name:
                            template = EmailTemplate(
                                template_name=name,
                                subject=generated.get("subject", ""),
                                body_content=generated.get("body", ""),
                                campaign_id=get_active_campaign_id(),
                                project_id=get_active_project_id()
                            )
                            session.add(template)
                            session.commit()
                            st.success("Created!")
                            st.session_state.app_state['email_template_cache'] = get_cached_templates(session)
                        st.subheader("Generated")
                        st.text(f"Subject: {generated.get('subject', '')}")
                        st.components.v1.html(wrap_email_body(generated.get('body', '')), height=400, scrolling=True)
            else:
                subject = st.text_input("Subject")
                body = st.text_area("Body", height=200)
                if st.button("Create") and all([name, subject, body]):
                    template = EmailTemplate(
                        template_name=name,
                        subject=subject,
                        body_content=body,
                        campaign_id=get_active_campaign_id(),
                        project_id=get_active_project_id()
                    )
                    session.add(template)
                    session.commit()
                    st.success("Created!")
                    st.session_state.app_state['email_template_cache'] = get_cached_templates(session)

        if templates:
            st.subheader("Existing Templates")
            for template in templates:
                with st.expander(f"Template: {template.template_name}"):
                    try:
                        c1, c2 = st.columns(2)
                        subject = c1.text_input("Subject", template.subject, key=f"s_{template.id}")
                        ai_custom = c2.checkbox("AI", template.is_ai_customizable, key=f"ai_{template.id}")
                        body = st.text_area("Body", template.body_content, height=200, key=f"b_{template.id}")
                        
                        preview = wrap_email_body(body)
                        st.components.v1.html(preview, height=400, scrolling=True)
                        
                        if st.button("Save", key=f"save_{template.id}"):
                            template.subject = subject
                            template.body_content = body
                            template.is_ai_customizable = ai_custom
                            session.commit()
                            st.success("Saved!")
                            st.session_state.app_state['email_template_cache'] = get_cached_templates(session)
                    except Exception as e:
                        st.error(f"Error updating template: {str(e)}")
                        session.rollback()

def save_lead_source(s, lead_id, term_id, url, status, duration, **meta):
    """Save lead source with metadata"""
    try:
        s.add(LeadSource(
            lead_id=lead_id,
            search_term_id=term_id,
            url=url,
            http_status=status,
            scrape_duration=duration,
            **{k: meta.get(k,'') for k in ['page_title','meta_description','content','tags','phone_numbers']}
        ))
        s.commit()
        return True
    except Exception as e:
        s.rollback()
        logging.error(f"Error saving lead source: {str(e)}")
        return False

def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
    """Log search term effectiveness metrics"""
    try:
        session.add(SearchTermEffectiveness(
            term=term,
            total_results=total_results,
            valid_leads=valid_leads,
            irrelevant_leads=total_results - valid_leads,
            blogs_found=blogs_found,
            directories_found=directories_found
        ))
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error logging search term effectiveness: {str(e)}")
        return False

def add_or_get_search_term(session, term, campaign_id, created_at=None):
    """Add new search term or get existing one"""
    try:
        existing = session.query(SearchTerm).filter_by(
            term=term,
            campaign_id=campaign_id
        ).first()
        
        if existing:
            return existing.id
            
        new_term = SearchTerm(
            term=term,
            campaign_id=campaign_id,
            created_at=created_at or datetime.utcnow()
        )
        session.add(new_term)
        session.commit()
        return new_term.id
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding/getting search term: {str(e)}")
        return None

# Session state management functions
get_active_project_id = lambda: st.session_state.get('active_project_id', 1)
get_active_campaign_id = lambda: st.session_state.get('active_campaign_id', 1)
set_active_project_id = lambda project_id: st.session_state.__setitem__('active_project_id', project_id)
set_active_campaign_id = lambda campaign_id: st.session_state.__setitem__('active_campaign_id', campaign_id)

def create_or_update_email_template(session, template_name, subject, body_content, template_id=None, is_ai_customizable=False, created_at=None, language='ES'):
    """Create or update email template"""
    try:
        if template_id:
            template = session.query(EmailTemplate).filter_by(id=template_id).first()
            if template:
                template.template_name = template_name
                template.subject = subject
                template.body_content = body_content
                template.is_ai_customizable = is_ai_customizable
        else:
            template = EmailTemplate(
                template_name=template_name,
                subject=subject,
                body_content=body_content,
                is_ai_customizable=is_ai_customizable,
                campaign_id=get_active_campaign_id(),
                created_at=created_at or datetime.utcnow()
            )
            session.add(template)
            
        template.language = language
        session.commit()
        return template.id
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating/updating template: {str(e)}")
        return None

def fetch_search_terms_with_lead_count(s):
    """Fetch search terms with lead and email counts"""
    try:
        return pd.DataFrame(
            s.query(
                SearchTerm.term,
                func.count(distinct(Lead.id)).label('lead_count'),
                func.count(distinct(EmailCampaign.id)).label('email_count')
            )
            .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
            .join(Lead, LeadSource.lead_id == Lead.id)
            .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
            .group_by(SearchTerm.term)
            .all(),
            columns=['Term', 'Lead Count', 'Email Count']
        )
    except Exception as e:
        logging.error(f"Error fetching search terms with counts: {str(e)}")
        return pd.DataFrame()

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    """Enhanced OpenAI chat completion with error handling and logging"""
    with safe_db_connection() as session:
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
        
        # Log the AI request
        with safe_db_connection() as session:
            log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id, model)
        
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    except Exception as e:
        st.error(f"Error in OpenAI API call: {str(e)}")
        with safe_db_connection() as session:
            log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id, model)
        return None

def log_ai_request(session, function_name, prompt, response, lead_id=None, email_campaign_id=None, model_used=None):
    """Log AI requests with proper error handling"""
    try:
        session.add(AIRequestLog(
            function_name=function_name,
            prompt=json.dumps(prompt),
            response=json.dumps(response) if response else None,
            lead_id=lead_id,
            email_campaign_id=email_campaign_id,
            model_used=model_used
        ))
        session.commit()
    except Exception as e:
        logging.error(f"Error logging AI request: {str(e)}")
        session.rollback()

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def make_request_with_retry(url, method='GET', **kwargs):
    """Make HTTP request with retry logic"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        response = session.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    except Exception as e:
        logging.error(f"Request failed: {str(e)}")
        raise

async def async_fetch_url(session, url):
    """Async URL fetcher"""
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return None

async def bulk_fetch_urls(urls):
    """Bulk URL fetcher using aiohttp"""
    async with aiohttp.ClientSession() as session:
        tasks = [async_fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )

def initialize_session_state():
    """Initialize session state variables"""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
            'email_template_cache': None,
            'automation_status': False,
            'automation_logs': [],
            'total_leads_found': 0,
            'total_emails_sent': 0,
            'log_entries': []
        }
    
    if 'active_project_id' not in st.session_state:
        st.session_state.active_project_id = 1
    
    if 'active_campaign_id' not in st.session_state:
        st.session_state.active_campaign_id = 1

def cleanup_resources():
    """Cleanup application resources"""
    try:
        engine.dispose()
        logging.info("Database connections cleaned up")
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

def handle_error(e: Exception, context: str = ""):
    """Centralized error handling"""
    error_id = str(uuid.uuid4())
    logging.error(f"Error ID: {error_id} - Context: {context} - Error: {str(e)}")
    st.error(f"""
        An error occurred: {str(e)}
        Error ID: {error_id}
        Please contact support if this persists.
    """)
    return error_id

def validate_data(data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, str]:
    """Validate data dictionary against required fields"""
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    return True, ""

def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    # Remove potentially dangerous characters
    text = re.sub(r'[<>]', '', text)
    # Limit length
    return text[:1000]

def main():
    st.set_page_config(
        page_title="Autoclient.ai | Lead Generation AI App",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=""
    )

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
        "🔄 Settings": settings_page,
        "📨 Sent Campaigns": view_sent_email_campaigns
    }

    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=list(pages.keys()),
            icons=["search", "send", "people", "key", "envelope", "folder", "book", "robot", "gear", "list-check", "gear", "envelope-open"],
            menu_icon="cast",
            default_index=0
        )

    try:
        initialize_session_state()
        pages[selected]()
    except Exception as e:
        handle_error(e, f"Page: {selected}")
    finally:
        cleanup_resources()

    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")

if __name__ == "__main__":
    setup_logging()
    initialize_session_state()
    main()
