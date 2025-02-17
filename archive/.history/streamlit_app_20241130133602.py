import os, json, re, logging, asyncio, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid, aiohttp, urllib3, random, html, smtplib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select, text, distinct, and_, Index, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed, wait_exponential
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
#database info
DB_HOST = os.getenv("SUPABASE_DB_HOST", "localhost")
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")
DB_PORT = os.getenv("SUPABASE_DB_PORT", "5432")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()
DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT = map(os.getenv, ["SUPABASE_DB_HOST", "SUPABASE_DB_NAME", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD", "SUPABASE_DB_PORT"])
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0, pool_pre_ping=True)
SessionLocal, Base = sessionmaker(bind=engine), declarative_base()

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

try:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0, pool_pre_ping=True)
    # Test the connection
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
except Exception as e:
    st.error(f"Failed to connect to database: {str(e)}")
    raise

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

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
    campaign_name = Column(Text, default="Default Campaign")
    campaign_type = Column(Text, default="Email")
    project_id = Column(BigInteger, ForeignKey('projects.id'), default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    auto_send = Column(Boolean, default=False)
    loop_automation = Column(Boolean, default=False)
    ai_customization = Column(Boolean, default=False)
    max_emails_per_group = Column(BigInteger, default=500)
    loop_interval = Column(BigInteger, default=60)
    project = relationship("Project", back_populates="campaigns")
    email_campaigns = relationship("EmailCampaign", back_populates="campaign")
    search_terms = relationship("SearchTerm", back_populates="campaign")
    campaign_leads = relationship("CampaignLead", back_populates="campaign")


# Replace the existing SearchProcess class with this updated version
class SearchProcess(Base):
    __tablename__ = 'search_processes'
    __table_args__ = (
        Index('idx_search_process_status', 'status'),
        Index('idx_search_process_created', 'created_at'),
    )
    id = Column(BigInteger, primary_key=True)
    search_terms = Column(JSON)  # Store list of search terms
    settings = Column(JSON)      # Store search settings
    status = Column(Text)        # 'running', 'completed', 'failed'
    results = Column(JSON)       # Store search results
    logs = Column(JSON)          # Store process logs
    total_leads_found = Column(BigInteger, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    campaign = relationship("Campaign")
 

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
    __table_args__ = (
        Index('idx_lead_email', 'email'),
        Index('idx_lead_created_at', 'created_at'),
        Index('idx_lead_status', 'status'),
    )
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    phone, first_name, last_name, company, job_title = [Column(Text) for _ in range(5)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_processed = Column(Boolean, default=False)
    last_contacted = Column(DateTime(timezone=True))
    notes = Column(Text)
    status = Column(Text, default='New')
    source = Column(Text)
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
    language = Column(Text, default='ES')  # Add language column
    campaign = relationship("Campaign")
    email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
    __tablename__ = 'email_campaigns'
    __table_args__ = (
        Index('idx_campaign_sent_at', 'sent_at'),
        Index('idx_campaign_status', 'status'),
        Index('idx_campaign_tracking', 'tracking_id'),
    )
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
    __table_args__ = (
        Index('idx_search_term', 'term'),
        Index('idx_search_created_at', 'created_at'),
    )
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
    __table_args__ = (
        Index('idx_source_url', 'url'),
        Index('idx_source_domain', 'domain'),
    )
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
        general_settings = session.query(Settings).filter_by(setting_type='general').first() or Settings(name='General Settings', setting_type='general', value={})
        st.header("General Settings")
        
        with st.form("general_settings_form"):
            # Predefined model configurations
            model_configs = {
                "Qwen 2.5 72B": {
                    "model": "Qwen/Qwen2.5-72B-Instruct",
                    "api_base": "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1/",
                    "api_key": "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"
                },
                "GPT-4": {
                    "model": "gpt-4",
                    "api_base": "https://api.openai.com/v1/",
                    "api_key": ""  # Will be filled by user input
                },
                "Custom Configuration": {
                    "model": "",
                    "api_base": "",
                    "api_key": ""
                }
            }
            
            selected_model = st.selectbox(
                "Select Model",
                options=list(model_configs.keys()),
                index=0
            )
            
            if selected_model == "Custom Configuration":
                openai_api_key = st.text_input("API Key", value=general_settings.value.get('openai_api_key', ''), type="password")
                openai_api_base = st.text_input("API Base URL", value=general_settings.value.get('openai_api_base', ''))
                openai_model = st.text_input("Model Name", value=general_settings.value.get('openai_model', ''))
            else:
                config = model_configs[selected_model]
                openai_api_key = st.text_input("API Key", 
                    value=config['api_key'] or general_settings.value.get('openai_api_key', ''),
                    type="password")
                openai_api_base = config['api_base']
                openai_model = config['model']
                
                if not config['api_key']:  # If API key is not predefined
                    st.info(f"Please enter your API key for {selected_model}")
            
            if st.form_submit_button("Save General Settings"):
                general_settings.value = {
                    'openai_api_key': openai_api_key,
                    'openai_api_base': openai_api_base,
                    'openai_model': openai_model
                }
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    try:
        email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
        if not email_settings:
            logging.error(f"No email settings found for {from_email}")
            return None, None

        if not all([to_email, subject, body]):
            logging.error("Missing required email parameters")
            return None, None

        tracking_id = str(uuid.uuid4())
        # Add a try-except to handle tracking pixel failures gracefully
        try:
            tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"  # Add missing }
            tracked_body = wrap_email_body(body).replace('</body>', f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/></body>')
        except Exception as e:
            logging.warning(f"Failed to add tracking pixel: {str(e)}")
            tracked_body = wrap_email_body(body)  # Use original body if tracking fails

        if email_settings.provider == 'ses':
            if not all([email_settings.aws_access_key_id, email_settings.aws_secret_access_key, email_settings.aws_region]):
                logging.error("Missing AWS credentials")
                return None, None

            try:
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
            except Exception as e:
                logging.error(f"AWS SES error: {str(e)}")
                return None, None

        elif email_settings.provider == 'smtp':
            if not all([email_settings.smtp_server, email_settings.smtp_port, email_settings.smtp_username, email_settings.smtp_password]):
                logging.error("Missing SMTP credentials")
                return None, None

            try:
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
            except Exception as e:
                logging.error(f"SMTP error: {str(e)}")
                return None, None
        else:
            logging.error(f"Unknown email provider: {email_settings.provider}")
            return None, None
    except Exception as e:
        logging.error(f"Error in send_email_ses after retries: {str(e)}")
        return None, None

def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    try:
        lead = session.query(Lead).filter_by(email=lead_email).first()
        if not lead:
            logging.error(f"Lead with email {lead_email} not found.")
            return None

        new_campaign = EmailCampaign(
            lead_id=lead.id,
            template_id=template_id,
            status=status,
            sent_at=sent_at,
            customized_subject=subject or "No subject",
            message_id=message_id or f"unknown-{uuid.uuid4()}",
            customized_content=email_body or "No content",
            campaign_id=get_active_campaign_id(),
            tracking_id=str(uuid.uuid4())
        )
        session.add(new_campaign)
        session.commit()
        return new_campaign
    except SQLAlchemyError as e:
        logging.error(f"Database error in save_email_campaign: {str(e)}")
        session.rollback()
        return None
    except Exception as e:
        logging.error(f"Error in save_email_campaign: {str(e)}")
        session.rollback()
        return None

def update_log(log_container, message, level='info'):
    icon = {
        'info': '🔵',
        'success': '🟢',
        'warning': '🟠',
        'error': '🔴',
        'email_sent': '🟣'
    }.get(level, '⚪')
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"{icon} [{timestamp}] {message}"
    
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    
    st.session_state.log_entries.append(log_entry)
    
    # Keep only last 1000 entries to prevent memory issues
    if len(st.session_state.log_entries) > 1000:
        st.session_state.log_entries = st.session_state.log_entries[-1000:]
    
    # Create scrollable log container with auto-scroll and improved styling
    log_html = f"""
        <div style='
            background: rgba(0,0,0,0.05);
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        '>
            <div id="log-container" style='
                height: 300px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 0.9em;
                line-height: 1.4;
                padding: 10px;
                background: rgba(255,255,255,0.8);
                border-radius: 3px;
            '>
                {'<br>'.join(st.session_state.log_entries)}
            </div>
        </div>
        <script>
            function scrollToBottom() {{
                var container = document.getElementById('log-container');
                if (container) {{
                    container.scrollTop = container.scrollHeight;
                }}
            }}
            // Initial scroll
            scrollToBottom();
            // Set up a mutation observer to watch for changes
            var observer = new MutationObserver(scrollToBottom);
            var container = document.getElementById('log-container');
            if (container) {{
                observer.observe(container, {{ childList: true, subtree: true }});
            }}
        </script>
    """
    log_container.markdown(log_html, unsafe_allow_html=True)

def optimize_search_term(search_term, language):
    if language == 'english':
        return f'"{search_term}" email OR contact OR "get in touch" site:.com'
    elif language == 'spanish':
        return f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es'
    return search_term

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def get_domain_from_url(url):
    return urlparse(url).netloc

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def extract_emails_from_html(html_content):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, html_content)

def extract_info_from_page(soup):
    name = soup.find('meta', {'name': 'author'})
    name = name['content'] if name else ''
    
    company = soup.find('meta', {'property': 'og:site_name'})
    company = company['content'] if company else ''
    
    job_title = soup.find('meta', {'name': 'job_title'})
    job_title = job_title['content'] if job_title else ''
    
    return name, company, job_title

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def manual_search(session, terms, num_results, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False, language='ES', enable_email_sending=True, log_container=None, from_email=None, reply_to=None, email_template=None, process_id=None):
    """
    Execute manual search with optional background processing support
    """
    try:
        ua = UserAgent()
        results = []
        total_leads = 0
        domains_processed = set()
        
        for original_term in terms:
            try:
                search_term_id = add_or_get_search_term(session, original_term, get_active_campaign_id())
                search_term = shuffle_keywords(original_term) if shuffle_keywords_option else original_term
                search_term = optimize_search_term(search_term, 'english' if optimize_english else 'spanish') if optimize_english or optimize_spanish else search_term
                
                log_message = f"🔍 Searching: {original_term}"
                if process_id:
                    update_process_log(session, process_id, log_message, 'info')
                elif log_container:
                    update_log(log_container, log_message)
                
                for url in google_search(search_term, num_results, lang=language):
                    try:
                        domain = get_domain_from_url(url)
                        if ignore_previously_fetched and domain in domains_processed:
                            continue
                        
                        if not url.startswith(('http://', 'https://')):
                            url = 'http://' + url
                        
                        response = requests.get(url, timeout=10, verify=False, headers={'User-Agent': ua.random})
                        response.raise_for_status()
                        
                        html_content = response.text
                        soup = BeautifulSoup(response.text, 'html.parser')
                        emails = extract_emails_from_html(html_content)
                        valid_emails = [email.lower() for email in emails if is_valid_email(email)]
                        
                        if valid_emails:
                            log_message = f"📍 Found {len(valid_emails)} emails on {url}"
                            if process_id:
                                update_process_log(session, process_id, log_message, 'success')
                            elif log_container:
                                update_log(log_container, log_message)
                            
                            name, company, job_title = extract_info_from_page(soup)
                            page_title = get_page_title(html_content)
                            page_description = get_page_description(html_content)
                            
                            for email in valid_emails:
                                try:
                                    lead = save_lead(session, email=email, first_name=name, company=company, job_title=job_title, url=url, search_term_id=search_term_id, created_at=datetime.utcnow())
                                    if lead:
                                        total_leads += 1
                                        log_message = f"📧 Found email: {email}"
                                        if process_id:
                                            update_process_log(session, process_id, log_message, 'success')
                                        elif log_container:
                                            update_log(log_container, log_message)
                                        
                                        results.append({
                                            'Email': email,
                                            'URL': url,
                                            'Lead Source': original_term,
                                            'Title': page_title,
                                            'Description': page_description,
                                            'Tags': [],
                                            'Name': name,
                                            'Company': company,
                                            'Job Title': job_title,
                                            'Search Term ID': search_term_id
                                        })

                                        if enable_email_sending and from_email and email_template:
                                            template = session.query(EmailTemplate).filter_by(id=int(email_template.split(":")[0])).first()
                                            if template:
                                                wrapped_content = wrap_email_body(template.body_content)
                                                response, tracking_id = send_email_ses(session, from_email, email, template.subject, wrapped_content, reply_to=reply_to)
                                                
                                                if response:
                                                    log_message = f"✉️ Email sent to: {email}"
                                                    if process_id:
                                                        update_process_log(session, process_id, log_message, 'email_sent')
                                                    elif log_container:
                                                        update_log(log_container, log_message, 'email_sent')
                                                    save_email_campaign(session, email, template.id, 'Sent', datetime.utcnow(), template.subject, response['MessageId'], wrapped_content)
                                                else:
                                                    log_message = f"❌ Failed to send email to: {email}"
                                                    if process_id:
                                                        update_process_log(session, process_id, log_message, 'error')
                                                    elif log_container:
                                                        update_log(log_container, log_message, 'error')
                                                    save_email_campaign(session, email, template.id, 'Failed', datetime.utcnow(), template.subject, None, wrapped_content)
                                except Exception as e:
                                    error_message = f"Error processing email {email}: {str(e)}"
                                    if process_id:
                                        update_process_log(session, process_id, error_message, 'error')
                                    elif log_container:
                                        update_log(log_container, error_message, 'error')
                                    continue
                        
                        domains_processed.add(domain)
                        
                    except Exception as e:
                        error_message = f"Error processing URL {url}: {str(e)}"
                        if process_id:
                            update_process_log(session, process_id, error_message, 'error')
                        elif log_container:
                            update_log(log_container, error_message, 'error')
                        continue
                    
            except Exception as e:
                error_message = f"Error processing term {original_term}: {str(e)}"
                if process_id:
                    update_process_log(session, process_id, error_message, 'error')
                elif log_container:
                    update_log(log_container, error_message, 'error')
                continue
            
        if total_leads > 0:
            success_message = f"✨ Found {total_leads} total leads"
            if process_id:
                update_process_log(session, process_id, success_message, 'success')
            elif log_container:
                update_log(log_container, success_message, 'success')
                
        return {"total_leads": total_leads, "results": results}
    except Exception as e:
        error_message = f"Error in manual_search: {str(e)}"
        if process_id:
            update_process_log(session, process_id, error_message, 'error')
        elif log_container:
            update_log(log_container, error_message, 'error')
        return {"total_leads": 0, "results": []}

def analyze_content_for_tags(content, max_tags=15):
    """
    Enhanced function to extract meaningful tags from content using NLP techniques
    """
    try:
        # Generate tags using OpenAI
        prompt = f"Extract up to {max_tags} relevant business tags from this content:\n\n{content[:1000]}..."
        messages = [
            {"role": "system", "content": "You are a business analyst extracting relevant tags from web content."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai_chat_completion(messages, function_name="extract_tags")
        
        if isinstance(response, list):
            return response[:max_tags]
        elif isinstance(response, str):
            # Split by common separators and clean up
            tags = [tag.strip() for tag in response.replace('\n', ',').split(',')]
            return [tag for tag in tags if tag][:max_tags]
        else:
            return []
    except Exception as e:
        logging.error(f"Error analyzing content for tags: {str(e)}")
        return []

def generate_or_adjust_email_template(prompt, kb_info=None, current_template=None):
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in creating and refining email templates for marketing campaigns. Always respond with a JSON object containing 'subject' and 'body' keys. The 'body' should contain HTML formatted content suitable for insertion into an email body."},
        {"role": "user", "content": f"""{'Adjust the following email template based on the given instructions:' if current_template else 'Create an email template based on the following prompt:'} {prompt}

        {'Current Template:' if current_template else 'Guidelines:'}
        {current_template if current_template else '1. Focus on benefits to the reader, address potential customer doubts, include clear CTAs, use a natural tone, and be concise.'}

        Respond with a JSON object containing 'subject' and 'body' keys. The 'body' should contain HTML formatted content suitable for insertion into an email body.

        Follow these guidelines:
        1. Use proper HTML tags for structuring the email content (e.g., <p>, <h1>, <h2>, etc.).
        2. Include inline CSS for styling where appropriate.
        3. Ensure the content is properly structured and easy to read.
        4. Include a call-to-action button or link with appropriate styling.
        5. Make the design responsive for various screen sizes.
        6. Do not include <html>, <head>, or <body> tags.

        Example structure:
        {{
          "subject": "Your compelling subject line here",
          "body": "<h1>Welcome!</h1><p>Your email content here...</p><a href='#' style='display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px;'>Call to Action</a>"
        }}"""}
    ]
    if kb_info:
        messages.append({"role": "user", "content": f"Consider this knowledge base information: {json.dumps(kb_info)}"})

    response = openai_chat_completion(messages, function_name="generate_or_adjust_email_template")
    
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "subject": "AI Generated Subject",
                "body": f"<p>{response}</p>"
            }
    elif isinstance(response, dict):
        return response
    else:
        return {"subject": "", "body": "<p>Failed to generate email content.</p>"}

def fetch_leads_with_sources(session):
    try:
        query = session.query(Lead, func.string_agg(LeadSource.url, ', ').label('sources'), func.max(EmailCampaign.sent_at).label('last_contact'), func.string_agg(EmailCampaign.status, ', ').label('email_statuses')).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(Lead.id)
        return pd.DataFrame([{**{k: getattr(lead, k) for k in ['id', 'email', 'first_name', 'last_name', 'company', 'job_title', 'created_at']}, 'Source': sources, 'Last Contact': last_contact, 'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted', 'Delete': False} for lead, sources, last_contact, email_statuses in query.all()])
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_leads_with_sources: {str(e)}")
        return pd.DataFrame()

def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term, 
                          func.count(distinct(Lead.id)).label('lead_count'),
                          func.count(distinct(EmailCampaign.id)).label('email_count'))
             .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
             .join(Lead, LeadSource.lead_id == Lead.id)
             .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
             .group_by(SearchTerm.term))
    df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    return df

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

def update_search_term_group(session, group_id, updated_terms):
    try:
        current_term_ids = set(int(term.split(":")[0]) for term in updated_terms)
        existing_terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all()
        for term in existing_terms:
            term.group_id = None if term.id not in current_term_ids else group_id
        for term_str in updated_terms:
            term = session.query(SearchTerm).get(int(term_str.split(":")[0]))
            if term: term.group_id = group_id
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_search_term_group: {str(e)}")

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        new_search_term = SearchTerm(term=new_term, campaign_id=campaign_id, created_at=datetime.utcnow(), group_id=int(group_for_new_term.split(":")[0]) if group_for_new_term != "None" else None)
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
    existing_groups = session.query(SearchTermGroup).all()
    prompt = f"Categorize these search terms into existing groups or suggest new ones:\n{', '.join([term.term for term in ungrouped_terms])}\n\nExisting groups: {', '.join([group.name for group in existing_groups])}\n\nRespond with a JSON object: {{group_name: [term1, term2, ...]}}"
    messages = [{"role": "system", "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."}, {"role": "user", "content": prompt}]
    response = openai_chat_completion(messages, function_name="ai_group_search_terms")
    return response if isinstance(response, dict) else {}

def update_search_term_groups(session, grouped_terms):
    for group_name, terms in grouped_terms.items():
        group = session.query(SearchTermGroup).filter_by(name=group_name).first() or SearchTermGroup(name=group_name)
        if not group.id: session.add(group); session.flush()
        for term in terms:
            search_term = session.query(SearchTerm).filter_by(term=term).first()
            if search_term: search_term.group_id = group.id
    session.commit()

def create_search_term_group(session, group_name):
    try:
        session.add(SearchTermGroup(name=group_name))
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating search term group: {str(e)}")

def delete_search_term_group(session, group_id):
    try:
        group = session.query(SearchTermGroup).get(group_id)
        if group:
            session.query(SearchTerm).filter(SearchTerm.group_id == group_id).update({SearchTerm.group_id: None})
            session.delete(group)
            session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting search term group: {str(e)}")

def ai_automation_loop(session, log_container, leads_container):
    automation_logs, total_search_terms, total_emails_sent = [], 0, 0
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

            total_search_terms = len(optimized_terms)
            progress_bar = st.progress(0)
            for idx, term in enumerate(optimized_terms):
                results = manual_search(session, [term], 10, ignore_previously_fetched=True)
                new_leads = []
                for res in results['results']:
                    lead = save_lead(session, res['Email'], url=res['URL'])
                    if lead:
                        new_leads.append((lead.id, lead.email))
                if new_leads:
                    template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                    if template:
                        from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                        reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                        logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': email} for _, email in new_leads])
                        automation_logs.extend(logs)
                        total_emails_sent += sent_count
                leads_container.text_area("New Leads Found", "\n".join([email for _, email in new_leads]), height=200)
                progress_bar.progress((idx + 1) / len(optimized_terms))
            st.success(f"Automation cycle completed. Total search terms: {total_search_terms}, Total emails sent: {total_emails_sent}")
            time.sleep(3600)
        except Exception as e:
            log_container.error(f"Critical error in automation cycle: {str(e)}")
            time.sleep(300)
    log_container.info("Automation stopped")
    st.session_state.automation_logs = automation_logs
    st.session_state.total_leads_found = total_search_terms
    st.session_state.total_emails_sent = total_emails_sent

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    try:
        with db_session() as session:
            general_settings = session.query(Settings).filter_by(setting_type='general').first()
            if not general_settings or 'openai_api_key' not in general_settings.value:
                st.error("OpenAI API key not set. Please configure it in the settings.")
                return None

            # Initialize OpenAI client with Hugging Face endpoint
            client = OpenAI(
                base_url=general_settings.value.get('openai_api_base', 'https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1/'),
                api_key=general_settings.value.get('openai_api_key', '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')
            )

            try:
                # Make request to Hugging Face endpoint
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-72B-Instruct",  # Model name is required but may be ignored
                    messages=messages,
                    temperature=temperature,
                    max_tokens=500
                )
                
                result = response.choices[0].message.content
                log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id, "Qwen/Qwen2.5-72B-Instruct")
                
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return result
            except Exception as e:
                st.error(f"Error in API call: {str(e)}")
                log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id, "Qwen/Qwen2.5-72B-Instruct")
                return None
    except Exception as e:
        st.error(f"Error in database operation: {str(e)}")
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
        # Check if lead already exists
        existing_lead = session.query(Lead).filter_by(email=email).first()
        if existing_lead:
            # Update existing lead with new information if provided
            for attr in ['first_name', 'last_name', 'company', 'job_title', 'phone', 'created_at']:
                if locals()[attr]:
                    setattr(existing_lead, attr, locals()[attr])
            lead = existing_lead
        else:
            # Create new lead
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
        
        session.flush()  # Flush changes to get lead ID
        
        # Save lead source if URL is provided
        if url and search_term_id:
            lead_source = LeadSource(
                lead_id=lead.id,
                url=url,
                search_term_id=search_term_id
            )
            session.add(lead_source)
        
        # Create campaign lead association
        campaign_lead = CampaignLead(
            campaign_id=get_active_campaign_id(),
            lead_id=lead.id,
            status="Not Contacted",
            created_at=datetime.utcnow()
        )
        session.add(campaign_lead)
        
        # Commit all changes
        session.commit()
        logging.info(f"Successfully saved/updated lead: {email}")
        return lead
        
    except Exception as e:
        logging.error(f"Error saving lead {email}: {str(e)}")
        session.rollback()
        return None

def save_lead_source(session, lead_id, search_term_id, url, http_status, scrape_duration, page_title=None, meta_description=None, content=None, tags=None, phone_numbers=None):
    session.add(LeadSource(lead_id=lead_id, search_term_id=search_term_id, url=url, http_status=http_status, scrape_duration=scrape_duration, page_title=page_title or get_page_title(url), meta_description=meta_description or get_page_description(url), content=content or extract_visible_text(BeautifulSoup(requests.get(url).text, 'html.parser')), tags=tags, phone_numbers=phone_numbers))
    session.commit()

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
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
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
def update_display(container, items, title, item_key):
    container.markdown(
        f"""
        <style>
            .container {{
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.25rem;
                padding: 1rem;
                background-color: rgba(49, 51, 63, 0.1);
            }}
            .entry {{
                margin-bottom: 0.5rem;
                padding: 0.5rem;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 0.25rem;
            }}
        </style>
        <div class="container">
            <h4>{title} ({len(items)})</h4>
            {"".join(f'<div class="entry">{item[item_key]}</div>' for item in items[-20:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def get_domain_from_url(url):
    return urlparse(url).netloc

# Initialize session state for search terms if not exists
if 'search_terms' not in st.session_state:
    st.session_state.search_terms = []

def manual_search_page():
    st.title("Manual Search")
    
    with db_session() as session:
        # Show active processes first
        active_processes = session.query(SearchProcess).filter(
            SearchProcess.status == 'running'
        ).order_by(SearchProcess.created_at.desc()).all()
        
        if active_processes:
            st.subheader("Active Search Processes")
            for process in active_processes:
                with st.expander(f"Process {process.id} - Started at {process.created_at.strftime('%Y-%m-%d %H:%M:%S')}", expanded=True):
                    display_process_logs(process.id)
        
        # Main search interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get recent search terms for suggestions
            recent_searches = session.query(SearchTerm).order_by(SearchTerm.created_at.desc()).limit(5).all()
            recent_terms = [term.term for term in recent_searches]
            
            # Search terms input with suggestions
            search_terms = st_tags(
                label='Enter Search Terms',
                text='Press enter after each term',
                value=st.session_state.search_terms,  # Use session state value
                suggestions=recent_terms,  # Show recent terms as suggestions
                key='search_terms_input'
            )
            
            # Update session state when terms change
            if search_terms != st.session_state.search_terms:
                st.session_state.search_terms = search_terms
            
            # Settings
            settings = {
                'num_results': st.number_input('Results per term', min_value=1, max_value=100, value=10),
                'ignore_previously_fetched': st.checkbox('Ignore previously fetched domains', value=True),
                'optimize_english': st.checkbox('Optimize for English'),
                'optimize_spanish': st.checkbox('Optimize for Spanish'),
                'shuffle_keywords_option': st.checkbox('Shuffle keywords'),
                'language': st.selectbox('Language', ['ES', 'EN'], index=0),
                'enable_email_sending': st.checkbox('Enable automatic email sending')
            }
            
            # Email settings if enabled
            if settings['enable_email_sending']:
                email_settings = fetch_email_settings(session)
                if email_settings:
                    email_setting = st.selectbox(
                        "From Email",
                        options=email_settings,
                        format_func=lambda x: f"{x['name']} ({x['email']})"
                    )
                    settings['from_email'] = email_setting['email']
                    settings['reply_to'] = st.text_input("Reply To", value=email_setting['email'])
                    
                    templates = fetch_email_templates(session)
                    if templates:
                        settings['email_template'] = st.selectbox("Email Template", options=templates)
                    else:
                        st.warning("No email templates found. Please create one first.")
                        settings['enable_email_sending'] = False
                else:
                    st.warning("No email settings found. Please configure email settings first.")
                    settings['enable_email_sending'] = False
        
        with col2:
            st.subheader("AI Assistant")
            
            # AI Propose button
            if st.button("PROPOSE NEW SEARCH TERMS", use_container_width=True):
                with st.spinner("Generating new search term proposals..."):
                    kb_info = get_knowledge_base_info(session, get_active_project_id())
                    if kb_info:
                        prompt = f"""Based on the following business context, propose 5 new search terms that would be effective for lead generation:
                        Business: {kb_info.get('company_description', '')}
                        Target Market: {kb_info.get('company_target_market', '')}
                        Product: {kb_info.get('product_description', '')}
                        Current Terms: {', '.join(search_terms) if search_terms else 'None'}
                        """
                        response = openai_chat_completion(
                            messages=[
                                {"role": "system", "content": "You are a lead generation expert. Propose specific, targeted search terms."},
                                {"role": "user", "content": prompt}
                            ],
                            function_name="propose_search_terms"
                        )
                        if isinstance(response, str):
                            new_terms = [term.strip() for term in response.split('\n') if term.strip()]
                            # Update session state without rerunning
                            st.session_state.search_terms = list(set(search_terms + new_terms))
                            # Show the new terms
                            st.success("New terms proposed!")
                            st.write("New terms added:")
                            for term in new_terms:
                                st.write(f"• {term}")
                    else:
                        st.warning("Please set up your Knowledge Base first to get better search term proposals.")
            
            st.markdown("---")
            
            # AI Optimize button
            if st.button("OPTIMIZE SEARCH TERMS", use_container_width=True):
                with st.spinner("Optimizing search terms..."):
                    if search_terms:
                        prompt = f"""Optimize these search terms for better lead generation results:
                        Terms: {', '.join(search_terms)}
                        
                        For each term:
                        1. Add relevant industry-specific keywords
                        2. Include intent-based modifiers
                        3. Consider geographic or demographic targeting if applicable
                        4. Format for better search engine compatibility
                        """
                        response = openai_chat_completion(
                            messages=[
                                {"role": "system", "content": "You are a search optimization expert. Improve search terms for maximum effectiveness."},
                                {"role": "user", "content": prompt}
                            ],
                            function_name="optimize_search_terms"
                        )
                        if isinstance(response, str):
                            optimized_terms = [term.strip() for term in response.split('\n') if term.strip()]
                            # Update session state without rerunning
                            st.session_state.search_terms = optimized_terms
                            # Show the optimized terms
                            st.success("Terms optimized!")
                            st.write("Optimized terms:")
                            for term in optimized_terms:
                                st.write(f"• {term}")
                    else:
                        st.warning("Please enter some search terms to optimize.")
            
            st.markdown("---")
            
            # Start Search button
            if st.button("Start Search", type="primary", use_container_width=True):
                if not search_terms:
                    st.warning("Please enter at least one search term")
                    return
                
                new_process = SearchProcess(
                    search_terms=search_terms,
                    settings=settings,
                    status='running',
                    results={},
                    logs=[],
                    total_leads_found=0,
                    campaign_id=get_active_campaign_id()
                )
                session.add(new_process)
                session.commit()
                
                # Start background thread
                import threading
                thread = threading.Thread(
                    target=background_manual_search,
                    args=(new_process.id, search_terms, settings)
                )
                thread.daemon = True
                thread.start()
                
                st.success("Search process started in background!")
                st.info(f"Process ID: {new_process.id}")
                
                # Show logs immediately
                display_process_logs(new_process.id)

def get_page_description(html_content):
    """Extract page description from HTML meta tags."""
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    return meta_desc['content'] if meta_desc else "No description found"

def fetch_search_terms_with_lead_count(session):
    """Fetch search terms with their associated lead and email counts."""
    query = (session.query(SearchTerm.term, 
                          func.count(distinct(Lead.id)).label('lead_count'),
                          func.count(distinct(EmailCampaign.id)).label('email_count'))
             .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
             .join(Lead, LeadSource.lead_id == Lead.id)
             .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
             .group_by(SearchTerm.term))
    df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    return df

def view_campaign_logs():
    st.title("Email Campaign Logs")
    
    with db_session() as session:
        logs = fetch_all_email_logs(session)
        if logs.empty:
            st.info("No email logs found.")
            return
        
        # Add CSS for styling
        st.markdown("""
            <style>
                .email-logs-container {
                    max-height: 600px;
                    overflow-y: auto;
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: 1rem;
                    background-color: rgba(49, 51, 63, 0.1);
                    margin-bottom: 1rem;
                }
                .email-log-entry {
                    padding: 0.75rem;
                    margin-bottom: 0.5rem;
                    border-radius: 0.25rem;
                    background-color: rgba(255, 255, 255, 0.05);
                    border-left: 4px solid;
                    animation: fadeIn 0.5s ease-in;
                }
                .email-log-entry.success {
                    border-left-color: #28a745;
                }
                .email-log-entry.error {
                    border-left-color: #dc3545;
                }
                .email-log-entry.pending {
                    border-left-color: #ffc107;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(-10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Emails", len(logs))
        with col2:
            success_rate = (logs['Status'] == 'sent').mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            st.metric("Unique Recipients", logs['Email'].nunique())
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=logs['Sent At'].min().date())
        with col2:
            end_date = st.date_input("End Date", value=logs['Sent At'].max().date())
        
        search_term = st.text_input("🔍 Search by email or subject")
        
        # Filter logs
        filtered_logs = logs[
            (logs['Sent At'].dt.date >= start_date) & 
            (logs['Sent At'].dt.date <= end_date)
        ]
        
        if search_term:
            filtered_logs = filtered_logs[
                filtered_logs['Email'].str.contains(search_term, case=False) | 
                filtered_logs['Subject'].str.contains(search_term, case=False)
            ]
        
        # Display logs
        st.markdown("<div class='email-logs-container'>", unsafe_allow_html=True)
        for _, log in filtered_logs.iterrows():
            status_class = {
                'sent': 'success',
                'failed': 'error'
            }.get(log['Status'], 'pending')
            
            st.markdown(f"""
                <div class='email-log-entry {status_class}'>
                    <div style='display: flex; justify-content: space-between;'>
                        <strong>{log['Email']}</strong>
                        <span>{log['Sent At'].strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    <div style='margin-top: 0.5rem;'>
                        <strong>Subject:</strong> {log['Subject']}<br>
                        <strong>Status:</strong> {log['Status']}<br>
                        <strong>Template:</strong> {log['Template']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add auto-scroll JavaScript
        st.markdown("""
            <script>
                function scrollToBottom() {
                    const container = document.querySelector('.email-logs-container');
                    if (container) {
                        container.scrollTop = container.scrollHeight;
                    }
                }
                
                // Initial scroll
                scrollToBottom();
                
                // Set up a mutation observer to watch for changes
                const observer = new MutationObserver(scrollToBottom);
                const container = document.querySelector('.email-logs-container');
                if (container) {
                    observer.observe(container, { childList: true, subtree: true });
                }
            </script>
        """, unsafe_allow_html=True)
        
        # Export option
        if st.button("Export Logs to CSV"):
            csv = filtered_logs.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="email_logs.csv",
                mime="text/csv"
            )

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

def update_lead(session, lead_id, updated_data):
    try:
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            for key, value in updated_data.items():
                setattr(lead, key, value)
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
            return True
    except SQLAlchemyError as e:
        logging.error(f"Error deleting lead {lead_id}: {str(e)}")
        session.rollback()
    return False

def is_valid_email(email):
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False






def view_leads_page():
    st.title("Lead Management Dashboard")
    with db_session() as session:
        if 'leads' not in st.session_state or st.button("Refresh Leads"):
            st.session_state.leads = fetch_leads_with_sources(session)
        if not st.session_state.leads.empty:
            total_leads = len(st.session_state.leads)
            contacted_leads = len(st.session_state.leads[st.session_state.leads['Last Contact'].notna()])
            conversion_rate = (st.session_state.leads['Last Email Status'] == 'sent').mean()

            st.columns(3)[0].metric("Total Leads", f"{total_leads:,}")
            st.columns(3)[1].metric("Contacted Leads", f"{contacted_leads:,}")
            st.columns(3)[2].metric("Conversion Rate", f"{conversion_rate:.2%}")

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
    df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    return df

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
    """Get the currently active campaign ID from session state"""
    return st.session_state.get('active_campaign_id', 1)  # Default to 1 if not set

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
        current_term_ids = set(int(term.split(":")[0]) for term in updated_terms)
        existing_terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all()
        
        for term in existing_terms:
            if term.id not in current_term_ids:
                term.group_id = None
        
        for term_str in updated_terms:
            term_id = int(term_str.split(":")[0])
            term = session.query(SearchTerm).get(term_id)
            if term:
                term.group_id = group_id
        
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_search_term_group: {str(e)}")

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
    existing_groups = session.query(SearchTermGroup).all()
    
    prompt = f"""
    Categorize these search terms into existing groups or suggest new ones:
    {', '.join([term.term for term in ungrouped_terms])}

    Existing groups: {', '.join([group.name for group in existing_groups])}

    Respond with a JSON object: {{group_name: [term1, term2, ...]}}
    """

    messages = [
        {"role": "system", "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."},
        {"role": "user", "content": prompt}
    ]

    response = openai_chat_completion(messages, function_name="ai_group_search_terms")

    return response if isinstance(response, dict) else {}

def update_search_term_groups(session, grouped_terms):
    for group_name, terms in grouped_terms.items():
        group = session.query(SearchTermGroup).filter_by(name=group_name).first()
        if not group:
            group = SearchTermGroup(name=group_name)
            session.add(group)
            session.flush()
        
        for term in terms:
            search_term = session.query(SearchTerm).filter_by(term=term).first()
            if search_term:
                search_term.group_id = group.id

    session.commit()

def create_search_term_group(session, group_name):
    try:
        new_group = SearchTermGroup(name=group_name)
        session.add(new_group)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating search term group: {str(e)}")

def delete_search_term_group(session, group_id):
    try:
        group = session.query(SearchTermGroup).get(group_id)
        if group:
            # Set group_id to None for all search terms in this group
            session.query(SearchTerm).filter(SearchTerm.group_id == group_id).update({SearchTerm.group_id: None})
            session.delete(group)
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting search term group: {str(e)}")


def email_templates_page():
    st.header("Email Templates")
    with db_session() as session:
        templates = session.query(EmailTemplate).all()
        with st.expander("Create New Template", expanded=False):
            new_template_name = st.text_input("Template Name", key="new_template_name")
            use_ai = st.checkbox("Use AI to generate template", key="use_ai")
            if use_ai:
                ai_prompt = st.text_area("Enter prompt for AI template generation", key="ai_prompt")
                use_kb = st.checkbox("Use Knowledge Base information", key="use_kb")
                if st.button("Generate Template", key="generate_ai_template"):
                    with st.spinner("Generating template..."):
                        kb_info = get_knowledge_base_info(session, get_active_project_id()) if use_kb else None
                        generated_template = generate_or_adjust_email_template(ai_prompt, kb_info)
                        new_template_subject = generated_template.get("subject", "AI Generated Subject")
                        new_template_body = generated_template.get("body", "")
                        
                        if new_template_name:
                            new_template = EmailTemplate(
                                template_name=new_template_name,
                                subject=new_template_subject,
                                body_content=new_template_body,
                                campaign_id=get_active_campaign_id()
                            )
                            session.add(new_template)
                            session.commit()
                            st.success("AI-generated template created and saved!")
                            templates = session.query(EmailTemplate).all()
                        else:
                            st.warning("Please provide a name for the template before generating.")
                        
                        st.subheader("Generated Template")
                        st.text(f"Subject: {new_template_subject}")
                        st.components.v1.html(wrap_email_body(new_template_body), height=400, scrolling=True)
            else:
                new_template_subject = st.text_input("Subject", key="new_template_subject")
                new_template_body = st.text_area("Body Content", height=200, key="new_template_body")

            if st.button("Create Template", key="create_template_button"):
                if all([new_template_name, new_template_subject, new_template_body]):
                    new_template = EmailTemplate(
                        template_name=new_template_name,
                        subject=new_template_subject,
                        body_content=new_template_body,
                        campaign_id=get_active_campaign_id()
                    )
                    session.add(new_template)
                    session.commit()
                    st.success("New template created successfully!")
                    templates = session.query(EmailTemplate).all()
                else:
                    st.warning("Please fill in all fields to create a new template.")

        if templates:
            st.subheader("Existing Templates")
            for template in templates:
                with st.expander(f"Template: {template.template_name}", expanded=False):
                    col1, col2 = st.columns(2)
                    edited_subject = col1.text_input("Subject", value=template.subject, key=f"subject_{template.id}")
                    is_ai_customizable = col2.checkbox("AI Customizable", value=template.is_ai_customizable, key=f"ai_{template.id}")
                    edited_body = st.text_area("Body Content", value=template.body_content, height=200, key=f"body_{template.id}")
                    
                    ai_adjustment_prompt = st.text_area("AI Adjustment Prompt", key=f"ai_prompt_{template.id}", placeholder="E.g., Make it less marketing-like and mention our main features")
                    
                    col3, col4 = st.columns(2)
                    if col3.button("Apply AI Adjustment", key=f"apply_ai_{template.id}"):
                        with st.spinner("Applying AI adjustment..."):
                            kb_info = get_knowledge_base_info(session, get_active_project_id())
                            adjusted_template = generate_or_adjust_email_template(ai_adjustment_prompt, kb_info, current_template=edited_body)
                            edited_subject = adjusted_template.get("subject", edited_subject)
                            edited_body = adjusted_template.get("body", edited_body)
                            st.success("AI adjustment applied. Please review and save changes.")
                    
                    if col4.button("Save Changes", key=f"save_{template.id}"):
                        template.subject = edited_subject
                        template.body_content = edited_body
                        template.is_ai_customizable = is_ai_customizable
                        session.commit()
                        st.success("Template updated successfully!")
                    
                    st.markdown("### Preview")
                    st.text(f"Subject: {edited_subject}")
                    st.components.v1.html(wrap_email_body(edited_body), height=400, scrolling=True)
                    
                    if st.button("Delete Template", key=f"delete_{template.id}"):
                        session.delete(template)
                        session.commit()
                        st.success("Template deleted successfully!")
                        st.rerun()
        else:
            st.info("No email templates found. Create a new template to get started.")

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
            email_setting_option = st.selectbox("From Email", options=email_settings, format_func=lambda x: f"{x['name']} ({x['email']}")
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
    df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    return df

def fetch_search_term_groups(session):
    return [f"{group.id}: {group.name}" for group in session.query(SearchTermGroup).all()]

def fetch_search_terms_for_groups(session, group_ids):
    terms = session.query(SearchTerm).filter(SearchTerm.group_id.in_(group_ids)).all()
    return [term.term for term in terms]

def ai_automation_loop(session, log_container, leads_container):
    automation_logs, total_search_terms, total_emails_sent = [], 0, 0
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
            total_search_terms = len(optimized_terms)
            progress_bar = st.progress(0)
            for idx, term in enumerate(optimized_terms):
                results = manual_search(session, [term], 10, ignore_previously_fetched=True)
                new_leads = []
                for res in results['results']:
                    lead = save_lead(session, res['Email'], url=res['URL'])
                    if lead:
                        new_leads.append((lead.id, lead.email))
                if new_leads:
                    template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                    if template:
                        from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                        reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                        logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': email} for _, email in new_leads])
                        automation_logs.extend(logs)
                        total_emails_sent += sent_count
                leads_container.text_area("New Leads Found", "\n".join([email for _, email in new_leads]), height=200)
                progress_bar.progress((idx + 1) / len(optimized_terms))
            st.success(f"Automation cycle completed. Total search terms: {total_search_terms}, Total emails sent: {total_emails_sent}")
            time.sleep(3600)
        except Exception as e:
            log_container.error(f"Critical error in automation cycle: {str(e)}")
            time.sleep(300)
    log_container.info("Automation stopped")
    st.session_state.update({"automation_logs": automation_logs, "total_leads_found": total_search_terms, "total_emails_sent": total_emails_sent})

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
    df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])
    return df

def fetch_leads_for_search_terms(session, search_term_ids) -> List[Lead]:
    return session.query(Lead).distinct().join(LeadSource).filter(LeadSource.search_term_id.in_(search_term_ids)).all()

def projects_campaigns_page():
    with db_session() as session:
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
                    form_data.update({'project_id': project_id, 'created_at': datetime.utcnow()})
                    if kb_entry:
                        for k, v in form_data.items(): setattr(kb_entry, k, v)
                    else: session.add(KnowledgeBase(**form_data))
                    session.commit()
                    st.success("Knowledge Base saved successfully!", icon="✅")
                except Exception as e: st.error(f"An error occurred while saving the Knowledge Base: {str(e)}")

def autoclient_ai_page():
    st.header("AutoclientAI - Automated Lead Generation")
    with st.expander("Knowledge Base Information", expanded=False):
        with db_session() as session:
            kb_info = get_knowledge_base_info(session, get_active_project_id())
        if not kb_info:
            return st.error("Knowledge Base not found for the active project. Please set it up first.")
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
            else:
                st.error("Failed to generate optimized search terms. Please try again.")
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

def update_search_terms(session, classified_terms):
    for group, terms in classified_terms.items():
        for term in terms:
            existing_term = session.query(SearchTerm).filter_by(term=term, project_id=get_active_project_id()).first()
            if existing_term:
                existing_term.group = group
            else:
                session.add(SearchTerm(term=term, group=group, project_id=get_active_project_id()))
    session.commit()

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
            # Remove st.rerun() - Streamlit will handle the update automatically

    if st.button("Perform Quick Scan", use_container_width=True):
        with st.spinner("Performing quick scan..."):
            try:
                with db_session() as session:
                    new_leads = session.query(Lead).filter(Lead.is_processed == False).count()
                    session.query(Lead).filter(Lead.is_processed == False).update(
                        {Lead.is_processed: True}  # Remove extra )
                    )
                    session.commit()
                    st.success(f"Quick scan completed! Found {new_leads} new leads.")
            except Exception as e:
                st.error(f"An error occurred during quick scan: {str(e)}")

    st.subheader("Real-Time Analytics")
    try:
        with db_session() as session:
            total_leads = session.query(Lead).count()
            emails_sent = session.query(EmailCampaign).count()
            col1, col2 = st.columns(2)
            col1.metric("Total Leads", total_leads)
            col2.metric("Emails Sent", emails_sent)
    except Exception as e:
        st.error(f"An error occurred while displaying analytics: {str(e)}")

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
                    else:
                        leads_container.info("No new leads found in this cycle.")

                    update_display(log_container, st.session_state.get('automation_logs', []), "Latest Logs", "log")
                    time.sleep(3600)
        except Exception as e:
            st.error(f"An error occurred in the automation process: {str(e)}")

def get_knowledge_base_info(session, project_id):
    kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb.to_dict() if kb else None

def generate_optimized_search_terms(session, base_terms, kb_info):
    ai_prompt = f"Generate 5 optimized search terms based on: {', '.join(base_terms)}. Context: {kb_info}"
    return get_ai_response(ai_prompt).split('\n')

def update_display(container, items, title, item_type):
    container.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True)
    for item in items[-10:]:
        container.text(item)

def get_search_terms(session):
    return [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]

def get_ai_response(prompt):
    return openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100).choices[0].text.strip()

def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        return []

def bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_bar=None, status_text=None, results=None, log_container=None):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        logging.error(f"Email template with ID {template_id} not found.")
        return [], 0

    email_subject = template.subject
    email_content = template.body_content

    logs, sent_count = [], 0
    total_leads = len(leads)

    for index, lead in enumerate(leads):
        try:
            validate_email(lead['Email'])
            response, tracking_id = send_email_ses(session, from_email, lead['Email'], email_subject, email_content, reply_to=reply_to)
            if response:
                status = 'sent'
                message_id = response.get('MessageId', f"sent-{uuid.uuid4()}")
                sent_count += 1
                log_message = f"✅ Email sent to: {lead['Email']}"
            else:
                status = 'failed'
                message_id = f"failed-{uuid.uuid4()}"
                log_message = f"❌ Failed to send email to: {lead['Email']}"
            
            save_email_campaign(session, lead['Email'], template_id, status, datetime.utcnow(), email_subject, message_id, email_content)
            logs.append(log_message)

            if progress_bar:
                progress_bar.progress((index + 1) / total_leads)
            if status_text:
                status_text.text(f"Processed {index + 1}/{total_leads} leads")
            if results is not None:
                results.append({"Email": lead['Email'], "Status": status})

            if log_container:
                log_container.text(log_message)

        except EmailNotValidError:
            log_message = f"❌ Invalid email address: {lead['Email']}"
            logs.append(log_message)
        except Exception as e:
            error_message = f"Error sending email to {lead['Email']}: {str(e)}"
            logging.error(error_message)
            save_email_campaign(session, lead['Email'], template_id, 'failed', datetime.utcnow(), email_subject, f"error-{uuid.uuid4()}", email_content)
            logs.append(f"❌ Error sending email to: {lead['Email']} (Error: {str(e)})")

    return logs, sent_count

def wrap_email_body(body_content):
    """Wrap email body content with HTML template"""
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
            }}
            .signature {{
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        {body_content}
    </body>
    </html>
    """

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

def display_logs(log_container, logs):
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

def view_sent_email_campaigns():
    st.header("Sent Email Campaigns")
    try:
        with db_session() as session:
            email_campaigns = fetch_sent_email_campaigns(session)
        if not email_campaigns.empty:
            st.dataframe(email_campaigns)
            st.subheader("Detailed Content")
            selected_campaign = st.selectbox("Select a campaign to view details", email_campaigns['ID'].tolist())
            if selected_campaign:
                campaign_content = email_campaigns[email_campaigns['ID'] == selected_campaign]['Content'].iloc[0]
                st.text_area("Content", campaign_content if campaign_content else "No content available", height=300)
        else:
            st.info("No sent email campaigns found.")
    except Exception as e:
        st.error(f"An error occurred while fetching sent email campaigns: {str(e)}")
        logging.error(f"Error in view_sent_email_campaigns: {str(e)}")

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
        pages[selected]()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("An error occurred in the main function")
        st.write("Please try refreshing the page or contact support if the issue persists.")

    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")


def update_process_log(session, process_id, message, level='info'):
    """Update the logs for a search process"""
    try:
        process = session.query(SearchProcess).get(process_id)
        if not process:
            logging.error(f"Process {process_id} not found")
            return False
            
        # Initialize logs array if None
        if process.logs is None:
            process.logs = []
            
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message
        }
        
        # Append new log entry
        process.logs.append(log_entry)
        
        # Update the process
        session.add(process)
        session.commit()
        
        return True
    except Exception as e:
        logging.error(f"Error updating process log: {str(e)}")
        session.rollback()
        return False

def display_process_logs(process_id):
    """Display logs for a search process"""
    with db_session() as session:
        process = session.query(SearchProcess).get(process_id)
        if not process:
            st.info("Process not found")
            return
            
        if not process.logs:
            st.info("No logs available yet. Logs will appear here as the process runs.")
            return
        
        # Use container for better stability
        log_container = st.container()
        
        with log_container:
            # Add CSS for scrollable container with unique class
            st.markdown("""
                <style>
                    .process-logs-container {
                        max-height: 400px;
                        overflow-y: auto;
                        border: 1px solid rgba(49, 51, 63, 0.2);
                        border-radius: 0.25rem;
                        padding: 1rem;
                        background-color: rgba(49, 51, 63, 0.1);
                        margin-bottom: 1rem;
                        font-family: monospace;
                    }
                    .process-log-entry {
                        padding: 0.25rem 0;
                        border-bottom: 1px solid rgba(49, 51, 63, 0.1);
                        animation: fadeIn 0.5s ease-in;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(-10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Format logs with icons
            log_entries = []
            for log in process.logs:
                timestamp = datetime.fromisoformat(log['timestamp']).strftime('%H:%M:%S')
                level = log['level']
                message = log['message']
                icon = {
                    'info': '🔵',
                    'success': '🟢',
                    'warning': '🟠',
                    'error': '🔴',
                    'email_sent': '🟣'
                }.get(level, '⚪')
                log_entries.append(f'<div class="process-log-entry">{icon} [{timestamp}] {message}</div>')
            
            # Display all logs at once in the container
            st.markdown(f'<div class="process-logs-container">{"".join(log_entries)}</div>', unsafe_allow_html=True)
            
            # Add auto-scroll JavaScript
            st.markdown("""
                <script>
                    function scrollToBottom() {
                        const containers = document.getElementsByClassName('process-logs-container');
                        for (let container of containers) {
                            container.scrollTop = container.scrollHeight;
                        }
                    }
                    
                    // Initial scroll
                    scrollToBottom();
                    
                    // Set up a mutation observer to watch for changes
                    const observer = new MutationObserver(scrollToBottom);
                    const containers = document.getElementsByClassName('process-logs-container');
                    for (let container of containers) {
                        observer.observe(container, { childList: true, subtree: true });
                    }
                </script>
            """, unsafe_allow_html=True)

def background_manual_search(process_id, search_terms, settings):
    """Execute manual search in background"""
    with db_session() as session:
        try:
            process = session.query(SearchProcess).get(process_id)
            if not process:
                logging.error(f"Process {process_id} not found")
                return
                
            process.status = 'running'
            session.commit()
            
            # Log start
            update_process_log(session, process_id, f"Starting search process with {len(search_terms)} terms", 'info')
            
            results = manual_search(
                session=session,
                terms=search_terms,
                num_results=settings.get('num_results', 10),
                ignore_previously_fetched=settings.get('ignore_previously_fetched', True),
                optimize_english=settings.get('optimize_english', False),
                optimize_spanish=settings.get('optimize_spanish', False),
                shuffle_keywords_option=settings.get('shuffle_keywords_option', False),
                language=settings.get('language', 'ES'),
                enable_email_sending=settings.get('enable_email_sending', False),
                from_email=settings.get('from_email'),
                reply_to=settings.get('reply_to'),
                email_template=settings.get('email_template'),
                process_id=process_id
            )
            
            # Log completion
            update_process_log(session, process_id, f"Search completed. Found {results.get('total_leads', 0)} leads", 'success')
            
            process.status = 'completed'
            process.results = results
            session.commit()
            
        except Exception as e:
            error_msg = f"Error in background search: {str(e)}"
            logging.error(error_msg)
            update_process_log(session, process_id, error_msg, 'error')
            process.status = 'failed'
            session.commit()

def view_campaign_logs():
    st.title("Email Campaign Logs")
    
    with db_session() as session:
        logs = fetch_all_email_logs(session)
        if logs.empty:
            st.info("No email logs found.")
            return
        
        # Add CSS for styling
        st.markdown("""
            <style>
                .email-logs-container {
                    max-height: 600px;
                    overflow-y: auto;
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem;
                    padding: 1rem;
                    background-color: rgba(49, 51, 63, 0.1);
                    margin-bottom: 1rem;
                }
                .email-log-entry {
                    padding: 0.75rem;
                    margin-bottom: 0.5rem;
                    border-radius: 0.25rem;
                    background-color: rgba(255, 255, 255, 0.05);
                    border-left: 4px solid;
                    animation: fadeIn 0.5s ease-in;
                }
                .email-log-entry.success {
                    border-left-color: #28a745;
                }
                .email-log-entry.error {
                    border-left-color: #dc3545;
                }
                .email-log-entry.pending {
                    border-left-color: #ffc107;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(-10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Emails", len(logs))
        with col2:
            success_rate = (logs['Status'] == 'sent').mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            st.metric("Unique Recipients", logs['Email'].nunique())
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=logs['Sent At'].min().date())
        with col2:
            end_date = st.date_input("End Date", value=logs['Sent At'].max().date())
        
        search_term = st.text_input("🔍 Search by email or subject")
        
        # Filter logs
        filtered_logs = logs[
            (logs['Sent At'].dt.date >= start_date) & 
            (logs['Sent At'].dt.date <= end_date)
        ]
        
        if search_term:
            filtered_logs = filtered_logs[
                filtered_logs['Email'].str.contains(search_term, case=False) | 
                filtered_logs['Subject'].str.contains(search_term, case=False)
            ]
        
        # Display logs
        st.markdown("<div class='email-logs-container'>", unsafe_allow_html=True)
        for _, log in filtered_logs.iterrows():
            status_class = {
                'sent': 'success',
                'failed': 'error'
            }.get(log['Status'], 'pending')
            
            st.markdown(f"""
                <div class='email-log-entry {status_class}'>
                    <div style='display: flex; justify-content: space-between;'>
                        <strong>{log['Email']}</strong>
                        <span>{log['Sent At'].strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    <div style='margin-top: 0.5rem;'>
                        <strong>Subject:</strong> {log['Subject']}<br>
                        <strong>Status:</strong> {log['Status']}<br>
                        <strong>Template:</strong> {log['Template']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add auto-scroll JavaScript
        st.markdown("""
            <script>
                function scrollToBottom() {
                    const container = document.querySelector('.email-logs-container');
                    if (container) {
                        container.scrollTop = container.scrollHeight;
                    }
                }
                
                // Initial scroll
                scrollToBottom();
                
                // Set up a mutation observer to watch for changes
                const observer = new MutationObserver(scrollToBottom);
                const container = document.querySelector('.email-logs-container');
                if (container) {
                    observer.observe(container, { childList: true, subtree: true });
                }
            </script>
        """, unsafe_allow_html=True)
        
        # Export option
        if st.button("Export Logs to CSV"):
            csv = filtered_logs.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="email_logs.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
