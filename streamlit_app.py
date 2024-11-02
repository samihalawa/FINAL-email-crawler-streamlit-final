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

# Initialize database connection once
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Remove duplicate initialization
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("Database configuration incomplete")

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
def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, campaign=None):
    """Send email with proper error handling and tracking"""
    if not all([from_email, to_email, subject, body]):
        logging.error("Missing required email parameters")
        return None, None
        
    try:
        # Get email settings
        email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
        if not email_settings:
            logging.error(f"No email settings found for {from_email}")
            return None, None

        # Generate tracking ID
        tracking_id = str(uuid.uuid4())
        
        # Add tracking pixel and wrap links
        tracked_body = wrap_email_body(body)
        tracked_body = tracked_body.replace('</body>',
            f'<img src="https://autoclient-email-analytics.trigox.workers.dev/track?id={tracking_id}&type=open" style="display:none;"/></body>')
        
        soup = BeautifulSoup(tracked_body, 'html.parser')
        for a in soup('a', href=True):
            a['href'] = f"https://autoclient-email-analytics.trigox.workers.dev/track?id={tracking_id}&type=click&url={a['href']}"

        if email_settings.provider == 'ses':
            # Configure AWS SES
            ses_client = boto3.Session(
                aws_access_key_id=email_settings.aws_access_key_id,
                aws_secret_access_key=email_settings.aws_secret_access_key,
                region_name=email_settings.aws_region
            ).client('ses')

            # Send email
            response = ses_client.send_email(
                Source=from_email,
                Destination={'ToAddresses': [to_email]},
                Message={
                    'Subject': {'Data': subject, 'Charset': charset},
                    'Body': {'Html': {'Data': str(soup), 'Charset': charset}}
                },
                ReplyToAddresses=[reply_to] if reply_to else []
            )
            return response, tracking_id
        else:
            # Handle SMTP or other providers
            return {'MessageId': f'smtp-{uuid.uuid4()}'}, tracking_id

    except Exception as e:
        logging.error(f"Email sending error: {str(e)}")
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
def save_lead(session: Session, email: str, **kwargs) -> Optional[Lead]:
    """Save lead with proper error handling"""
    try:
        with session.begin_nested():  # Use savepoint
            lead = session.query(Lead).filter_by(email=email).first() or Lead(
                email=email,
                **{x:kwargs.get(x) for x in ['first_name','last_name','company','job_title','phone']},
                created_at=kwargs.get('created_at',datetime.utcnow())
            )
            if not lead.id:
                session.add(lead)
                session.flush()
            
            session.add(LeadSource(
                lead_id=lead.id,
                url=kwargs.get('url'),
                search_term_id=kwargs.get('search_term_id')
            ))
            session.add(CampaignLead(
                campaign_id=get_active_campaign_id(),
                lead_id=lead.id,
                status="Not Contacted"
            ))
            session.commit()
            return lead
    except SQLAlchemyError as e:  # Specific exception
        logging.error(f"Database error: {str(e)}")
        session.rollback()
        return None
    except Exception as e:  # General exception as fallback
        logging.error(f"Unexpected error: {str(e)}")
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
    <style>
        body {{
            font-family: Arial;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px
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

def update_search_term_group(s,gid,ut):
    try:
        [setattr(t,'group_id',None) for t in s.query(SearchTerm).filter(SearchTerm.group_id==gid).all() if t.id not in {int(x.split(":")[0]) for x in ut if ":" in x}]
        [setattr(t,'group_id',gid) for x in ut if ":" in x and (t:=s.query(SearchTerm).get(int(x.split(":")[0])))]
        return s.commit() or True
    except Exception as e: return s.rollback() or logging.error(f"Error updating group: {str(e)}") or False

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try: session.add(SearchTerm(term=new_term, campaign_id=campaign_id, created_at=datetime.utcnow(), 
                    group_id=int(group_for_new_term.split(":")[0]) if group_for_new_term != "None" else None)) and session.commit()
    except Exception as e: session.rollback(); logging.error(f"Error adding search term: {str(e)}")

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

def ai_automation_loop(session, log_container, leads_container):
    if not all([session, log_container, leads_container]):
        raise ValueError("Missing required automation parameters")
    
    try:
        with session.begin_nested():
            automation_state = get_automation_state()
            if not automation_state['running']:
                automation_state['running'] = True
                automation_state['last_run'] = datetime.now()
            
            # Fix walrus operator by using regular assignment
            automation_status = st.session_state.get('automation_status', False)
            while automation_status:
                automation_state['current_cycle'] += 1
                automation_logs, total_search_terms, total_emails_sent = [], 0, 0
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
                        template = session.query(EmailTemplate).filter_by(
                            project_id=get_active_project_id(),
                            campaign_id=get_active_campaign_id()
                        ).first()
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
        logging.error(f"Automation error: {str(e)}")
        if log_container:
            log_container.error(f"Automation failed: {str(e)}")
        raise

def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    # Add caching for API responses
    @st.cache_data(ttl=3600)
    def get_cached_response(messages_key):
        with safe_db_connection() as session:
            general_settings = session.query(Settings).filter_by(setting_type='general').first()
            if not general_settings or 'openai_api_key' not in general_settings.value:
                raise ValueError("OpenAI API key not configured")
            
            client = OpenAI(api_key=general_settings.value['openai_api_key'])
            response = client.chat.completions.create(
                model=general_settings.value.get('openai_model', "gpt-4o-mini"),
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content

    try:
        result = get_cached_response(str(messages))
        with safe_db_connection() as session:
            log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id)
        return json.loads(result) if isinstance(result, str) else result
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
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

def save_lead(
    session: Session, 
    email: str, 
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    company: Optional[str] = None,
    job_title: Optional[str] = None,
    phone: Optional[str] = None,
    url: Optional[str] = None,
    search_term_id: Optional[int] = None,
    created_at: Optional[datetime] = None
) -> Optional[Lead]:
    """Save lead with proper type hints"""
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

def save_lead_source(s, lead_id, term_id, url, status, duration, **meta):
    s.add(LeadSource(lead_id=lead_id,search_term_id=term_id,url=url,http_status=status,scrape_duration=duration,**{k:meta.get(k,'') for k in ['page_title','meta_description','content','tags','phone_numbers']})) and s.commit()

def get_page_title(html_content):
    try: return BeautifulSoup(html_content, 'html.parser').title.string.strip() if BeautifulSoup(html_content, 'html.parser').title else "No title found"
    except Exception as e: logging.error(f"Error getting page title: {str(e)}"); return "No title found"

def get_page_description(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    return meta_desc['content'] if meta_desc else "No description found"

def extract_visible_text(soup):
    [script.extract() for script in soup(["script", "style"])]
    return ' '.join(phrase.strip() for line in soup.get_text().splitlines() if line.strip() 
                   for phrase in line.split("  ") if phrase.strip())

def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
    session.add(SearchTermEffectiveness(term=term, total_results=total_results, valid_leads=valid_leads, 
                irrelevant_leads=total_results - valid_leads, blogs_found=blogs_found, directories_found=directories_found)) and session.commit()

get_active_project_id = lambda: st.session_state.get('active_project_id', 1)
get_active_campaign_id = lambda: st.session_state.get('active_campaign_id', 1)
set_active_project_id = lambda project_id: st.session_state.__setitem__('active_project_id', project_id)
set_active_campaign_id = lambda campaign_id: st.session_state.__setitem__('active_campaign_id', campaign_id)

def add_or_get_search_term(session, term, campaign_id, created_at=None):
    return (st := session.query(SearchTerm).filter_by(term=term,campaign_id=campaign_id).first()) and st.id or session.add(SearchTerm(term=term,campaign_id=campaign_id,created_at=created_at or datetime.utcnow())) or session.commit() or st.id

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
def get_domain_from_url(url): return urlparse(url).netloc

def manual_search_page():
    st.title("Manual Search")
    with safe_db_connection() as session:
        recent_search_terms = [t.term for t in session.query(SearchTerm).order_by(SearchTerm.created_at.desc()).limit(5)]
        email_templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)

    col1, col2 = st.columns([2,1])
    with col1:
        search_terms = st_tags(label='Enter search terms:', text='Press enter to add more', value=recent_search_terms, suggestions=['software engineer', 'data scientist', 'product manager'], maxtags=10, key='search_terms_input')
        num_results = st.slider("Results per term", 1, 500000, 1000)
    with col2:
        enable_email_sending = st.checkbox("Enable email sending", True)
        ignore_previously_fetched = st.checkbox("Ignore fetched domains", True) 
        shuffle_keywords_option = st.checkbox("Shuffle Keywords", True)
        optimize_english = st.checkbox("Optimize (English)", False)
        optimize_spanish = st.checkbox("Optimize (Spanish)", False)
        language = st.selectbox("Select Language", ["ES","EN"], 0)

    if enable_email_sending:
        if not email_templates or not email_settings: return st.error("Missing templates or settings")
        col3, col4 = st.columns(2)
        with col3: email_template = st.selectbox("Email template", email_templates, format_func=lambda x:x.split(":")[1].strip())
        with col4:
            email_setting = st.selectbox("From Email", email_settings, format_func=lambda x:f"{x['name']} ({x['email']}")
            if not email_setting: return st.error("Select email setting")
            from_email = email_setting['email']
            reply_to = st.text_input("Reply To", from_email)

    if st.button("Search") and search_terms:
        progress_bar, status_text, email_status = st.progress(0), st.empty(), st.empty()
        results, leads_found, emails_sent = [], [], []
        log_container, leads_container = st.empty(), st.empty()

        for i, term in enumerate(search_terms):
            status_text.text(f"Searching: '{term}' ({i+1}/{len(search_terms)})")
            with safe_db_connection() as session:
                term_results = manual_search(session, [term], num_results, ignore_previously_fetched, optimize_english, optimize_spanish, shuffle_keywords_option, language, enable_email_sending, log_container, from_email, reply_to, email_template)
                results.extend(term_results['results'])
                leads_found.extend([f"{r['Email']} - {r['Company']}" for r in term_results['results'] if 'Email' in r and 'Company' in r])
                
                if enable_email_sending:
                    template = session.query(EmailTemplate).filter_by(id=int(email_template.split(":")[0])).first()
                    for result in term_results['results']:
                        if not result or 'Email' not in result or not is_valid_email(result['Email']): continue
                        wrapped_content = wrap_email_body(template.body_content)
                        response, tracking_id = send_email_ses(session, from_email, result['Email'], template.subject, wrapped_content, reply_to=reply_to)
                        if response:
                            update_log(log_container, f"Sent email to: {result['Email']}", 'email_sent')
                            save_email_campaign(session, result['Email'], template.id, 'Sent', datetime.utcnow(), template.subject, response['MessageId'], wrapped_content)
                            emails_sent.append(f"✅ {result['Email']}")
                        else:
                            update_log(log_container, f"Failed to send email to: {result['Email']}", 'error')
                            save_email_campaign(session, result['Email'], template.id, 'Failed', datetime.utcnow(), template.subject, None, wrapped_content)
                            emails_sent.append(f"❌ {result['Email']}")

            leads_container.dataframe(pd.DataFrame({"Leads Found": leads_found, "Emails Sent": emails_sent + [""] * (len(leads_found) - len(emails_sent))}))
            progress_bar.progress((i + 1) / len(search_terms))

        st.subheader("Search Results")
        st.dataframe(pd.DataFrame(results))
        if enable_email_sending:
            st.subheader("Email Sending Results")
            success_rate = sum(1 for e in emails_sent if e.startswith("✅")) / len(emails_sent) if emails_sent else 0
            st.metric("Email Sending Success Rate", f"{success_rate:.2%}")
        st.download_button("Download CSV", pd.DataFrame(results).to_csv(index=False).encode('utf-8'), "search_results.csv", "text/csv")

def fetch_search_terms_with_lead_count(s): return pd.DataFrame(s.query(SearchTerm.term, func.count(distinct(Lead.id)).label('lead_count'), func.count(distinct(EmailCampaign.id)).label('email_count')).join(LeadSource, SearchTerm.id == LeadSource.search_term_id).join(Lead, LeadSource.lead_id == Lead.id).outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id).group_by(SearchTerm.term).all(), columns=['Term', 'Lead Count', 'Email Count'])

def ai_automation_loop(s, log_c, leads_c):
    if not all([s,log_c,leads_c]): raise ValueError("Missing params")
    try:
        with s.begin_nested():
            automation_state = get_automation_state()
            automation_state.update({'running':True, 'last_run':datetime.now()})
            while st.session_state.get('automation_status',False):
                automation_state['current_cycle'] += 1
                automation_logs, total_search_terms, total_emails_sent = [], 0, 0
                log_c.info("Starting automation cycle")
                if not (kb_info:=get_knowledge_base_info(s,get_active_project_id())): log_c.warning("Knowledge Base not found"); time.sleep(3600); continue
                optimized_terms = generate_optimized_search_terms(s,[t.term for t in s.query(SearchTerm).filter_by(project_id=get_active_project_id())],kb_info)
                st.subheader("Optimized Search Terms"); st.write(", ".join(optimized_terms))
                progress_bar = st.progress(0)
                for idx,term in enumerate(optimized_terms):
                    results = manual_search(s,[term],10,True)
                    new_leads = [(lead.id,lead.email) for res in results['results'] if (lead:=save_lead(s,res['Email'],url=res['URL']))]
                    if new_leads and (template:=s.query(EmailTemplate).filter_by(project_id=get_active_project_id(),campaign_id=get_active_campaign_id()).first()):
                        from_email = kb_info.get('contact_email','hello@indosy.com')
                        reply_to = kb_info.get('contact_email','eugproductions@gmail.com')
                        logs,sent_count = bulk_send_emails(s,template.id,from_email,reply_to,[{'Email':email} for _,email in new_leads])
                        automation_logs.extend(logs); total_emails_sent += sent_count
                    leads_c.text_area("New Leads Found","\n".join(email for _,email in new_leads),height=200)
                    progress_bar.progress((idx+1)/len(optimized_terms))
                st.success(f"Cycle done: {total_search_terms} terms, {total_emails_sent} emails")
                time.sleep(3600)
    except Exception as e: logging.error(f"Automation error: {e}"); log_c and log_c.error(f"Failed: {e}"); raise

def shuffle_keywords(term): return " ".join(random.sample(term.split(),len(term.split())))

def get_page_description(html): return (soup:=BeautifulSoup(html,'html.parser')).find('meta',attrs={'name':'description'}) and soup.find('meta',attrs={'name':'description'})['content'] or "No description found"

def is_valid_email(email):
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
        email_cfg = fetch_email_settings(session)[0] if fetch_email_settings(session) else {}
        email = email_cfg.get('email')
        template = session.query(EmailTemplate).first()
        res = manual_search(session, [t.term for t in terms], 10, True, False, False, True, "EN", True, st.empty(), email, email, f"{template.id}: {template.template_name}" if template else None)
    st.success(f"Quick scan completed! Found {len(res['results'])} new leads.")
    return {"new_leads": len(res['results']), "terms_used": [t.term for t in terms]}

def bulk_send_emails(
    session: Session,
    template_id: int,
    from_email: str,
    reply_to: str,
    leads: List[Dict[str, str]],
    progress_bar: Optional[Any] = None,
    status_text: Optional[Any] = None,
    results: Optional[List] = None,
    log_container: Optional[Any] = None
) -> Tuple[List[str], int]:
    """Send bulk emails with proper error handling and tracking"""
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        return [], 0

    logs, success_count = [], 0
    
    for i, lead in enumerate(leads):
        try:
            if is_valid_email(lead['Email']):
                response, tracking_id = send_email_ses(
                    session, from_email, lead['Email'],
                    template.subject, template.body_content,
                    reply_to=reply_to
                )
                
                save_email_campaign(
                    session, lead['Email'], template_id,
                    'sent' if response else 'failed',
                    datetime.utcnow(),
                    template.subject,
                    response.get('MessageId', f"{'sent' if response else 'failed'}-{uuid.uuid4()}"),
                    template.body_content
                )
                
                success_count += bool(response)
                logs.append(f"{'✅' if response else '❌'} {'Sent to' if response else 'Failed:'} {lead['Email']}")
                
                if progress_bar:
                    progress_bar.progress((i + 1) / len(leads))
                if status_text:
                    status_text.text(f"{i + 1}/{len(leads)}")
                if results is not None:
                    results.append({"Email": lead['Email'], "Status": 'sent' if response else 'failed'})
                if log_container:
                    log_container.text(logs[-1])
                
        except Exception as e:
            logs.append(f"❌ Error: {lead['Email']} ({str(e)})")
            
    return logs, success_count

def view_campaign_logs():
    st.header("Email Logs")
    with safe_db_connection() as session:
        logs = fetch_all_email_logs(session)
        if logs.empty: st.info("No email logs found."); return
        st.write(f"Total emails sent: {len(logs)}\nSuccess rate: {(logs['Status'] == 'sent').mean():.2%}")
        start_date, end_date = st.columns(2)[0].date_input("Start Date", logs['Sent At'].min().date()), st.columns(2)[1].date_input("End Date", logs['Sent At'].max().date())
        filtered = logs[(logs['Sent At'].dt.date >= start_date) & (logs['Sent At'].dt.date <= end_date)]
        if search := st.text_input("Search by email or subject"): filtered = filtered[filtered['Email'].str.contains(search, case=False) | filtered['Subject'].str.contains(search, case=False)]
        st.columns(3)[0].metric("Emails Sent", len(filtered))
        st.columns(3)[1].metric("Unique Recipients", filtered['Email'].nunique())
        st.columns(3)[2].metric("Success Rate", f"{(filtered['Status'] == 'sent').mean():.2%}")
        st.bar_chart(filtered.resample('D', on='Sent At')['Email'].count())
        for _, log in filtered.iterrows():
            with st.expander(f"{log['Sent At'].strftime('%Y-%m-%d %H:%M:%S')} - {log['Email']} - {log['Status']}"):
                st.write(f"**Subject:** {log['Subject']}\n**Content Preview:** {log['Content'][:100]}...")
                if st.button("View Full Email", key=f"view_email_{log['ID']}"): st.components.v1.html(wrap_email_body(log['Content']), height=400, scrolling=True)
                if log['Status'] != 'sent': st.error(f"Status: {log['Status']}")
        page = st.number_input("Page", 1, (len(filtered)-1)//20+1, 1)
        st.table(filtered.iloc[(page-1)*20:page*20][['Sent At', 'Email', 'Subject', 'Status']])
        if st.button("Export Logs to CSV"): st.download_button("Download CSV", filtered.to_csv(index=False), "email_logs.csv", "text/csv")

def fetch_all_email_logs(session):
    try: return pd.DataFrame({k: [getattr(ec, k) if k != 'sent_at' else ec.sent_at for ec in session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()] for k in ['ID', 'Sent At', 'Email', 'Template', 'Subject', 'Content', 'Status', 'Message ID', 'Campaign ID', 'Lead Name', 'Lead Company']})
    except SQLAlchemyError as e: logging.error(f"Database error in fetch_all_email_logs: {str(e)}"); return pd.DataFrame()

def update_lead(session, lead_id, data): 
    try: return bool(session.query(Lead).filter(Lead.id == lead_id).update(data))
    except SQLAlchemyError as e: logging.error(f"Error updating lead {lead_id}: {str(e)}"); session.rollback(); return False

def delete_lead(session, lead_id):
    try: return bool(session.query(Lead).filter(Lead.id == lead_id).delete())
    except SQLAlchemyError as e: logging.error(f"Error deleting lead {lead_id}: {str(e)}"); session.rollback(); return False

def is_valid_email(email):
    try: validate_email(email); return True 
    except EmailNotValidError: 
        return False
def email_templates_page():
    def validate_template(n,s,b): return (True,"") if all([n and len(n)>=3,s,b and len(b)>=10]) else (False,"Invalid")
    @st.cache_data(ttl=300)
    def get_cached_templates(s): return s.query(EmailTemplate).all()
    @st.cache_data(ttl=300)
    def get_cached_preview(i,b): return wrap_email_body(b)

    with safe_db_connection() as session:
        # Fix template cache initialization
        if not st.session_state.app_state.get('email_template_cache'):
            st.session_state.app_state['email_template_cache'] = get_cached_templates(session)
        
        templates = st.session_state.app_state['email_template_cache']
        
        st.header("Email Templates")
        
        with st.expander("Create New Template", False):
            n = st.text_input("Template Name", key="new_template_name")
            if st.checkbox("Use AI", key="use_ai"):
                p = st.text_area("AI Prompt", key="ai_prompt")
                kb = get_knowledge_base_info(s,get_active_project_id()) if st.checkbox("Use KB") else None
                if st.button("Generate", key="gen_ai"):
                    with st.spinner("Generating..."):
                        g = generate_or_adjust_email_template(p,kb)
                        if n:
                            t = EmailTemplate(template_name=n,subject=g.get("subject",""),body_content=g.get("body",""),campaign_id=get_active_campaign_id(), project_id=get_active_project_id())
                            session.add(t); session.commit(); st.success("Created!"); templates = get_cached_templates(s)
                        st.subheader("Generated"); st.text(f"Subject: {g.get('subject','')}"); st.components.v1.html(wrap_email_body(g.get('body','')),height=400,scrolling=True)
            else:
                sj,bd = st.text_input("Subject"),st.text_area("Body",height=200)
                if st.button("Create") and all([n,sj,bd]):
                    t = EmailTemplate(
                        template_name=n,
                        subject=sj,
                        body_content=bd,
                        campaign_id=get_active_campaign_id(),
                        project_id=get_active_project_id()  # Add project_id
                    )
                    session.add(t)
                    session.commit()
                    st.success("Created!")
                    st.session_state.app_state['email_template_cache'] = get_cached_templates(session)

        if templates:
            st.subheader("Existing Templates")
            for template in templates:
                with st.expander(f"Template: {template.template_name}"):
                    # Add error handling for template operations
                    try:
                        c1, c2 = st.columns(2)
                        subject = c1.text_input("Subject", template.subject, key=f"s_{template.id}")
                        ai_custom = c2.checkbox("AI", template.is_ai_customizable, key=f"ai_{template.id}")
                        body = st.text_area("Body", template.body_content, height=200, key=f"b_{template.id}")
                        
                        # Fix preview rendering
                        preview = wrap_email_body(body)
                        st.components.v1.html(preview, height=400, scrolling=True)
                        
                        # Add save confirmation
                        if st.button("Save", key=f"save_{template.id}"):
                            template.subject = subject
                            template.body_content = body 
                            template.is_ai_customizable = ai_custom
                            session.commit()
                            st.success("Saved!")
                            # Refresh cache
                            st.session_state.app_state['email_template_cache'] = get_cached_templates(session)
                            
                    except Exception as e:
                        st.error(f"Error updating template: {str(e)}")
                        session.rollback()

def get_email_preview(session, template_id, from_email, reply_to):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        return "<p>Not found</p>"
    return f"""
        <div style='margin-bottom:20px;color:#666'>
            <strong>From:</strong> {from_email}<br>
            <strong>Reply-To:</strong> {reply_to}<br>
            <strong>Subject:</strong> {template.subject}
        </div>
        {wrap_email_body(template.body_content)}
    """

def fetch_all_search_terms(s): return s.query(SearchTerm).all()
def get_knowledge_base_info(s,p): return (k:=s.query(KnowledgeBase).filter_by(project_id=p).first()) and k.to_dict()
def get_email_template_by_name(s,n): return s.query(EmailTemplate).filter_by(template_name=n).first()
def knowledge_base_page():
    st.title("Knowledge Base")
    with safe_db_connection() as session:
        if not (projects := fetch_projects(session)): return st.warning("No projects found. Please create a project first.")
        project_id = int((selected := st.selectbox("Select Project", projects)).split(":")[0])
        set_active_project_id(project_id)
        kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        with st.form("knowledge_base_form"):
            fields = ['kb_name','kb_bio','kb_values','contact_name','contact_role','contact_email','company_description','company_mission','company_target_market','company_other','product_name','product_description','product_target_customer','product_other','other_context','example_email']
            form_data = {f: (st.text_input if f in ['kb_name','contact_name','contact_role','contact_email','product_name'] else st.text_area)(f.replace('_',' ').title(), value=getattr(kb,f,'')) for f in fields}
            if st.form_submit_button("Save Knowledge Base"):
                try:
                    form_data.update({'project_id':project_id,'created_at':datetime.utcnow()})
                    [setattr(kb,k,v) for k,v in form_data.items()] if kb else session.add(KnowledgeBase(**form_data))
                    session.commit()
                    st.success("Knowledge Base saved successfully!", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving Knowledge Base: {str(e)}")
                    session.rollback()

def autoclient_ai_page():
    if 'automation_state' not in st.session_state: st.session_state.automation_state = {'status':False,'logs':[],'leads_found':0,'emails_sent':0,'last_run':None}
    safe_automation = lambda s,l,c: ai_automation_loop(s,l,c) if st.spinner("Running automation...") else None
    st.header("AutoclientAI - Automated Lead Generation")
    with st.expander("Knowledge Base Info",False):
        if not (kb:=get_knowledge_base_info(safe_db_connection().__enter__(),get_active_project_id())): return st.error("Knowledge Base not found. Set up first.")
        st.json(kb)
    if st.button("Generate Terms",key="gen_terms") and (user_input:=st.text_area("Enter context/goals:",help="For targeted search terms")):
        with st.spinner("Generating..."):
            if terms:=generate_optimized_search_terms(safe_db_connection().__enter__(),[t.term for t in safe_db_connection().__enter__().query(SearchTerm).filter_by(project_id=get_active_project_id()).all()],kb):
                st.session_state.optimized_terms=terms; st.success("Terms optimized!"); st.subheader("Results"); st.write(", ".join(terms))
            else: st.error("Failed to generate terms")
    if st.button("Start",key="start"): st.session_state.update({"automation_status":True,"automation_logs":[],"total_leads_found":0,"total_emails_sent":0}); st.success("Started!")
    if st.session_state.get('automation_status',False):
        try: safe_automation(safe_db_connection().__enter__(),st.empty(),st.empty())
        except Exception as e: st.error(f"Error: {e}"); st.session_state.automation_status=False
    if not st.session_state.get('automation_status',False) and (logs:=st.session_state.get('automation_logs')):
        st.subheader("Results"); st.metric("Leads Found",st.session_state.total_leads_found); st.metric("Emails Sent",st.session_state.total_emails_sent)
        st.subheader("Logs"); st.text_area("","\n".join(logs),height=300)
    if 'email_logs' in st.session_state:
        st.subheader("Email Logs"); df=pd.DataFrame(st.session_state.email_logs); st.dataframe(df)
        st.metric("Success Rate",f"{(df['Status']=='sent').mean()*100:.2f}%")
    st.subheader("Debug"); st.json(st.session_state); st.write("Func:",autoclient_ai_page.__name__); st.write("State:",list(st.session_state.keys()))

def update_search_terms(s,t): [setattr(s.query(SearchTerm).filter_by(term=term,project_id=get_active_project_id()).first(),'group',g) if s.query(SearchTerm).filter_by(term=term,project_id=get_active_project_id()).first() else s.add(SearchTerm(term=term,group=g,project_id=get_active_project_id())) for g,terms in t.items() for term in terms]; s.commit()

def update_results_display(c,r): c.markdown(f'<style>.rc{{max-height:400px;overflow-y:auto;border:1px solid rgba(49,51,63,.2);border-radius:.25rem;padding:1rem;background:rgba(49,51,63,.1)}}.re{{margin-bottom:.5rem;padding:.5rem;background:rgba(255,255,255,.1);border-radius:.25rem}}</style><div class="rc"><h4>Leads ({len(r)})</h4>{"".join(f"<div class=re><strong>{x["Email"]}</strong><br>{x["URL"]}</div>" for x in r[-10:])}</div>',unsafe_allow_html=True)

def automation_control_panel_page():
    """Control panel for automation with proper state management"""
    try:
        with safe_db_connection() as session:  # Use context manager
            if st.session_state.get('automation_running', False):
                run_automation(session)
    except Exception as e:
        st.error(f"Automation error: {str(e)}")
        st.session_state.automation_running = False

def generate_optimized_search_terms(
    session: Session,
    base_terms: List[str],
    knowledge_base: Dict[str, Any]
) -> List[str]:
    """Generate optimized search terms using AI"""
    try:
        prompt = f"Generate 5 optimized search terms based on: {', '.join(base_terms)}. Context: {knowledge_base}"
        response = get_ai_response(prompt)
        return response.split('\n') if response else []
    except Exception as e:
        logging.error(f"Error generating search terms: {e}")
        return []

def update_display(c,i,t,y): c.markdown(f"<h4>{t}</h4>",unsafe_allow_html=True); [c.text(x) for x in i[-10:]]

def get_search_terms(session: Session) -> List[str]:
    return [t.term for t in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]

def get_ai_response(prompt: str) -> str:
    """Get AI response with proper error handling"""
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        return ""

def fetch_email_settings(s): return [{"id":s.id,"name":s.name,"email":s.email} for s in s.query(EmailSettings).all()] if s else []

def wrap_email_body(b,t=None): return f'<!DOCTYPE html><html><head><meta charset="UTF-8"><style>body{{font:16px Arial;max-width:600px;margin:auto;padding:20px}}</style></head><body>{b}{f"""<img src="https://autoclient-email-analytics.trigox.workers.dev/track?id={t}&type=open" style="display:none"/>""" if t else ""}</body></html>'

def fetch_sent_email_campaigns(s): return pd.DataFrame({k:[getattr(x,k) for x in s.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead),joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()] for k in ['id','sent_at','email','template_name','subject','content','status','message_id','campaign_id','lead_name','company']}) if True else pd.DataFrame()

def display_logs(c,l): c.markdown(f'<style>.lc{{max-height:300px;overflow-y:auto;border:1px solid rgba(49,51,63,.2);border-radius:.25rem;padding:1rem}}.le{{margin-bottom:.5rem;padding:.5rem;border-radius:.25rem;background:rgba(28,131,225,.1)}}</style><div class="lc">{"".join(f"<div class=le>{x}</div>" for x in l[-20:])}</div>',unsafe_allow_html=True) if l else c.info("No logs")

def view_sent_email_campaigns():
    st.header("Sent Campaigns")
    try:
        with safe_db_connection() as session:  # Use context manager
            df = fetch_sent_email_campaigns(session)
            if not df.empty:
                st.dataframe(df)
                st.subheader("Details")
                sel = st.selectbox("Select campaign", df['ID'].tolist())
                if sel:
                    content = df[df['ID']==sel]['Content'].iloc[0]
                    st.text_area("Content", content or "No content", height=300)
            else:
                st.info("No campaigns")
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error in view_sent_email_campaigns: {e}")

@st.cache_resource(ttl=3600)
def get_database_engine():
    """Get database engine with connection pooling"""
    try:
        return create_engine(
            DATABASE_URL,
            pool_size=20,
            max_overflow=0,
            pool_pre_ping=True  # Add connection testing
        )
    except Exception as e:
        logging.error(f"Failed to create database engine: {e}")
        raise

@st.cache_resource
def get_cached_session(): return sessionmaker(bind=get_database_engine())()

@st.cache_data(ttl=300)
def fetch_cached_leads(s): return fetch_leads_with_sources(s)

@st.cache_data(ttl=300)
def fetch_cached_templates(s): return fetch_email_templates(s)

@st.cache_data(ttl=300)
def fetch_cached_settings(s): return fetch_email_settings(s)

def init_session_state():
    """Initialize all required session state variables"""
    defaults = {
        'app_init': False,
        'app_state': {
            'search_in_progress': False,
            'current_search_results': [],
            'last_search_terms': [],
            'automation': {'running': False},
            'cache': {},
            'window_id': str(uuid.uuid4()),
            'last_page': None
        },
        'automation_state': {
            'running': False,
            'current_cycle': 0,
            'last_run': None,
            'logs': [],
            'total_leads': 0,
            'total_emails': 0
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Use in main
def main():
    try:
        init_session_state()
        ctx = get_streamlit_context()
        if not ctx:
            st.error("Failed to get Streamlit context")
            return

        # Initialize session state
        if not hasattr(st.session_state, 'app_init'):
            initialize_session_state()
            st.session_state.app_init = True

        # Sidebar navigation
        with st.sidebar:
            selected_page = option_menu(
                "Navigation",
                list(pages.keys()),
                ["search", "send", "people", "key", "envelope", "folder", 
                 "book", "robot", "gear", "list-check", "gear", "envelope-open"],
                menu_icon="cast",
                default_index=0,
                key=f'nav_menu_{st.session_state.app_state["window_id"]}'
            )

        # Handle page changes
        if st.session_state.app_state['last_page'] != selected_page:
            st.session_state.app_state['last_page'] = selected_page
            st.session_state.app_state['needs_reload'] = True
            st.session_state.app_state['cache'] = {k: None for k in st.session_state.app_state['cache']}

        # Render selected page
        if selected_page in pages:
            pages[selected_page]()
        else:
            st.error(f"Page {selected_page} not found")

    except Exception as e:
        logging.exception("Critical error in main application")
        st.error(f"An error occurred: {str(e)}")
        
def validate_environment():
    """Validate required environment variables"""
    required_vars = {
        "SUPABASE_DB_HOST": DB_HOST,
        "SUPABASE_DB_NAME": DB_NAME,
        "SUPABASE_DB_USER": DB_USER,
        "SUPABASE_DB_PASSWORD": DB_PASSWORD,
        "SUPABASE_DB_PORT": DB_PORT
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Add to main
if __name__ == "__main__":
    try:
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # Initialize database
        engine = init_database()
        SessionLocal = sessionmaker(bind=engine)
        
        # Run main application
        main()
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        st.error("Application failed to start. Please check the logs.")

# Remove duplicate get_automation_state functions and keep this single implementation
def get_automation_state() -> Dict[str, Any]:
    """Get current automation state with defaults"""
    if 'automation_state' not in st.session_state:
        st.session_state.automation_state = {
            'running': False,
            'current_cycle': 0,
            'last_run': None,
            'logs': [],
            'total_leads': 0,
            'total_emails': 0
        }
    return st.session_state.automation_state

def get_streamlit_context():
    """Get Streamlit runtime context safely"""
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            # Create new context if none exists
            ctx = type('Context', (), {'session_id': str(uuid.uuid4())})()
            add_script_run_ctx(ctx)
        return ctx
    except Exception as e:
        logging.warning(f"Could not get Streamlit context: {e}")
        return None

# Then replace direct get_script_run_ctx() calls with:
ctx = get_streamlit_context()

def get_openai_client():
    """Get OpenAI client with proper error handling"""
    try:
        with safe_db_connection() as session:
            settings = session.query(Settings).filter_by(setting_type='general').first()
            if not settings or 'openai_api_key' not in settings.value:
                raise ValueError("OpenAI API key not configured in settings")
            
            return OpenAI(
                api_key=settings.value['openai_api_key'],
                base_url=settings.value.get('openai_api_base', 'https://api.openai.com/v1'),
            )
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        raise

# Replace the current environment initialization with this:
def get_environment_settings():
    """Get environment settings with fallback to database settings"""
    try:
        # Try loading from .env first
        load_dotenv()
        
        # Get from environment
        env_settings = {
            "DB_HOST": os.getenv("SUPABASE_DB_HOST"),
            "DB_NAME": os.getenv("SUPABASE_DB_NAME"), 
            "DB_USER": os.getenv("SUPABASE_DB_USER"),
            "DB_PASSWORD": os.getenv("SUPABASE_DB_PASSWORD"),
            "DB_PORT": os.getenv("SUPABASE_DB_PORT")
        }

        # If any env var is missing, try getting from settings table
        if not all(env_settings.values()):
            with safe_db_connection() as session:
                db_settings = session.query(Settings).filter_by(setting_type='database').first()
                if db_settings and db_settings.value:
                    missing_keys = [k for k, v in env_settings.items() if not v]
                    for key in missing_keys:
                        env_settings[key] = db_settings.value.get(key.lower())
                        if env_settings[key]:  # Set environment variable if found in settings
                            os.environ[f"SUPABASE_{key}"] = str(env_settings[key])

        return env_settings
    except Exception as e:
        logging.error(f"Failed to get environment settings: {e}")
        raise

# Update the database initialization code:
def init_database():
    """Initialize database connection with settings fallback"""
    env_settings = get_environment_settings()
    
    if not all(env_settings.values()):
        raise ValueError("Database configuration incomplete. Check .env file or database settings.")
        
    DATABASE_URL = (
        f"postgresql://{env_settings['DB_USER']}:{env_settings['DB_PASSWORD']}"
        f"@{env_settings['DB_HOST']}:{env_settings['DB_PORT']}/{env_settings['DB_NAME']}"
    )
    
    return create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
