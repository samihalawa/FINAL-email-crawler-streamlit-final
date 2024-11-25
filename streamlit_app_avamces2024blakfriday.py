import os, json, re, logging, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select, text, distinct, and_, or_, Index
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from openai import OpenAI 
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from streamlit_tags import st_tags
import plotly.express as px
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager

DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()
DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT = map(os.getenv, ["SUPABASE_DB_HOST", "SUPABASE_DB_NAME", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD", "SUPABASE_DB_PORT"])
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=10, pool_timeout=30, pool_recycle=1800)
SessionLocal, Base = sessionmaker(bind=engine), declarative_base()

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

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

def transactional(func):
    """Transaction decorator with proper error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with db_session() as session:
            try:
                result = func(session, *args, **kwargs)
                return result
            except Exception as e:
                session.rollback()
                logging.error(f"Transaction failed: {str(e)}")
                raise
    return wrapper

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
    # Remove the domain column
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
    daily_limit = Column(BigInteger, default=200)
    quota = relationship("EmailQuota", back_populates="email_settings", uselist=False)

# Add the EmailQuota model after the EmailSettings model
class EmailQuota(Base):
    __tablename__ = 'email_quotas'
    id = Column(BigInteger, primary_key=True)
    email_settings_id = Column(BigInteger, ForeignKey('email_settings.id', ondelete='CASCADE'))
    emails_sent_today = Column(BigInteger, default=0)
    last_reset = Column(DateTime(timezone=True))
    error_count = Column(BigInteger, default=0)
    last_error = Column(Text)
    last_error_time = Column(DateTime(timezone=True))
    lock_version = Column(BigInteger, default=0)  # For optimistic locking
    email_settings = relationship("EmailSettings", back_populates="quota")

    __table_args__ = (
        Index('idx_email_quota_settings', email_settings_id),
        Index('idx_email_quota_reset', last_reset),
    )

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def test_email_settings(session, settings_id):
    """Test email settings by sending a test email"""
    try:
        settings = session.query(EmailSettings).get(settings_id)
        if not settings:
            return False, "Email settings not found"
            
        test_subject = "Test Email from AutoclientAI"
        test_body = f"""
        <h2>Test Email</h2>
        <p>This is a test email from your AutoclientAI email settings: {settings.name}</p>
        <p>Provider: {settings.provider}</p>
        <p>Sent at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        response, tracking_id = send_email_ses(
            session, 
            settings.email, 
            settings.email,  # Send to self
            test_subject,
            test_body,
            reply_to=settings.email
        )
        
        if response:
            return True, "Test email sent successfully!"
        return False, "Failed to send test email"
        
    except Exception as e:
        return False, f"Error testing email settings: {str(e)}"

def get_settings_from_db(session, setting_type):
    """Get settings from database by type"""
    try:
        setting = session.query(Settings).filter_by(setting_type=setting_type).first()
        return setting.value if setting else None
    except Exception as e:
        logging.error(f"Error fetching {setting_type} settings: {str(e)}")
        return None

def update_settings_in_db(session, setting_type, value):
    """Update or create settings in database"""
    try:
        setting = session.query(Settings).filter_by(setting_type=setting_type).first()
        if setting:
            setting.value = value
            setting.updated_at = datetime.utcnow()
        else:
            setting = Settings(
                name=f"{setting_type} Settings",
                setting_type=setting_type,
                value=value
            )
            session.add(setting)
        session.commit()
        return True
    except Exception as e:
        logging.error(f"Error updating {setting_type} settings: {str(e)}")
        session.rollback()
        return False

def settings_page():
    st.title("Settings")
    
    with db_session() as session:
        tab1, tab2, tab3 = st.tabs(["Email Settings", "OpenAI Settings", "AWS Settings"])
        
        with tab1:
            st.header("Email Settings")
            email_settings = session.query(EmailSettings).all()
            
            # Display existing settings
            if email_settings:
                st.subheader("Existing Email Settings")
                for setting in email_settings:
                    with st.expander(f"{setting.name} ({setting.email})"):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"Provider: {setting.provider}")
                            st.write(f"Daily Limit: {setting.daily_limit}")
                            if setting.quota:
                                st.write(f"Emails Sent Today: {setting.quota.emails_sent_today}")
                                st.write(f"Error Count: {setting.quota.error_count}")
                        
                        with col2:
                            if st.button("Test Email", key=f"test_{setting.id}"):
                                with st.spinner("Sending test email..."):
                                    success, message = test_email_settings(session, setting.id)
                                    if success:
                                        st.success(message)
                                    else:
                                        st.error(message)
                        
                        with col3:
                            if st.button("Delete", key=f"delete_{setting.id}"):
                                session.delete(setting)
                                session.commit()
                                st.success(f"Deleted {setting.name}")
                                st.experimental_rerun()
            
            # Add new email settings form
            st.subheader("Add New Email Settings")
            with st.form("email_settings_form"):
                name = st.text_input("Settings Name", placeholder="e.g., Company Gmail")
                email = st.text_input("Email Address")
                provider = st.selectbox("Provider", ["smtp", "ses"])
                daily_limit = st.number_input("Daily Email Limit", min_value=1, value=200)
                
                if provider == "smtp":
                    smtp_server = st.text_input("SMTP Server", placeholder="smtp.gmail.com")
                    smtp_port = st.number_input("SMTP Port", value=587)
                    smtp_username = st.text_input("SMTP Username")
                    smtp_password = st.text_input("SMTP Password", type="password")
                    
                else:  # ses
                    aws_settings = get_settings_from_db(session, 'aws')
                    aws_access_key_id = st.text_input("AWS Access Key ID", 
                                                    value=aws_settings.get('aws_access_key_id', '') if aws_settings else '')
                    aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password",
                                                        value=aws_settings.get('aws_secret_access_key', '') if aws_settings else '')
                    aws_region = st.text_input("AWS Region", 
                                             value=aws_settings.get('aws_region', 'us-east-1') if aws_settings else 'us-east-1')
                
                if st.form_submit_button("Save & Test"):
                    try:
                        # Create settings object
                        settings_data = {
                            "name": name,
                            "email": email,
                            "provider": provider,
                            "daily_limit": daily_limit
                        }
                        
                        if provider == "smtp":
                            settings_data.update({
                                "smtp_server": smtp_server,
                                "smtp_port": smtp_port,
                                "smtp_username": smtp_username,
                                "smtp_password": smtp_password
                            })
                        else:
                            settings_data.update({
                                "aws_access_key_id": aws_access_key_id,
                                "aws_secret_access_key": aws_secret_access_key,
                                "aws_region": aws_region
                            })
                            # Update AWS settings in database
                            update_settings_in_db(session, 'aws', {
                                "aws_access_key_id": aws_access_key_id,
                                "aws_secret_access_key": aws_secret_access_key,
                                "aws_region": aws_region
                            })
                        
                        # Save settings
                        new_settings = EmailSettings(**settings_data)
                        session.add(new_settings)
                        session.flush()
                        
                        # Create quota record
                        quota = EmailQuota(
                            email_settings_id=new_settings.id,
                            last_reset=datetime.utcnow()
                        )
                        session.add(quota)
                        
                        # Test the settings
                        success, message = test_email_settings(session, new_settings.id)
                        
                        if success:
                            session.commit()
                            st.success("Settings saved and test email sent successfully!")
                        else:
                            session.rollback()
                            st.error(f"Settings test failed: {message}")
                        
                    except Exception as e:
                        session.rollback()
                        st.error(f"Error saving settings: {str(e)}")
        
        with tab2:
            st.header("OpenAI Settings")
            openai_settings = get_settings_from_db(session, 'openai')
            
            with st.form("openai_settings_form"):
                api_key = st.text_input("API Key", 
                                      value=openai_settings.get('api_key', '') if openai_settings else '',
                                      type="password")
                model = st.selectbox("Model", 
                                   ["gpt-4", "gpt-3.5-turbo"],
                                   index=0 if openai_settings and openai_settings.get('model') == 'gpt-4' else 1)
                temperature = st.slider("Temperature", 
                                     min_value=0.0, max_value=1.0, value=0.7, step=0.1)
                
                if st.form_submit_button("Save OpenAI Settings"):
                    settings_value = {
                        "api_key": api_key,
                        "model": model,
                        "temperature": temperature
                    }
                    if update_settings_in_db(session, 'openai', settings_value):
                        st.success("OpenAI settings saved successfully!")
                    else:
                        st.error("Failed to save OpenAI settings")
        
        with tab3:
            st.header("AWS Settings")
            aws_settings = get_settings_from_db(session, 'aws')
            
            with st.form("aws_settings_form"):
                aws_access_key_id = st.text_input("AWS Access Key ID",
                                                value=aws_settings.get('aws_access_key_id', '') if aws_settings else '')
                aws_secret_access_key = st.text_input("AWS Secret Access Key",
                                                    value=aws_settings.get('aws_secret_access_key', '') if aws_settings else '',
                                                    type="password")
                aws_region = st.text_input("AWS Region",
                                         value=aws_settings.get('aws_region', 'us-east-1') if aws_settings else 'us-east-1')
                
                if st.form_submit_button("Save AWS Settings"):
                    settings_value = {
                        "aws_access_key_id": aws_access_key_id,
                        "aws_secret_access_key": aws_secret_access_key,
                        "aws_region": aws_region
                    }
                    if update_settings_in_db(session, 'aws', settings_value):
                        st.success("AWS settings saved successfully!")
                    else:
                        st.error("Failed to save AWS settings")

def send_email_ses(session, from_email, to_email, subject, body, reply_to=None):
    """Send email using AWS SES with proper quota management"""
    try:
        # Get email settings and check quota
        email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
        if not email_settings:
            logging.error(f"No email settings found for {from_email}")
            return None, None

        # Check daily quota
        quota = email_settings.quota
        if quota and quota.emails_sent_today >= email_settings.daily_limit:
            logging.warning(f"Daily email quota exceeded for {from_email}")
            return None, None

        if email_settings.provider.lower() == 'ses':
            # Configure AWS SES client
            ses_client = boto3.client(
                'ses',
                aws_access_key_id=email_settings.aws_access_key_id,
                aws_secret_access_key=email_settings.aws_secret_access_key,
                region_name=email_settings.aws_region
            )

            # Prepare email
            email_msg = {
                'Source': from_email,
                'Destination': {'ToAddresses': [to_email]},
                'Message': {
                    'Subject': {'Data': subject},
                    'Body': {'Html': {'Data': body}}
                }
            }
            if reply_to:
                email_msg['ReplyToAddresses'] = [reply_to]

            # Send email
            response = ses_client.send_email(**email_msg)
            tracking_id = str(uuid.uuid4())

            # Update quota
            if quota:
                quota.emails_sent_today += 1
                quota.last_error = None
                quota.last_error_time = None
                session.commit()

            return response, tracking_id

        elif email_settings.provider.lower() == 'smtp':
            # SMTP sending logic with quota management
            smtp_server = smtplib.SMTP(email_settings.smtp_server, email_settings.smtp_port)
            smtp_server.starttls()
            smtp_server.login(email_settings.smtp_username, email_settings.smtp_password)

            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            if reply_to:
                msg.add_header('Reply-To', reply_to)

            msg.attach(MIMEText(body, 'html'))

            smtp_server.send_message(msg)
            smtp_server.quit()

            tracking_id = str(uuid.uuid4())
            
            # Update quota
            if quota:
                quota.emails_sent_today += 1
                quota.last_error = None
                quota.last_error_time = None
                session.commit()

            return {'MessageId': f'smtp-{tracking_id}'}, tracking_id

    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        if quota:
            quota.error_count += 1
            quota.last_error = str(e)
            quota.last_error_time = datetime.utcnow()
            session.commit()
        return None, None

def categorize_email_error(error_msg):
    """Categorize email errors as temporary or permanent"""
    permanent_errors = [
        'invalid email address',
        'address rejected',
        'permanent failure',
        'invalid credentials',
        'authentication failed'
    ]
    
    error_msg = error_msg.lower()
    return 'permanent' if any(err in error_msg for err in permanent_errors) else 'temporary'

@transactional
def record_email_error(session, settings_id, error_message, error_category):
    """Record an error with categorization"""
    quota = session.query(EmailQuota).filter_by(email_settings_id=settings_id).with_for_update().first()
    if quota:
        quota.error_count += 2 if error_category == 'permanent' else 1
        quota.last_error = error_message
        quota.last_error_time = datetime.utcnow()
        quota.lock_version += 1

@transactional
def record_email_success(session, settings_id):
    """Record a successful email send with optimistic locking"""
    quota = session.query(EmailQuota).filter_by(email_settings_id=settings_id).with_for_update().first()
    if quota:
        quota.emails_sent_today += 1
        quota.error_count = 0
        quota.last_error = None
        quota.last_error_time = None
        quota.lock_version += 1

def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    """Save email campaign with proper error handling and validation"""
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
        logging.error(f"Database error saving email campaign: {str(e)}")
        session.rollback()
        return None
    except Exception as e:
        logging.error(f"Unexpected error saving email campaign: {str(e)}")
        session.rollback()
        return None

def update_log(log_container, message, level='info'):
    icon = {'info': '🔵', 'success': '🟢', 'warning': '🟠', 'error': '🔴', 'email_sent': '🟣'}.get(level, '⚪')
    log_entry = f"{icon} {message}"
    
    # Simple console logging without HTML
    print(f"{icon} {message.split('<')[0]}")  # Only print the first part of the message before any HTML tags
    
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    
    # HTML-formatted log entry for Streamlit display
    html_log_entry = f"{icon} {message}"
    st.session_state.log_entries.append(html_log_entry)
    
    # Update the Streamlit display with all logs
    log_html = f"<div style='height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.8em; line-height: 1.2;'>{'<br>'.join(st.session_state.log_entries)}</div>"
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

def save_lead_with_source(session, email, url, name, company, job_title, search_term_id, html_content, domain):
    try:
        # Add content extraction logic directly here
        soup = BeautifulSoup(html_content, 'html.parser')
        page_title = soup.title.string if soup.title else "No title"
        visible_text = ' '.join([text for text in soup.stripped_strings])
        
        lead = session.query(Lead).filter_by(email=email).first()
        if not lead:
            lead = Lead(
                email=email,
                first_name=name,
                last_name="",
                company=company,
                job_title=job_title,
                created_at=datetime.utcnow()
            )
            session.add(lead)
            session.flush()

        source = LeadSource(
            lead_id=lead.id,
            search_term_id=search_term_id,
            url=url,
            domain=domain,
            page_title=page_title,
            meta_description=get_page_description(html_content),
            content=visible_text[:1000],  # Store first 1000 chars of visible text
            http_status=200,
            created_at=datetime.utcnow()
        )
        session.add(source)
        session.commit()
        return lead
    except Exception as e:
        session.rollback()
        logging.error(f"Error saving lead and source: {str(e)}")
        return None

def manual_search(session, terms, num_results, ignore_previously_fetched=True, 
                 optimize_english=False, optimize_spanish=False, 
                 shuffle_keywords_option=False, language='ES', 
                 enable_email_sending=False, log_container=None, 
                 from_email=None, reply_to=None, email_template=None, 
                 lead_save_strategy="All"):
    """Updated manual search with proper lead saving strategies"""
    try:
        ua = UserAgent()
        results = []
        total_leads = 0
        domains_processed = set()
        domain_lead_count = {}

        # Add error handling for missing search term
        if not terms:
            if log_container:
                update_log(log_container, "No search terms provided", 'error')
            return {"total_leads": 0, "results": []}

        for original_term in terms:
            try:
                # Add campaign ID validation
                campaign_id = get_active_campaign_id()
                if not campaign_id:
                    if log_container:
                        update_log(log_container, "No active campaign selected", 'error')
                    return {"total_leads": 0, "results": []}

                search_term_id = add_or_get_search_term(session, original_term, campaign_id)
                search_term = shuffle_keywords(original_term) if shuffle_keywords_option else original_term
                search_term = optimize_search_term(search_term, 'english' if optimize_english else 'spanish') if (optimize_english or optimize_spanish) else search_term

                if log_container:
                    update_log(log_container, f"Searching for '{original_term}' (Used '{search_term}')")

                # Add timeout and retry logic for Google search
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        urls = google_search(search_term, num_results, lang=language)
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            raise
                        time.sleep(2 ** retry)  # Exponential backoff

                for url in urls:
                    domain = get_domain_from_url(url)
                    domain = domain.replace('www.', '')  # Normalize domain

                    # Domain processing logic based on strategy
                    if ignore_previously_fetched and domain in domains_processed:
                        if log_container:
                            update_log(log_container, f"Skipping Previously Fetched: {domain}", 'warning')
                        continue

                    if lead_save_strategy == "1 per domain" and domain_lead_count.get(domain, 0) > 0:
                        if log_container:
                            update_log(log_container, f"Skipping domain {domain}: Already have a lead", 'warning')
                        continue

                    try:
                        url = f"https://{url}" if not url.startswith(('http://', 'https://')) else url
                        response = requests.get(url, timeout=10, verify=False, headers={'User-Agent': ua.random})
                        response.raise_for_status()
                        html_content = response.text
                        soup = BeautifulSoup(html_content, 'html.parser')
                        emails = extract_emails_from_html(html_content)

                        if log_container:
                            update_log(log_container, f"Found {len(emails)} email(s) on {url}", 'success')

                        for email in filter(is_valid_email, emails):
                            email_domain = email.split('@')[1]
                            
                            # Apply lead saving strategy
                            if lead_save_strategy == "Same domain only" and email_domain != domain:
                                continue
                            
                            if domain not in domains_processed or lead_save_strategy == "All":
                                name, company, job_title = extract_info_from_page(soup)
                                
                                lead = save_lead_with_source(
                                    session, email, url, name, company, job_title,
                                    search_term_id, html_content, domain
                                )

                                if lead:
                                    total_leads += 1
                                    domain_lead_count[domain] = domain_lead_count.get(domain, 0) + 1
                                    
                                    results.append({
                                        'Email': email,
                                        'URL': url,
                                        'Lead Source': original_term,
                                        'Title': get_page_title(html_content),
                                        'Description': get_page_description(html_content),
                                        'Name': name,
                                        'Company': company,
                                        'Job Title': job_title,
                                        'Domain': domain,
                                        'Search Term ID': search_term_id
                                    })

                                    if log_container:
                                        update_log(log_container, f"Saved lead: {email}", 'success')
                                    
                                    domains_processed.add(domain)

                                    # Handle email sending if enabled
                                    if enable_email_sending and all([from_email, email_template, reply_to]):
                                        template = session.query(EmailTemplate).filter_by(id=int(email_template.split(":")[0])).first()
                                        if template:
                                            wrapped_content = wrap_email_body(template.body_content)
                                            response, tracking_id = send_email_ses(
                                                session, 
                                                from_email, 
                                                email, 
                                                template.subject, 
                                                wrapped_content, 
                                                reply_to=reply_to
                                            )
                                            
                                            if response:
                                                if log_container:
                                                    update_log(log_container, f"Sent email to: {email}", 'email_sent')
                                                save_email_campaign(
                                                    session, 
                                                    email, 
                                                    template.id, 
                                                    'Sent', 
                                                    datetime.utcnow(), 
                                                    template.subject, 
                                                    response['MessageId'], 
                                                    wrapped_content
                                                )
                                            else:
                                                if log_container:
                                                    update_log(log_container, f"Failed to send email to: {email}", 'error')
                                                save_email_campaign(
                                                    session, 
                                                    email, 
                                                    template.id, 
                                                    'Failed', 
                                                    datetime.utcnow(), 
                                                    template.subject, 
                                                    None, 
                                                    wrapped_content
                                                )
                                    break

                    except requests.RequestException as e:
                        if log_container:
                            update_log(log_container, f"Error processing URL {url}: {str(e)}", 'error')

            except Exception as e:
                if log_container:
                    update_log(log_container, f"Error processing term '{original_term}': {str(e)}", 'error')

        if log_container:
            update_log(log_container, f"Total leads found: {total_leads}", 'info')

        return {"total_leads": total_leads, "results": results}
        
    except Exception as e:
        logging.error(f"Critical error in manual_search: {str(e)}")
        if log_container:
            update_log(log_container, f"Critical error: {str(e)}", 'error')
        return {"total_leads": 0, "results": []}

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
            return {"subject": "AI Generated Subject", "body": f"<p>{response}</p>"}
    elif isinstance(response, dict):
        return response
    return {"subject": "", "body": "<p>Failed to generate email content.</p>"}

def fetch_leads_with_sources(session):
    """Fetch leads with their associated sources and email campaign data"""
    try:
        # Fix unclosed parenthesis in query
        query = (
            session.query(
                Lead,
                func.string_agg(LeadSource.url, ', ').label('sources'),
                func.max(EmailCampaign.sent_at).label('last_contact'),
                func.string_agg(EmailCampaign.status, ', ').label('email_statuses')
            )
            .outerjoin(LeadSource)
            .outerjoin(EmailCampaign)
            .group_by(Lead.id)
        )  # Close the query parenthesis
        results = query.all()
        
        # Create DataFrame with proper error handling and data validation
        try:
            df = pd.DataFrame([{
                'ID': lead.id,
                'Email': lead.email,
                'First Name': lead.first_name,
                'Last Name': lead.last_name, 
                'Company': lead.company,
                'Job Title': lead.job_title,
                'Created At': lead.created_at,
                'Source': sources if sources else '',
                'Last Contact': last_contact,
                'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted',
                'Delete': False
            } for lead, sources, last_contact, email_statuses in results])
            
            # Ensure all required columns exist
            required_cols = ['ID', 'Email', 'First Name', 'Last Name', 'Company', 'Job Title', 
                           'Created At', 'Source', 'Last Contact', 'Last Email Status', 'Delete']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = ''
                    
            return df
            
        except Exception as e:
            logging.error(f"Error creating DataFrame in fetch_leads_with_sources: {str(e)}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=required_cols)
            
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_leads_with_sources: {str(e)}")
        return pd.DataFrame(columns=['ID', 'Email', 'First Name', 'Last Name', 'Company', 'Job Title',
                                   'Created At', 'Source', 'Last Contact', 'Last Email Status', 'Delete'])
    except Exception as e:
        logging.error(f"Unexpected error in fetch_leads_with_sources: {str(e)}")
        return pd.DataFrame(columns=['ID', 'Email', 'First Name', 'Last Name', 'Company', 'Job Title',
                                   'Created At', 'Source', 'Last Contact', 'Last Email Status', 'Delete'])

def fetch_search_terms_with_lead_count(session):
    """Fetch search terms with their associated lead and email counts"""
    try:
        query = (
            session.query(
                SearchTerm.term,
                func.count(distinct(Lead.id)).label('lead_count'),
                func.count(distinct(EmailCampaign.id)).label('email_count')
            )
            .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
            .join(Lead, LeadSource.lead_id == Lead.id)
            .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
            .group_by(SearchTerm.term)
        )
        
        results = query.all()
        
        # Create DataFrame with proper error handling
        try:
            df = pd.DataFrame(results, columns=['Term', 'Lead Count', 'Email Count'])
            return df
        except Exception as e:
            logging.error(f"Error creating DataFrame in fetch_search_terms_with_lead_count: {str(e)}")
            return pd.DataFrame(columns=['Term', 'Lead Count', 'Email Count'])
            
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_search_terms_with_lead_count: {str(e)}")
        return pd.DataFrame(columns=['Term', 'Lead Count', 'Email Count'])
    except Exception as e:
        logging.error(f"Unexpected error in fetch_search_terms_with_lead_count: {str(e)}")
        return pd.DataFrame(columns=['Term', 'Lead Count', 'Email Count'])

def add_search_term(session, term, campaign_id):
    """Add a new search term with proper error handling"""
    try:
        new_term = SearchTerm(
            term=term,
            campaign_id=campaign_id,
            created_at=datetime.utcnow()
        )
        session.add(new_term)
        session.commit()
        return new_term.id
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error in add_search_term: {str(e)}")
        raise
    except Exception as e:
        session.rollback() 
        logging.error(f"Unexpected error in add_search_term: {str(e)}")
        raise

def update_search_term_group(session, group_id, updated_terms):
    """Update search term group assignments with proper error handling"""
    try:
        # Get current term IDs from updated terms list
        current_term_ids = {int(term.split(":")[0]) for term in updated_terms}
        
        # Update existing terms
        existing_terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all()
        for term in existing_terms:
            term.group_id = group_id if term.id in current_term_ids else None
            
        # Update new terms
        for term_str in updated_terms:
            term_id = int(term_str.split(":")[0])
            term = session.query(SearchTerm).get(term_id)
            if term:
                term.group_id = group_id
                
        session.commit()
        
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error in update_search_term_group: {str(e)}")
    except Exception as e:
        session.rollback()
        logging.error(f"Unexpected error in update_search_term_group: {str(e)}")

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    """Add new search term with group assignment"""
    try:
        group_id = None
        if group_for_new_term != "None":
            try:
                group_id = int(group_for_new_term.split(":")[0])
            except (ValueError, IndexError):
                logging.error(f"Invalid group format: {group_for_new_term}")
        
        new_search_term = SearchTerm(
            term=new_term,
            campaign_id=campaign_id,
            created_at=datetime.utcnow(),
            group_id=group_id
        )
        session.add(new_search_term)
        session.commit()
        
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error in add_new_search_term: {str(e)}")
    except Exception as e:
        session.rollback()
        logging.error(f"Unexpected error in add_new_search_term: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
    """Use AI to group search terms with proper error handling"""
    try:
        existing_groups = session.query(SearchTermGroup).all()
        prompt = (
            f"Categorize these search terms into existing groups or suggest new ones:\n"
            f"{', '.join([term.term for term in ungrouped_terms])}\n\n"
            f"Existing groups: {', '.join([group.name for group in existing_groups])}\n\n"
            f"Respond with a JSON object: {{group_name: [term1, term2, ...]}}"
        )
        
        messages = [
            {
                "role": "system", 
                "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = openai_chat_completion(messages, function_name="ai_group_search_terms")
        return response if isinstance(response, dict) else {}
        
    except Exception as e:
        logging.error(f"Error in ai_group_search_terms: {str(e)}")
        return {}

def update_search_term_groups(session, grouped_terms):
    """Update search term groups with proper error handling"""
    try:
        for group_name, terms in grouped_terms.items():
            # Get or create group
            group = (session.query(SearchTermGroup)
                    .filter_by(name=group_name)
                    .first() or SearchTermGroup(name=group_name))
            
            if not group.id:
                session.add(group)
                session.flush()
            
            # Update terms
            for term in terms:
                search_term = session.query(SearchTerm).filter_by(term=term).first()
                if search_term:
                    search_term.group_id = group.id
                    
        session.commit()
        
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error in update_search_term_groups: {str(e)}")
    except Exception as e:
        session.rollback()
        logging.error(f"Unexpected error in update_search_term_groups: {str(e)}")

def create_search_term_group(session, group_name):
    """Create new search term group with proper error handling"""
    try:
        group = SearchTermGroup(name=group_name)
        session.add(group)
        session.commit()
        return group.id
        
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error in create_search_term_group: {str(e)}")
    except Exception as e:
        session.rollback()
        logging.error(f"Unexpected error in create_search_term_group: {str(e)}")

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def get_page_description(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        return meta_desc['content'].strip() if meta_desc else "No description found"
    except Exception as e:
        logging.error(f"Error extracting page description: {str(e)}")
        return "Error extracting description"

def is_valid_email(email):
    if email is None: return False
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^(email|info|contact|support|hello|hola|hi|salutations|greetings|inquiries|questions)@.*",
        r"^email@email\.com$",
        r".*@example\.com$",
        r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"
    ]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    if any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns): return False
    if any(email.lower().endswith(f"@{domain}") for domain in typo_domains): return False
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

def bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_bar=None, status_text=None, results=None, log_container=None):
    """Send bulk emails with proper error handling and rate limiting"""
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        logging.error(f"Email template with ID {template_id} not found.")
        return [], 0

    logs, sent_count = [], 0
    total_leads = len(leads)

    # Get email settings for quota check
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings:
        return [], 0

    for index, lead in enumerate(leads):
        try:
            # Check quota before sending
            if email_settings.quota and email_settings.quota.emails_sent_today >= email_settings.daily_limit:
                logs.append(f"❌ Daily quota exceeded for {from_email}")
                break

            validate_email(lead['Email'])
            response, tracking_id = send_email_ses(session, from_email, lead['Email'], template.subject, template.body_content, reply_to=reply_to)
            
            status = 'sent' if response else 'failed'
            message_id = response.get('MessageId', f"{'sent' if response else 'failed'}-{uuid.uuid4()}")
            if response: 
                sent_count += 1
            
            save_email_campaign(session, lead['Email'], template_id, status, datetime.utcnow(), template.subject, message_id, template.body_content)
            log_message = f"{'✅' if response else '❌'} {'Sent email to' if response else 'Failed to send email to'}: {lead['Email']}"
            logs.append(log_message)

            # Update progress indicators
            if progress_bar: 
                progress_bar.progress((index + 1) / total_leads)
            if status_text: 
                status_text.text(f"Processed {index + 1}/{total_leads} leads")
            if results is not None: 
                results.append({"Email": lead['Email'], "Status": status})
            if log_container: 
                log_container.text(log_message)

            # Add small delay to prevent rate limiting
            time.sleep(0.1)

        except EmailNotValidError:
            logs.append(f"❌ Invalid email address: {lead['Email']}")
        except Exception as e:
            error_message = f"Error sending email to {lead['Email']}: {str(e)}"
            logging.error(error_message)
            save_email_campaign(session, lead['Email'], template_id, 'failed', datetime.utcnow(), template.subject, f"error-{uuid.uuid4()}", template.body_content)
            logs.append(f"❌ Error sending email to: {lead['Email']} (Error: {str(e)})")

    return logs, sent_count

def view_campaign_logs():
    st.header("Email Logs")
    with db_session() as session:
        logs = fetch_all_email_logs(session)
        if logs.empty:
            st.info("No email logs found.")
        else:
            st.write(f"Total emails sent: {len(logs)}")
            st.write(f"Success rate: {(logs['Status'] == 'sent').mean():.2%}")

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=logs['Sent At'].min().date())
            with col2:
                end_date = st.date_input("End Date", value=logs['Sent At'].max().date())

            filtered_logs = logs[(logs['Sent At'].dt.date >= start_date) & (logs['Sent At'].dt.date <= end_date)]

            search_term = st.text_input("Search by email or subject")
            if search_term:
                filtered_logs = filtered_logs[filtered_logs['Email'].str.contains(search_term, case=False) | 
                                              filtered_logs['Subject'].str.contains(search_term, case=False)]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Emails Sent", len(filtered_logs))
            with col2:
                st.metric("Unique Recipients", filtered_logs['Email'].nunique())
            with col3:
                st.metric("Success Rate", f"{(filtered_logs['Status'] == 'sent').mean():.2%}")

            daily_counts = filtered_logs.resample('D', on='Sent At')['Email'].count()
            st.bar_chart(daily_counts)

            st.subheader("Detailed Email Logs")
            for _, log in filtered_logs.iterrows():
                with st.expander(f"{log['Sent At'].strftime('%Y-%m-%d %H:%M:%S')} - {log['Email']} - {log['Status']}"):
                    st.write(f"**Subject:** {log['Subject']}")
                    st.write(f"**Content Preview:** {log['Content'][:100]}...")
                    if st.button("View Full Email", key=f"view_email_{log['ID']}"):
                        st.components.v1.html(wrap_email_body(log['Content']), height=400, scrolling=True)
                    if log['Status'] != 'sent':
                        st.error(f"Status: {log['Status']}")

            logs_per_page = 20
            total_pages = (len(filtered_logs) - 1) // logs_per_page + 1
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            start_idx = (page - 1) * logs_per_page
            end_idx = start_idx + logs_per_page

            st.table(filtered_logs.iloc[start_idx:end_idx][['Sent At', 'Email', 'Subject', 'Status']])

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
        query = (
            session.query(
                Lead, 
                func.string_agg(LeadSource.url, ', ').label('sources'),
                func.max(EmailCampaign.sent_at).label('last_contact'),
                func.string_agg(EmailCampaign.status, ', ').label('email_statuses')
            )
            .outerjoin(LeadSource)
            .outerjoin(EmailCampaign)
            .group_by(Lead.id)
        )  # Close query definition
        return pd.DataFrame(
            [{
                **{k: getattr(lead, k) for k in ['id', 'email', 'first_name', 'last_name', 'company', 'job_title', 'created_at']},
                'Source': sources,
                'Last Contact': last_contact,
                'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted',
                'Delete': False
            } for lead, sources, last_contact, email_statuses in query.all()]
        )  # Close DataFrame
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
    query = (
        session.query(
            SearchTerm.term,
            func.count(distinct(Lead.id)).label('lead_count'),
            func.count(distinct(EmailCampaign.id)).label('email_count')
        )
        .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
        .join(Lead, LeadSource.lead_id == Lead.id)
        .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
        .group_by(SearchTerm.term)
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

def add_search_term(session, term, campaign_id):
    """Add a new search term to the database"""
    try:
        new_term = SearchTerm(
            term=term,
            campaign_id=campaign_id,
            created_at=datetime.utcnow()
        )
        session.add(new_term)
        session.commit()
        return new_term.id
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")
        raise

def get_active_campaign_id():
    """Get the active campaign ID from the session state"""
    return st.session_state.get('active_campaign_id', 1)  # Default to 1 if not set

def set_active_campaign_id(campaign_id):
    """Set the active campaign ID in the session state"""
    st.session_state['active_campaign_id'] = campaign_id

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
                col1, col2, col3 = st.columns([2, 1, 1])
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
        new_search_term = SearchTerm(
            term=new_term,
            campaign_id=campaign_id,
            created_at=datetime.utcnow(),
            group_id=int(group_for_new_term.split(":")[0]) if group_for_new_term != "None" else None
        )  # Corrected syntax
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
        cleanup_stale_email_queue(session)  # Add cleanup call
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
    """Fetch email settings with proper error handling"""
    try:
        settings = session.query(EmailSettings).all()
        if not settings:
            logging.warning("No email settings found")
            return []
        
        return [{
            "id": setting.id,
            "name": setting.name,
            "email": setting.email,
            "provider": setting.provider,
            "daily_limit": setting.daily_limit
        } for setting in settings]
    except SQLAlchemyError as e:
        logging.error(f"Database error in fetch_email_settings: {str(e)}")
        return []

def fetch_search_terms_with_lead_count(session):
    query = (
        session.query(
            SearchTerm.term,
            func.count(distinct(Lead.id)).label('lead_count'),
            func.count(distinct(EmailCampaign.id)).label('email_count')
        )
        .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
        .join(Lead, LeadSource.lead_id == Lead.id)
        .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
        .group_by(SearchTerm.term)
    )
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

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
                    # Add lead saving logic directly here
                    lead = save_lead_with_source(
                        session, 
                        res['Email'], 
                        res['URL'],
                        res.get('Name', ''),
                        res.get('Company', ''),
                        res.get('Job Title', ''),
                        search_term_id,
                        res.get('html_content', ''),
                        res.get('Domain', '')
                    )  # Fixed missing closing parenthesis
                    if lead:
                        new_leads.append((lead.id, lead.email))
                if new_leads:
                    template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                    if template:
                        from_email = kb_info.get('contact_email', 'hello@indosy.com')
                        reply_to = kb_info.get('contact_email', 'eugproductions@gmail.com')
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
    if not results:
        return st.warning("No results to display.")
    with st.expander("Search Results", expanded=True):
        st.markdown(f"### Total Leads Found: **{len(results)}**")
        for i, res in enumerate(results):
            with st.expander(f"Lead: {res['Email']}", key=f"lead_expander_{key_suffix}_{i}"):
                st.markdown(
                    f"**URL:** [{res['URL']}]({res['URL']})  \n"
                    f"**Title:** {res['Title']}  \n"
                    f"**Description:** {res['Description']}  \n"
                    f"**Tags:** {', '.join(res['Tags'])}  \n"
                    f"**Lead Source:** {res['Lead Source']}  \n"
                    f"**Lead Email:** {res['Email']}"
                )  # Close markdown()

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
    query = (
        session.query(
            SearchTerm.term,
            func.count(distinct(Lead.id)).label('lead_count'),
            func.count(distinct(EmailCampaign.id)).label('email_count')
        )
        .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
        .join(Lead, LeadSource.lead_id == Lead.id)
        .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
        .group_by(SearchTerm.term)
    )
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

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
        # Add project fetching logic directly here
        projects = session.query(Project).all()
        project_options = [f"{p.id}: {p.project_name}" for p in projects]
        if not project_options: 
            return st.warning("No projects found. Please create a project first.")
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

    with db_session() as session:
        cleanup_stale_email_queue(session)  # Add cleanup call
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

    if st.button("Perform Quick Scan", use_container_width=True):
        with st.spinner("Performing quick scan..."):
            try:
                with db_session() as session:
                    new_leads = session.query(Lead).filter(Lead.is_processed == False).count()
                    session.query(Lead).filter(Lead.is_processed == False).update({Lead.is_processed: True})
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

def wrap_email_body(content: str) -> str:
    """Wrap email content in proper HTML structure"""
    return f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        {content}
    </div>
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
    st.set_page_config(page_title="AutoclientAI", layout="wide")
    
    # Initialize session state
    if 'active_project_id' not in st.session_state:
        st.session_state.active_project_id = 1
    if 'active_campaign_id' not in st.session_state:
        st.session_state.active_campaign_id = 1
    if 'automation_status' not in st.session_state:
        st.session_state.automation_status = False
    
    # Navigation
    with st.sidebar:
        selected = option_menu(
            "AutoclientAI",
            ["Projects & Campaigns", "Lead Management", "Search Terms", 
             "Email Templates", "Bulk Email", "Email Logs", "Settings",
             "Knowledge Base", "Quick Scan", "AI Automation", "Analytics",
             "Search Results", "Lead Sources"],
            icons=['folder', 'person', 'search', 'envelope', 'send', 
                  'clock-history', 'gear', 'book', 'lightning', 'robot',
                  'graph-up', 'list', 'database'],
            menu_icon="app-indicator"
        )
    
    # Page routing
    try:
        if selected == "Projects & Campaigns":
            projects_campaigns_page()
        elif selected == "Lead Management":
            view_leads_page()
        elif selected == "Search Terms":
            search_terms_page()
        elif selected == "Email Templates":
            email_templates_page()
        elif selected == "Bulk Email":
            bulk_send_page()
        elif selected == "Email Logs":
            view_campaign_logs()
        elif selected == "Settings":
            settings_page()
        elif selected == "Knowledge Base":
            knowledge_base_page()
        elif selected == "Quick Scan":
            quick_scan_page()
        elif selected == "AI Automation":
            ai_automation_page()
        elif selected == "Analytics":
            analytics_page()
        elif selected == "Search Results":
            search_results_page()
        elif selected == "Lead Sources":
            lead_sources_page()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in main navigation: {str(e)}")

def quick_scan_page():
    st.title("Quick Scan")
    with db_session() as session:
        if st.button("Start Quick Scan"):
            results = perform_quick_scan(session)
            st.success(f"Found {results['new_leads']} new leads using terms: {', '.join(results['terms_used'])}")

def analytics_page():
    st.title("Analytics Dashboard")
    with db_session() as session:
        col1, col2, col3 = st.columns(3)
        
        # Lead Growth
        leads_df = pd.read_sql(
            "SELECT DATE(created_at) as date, COUNT(*) as count FROM leads GROUP BY DATE(created_at)",
            session.bind
        )
        if not leads_df.empty:
            leads_df['cumulative'] = leads_df['count'].cumsum()
            col1.metric("Total Leads", int(leads_df['count'].sum()))
            st.line_chart(leads_df.set_index('date')['cumulative'])
        
        # Email Performance
        email_stats = pd.read_sql("""
            SELECT status, COUNT(*) as count 
            FROM email_campaigns 
            GROUP BY status
        """, session.bind)
        if not email_stats.empty:
            col2.metric("Emails Sent", int(email_stats['count'].sum()))
            st.bar_chart(email_stats.set_index('status'))
        
        # Search Term Performance
        term_stats = pd.read_sql("""
            SELECT st.term, COUNT(DISTINCT l.id) as lead_count
            FROM search_terms st
            JOIN lead_sources ls ON st.id = ls.search_term_id
            JOIN leads l ON ls.lead_id = l.id
            GROUP BY st.term
            ORDER BY lead_count DESC
            LIMIT 10
        """, session.bind)
        if not term_stats.empty:
            col3.metric("Active Search Terms", len(term_stats))
            st.bar_chart(term_stats.set_index('term'))

def search_results_page():
    st.title("Search Results")
    with db_session() as session:
        search_terms = session.query(SearchTerm).all()
        selected_term = st.selectbox("Select Search Term", 
                                   options=[term.term for term in search_terms])
        if selected_term:
            results = manual_search(session, [selected_term], 10, True)
            display_search_results(results['results'], selected_term)

def lead_sources_page():
    st.title("Lead Sources")
    with db_session() as session:
        sources = pd.read_sql("""
            SELECT 
                ls.domain,
                COUNT(DISTINCT l.id) as lead_count,
                STRING_AGG(DISTINCT st.term, ', ') as search_terms
            FROM lead_sources ls
            JOIN leads l ON ls.lead_id = l.id
            JOIN search_terms st ON ls.search_term_id = st.id
            GROUP BY ls.domain
            ORDER BY lead_count DESC
        """, session.bind)
        
        st.dataframe(sources)
        
        # Domain Analysis
        st.subheader("Domain Analysis")
        domain_stats = sources['domain'].str.split('.').str[-1].value_counts()
        st.bar_chart(domain_stats)

def knowledge_base_page():
    st.title("Knowledge Base")
    with db_session() as session:
        kb = session.query(KnowledgeBase).filter_by(project_id=get_active_project_id()).first()
        
        with st.form("knowledge_base_form"):
            kb_data = {
                'kb_name': st.text_input("Knowledge Base Name", value=kb.kb_name if kb else ""),
                'kb_bio': st.text_area("Bio", value=kb.kb_bio if kb else ""),
                'kb_values': st.text_area("Values", value=kb.kb_values if kb else ""),
                'contact_name': st.text_input("Contact Name", value=kb.contact_name if kb else ""),
                'contact_role': st.text_input("Contact Role", value=kb.contact_role if kb else ""),
                'contact_email': st.text_input("Contact Email", value=kb.contact_email if kb else ""),
                'company_description': st.text_area("Company Description", value=kb.company_description if kb else ""),
                'company_mission': st.text_area("Company Mission", value=kb.company_mission if kb else ""),
                'company_target_market': st.text_area("Target Market", value=kb.company_target_market if kb else ""),
                'product_name': st.text_input("Product Name", value=kb.product_name if kb else ""),
                'product_description': st.text_area("Product Description", value=kb.product_description if kb else ""),
                'product_target_customer': st.text_area("Target Customer", value=kb.product_target_customer if kb else ""),
                'example_email': st.text_area("Example Email", value=kb.example_email if kb else "")
            }
            
            if st.form_submit_button("Save Knowledge Base"):
                try:
                    if kb:
                        for key, value in kb_data.items():
                            setattr(kb, key, value)
                    else:
                        kb = KnowledgeBase(project_id=get_active_project_id(), **kb_data)
                        session.add(kb)
                    session.commit()
                    st.success("Knowledge base updated successfully!")
                except Exception as e:
                    st.error(f"Error saving knowledge base: {str(e)}")

def ai_automation_page():
    st.title("AI Automation")
    
    automation_status = st.session_state.get('automation_status', False)
    if st.button("Start Automation" if not automation_status else "Stop Automation"):
        st.session_state.automation_status = not automation_status
    
    st.info("Current Status: " + ("Running" if automation_status else "Stopped"))
    
    if automation_status:
        log_container = st.empty()
        leads_container = st.empty()
        with db_session() as session:
            ai_automation_loop(session, log_container, leads_container)

def openai_chat_completion(messages: List[Dict[str, str]], function_name: str) -> Any:
    """Make OpenAI chat completion call with settings from database"""
    try:
        with db_session() as session:
            openai_settings = get_settings_from_db(session, 'openai')
            if not openai_settings:
                logging.error("OpenAI settings not found in database")
                return None

            client = OpenAI(api_key=openai_settings['api_key'])
            response = client.chat.completions.create(
                model=openai_settings['model'],
                messages=messages,
                temperature=openai_settings.get('temperature', 0.7),
                max_tokens=1500
            )
            return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in {function_name}: {str(e)}")
        return None

if __name__ == "__main__":
    main()