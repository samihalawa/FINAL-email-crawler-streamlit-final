import os, json, re, logging, asyncio, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid, aiohttp, urllib3, random, html, smtplib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from openai import OpenAI
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, Float, Index, select, text, distinct, and_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, joinedload
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from typing import List, Optional
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
import plotly.express as px
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import functools
from sqlalchemy.dialects.postgresql import insert
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from streamlit.runtime.scriptrunner import RerunData, RerunException
from streamlit.source_util import get_pages
from sqlalchemy.pool import QueuePool

# 1. Load environment variables and set up database connection
load_dotenv()
DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT = map(os.getenv, ["SUPABASE_DB_HOST", "SUPABASE_DB_NAME", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD", "SUPABASE_DB_PORT"])
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

engine = create_engine(DATABASE_URL, poolclass=QueuePool, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# 2. Define database models (Input: None, Output: SQLAlchemy models)
class BackgroundProcessState(Base):
    __tablename__ = 'background_process_state'
    id = Column(BigInteger, primary_key=True)
    is_running = Column(Boolean, default=False)
    last_run = Column(DateTime(timezone=True))
    current_term = Column(Text)
    leads_found = Column(BigInteger, default=0)
    emails_sent = Column(BigInteger, default=0)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    job_type = Column(Text)
    job_params = Column(JSON)
    job_progress = Column(Float, default=0)
    job_id = Column(Text)
    campaign = relationship("Campaign", back_populates="background_state")

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
    background_state = relationship("BackgroundProcessState", back_populates="campaign", uselist=False)

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
    other_context, example_email = [Column(Text) for _ in range(2)]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    project = relationship("Project", back_populates="knowledge_base")

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']}

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
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    template_id = Column(BigInteger, ForeignKey('email_templates.id'))
    customized_subject, customized_content, original_subject, original_content, status = [Column(Text) for _ in range(5)]
    engagement_data = Column(JSON)
    message_id, tracking_id = [Column(Text) for _ in range(2)]
    sent_at, opened_at, clicked_at = [Column(DateTime(timezone=True)) for _ in range(3)]
    ai_customized = Column(Boolean, default=False)
    open_count, click_count = [Column(BigInteger, default=0) for _ in range(2)]
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
    url, domain, page_title, meta_description, scrape_duration, meta_tags, phone_numbers, content, tags = [Column(Text) for _ in range(9)]
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
    leads_gathered, emails_sent = [Column(BigInteger) for _ in range(2)]
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    status = Column(Text)
    logs = Column(JSON)
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
    name, email, provider, smtp_server, smtp_username, smtp_password, aws_access_key_id, aws_secret_access_key, aws_region = [Column(Text) for _ in range(9)]
    smtp_port = Column(BigInteger)

# 3. Context manager for database sessions (Input: None, Output: Session)
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

import openai
import streamlit as st
import threading
import time
from sqlalchemy.orm import sessionmaker
from models import Base, engine, KnowledgeBase, SearchTerm, SearchTermGroup, EmailSettings, Lead, EmailCampaign
from sqlalchemy import func
import json
import logging
import functools

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'automation_status' not in st.session_state:
    st.session_state.automation_status = False
if 'optimized_terms' not in st.session_state:
    st.session_state.optimized_terms = []

# Create database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Get OpenAI client (Input: None, Output: OpenAI client)
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 5. OpenAI chat completion with retry (Input: messages, function_name, model; Output: JSON response)
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
def openai_chat_completion(messages, function_name, model="gpt-4"):
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=[{"name": function_name, "parameters": {"type": "object"}}],
            function_call={"name": function_name}
        )
        return json.loads(response.choices[0].function_call.arguments)
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {str(e)}")
        return None

# 6. Fetch search terms with lead count (Input: session; Output: Query result)
@st.cache_data(ttl=3600)
def fetch_search_terms_with_lead_count(session):
    query = session.query(
        SearchTerm.term,
        func.count(func.distinct(Lead.id)).label('lead_count'),
        func.count(func.distinct(EmailCampaign.id)).label('email_count')
    ).join(Lead, SearchTerm.leads).outerjoin(EmailCampaign, Lead.email_campaigns
    ).group_by(SearchTerm.term)

    return query.all()

# 7. Update search term group (Input: session, group_id, updated_terms; Output: None)
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

# 8. Add new search term (Input: session, new_term, campaign_id, group_for_new_term; Output: None)
def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        new_search_term = SearchTerm(term=new_term, campaign_id=campaign_id, created_at=func.now())
        if group_for_new_term != "None":
            new_search_term.group_id = int(group_for_new_term.split(":")[0])
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")

# 9. AI group search terms (Input: session, ungrouped_terms; Output: Grouped terms)
def ai_group_search_terms(session, ungrouped_terms):
    existing_groups = session.query(SearchTermGroup).all()
    prompt = f"""
    Categorize these search terms into existing groups or suggest new ones:
    {', '.join([term.term for term in ungrouped_terms])}

    Existing groups: {', '.join([group.name for group in existing_groups])}

    Respond with a JSON object where keys are group names and values are arrays of terms.
    Schema: {{"group_name1": ["term1", "term2"], "group_name2": ["term3", "term4"]}}
    """
    messages = [
        {"role": "system", "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."},
        {"role": "user", "content": prompt}
    ]
    return openai_chat_completion(messages, "ai_group_search_terms")

# 10. Update search term groups (Input: session, grouped_terms; Output: None)
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

# 11. Create search term group (Input: session, group_name; Output: None)
def create_search_term_group(session, group_name):
    try:
        new_group = SearchTermGroup(name=group_name)
        session.add(new_group)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating search term group: {str(e)}")

# 12. Delete search term group (Input: session, group_id; Output: None)
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

# 13. Get knowledge base info (Input: session, project_id; Output: Knowledge base dict)
@functools.lru_cache(maxsize=10)
def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

# 14. Generate optimized search terms: Input: session, base_terms, kb_info; Output: List of optimized terms; Uses OpenAI API
def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"""
    Generate 5 optimized search terms based on the following base terms and context:
    Base terms: {', '.join(base_terms)}
    Context: {json.dumps(kb_info)}

    Respond with a JSON object containing an array of optimized search terms.
    Schema: {{"optimized_terms": ["term1", "term2", "term3", "term4", "term5"]}}
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai_chat_completion(messages, "generate_optimized_search_terms")
    return response.get("optimized_terms", []) if response else []

# 15. Bulk send email page: Input: None; Output: Streamlit UI; Manages email sending process
def bulk_send_page():
    st.title("Bulk Email Sending")
    with db_session() as session:
        templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)
        if not templates or not email_settings:
            return st.error("No email templates or settings available. Please set them up first.")
        template_option = st.selectbox("Email Template", options=templates, format_func=lambda x: x.split(":")[1].strip())
        template_id = int(template_option.split(":")[0])
        template = session.query(EmailTemplate).filter_by(id=template_id).first()
        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input("Subject", value=template.subject if template else "")
            email_setting_option = st.selectbox("From Email", options=email_settings, format_func=lambda x: f"{x['name']} ({x['email']})")
            if email_setting_option:
                from_email = email_setting_option['email']
                reply_to = st.text_input("Reply To", email_setting_option['email'])
            else:
                return st.error("Selected email setting not found. Please choose a valid email setting.")
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
        st.info(f"Total leads: {total_leads}\nLeads matching template language ({template.language}): {len(eligible_leads)}\nLeads to be contacted: {len(contactable_leads)}")
        user_settings = get_user_settings()
        enable_email_sending = user_settings['enable_email_sending']
        if st.button("Send Emails", type="primary"):
            if not contactable_leads:
                return st.warning("No leads found matching the selected criteria.")
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

# 16. Fetch search term groups: Input: session; Output: List[str]; Retrieves all search term groups
def fetch_search_term_groups(session):
    return [f"{group.id}: {group.name}" for group in session.query(SearchTermGroup).all()]

# 17. Fetch search terms for groups: Input: session, List[int]; Output: List[str]; Retrieves terms for specified groups
def fetch_search_terms_for_groups(session, group_ids):
    return [term.term for term in session.query(SearchTerm).filter(SearchTerm.group_id.in_(group_ids)).all()]

# 18. Automation control panel page: Input: None; Output: Streamlit UI; Manages automation process
def automation_control_panel_page():
    st.title("Automation Control Panel")
    col1, col2 = st.columns([2, 1])
    with col1:
        status = "Active" if scheduler.get_job('automation_job') else "Inactive"
        st.metric("Automation Status", status)
    with col2:
        button_text = "Stop Automation" if scheduler.get_job('automation_job') else "Start Automation"
        if st.button(button_text, use_container_width=True):
            if scheduler.get_job('automation_job'):
                scheduler.remove_job('automation_job')
                update_automation_status(False)
                st.success("Automation stopped.")
            else:
                scheduler.add_job(automation_job, IntervalTrigger(minutes=60), id='automation_job', replace_existing=True)
                update_automation_status(True)
                st.success("Automation started.")
    
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
    
    # Display Real-Time Analytics
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
    
    # Display Automation Logs
    st.subheader("Automation Logs")
    log_container = st.empty()
    automation_logs = get_automation_logs(session)
    display_results_and_logs(log_container, automation_logs, "Latest Logs", "logs")
    
    # Display Recently Found Leads
    st.subheader("Recently Found Leads")
    leads_container = st.empty()
    recent_leads = fetch_recent_leads(session)
    if recent_leads:
        leads_container.dataframe(pd.DataFrame(recent_leads, columns=["Email", "Company", "Created At"]))
    else:
        leads_container.info("No new leads found.")

# 19. Update search term group: Input: session, int, List[str]; Output: None; Updates terms in a group
def update_search_term_group(session, group_id, updated_terms):
    try:
        current_term_ids = set(int(term.split(":")[0]) for term in updated_terms)
        existing_terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all()
        
        # Ungroup terms not present in updated_terms
        for term in existing_terms:
            if term.id not in current_term_ids:
                term.group_id = None
        
        # Group terms based on updated_terms
        for term_str in updated_terms:
            term_id = int(term_str.split(":")[0])
            term = session.query(SearchTerm).get(term_id)
            if term:
                term.group_id = group_id
        
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_search_term_group: {str(e)}")

# 20. Add new search term: Input: session, str, int, str; Output: None; Adds a new search term to the database
def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        new_search_term = SearchTerm(term=new_term, campaign_id=campaign_id, created_at=func.now())
        if group_for_new_term != "None":
            new_search_term.group_id = int(group_for_new_term.split(":")[0])
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding search term: {str(e)}")

# 21. AI group search terms: Input: session, List[SearchTerm]; Output: Dict[str, List[str]]; Groups search terms using AI
def ai_group_search_terms(session, ungrouped_terms):
    existing_groups = session.query(SearchTermGroup).all()
    prompt = f"""
    Categorize these search terms into existing groups or suggest new ones:
    {', '.join([term.term for term in ungrouped_terms])}
    Existing groups: {', '.join([group.name for group in existing_groups])}
    Respond with a JSON object where keys are group names and values are arrays of terms.
    Schema: {{"group_name1": ["term1", "term2"], "group_name2": ["term3", "term4"]}}
    """
    messages = [
        {"role": "system", "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."},
        {"role": "user", "content": prompt}
    ]
    return openai_chat_completion(messages, "ai_group_search_terms")

# 22. Update search term groups: Input: session, Dict[str, List[str]]; Output: None; Updates database with grouped terms
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

# 23. Create search term group: Input: session, str; Output: None; Creates a new search term group
def create_search_term_group(session, group_name):
    try:
        new_group = SearchTermGroup(name=group_name)
        session.add(new_group)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating search term group: {str(e)}")

# 24. Delete search term group: Input: session, int; Output: None; Deletes a search term group and ungroups its terms
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

# 25. Get knowledge base info: Input: session, int; Output: Dict or None; Retrieves knowledge base info for a project
@functools.lru_cache(maxsize=10)
def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

# 26. Generate optimized search terms: Input: session, List[str], Dict; Output: List[str]; Generates optimized search terms
def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"""
    Generate 5 optimized search terms based on the following base terms and context:
    Base terms: {', '.join(base_terms)}
    Context: {json.dumps(kb_info)}
    Respond with a JSON object containing an array of optimized search terms.
    Schema: {{"optimized_terms": ["term1", "term2", "term3", "term4", "term5"]}}
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai_chat_completion(messages, "generate_optimized_search_terms")
    return response.get("optimized_terms", []) if response else []

# 27. Display results and logs: Input: st.container, List[str], str, str; Output: None; Displays results or logs in Streamlit
def display_results_and_logs(container, items, title, item_type):
    if not items:
        container.info(f"No {item_type} to display yet.")
        return
    container.markdown(
        f"""
        <style>
        .results-container {{
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
        }}
        .result-entry {{
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            border-radius: 0.25rem;
            background-color: rgba(28, 131, 225, 0.1);
        }}
        </style>
        <h4>{title}</h4>
        <div class="results-container">
        {"".join(f'<div class="result-entry">{item}</div>' for item in items[-20:])}
        </div>
        """,
        unsafe_allow_html=True
    )

# 28. Get search terms: Input: session; Output: List[str]; Retrieves search terms for the active project
def get_search_terms(session):
    return [term.term for term in session.query(SearchTerm).filter_by(project_id=st.session_state.get('active_project_id')).all()]

# 29. Get AI response: Input: str; Output: Dict; Gets AI response for a given prompt
def get_ai_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    return openai_chat_completion(messages, "get_ai_response")

# 30. Fetch email settings: Input: session; Output: List[Dict]; Retrieves email settings from the database
def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        return []

# 31. Auto refresh: Input: None; Output: None; Triggers a Streamlit rerun after a delay
def auto_refresh():
    time.sleep(300)  # 5 minutes
    raise RerunException(RerunData(None))

# 32. AutoclientAI page: Input: None; Output: None; Renders the AutoclientAI Streamlit page
def autoclient_ai_page():
    st.header("AutoclientAI - Automated Lead Generation")
    with st.expander("Knowledge Base Information", expanded=False):
        with SessionLocal() as session:
            kb_info = get_knowledge_base_info(session, st.session_state.get('active_project_id'))
        if not kb_info:
            return st.error("Knowledge Base not found for the active project. Please set it up first.")
        st.json(kb_info)
    user_input = st.text_area("Enter additional context or specific goals for lead generation:", help="This information will be used to generate more targeted search terms.")
    if st.button("Generate Optimized Search Terms", key="generate_optimized_terms"):
        with st.spinner("Generating optimized search terms..."):
            with SessionLocal() as session:
                base_terms = get_search_terms(session)
                optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            if optimized_terms:
                st.session_state.optimized_terms = optimized_terms
                st.success("Search terms optimized successfully!")
                st.subheader("Optimized Search Terms")
                st.write(", ".join(optimized_terms))
            else:
                st.error("Failed to generate optimized search terms. Please try again.")
    if st.button("Start Automation", key="start_automation"):
        st.session_state.automation_status = True
        st.session_state.automation_logs = []
        st.session_state.total_leads_found = 0
        st.session_state.total_emails_sent = 0
        threading.Thread(target=background_task, args=(st.session_state,), daemon=True).start()
        st.success("Automation started!")
    if st.button("Stop Automation", key="stop_automation"):
        st.session_state.automation_status = False
        st.success("Automation stopped!")
    st.subheader("Automation Status")
    st.write(f"Running: {'Yes' if st.session_state.automation_status else 'No'}")
    st.write(f"Total Leads Found: {st.session_state.total_leads_found}")
    st.write(f"Total Emails Sent: {st.session_state.total_emails_sent}")
    st.subheader("Automation Logs")
    display_results_and_logs(st.empty(), st.session_state.get('automation_logs', []), "Latest Logs", "logs")

# 33. Background task: Input: SessionState; Output: None; Runs the automation process in the background
def background_task(session_state):
    while session_state.automation_status:
        try:
            with SessionLocal() as session:
                optimized_terms = session_state.get('optimized_terms', [])
                if not optimized_terms:
                    optimized_terms = generate_optimized_search_terms(session, get_search_terms(session), get_knowledge_base_info(session, session_state.get('active_project_id')))
                for term in optimized_terms:
                    if not session_state.automation_status:
                        break
                    results = auto_perform_optimized_search(session, term, 10)
                    session_state.total_leads_found += len(results)
                    session_state.automation_logs.append(f"Found {len(results)} leads for term: {term}")
                    if session_state.user_settings.get('enable_email_sending', True):
                        for lead in results:
                            if not session_state.automation_status:
                                break
                            send_email_to_lead(session, lead)
                            session_state.total_emails_sent += 1
                            session_state.automation_logs.append(f"Sent email to: {lead.email}")
                    time.sleep(random.uniform(5, 15))  # Random delay between searches
        except Exception as e:
            session_state.automation_logs.append(f"Error in background task: {str(e)}")
        finally:
            time.sleep(60)  # Wait for 1 minute before the next iteration

# 34. Main function: Input: None; Output: None; Handles main navigation and page routing
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["AutoclientAI", "Search Terms", "Email Templates", "Settings"])
    if page == "AutoclientAI":
        autoclient_ai_page()
    elif page == "Search Terms":
        search_terms_page()
    elif page == "Email Templates":
        email_templates_page()
    elif page == "Settings":
        settings_page()

if __name__ == "__main__":
    main()

# 35. Bulk send page: Input: None; Output: None; Renders bulk email sending UI
def bulk_send_page():
    with db_session() as session:
        st.title("Bulk Email Sending")
        email_templates = fetch_email_templates(session)
        email_template = st.selectbox("Select Email Template", options=email_templates, format_func=lambda x: x.split(":")[1].strip())
        if not email_template:
            return st.error("Please select an email template.")
        template_id = int(email_template.split(":")[0])
        send_option = st.radio("Send to:", ["All Leads", "Specific Email", "Leads from Chosen Search Terms"], index=0)
        specific_email = st.text_input("Enter Email Address") if send_option == "Specific Email" else None
        selected_terms = st.multiselect("Select Search Terms", options=fetch_search_terms(session)) if send_option == "Leads from Chosen Search Terms" else None
        email_settings = fetch_email_settings(session)
        if not email_settings:
            return st.error("No email settings available. Please configure them first.")
        from_email_option = st.selectbox("From Email", options=email_settings, format_func=lambda x: f"{x['name']} ({x['email']})")
        if not from_email_option:
            return st.error("Please select a valid 'From Email'.")
        from_email = from_email_option['email']
        reply_to = st.text_input("Reply-To", value=from_email)
        if st.button("Send Emails", type="primary"):
            leads = fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted=True)
            if not leads:
                return st.warning("No leads found matching the selected criteria.")
            progress_bar, status_text, log_container = st.progress(0), st.empty(), st.empty()
            results = []
            try:
                logs, sent_count = bulk_send_emails(session, template_id, from_email, reply_to, leads, progress_bar=progress_bar, status_text=status_text, results=results, log_container=log_container)
                st.success(f"Emails sent successfully to {sent_count} leads.")
                st.subheader("Sending Results")
                st.dataframe(pd.DataFrame(results))
                success_rate = (pd.DataFrame(results)['Status'] == 'sent').mean() * 100
                st.metric("Email Sending Success Rate", f"{success_rate:.2f}%")
            except Exception as e:
                st.error(f"An error occurred during bulk email sending: {str(e)}")

# 36. Bulk send emails: Input: session, int, str, str, List[Dict], **kwargs; Output: Dict; Sends bulk emails and returns results
def bulk_send_emails(session, template_id, from_email, reply_to, leads, **kwargs):
    progress_bar, status_text, log_container, results = kwargs.get('progress_bar'), kwargs.get('status_text'), kwargs.get('log_container'), kwargs.get('results', [])
    sent_count, total = 0, len(leads)
    template = session.query(EmailTemplate).get(template_id)
    if not template:
        raise ValueError("Email template not found")
    for idx, lead in enumerate(leads):
        try:
            response, tracking_id = send_email_ses(session, from_email, lead['Email'], template.subject, template.body_content, reply_to=reply_to)
            status = 'sent' if response else 'failed'
            results.append({'Email': lead['Email'], 'Status': status})
            sent_count += status == 'sent'
            if log_container:
                log_container.text(f"Sent to {lead['Email']}: {status}")
            if status == 'sent':
                save_email_campaign(session, lead['Email'], template_id, 'Sent', datetime.utcnow(), template.subject, response['MessageId'], template.body_content)
        except Exception as e:
            results.append({'Email': lead['Email'], 'Status': 'error'})
            if log_container:
                log_container.error(f"Error sending to {lead['Email']}: {str(e)}")
        if progress_bar:
            progress_bar.progress((idx + 1) / total)
        if status_text:
            status_text.text(f"Sending emails: {int((idx + 1) / total * 100)}% completed.")
    return {"logs": results, "sent_count": sent_count}

# 37. View campaign logs: Input: None; Output: None; Displays email campaign logs
def view_campaign_logs():
    with db_session() as session:
        st.header("Email Logs")
        logs = fetch_all_email_logs(session)
        if logs.empty:
            return st.info("No email logs found.")
        st.write(f"Total emails sent: {len(logs)}")
        st.write(f"Success rate: {(logs['Status'] == 'sent').mean():.2%}")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", value=logs['Sent At'].min().date())
        end_date = col2.date_input("End Date", value=logs['Sent At'].max().date())
        filtered_logs = logs[(logs['Sent At'].dt.date >= start_date) & (logs['Sent At'].dt.date <= end_date)]
        search_term = st.text_input("Search by email or subject")
        if search_term:
            filtered_logs = filtered_logs[filtered_logs['Email'].str.contains(search_term, case=False, na=False) | filtered_logs['Subject'].str.contains(search_term, case=False, na=False)]
        col1, col2, col3 = st.columns(3)
        col1.metric("Emails Sent", len(filtered_logs))
        col2.metric("Unique Recipients", filtered_logs['Email'].nunique())
        col3.metric("Success Rate", f"{(filtered_logs['Status'] == 'sent').mean():.2%}")
        st.bar_chart(filtered_logs.resample('D', on='Sent At')['Email'].count())
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
        page = st.number_input("Page", min_value=1, max_value=(len(filtered_logs) - 1) // logs_per_page + 1, value=1)
        start_idx = (page - 1) * logs_per_page
        st.table(filtered_logs.iloc[start_idx:start_idx + logs_per_page][['Sent At', 'Email', 'Subject', 'Status']])

# 38. Fetch all email logs: Input: session; Output: DataFrame; Retrieves all email campaign logs
def fetch_all_email_logs(session):
    try:
        query = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc())
        data = [{
            'ID': ec.id, 'Sent At': ec.sent_at, 'Email': ec.lead.email, 'Template': ec.template.template_name,
            'Subject': ec.customized_subject or "No subject", 'Content': ec.customized_content or "No content",
            'Status': ec.status, 'Message ID': ec.message_id or "No message ID", 'Campaign ID': ec.campaign_id,
            'Lead Name': f"{ec.lead.first_name or ''} {ec.lead.last_name or ''}".strip() or "Unknown",
            'Lead Company': ec.lead.company or "Unknown"
        } for ec in query.all()]
        return pd.DataFrame(data)
    except SQLAlchemyError as e:
        st.error(f"Database error while fetching email logs: {str(e)}")
        return pd.DataFrame()

# 39. Wrap email body: Input: str; Output: str; Wraps email content in HTML tags
def wrap_email_body(content):
    return f"<html><body>{content}</body></html>"

# 40. Main function: Input: None; Output: None; Handles main navigation and page routing
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["AutoclientAI", "Search Terms", "Email Templates", "Settings"])
    if page == "AutoclientAI":
        autoclient_ai_page()
    elif page == "Search Terms":
        search_terms_page()
    elif page == "Email Templates":
        email_templates_page()
    elif page == "Settings":
        settings_page()

# 41. Save email campaign: Input: session, str, int, str, datetime, str, str, str; Output: None; Saves email campaign details
def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    try:
        lead = session.query(Lead).filter_by(email=lead_email).first()
        if not lead:
            return logging.error(f"Lead with email {lead_email} not found.")
        new_campaign = EmailCampaign(
            lead_id=lead.id, template_id=template_id, status=status, sent_at=sent_at,
            customized_subject=subject or "No subject", message_id=message_id or f"unknown-{uuid.uuid4()}",
            customized_content=email_body or "No content", campaign_id=get_active_campaign_id(),
            tracking_id=str(uuid.uuid4())
        )
        session.add(new_campaign)
        session.commit()
    except Exception as e:
        logging.error(f"Error saving email campaign: {str(e)}")
        session.rollback()

# 42. Update log: Input: st.container, str, str; Output: None; Updates and displays log messages
def update_log(log_container, message, level='info'):
    icon = {'info': '🔵', 'success': '🟢', 'warning': '🟠', 'error': '🔴', 'email_sent': '🟣'}.get(level, '⚪')
    log_entry = f"{icon} {message}"
    logging.log(getattr(logging, level.upper(), logging.INFO), message.split('<')[0])
    print(f"{icon} {message.split('<')[0]}")
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    st.session_state.log_entries.append(f"{icon} {message}")
    if log_container:
        log_html = f"<div style='height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.8em; line-height: 1.2;'>{'<br>'.join(st.session_state.log_entries)}</div>"
        log_container.markdown(log_html, unsafe_allow_html=True)

# 43. Optimize search term: Input: str, str; Output: str; Optimizes search term based on language
def optimize_search_term(search_term, language):
    return f'"{search_term}" email OR contact OR "get in touch" site:.com' if language == 'english' else f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es' if language == 'spanish' else search_term

# 44. Shuffle keywords: Input: str; Output: str; Randomly shuffles words in a search term
def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

# 45. Get domain from URL: Input: str; Output: str; Extracts domain from a given URL
def get_domain_from_url(url):
    return urlparse(url).netloc

# 46. Is valid email: Input: str; Output: bool; Checks if an email address is valid
def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

# 47. Extract emails from HTML: Input: str, str; Output: List[Dict]; Extracts email addresses and related info from HTML content
def extract_emails_from_html(html_content, domain):
    soup = BeautifulSoup(html_content, 'html.parser')
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, html_content)
    page_title = soup.title.string if soup.title else "No title found"
    meta_description = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_description['content'] if meta_description else "No description found"
    return [{
        'email': email,
        'name': extract_info_from_page(soup)[0],
        'company': extract_info_from_page(soup)[1],
        'job_title': extract_info_from_page(soup)[2],
        'page_title': page_title,
        'meta_description': meta_description,
        'tags': [tag.name for tag in soup.find_all()],
        'domain': domain
    } for email in emails]

# 48. Extract info from page: Input: BeautifulSoup; Output: Tuple[str, str, str]; Extracts name, company, and job title from a webpage
def extract_info_from_page(soup):
    return (
        soup.find('meta', {'name': 'author'})['content'] if soup.find('meta', {'name': 'author'}) else '',
        soup.find('meta', {'property': 'og:site_name'})['content'] if soup.find('meta', {'property': 'og:site_name'}) else '',
        soup.find('meta', {'name': 'job_title'})['content'] if soup.find('meta', {'name': 'job_title'}) else ''
    )

# 49. Optimized search: Input: str, int; Output: List[str]; Performs an optimized Google search
@functools.lru_cache(maxsize=100)
def optimized_search(term, num_results):
    return google_search(term, num_results)

# 50. Manual search: Input: session, List[str], int, **kwargs; Output: Dict; Performs manual search and processes results
def manual_search(session, terms, num_results, **kwargs):
    ua, results, total_leads, domains_processed, emails_processed = UserAgent(), [], 0, set(), set()
    for original_term in terms:
        try:
            search_term_id = add_or_get_search_term(session, original_term, get_active_campaign_id())
            search_term = shuffle_keywords(original_term) if kwargs.get('shuffle_keywords_option') else original_term
            search_term = optimize_search_term(search_term, 'english' if kwargs.get('optimize_english') else 'spanish' if kwargs.get('optimize_spanish') else None)
            update_log(kwargs.get('log_container'), f"Searching for '{original_term}' (Used '{search_term}')")
            for url in optimized_search(search_term, num_results):
                domain = get_domain_from_url(url)
                if kwargs.get('ignore_previously_fetched', True) and domain in domains_processed:
                    update_log(kwargs.get('log_container'), f"Skipping Previously Fetched: {domain}", 'warning')
                    continue
                update_log(kwargs.get('log_container'), f"Fetching: {url}")
                try:
                    url = f"http://{url}" if not url.startswith(('http://', 'https://')) else url
                    response = requests.get(url, timeout=10, verify=False, headers={'User-Agent': ua.random})
                    response.raise_for_status()
                    html_content, soup = response.text, BeautifulSoup(response.text, 'html.parser')
                    emails = extract_emails_from_html(html_content, domain)
                    update_log(kwargs.get('log_container'), f"Found {len(emails)} email(s) on {url}", 'success')
                    for email_info in emails:
                        email = email_info['email']
                        if (kwargs.get('one_email_per_url', True) and email in emails_processed) or (kwargs.get('one_email_per_domain', True) and email.split('@')[1] in domains_processed):
                            continue
                        emails_processed.add(email)
                        domains_processed.add(email.split('@')[1])
                        lead = save_lead(session, email=email, first_name=email_info['name'], company=email_info['company'], job_title=email_info['job_title'], url=domain, search_term_id=search_term_id, created_at=datetime.utcnow())
                        if lead:
                            total_leads += 1
                            results.append({
                                'Email': email, 'URL': domain, 'Lead Source': original_term,
                                'Title': email_info['page_title'], 'Description': email_info['meta_description'],
                                'Tags': email_info['tags'], 'Name': email_info['name'],
                                'Company': email_info['company'], 'Job Title': email_info['job_title'],
                                'Search Term ID': search_term_id
                            })
                            update_log(kwargs.get('log_container'), f"Saved lead: {email}", 'success')
                            if kwargs.get('enable_email_sending') and kwargs.get('from_email') and kwargs.get('email_template'):
                                template = session.query(EmailTemplate).filter_by(id=int(kwargs['email_template'].split(":")[0])).first()
                                if template:
                                    wrapped_content = wrap_email_body(template.body_content)
                                    try:
                                        response, tracking_id = rate_limited_send_email_ses(session, kwargs['from_email'], email, template.subject, wrapped_content, reply_to=kwargs.get('reply_to'))
                                        if response:
                                            update_log(kwargs.get('log_container'), f"Sent email to: {email}", 'email_sent')
                                            save_email_campaign(session, email, template.id, 'Sent', datetime.utcnow(), template.subject, response['MessageId'], wrapped_content)
                                        else:
                                            update_log(kwargs.get('log_container'), f"Failed to send email to: {email}", 'error')
                                            save_email_campaign(session, email, template.id, 'Failed', datetime.utcnow(), template.subject, None, wrapped_content)
                                    except Exception as e:
                                        update_log(kwargs.get('log_container'), f"Error sending email to {email}: {str(e)}", 'error')
                except requests.RequestException as e:
                    update_log(kwargs.get('log_container'), f"Error processing URL {url}: {str(e)}", 'error')
        except Exception as e:
            update_log(kwargs.get('log_container'), f"Error processing term '{original_term}': {str(e)}", 'error')
    update_log(kwargs.get('log_container'), f"Total leads found: {total_leads}", 'info')
    return {"total_leads": total_leads, "results": results}

# 51. Delete lead: Input: session, int; Output: bool; Deletes a lead and its associated sources
def delete_lead(session, lead_id):
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

# 52. Get knowledge base info: Input: session, int; Output: Dict or None; Retrieves knowledge base info for a project
@functools.lru_cache(maxsize=100)
def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

# 53. Background manual search: Input: session_factory; Output: None; Performs background search and email sending
def background_manual_search(session_factory):
    with db_session() as session:
        state = get_background_state()
        if not state['is_running']:
            return
        search_terms = session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()
        for term in search_terms:
            update_background_state(session, current_term=term.term)
            results = manual_search(session, [term.term], 10, ignore_previously_fetched=True)
            update_background_state(session, leads_found=state['leads_found'] + results['total_leads'])
            if results['total_leads'] > 0:
                template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                if template:
                    kb_info = get_knowledge_base_info(session, get_active_project_id())
                    from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                    reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                    logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': res['Email']} for res in results['results']])
                    update_background_state(session, emails_sent=state['emails_sent'] + sent_count)
        update_background_state(session, last_run=datetime.utcnow())

# 54. Start background process: Input: None; Output: None; Initiates the background search process
def start_background_process():
    with db_session() as session:
        update_background_state(session, is_running=True)

# 55. Pause background search: Input: None; Output: None; Pauses the background search process
def pause_background_search():
    with db_session() as session:
        update_background_state(session, is_running=False)

# 56. Resume background search: Input: None; Output: None; Resumes the background search process
def resume_background_search():
    with db_session() as session:
        update_background_state(session, is_running=True)

# 57. Stop background search: Input: None; Output: None; Stops the background search process and resets state
def stop_background_search():
    with db_session() as session:
        update_background_state(session, is_running=False, current_term=None, job_progress=0)

# 58. Settings page: Input: None; Output: None; Renders the settings page UI
def settings_page():
    st.title("Settings")
    with db_session() as session:
        email_settings = fetch_email_settings(session)
        st.subheader("Email Settings")
        for setting in email_settings:
            with st.expander(f"{setting['name']} ({setting['email']})"):
                name = st.text_input("Name", value=setting['name'], key=f"name_{setting['id']}")
                email = st.text_input("Email", value=setting['email'], key=f"email_{setting['id']}")
                provider = st.selectbox("Provider", ["SMTP", "AWS SES"], key=f"provider_{setting['id']}")
                if provider == "SMTP":
                    smtp_server = st.text_input("SMTP Server", key=f"smtp_server_{setting['id']}")
                    smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535, key=f"smtp_port_{setting['id']}")
                    smtp_username = st.text_input("SMTP Username", key=f"smtp_username_{setting['id']}")
                    smtp_password = st.text_input("SMTP Password", type="password", key=f"smtp_password_{setting['id']}")
                else:
                    aws_access_key_id = st.text_input("AWS Access Key ID", key=f"aws_access_key_id_{setting['id']}")
                    aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password", key=f"aws_secret_access_key_{setting['id']}")
                    aws_region = st.text_input("AWS Region", key=f"aws_region_{setting['id']}")
                if st.button("Update", key=f"update_{setting['id']}"):
                    update_email_setting(session, setting['id'], name, email, provider, smtp_server, smtp_port, smtp_username, smtp_password, aws_access_key_id, aws_secret_access_key, aws_region)
                    st.success("Email setting updated successfully!")
        if st.button("Add New Email Setting"):
            add_new_email_setting(session)
            st.success("New email setting added successfully!")
            st.rerun()

# 59. Update email setting: Input: session, int, str, str, str, Optional[str], Optional[int], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]; Output: None; Updates an email setting
def update_email_setting(session, setting_id, name, email, provider, smtp_server=None, smtp_port=None, smtp_username=None, smtp_password=None, aws_access_key_id=None, aws_secret_access_key=None, aws_region=None):
    setting = session.query(EmailSettings).filter_by(id=setting_id).first()
    if setting:
        setting.name, setting.email, setting.provider = name, email, provider
        setting.smtp_server, setting.smtp_port, setting.smtp_username, setting.smtp_password = smtp_server, smtp_port, smtp_username, smtp_password
        setting.aws_access_key_id, setting.aws_secret_access_key, setting.aws_region = aws_access_key_id, aws_secret_access_key, aws_region
        session.commit()

# 60. Add new email setting: Input: session; Output: None; Adds a new email setting to the database
def add_new_email_setting(session):
    new_setting = EmailSettings(name="New Email Setting", email="example@example.com", provider="SMTP")
    session.add(new_setting)
    session.commit()

# 61. Search terms page: Input: None; Output: None; Renders search terms management UI
def search_terms_page():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Search Terms Dashboard</h1>", unsafe_allow_html=True)
    with db_session() as session:
        search_terms_df = fetch_search_terms_with_lead_count(session)
        if search_terms_df.empty:
            return st.info("No search terms available. Add some to your campaigns.")
        
        st.columns(3)[0].metric("Total Search Terms", len(search_terms_df))
        st.columns(3)[1].metric("Total Leads", search_terms_df['Lead Count'].sum())
        st.columns(3)[2].metric("Total Emails Sent", search_terms_df['Email Count'].sum())
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Search Term Groups", "Performance", "Add New Term", "AI Grouping", "Manage Groups"])
        
        with tab1:
            groups = session.query(SearchTermGroup).all() + ["Ungrouped"]
            for group in groups:
                with st.expander(group.name if isinstance(group, SearchTermGroup) else group, expanded=True):
                    group_id = group.id if isinstance(group, SearchTermGroup) else None
                    terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all() if group_id else session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
                    updated_terms = st_tags(label="", text="Add or remove terms", value=[f"{term.id}: {term.term}" for term in terms], suggestions=[term for term in search_terms_df['Term'] if term not in [f"{t.id}: {t.term}" for t in terms]], key=f"group_{group_id}")
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
                group_to_delete = st.selectbox("Select Group to Delete", [f"{g.id}: {g.name}" for g in groups if isinstance(g, SearchTermGroup)], format_func=lambda x: x.split(":")[1])
                if st.button("Delete Group") and group_to_delete:
                    group_id = int(group_to_delete.split(":")[0])
                    delete_search_term_group(session, group_id)
                    st.success(f"Deleted group: {group_to_delete.split(':')[1]}")
                    st.rerun()

# 62. Email templates page: Input: None; Output: None; Manages email templates
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
                            new_template = EmailTemplate(template_name=new_template_name, subject=new_template_subject, body_content=new_template_body, campaign_id=get_active_campaign_id())
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
                    new_template = EmailTemplate(template_name=new_template_name, subject=new_template_subject, body_content=new_template_body, campaign_id=get_active_campaign_id())
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

# 63. Get email preview: Input: Session, int, str, str; Output: str; Returns HTML preview of email template
def get_email_preview(session, template_id, from_email, reply_to):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    return wrap_email_body(template.body_content) if template else "<p>Template not found</p>"

# 64. Fetch all search terms: Input: Session; Output: List[SearchTerm]; Retrieves all search terms
def fetch_all_search_terms(session):
    return session.query(SearchTerm).all()

# 65. Get email template by name: Input: Session, str; Output: EmailTemplate; Retrieves template by name
def get_email_template_by_name(session, template_name):
    return session.query(EmailTemplate).filter_by(template_name=template_name).first()

# 66. Bulk save email campaigns: Input: Session, List[dict]; Output: None; Saves multiple email campaigns
def bulk_save_email_campaigns(session: Session, campaigns: List[dict]):
    stmt = insert(EmailCampaign).values(campaigns)
    stmt = stmt.on_conflict_do_update(
        index_elements=['lead_id', 'template_id'],
        set_={
            'status': stmt.excluded.status,
            'sent_at': stmt.excluded.sent_at,
            'customized_subject': stmt.excluded.customized_subject,
            'message_id': stmt.excluded.message_id,
            'customized_content': stmt.excluded.customized_content
        }
    )
    session.execute(stmt)
    session.commit()

# 67. Fetch leads: Input: Session, int, str, str, List[str], bool; Output: List[dict]; Retrieves leads based on criteria
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
        log_error(f"Error fetching leads: {str(e)}")
        return []

# 68. Fetch email settings: Input: Session; Output: List[dict]; Retrieves email settings
def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        st.error("Failed to fetch email settings. Please try again later.")
        return []

# 69. Fetch search terms with lead count: Input: Session; Output: pd.DataFrame; Retrieves search terms with lead and email counts
@st.cache_data(ttl=3600, max_entries=100)
def fetch_search_terms_with_lead_count(session):
    query = session.query(
        SearchTerm.term,
        func.count(distinct(Lead.id)).label('lead_count'),
        func.count(distinct(EmailCampaign.id)).label('email_count')
    ).join(LeadSource, SearchTerm.id == LeadSource.search_term_id
    ).join(Lead, LeadSource.lead_id == Lead.id
    ).outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id
    ).group_by(SearchTerm.term)
    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

# 70. Fetch search term groups: Input: Session; Output: List[str]; Retrieves search term group names
def fetch_search_term_groups(session):
    return [f"{group.id}: {group.name}" for group in session.query(SearchTermGroup).all()]

# 71. Fetch search terms for groups: Input: Session, List[int]; Output: List[str]; Retrieves search terms for given group IDs
def fetch_search_terms_for_groups(session, group_ids):
    return [term.term for term in session.query(SearchTerm).filter(SearchTerm.group_id.in_(group_ids)).all()]

# 72. Projects and campaigns page: Input: None; Output: None; Manages projects and campaigns
def projects_campaigns_page():
    with db_session() as session:
        st.header("Projects and Campaigns")
        st.subheader("Add New Project")
        with st.form("add_project_form"):
            project_name = st.text_input("Project Name")
            submitted = st.form_submit_button("Add Project")
            if submitted:
                st.session_state.add_project_form_submitted = True
        handle_form_submit("add_project_form", lambda: add_project(session, project_name))

        st.subheader("Existing Projects and Campaigns")
        projects = session.query(Project).all()
        for project in projects:
            with st.expander(f"{project.project_name} (ID: {project.id})"):
                campaigns = session.query(Campaign).filter_by(project_id=project.id).all()
                for campaign in campaigns:
                    with st.expander(f"{campaign.campaign_name} (ID: {campaign.id})"):
                        st.write(f"Created At: {campaign.created_at}")
                        st.write(f"Last Run: {campaign.last_run or 'Never'}")
                        st.write(f"Total Leads Found: {campaign.total_leads_found}")
                        st.write(f"Total Emails Sent: {campaign.total_emails_sent}")
                        if st.button(f"Delete Campaign {campaign.id}", key=f"delete_campaign_{campaign.id}"):
                            session.delete(campaign)
                            session.commit()
                            st.success(f"Campaign {campaign.id} deleted successfully.")
                            st.experimental_rerun()
                with st.form(f"add_campaign_form_{project.id}"):
                    campaign_name = st.text_input("Campaign Name", key=f"campaign_name_{project.id}")
                    submitted = st.form_submit_button("Add Campaign", key=f"add_campaign_{project.id}")
                    if submitted:
                        st.session_state[f"add_campaign_form_{project.id}_submitted"] = True

                handle_form_submit(f"add_campaign_form_{project.id}", lambda: add_campaign(session, project.id, campaign_name))

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

# 73. Knowledge base page: Input: None; Output: None; Manages knowledge base for projects
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
            fields = ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']
            form_data = {field: st.text_input(field.replace('_', ' ').title(), value=getattr(kb_entry, field, '')) if field in ['kb_name', 'contact_name', 'contact_role', 'contact_email', 'product_name'] else st.text_area(field.replace('_', ' ').title(), value=getattr(kb_entry, field, '')) for field in fields}
            submitted = st.form_submit_button("Save Knowledge Base")
            if submitted:
                st.session_state.knowledge_base_form_submitted = True

        handle_form_submit("knowledge_base_form", lambda: save_knowledge_base(session, project_id, form_data))

# 74. Handle form submit: Input: str, Callable; Output: None; Handles form submission and rerun
def handle_form_submit(form_key, action):
    if st.session_state.get(f"{form_key}_submitted", False):
        action()
        st.session_state[f"{form_key}_submitted"] = False
        st.experimental_rerun()

# 75. Add project: Input: Session, str; Output: None; Adds a new project to the database
def add_project(session, project_name):
    if project_name.strip():
        try:
            session.add(Project(project_name=project_name, created_at=datetime.utcnow()))
            session.commit()
            st.success(f"Project '{project_name}' added successfully.")
        except SQLAlchemyError as e:
            st.error(f"Error adding project: {str(e)}")
    else:
        st.warning("Please enter a project name.")

# 76. Add campaign: Input: Session, int, str; Output: None; Adds a new campaign to the database
def add_campaign(session, project_id, campaign_name):
    if campaign_name.strip():
        try:
            session.add(Campaign(campaign_name=campaign_name, project_id=project_id, created_at=datetime.utcnow()))
            session.commit()
            st.success(f"Campaign '{campaign_name}' added successfully.")
        except SQLAlchemyError as e:
            st.error(f"Error adding campaign: {str(e)}")
    else:
        st.warning("Please enter a campaign name.")

# 77. Save knowledge base: Input: Session, int, dict; Output: None; Saves knowledge base information
def save_knowledge_base(session, project_id, form_data):
    form_data.update({'project_id': project_id, 'created_at': datetime.utcnow()})
    kb_entry = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    if kb_entry:
        for k, v in form_data.items():
            setattr(kb_entry, k, v)
    else:
        session.add(KnowledgeBase(**form_data))
    session.commit()
    st.success("Knowledge Base saved successfully!", icon="✅")

# 78. Main function: Input: None; Output: None; Main entry point for the Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["AutoclientAI", "Search Terms", "Email Templates", "Settings", "Automation Control Panel"])

    if page == "AutoclientAI":
        autoclient_ai_page()
    elif page == "Search Terms":
        search_terms_page()
    elif page == "Email Templates":
        email_templates_page()
    elif page == "Settings":
        settings_page()
    elif page == "Automation Control Panel":
        automation_control_panel_page()

    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")

if __name__ == "__main__":
    main()

# 79. AutoclientAI page: Input: None; Output: None; Manages automated lead generation process
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
        start_automation()
        st.success("Automation started!")
    if st.session_state.get('automation_logs'):
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

# 80. Automation control panel page: Input: None; Output: None; Manages automation control panel
def automation_control_panel_page():
    st.title("Automation Control Panel")
    col1, col2 = st.columns([2, 1])
    with col1:
        status = "Active" if scheduler.get_job('automation_job') else "Inactive"
        st.metric("Automation Status", status)
    with col2:
        button_text = "Stop Automation" if scheduler.get_job('automation_job') else "Start Automation"
        if st.button(button_text, use_container_width=True):
            if scheduler.get_job('automation_job'):
                stop_automation()
                st.success("Automation stopped.")
            else:
                start_automation()
                st.success("Automation started.")
    
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
    
    # Display Real-Time Analytics
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
    
    # Display Automation Logs
    st.subheader("Automation Logs")
    log_container = st.empty()
    automation_logs = get_automation_logs(session)
    display_results_and_logs(log_container, automation_logs, "Latest Logs", "logs")
    
    # Display Recently Found Leads
    st.subheader("Recently Found Leads")
    leads_container = st.empty()
    recent_leads = fetch_recent_leads(session)
    if recent_leads:
        leads_container.dataframe(pd.DataFrame(recent_leads, columns=["Email", "Company", "Created At"]))
    else:
        leads_container.info("No new leads found.")

# 81. Get knowledge base info: Input: Session, int; Output: dict; Retrieves knowledge base information for a project
@functools.lru_cache(maxsize=10)
def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

# 82. Generate optimized search terms: Input: Session, list, dict; Output: list; Generates optimized search terms using AI
def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"Generate 5 optimized search terms based on the following base terms and context:\nBase terms: {', '.join(base_terms)}\nContext: {json.dumps(kb_info)}\n\nRespond with a JSON object containing an array of optimized search terms.\nSchema: {{\"optimized_terms\": [\"term1\", \"term2\", \"term3\", \"term4\", \"term5\"]}}"
    messages = [{"role": "user", "content": prompt}]
    response = openai_chat_completion(messages, "generate_optimized_search_terms")
    return response.get("optimized_terms", []) if response else []

# 83. Display results and logs: Input: Container, list, str, str; Output: None; Displays results or logs in a scrollable container
def display_results_and_logs(container, items, title, item_type):
    if not items:
        container.info(f"No {item_type} to display yet.")
        return
    container.markdown(f"<style>.results-container {{max-height: 300px;overflow-y: auto;border: 1px solid rgba(49, 51, 63, 0.2);border-radius: 0.25rem;padding: 1rem;}}.result-entry {{margin-bottom: 0.5rem;padding: 0.5rem;border-radius: 0.25rem;background-color: rgba(28, 131, 225, 0.1);}}</style><h4>{title}</h4><div class='results-container'>{''.join(f'<div class=\"result-entry\">{item}</div>' for item in items[-20:])}</div>", unsafe_allow_html=True)

# 85. Get search terms: Input: Session; Output: list; Retrieves search terms for the active project
def get_search_terms(session):
    return [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]

# 86. Get AI response: Input: str; Output: str; Generates AI response based on given prompt
def get_ai_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    return openai_chat_completion(messages, "get_ai_response")

# 87. Fetch email settings: Input: Session; Output: list; Retrieves email settings from the database
def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        logging.error(f"Error fetching email settings: {e}")
        st.error("Failed to fetch email settings. Please try again later.")
        return []

# 88. Wrap email body: Input: str; Output: str; Wraps email body content in HTML template
def wrap_email_body(body_content):
    return f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>Email Template</title><style>body {{font-family: Arial, sans-serif;line-height: 1.6;color: #333;max-width: 600px;margin: 0 auto;padding: 20px;}}</style></head><body>{body_content}</body></html>"

# 89. Fetch sent email campaigns: Input: Session; Output: DataFrame; Retrieves sent email campaigns data
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

# 90. View sent email campaigns: Input: None; Output: None; Displays sent email campaigns in the UI
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

# 91. Get user settings: Input: None; Output: dict; Retrieves or initializes user settings
def get_user_settings():
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = {
            'enable_email_sending': True,
            'ignore_previously_fetched': True,
            'shuffle_keywords_option': True,
            'optimize_english': False,
            'optimize_spanish': False,
            'language': 'EN'
        }
    return st.session_state.user_settings

# 92. Rate limited API call: Input: function, *args, **kwargs; Output: Any; Executes API calls with rate limiting
@sleep_and_retry
@limits(calls=100, period=60)
def rate_limited_api_call(func, *args, **kwargs):
    return func(*args, **kwargs)

# 93. Load leads progressively: Input: Session, int; Output: Generator; Yields leads in batches
def load_leads_progressively(session, page_size=100):
    offset = 0
    while True:
        leads = session.query(Lead).order_by(Lead.id).offset(offset).limit(page_size).all()
        if not leads:
            break
        yield from leads
        offset += page_size

# 94. Optimized manual search: Input: Session, list, int; Output: list; Performs optimized search for leads
def optimized_manual_search(session, search_terms, num_results):
    results = []
    for term in search_terms:
        results.extend(auto_perform_optimized_search(session, term, num_results))
    return results[:num_results]

# 95. Auto perform optimized search: Input: Session, str, int; Output: list; Performs optimized search for a single term
def auto_perform_optimized_search(session, term, num_results):
    search_query = session.query(Lead).filter(Lead.email.ilike(f"%{term}%"))
    if session.bind.dialect.name == 'postgresql':
        search_query = search_query.with_hint(Lead, 'USE INDEX (ix_lead_email)')
    search_results = search_query.limit(num_results).all()
    return [{"id": lead.id, "email": lead.email, "first_name": lead.first_name, "last_name": lead.last_name, "company": lead.company, "position": lead.position, "country": lead.country, "industry": lead.industry} for lead in search_results]

# 96. Manual search page: Input: None; Output: None; Handles manual search UI and functionality
def manual_search_page():
    st.title("Manual Search")
    start_background_process()
    user_settings = get_user_settings()
    with db_session() as session:
        state = get_background_state()
        if state:
            col1, col2, col3 = st.columns(3)
            col1.metric("Background Status", "Running" if state['is_running'] else "Paused")
            col2.metric("Total Leads Found", state['leads_found'])
            col3.metric("Total Emails Sent", state['emails_sent'])
            st.info(f"Last run: {state['last_run'] or 'Never'} | Current term: {state['current_term'] or 'None'}")
        latest_leads = session.query(Lead).order_by(Lead.created_at.desc()).limit(5).all()
        latest_leads_data = [(lead.email, lead.company, lead.created_at) for lead in latest_leads]
        latest_campaigns = session.query(EmailCampaign).order_by(EmailCampaign.sent_at.desc()).limit(5).all()
        latest_campaigns_data = [(campaign.lead.email, campaign.template.template_name, campaign.sent_at, campaign.status) for campaign in latest_campaigns]
        recent_searches = session.query(SearchTerm).order_by(SearchTerm.created_at.desc()).limit(5).all()
        recent_search_terms = [term.term for term in recent_searches]
    col1, col2 = st.columns([2, 1])
    with col1:
        search_terms = st_tags(label='Enter search terms:', text='Press enter to add more', value=recent_search_terms, suggestions=['software engineer', 'data scientist', 'product manager'], maxtags=10, key='search_terms_input')
        num_results = st.slider("Results per term", 1, 500, 10)
    with col2:
        st.subheader("Search Options")
        enable_email_sending = st.checkbox("Enable email sending", value=user_settings['enable_email_sending'])
        ignore_previously_fetched = st.checkbox("Ignore fetched domains", value=user_settings['ignore_previously_fetched'])
        shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=user_settings['shuffle_keywords_option'])
        optimize_english = st.checkbox("Optimize (English)", value=user_settings['optimize_english'])
        optimize_spanish = st.checkbox("Optimize (Spanish)", value=user_settings['optimize_spanish'])
        language = st.selectbox("Select Language", options=["ES", "EN"], index=0 if user_settings['language'] == "ES" else 1)
        one_email_per_url = st.checkbox("Only One Email per URL", value=True)
        one_email_per_domain = st.checkbox("Only One Email per Domain", value=True)
        if enable_email_sending:
            with db_session() as session:
                email_templates = fetch_email_templates(session)
            email_settings = fetch_email_settings(session)
            if not email_templates or not email_settings:
                st.error("No email templates or settings available. Please set them up first.")
                return
            col3, col4 = st.columns(2)
            with col3:
                email_template = st.selectbox("Email template", options=email_templates, format_func=lambda x: x.split(":")[1].strip())
            with col4:
                email_setting_option = st.selectbox("From Email", options=email_settings, format_func=lambda x: f"{x['name']} ({x['email']})")
                if email_setting_option:
                    from_email = email_setting_option['email']
                    reply_to = st.text_input("Reply To", email_setting_option['email'])
                else:
                    st.error("No email setting selected. Please select an email setting.")
                    return
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Run Manual Search", use_container_width=True):
            st.session_state.run_manual_search = True
        if st.session_state.get('run_manual_search', False):
            with db_session() as session:
                job_enqueued = enqueue_job(session, 'manual_search', {
                    'search_terms': search_terms,
                    'num_results': num_results,
                    'ignore_previously_fetched': ignore_previously_fetched,
                    'optimize_english': optimize_english,
                    'optimize_spanish': optimize_spanish,
                    'shuffle_keywords_option': shuffle_keywords_option,
                    'language': language,
                    'enable_email_sending': enable_email_sending,
                    'from_email': from_email,
                    'reply_to': reply_to,
                    'email_template': email_template,
                    'one_email_per_url': one_email_per_url,
                    'one_email_per_domain': one_email_per_domain
                })
                if job_enqueued:
                    st.success("Manual search job enqueued successfully!")
                else:
                    st.warning("A job is already running. Please wait for it to complete.")
            with db_session() as session:
                state = get_background_state()
                if state['is_running'] and state['job_type'] == 'manual_search':
                    st.info(f"Manual search in progress. Current term: {state['current_term']}")
                    st.progress(state['job_progress'])
    with col2:
        if st.button("Pause/Resume Background Search", use_container_width=True):
            with db_session() as session:
                state = get_background_state()
                if state['is_running']:
                    pause_background_search()
                    st.success("Background search paused.")
                else:
                    resume_background_search()
                    st.success("Background search resumed.")
    with col3:
        if st.button("Stop Background Search", use_container_width=True):
            stop_background_search()
            st.success("Background search stopped.")
    st.subheader("Latest Leads")
    st.table(pd.DataFrame(latest_leads_data, columns=["Email", "Company", "Created At"]))
    st.subheader("Latest Email Campaigns")
    st.table(pd.DataFrame(latest_campaigns_data, columns=["Email", "Template", "Sent At", "Status"]))

# 97. View leads page: Input: None; Output: None; Displays and manages leads data
def view_leads_page():
    st.title("View Leads")
    
    with db_session() as session:
        page_size = 50
        page_number = st.number_input("Page", min_value=1, value=1)
        offset = (page_number - 1) * page_size
        
        leads = session.query(Lead).order_by(Lead.created_at.desc()).offset(offset).limit(page_size).all()
        
        leads_df = pd.DataFrame([
            {
                "ID": lead.id,
                "Email": lead.email,
                "Name": f"{lead.first_name} {lead.last_name}",
                "Company": lead.company,
                "Job Title": lead.job_title,
                "Created At": lead.created_at
            } for lead in leads
        ])
        
        st.dataframe(leads_df)
        
        search_term = st.text_input("Search leads")
        if search_term:
            filtered_df = leads_df[leads_df.apply(lambda row: search_term.lower() in row.astype(str).str.lower().sum(), axis=1)]
            st.dataframe(filtered_df)
        
        if st.button("Export Leads"):
            csv = leads_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="leads_export.csv",
                mime="text/csv"
            )

# 98. Main function: Input: None; Output: None; Manages the main application flow and UI
def main():
    try:
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
        if selected in pages:
            pages[selected]()
        else:
            st.error(f"Selected page '{selected}' not found.")
        st.sidebar.markdown("---")
        st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main function: {str(e)}")
        st.error("An unexpected error occurred. Please try refreshing the page or contact support if the issue persists.")
        st.exception(e)
    auto_refresh()

if __name__ == "__main__":
    main()
