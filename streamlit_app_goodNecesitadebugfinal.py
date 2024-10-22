import os
import json
import re
import logging
import asyncio
import time
import requests
import pandas as pd
import streamlit as st
import openai
import boto3
import uuid
import aiohttp
import urllib3
import random
import html
import smtplib
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

scheduler = BackgroundScheduler()
scheduler.start()

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

# ... (rest of the code remains unchanged)

# ... (rest of the code remains unchanged)

# ... (rest of the code remains unchanged)

