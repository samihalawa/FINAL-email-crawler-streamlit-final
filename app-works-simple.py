"""
AUTOCLIENT - Lead Generation App

This application facilitates lead generation through manual and bulk searches,
manages campaigns, templates, and integrates AI functionalities for optimizing
search terms and email templates.
"""

# --------------------- Standard Imports ---------------------
import os
import json
import re
import logging
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# --------------------- Third-Party Imports ---------------------
from streamlit_option_menu import option_menu
# --------------------- Third-Party Imports ---------------------
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from fake_useragent import UserAgent
import pandas as pd
import streamlit as st
import openai
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func
import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
import time
import logging

# Initialize OpenAI client
client = None
try:
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except openai.OpenAIError as e:
    logging.error(f"Failed to initialize OpenAI client: {str(e)}")

@retry(wait=wait_fixed(20) + wait_random_exponential(min=1, max=40), stop=stop_after_attempt(6))
def chat_completion_with_backoff(**kwargs):
    if client is None:
        raise ValueError("OpenAI client is not initialized")

    try:
        return client.chat.completions.create(**kwargs)
    except openai.RateLimitError:
        logging.warning("OpenAI API rate limit exceeded. Retrying...")
        raise
    except openai.APIError as e:
        logging.error(f"OpenAI API error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in OpenAI API call: {str(e)}")
        raise

def safe_chat_completion(**kwargs):
    try:
        return chat_completion_with_backoff(**kwargs)
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {str(e)}")
        return None

def send_email_ses(from_email, to_email, subject, body, charset='UTF-8'):
    client = boto3.client('ses', 
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
                          region_name=os.getenv('AWS_REGION'))
    try:
        response = client.send_email(
            Source=from_email,
            Destination={
                'ToAddresses': [to_email],
            },
            Message={
                'Subject': {
                    'Data': subject,
                    'Charset': charset,
                },
                'Body': {
                    'Text': {
                        'Data': body,
                        'Charset': charset,
                    },
                },
            },
        )
        return response
    except ClientError as e:
        logging.error(f"Error sending email: {e.response['Error']['Message']}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error sending email: {str(e)}")
        return None

def send_email_to_lead(session, lead, template, from_email, reply_to):
    try:
        # Customize the email content (you might want to add more sophisticated customization)
        subject = template['subject']
        body = template['body']

        # Send the email using send_email_ses function
        ses_response = send_email_ses(from_email, lead['Email'], subject, body)

        # Log the email campaign
        campaign = EmailCampaign(
            lead_id=lead['Lead ID'],
            status='sent',
            sent_at=datetime.utcnow(),
            message_id=ses_response['MessageId'],
            customized_subject=subject,
            customized_content=body
        )
        session.add(campaign)
        session.commit()

        return {
            "Email": lead['Email'],
            "Status": "sent",
            "Message ID": ses_response['MessageId']
        }
    except Exception as e:
        return {
            "Email": lead['Email'],
            "Status": "failed",
            "Error": str(e)
        }

# Load environment variables
load_dotenv()

# --------------------- OpenAI Configuration ---------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE
openai.model = "gpt-4o-mini"

# --------------------- Database Schema ---------------------
from sqlalchemy import (
    create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"  # Ensure these env variables are set

# Create engine and session
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Define database models
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
    kb_name = Column(Text)
    kb_bio = Column(Text)
    kb_values = Column(Text)
    contact_name = Column(Text)
    contact_role = Column(Text)
    contact_email = Column(Text)
    company_description = Column(Text)
    company_mission = Column(Text)
    company_target_market = Column(Text)
    company_other = Column(Text)
    product_name = Column(Text)
    product_description = Column(Text)
    product_target_customer = Column(Text)
    product_other = Column(Text)
    other_context = Column(Text)
    example_email = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    project = relationship("Project", back_populates="knowledge_base")

class LeadSource(Base):
    __tablename__ = 'lead_sources'
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    url = Column(Text)
    page_title = Column(Text)
    meta_description = Column(Text)
    scrape_duration = Column(Text)
    meta_tags = Column(Text)
    phone_numbers = Column(Text)
    content = Column(Text)
    tags = Column(Text)
    http_status = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="lead_sources")
    search_term = relationship("SearchTerm", back_populates="lead_sources")

class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text)
    phone = Column(Text)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text)
    job_title = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaign_leads = relationship("CampaignLead", back_populates="lead")
    lead_sources = relationship("LeadSource", back_populates="lead")
    email_campaigns = relationship("EmailCampaign", back_populates="lead")

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    template_name = Column(Text)
    subject = Column(Text)
    body_content = Column(Text)
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
    customized_subject = Column(Text)
    customized_content = Column(Text)
    original_subject = Column(Text)
    original_content = Column(Text)
    status = Column(Text)
    engagement_data = Column(JSON)
    message_id = Column(Text)
    sent_at = Column(DateTime(timezone=True))
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
    total_results = Column(BigInteger)
    valid_leads = Column(BigInteger)
    irrelevant_leads = Column(BigInteger)
    blogs_found = Column(BigInteger)
    directories_found = Column(BigInteger)
    term = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class SearchTermGroup(Base):
    __tablename__ = 'search_term_groups'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    email_template = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_terms = relationship("SearchTerm", back_populates="group")

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    id = Column(BigInteger, primary_key=True)
    group_id = Column(BigInteger, ForeignKey('search_term_groups.id'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    term = Column(Text)
    category = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    group = relationship("SearchTermGroup", back_populates="search_terms")
    campaign = relationship("Campaign", back_populates="search_terms")
    optimized_terms = relationship("OptimizedSearchTerm", back_populates="original_term")
    lead_sources = relationship("LeadSource", back_populates="search_term")

class AIRequestLog(Base):
    __tablename__ = 'ai_request_logs'
    id = Column(BigInteger, primary_key=True)
    function_name = Column(Text)
    prompt = Column(Text)
    response = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    email_campaign_id = Column(BigInteger, ForeignKey('email_campaigns.id'))
    model_used = Column(Text)
    lead = relationship("Lead")
    email_campaign = relationship("EmailCampaign")

class AutomationLog(Base):
    __tablename__ = 'automation_logs'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    leads_gathered = Column(BigInteger)
    emails_sent = Column(BigInteger)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    status = Column(Text)
    logs = Column(JSON)
    campaign = relationship("Campaign")
    search_term = relationship("SearchTerm")

# Create all tables
Base.metadata.create_all(engine)

# --------------------- Helper Functions ---------------------
def initialize_aws_session():
    return boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

# Only one get_db_connection function
from contextlib import contextmanager

# Remove async from get_db_connection and use it synchronously
def get_db_connection():
    """Create a new database session."""
    return SessionLocal()

async def get_db():
    db = await get_db_connection()
    try:
        yield db
    finally:
        await db.close()




def is_valid_email(email):
    pattern = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    return re.match(pattern, email) is not None

def find_emails(text):
    email_regex = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
    return set(email_regex.findall(text))

def extract_phone_numbers(text):
    phone_pattern = re.compile(
        r'\b(?:\+?(\d{1,3}))?[\s.-]?'  # Country code
        r'(?:\(?(\d{2,4})\)?[\s.-]?)?'  # Area code
        r'(\d{3})[\s.-]?'  # First 3 digits
        r'(\d{4})\b'  # Last 4 digits
    )
    matches = phone_pattern.findall(text)
    phone_numbers = []
    for match in matches:
        country_code, area_code, first_three, last_four = match
        phone_number = ""
        if country_code:
            phone_number += f"+{country_code} "
        if area_code:
            phone_number += f"({area_code}) "
        phone_number += f"{first_three}-{last_four}"
        phone_numbers.append(phone_number.strip())
    return phone_numbers

def extract_visible_text(soup):
    for element in soup(['style', 'script', 'head', 'title', 'meta', '[document]']):
        element.extract()
    return ' '.join(soup.stripped_strings)

def get_domain_from_url(url):
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    return domain

def save_lead(session, email, phone=None, first_name=None, last_name=None, company=None, job_title=None, domain=None):
    if not is_valid_email(email):
        return None
    existing_lead = session.query(Lead).filter_by(email=email).first()
    if existing_lead:
        return existing_lead.id
    if domain:
        domain_lead = session.query(Lead).filter(Lead.email.like(f"%@{domain}")).first()
        if domain_lead:
            return None  # Skip if email from this domain already exists
    new_lead = Lead(email=email, phone=phone, first_name=first_name, last_name=last_name, company=company, job_title=job_title)
    session.add(new_lead)
    session.commit()
    session.refresh(new_lead)
    return new_lead.id

def save_lead_source(session, lead_id, search_term_id, url, page_title, meta_description, phone_numbers, content, tags):
    lead_source = LeadSource(
        lead_id=lead_id,
        search_term_id=search_term_id,
        url=url,
        page_title=page_title,
        meta_description=meta_description,
        content=content,
        tags=json.dumps(tags),
        phone_numbers=json.dumps(phone_numbers)
    )
    session.add(lead_source)
    session.commit()

def save_email_campaign(session, lead_id, template_id, status, sent_at=None, subject=None, message_id=None, customized_content=None):
    email_campaign = EmailCampaign(
        lead_id=lead_id,
        template_id=template_id,
        status=status,
        sent_at=sent_at,
        customized_subject=subject,
        message_id=message_id,
        customized_content=customized_content
    )
    session.add(email_campaign)
    session.commit()

def log_ai_request(session, function_name, prompt, response, lead_id=None, email_campaign_id=None):
    log_entry = AIRequestLog(
        function_name=function_name,
        prompt=json.dumps(prompt),
        response=json.dumps(response) if response else None,
        lead_id=lead_id,
        email_campaign_id=email_campaign_id,
        model_used=openai.model
    )
    session.add(log_entry)
    session.commit()

def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
    effectiveness = SearchTermEffectiveness(
        term=term,
        total_results=total_results,
        valid_leads=valid_leads,
        irrelevant_leads=total_results - valid_leads,
        blogs_found=blogs_found,
        directories_found=directories_found
    )
    session.add(effectiveness)
    session.commit()

def get_active_project_id():
    return st.session_state.get('active_project_id', 1)

def get_active_campaign_id():
    return st.session_state.get('active_campaign_id', 1)

def set_active_project_id(project_id):
    st.session_state['active_project_id'] = project_id

def set_active_campaign_id(campaign_id):
    st.session_state['active_campaign_id'] = campaign_id

def add_search_term(session, term, campaign_id):
    campaign_id = get_active_campaign_id()
    new_term = SearchTerm(term=term, campaign_id=campaign_id)
    session.add(new_term)
    session.commit()

def add_or_get_search_term(session, term, campaign_id):
    search_term = session.query(SearchTerm).filter_by(term=term, campaign_id=campaign_id).first()
    if search_term:
        return search_term.id
    else:
        new_term = SearchTerm(term=term, campaign_id=campaign_id)
        session.add(new_term)
        session.commit()
        session.refresh(new_term)
        new_term.leads_count = 1
        session.commit()
        return new_term.id

def fetch_campaigns(session):
    try:
        campaigns = session.query(Campaign).all()
        return [f"{camp.id}: {camp.campaign_name}" for camp in campaigns]
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return []

def fetch_projects(session):
    try:
        projects = session.query(Project).all()
        return [f"{project.id}: {project.project_name}" for project in projects]
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return []

def fetch_email_templates(session, campaign_id=None):
    try:
        templates = []
        if campaign_id:
            templates = session.query(EmailTemplate).filter_by(campaign_id=campaign_id).all()
        else:
            templates = session.query(EmailTemplate).all()
        return [f"{template.id}: {template.template_name}" for template in templates]
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return []

def create_email_template(session, template_name, subject, body_content):
    campaign_id = get_active_campaign_id()
    # Check if template already exists
    existing_template = session.query(EmailTemplate).filter_by(template_name=template_name, campaign_id=campaign_id).first()
    if existing_template:
        return existing_template.id
    new_template = EmailTemplate(
        template_name=template_name,
        subject=subject,
        body_content=body_content,
        campaign_id=campaign_id
    )
    session.add(new_template)
    session.commit()
    return new_template.id

def fetch_search_terms(session, campaign_id=None):
    if campaign_id:
        terms = session.query(SearchTerm).filter_by(campaign_id=campaign_id).all()
    else:
        terms = session.query(SearchTerm).all()
    data = {
        "ID": [term.id for term in terms],
        "Search Term": [term.term for term in terms],
        "Campaign": [term.campaign.campaign_name if term.campaign else '' for term in terms],
        "Created At": [term.created_at.strftime("%Y-%m-%d %H:%M:%S") for term in terms]
    }
    return pd.DataFrame(data)

def fetch_leads(session):
    leads = []
    try:
        try:
            leads = session.query(Lead).all()
        except SQLAlchemyError as e:
            st.error(f"Database error: {str(e)}")
            return []
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
    data = {
        "ID": [lead.id for lead in leads],
        "Email": [lead.email for lead in leads],
        "Phone": [lead.phone for lead in leads],
        "First Name": [lead.first_name for lead in leads],
        "Last Name": [lead.last_name for lead in leads],
        "Company": [lead.company for lead in leads],
        "Job Title": [lead.job_title for lead in leads],
        "Created At": [lead.created_at.strftime("%Y-%m-%d %H:%M:%S") for lead in leads]
    }
    return pd.DataFrame(data)

def fetch_sent_email_campaigns(session):
    try:
        email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).order_by(EmailCampaign.sent_at.desc()).all()
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return []
    data = {
        'ID': [ec.id for ec in email_campaigns],
        'Sent At': [ec.sent_at.strftime("%Y-%m-%d %H:%M:%S") if ec.sent_at else "" for ec in email_campaigns],
        'Email': [ec.lead.email for ec in email_campaigns],
        'Template': [ec.template.template_name for ec in email_campaigns],
        'Subject': [ec.customized_subject for ec in email_campaigns],
        'Content': [ec.customized_content for ec in email_campaigns],
        'Status': [ec.status for ec in email_campaigns],
        'Message ID': [ec.message_id for ec in email_campaigns]
    }
    return pd.DataFrame(data)

def view_sent_email_campaigns_page():
    st.header("Sent Email Campaigns")
    with get_db_connection() as session:
        email_campaigns = fetch_sent_email_campaigns(session)
    if email_campaigns.empty:
        st.info("No sent email campaigns found.")
    else:
        st.dataframe(email_campaigns)
    session.close()

def fetch_leads_for_bulk_send(session, template_id, send_option, filter_option):
    if send_option == "All Not Contacted with this Template":
        subquery = session.query(EmailCampaign.lead_id).filter(EmailCampaign.template_id == template_id).subquery()
        query = session.query(Lead.id, Lead.email).filter(~Lead.id.in_(subquery))
    else:
        query = session.query(Lead.id, Lead.email)
    if filter_option == "Filter Out blog-directory":
        query = query.join(LeadSource).filter(~LeadSource.tags.contains('blog-directory'))
    leads = query.distinct().all()
    return leads

def categorize_page_content(soup, url):
    title = soup.title.string.lower() if soup.title else ''
    meta_description = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_description['content'].lower() if meta_description else ''
    if 'blog' in url.lower() or 'blog' in title or 'blog' in meta_description:
        return 'blog'
    elif 'directory' in url.lower() or 'directory' in title or 'directory' in meta_description:
        return 'directory'
    else:
        return 'company'

def extract_emails_from_html(html_content):
    emails = set()
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Extract emails from mailto links
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('mailto:'):
                emails.add(href[7:])
        # Extract emails from scripts and other tags
        text_content = soup.get_text()
        email_matches = find_emails(text_content)
        emails.update(email_matches)
    return list(emails)
import time
from requests.exceptions import RequestException

def manual_search(session, terms, num_results, search_type="All Leads", search_language="en-US", log_container=None, leads_container=None):
    all_results, logs, emails_found = [], [], []
    ua = UserAgent()
    headers = {'User-Agent': ua.random}
    campaign_id = get_active_campaign_id()

    term_progress = st.progress(0)
    for term_counter, term in enumerate(terms, 1):
        st.write(f"Searching for: {term}")
        try:
            term_id = add_or_get_search_term(session, term, campaign_id)
        except Exception as e:
            logs.append(f"Error adding/getting search term: {str(e)}")
            continue

        valid_leads_count = blogs_found = directories_found = 0

        max_retries = 3
        retry_delay = 5

        search_urls = []
        for attempt in range(max_retries):
            try:
                search_urls = list(search(term, num_results=num_results, lang=search_language))
                break
            except RequestException as e:
                if attempt < max_retries - 1:
                    log_message = f"Error searching for '{term}'. Retrying in {retry_delay} seconds... Error: {str(e)}"
                    if log_container:
                        log_container.warning(log_message)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    log_message = f"Failed to search for '{term}' after {max_retries} attempts: {str(e)}"
                    if log_container:
                        log_container.error(log_message)
                    logs.append(log_message)

        if not search_urls:
            logs.append(f"No search results found for term: {term}")
            continue

        url_progress = st.progress(0)
        for url_counter, url in enumerate(search_urls, 1):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                emails = extract_emails_from_html(response.text)
                phone_numbers = extract_phone_numbers(response.text)
                content = extract_visible_text(soup)
                title = soup.title.string.strip() if soup.title else ""
                meta_description = soup.find('meta', attrs={'name': 'description'})
                meta_description = meta_description['content'].strip() if meta_description else ""
                categorized = categorize_page_content(soup, url)
                tags = ['blog'] if categorized == 'blog' else ['directory'] if categorized == 'directory' else ['company']
                blogs_found += categorized == 'blog'
                directories_found += categorized == 'directory'

                domain = get_domain_from_url(url)

                for email in emails:
                    if is_valid_email(email):
                        try:
                            lead_id = save_lead(session, email, phone=phone_numbers[0] if phone_numbers else None, domain=domain)
                            if lead_id:
                                save_lead_source(session, lead_id, term_id, url, title, meta_description, phone_numbers, content, tags)
                                all_results.append({"Email": email, "URL": url, "Title": title, "Description": meta_description, "Tags": tags, "Lead Source": url, "Lead ID": lead_id})
                                emails_found.append(email)
                                if log_container:
                                    log_container.text_area("Emails Found", "\n".join(emails_found[-50:]), height=200)
                                if leads_container:
                                    leads_container.text_area("Emails Found", "\n".join(emails_found[-50:]), height=200)
                                logs.append(f"New Lead! {email} from {url}")
                                if log_container:
                                    log_container.text_area("Search Logs", "\n".join(logs[-10:]), height=200)
                                valid_leads_count += 1
                                break
                        except Exception as e:
                            logs.append(f"Error saving lead {email}: {str(e)}")
            except requests.exceptions.RequestException as e:
                logs.append(f"Error processing {url}: {str(e)}")
                if log_container:
                    log_container.text_area("Search Logs", "\n".join(logs[-10:]), height=200)
            except Exception as e:
                logs.append(f"Unexpected error processing {url}: {str(e)}")
                if log_container:
                    log_container.text_area("Search Logs", "\n".join(logs[-10:]), height=200)

            url_progress.progress(url_counter / len(search_urls))

        try:
            log_search_term_effectiveness(session, term, len(search_urls), valid_leads_count, blogs_found, directories_found)
        except Exception as e:
            logs.append(f"Error logging search term effectiveness: {str(e)}")

        term_progress.progress(term_counter / len(terms))

    st.success("Search completed!")
    return all_results

async def save_email_campaign_async(session, lead_id, template_id, status, sent_at, subject, message_id, customized_content):
    try:
        email_campaign = EmailCampaign(
            lead_id=lead_id,
            template_id=template_id,
            status=status,
            sent_at=sent_at,
            customized_subject=subject,
            message_id=message_id,
            customized_content=customized_content
        )
        session.add(email_campaign)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise e




def save_search_term_groups_and_templates(session, search_term_groups, email_templates):
    try:
        # Save search terms
        for group_name, terms in search_term_groups.items():
            group = session.query(SearchTermGroup).filter_by(name=group_name).first()
            if not group:
                group = SearchTermGroup(name=group_name)
                session.add(group)
                session.flush()

            # Delete existing terms for this group
            session.query(SearchTerm).filter_by(group_id=group.id).delete()

            # Add new terms
            for term in terms:
                new_term = SearchTerm(term=term, group_id=group.id)
                session.add(new_term)

        # Save email templates
        for group_name, template_data in email_templates.items():
            template = session.query(EmailTemplate).filter_by(template_name=f"Template_{group_name}").first()
            if template:
                template.subject = template_data['subject']
                template.body_content = template_data['body']
            else:
                new_template = EmailTemplate(
                    template_name=f"Template_{group_name}",
                    subject=template_data['subject'],
                    body_content=template_data['body']
                )
                session.add(new_template)

        session.commit()
        return True
    except Exception as e:
        session.rollback()
        st.error(f"Error saving search term groups and templates: {str(e)}")
        return False


async def bulk_send_coroutine(session, template_id, from_email, reply_to, leads):
    logs = []
    all_results = []
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        logs.append("Template not found.")
        return logs, all_results
    total_leads = len(leads)
    progress = st.progress(0)
    for index, (lead_id, email) in enumerate(leads, start=1):
        logs.append(f"Processing lead {lead_id} with email {email}...")
        try:
            sent_at = datetime.utcnow()
            message_id = f"msg-{lead_id}-{int(time.time())}"
            status = 'sent'
            customized_content = template.body_content

            # Send email using AWS SES
            ses_response = send_email_ses(from_email, email, template.subject, customized_content)


            # Send email using AWS SES
            ses_response = send_email_ses(from_email, email, template.subject, customized_content)

            if ses_response:
                message_id = ses_response['MessageId']
                await save_email_campaign_async(session, lead_id, template_id, status, sent_at, template.subject, message_id, customized_content)
                logs.append(f"[{index}/{total_leads}] Sent email to {email} - Message ID: {message_id}")
                all_results.append({"Email": email, "Status": status, "Message ID": message_id})
            else:
                raise Exception("Failed to send email via SES")


            await asyncio.sleep(0.1)
            progress.progress(index / total_leads)
        except Exception as e:
            await save_email_campaign_async(session, lead_id, template_id, 'failed', None, template.subject, None, str(e))
            logs.append(f"[{index}/{total_leads}] Failed to send email to {email}: {e}")
            all_results.append({"Email": email, "Status": 'failed', "Message ID": None})
    return logs, all_results

def update_log_display(log_container, log_key):
    if log_key not in st.session_state:
        st.session_state[log_key] = []
    log_container.markdown(f"""
    <div style="height:200px;overflow:auto;">
    {'<br>'.join(st.session_state[log_key][-20:])}
    </div>
    """, unsafe_allow_html=True)

async def continuous_automation_process(session, log_container, leads_container):
    emails_found = []
    while st.session_state.get('automation_status', False):
        try:
            search_terms = fetch_all_search_terms(session)
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not search_terms or not kb_info:
                st.warning("Missing search terms or knowledge base.")
                return

            classified_terms = classify_search_terms(session, search_terms, kb_info)
            update_log_display(log_container, "automation_logs")
            log_container.info(f"Classified search terms into {len(classified_terms)} groups.")

            reference_template = get_email_template_by_name(session, "REFERENCE")
            if not reference_template:
                st.warning("REFERENCE email template not found.")
                return

            for group_name, terms in classified_terms.items():
                if group_name == "low_quality_search_terms":
                    continue

                adjusted_template_data = adjust_email_template_api(session, reference_template.body_content, f"Adjust for terms: {', '.join(terms)}", kb_info)
                if not adjusted_template_data or 'body' not in adjusted_template_data:
                    st.warning(f"Failed to adjust template for '{group_name}'.")
                    continue

                new_template_id = create_email_template(session, f"{reference_template.template_name}_{group_name}", adjusted_template_data['subject'], adjusted_template_data['body'])
                update_log_display(log_container, "automation_logs")
                log_container.info(f"Created template '{reference_template.template_name}_{group_name}' for '{group_name}'.")

                results = manual_search(session, terms, 10, "All Leads", log_container, leads_container)
                new_emails = [res['Email'] for res in results if res['Email'] not in emails_found]
                emails_found.extend(new_emails)

                leads_container.text_area("Emails Found", "\n".join(emails_found[-50:]), height=200)

                leads_to_send = [(res['Lead ID'], email) for res, email in zip(results, new_emails)]
                from_email = kb_info.get('contact_email', 'default@example.com')
                reply_to = kb_info.get('contact_email', 'default@example.com')
                await bulk_send_coroutine(session, new_template_id, from_email, reply_to, leads_to_send, log_container)

            await asyncio.sleep(60)

        except Exception as e:
            update_log_display(log_container, "automation_logs")
            log_container.error(f"Automation error: {e}")
            break

def display_search_results(results):
    if not results:
        st.warning("No results to display.")
        return
    st.markdown(f"### Total Leads Found: **{len(results)}**")
    with st.expander("View All Results", expanded=False):
        for res in results:
            st.markdown(f"""
            **Email:** {res['Email']}  
            **URL:** [{res['URL']}]({res['URL']})  
            **Title:** {res['Title']}  
            **Description:** {res['Description']}  
            **Tags:** {', '.join(res['Tags'])}  
            **Lead Source:** {res['Lead Source']}
            ---
            """)

def perform_quick_scan(session):
    terms = get_least_searched_terms(session, 3)
    results = manual_search(session, terms, 10, "All Leads")
    return {"new_leads": len(results)}

# --------------------- Page Functions ---------------------
import time

def manual_search_page():
    st.header("Manual Search")

    if not st.session_state.get('search_in_progress', False):
        with st.form("manual_search_form"):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("Enter your search terms below to find leads. Each term should be on a new line.")
                search_terms = st.text_area("Enter Search Terms (one per line)", height=150)
            with col2:
                with get_db_connection() as session:
                    campaigns = fetch_campaigns(session)
                campaign = st.selectbox("Select Campaign", options=campaigns)
                set_active_campaign_id(int(campaign.split(":")[0]))
                num_results = st.slider("Number of Results per Term", 10, 200, 30, step=10)
                search_type = st.selectbox("Search Type", ["All Leads", "Exclude Probable Blogs/Directories"])
            submit = st.form_submit_button("Start Search")

        if submit:
            terms = [term.strip() for term in search_terms.split('\n') if term.strip()]
            if not terms:
                st.warning("Please enter at least one valid search term.")
            else:
                st.session_state.search_params = {
                    'terms': terms,
                    'num_results': num_results,
                    'search_type': search_type
                }
                st.session_state.search_in_progress = True
                st.session_state.search_results = []
                st.session_state.current_term_index = 0

    if st.session_state.get('search_in_progress', False):
        st.subheader("Search Progress")

        col1, col2 = st.columns([2, 1])
        with col1:
            progress_bar = st.progress(0)
            status_text = st.empty()
        with col2:
            stats_container = st.container()

        results_container = st.container()

        terms = st.session_state.search_params['terms']
        num_results = st.session_state.search_params['num_results']
        search_type = st.session_state.search_params['search_type']

        while st.session_state.current_term_index < len(terms):
            term = terms[st.session_state.current_term_index]
            try:
                with get_db_connection() as session:
                    term_results = manual_search(session, [term], num_results, search_type)
                st.session_state.search_results.extend(term_results)

                progress = (st.session_state.current_term_index + 1) / len(terms)
                progress_bar.progress(progress)
                status_text.text(f"Searching term {st.session_state.current_term_index + 1} of {len(terms)}: {term}")

                with stats_container:
                    st.metric("Terms Searched", st.session_state.current_term_index + 1)
                    st.metric("Leads Found", len(st.session_state.search_results))

                with results_container:
                    st.subheader(f"Results for: {term}")
                    display_search_results(term_results, key_suffix=st.session_state.current_term_index)
            except Exception as e:
                st.error(f"Error occurred for term '{term}': {str(e)}")

            st.session_state.current_term_index += 1
            if st.session_state.current_term_index < len(terms):
                time.sleep(1)  # Add a small delay to avoid excessive reruns
                st.rerun()

        st.session_state.search_in_progress = False
        st.success(f"Search completed! Found {len(st.session_state.search_results)} leads across {len(terms)} terms.")

    if not st.session_state.get('search_in_progress', False) and st.session_state.get('search_results'):
        st.subheader("Overall Search Results")
        display_search_results(st.session_state.search_results, key_suffix="overall")

def display_search_results(results, key_suffix):
    if not results:
        st.warning("No results to display.")
        return

    for i, res in enumerate(results):
        with st.expander(f"Lead: {res['Email']}", expanded=False, key=f"lead_expander_{key_suffix}_{i}"):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"**URL:** [{res['URL']}]({res['URL']})")
                st.markdown(f"**Title:** {res['Title']}")
                st.markdown(f"**Description:** {res['Description']}")
            with col2:
                st.markdown(f"**Tags:** {', '.join(res['Tags'])}")
                st.markdown(f"**Lead Source:** {res['Lead Source']}")
                st.markdown(f"**Lead Email:** {res['Email']}")
def bulk_send_page():
    st.header("Bulk Send Emails")

    if not st.session_state.get('bulk_send_in_progress', False):
        with st.form("bulk_send_form"):
            col1, col2 = st.columns(2)
            with col1:
                with get_db_connection() as session:
                    templates = fetch_email_templates(session)
                if not templates:
                    st.error("No email templates found. Please create a template first.")
                    return
                template = st.selectbox("Select Email Template", options=templates)
                template_id = int(template.split(":")[0])
                from_email = st.text_input("From Email", value="Sami Halawa AI <hello@indosy.com>")
                reply_to = st.text_input("Reply To", value="eugproductions@gmail.com")

            with col2:
                send_option = st.radio(
                    "Send to:",
                    ["All Leads", "All Not Contacted with this Template", "All Not Contacted with Templates from this Campaign", "Leads from Selected Search Terms", "Leads from Selected Search Term Groups"]
                )
                if send_option in ["Leads from Selected Search Terms", "Leads from Selected Search Term Groups"]:
                    if send_option == "Leads from Selected Search Terms":
                        search_terms = fetch_search_terms_with_lead_count(session)
                        selected_items = st.multiselect(
                            "Select Search Terms",
                            options=search_terms,
                            format_func=lambda x: x.split(": ", 1)[1]
                        )
                    else:
                        search_term_groups = fetch_search_term_groups(session)
                        selected_items = st.multiselect(
                            "Select Search Term Groups",
                            options=search_term_groups
                        )
                filter_option = st.radio(
                    "Filter:",
                    ["Not Filter Out Leads", "Filter Out blog-directory"]
                )

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                preview_button = st.form_submit_button(label="Preview Email", help="Preview the email before sending.")
            with col2:
                send_button = st.form_submit_button(label="Start Bulk Send")

        if preview_button:
            with get_db_connection() as session:
                preview = get_email_preview(session, template_id, from_email, reply_to)
            st.subheader("Email Preview")
            st.components.v1.html(preview, height=400, scrolling=True)

        if send_button:
            st.session_state.bulk_send_in_progress = True
            st.session_state.bulk_send_params = {
                'template_id': template_id,
                'from_email': from_email,
                'reply_to': reply_to,
                'send_option': send_option,
                'filter_option': filter_option,
                'selected_items': selected_items if send_option in ["Leads from Selected Search Terms", "Leads from Selected Search Term Groups"] else None
            }
            st.session_state.bulk_send_results = []
            st.session_state.current_lead_index = 0

    if st.session_state.get('bulk_send_in_progress', False):
        st.subheader("Bulk Send Progress")

        col1, col2 = st.columns([2, 1])
        with col1:
            progress_bar = st.progress(0)
            status_text = st.empty()
        with col2:
            stats_container = st.container()

        email_list_container = st.container()

        with get_db_connection() as session:
            params = st.session_state.bulk_send_params
            if params['send_option'] == "Leads from Selected Search Terms":
                search_term_ids = [int(term.split(":")[0]) for term in params['selected_items']]
                leads_to_send = fetch_leads_for_search_terms(session, search_term_ids)
            elif params['send_option'] == "Leads from Selected Search Term Groups":
                leads_to_send = fetch_leads_for_search_term_groups(session, params['selected_items'])
            else:
                leads_to_send = fetch_leads_for_bulk_send(session, params['template_id'], params['send_option'], params['filter_option'])

            total_leads = len(leads_to_send)

            while st.session_state.current_lead_index < total_leads:
                lead = leads_to_send[st.session_state.current_lead_index]
                try:
                    result = send_email_to_lead(session, lead, params['template_id'], params['from_email'], params['reply_to'])
                    st.session_state.bulk_send_results.append(result)
                except Exception as e:
                    st.error(f"Error sending email to {lead[1]}: {str(e)}")

                st.session_state.current_lead_index += 1
                progress = st.session_state.current_lead_index / total_leads
                progress_bar.progress(progress)
                status_text.text(f"Sending email {st.session_state.current_lead_index} of {total_leads}")

                with stats_container:
                    st.metric("Emails Sent", st.session_state.current_lead_index)
                    st.metric("Success Rate", f"{(sum(1 for r in st.session_state.bulk_send_results if r['Status'] == 'sent') / len(st.session_state.bulk_send_results) * 100):.2f}%")

                with email_list_container:
                    st.subheader("Recently Sent Emails")
                    for result in st.session_state.bulk_send_results[-10:]:
                        st.markdown(f"**{result['Email']}** - Status: {result['Status']}")

                if st.session_state.current_lead_index % 10 == 0:  # Update every 10 leads
                    st.rerun()

            st.session_state.bulk_send_in_progress = False
            st.success(f"Bulk send completed. Sent {total_leads} emails.")

    if not st.session_state.get('bulk_send_in_progress', False) and st.session_state.get('bulk_send_results'):
        st.subheader("Bulk Send Results")
        display_bulk_send_results(st.session_state.bulk_send_results)

def display_bulk_send_results(results):
    total_sent = len(results)
    successful_sends = sum(1 for r in results if r['Status'] == 'sent')
    failed_sends = total_sent - successful_sends

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Emails Sent", total_sent)
    col2.metric("Successful Sends", successful_sends)
    col3.metric("Failed Sends", failed_sends)

    st.subheader("Detailed Results")
    df = pd.DataFrame(results)
    st.dataframe(df)

    if failed_sends > 0:
        st.subheader("Failed Sends")
        failed_df = df[df['Status'] != 'sent']
        st.dataframe(failed_df)

def view_leads_page():
    st.header("View Leads")
    with get_db_connection() as session:
        if st.button("Refresh Leads"):
            st.session_state.leads = fetch_leads(session)
        if 'leads' not in st.session_state:
            st.session_state.leads = fetch_leads(session)
        if st.session_state.leads.empty:
            st.info("No leads available.")
        else:
            st.dataframe(st.session_state.leads)
    session.close()

def search_terms_page():
    st.header("Search Terms")
    with get_db_connection() as session:
        with st.form("add_search_term_form"):
            search_term = st.text_input("Enter New Search Term")
            group = st.text_input("Group (optional)")
            submit_button = st.form_submit_button("Add Search Term")

        if submit_button:
            if search_term.strip():
                try:
                    add_search_term(session, search_term, group)
                    st.success(f"Search term '{search_term}' added to group '{group or 'Ungrouped'}'.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a valid search term.")

        view_option = st.radio("VIEW OPTIONS", ["Grouped", "Last Modified"])

        if view_option == "Grouped":
            display_grouped_terms(session)
        else:
            display_sorted_terms(session)

    session.close()

def display_grouped_terms(session):
    grouped_terms = fetch_grouped_search_terms(session)
    for group, terms in grouped_terms.items():
        total_leads = sum(term['lead_count'] for term in terms)
        with st.expander(f"{group.upper()} ({total_leads})"):
            cols = st.columns(2)
            for i, term in enumerate(terms):
                cols[i % 2].write(f"{term['term']} ({term['lead_count']})")

def display_sorted_terms(session):
    search_terms_df = fetch_search_terms_sorted(session)
    st.dataframe(search_terms_df)

def fetch_grouped_search_terms(session):
    try:
        query = (
            session.query(
                SearchTerm.id,
                SearchTerm.term,
                func.coalesce(SearchTerm.group, 'Ungrouped').label('group'),
                func.count(CampaignLead.id).label('lead_count')
            )
            .outerjoin(CampaignLead, SearchTerm.id == CampaignLead.search_term_id)
            .group_by(SearchTerm.id, SearchTerm.term, SearchTerm.group)
            .order_by('group', SearchTerm.term)
        )
        result = query.all()

        grouped_terms = {}
        for row in result:
            group = row.group
            if group not in grouped_terms:
                grouped_terms[group] = []
            grouped_terms[group].append({
                'id': row.id,
                'term': row.term,
                'lead_count': row.lead_count
            })

        return grouped_terms
    except Exception as e:
        logging.error(f"Error in fetch_grouped_search_terms: {str(e)}")
        return {}

def generate_search_term_groups(kb_info, user_input):
    prompt = f"""
    Based on the following knowledge base information and user input, generate multiple search term groups 
    to target different types of potential leads. Each group should have a descriptive name and 5-10 search terms.

    Knowledge Base:
    {json.dumps(kb_info, indent=2)}

    User Input:
    {user_input}

    Respond with a JSON object where keys are group names and values are lists of search terms.
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant that generates optimized search term groups for lead generation."},
        {"role": "user", "content": prompt}
    ]
    return openai_chat_completion(messages, function_name="generate_search_term_groups") or {}

def fetch_search_terms_sorted(session):
    try:
        # Log the start of the function
        logging.info("Starting fetch_search_terms_sorted")

        query = (
            session.query(
                SearchTerm.id,
                SearchTerm.term,
                func.coalesce(SearchTerm.group, 'Ungrouped').label('group'),
                func.count(CampaignLead.id).label('lead_count'),
                func.max(CampaignLead.created_at).label('last_lead_added')
            )
            .outerjoin(CampaignLead, SearchTerm.id == CampaignLead.search_term_id)
            .group_by(SearchTerm.id, SearchTerm.term, SearchTerm.group)
            .order_by(func.max(CampaignLead.created_at).desc().nullslast(), func.count(CampaignLead.id).desc())
        )

        # Log the SQL query
        logging.info(f"SQL Query: {query}")

        result = query.all()

        # Log the number of results
        logging.info(f"Number of results from query: {len(result)}")

        df = pd.DataFrame([(r.id, r.term, r.group, r.lead_count, r.last_lead_added) for r in result],
                          columns=['ID', 'Term', 'Group', 'Lead Count', 'Last Lead Added'])

        # Add debugging information
        logging.info(f"Number of search terms fetched: {len(df)}")
        logging.info(f"Columns in DataFrame: {df.columns}")
        logging.info(f"First few rows of DataFrame:\n{df.head().to_string()}")

        # Check for any NaN or null values
        if df.isnull().values.any():
            logging.warning("DataFrame contains NaN or null values")
            logging.info(f"Columns with null values: {df.columns[df.isnull().any()].tolist()}")

        return df
    except Exception as e:
        logging.error(f"Error in fetch_search_terms_sorted: {str(e)}")
        logging.exception("Full traceback:")
        return pd.DataFrame()

def add_search_term(session, search_term, group=None):
    try:
        logging.info(f"Adding search term: {search_term}, group: {group}")
        new_term = SearchTerm(term=search_term, group=group)
        session.add(new_term)
        session.commit()
        logging.info(f"Successfully added search term: {search_term}")
    except Exception as e:
        logging.error(f"Error in add_search_term: {str(e)}")
        session.rollback()
        raise

def fetch_search_term_groups(session):
    try:
        logging.info("Fetching search term groups")
        query = session.query(func.coalesce(SearchTerm.group, 'Ungrouped')).distinct()
        groups = [group[0] for group in query.all()]
        logging.info(f"Fetched groups: {groups}")
        return groups
    except Exception as e:
        logging.error(f"Error in fetch_search_term_groups: {str(e)}")
        return []

def fetch_leads_for_search_term_groups(session, groups):
    try:
        logging.info(f"Fetching leads for groups: {groups}")
        query = (
            session.query(Lead)
            .join(CampaignLead)
            .join(SearchTerm)
            .filter(func.coalesce(SearchTerm.group, 'Ungrouped').in_(groups))
        )
        leads = query.all()
        logging.info(f"Number of leads fetched: {len(leads)}")
        return leads
    except Exception as e:
        logging.error(f"Error in fetch_leads_for_search_term_groups: {str(e)}")
        return []

def email_templates_page():
    st.header("Email Templates")
    with get_db_connection() as session:

        with st.form("add_email_template_form"):
            template_name = st.text_input("Template Name")
        subject = st.text_input("Subject")
        body_content = st.text_area("Body Content (HTML)", height=400)
        submit = st.form_submit_button("Add Email Template")

    if submit:
        if not template_name.strip() or not subject.strip() or not body_content.strip():
            st.warning("Please fill in all fields.")
        else:
            try:
                template_id = create_email_template(session, template_name, subject, body_content)
                st.success(f"Email template '{template_name}' added with ID: {template_id}.")
            except Exception as e:
                st.error(f"Fatal error: {str(e)}")

    st.subheader("Manage Email Templates")

    with st.form("manage_email_template_form"):
        template_name = st.text_input("Template Name")
        load_button = st.form_submit_button("Load Template")

    if load_button:
        try:
            template = get_email_template_by_name(session, template_name)
            if template:
                # Add code here to display or edit the loaded template
                st.write("Template loaded successfully.")
            else:
                st.warning("Template not found.")
        except Exception as e:
            st.error(f"Fatal error: {str(e)}")
# AI Functions

def get_knowledge_base_info(session, project_id=None):
    if not project_id:
        project_id = get_active_project_id()
    try:
        kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        if not kb:
            return {}
        return {
            "kb_name": kb.kb_name,
            "kb_bio": kb.kb_bio,
            "kb_values": kb.kb_values,
            "contact_name": kb.contact_name,
            "contact_role": kb.contact_role,
            "contact_email": kb.contact_email,
            "company_description": kb.company_description,
            "company_mission": kb.company_mission,
            "company_target_market": kb.company_target_market,
            "company_other": kb.company_other,
            "product_name": kb.product_name,
            "product_description": kb.product_description,
            "product_target_customer": kb.product_target_customer,
            "product_other": kb.product_other,
            "other_context": kb.other_context,
            "example_email": kb.example_email
        }
    except Exception as e:
        st.error(f"Fatal error: {str(e)}")
        return {}

def classify_search_terms(session, search_terms, kb_info):
    prompt = f"""
    Classify the following search terms into strategic groups based on their relevance and potential for lead generation: {', '.join(search_terms)}. 

    Knowledge Base Info:
    {json.dumps(kb_info)}

    Create groups that allow for tailored, personalized email content. Consider the product/service features, target audience, and potential customer pain points. Ensure groups are specific enough for customization but broad enough for efficiency. Include a 'low_quality_search_terms' category for irrelevant or overly broad terms.

    Respond with a JSON object in the following format:
    {{
        "group_name_1": ["term1", "term2", "term3"],
        "group_name_2": ["term4", "term5", "term6"],
        "low_quality_search_terms": ["term7", "term8"]
    }}
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in lead generation."},
        {"role": "user", "content": prompt}
    ]
    return openai_chat_completion(messages) or {}

def generate_email_template(session, terms, kb_info):
    prompt = f"""
    Create an email template for the following search terms: {', '.join(terms)}.

    Knowledge Base Info: 
    {json.dumps(kb_info)}

    Guidelines: Focus on benefits to the reader, address potential customer doubts, 
    include clear CTAs, use a natural tone, and be concise. 

    Respond with a JSON object in the format: 
    {{
        "subject": "Subject line here",
        "body": "HTML body content here"
    }}
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in creating high-converting email templates for targeted marketing campaigns."},
        {"role": "user", "content": prompt}
    ]
    return openai_chat_completion(messages) or {"subject": "", "body": ""}

def adjust_email_template_api(session, current_template, adjustment_prompt, kb_info):
    prompt = f"""
    Adjust the following email template based on the given instructions:

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
    }}
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in refining high-converting email templates for targeted marketing campaigns."},
        {"role": "user", "content": prompt}
    ]
    return openai_chat_completion(messages) or {"subject": "", "body": ""}

def generate_optimized_search_terms(session, current_terms, kb_info):
    prompt = f"""
    Optimize the following search terms for targeted email campaigns:

    Current Terms: {', '.join(current_terms)}

    Knowledge Base Info:
    {json.dumps(kb_info)}

    Guidelines:
    1. Focus on terms likely to attract high-quality leads
    2. Consider product/service features, target audience, and customer pain points
    3. Optimize for specificity and relevance
    4. Think about how each term could lead to a compelling email strategy
    5. Remove or improve low-quality or overly broad terms
    6. Add new, highly relevant terms based on the knowledge base information

    Provide a list of optimized search terms, aiming for quality over quantity.

    Respond with a JSON object in the following format:
    {{
        "optimized_terms": ["term1", "term2", "term3", "term4", "term5"]
    }}
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in optimizing search terms for targeted email marketing campaigns."},
        {"role": "user", "content": prompt}
    ]
    response = openai_chat_completion(messages)
    return response.get("optimized_terms", []) if response else []

def fetch_search_term_groups(session):
    groups = session.query(SearchTermGroup).all()
    return [f"{group.id}:{group.name}" for group in groups]

def get_group_data(session, group_id):
    if not group_id:
        return None
    group_id = int(group_id.split(':')[0])
    group = session.query(SearchTermGroup).filter_by(id=group_id).first()
    if not group:
        return None
    search_terms = [term.term for term in group.search_terms]
    return {
        "name": group.name,
        "description": group.description,
        "email_template": group.email_template,
        "search_terms": search_terms
    }

def save_optimized_group(session, group_id, optimized_data):
    group_id = int(group_id.split(':')[0])
    group = session.query(SearchTermGroup).filter_by(id=group_id).first()
    if not group:
        return
    # Delete existing terms for this group
    session.query(SearchTerm).filter_by(group_id=group_id).delete()
    # Insert new terms
    for category, terms in optimized_data.items():
        for term in terms:
            new_term = SearchTerm(term=term, group_id=group_id, category=category)
            session.add(new_term)
    session.commit()

def fetch_all_search_terms(session):
    terms = session.query(SearchTerm).all()
    return [term.term for term in terms]

def save_new_group(session, category, terms, email_template):
    new_group = SearchTermGroup(name=category, email_template=email_template)
    session.add(new_group)
    session.commit()
    for term in terms:
        new_term = SearchTerm(term=term, group_id=new_group.id, category=category)
        session.add(new_term)
    session.commit()

def save_adjusted_template(session, group_id, adjusted_template):
    group_id = int(group_id.split(':')[0])
    group = session.query(SearchTermGroup).filter_by(id=group_id).first()
    if not group:
        return
    group.email_template = adjusted_template
    session.commit()

def save_optimized_search_terms(session, optimized_terms):
    campaign_id = get_active_campaign_id()
    for term in optimized_terms:
        new_term = SearchTerm(term=term, campaign_id=campaign_id)
        session.add(new_term)
    session.commit()

def fetch_ai_request_logs(session):
    logs = session.query(AIRequestLog).order_by(AIRequestLog.created_at.desc()).limit(10).all()
    return [
        {
            "function_name": log.function_name,
            "prompt": log.prompt,
            "response": log.response,
            "created_at": log.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for log in logs
    ]

def get_email_template_by_name(session, template_name):
    return session.query(EmailTemplate).filter_by(template_name=template_name).first()

# --------------------- Page Functions ---------------------

def get_least_searched_terms(session, n):
    terms = session.query(SearchTerm).order_by(SearchTerm.id).limit(n).all()
    return [term.term for term in terms]

def get_ineffective_search_terms(session, threshold=0.3):
    ineffective_terms = session.query(SearchTermEffectiveness).filter(
        (SearchTermEffectiveness.valid_leads / SearchTermEffectiveness.total_results) < threshold
    ).all()
    return [term.term for term in ineffective_terms]

def count_total_leads(session):
    return session.query(Lead).count()

def count_leads_last_24_hours(session):
    return session.query(Lead).filter(Lead.created_at >= datetime.utcnow() - timedelta(days=1)).count()

def count_emails_sent(session):
    return session.query(EmailCampaign).filter_by(status='sent').count()

def count_optimized_search_terms(session):
    return session.query(OptimizedSearchTerm).count()

def display_real_time_analytics(session):
    total_leads = count_total_leads(session)
    leads_last_24h = count_leads_last_24_hours(session)
    emails_sent = count_emails_sent(session)
    optimized_terms = count_optimized_search_terms(session)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Leads", total_leads)
    col2.metric("Leads in Last 24 Hours", leads_last_24h)
    col3.metric("Emails Sent", emails_sent)
    col4.metric("Optimized Terms", optimized_terms)

def display_result_card(result):
    st.markdown(f"""
- **Email:** {result['Email']}
  **URL:** [{result['URL']}]({result['URL']})
  **Title:** {result['Title']}
  **Description:** {result['Description']}
  **Tags:** {', '.join(result['Tags'])}
""")
    st.write("---")  # Add a separator for better UI
def get_email_preview(session, template_id, from_email, reply_to):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        st.error("Template not found.")
        return "<p>Template not found</p>"
    return f"""
    <h3>Email Preview</h3>
    <strong>From:</strong> {from_email}<br>
    <strong>Reply-To:</strong> {reply_to}<br>
    <strong>Subject:</strong> {template.subject}<br>
    <strong>From:</strong> {from_email}<br>
    <strong>Reply-To:</strong> {reply_to}<br>
    <hr>
    <h4>Body:</h4>
    <div>{template.body_content}</div>
    """

async def bulk_send_coroutine(session, template_id, from_email, reply_to, leads):
    logs = []
    all_results = []
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        logs.append("Template not found.")
        return logs, all_results
    total_leads = len(leads)
    progress = st.progress(0)
    for index, (lead_id, email) in enumerate(leads, start=1):
        logs.append(f"Processing lead {lead_id} with email {email}...")
        try:
            sent_at = datetime.utcnow()
            message_id = f"msg-{lead_id}-{int(time.time())}"
            status = 'sent'
            customized_content = template.body_content

            # Send email using AWS SES
            ses_response = send_email_ses(from_email, email, template.subject, customized_content)

            await save_email_campaign_async(session, lead_id, template_id, status, sent_at, template.subject, message_id, customized_content)
            logs.append(f"[{index}/{total_leads}] Sent email to {email} - Message ID: {message_id}")
            all_results.append({"Email": email, "Status": status, "Message ID": message_id})
            await asyncio.sleep(0.1)
            progress.progress(index / total_leads)
        except Exception as e:
            save_email_campaign(session, lead_id, template_id, 'failed', None, template.subject, None, str(e))
            logs.append(f"[{index}/{total_leads}] Failed to send email to {email}: {e}")
            all_results.append({"Email": email, "Status": 'failed', "Message ID": None})
    return logs, all_results

async def continuous_automation_process(session, log_container, leads_container):
    emails_found = []
    while st.session_state.get('automation_status', False):
        try:
            search_terms = fetch_all_search_terms(session)
            if not search_terms or len(search_terms) == 0:
                st.warning("No search terms available.")
                return

            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                st.warning("Knowledge Base not found for the active project.")
                return

            classified_terms = classify_search_terms(session, search_terms, kb_info)
            update_log_display(log_container, "automation_logs")
            log_container.info(f"Classified search terms into {len(classified_terms)} groups.")

            reference_template = get_email_template_by_name(session, "REFERENCE")
            if not reference_template:
                st.warning("REFERENCE email template not found.")
                return

            for group_name, terms in classified_terms.items():
                if group_name == "low_quality_search_terms":
                    continue

                adjustment_prompt = f"Adjust the email template to better appeal to recipients found using the search terms: {', '.join(terms)}"
                adjusted_template_data = adjust_email_template_api(session, reference_template.body_content, adjustment_prompt, kb_info)
                if not adjusted_template_data or not adjusted_template_data.get('body'):
                    st.warning(f"Failed to adjust email template for group '{group_name}'.")
                    continue

                adjusted_template_name = f"{reference_template.template_name}_{group_name}"
                new_template_id = create_email_template(session, adjusted_template_name, adjusted_template_data['subject'], adjusted_template_data['body'])
                update_log_display(log_container, "automation_logs")
                log_container.info(f"Created template '{reference_template.template_name}_{group_name}' for '{group_name}'.")

                results = manual_search(session, terms, 10, "All Leads", log_container, leads_container)
                new_emails = [res['Email'] for res in results if res['Email'] not in emails_found]
                emails_found.extend(new_emails)

                leads_container.text_area(
                    "Emails Found",
                    "\n".join(emails_found[-50:]),  # Display last 50 emails
                    height=200,
                    key=f"emails_found_{len(emails_found)}_{len(emails_found)}"  # Use current counts
                )

                lead_ids = [res['Lead ID'] for res in results]
                leads_to_send = [(lead_id, email) for lead_id, email in zip(lead_ids, new_emails)]

                from_email = kb_info.get('contact_email', 'hello@indosy.com')
                reply_to = kb_info.get('contact_email', 'eugproductions@gmail.com')
                logs, _ = await bulk_send_coroutine(session, new_template_id, from_email, reply_to, leads_to_send, log_container)
                st.session_state.automation_logs.extend(logs)
                update_log_display(log_container, "automation_logs")

            await asyncio.sleep(60)

        except Exception as e:
            update_log_display(log_container, "automation_logs")
            log_container.error(f"Automation error: {e}")
            break

def update_log_display(log_container, log_key):
    if log_key not in st.session_state:
        st.session_state[log_key] = []
    log_container.markdown(f"""
    <div style="height:200px;overflow:auto;">
    {'<br>'.join(st.session_state[log_key][-20:])}
    </div>
    """, unsafe_allow_html=True)

def display_search_results(results, key_suffix):
    if not results:
        st.warning("No results to display.")
        return
    with st.expander("Search Results", expanded=True):
        st.markdown(f"### Total Leads Found: **{len(results)}**")
    for i, res in enumerate(results):
        with st.expander(f"Lead: {res['Email']}", key=f"lead_expander_{key_suffix}_{i}"):
            st.markdown(f"**URL:** [{res['URL']}]({res['URL']})  \n"
                        f"**Title:** {res['Title']}  \n"
                        f"**Description:** {res['Description']}  \n"
                        f"**Tags:** {', '.join(res['Tags'])}  \n"
                        f"**Lead Source:** {res['Lead Source']}  \n"
                        f"**Lead Email:** {res['Email']}")

def perform_quick_scan(session):
    terms = get_least_searched_terms(session, 3)
    results = manual_search(session, terms, 10, "All Leads")
    return {"new_leads": len(results)}

def fetch_search_terms_with_lead_count(session, campaign_id=None):
    try:
        if campaign_id:
            query = session.query(SearchTerm, func.count(LeadSource.id).label('lead_count')).\
                outerjoin(LeadSource).\
                filter(SearchTerm.campaign_id == campaign_id).\
                group_by(SearchTerm.id)
        else:
            query = session.query(SearchTerm, func.count(LeadSource.id).label('lead_count')).\
                outerjoin(LeadSource).\
                group_by(SearchTerm.id)

        terms = query.all()
        return [f"{term.SearchTerm.id}: {term.SearchTerm.term} ({term.lead_count} leads)" for term in terms]
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return []

def fetch_leads_for_search_terms(session, search_term_ids):
    try:
        leads = session.query(Lead).distinct().\
            join(LeadSource).\
            filter(LeadSource.search_term_id.in_(search_term_ids)).\
            all()
        return [(lead.id, lead.email) for lead in leads]
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return []
def projects_campaigns_page():
    with get_db_connection() as session:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Add Project")
            with st.form("add_project_form"):
                project_name = st.text_input("Project Name")
                submit = st.form_submit_button("Add Project")
            if submit:
                if not project_name.strip():
                    st.warning("Please enter a valid project name.")
                else:
                    try:
                        new_project = Project(project_name=project_name)
                        session.add(new_project)
                        session.commit()
                        st.success(f"Project '{project_name}' added.")
                    except SQLAlchemyError as e:
                        st.error(f"Database error: {str(e)}")

        with col2:
            st.subheader("Add Campaign")
            with st.form("add_campaign_form"):
                campaign_name = st.text_input("Campaign Name")
                campaign_type = st.selectbox("Campaign Type", ["Email", "SMS"])
                project_options = fetch_projects(session)
                selected_project = st.selectbox("Select Project", options=project_options)
                submit = st.form_submit_button("Add Campaign")
            if submit:
                if not campaign_name.strip():
                    st.warning("Please enter a valid campaign name.")
                else:
                    try:
                        project_id = int(selected_project.split(":")[0])
                        new_campaign = Campaign(
                            campaign_name=campaign_name,
                            campaign_type=campaign_type,
                            project_id=project_id
                        )
                        session.add(new_campaign)
                        session.commit()
                        st.success(f"Campaign '{campaign_name}' added to project '{selected_project}'.", icon="✅")
                    except SQLAlchemyError as e:
                        st.error(f"Database error: {str(e)}")

        st.subheader("Existing Projects")
        try:
            projects = session.query(Project).all()
            for project in projects:
                st.write(f"- **{project.project_name}** (ID: {project.id})")
        except SQLAlchemyError as e:
            st.error(f"Database error: {str(e)}")

        st.subheader("Existing Campaigns")
        try:
            campaigns = session.query(Campaign).all()
            for campaign in campaigns:
                st.write(f"- **{campaign.campaign_name}** (ID: {campaign.id}, Type: {campaign.campaign_type})")
        except SQLAlchemyError as e:
            st.error(f"Database error: {str(e)}")

def knowledge_base_page():
    with get_db_connection() as session:
        project_options = fetch_projects(session)

    if not project_options:
        st.warning("No projects found. Please create a project first.")
        return

    selected_project = st.selectbox("Select Project", options=project_options)
    project_id = int(selected_project.split(":")[0])
    set_active_project_id(project_id)

    with get_db_connection() as session:
        kb_entry = session.query(KnowledgeBase).filter_by(project_id=project_id).first()

    with st.form("knowledge_base_form"):
        kb_name = st.text_input("Knowledge Base Name", value=kb_entry.kb_name if kb_entry else "")
        kb_bio = st.text_area("Bio", value=kb_entry.kb_bio if kb_entry else "")
        kb_values = st.text_area("Values", value=kb_entry.kb_values if kb_entry else "")
        contact_name = st.text_input("Contact Name", value=kb_entry.contact_name if kb_entry else "")
        contact_role = st.text_input("Contact Role", value=kb_entry.contact_role if kb_entry else "")
        contact_email = st.text_input("Contact Email", value=kb_entry.contact_email if kb_entry else "")
        company_description = st.text_area("Company Description", value=kb_entry.company_description if kb_entry else "")
        company_mission = st.text_area("Company Mission/Vision", value=kb_entry.company_mission if kb_entry else "")
        company_target_market = st.text_area("Company Target Market", value=kb_entry.company_target_market if kb_entry else "")
        company_other = st.text_area("Company Other", value=kb_entry.company_other if kb_entry else "")
        product_name = st.text_input("Product Name", value=kb_entry.product_name if kb_entry else "")
        product_description = st.text_area("Product Description", value=kb_entry.product_description if kb_entry else "")
        product_target_customer = st.text_area("Product Target Customer", value=kb_entry.product_target_customer if kb_entry else "")
        product_other = st.text_area("Product Other", value=kb_entry.product_other if kb_entry else "")
        other_context = st.text_area("Other Context", value=kb_entry.other_context if kb_entry else "")
        example_email = st.text_area("Example Email", value=kb_entry.example_email if kb_entry else "")
        submit = st.form_submit_button("Save Knowledge Base")

    if submit:
        try:
            with get_db_connection() as session:
                if kb_entry:
                    kb_entry.kb_name = kb_name
                    kb_entry.kb_bio = kb_bio
                    kb_entry.kb_values = kb_values
                    kb_entry.contact_name = contact_name
                    kb_entry.contact_role = contact_role
                    kb_entry.contact_email = contact_email
                    kb_entry.company_description = company_description
                    kb_entry.company_mission = company_mission
                    kb_entry.company_target_market = company_target_market
                    kb_entry.company_other = company_other
                    kb_entry.product_name = product_name
                    kb_entry.product_description = product_description
                    kb_entry.product_target_customer = product_target_customer
                    kb_entry.product_other = product_other
                    kb_entry.other_context = other_context
                    kb_entry.example_email = example_email
                else:
                    new_kb = KnowledgeBase(
                        project_id=project_id,
                        kb_name=kb_name,
                        kb_bio=kb_bio,
                        kb_values=kb_values,
                        contact_name=contact_name,
                        contact_role=contact_role,
                        contact_email=contact_email,
                        company_description=company_description,
                        company_mission=company_mission,
                        company_target_market=company_target_market,
                        company_other=company_other,
                        product_name=product_name,
                        product_description=product_description,
                        product_target_customer=product_target_customer,
                        product_other=product_other,
                        other_context=other_context,
                        example_email=example_email
                    )
                    session.add(new_kb)
                session.commit()
            st.success("Knowledge Base saved successfully!", icon="✅")
        except Exception as e:
            st.error(f"An error occurred while saving the Knowledge Base: {str(e)}")

def autoclient_ai_page():
    st.header("AutoclientAI - Automated Lead Generation")

    with st.expander("Knowledge Base Information", expanded=False):
        with get_db_connection() as session:
            kb_info = get_knowledge_base_info(session, get_active_project_id())
        if not kb_info:
            st.error("Knowledge Base not found for the active project. Please set it up first.")
            return
        st.json(kb_info)

    user_input = st.text_area("Enter additional context or specific goals for lead generation:", 
                              help="This information will be used to generate more targeted search terms.")

    if st.button("Generate Search Term Groups and Email Templates", key="generate_groups_and_templates"):
        with st.spinner("Generating search term groups and email templates..."):
            search_term_groups, email_templates = generate_search_term_groups_and_templates(kb_info, user_input)
            
            if not search_term_groups:
                st.error("Failed to generate search term groups and email templates. Please try again.")
                return

            st.session_state.search_term_groups = search_term_groups
            st.session_state.email_templates = email_templates

        st.success("Search term groups and email templates generated successfully!")

        # Display generated search terms and email templates
        st.subheader("Generated Search Term Groups and Email Templates")
        for group_name, terms in search_term_groups.items():
            with st.expander(f"Group: {group_name}"):
                st.write("**Search Terms:**")
                st.write(", ".join(terms))
                st.write("**Email Template:**")
                st.write(f"Subject: {email_templates[group_name]['subject']}")
                st.text_area("Body:", email_templates[group_name]['body'], height=200)

    if st.button("Start Automation", key="start_automation"):
        st.session_state.automation_status = True
        st.session_state.automation_logs = []
        st.session_state.total_leads_found = 0
        st.session_state.total_emails_sent = 0
        st.success("Automation started!")

    if st.session_state.get('automation_status', False):
        st.subheader("Automation in Progress")
        
        progress_bar = st.progress(0)
        log_container = st.empty()
        leads_container = st.empty()
        analytics_container = st.empty()

        try:
            with get_db_connection() as session:
                for idx, (group_name, terms) in enumerate(st.session_state.search_term_groups.items()):
                    st.subheader(f"Processing Group: {group_name}")
                    st.write("Search Terms:", ", ".join(terms))
                    
                    results = manual_search(session, terms, 10, "All Leads")

                    new_leads = []
                    for res in results:
                        if save_lead(session, res['Email'], domain=get_domain_from_url(res['URL'])):
                            new_leads.append((res['Lead ID'], res['Email']))

                    st.session_state.total_leads_found += len(new_leads)

                    if new_leads:
                        template = st.session_state.email_templates[group_name]
                        from_email = kb_info.get('contact_email', 'hello@indosy.com')
                        reply_to = kb_info.get('contact_email', 'eugproductions@gmail.com')
                        logs, sent_count = bulk_send_coroutine(session, template, from_email, reply_to, new_leads)
                        st.session_state.automation_logs.extend(logs)
                        st.session_state.total_emails_sent += sent_count

                    leads_container.text_area("New Leads Found", "\n".join([email for _, email in new_leads]), height=200)
                    progress_bar.progress((idx + 1) / len(st.session_state.search_term_groups))
                    
                    analytics_container.metric("Total Leads Found", st.session_state.total_leads_found)
                    analytics_container.metric("Total Emails Sent", st.session_state.total_emails_sent)

                    update_log_display(log_container, "automation_logs")

            st.success(f"Automation completed. Total leads found: {st.session_state.total_leads_found}, Total emails sent: {st.session_state.total_emails_sent}")
            st.session_state.automation_status = False

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

    # Debug information
    st.subheader("Debug Information")
    st.json(st.session_state)
    st.write("Current function:", autoclient_ai_page.__name__)
    st.write("Session state keys:", list(st.session_state.keys()))


def optimize_existing_groups(kb_info):
    try:
        with get_db_connection() as session:
            search_terms = session.query(SearchTerm).all()
            terms = [term.term for term in search_terms]
            if not terms:
                st.info("No search terms available to optimize.")
                return
            classification = classify_search_terms(terms, kb_info)
            st.write("**Optimized Search Term Groups:**")
            st.json(classification)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
    try:
        # First attempt: with JSON response format
        response = client.chat.completions.create(
            model=openai.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        
        # Log the successful request
        with get_db_connection() as session:
            log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id)
        
        return result
    except OpenAIError as e:
        if "Invalid parameter: 'response_format'" in str(e):
            # Second attempt: without JSON response format
            try:
                response = client.chat.completions.create(
                    model=openai.model,
                    messages=messages,
                    temperature=temperature
                )
                content = response.choices[0].message.content
                try:
                    result = json.loads(content)
                    
                    # Log the successful request (second attempt)
                    with get_db_connection() as session:
                        log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id)
                    
                    return result
                except json.JSONDecodeError:
                    st.error(f"Failed to parse JSON from API response: {content}")
                    
                    # Log the failed parsing
                    with get_db_connection() as session:
                        log_ai_request(session, function_name, messages, content, lead_id, email_campaign_id)
                    
                    return None
            except Exception as inner_e:
                st.error(f"Error in OpenAI API call (second attempt): {str(inner_e)}")
                
                # Log the failed request (second attempt)
                with get_db_connection() as session:
                    log_ai_request(session, function_name, messages, str(inner_e), lead_id, email_campaign_id)
                
                return None
        else:
            st.error(f"Error in OpenAI API call: {str(e)}")
            
            # Log the failed request
            with get_db_connection() as session:
                log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id)
            
            return None
    except Exception as e:
        st.error(f"Unexpected error in OpenAI API call: {str(e)}")
        
        # Log the unexpected error
        with get_db_connection() as session:
            log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id)
        
        return None


def generate_optimized_search_terms(session, current_terms, kb_info):
    prompt = f"""
    Optimize the following search terms for targeted email campaigns:

    Current Terms: {', '.join(current_terms)}

    Knowledge Base Info:
    {json.dumps(kb_info)}

    Guidelines:
    1. Focus on terms likely to attract high-quality leads
    2. Consider product/service features, target audience, and customer pain points
    3. Optimize for specificity and relevance
    4. Think about how each term could lead to a compelling email strategy
    5. Remove or improve low-quality or overly broad terms
    6. Add new, highly relevant terms based on the knowledge base information

    Provide a list of optimized search terms, aiming for quality over quantity.

    Respond with a JSON object in the following format:
    {{
        "optimized_terms": ["term1", "term2", "term3", "term4", "term5"]
    }}
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in optimizing search terms for targeted email marketing campaigns."},
        {"role": "user", "content": prompt}
    ]
    response = openai_chat_completion(messages)
    return response.get("optimized_terms", []) if response else []

def generate_search_term_groups_and_templates(kb_info, user_input):
    prompt = f"""
    Based on the following knowledge base information and user input, generate 5 groups of search terms for lead generation, along with corresponding email templates for each group.

    Knowledge Base Info:
    {json.dumps(kb_info)}

    User Input:
    {user_input}

    For each group, provide:
    1. A group name
    2. 5-10 relevant search terms
    3. An email template with a subject line and body

    Respond with a JSON object in the following format:
    {{
        "group_name_1": {{
            "terms": ["term1", "term2", "term3", "term4", "term5"],
            "email_template": {{
                "subject": "Subject line for group 1",
                "body": "Email body for group 1"
            }}
        }},
        "group_name_2": {{
            "terms": ["term6", "term7", "term8", "term9", "term10"],
            "email_template": {{
                "subject": "Subject line for group 2",
                "body": "Email body for group 2"
            }}
        }},
        ...
    }}
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant that generates optimized search term groups and email templates for lead generation."},
        {"role": "user", "content": prompt}
    ]
    response = openai_chat_completion(messages, function_name="generate_search_term_groups_and_templates")
    if response:
        search_term_groups = {group: data['terms'] for group, data in response.items()}
        email_templates = {group: data['email_template'] for group, data in response.items()}
        return search_term_groups, email_templates
    return {}, {}



def classify_search_terms(session, search_terms, kb_info):
    prompt = f"""
    Classify the following search terms into strategic groups based on their relevance and potential for lead generation: {', '.join(search_terms)}. 

    Knowledge Base Info:
    {json.dumps(kb_info)}

    Create groups that allow for tailored, personalized email content. Consider the product/service features, target audience, and potential customer pain points. Ensure groups are specific enough for customization but broad enough for efficiency. Include a 'low_quality_search_terms' category for irrelevant or overly broad terms.

    Respond with a JSON object in the following format:
    {{
        "group_name_1": ["term1", "term2", "term3"],
        "group_name_2": ["term4", "term5", "term6"],
        "low_quality_search_terms": ["term7", "term8"]
    }}
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in lead generation."},
        {"role": "user", "content": prompt}
    ]
    return openai_chat_completion(messages) or {}

def adjust_email_template_api(session, template_content, adjustment_prompt, kb_info):
    prompt = f"Adjust this email template: {template_content}\n\nBased on this knowledge base: {json.dumps(kb_info)}\n\nAdjustment instructions: {adjustment_prompt}\n\nRespond with a JSON object containing 'subject' and 'body' keys."
    messages = [
        {"role": "system", "content": "You are an AI assistant that adjusts email templates for targeted marketing campaigns."},
        {"role": "user", "content": prompt}
    ]
    response = openai_chat_completion(messages)
    return json.loads(response)

import asyncio
import time
from tenacity import retry, stop_after_attempt, wait_exponential

def get_domain_from_url(url):
    return url.split('//')[-1].split('/')[0]

def save_lead(session, email, domain):
    existing_lead = session.query(Lead).filter_by(email=email).first()
    if existing_lead:
        return False
    new_lead = Lead(email=email, domain=domain)
    session.add(new_lead)
    session.commit()
    return True

def update_log_display(log_container, log_key):
    log_container.text_area("Automation Logs", "\n".join(st.session_state.get(log_key, [])), height=200)

def display_real_time_analytics(session):
    total_leads = session.query(Lead).count()
    total_emails_sent = session.query(EmailCampaign).count()
    st.write(f"Total Leads: {total_leads}")
    st.write(f"Total Emails Sent: {total_emails_sent}")

def perform_quick_scan(session): 
    new_leads = 0  # Replace with actual count of new leads found
    return {"new_leads": new_leads}

def create_new_groups(kb_info):
    try:
        with get_db_connection() as session:
            all_terms = session.query(SearchTerm).all()
        terms = [term.term for term in all_terms]
        if not terms:
            st.info("No search terms available to create groups.")
            return
        classification = classify_search_terms(terms, kb_info)
        st.write("**Created New Search Term Groups:**")
        st.json(classification)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def adjust_email_template(kb_info):
    with get_db_connection() as session:
        templates = fetch_email_templates(session)
    if not templates:
        st.info("No email templates available to adjust.")
        return
    template = st.selectbox("Select Template to Adjust", options=templates)
    template_id = int(template.split(":")[0])
    with st.form("adjust_template_form"):
        adjustment_prompt = st.text_area("Enter Adjustment Instructions")
        submit = st.form_submit_button("Adjust Template")
    if submit:
        try:
            with get_db_connection() as session:
                current_template = session.query(EmailTemplate).filter_by(id=template_id).first()
                if not current_template:
                    st.warning("Selected template not found.")
                    return
                adjusted_content = adjust_email_template_api(
                    current_template.body_content,
                    adjustment_prompt,
                    kb_info
                )
                current_template.body_content = adjusted_content
                session.commit()
            st.success("Email template adjusted successfully.")
        except Exception as e:
            st.error(f"An error occurred while adjusting the template: {str(e)}")

def optimize_search_terms_ai(kb_info):
    with get_db_connection() as session:
        current_terms = [term.term for term in session.query(SearchTerm).all()]
    if not current_terms:
        st.info("No search terms available to optimize.")
        return
    optimized_terms = generate_optimized_search_terms(current_terms, kb_info)
    st.write("**Optimized Search Terms:**")
    st.write("\n".join(optimized_terms))
    if st.button("Save Optimized Search Terms"):
        try:
            with get_db_connection() as session:
                save_optimized_search_terms(session, optimized_terms)
            st.success("Optimized search terms saved successfully.")
        except Exception as e:
            st.error(f"An error occurred while saving optimized search terms: {str(e)}")

async def bulk_send_coroutine(session, template_id, from_email, reply_to, leads, log_container):
    logs = []
    for lead_id, email in leads:
        try:
            template = session.query(EmailTemplate).filter_by(id=template_id).first()
            if not template:
                raise ValueError(f"Email template with id {template_id} not found")
            email_content = template.content
            send_email_ses(from_email, email, template.subject, email_content)
            logs.append(f"Email sent to: {email}")
        except Exception as e:
            logs.append(f"Failed to send email to {email}: {str(e)}")
    return logs, len(leads)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def ai_automation_loop(session, log_container, leads_container):
    st.session_state.automation_logs = []
    total_search_terms = 0
    total_emails_sent = 0

    while st.session_state.get('automation_status', False):
        try:
            log_container.info("Starting automation cycle")

            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                log_container.warning("Knowledge Base not found. Skipping cycle.")
                await asyncio.sleep(3600)
                continue

            base_terms = session.query(SearchTerm.term).filter_by(project_id=get_active_project_id()).all()
            base_terms = [term[0] for term in base_terms]
            optimized_terms = generate_optimized_search_terms(base_terms, kb_info)

            st.subheader("Optimized Search Terms")
            st.write(", ".join(optimized_terms))
            
            if st.button("Confirm Search Terms", key="confirm_search_terms"):
                total_search_terms = len(optimized_terms)
                classified_terms = classify_search_terms(optimized_terms, kb_info)

                progress_bar = st.progress(0)
                for idx, (group_name, terms) in enumerate(classified_terms.items()):
                    if group_name == "low_quality_search_terms":
                        continue

                    reference_template = session.query(EmailTemplate).filter_by(template_name="REFERENCE").first()
                    if not reference_template:
                        continue

                    adjusted_template = adjust_email_template_api(session, reference_template.body_content, 
                                                                  f"Adjust for: {', '.join(terms[:5])}", kb_info)
                    if not adjusted_template or 'body' not in adjusted_template:
                        continue

                    new_template = EmailTemplate(
                        template_name=f"{reference_template.template_name}_{group_name}",
                        subject=adjusted_template['subject'],
                        body_content=adjusted_template['body'],
                        project_id=get_active_project_id()
                    )
                    session.add(new_template)
                    session.commit()

                    results = []
                    for term in terms:
                        term_results = manual_search(session, [term], 10, "All Leads", log_container, leads_container)
                        results.extend(term_results)

                    new_leads = []
                    for res in results:
                        if save_lead(session, res['Email'], domain=get_domain_from_url(res['URL'])):
                            new_leads.append((res['Lead ID'], res['Email']))

                    if new_leads:
                        from_email = kb_info.get('contact_email', 'hello@indosy.com')
                        reply_to = kb_info.get('contact_email', 'eugproductions@gmail.com')
                        logs, sent_count = await bulk_send_coroutine(session, new_template.id, from_email, reply_to, new_leads, log_container)
                        st.session_state.automation_logs.extend(logs)
                        total_emails_sent += sent_count

                    leads_container.text_area("New Leads Found", "\n".join([email for _, email in new_leads]), height=200)
                    progress_bar.progress((idx + 1) / len(classified_terms))

                st.success(f"Automation cycle completed. Total search terms: {total_search_terms}, Total emails sent: {total_emails_sent}")
                await asyncio.sleep(3600)
            else:
                log_container.warning("Please confirm the search terms to proceed.")
                continue

        except Exception as e:
            log_container.error(f"Critical error in automation cycle: {str(e)}")
            await asyncio.sleep(300)

    log_container.info("Automation stopped")

def automation_control_panel_page():
    st.subheader("Control Your Automation Settings")
    if 'automation_status' not in st.session_state:
        st.session_state.automation_status = False
    if 'automation_logs' not in st.session_state:
        st.session_state.automation_logs = []

    col1, col2 = st.columns(2)
    with col1:
        button_text = "Stop Automation" if st.session_state.automation_status else "Start Automation"
        if st.button(button_text):
            st.session_state.automation_status = not st.session_state.automation_status
            if st.session_state.automation_status:
                st.session_state.automation_logs = []
    with col2:
        if st.button("Quick Scan"):
            with st.spinner("Performing quick scan..."):
                try:
                    with get_db_connection() as session:
                        quick_scan_results = perform_quick_scan(session)
                    st.success(f"Quick scan completed! Found {quick_scan_results['new_leads']} new leads.")
                except Exception as e:
                    st.error(f"An error occurred during quick scan: {str(e)}")

    st.subheader("Current Automation Status")
    status = "ON" if st.session_state.automation_status else "OFF"
    st.markdown(f"**Automation Status:** {status}")

    st.subheader("Real-Time Analytics")
    try:
        with get_db_connection() as session:
            display_real_time_analytics(session)
    except Exception as e:
        st.error(f"An error occurred while displaying analytics: {str(e)}")

    st.subheader("Automation Logs")
    log_container = st.empty()
    update_log_display(log_container, "automation_logs")

    st.subheader("Emails Found")
    leads_container = st.empty()

    if st.session_state.get('automation_status', False):
        st.write("Automation is currently running in the background.")
        try:
            with get_db_connection() as session:
                asyncio.run(ai_automation_loop(session, log_container, leads_container))
        except Exception as e:
            st.error(f"An error occurred in the automation process: {str(e)}")



            
            
def main():
    st.set_page_config(
        page_title="AutoclientAI - Lead Generation",
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
        "📈 View Sent Email Campaigns": view_sent_email_campaigns_page,
        "🚀 Projects & Campaigns": projects_campaigns_page,
        "📚 Knowledge Base": knowledge_base_page,
        "🤖 AutoclientAI": autoclient_ai_page,
        "⚙️ Automation Control": automation_control_panel_page
    }

    # Sidebar navigation menu
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=list(pages.keys()),
            icons=["search", "send", "people", "key", "envelope", "graph-up", "folder", "book", "robot", "gear"],
            menu_icon="cast",
            default_index=0
        )
    
    # Display the main title only once
    st.title("AutoclientAI")
    st.markdown("---")
    
    # Wrap the page function call in a try-except block
    try:
        pages[selected]()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("An error occurred in the main function")

    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")

if __name__ == "__main__":
    main()
