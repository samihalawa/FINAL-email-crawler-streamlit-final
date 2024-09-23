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
import time
import logging
import asyncio
from datetime import datetime, timedelta

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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --------------------- OpenAI Configuration ---------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE
openai_model = "gpt-4"

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
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

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

def get_db_connection():
    return SessionLocal()

def validate_email(email):
    pattern = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    return re.match(pattern, email) is not None

def is_valid_email(email):
    # Exclude emails that are likely to be invalid or unwanted
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js)$",  # File extensions
        r"^nr@.*",  # Patterns starting with 'nr@'
        r"^bootstrap@.*",
        r"^jquery@.*",
        r"^core@.*",
        r"^email@email\.com$",
        r"^icon-.*",
        r"^noreply@.*",
        r".*@example\.com$",
        r".*@.*\.(png|jpg|jpeg|gif|css|js)$",  # Emails with file extensions
        r".*@.*\.(jpga|js)$",  # Additional unwanted patterns
        r".*@.*\.(PM|HL)$"  # Additional unwanted patterns
    ]
    for pattern in invalid_patterns:
        if re.match(pattern, email, re.IGNORECASE):
            return False
    return validate_email(email)

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
    # Check if email or domain already exists
    existing_lead = session.query(Lead).filter_by(email=email).first()
    if existing_lead:
        return existing_lead.id
    if domain:
        domain_lead = session.query(Lead).filter(Lead.email.like(f"%@{domain}")).first()
        if domain_lead:
            return None  # Skip if email from this domain already exists
    new_lead = Lead(
        email=email,
        phone=phone,
        first_name=first_name,
        last_name=last_name,
        company=company,
        job_title=job_title
    )
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
        prompt=prompt,
        response=response,
        lead_id=lead_id,
        email_campaign_id=email_campaign_id,
        model_used=openai_model
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

def add_search_term(session, term):
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
        return new_term.id

def fetch_campaigns(session):
    campaigns = session.query(Campaign).all()
    return [f"{camp.id}: {camp.campaign_name}" for camp in campaigns]

def fetch_projects(session):
    try:
        projects = session.query(Project).all()
        return [f"{project.id}: {project.project_name}" for project in projects]
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return []

def fetch_email_templates(session, campaign_id=None):
    if campaign_id:
        templates = session.query(EmailTemplate).filter_by(campaign_id=campaign_id).all()
    else:
        templates = session.query(EmailTemplate).all()
    return [f"{template.id}: {template.template_name}" for template in templates]

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
    leads = session.query(Lead).all()
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
    email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).order_by(EmailCampaign.sent_at.desc()).all()
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

def manual_search(session, terms, num_results, search_type="All Leads", log_container=None, leads_container=None):
    all_results = []
    logs = []
    emails_found = []
    ua = UserAgent()
    headers = {'User-Agent': ua.random}
    campaign_id = get_active_campaign_id()

    total_terms = len(terms)
    term_progress = st.progress(0)
    term_counter = 0
    for term in terms:
        st.write(f"Currently searching for leads for: {term}")
        search_urls = list(search(term, num_results=num_results))
        term_id = add_or_get_search_term(session, term, campaign_id)
        valid_leads_count = 0
        blogs_found = 0
        directories_found = 0

        total_urls = len(search_urls)
        url_progress = st.progress(0)
        url_counter = 0
        for url in search_urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                emails = extract_emails_from_html(response.text)
                phone_numbers = extract_phone_numbers(response.text)
                content = extract_visible_text(soup)
                title = soup.title.string.strip() if soup.title else ""
                meta_description_tag = soup.find('meta', attrs={'name': 'description'})
                meta_description = meta_description_tag['content'].strip() if meta_description_tag else ""
                categorized = categorize_page_content(soup, url)
                tags = []
                if categorized == 'blog':
                    blogs_found += 1
                    tags.append('blog')
                elif categorized == 'directory':
                    directories_found += 1
                    tags.append('directory')
                else:
                    tags.append('company')

                domain = get_domain_from_url(url)

                for email in emails:
                    if is_valid_email(email):
                        lead_id = save_lead(session, email, phone=phone_numbers[0] if phone_numbers else None, domain=domain)
                        if lead_id:
                            save_lead_source(
                                session,
                                lead_id,
                                term_id,
                                url,
                                title,
                                meta_description,
                                phone_numbers,
                                content,
                                tags
                            )
                            all_results.append({
                                "Email": email,
                                "URL": url,
                                "Title": title,
                                "Description": meta_description,
                                "Tags": tags,
                                "Lead Source": url,
                                "Lead ID": lead_id
                            })
                            emails_found.append(email)
                            # Update leads display
                            if leads_container:
                                leads_container.text_area(
                                    "Emails Found",
                                    "\n".join(emails_found[-50:]),  # Display last 50 emails
                                    height=200
                                )
                            logs.append(f"New Lead! {email} from {url}")
                            if log_container:
                                log_container.text_area("Search Logs", "\n".join(logs[-10:]), height=200)
                            valid_leads_count += 1
                            break  # Only save one email per domain
            except requests.exceptions.RequestException as e:
                logs.append(f"Error processing {url}: {e}")
                if log_container:
                    log_container.text_area("Search Logs", "\n".join(logs[-10:]), height=200)
                continue

            url_counter += 1
            url_progress.progress(url_counter / total_urls)
        log_search_term_effectiveness(
            session,
            term,
            len(search_urls),
            valid_leads_count,
            blogs_found,
            directories_found
        )

        term_counter += 1
        term_progress.progress(term_counter / total_terms)

    st.write("Search completed!")
    return all_results

# AI Functions

def get_knowledge_base_info(session, project_id=None):
    if not project_id:
        project_id = get_active_project_id()
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

def classify_search_terms(session, search_terms, kb_info):
    prompt = f"""
Classify the following search terms into strategic groups:

Search Terms: {', '.join(search_terms)}

Knowledge Base Info:
{json.dumps(kb_info)}

Create groups that allow for tailored, personalized email content. Consider the product/service features, target audience, and potential customer pain points. Groups should be specific enough for customization but broad enough to be efficient. Always include a 'low_quality_search_terms' category for irrelevant or overly broad terms.

Respond with a JSON object in the following format:
{{
    "group_name_1": ["term1", "term2", "term3"],
    "group_name_2": ["term4", "term5", "term6"],
    "low_quality_search_terms": ["term7", "term8"]
}}
"""
    response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in strategic search term classification for targeted email marketing campaigns. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content.strip()
    log_ai_request(session, "classify_search_terms", prompt, content)
    try:
        classified_terms = json.loads(content)
        return classified_terms
    except json.JSONDecodeError:
        st.error("Failed to parse AI response. Please try again.")
        return {}

def generate_email_template(session, terms, kb_info):
    prompt = f"""
Create an email template for the following search terms:

Search Terms: {', '.join(terms)}

Knowledge Base Info:
{json.dumps(kb_info)}

Guidelines:
1. Focus on benefits to the reader
2. Address potential customer doubts and fears
3. Include clear CTAs at the beginning and end
4. Use a natural, conversational tone
5. Be concise but impactful
6. Use minimal formatting - remember this is an email, not a landing page

Provide the email body content in HTML format, excluding <body> tags. Use <p>, <strong>, <em>, and <a> tags as needed.

Respond with a JSON object in the following format:
{{
    "subject": "Your email subject here",
    "body": "Your HTML email body here"
}}
"""
    response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in creating high-converting email templates for targeted marketing campaigns. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content.strip()
    log_ai_request(session, "generate_email_template", prompt, content)
    try:
        email_template = json.loads(content)
        return email_template
    except json.JSONDecodeError:
        st.error("Failed to parse AI response. Please try again.")
        return {"subject": "", "body": ""}

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
1. Maintain focus on conversion and avoiding spam filters
2. Preserve the natural, conversational tone
3. Ensure benefits to the reader remain highlighted
4. Continue addressing potential customer doubts and fears
5. Keep clear CTAs at the beginning and end
6. Remain concise and impactful
7. Maintain minimal formatting suitable for an email

Provide the adjusted email body content in HTML format, excluding <body> tags.

Respond with a JSON object in the following format:
{{
    "subject": "Your adjusted email subject here",
    "body": "Your adjusted HTML email body here"
}}
"""
    response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in refining high-converting email templates for targeted marketing campaigns. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content.strip()
    log_ai_request(session, "adjust_email_template_api", prompt, content)
    try:
        adjusted_template = json.loads(content)
        return adjusted_template
    except json.JSONDecodeError:
        st.error("Failed to parse AI response. Please try again.")
        return {"subject": "", "body": ""}

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
    response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in optimizing search terms for targeted email marketing campaigns. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content.strip()
    log_ai_request(session, "generate_optimized_search_terms", prompt, content)
    try:
        optimized_terms = json.loads(content).get("optimized_terms", [])
        return optimized_terms
    except json.JSONDecodeError:
        st.error("Failed to parse AI response. Please try again.")
        return []

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

def get_email_preview(session, template_id, from_email, reply_to):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        return "<p>Template not found</p>"
    preview = f"""
    <h3>Email Preview</h3>
    <strong>Subject:</strong> {template.subject}<br>
    <strong>From:</strong> {from_email}<br>
    <strong>Reply-To:</strong> {reply_to}<br>
    <hr>
    <h4>Body:</h4>
    <div>{template.body_content}</div>
    """
    return preview

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
        try:
            sent_at = datetime.utcnow()
            message_id = f"msg-{lead_id}-{int(time.time())}"
            status = 'sent'
            customized_content = template.body_content
            save_email_campaign(session, lead_id, template_id, status, sent_at, template.subject, message_id, customized_content)
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
    while st.session_state.automation_status:
        try:
            # Fetch all search terms
            search_terms = fetch_all_search_terms(session)
            if not search_terms:
                st.warning("No search terms available.")
                return

            # Get knowledge base info
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                st.warning("Knowledge Base not found for the active project.")
                return

            # Classify search terms into groups
            classified_terms = classify_search_terms(session, search_terms, kb_info)
            log_message = f"Classified search terms into {len(classified_terms)} groups."
            st.session_state.automation_logs.append(log_message)
            update_log_display(log_container)

            # Get REFERENCE template
            reference_template = get_email_template_by_name(session, "REFERENCE")
            if not reference_template:
                st.warning("REFERENCE email template not found.")
                return

            # For each group
            for group_name, terms in classified_terms.items():
                if group_name == "low_quality_search_terms":
                    continue  # Skip low-quality terms

                # Create an email variation of the REFERENCE template
                adjustment_prompt = f"Adjust the email template to better appeal to recipients found using the search terms: {', '.join(terms)}"

                adjusted_template_data = adjust_email_template_api(session, reference_template.body_content, adjustment_prompt, kb_info)
                if not adjusted_template_data or not adjusted_template_data.get('body'):
                    st.warning(f"Failed to adjust email template for group '{group_name}'.")
                    continue

                # Save the new email template
                adjusted_template_name = f"{reference_template.template_name}_{group_name}"
                new_template_id = create_email_template(session, adjusted_template_name, adjusted_template_data['subject'], adjusted_template_data['body'])
                log_message = f"Created new email template '{adjusted_template_name}' for group '{group_name}'."
                st.session_state.automation_logs.append(log_message)
                update_log_display(log_container)

                # Search for leads using the group's search terms
                results = manual_search(session, terms, 10, "All Leads", log_container, leads_container)
                emails_found.extend([res['Email'] for res in results])

                # Update leads display
                leads_container.text_area("Emails Found", "\n".join(emails_found[-50:]), height=200)

                # Fetch leads found in this search
                lead_ids = [res['Lead ID'] for res in results]
                leads_to_send = [(lead_id, email) for lead_id, email in zip(lead_ids, emails_found)]

                # Send emails to the leads found
                from_email = kb_info.get('contact_email', 'hello@indosy.com')
                reply_to = kb_info.get('contact_email', 'eugproductions@gmail.com')
                logs, _ = await bulk_send_coroutine(session, new_template_id, from_email, reply_to, leads_to_send)
                st.session_state.automation_logs.extend(logs)
                update_log_display(log_container)

            await asyncio.sleep(60)

        except Exception as e:
            logging.error(f"Error during automation: {e}")
            st.session_state.automation_logs.append(f"Error during automation: {e}")
            update_log_display(log_container)
            break

def update_log_display(log_container):
    log_container.text_area("Automation Logs", "\n".join(st.session_state.automation_logs[-20:]), height=300)

def display_search_results(results):
    st.subheader("Search Results")
    st.write(f"Total Leads Found: {len(results)}")
    for res in results:
        with st.expander(f"Lead: {res['Email']}"):
            st.write(f"**URL:** [{res['URL']}]({res['URL']})")
            st.write(f"**Title:** {res['Title']}")
            st.write(f"**Description:** {res['Description']}")
            st.write(f"**Tags:** {', '.join(res['Tags'])}")
            st.write(f"**Lead Source:** {res['Lead Source']}")
            st.write(f"**Lead Email:** {res['Email']}")

def perform_quick_scan(session):
    terms = get_least_searched_terms(session, 3)
    results = manual_search(session, terms, 10, "All Leads")
    return {"new_leads": len(results)}

# --------------------- Page Functions ---------------------
def manual_search_page():
    st.header("Manual Search")
    session = get_db_connection()
    with st.form("manual_search_form"):
        campaigns = fetch_campaigns(session)
        campaign = st.selectbox("Select Campaign", options=campaigns)
        set_active_campaign_id(int(campaign.split(":")[0]))
        search_terms = st.text_area("Enter Search Terms (one per line)")
        num_results = st.slider("Number of Results per Term", 10, 200, 30, step=10)
        search_type = st.selectbox("Search Type", ["All Leads", "Exclude Probable Blogs/Directories"])
        submit = st.form_submit_button("Search")

    if submit:
        terms = [term.strip() for term in search_terms.split('\n') if term.strip()]
        if not terms:
            st.warning("Please enter at least one search term.")
        else:
            with st.spinner("Performing manual search..."):
                progress_bar = st.progress(0)
                total_terms = len(terms)
                results = []
                log_container = st.empty()
                leads_container = st.empty()  # Placeholder for emails
                for i, term in enumerate(terms):
                    st.info(f"Currently searching for leads for: {term}")
                    try:
                        term_results = manual_search(session, [term], num_results, search_type, log_container, leads_container)
                        results.extend(term_results)
                        st.success(f"Search for term '{term}' completed! Found {len(term_results)} leads.")
                    except ValueError as e:
                        if "NUL" in str(e):
                            st.error(f"Error: The term '{term}' contains invalid characters.")
                        else:
                            st.error(f"Error during search for term '{term}': {e}")
                    progress_bar.progress((i + 1) / total_terms)
                    st.write(f"Processed {i + 1} out of {total_terms} terms.")
                    log_message = f"Search for term '{term}' completed! Found {len(term_results)} leads."
                    st.session_state.automation_logs.append(log_message)
                    update_log_display(log_container)
                    display_search_results(term_results)
                st.success(f"Search completed! Found {len(results)} valid leads in total.")
                display_search_results(results)
        session.close()

def bulk_send_page():
    st.header("Bulk Send")
    session = get_db_connection()
    with st.form("bulk_send_form"):
        templates = fetch_email_templates(session)
        if not templates:
            st.warning("No email templates found. Please create a template first.")
            return
        template = st.selectbox("Select Email Template", options=templates)
        template_id = int(template.split(":")[0])
        from_email = st.text_input("From Email", value="Sami Halawa AI <hello@indosy.comm>")
        reply_to = st.text_input("Reply To", value="eugproductions@gmail.com")
        send_option = st.radio(
            "Send to:",
            ["All Leads", "All Not Contacted with this Template", "All Not Contacted with Templates from this Campaign"]
        )
        filter_option = st.radio(
            "Filter:",
            ["Not Filter Out Leads", "Filter Out blog-directory"]
        )
        preview_button = st.form_submit_button(label="Preview Email")
        send_button = st.form_submit_button(label="Start Bulk Send")
    if preview_button:
        preview = get_email_preview(session, template_id, from_email, reply_to)
        st.components.v1.html(preview, height=600, scrolling=True)
    if send_button:
        with st.spinner("Starting bulk send..."):
            leads_to_send = fetch_leads_for_bulk_send(session, template_id, send_option, filter_option)
            logs, _ = asyncio.run(bulk_send_coroutine(session, template_id, from_email, reply_to, leads_to_send))
            for log in logs:
                st.write(log)
            st.success(f"Bulk send completed. Sent {len(leads_to_send)} emails.")
    session.close()

def view_leads_page():
    st.header("View Leads")
    session = get_db_connection()
    if st.button("Refresh Leads"):
        st.session_state.leads = fetch_leads(session)
    if 'leads' not in st.session_state:
        st.session_state.leads = fetch_leads(session)
    st.dataframe(st.session_state.leads)
    session.close()

def search_terms_page():
    st.header("Search Terms")
    session = get_db_connection()
    with st.form("add_search_term_form"):
        search_term = st.text_input("Enter New Search Term")
        submit = st.form_submit_button("Add Search Term")
    if submit:
        if not search_term.strip():
            st.warning("Please enter a valid search term.")
        else:
            add_search_term(session, search_term)
            st.success(f"Search term '{search_term}' added.")
    st.subheader("Existing Search Terms")
    search_terms_df = fetch_search_terms(session)
    st.dataframe(search_terms_df)
    session.close()

def email_templates_page():
    st.header("Email Templates")
    session = get_db_connection()
    with st.form("add_email_template_form"):
        template_name = st.text_input("Template Name")
        subject = st.text_input("Subject")
        body_content = st.text_area("Body Content (HTML)", height=400)
        submit = st.form_submit_button("Add Email Template")
    if submit:
        if not template_name.strip() or not subject.strip() or not body_content.strip():
            st.warning("Please fill in all fields.")
        else:
            template_id = create_email_template(session, template_name, subject, body_content)
            st.success(f"Email template '{template_name}' added with ID: {template_id}.")
    st.subheader("Existing Email Templates")
    templates = fetch_email_templates(session)
    if templates:
        templates_df = pd.DataFrame({
            "ID": [int(t.split(":")[0]) for t in templates],
            "Template Name": [t.split(": ")[1] for t in templates]
        })
        st.dataframe(templates_df)
    else:
        st.info("No email templates found.")
    session.close()

def view_sent_email_campaigns_page():
    st.header("View Sent Email Campaigns")
    session = get_db_connection()
    if st.button("Refresh Sent Email Campaigns"):
        st.session_state.sent_email_campaigns = fetch_sent_email_campaigns(session)
    if 'sent_email_campaigns' not in st.session_state:
        st.session_state.sent_email_campaigns = fetch_sent_email_campaigns(session)
    st.dataframe(st.session_state.sent_email_campaigns)
    session.close()

def projects_campaigns_page():
    st.header("Projects & Campaigns")
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
                        st.success(f"Campaign '{campaign_name}' added to project '{selected_project}'.")
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
    st.header("Knowledge Base")
    session = get_db_connection()
    project_options = fetch_projects(session)
    selected_project = st.selectbox("Select Project", options=project_options)
    project_id = int(selected_project.split(":")[0])
    set_active_project_id(project_id)
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
        st.success("Knowledge Base saved successfully.")
    session.close()

def autoclient_ai_page():
    st.header("AutoclientAI")
    session = get_db_connection()
    project_options = fetch_projects(session)
    selected_project = st.selectbox("Select Project for AI Enhancements", options=project_options)
    project_id = int(selected_project.split(":")[0])
    set_active_project_id(project_id)
    kb_info = get_knowledge_base_info(session, project_id)
    if not kb_info:
        st.warning("Knowledge Base not found for the selected project. Please set it up first.")
        session.close()
        return
    tab1, tab2, tab3, tab4 = st.tabs([
        "Optimize Existing Groups",
        "Create New Groups",
        "Adjust Email Templates",
        "Optimize Search Terms"
    ])
    with tab1:
        optimize_existing_groups(session, kb_info)
    with tab2:
        create_new_groups(session, kb_info)
    with tab3:
        adjust_email_template(session, kb_info)
    with tab4:
        optimize_search_terms_ai(session, kb_info)
    session.close()

def optimize_existing_groups(session, kb_info):
    try:
        search_terms = session.query(SearchTerm).all()
        terms = [term.term for term in search_terms]
        if not terms:
            st.info("No search terms available to optimize.")
            return
        classification = classify_search_terms(session, terms, kb_info)
        st.write("**Optimized Search Term Groups:**")
        st.json(classification)
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")

def create_new_groups(session, kb_info):
    try:
        all_terms = session.query(SearchTerm).all()
        terms = [term.term for term in all_terms]
        if not terms:
            st.info("No search terms available to create groups.")
            return
        classification = classify_search_terms(session, terms, kb_info)
        st.write("**Created New Search Term Groups:**")
        st.json(classification)
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")

def adjust_email_template(session, kb_info):
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
        current_template = session.query(EmailTemplate).filter_by(id=template_id).first()
        if not current_template:
            st.warning("Selected template not found.")
            return
        adjusted_content = adjust_email_template_api(
            session,
            current_template.body_content,
            adjustment_prompt,
            kb_info
        )
        current_template.body_content = adjusted_content
        session.commit()
        st.success("Email template adjusted successfully.")

def optimize_search_terms_ai(session, kb_info):
    current_terms = [term.term for term in session.query(SearchTerm).all()]
    if not current_terms:
        st.info("No search terms available to optimize.")
        return
    optimized_terms = generate_optimized_search_terms(session, current_terms, kb_info)
    st.write("**Optimized Search Terms:**")
    st.write("\n".join(optimized_terms))
    if st.button("Save Optimized Search Terms"):
        save_optimized_search_terms(session, optimized_terms)
        st.success("Optimized search terms saved successfully.")

def automation_control_panel_page():
    st.header("Automation Control Panel")
    session = get_db_connection()

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
                quick_scan_results = perform_quick_scan(session)
                st.success(f"Quick scan completed! Found {quick_scan_results['new_leads']} new leads.")

    st.subheader("Current Automation Status")
    status = "ON" if st.session_state.automation_status else "OFF"
    st.write(f"**Automation is currently:** {status}")

    st.subheader("Real-Time Analytics")
    display_real_time_analytics(session)

    st.subheader("Automation Logs")
    log_container = st.empty()
    update_log_display(log_container)

    st.subheader("Emails Found")
    leads_container = st.empty()  # Placeholder for emails

    if st.session_state.get('automation_status', False):
        st.write("Automation is running in the background.")
        asyncio.run(continuous_automation_process(session, log_container, leads_container))

    session.close()


def main():
    st.set_page_config(page_title="AUTOCLIENT - Lead Generation", layout="wide")
    st.title("AUTOCLIENT - Lead Generation App")
    st.sidebar.title("Navigation")
    
    pages = {
        "Manual Search": manual_search_page,
        "Bulk Send": bulk_send_page,
        "View Leads": view_leads_page,
        "Search Terms": search_terms_page,
        "Email Templates": email_templates_page,
        "View Sent Email Campaigns": view_sent_email_campaigns_page,
        "Projects & Campaigns": projects_campaigns_page,
        "Knowledge Base": knowledge_base_page,
        "AutoclientAI": autoclient_ai_page,
        "Automation Control": automation_control_panel_page
    }

    selected_page = st.sidebar.radio("Select a page", list(pages.keys()))
    pages[selected_page]()

if __name__ == "__main__":
    main()
