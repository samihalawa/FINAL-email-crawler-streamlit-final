import os, json, re, logging, time, uuid, random, requests
from datetime import datetime
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, distinct, and_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, joinedload
from sqlalchemy.exc import SQLAlchemyError
from email_validator import validate_email, EmailNotValidError
import boto3
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import urllib3
from urllib.parse import urlparse, urlencode
from botocore.exceptions import ClientError
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from googlesearch import search as google_search

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key

# MODELS

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

    def to_dict(self):
        fields = ['kb_name','kb_bio','kb_values','contact_name','contact_role','contact_email',
                  'company_description','company_mission','company_target_market','company_other',
                  'product_name','product_description','product_target_customer','product_other',
                  'other_context','example_email']
        return {attr: getattr(self, attr) for attr in fields}

class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
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
    language = Column(Text, default='ES')
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

class SearchTermEffectiveness(Base):
    __tablename__ = 'search_term_effectiveness'
    id = Column(BigInteger, primary_key=True)
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    total_results = Column(BigInteger)
    valid_leads = Column(BigInteger)
    irrelevant_leads = Column(BigInteger)
    blogs_found = Column(BigInteger)
    directories_found = Column(BigInteger)
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
    language = Column(Text, default='ES')
    group = relationship("SearchTermGroup", back_populates="search_terms")
    campaign = relationship("Campaign", back_populates="search_terms")
    optimized_terms = relationship("OptimizedSearchTerm", backref="original_term")
    lead_sources = relationship("LeadSource", back_populates="search_term")
    effectiveness = relationship("SearchTermEffectiveness", back_populates="search_term", uselist=False)

class LeadSource(Base):
    __tablename__ = 'lead_sources'
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    url = Column(Text)
    domain = Column(Text)
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

class AIRequestLog(Base):
    __tablename__ = 'ai_request_logs'
    id = Column(BigInteger, primary_key=True)
    function_name = Column(Text)
    prompt = Column(Text)
    response = Column(Text)
    model_used = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    email_campaign_id = Column(BigInteger, ForeignKey('email_campaigns.id'))

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

class Settings(Base):
    __tablename__ = 'settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    setting_type = Column(Text, nullable=False)
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

Base.metadata.create_all(bind=engine)

def db_session():
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def is_valid_email_address(email):
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def wrap_email_body(body_content):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"></head>
    <body>
        {body_content}
    </body>
    </html>
    """

def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None):
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings:
        logging.error(f"No email settings found for {from_email}")
        return None, None
    tracking_id = str(uuid.uuid4())
    wrapped_body = wrap_email_body(body)
    try:
        if email_settings.provider == 'ses':
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
                    'Body': {'Html': {'Data': wrapped_body, 'Charset': charset}}
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
            msg.attach(MIMEText(wrapped_body, 'html'))

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

def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    lead = session.query(Lead).filter_by(email=lead_email).first()
    if not lead:
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
        tracking_id=str(uuid.uuid4()))
    session.add(new_campaign)
    session.commit()
    return new_campaign

def get_active_campaign_id():
    # For simplicity, return 1 or read from environment/session
    return 1

def get_active_project_id():
    # For simplicity, return 1 or read from environment/session
    return 1

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

def openai_chat_completion(messages, temperature=0.7, lead_id=None, email_campaign_id=None):
    # Generic OpenAI chat completion using openai.ChatCompletion
    model = "gpt-4"  # Example model
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        content = response.choices[0].message.content
        with SessionLocal() as session:
            log_ai_request(session, "chat_completion", messages, content, lead_id, email_campaign_id, model)
        # Try to parse as JSON
        try:
            return json.loads(content)
        except:
            return content
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        with SessionLocal() as session:
            log_ai_request(session, "chat_completion", messages, str(e), lead_id, email_campaign_id, model)
        return None

def generate_or_adjust_email_template(prompt, kb_info=None, current_template=None):
    messages = [
        {"role": "system", "content": "You are an AI assistant for creating and refining email templates."},
        {"role": "user", "content": f"""{'Adjust the following email template:' if current_template else 'Create an email template:'} {prompt}
        {'Current Template:' if current_template else ''}{current_template if current_template else ''}
        Respond with JSON containing 'subject' and 'body' keys.
        """}
    ]
    if kb_info:
        messages.append({"role": "user", "content": f"Consider this KB: {json.dumps(kb_info)}"})

    result = openai_chat_completion(messages)
    if isinstance(result, dict):
        return result
    return {"subject": "Subject", "body": "<p>Generated content</p>"}

def fetch_leads_with_sources(session):
    query = session.query(
        Lead,
        func.string_agg(LeadSource.url, ', ').label('sources'),
        func.max(EmailCampaign.sent_at).label('last_contact'),
        func.string_agg(EmailCampaign.status, ', ').label('email_statuses')
    ).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(Lead.id)

    results = []
    for lead, sources, last_contact, email_statuses in query.all():
        last_status = email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted'
        results.append({
            'ID': lead.id,
            'Email': lead.email,
            'First Name': lead.first_name,
            'Last Name': lead.last_name,
            'Company': lead.company,
            'Job Title': lead.job_title,
            'Created At': lead.created_at.isoformat(),
            'Source': sources,
            'Last Contact': last_contact.isoformat() if last_contact else None,
            'Last Email Status': last_status,
            'Delete': False
        })
    return results

def get_domain_from_url(url):
    return urlparse(url).netloc

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def extract_emails_from_html(html_content):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, html_content)

def extract_info_from_page(soup):
    # Basic info extraction from meta tags
    name = soup.find('meta', {'name': 'author'})
    name = name['content'] if name else ''
    company = soup.find('meta', {'property': 'og:site_name'})
    company = company['content'] if company else ''
    job_title = soup.find('meta', {'name': 'job_title'})
    job_title = job_title['content'] if job_title else ''
    return name, company, job_title

def get_page_title(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.title.string.strip() if soup.title else "No title found"

def get_page_description(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    return meta_desc['content'].strip() if meta_desc else "No description found"

def save_lead(session, email, first_name=None, last_name=None, company=None, job_title=None, phone=None, url=None, search_term_id=None, created_at=None):
    try:
        existing_lead = session.query(Lead).filter_by(email=email).first()
        if existing_lead:
            # Update existing lead if needed
            if first_name: existing_lead.first_name = first_name
            if last_name: existing_lead.last_name = last_name
            if company: existing_lead.company = company
            if job_title: existing_lead.job_title = job_title
            if phone: existing_lead.phone = phone
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

def optimize_search_term(search_term, language):
    if language == 'english':
        return f'"{search_term}" email OR contact site:.com'
    elif language == 'spanish':
        return f'"{search_term}" correo OR contacto site:.es'
    return search_term

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def add_or_get_search_term(session, term, campaign_id, created_at=None):
    search_term = session.query(SearchTerm).filter_by(term=term, campaign_id=campaign_id).first()
    if not search_term:
        search_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=created_at or datetime.utcnow())
        session.add(search_term)
        session.commit()
        session.refresh(search_term)
    return search_term.id

def manual_search(session, terms, num_results, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False, language='ES', enable_email_sending=False, from_email=None, reply_to=None, email_template=None):
    results, total_leads = [], 0
    domains_processed = set()
    processed_emails_per_domain = {}

    for original_term in terms:
        try:
            search_term_id = add_or_get_search_term(session, original_term, get_active_campaign_id())
            search_term = shuffle_keywords(original_term) if shuffle_keywords_option else original_term
            if optimize_english:
                search_term = optimize_search_term(search_term, 'english')
            elif optimize_spanish:
                search_term = optimize_search_term(search_term, 'spanish')

            for url in google_search(search_term, num_results=num_results, lang=language):
                domain = get_domain_from_url(url)
                if ignore_previously_fetched and domain in domains_processed:
                    continue

                try:
                    if not url.startswith(('http://', 'https://')):
                        url = 'http://' + url
                    response = requests.get(url, timeout=10, verify=False)
                    response.raise_for_status()
                    html_content = response.text
                    soup = BeautifulSoup(html_content, 'html.parser')
                    emails = extract_emails_from_html(html_content)
                    valid_emails = [e for e in emails if is_valid_email(e)]

                    if valid_emails:
                        name, company, job_title = extract_info_from_page(soup)
                        page_title = get_page_title(html_content)
                        page_description = get_page_description(html_content)

                        if domain not in processed_emails_per_domain:
                            processed_emails_per_domain[domain] = set()

                        for email in valid_emails:
                            if email in processed_emails_per_domain[domain]:
                                continue
                            processed_emails_per_domain[domain].add(email)
                            lead = save_lead(session, email=email, first_name=name, company=company,
                                             job_title=job_title, url=url, search_term_id=search_term_id,
                                             created_at=datetime.utcnow())
                            if lead:
                                total_leads += 1
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
                                            save_email_campaign(session, email, template.id, 'Sent', datetime.utcnow(), template.subject, response['MessageId'], wrapped_content)
                                        else:
                                            save_email_campaign(session, email, template.id, 'Failed', datetime.utcnow(), template.subject, None, wrapped_content)

                    domains_processed.add(domain)
                except requests.RequestException:
                    pass
        except Exception as e:
            logging.error(f"Error processing term '{original_term}': {str(e)}")

    return {"total_leads": total_leads, "results": results}

def bulk_send_emails(session, template_id, from_email, reply_to, leads):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if not template:
        logging.error(f"Email template with ID {template_id} not found.")
        return [], 0

    email_subject = template.subject
    email_content = template.body_content

    logs, sent_count = [], 0
    for lead in leads:
        if not is_valid_email_address(lead['Email']):
            logs.append(f"Invalid email: {lead['Email']}")
            continue

        response, tracking_id = send_email_ses(session, from_email, lead['Email'], email_subject, email_content, reply_to=reply_to)
        if response:
            status = 'sent'
            message_id = response.get('MessageId', f"sent-{uuid.uuid4()}")
            sent_count += 1
            logs.append(f"Email sent to: {lead['Email']}")
        else:
            status = 'failed'
            message_id = f"failed-{uuid.uuid4()}"
            logs.append(f"Failed to send email to: {lead['Email']}")

        save_email_campaign(session, lead['Email'], template_id, status, datetime.utcnow(), email_subject, message_id, email_content)
    return logs, sent_count

def fetch_all_email_logs(session):
    email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
    data = []
    for ec in email_campaigns:
        data.append({
            'ID': ec.id,
            'Sent At': ec.sent_at.isoformat() if ec.sent_at else None,
            'Email': ec.lead.email,
            'Template': ec.template.template_name,
            'Subject': ec.customized_subject,
            'Content': ec.customized_content,
            'Status': ec.status,
            'Message ID': ec.message_id,
            'Campaign ID': ec.campaign_id,
            'Lead Name': f"{ec.lead.first_name or ''} {ec.lead.last_name or ''}".strip() or "Unknown",
            'Lead Company': ec.lead.company or "Unknown"
        })
    return data

def ai_group_search_terms(session, ungrouped_terms):
    existing_groups = session.query(SearchTermGroup).all()
    prompt = f"Categorize these search terms into existing groups or suggest new ones:\n{', '.join([t.term for t in ungrouped_terms])}\n\nExisting groups: {', '.join([g.name for g in existing_groups])}\n\nRespond with a JSON object: {{group_name: [term1, term2, ...]}}"
    messages = [
        {"role": "system", "content": "You're an AI that categorizes search terms."},
        {"role": "user", "content": prompt}
    ]
    response = openai_chat_completion(messages)
    if isinstance(response, dict):
        return response
    return {}

def update_search_term_groups(session, grouped_terms):
    for group_name, terms in grouped_terms.items():
        group = session.query(SearchTermGroup).filter_by(name=group_name).first()
        if not group:
            group = SearchTermGroup(name=group_name)
            session.add(group)
            session.flush()
        for term in terms:
            st = session.query(SearchTerm).filter_by(term=term).first()
            if st:
                st.group_id = group.id
    session.commit()

def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"Optimize and expand these terms:\n{', '.join(base_terms)}\nConsider target market and mission.\nRespond with {{'optimized_terms': [...terms...]}}"
    messages = [{"role": "system", "content": "Optimize search terms."}, {"role": "user", "content": prompt}]
    response = openai_chat_completion(messages)
    if isinstance(response, dict) and 'optimized_terms' in response:
        return response['optimized_terms']
    return base_terms

def fetch_search_terms_with_lead_count(session):
    query = (session.query(SearchTerm.term,
                           func.count(distinct(Lead.id)).label('lead_count'),
                           func.count(distinct(EmailCampaign.id)).label('email_count'))
             .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
             .join(Lead, LeadSource.lead_id == Lead.id)
             .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
             .group_by(SearchTerm.term))
    df = query.all()
    result = []
    for row in df:
        result.append({'Term': row[0], 'Lead Count': row[1], 'Email Count': row[2]})
    return result

def fetch_leads_for_search_terms(session, search_term_ids):
    return session.query(Lead).distinct().join(LeadSource).filter(LeadSource.search_term_id.in_(search_term_ids)).all()

def perform_quick_scan(session):
    # Just a demo endpoint: run a short search on random terms
    terms = session.query(SearchTerm).order_by(func.random()).limit(3).all()
    from_email = session.query(EmailSettings).first().email if session.query(EmailSettings).first() else None
    reply_to = from_email
    email_template = session.query(EmailTemplate).first()
    if not email_template:
        return {"error": "No email template available"}
    res = manual_search(session, [t.term for t in terms], 10, True, False, False, True, "EN", True, from_email, reply_to, f"{email_template.id}: {email_template.template_name}")
    return {"new_leads": len(res['results']), "terms_used": [t.term for t in terms]}

def bulk_send_page(session, template_id, from_email, reply_to, send_option, specific_email=None, selected_terms=None, exclude_previously_contacted=True):
    leads = fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted)
    logs, sent_count = bulk_send_emails(session, template_id, from_email, reply_to, leads)
    return {"logs": logs, "sent_count": sent_count}

def fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted):
    query = session.query(Lead)
    if send_option == "Specific Email" and specific_email:
        query = query.filter(Lead.email == specific_email)
    elif send_option in ["Leads from Chosen Search Terms"] and selected_terms:
        query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))

    if exclude_previously_contacted:
        subquery = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None)).subquery()
        query = query.outerjoin(subquery, Lead.id == subquery.c.lead_id).filter(subquery.c.lead_id.is_(None))

    return [{"Email": lead.email, "ID": lead.id} for lead in query.all()]

# Flask App Endpoints

app = Flask(__name__)

@app.route('/projects', methods=['GET'])
def list_projects():
    with SessionLocal() as session:
        projects = session.query(Project).all()
        return jsonify([{"id": p.id, "project_name": p.project_name} for p in projects])

@app.route('/campaigns', methods=['GET'])
def list_campaigns():
    with SessionLocal() as session:
        campaigns = session.query(Campaign).all()
        return jsonify([{"id": c.id, "campaign_name": c.campaign_name, "project_id": c.project_id} for c in campaigns])

@app.route('/leads', methods=['GET'])
def list_leads_endpoint():
    with SessionLocal() as session:
        leads = session.query(Lead).limit(50).all()
        return jsonify([{
            "id": l.id,
            "email": l.email,
            "first_name": l.first_name,
            "last_name": l.last_name,
            "company": l.company,
            "job_title": l.job_title
        } for l in leads])

@app.route('/send_test_email', methods=['POST'])
def send_test_email():
    data = request.json
    from_email = data.get('from_email')
    to_email = data.get('to_email')
    subject = data.get('subject', 'Test Subject')
    body = data.get('body', '<p>Hello, this is a test email.</p>')
    with SessionLocal() as session:
        if not is_valid_email_address(to_email):
            return jsonify({"error": "Invalid recipient email"}), 400
        response, tracking_id = send_email_ses(session, from_email, to_email, subject, body)
        if response:
            save_email_campaign(session, to_email, 1, 'Sent', datetime.utcnow(), subject, response.get('MessageId'), body)
            return jsonify({"message": "Email sent successfully", "tracking_id": tracking_id})
        else:
            return jsonify({"error": "Failed to send email"}), 500

@app.route('/knowledge_base', methods=['GET'])
def get_knowledge_base():
    with SessionLocal() as session:
        kb = session.query(KnowledgeBase).filter_by(project_id=get_active_project_id()).first()
        if kb:
            return jsonify(kb.to_dict())
        return jsonify({"error": "Knowledge Base not found"}), 404

@app.route('/manual_search', methods=['POST'])
def manual_search_endpoint():
    data = request.json
    terms = data.get('terms', [])
    num_results = data.get('num_results', 10)
    ignore_previously_fetched = data.get('ignore_previously_fetched', True)
    optimize_english = data.get('optimize_english', False)
    optimize_spanish = data.get('optimize_spanish', False)
    shuffle_keywords_option = data.get('shuffle_keywords_option', False)
    language = data.get('language', 'ES')
    enable_email_sending = data.get('enable_email_sending', False)
    from_email = data.get('from_email')
    reply_to = data.get('reply_to')
    email_template = data.get('email_template')

    with SessionLocal() as session:
        result = manual_search(session, terms, num_results, ignore_previously_fetched,
                               optimize_english, optimize_spanish, shuffle_keywords_option,
                               language, enable_email_sending, from_email, reply_to, email_template)
        return jsonify(result)

@app.route('/generate_template', methods=['POST'])
def generate_template_endpoint():
    data = request.json
    prompt = data.get('prompt')
    current_template = data.get('current_template')
    with SessionLocal() as session:
        kb = session.query(KnowledgeBase).filter_by(project_id=get_active_project_id()).first()
        kb_info = kb.to_dict() if kb else None
    result = generate_or_adjust_email_template(prompt, kb_info, current_template)
    return jsonify(result)

@app.route('/ai_group_search_terms', methods=['POST'])
def ai_group_search_terms_endpoint():
    with SessionLocal() as session:
        ungrouped_terms = session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
        if ungrouped_terms:
            grouped = ai_group_search_terms(session, ungrouped_terms)
            update_search_term_groups(session, grouped)
            return jsonify({"message": "Search terms grouped successfully", "groups": grouped})
        else:
            return jsonify({"message": "No ungrouped terms found."})

@app.route('/bulk_send', methods=['POST'])
def bulk_send_endpoint():
    data = request.json
    template_id = data.get('template_id')
    from_email = data.get('from_email')
    reply_to = data.get('reply_to')
    send_option = data.get('send_option', 'All Leads')
    specific_email = data.get('specific_email')
    selected_terms = data.get('selected_terms')
    exclude_previously_contacted = data.get('exclude_previously_contacted', True)
    with SessionLocal() as session:
        result = bulk_send_page(session, template_id, from_email, reply_to, send_option, specific_email, selected_terms, exclude_previously_contacted)
        return jsonify(result)

@app.route('/fetch_email_logs', methods=['GET'])
def fetch_email_logs():
    with SessionLocal() as session:
        logs = fetch_all_email_logs(session)
        return jsonify(logs)

@app.route('/perform_quick_scan', methods=['GET'])
def perform_quick_scan_endpoint():
    with SessionLocal() as session:
        result = perform_quick_scan(session)
        return jsonify(result)

# You can add more endpoints for leads management, templates, projects/campaigns CRUD, etc., following the same pattern.

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
