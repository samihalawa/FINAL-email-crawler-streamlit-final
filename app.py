    import os
    import json
    import re
    import logging
    import random
    import time
    import traceback
    import uuid
    import requests
    import asyncio
    import urllib3
    import smtplib
    import sys
    from googlesearch import search
    from datetime import datetime, timedelta
    from contextlib import contextmanager
    from typing import List, Optional, Dict
    from urllib.parse import urlparse

    # FastAPI / Jinja2
    from fastapi import (
        FastAPI, Request, Form, Depends, HTTPException, BackgroundTasks, Response, status
    )
    from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware

    # SQLAlchemy
    from sqlalchemy import (
        create_engine,
        Column,
        BigInteger,
        Text,
        DateTime,
        ForeignKey,
        Boolean,
        JSON,
        func,
        distinct,
        text,
        event
    )
    from sqlalchemy.orm import (
        sessionmaker,
        declarative_base,
        relationship,
        joinedload,
        Session,
    )
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.pool import QueuePool

    # Additional libs
    import boto3
    from botocore.exceptions import ClientError
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email_validator import validate_email, EmailNotValidError
    from dotenv import load_dotenv
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
    from pydantic import BaseModel

    """
    FASTAPI SINGLE-FILE APPLICATION
    Replicates all pages, routes, and logic from a Streamlit code,
    but using FastAPI + Jinja2 for the user interface, and a single file approach.
    To run locally:
    python main.py
    Then visit http://localhost:8000 in your browser.
    """

    # ------------- ENV / DATABASE SETUP -------------
    load_dotenv()

    DB_HOST = os.getenv("SUPABASE_DB_HOST")
    DB_NAME = os.getenv("SUPABASE_DB_NAME")
    DB_USER = os.getenv("SUPABASE_DB_USER")
    DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
    DB_PORT = os.getenv("SUPABASE_DB_PORT")

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
        raise ValueError("One or more required database environment variables are not set")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Configure SQLAlchemy engine with connection pooling
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,  # Recycle connections after 30 minutes
        echo=False  # Set to True for debugging SQL queries
    )

    @event.listens_for(engine, "connect")
    def connect(dbapi_connection, connection_record):
        connection_record.info['pid'] = os.getpid()

    @event.listens_for(engine, "checkout")
    def checkout(dbapi_connection, connection_record, connection_proxy):
        pid = os.getpid()
        if connection_record.info['pid'] != pid:
            connection_record.connection = None
            raise SQLAlchemyError(
                f"Connection record belongs to pid {connection_record.info['pid']}, "
                f"attempting to check out in pid {pid}"
            )

    SessionLocal = sessionmaker(bind=engine)
    Base = declarative_base()

    # ------------- MODELS (replicated from the original code) -------------
    class Project(Base):
        __tablename__ = "projects"
        id = Column(BigInteger, primary_key=True)
        project_name = Column(Text, default="Default Project")
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        campaigns = relationship("Campaign", back_populates="project")
        knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False)


    class Campaign(Base):
        __tablename__ = "campaigns"
        id = Column(BigInteger, primary_key=True)
        campaign_name = Column(Text, default="Default Campaign")
        campaign_type = Column(Text, default="Email")
        project_id = Column(BigInteger, ForeignKey("projects.id"), default=1)
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
        __tablename__ = "campaign_leads"
        id = Column(BigInteger, primary_key=True)
        campaign_id = Column(BigInteger, ForeignKey("campaigns.id"))
        lead_id = Column(BigInteger, ForeignKey("leads.id"))
        status = Column(Text)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        lead = relationship("Lead", back_populates="campaign_leads")
        campaign = relationship("Campaign", back_populates="campaign_leads")


    class KnowledgeBase(Base):
        __tablename__ = "knowledge_base"
        id = Column(BigInteger, primary_key=True)
        project_id = Column(BigInteger, ForeignKey("projects.id"), nullable=False)
        kb_name = Column(Text)
        kb_bio = Column(Text)
        kb_values = Column(JSON)
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
        created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
        updated_at = Column(DateTime(timezone=True), onupdate=func.now(), index=True)
        project = relationship("Project", back_populates="knowledge_base")

        def to_dict(self):
            fields = [
                "kb_name",
                "kb_bio",
                "kb_values",
                "contact_name",
                "contact_role",
                "contact_email",
                "company_description",
                "company_mission",
                "company_target_market",
                "company_other",
                "product_name",
                "product_description",
                "product_target_customer",
                "product_other",
                "other_context",
                "example_email",
            ]
            return {attr: getattr(self, attr) for attr in fields}


    class Lead(Base):
        __tablename__ = "leads"
        id = Column(BigInteger, primary_key=True)
        email = Column(Text, unique=True, index=True)
        phone = Column(Text, index=True)
        first_name = Column(Text)
        last_name = Column(Text)
        company = Column(Text, index=True)
        job_title = Column(Text)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        campaign_leads = relationship("CampaignLead", back_populates="lead")
        lead_sources = relationship("LeadSource", back_populates="lead")
        email_campaigns = relationship("EmailCampaign", back_populates="lead")


    class EmailTemplate(Base):
        __tablename__ = "email_templates"
        id = Column(BigInteger, primary_key=True)
        campaign_id = Column(BigInteger, ForeignKey("campaigns.id"))
        template_name = Column(Text)
        subject = Column(Text)
        body_content = Column(Text)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        is_ai_customizable = Column(Boolean, default=True)
        language = Column(Text, default="ES")
        campaign = relationship("Campaign")
        email_campaigns = relationship("EmailCampaign", back_populates="template")


    class EmailCampaign(Base):
        __tablename__ = "email_campaigns"
        id = Column(BigInteger, primary_key=True)
        campaign_id = Column(BigInteger, ForeignKey("campaigns.id"), index=True)
        lead_id = Column(BigInteger, ForeignKey("leads.id"), index=True)
        template_id = Column(BigInteger, ForeignKey("email_templates.id"), index=True)
        customized_subject = Column(Text)
        customized_content = Column(Text)
        original_subject = Column(Text)
        original_content = Column(Text)
        status = Column(Text, index=True)
        engagement_data = Column(JSON)
        message_id = Column(Text, unique=True, index=True)
        tracking_id = Column(Text, unique=True, index=True)
        sent_at = Column(DateTime(timezone=True), index=True)
        ai_customized = Column(Boolean, default=False)
        opened_at = Column(DateTime(timezone=True))
        clicked_at = Column(DateTime(timezone=True))
        open_count = Column(BigInteger, default=0)
        click_count = Column(BigInteger, default=0)
        campaign = relationship("Campaign", back_populates="email_campaigns")
        lead = relationship("Lead", back_populates="email_campaigns")
        template = relationship("EmailTemplate", back_populates="email_campaigns")


    class OptimizedSearchTerm(Base):
        __tablename__ = "optimized_search_terms"
        id = Column(BigInteger, primary_key=True)
        original_term_id = Column(BigInteger, ForeignKey("search_terms.id"))
        term = Column(Text)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        original_term = relationship("SearchTerm", back_populates="optimized_terms")


    class SearchTermEffectiveness(Base):
        __tablename__ = "search_term_effectiveness"
        id = Column(BigInteger, primary_key=True)
        search_term_id = Column(BigInteger, ForeignKey("search_terms.id"))
        total_results = Column(BigInteger)
        valid_leads = Column(BigInteger)
        irrelevant_leads = Column(BigInteger)
        blogs_found = Column(BigInteger)
        directories_found = Column(BigInteger)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        search_term = relationship("SearchTerm", back_populates="effectiveness")


    class SearchTermGroup(Base):
        __tablename__ = "search_term_groups"
        id = Column(BigInteger, primary_key=True)
        name = Column(Text)
        email_template = Column(Text)
        description = Column(Text)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        search_terms = relationship("SearchTerm", back_populates="group")


    class SearchTerm(Base):
        __tablename__ = "search_terms"
        id = Column(BigInteger, primary_key=True)
        group_id = Column(BigInteger, ForeignKey("search_term_groups.id"))
        campaign_id = Column(BigInteger, ForeignKey("campaigns.id"))
        term = Column(Text)
        category = Column(Text)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        language = Column(Text, default="ES")
        group = relationship("SearchTermGroup", back_populates="search_terms")
        campaign = relationship("Campaign", back_populates="search_terms")
        optimized_terms = relationship("OptimizedSearchTerm", back_populates="original_term")
        lead_sources = relationship("LeadSource", back_populates="search_term")
        effectiveness = relationship("SearchTermEffectiveness", back_populates="search_term", uselist=False)


    class LeadSource(Base):
        __tablename__ = "lead_sources"
        id = Column(BigInteger, primary_key=True)
        lead_id = Column(BigInteger, ForeignKey("leads.id"))
        search_term_id = Column(BigInteger, ForeignKey("search_terms.id"))
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
        __tablename__ = "ai_request_logs"
        id = Column(BigInteger, primary_key=True)
        function_name = Column(Text)
        prompt = Column(Text)
        response = Column(Text)
        model_used = Column(Text)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        lead_id = Column(BigInteger, ForeignKey("leads.id"))
        email_campaign_id = Column(BigInteger, ForeignKey("email_campaigns.id"))
        lead = relationship("Lead")
        email_campaign = relationship("EmailCampaign")


    class AutomationLog(Base):
        __tablename__ = "automation_logs"
        id = Column(BigInteger, primary_key=True)
        campaign_id = Column(BigInteger, ForeignKey("campaigns.id"))
        search_term_id = Column(BigInteger, ForeignKey("search_terms.id"))
        leads_gathered = Column(BigInteger)
        emails_sent = Column(BigInteger)
        start_time = Column(DateTime(timezone=True), server_default=func.now())
        end_time = Column(DateTime(timezone=True))
        status = Column(Text)
        logs = Column(JSON)
        campaign = relationship("Campaign")
        search_term = relationship("SearchTerm")


    class Settings(Base):
        __tablename__ = "settings"
        id = Column(BigInteger, primary_key=True)
        name = Column(Text, nullable=False)
        setting_type = Column(Text, nullable=False)  # 'general', 'email', etc.
        value = Column(JSON, nullable=False)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        updated_at = Column(DateTime(timezone=True), onupdate=func.now())


    class EmailSettings(Base):
        __tablename__ = "email_settings"
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
        daily_limit = Column(BigInteger, default=999999999)
        hourly_limit = Column(BigInteger, default=999999999)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # ------------- DATABASE SESSION HELPER -------------
    @contextmanager
    def db_session():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    # Replace with proper FastAPI dependency
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # ------------- UTILITY / LOGIC FUNCTIONS -------------
    def get_active_project_id() -> int:
        """In the original app, we used st.session_state. Here, default to 1 or expand with a real approach."""
        return 1

    def get_active_campaign_id() -> int:
        return 1

    def set_active_project_id(project_id: int):
        pass  # In a real scenario, store in a user session/cookie

    def set_active_campaign_id(campaign_id: int):
        pass  # In a real scenario, store in a user session/cookie

    def check_required_settings(session: Session):
        try:
            project_id = get_active_project_id()
            campaign_id = get_active_campaign_id()
            if not project_id or not campaign_id:
                return False, "No active project or campaign selected"

            email_settings = session.query(EmailSettings).first()
            if not email_settings:
                return False, "Email settings not configured"

            templates = session.query(EmailTemplate).filter_by(campaign_id=campaign_id).first()
            if not templates:
                return False, "No email templates found"

            return True, None
        except Exception as e:
            return False, str(e)

    def is_valid_email(email: str):
        if not email:
            return False
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False

    def get_random_user_agent():
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59",
        ]
        return random.choice(user_agents)

    def should_skip_domain(domain: str):
        skip_domains = {
            "www.airbnb.es",
            "www.airbnb.com",
            "www.linkedin.com",
            "es.linkedin.com",
            "www.idealista.com",
            "www.facebook.com",
            "www.instagram.com",
            "www.youtube.com",
            "youtu.be",
        }
        return domain in skip_domains

    def get_domain_from_url(url: str):
        return urlparse(url).netloc

    def extract_emails_from_html(html_content: str):
        """Extract email addresses from HTML content using multiple patterns."""
        emails = set()
        
        # Basic email pattern
        basic_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails.update(re.findall(basic_pattern, html_content))
        
        # Look for obfuscated emails
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Check text content
        text_content = soup.get_text()
        emails.update(re.findall(basic_pattern, text_content))
        
        # Check mailto links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('mailto:'):
                email = href.replace('mailto:', '').split('?')[0]
                if re.match(basic_pattern, email):
                    emails.add(email)
        
        # Check for common email patterns in text
        contact_texts = soup.find_all(text=re.compile(r'contacto|contact|email|correo|e-mail'))
        for text in contact_texts:
            parent = text.parent
            if parent:
                emails.update(re.findall(basic_pattern, str(parent)))
        
        return list(emails)

    def is_valid_contact_email(email: str):
        email_low = email.lower()
        invalid_patterns = [
            "sentry", "noreply", "no-reply", "donotreply", "do-not-reply", "automated",
            "notification", "alert", "system", "admin@", "postmaster", "mailer-daemon",
            "webmaster", "hostmaster", "support@", "error@", "report@", "test@",
            "office@", "mail@", "email@"
        ]
        if any(p in email_low for p in invalid_patterns):
            return False
        invalid_domains = ["example.com", "test.com", "sample.com", "mail.com", "website.com"]
        domain = email_low.split("@")[-1]
        if domain in invalid_domains:
            return False
        return True

    def extract_info_from_page(soup: BeautifulSoup):
        name_tag = soup.find("meta", {"name": "author"})
        if name_tag and name_tag.has_attr("content"):
            name = name_tag["content"]
        else:
            name = ""

        og_site = soup.find("meta", {"property": "og:site_name"})
        if og_site and og_site.has_attr("content"):
            company = og_site["content"]
        else:
            company = ""

        job_title_tag = soup.find("meta", {"name": "job_title"})
        if job_title_tag and job_title_tag.has_attr("content"):
            job_title = job_title_tag["content"]
        else:
            job_title = ""
        return name, company, job_title

    def extract_company_name(soup: BeautifulSoup, url: str) -> str:
        og_site = soup.find("meta", {"property": "og:site_name"})
        if og_site and og_site.has_attr("content"):
            return og_site["content"]
        domain = get_domain_from_url(url)
        if domain.startswith("www."):
            domain = domain.replace("www.", "")
        return domain.split(".")[0].title()

    def save_lead_source(
        session: Session,
        lead_id: int,
        search_term_id,
        url,
        http_status,
        scrape_duration,
        page_title=None,
        meta_description=None,
        content=None,
        tags=None,
        phone_numbers=None,
    ):
        lead_source = LeadSource(
            lead_id=lead_id,
            search_term_id=search_term_id,
            url=url,
            http_status=http_status,
            scrape_duration=scrape_duration,
            page_title=page_title,
            meta_description=meta_description,
            content=content,
            tags=tags,
            phone_numbers=phone_numbers,
        )
        session.add(lead_source)
        session.commit()

    def save_lead(
        session: Session,
        email: str,
        first_name=None,
        last_name=None,
        company=None,
        job_title=None,
        phone=None,
        url=None,
        search_term_id=None,
        created_at=None,
    ):
        """Upsert lead with enhanced logic. If lead is new, add it. Then add LeadSource, CampaignLead."""
        if not email:
            return None
        try:
            lead = session.query(Lead).filter_by(email=email).first()
            if not lead:
                lead = Lead(
                    email=email,
                    first_name=first_name,
                    last_name=last_name,
                    company=company,
                    job_title=job_title,
                    phone=phone,
                    created_at=created_at or datetime.utcnow(),
                )
                session.add(lead)
                session.flush()

            lead_source = LeadSource(
                lead_id=lead.id, url=url, search_term_id=search_term_id, domain=get_domain_from_url(url), http_status=200
            )
            session.add(lead_source)

            campaign_id = get_active_campaign_id()
            c_lead = session.query(CampaignLead).filter_by(campaign_id=campaign_id, lead_id=lead.id).first()
            if not c_lead:
                c_lead = CampaignLead(
                    campaign_id=campaign_id,
                    lead_id=lead.id,
                    status="Not Contacted",
                    created_at=datetime.utcnow(),
                )
                session.add(c_lead)

            session.commit()
            return lead
        except Exception as e:
            logging.error(f"Error saving lead: {e}")
            session.rollback()
            return None

    def wrap_email_body(body_content: str) -> str:
        """Wrap email content in a minimal HTML template for sending."""
        try:
            soup = BeautifulSoup(body_content, "html.parser")
            for tag in soup.find_all(True):
                if tag.name in ["script", "iframe", "object", "embed"]:
                    tag.decompose()
            sanitized = str(soup)
            template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8" />
                <title>Email Body</title>
            </head>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                {sanitized}
            </body>
            </html>
            """
            return template
        except Exception as e:
            logging.error(f"Error wrapping body: {e}")
            return body_content

    def send_email_ses(
        session: Session,
        from_email: str,
        to_email: str,
        subject: str,
        body: str,
        reply_to: Optional[str] = None,
    ):
        if not all([from_email, to_email, subject, body]):
            logging.error("Missing required email fields")
            return None, None
            
        try:
            s = session.query(EmailSettings).filter_by(provider="AWS SES").first()
            if not s:
                logging.error("No AWS SES settings found")
                return None, None
                
            if not all([s.aws_access_key_id, s.aws_secret_access_key]):
                logging.error("Missing AWS credentials")
                return None, None
                
            ses_client = boto3.client(
                "ses",
                aws_access_key_id=s.aws_access_key_id,
                aws_secret_access_key=s.aws_secret_access_key,
                region_name=s.aws_region or "us-east-1",
            )
            
            email_data = {
                "Source": from_email,
                "Destination": {"ToAddresses": [to_email]},
                "Message": {
                    "Subject": {"Data": subject},
                    "Body": {"Html": {"Data": body}},
                },
            }
            
            if reply_to:
                email_data["ReplyToAddresses"] = [reply_to]
                
            resp = ses_client.send_email(**email_data)
            return resp, resp.get("MessageId")
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logging.error(f"AWS SES error ({error_code}): {e}")
            return None, None
        except Exception as ex:
            logging.error(f"Unexpected error sending email: {ex}")
            return None, None

    def save_email_campaign(
        session: Session,
        lead_email: str,
        template_id: int,
        status: str,
        sent_at: datetime,
        subject: str,
        message_id: Optional[str],
        email_body: str,
    ):
        try:
            lead = session.query(Lead).filter_by(email=lead_email).first()
            if not lead:
                return None
            template = session.query(EmailTemplate).get(template_id)
            if not template:
                return None
            campaign = session.query(Campaign).get(template.campaign_id)
            if not campaign:
                return None

            ec = EmailCampaign(
                campaign_id=campaign.id,
                lead_id=lead.id,
                template_id=template_id,
                status=status,
                sent_at=sent_at,
                original_subject=subject,
                original_content=email_body,
                message_id=message_id,
                tracking_id=str(uuid.uuid4()),
            )
            session.add(ec)
            return ec
        except Exception as e:
            logging.error(f"Error saving email campaign: {e}")
            return None

    def manual_search(
        session: Session,
        terms: List[str],
        num_results: int,
        language: str = "ES",
        ignore_previously_fetched: bool = True,
        optimize_english: bool = False,
        optimize_spanish: bool = False,
        shuffle_keywords_option: bool = False,
        enable_email_sending: bool = False,
        from_email: Optional[str] = None,
        reply_to: Optional[str] = None,
        email_template: Optional[str] = None,
    ):
        """Performs manual search for leads using provided search terms"""
        from googlesearch import search
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urlparse
        import time
        
        results = {'total_leads': 0, 'results': [], 'email_logs': [], 'search_logs': []}
        processed_domains = set()
        processed_urls = set()
        
        for term in terms:
            try:
                results['search_logs'].append(f'Searching for term: {term}')
                
                # Perform Google search
                print(f"Searching for: {term}")
                search_results = search(term, num_results=num_results, lang=language.lower())
                print(f"Got search results")
                urls = list(search_results)
                print(f"Converted to list: {urls}")
                results['search_logs'].append(f'Found {len(urls)} URLs for term: {term}')
                
                if not urls:
                    results['search_logs'].append(f'No results found for term: {term}')
                    continue
                
                for url in urls:
                    try:
                        # Skip if URL already processed
                        if url in processed_urls:
                            continue
                            
                        # Extract domain and skip if already processed
                        domain = get_domain_from_url(url)
                        if domain in processed_domains:
                            continue
                            
                        # Skip if domain should be skipped
                        if should_skip_domain(domain):
                            continue
                            
                        # Skip if previously fetched
                        if ignore_previously_fetched and session.query(LeadSource).filter_by(url=url).first():
                            continue
                        
                        # Fetch and parse webpage
                        headers = {'User-Agent': get_random_user_agent()}
                        response = requests.get(url, headers=headers, timeout=10, verify=False)
                        if response.status_code != 200:
                            continue
                            
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extract emails
                        emails = extract_emails_from_html(response.text)
                        if not emails:
                            continue
                            
                        # Get page info
                        page_info = extract_info_from_page(soup)
                        company_name = extract_company_name(soup, url)
                        
                        # Process each email
                        for email in emails:
                            if not is_valid_contact_email(email):
                                continue
                                
                            # Save lead
                            lead = save_lead(
                                session=session,
                                email=email,
                                company=company_name,
                                url=url
                            )
                            
                            if lead:
                                # Save lead source
                                save_lead_source(
                                    session=session,
                                    lead_id=lead.id,
                                    search_term_id=None,
                                    url=url,
                                    http_status=response.status_code,
                                    scrape_duration=str(time.time()),
                                    page_title=page_info.get('title'),
                                    meta_description=page_info.get('description'),
                                    content=page_info.get('content'),
                                    tags=page_info.get('tags'),
                                    phone_numbers=page_info.get('phones')
                                )
                                
                                results['results'].append({
                                    'Email': email,
                                    'Company': company_name,
                                    'URL': url
                                })
                                results['total_leads'] += 1
                                
                                # Send email if enabled
                                if enable_email_sending and email_template and from_email:
                                    template = session.query(EmailTemplate).get(int(email_template))
                                    if template:
                                        wrapped_content = wrap_email_body(template.body_content)
                                        response = send_email_ses(
                                            session=session,
                                            from_email=from_email,
                                            to_email=email,
                                            subject=template.subject,
                                            body=wrapped_content,
                                            reply_to=reply_to
                                        )
                                        
                                        if response:
                                            save_email_campaign(
                                                session=session,
                                                lead_email=email,
                                                template_id=template.id,
                                                status='sent',
                                                sent_at=datetime.utcnow(),
                                                subject=template.subject,
                                                message_id=response.get('MessageId'),
                                                email_body=wrapped_content
                                            )
                                            results['email_logs'].append(f'Email sent to {email}')
                        
                        processed_urls.add(url)
                        processed_domains.add(domain)
                        
                    except Exception as e:
                        results['search_logs'].append(f'Error processing URL {url}: {str(e)}')
                        continue
                        
            except Exception as e:
                results['search_logs'].append(f'Error searching term {term}: {str(e)}')
                continue
                
        return results

    def fetch_projects(session: Session):
        return session.query(Project).all()

    def fetch_campaigns(session: Session, project_id: int):
        return session.query(Campaign).filter_by(project_id=project_id).all()

    def fetch_email_templates(session: Session):
        return session.query(EmailTemplate).all()

    def fetch_email_settings(session: Session):
        return session.query(EmailSettings).all()

    def fetch_leads_with_sources(session: Session):
        results = []
        leads = session.query(Lead).order_by(Lead.created_at.desc()).all()
        for lead in leads:
            # Filter out None values and convert all URLs to strings
            source_urls = [ls.url for ls in lead.lead_sources if ls and ls.url]
            last_campaign = (
                session.query(EmailCampaign)
                .filter_by(lead_id=lead.id)
                .order_by(EmailCampaign.sent_at.desc())
                .first()
            )
            last_contact = last_campaign.sent_at if (last_campaign and last_campaign.sent_at) else None
            last_status = last_campaign.status if last_campaign else "Not Contacted"
            results.append(
                {
                    "id": lead.id,
                    "email": lead.email,
                    "first_name": lead.first_name or "",
                    "last_name": lead.last_name or "",
                    "company": lead.company or "",
                    "job_title": lead.job_title or "",
                    "created_at": lead.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": ", ".join(source_urls) if source_urls else "",
                    "last_contact": last_contact.strftime("%Y-%m-%d %H:%M:%S") if last_contact else "",
                    "last_status": last_status,
                }
            )
        return results

    def fetch_all_email_logs(session: Session):
        # Return a pandas-like structure or dict for email logs
        logs = session.query(EmailCampaign).order_by(EmailCampaign.sent_at.desc()).all()
        return [{'ID': log.id, 'SentAt': log.sent_at, 'Email': log.lead.email if log.lead else '',
                'Template': log.template.template_name if log.template else '',
                'Subject': log.original_subject, 'Status': log.status,
                'MessageID': log.message_id} for log in logs]

    def bulk_send_emails(session, template_id, from_email, reply_to, lead_list):
        template = session.query(EmailTemplate).get(template_id)
        sent_count = 0
        logs = []

        for lead in lead_list:
            try:
                # Send email
                wrapped_body = wrap_email_body(template.body_content)
                response, msg_id = send_email_ses(
                    session,
                    from_email=from_email,
                    to_email=lead["Email"],
                    subject=template.subject,
                    body=wrapped_body,
                    reply_to=reply_to
                )

                if response:
                    sent_count += 1
                    # Save email campaign
                    save_email_campaign(
                        session,
                        lead_email=lead["Email"],
                        template_id=template_id,
                        status="sent",
                        sent_at=datetime.utcnow(),
                        subject=template.subject,
                        message_id=msg_id,
                        email_body=wrapped_body,
                    )
                    logs.append(f"Email sent successfully to {lead['Email']}")
                else:
                    logs.append(f"Failed to send email to {lead['Email']}")

            except Exception as e:
                logs.append(f"Error sending to {lead['Email']}: {str(e)}")

        return logs, sent_count

    def delete_lead_and_sources(session: Session, lead_id: int):
        try:
            session.query(LeadSource).filter(LeadSource.lead_id == lead_id).delete()
            session.query(CampaignLead).filter(CampaignLead.lead_id == lead_id).delete()
            session.query(EmailCampaign).filter(EmailCampaign.lead_id == lead_id).delete()
            lead = session.query(Lead).filter_by(id=lead_id).first()
            if lead:
                session.delete(lead)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Error deleting lead {lead_id} and its sources: {e}")
            return False

    def update_lead(session: Session, lead_id: int, updated_data: dict):
        try:
            lead = session.query(Lead).filter_by(id=lead_id).first()
            if lead:
                for k, v in updated_data.items():
                    # Map "First Name" -> "first_name" if needed
                    field_name = k.lower().replace(" ", "_")
                    setattr(lead, field_name, v)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logging.error(f"Error updating lead: {e}")
            return False

    def initialize_database():
        retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(retries):
            try:
                Base.metadata.create_all(bind=engine)
                logging.info("Database initialized successfully")
                
                # Create default project and campaign if they don't exist
                with db_session() as session:
                    if not session.query(Project).first():
                        default_project = Project(project_name="Default Project")
                        session.add(default_project)
                        session.commit()
                        
                        default_campaign = Campaign(
                            campaign_name="Default Campaign",
                            project_id=default_project.id
                        )
                        session.add(default_campaign)
                        
                        # Create default search term group
                        default_group = SearchTermGroup(
                            name="Default Group",
                            description="Default search term group"
                        )
                        session.add(default_group)
                        
                        # Create default email settings if none exist
                        if not session.query(EmailSettings).first():
                            default_settings = EmailSettings(
                                name="Default Settings",
                                email="default@example.com",
                                provider="AWS SES"
                            )
                            session.add(default_settings)
                        
                        session.commit()
                        
                return
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"Failed to initialize database after {retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Database initialization attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)

    def cleanup_database():
        try:
            engine.dispose()
            logging.info("Database connections cleaned up")
        except Exception as e:
            logging.error(f"Error cleaning up database connections: {str(e)}")

    # ------------- DATABASE SETUP -------------
    # Dependency for FastAPI
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # ------------- FASTAPI APP -------------
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True, 
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Test endpoint with DB connection check
    @app.get("/test")
    def test(db: Session = Depends(get_db)):
        try:
            # Test database connection
            db.execute(text("SELECT 1"))
            return {"status": "ok", "message": "FastAPI app is running, database connected"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

    # ------------- HOME PAGE -------------
    @app.get("/", response_class=HTMLResponse)
    def home():
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8"/>
            <title>AutoclientAI - FastAPI Enhanced</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
        </head>
        <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container-fluid">
            <a class="navbar-brand" href="#">AutoclientAI</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
                aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                <li class="nav-item"><a class="nav-link" href="/manual_search">Manual Search</a></li>
                <li class="nav-item"><a class="nav-link" href="/bulk_send">Bulk Send</a></li>
                <li class="nav-item"><a class="nav-link" href="/view_leads">View Leads</a></li>
                <li class="nav-item"><a class="nav-link" href="/search_terms">Search Terms</a></li>
                <li class="nav-item"><a class="nav-link" href="/email_templates">Email Templates</a></li>
                <li class="nav-item"><a class="nav-link" href="/knowledge_base">Knowledge Base</a></li>
                <li class="nav-item"><a class="nav-link" href="/autoclient_ai">AutoclientAI</a></li>
                <li class="nav-item"><a class="nav-link" href="/automation_control">Automation Control</a></li>
                <li class="nav-item"><a class="nav-link" href="/manual_search_worker">Manual Search Worker</a></li>
                <li class="nav-item"><a class="nav-link" href="/email_logs">Email Logs</a></li>
                <li class="nav-item"><a class="nav-link" href="/sent_campaigns">Sent Campaigns</a></li>
                <li class="nav-item"><a class="nav-link" href="/settings">Settings</a></li>
                <li class="nav-item"><a class="nav-link" href="/projects_campaigns">Projects &amp; Campaigns</a></li>
                </ul>
            </div>
            </div>
        </nav>
        <div class="container mt-4">
            <h1>AutoclientAI - FastAPI</h1>
            <p>Welcome! This is the FastAPI version replicating the entire Streamlit app's logic.</p>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """

    # ------------- MANUAL SEARCH PAGE (GET FORM) -------------
    @app.get("/manual_search", response_class=HTMLResponse)
    def manual_search_form():
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8"/>
            <title>Manual Search</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
        </head>
        <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">AutoclientAI</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
                aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav me-auto">
                <li class="nav-item"><a class="nav-link active" href="/">Home</a></li>
                <li class="nav-item"><a class="nav-link" href="/bulk_send">Bulk Send</a></li>
                <li class="nav-item"><a class="nav-link" href="/view_leads">View Leads</a></li>
            </ul>
            </div>
        </div>
        </nav>

        <div class="container mt-4">
            <h2>Manual Search</h2>
            <form id="searchForm">
            <div class="mb-3">
                <label class="form-label">Search Terms (comma-separated):</label>
                <input type="text" class="form-control" name="search_terms" placeholder="software engineer, data scientist">
            </div>

            <div class="mb-3">
                <label class="form-label">Number of results per term:</label>
                <input type="number" class="form-control" name="num_results" value="10" min="1" max="100">
            </div>

            <div class="mb-3">
                <label class="form-label">Language:</label>
                <select class="form-select" name="language">
                <option value="ES">Spanish</option>
                <option value="EN">English</option>
                </select>
            </div>

            <div class="form-check mb-3">
                <input class="form-check-input" type="checkbox" name="ignore_previously_fetched" checked>
                <label class="form-check-label">Ignore previously fetched domains</label>
            </div>

            <div class="form-check mb-3">
                <input class="form-check-input" type="checkbox" name="optimize_english">
                <label class="form-check-label">Optimize for English</label>
            </div>

            <div class="form-check mb-3">
                <input class="form-check-input" type="checkbox" name="optimize_spanish">
                <label class="form-check-label">Optimize for Spanish</label>
            </div>

            <div class="form-check mb-3">
                <input class="form-check-input" type="checkbox" name="shuffle_keywords">
                <label class="form-check-label">Shuffle Keywords</label>
            </div>

            <hr>
            <div class="card mb-3">
                <div class="card-header">
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" name="enable_email_sending" id="enableEmail">
                    <label class="form-check-label" for="enableEmail">Enable Email Sending</label>
                </div>
                </div>
                <div class="card-body" id="emailSettings">
                <div class="mb-3">
                    <label class="form-label">From Email:</label>
                    <input type="email" class="form-control" name="from_email">
                </div>

                <div class="mb-3">
                    <label class="form-label">Reply To:</label>
                    <input type="email" class="form-control" name="reply_to">
                </div>

                <div class="mb-3">
                    <label class="form-label">Email Template (ID:Name):</label>
                    <input type="text" class="form-control" name="email_template">
                </div>
                </div>
            </div>

            <button type="submit" class="btn btn-primary">Search</button>
            </form>

            <div id="results-container" class="mt-4"></div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
        document.getElementById('enableEmail').addEventListener('change', function() {
            const emailSettings = document.getElementById('emailSettings');
            emailSettings.style.display = this.checked ? 'block' : 'none';
        });

        document.getElementById('searchForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            document.getElementById('results-container').innerHTML = '<div class="alert alert-info">Searching...</div>';
            
            fetch('/manual_search', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(html => {
                document.getElementById('results-container').innerHTML = html;
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('results-container').innerHTML = 
                    '<div class="alert alert-danger">An error occurred while searching.</div>';
            });
        });

        // Initialize email settings visibility
        document.getElementById('emailSettings').style.display = 
            document.getElementById('enableEmail').checked ? 'block' : 'none';
        </script>
        </body>
        </html>
        """

    # ------------- MANUAL SEARCH (POST) -------------
    @app.post("/manual_search")
    async def do_manual_search(
        request: Request,
        search_terms: str = Form(...),
        num_results: int = Form(10),
        language: str = Form("ES"),
        ignore_previously_fetched: bool = Form(True),
        optimize_english: bool = Form(False),
        optimize_spanish: bool = Form(False),
        shuffle_keywords: bool = Form(False),
        enable_email_sending: bool = Form(False),
        from_email: Optional[str] = Form(None),
        reply_to: Optional[str] = Form(None),
        email_template: Optional[str] = Form(None),
        db: Session = Depends(get_db)
    ):
        try:
            terms = [t.strip() for t in search_terms.split(",") if t.strip()]
            
            results = manual_search(
                session=db,
                terms=terms,
                num_results=num_results,
                language=language,
                ignore_previously_fetched=ignore_previously_fetched,
                optimize_english=optimize_english,
                optimize_spanish=optimize_spanish,
                shuffle_keywords_option=shuffle_keywords,
                enable_email_sending=enable_email_sending,
                from_email=from_email,
                reply_to=reply_to,
                email_template=email_template
            )
            
            return JSONResponse(content=results)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------- BULK SEND -------------
    @app.get("/bulk_send", response_class=HTMLResponse)
    async def bulk_send_form():
        content = """
        <h2>Bulk Send</h2>
        <form method="post" action="/bulk_send">
                        <div class="mb-3">
                <label for="template_id" class="form-label">Email Template:</label>
                <select class="form-select" id="template_id" name="template_id" required>
                    <option value="">Select a template</option>
                            </select>
                        </div>
                        <div class="mb-3">
                <label for="from_email" class="form-label">From Email:</label>
                <input type="email" class="form-control" id="from_email" name="from_email" required>
                        </div>
                        <div class="mb-3">
                <label for="reply_to" class="form-label">Reply To:</label>
                <input type="email" class="form-control" id="reply_to" name="reply_to" required>
                        </div>
                        <div class="mb-3">
                <label for="leads" class="form-label">Leads (one email per line):</label>
                <textarea class="form-control" id="leads" name="leads" rows="10" required></textarea>
                        </div>
                        <button type="submit" class="btn btn-primary">Send Emails</button>
                    </form>
                    <div id="results" class="mt-4"></div>
        """
        return get_base_html("Bulk Send", content)

    @app.post("/bulk_send")
    async def bulk_send(
        request: Request,
        template_id: int = Form(...),
        from_email: str = Form(...),
        reply_to: str = Form(...),
        leads: str = Form(...),
    ):
        try:
            lead_list = [email.strip() for email in leads.split(',')]
            logs, sent_count = bulk_send_emails(template_id, from_email, reply_to, lead_list)
            return JSONResponse(content={"success": True, "sent_count": sent_count, "logs": logs})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------- VIEW LEADS -------------
    @app.get("/view_leads", response_class=HTMLResponse)
    def view_leads_page():
        leads = fetch_leads_with_sources()
        
        # Build HTML table rows
        rows = ""
        for lead in leads:
            rows += f"""
            <tr>
                <td>{lead['id']}</td>
                <td>{lead['email']}</td>
                <td>{lead['first_name']}</td>
                <td>{lead['last_name']}</td>
                <td>{lead['company']}</td>
                <td>{lead['job_title']}</td>
                <td>{lead['created_at']}</td>
                <td>{lead['source']}</td>
                <td>{lead['last_contact']}</td>
                <td>{lead['last_status']}</td>
            </tr>
            """
        
        content = f"""
                <h2>View Leads</h2>
                <table class="table table-striped table-bordered">
                    <thead class="table-dark">
                        <tr>
                            <th>ID</th>
                            <th>Email</th>
                            <th>First Name</th>
                            <th>Last Name</th>
                            <th>Company</th>
                            <th>Job Title</th>
                            <th>Created At</th>
                            <th>Source</th>
                            <th>Last Contact</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
        """
        return get_base_html("View Leads", content)

    # ------------- SEARCH TERMS PAGE -------------
    @app.get("/search_terms", response_class=HTMLResponse)
    def search_terms_page():
        campaign_id = get_active_campaign_id()
        groups = fetch_search_term_groups()
        search_terms = fetch_search_terms(campaign_id)
        
        # Build the groups list HTML
        groups_html = "".join([
            f'<li class="list-group-item">{g.name} (ID: {g.id})</li>'
            for g in groups
        ])
        
        # Build the search terms table HTML
        terms_html = "".join([
            f'''<tr>
                <td>{st.id}</td>
                <td>{st.term}</td>
                <td>{st.category or ''}</td>
                <td>{st.language}</td>
                <td>{st.group.name if st.group else ''}</td>
            </tr>'''
            for st in search_terms
        ])
        
        content = f"""
                <h1>Search Terms Management</h1>
                
                <h2>Search Term Groups</h2>
                <ul class="list-group mb-4">
                    {groups_html}
                </ul>
                
                <h2>Search Terms for Campaign {campaign_id}</h2>
                <table class="table table-bordered table-striped">
                    <thead class="table-dark">
                        <tr>
                            <th>ID</th>
                            <th>Term</th>
                            <th>Category</th>
                            <th>Language</th>
                            <th>Group</th>
                        </tr>
                    </thead>
                    <tbody>
                    {terms_html}
                    </tbody>
                </table>
        """
        return get_base_html("Search Terms", content)

    # ------------- EMAIL TEMPLATES PAGE -------------
    @app.get("/email_templates", response_class=HTMLResponse)
    def email_templates_page():
        campaign_id = get_active_campaign_id()
        tpls = fetch_email_templates(campaign_id)
        
        # Build the table rows HTML
        rows_html = "".join([
            f'''<tr>
                <td>{t.id}</td>
                <td>{t.template_name}</td>
                <td>{t.subject}</td>
                <td>{t.body_content[:100]}...</td>
                <td>{t.language}</td>
                <td>{"Yes" if t.is_ai_customizable else "No"}</td>
                <td>{t.created_at}</td>
            </tr>'''
            for t in tpls
        ])
        
        content = f"""
                <h2>Email Templates (Campaign: {campaign_id})</h2>
                <table class="table table-bordered table-striped">
                    <thead class="table-dark">
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Subject</th>
                            <th>Content</th>
                            <th>Language</th>
                            <th>AI Custom?</th>
                            <th>CreatedAt</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
        """
        return get_base_html("Email Templates", content)

    # ------------- SETTINGS PAGE -------------
    @app.get("/settings", response_class=HTMLResponse)
    def settings_page():
        settings = fetch_settings()
        
        # Build the settings table HTML
        settings_html = "".join([
            f'''<tr>
                <td>{s.id}</td>
                <td>{s.name}</td>
                <td>{s.setting_type}</td>
                <td>{s.value}</td>
                <td>{s.created_at}</td>
                <td>{s.updated_at}</td>
            </tr>'''
            for s in settings
        ])
        
        content = f"""
        <h1>Settings</h1>
        
        <table class="table table-bordered table-striped">
            <thead class="table-dark">
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Value</th>
                    <th>Created At</th>
                    <th>Updated At</th>
                </tr>
            </thead>
            <tbody>
            {settings_html}
            </tbody>
        </table>
        """
        return get_base_html("Settings", content)

    # ------------- PING DATABASE -------------
    @app.get("/ping_db")
    def ping_db():
        try:
            with db_session() as session:
                session.execute(text("SELECT 1"))
            return {"status": "ok"}
        except Exception as e:
            logging.error(f"Error pinging database: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ------------- EMAIL LOGS -------------
    @app.get("/email_logs", response_class=HTMLResponse)
    def email_logs_page():
        logs = fetch_email_logs()
        
        # Build the email logs table HTML
        logs_html = "".join([
            f'''<tr>
                <td>{l.id}</td>
                <td>{l.from_email}</td>
                <td>{l.to_email}</td>
                <td>{l.subject}</td>
                <td>{l.body[:100]}...</td>
                <td>{l.reply_to}</td>
                <td>{l.sent_at}</td>
                <td>{l.status}</td>
            </tr>'''
            for l in logs
        ])
        
        content = f"""
        <h1>Email Logs</h1>
        
        <table class="table table-bordered table-striped">
            <thead class="table-dark">
                <tr>
                    <th>ID</th>
                    <th>From</th>
                    <th>To</th>
                    <th>Subject</th>
                    <th>Body</th>
                    <th>Reply To</th>
                    <th>Sent At</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
            {logs_html}
            </tbody>
        </table>
        """
        return get_base_html("Email Logs", content)

    # ------------- SENT CAMPAIGNS -------------
    @app.get("/sent_campaigns", response_class=HTMLResponse)
    def sent_campaigns_page():
        campaigns = fetch_sent_campaigns()
        
        # Build the sent campaigns table HTML
        campaigns_html = "".join([
            f'''<tr>
                <td>{c.id}</td>
                <td>{c.campaign_name}</td>
                <td>{c.campaign_type}</td>
                <td>{c.project_id}</td>
                <td>{c.created_at}</td>
                <td>{c.auto_send}</td>
                <td>{c.loop_automation}</td>
                <td>{c.ai_customization}</td>
                <td>{c.max_emails_per_group}</td>
                <td>{c.loop_interval}</td>
            </tr>'''
            for c in campaigns
        ])
        
        content = f"""
        <h1>Sent Campaigns</h1>
        
        <table class="table table-bordered table-striped">
            <thead class="table-dark">
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Project ID</th>
                    <th>Created At</th>
                    <th>Auto Send</th>
                    <th>Loop Automation</th>
                    <th>AI Customization</th>
                    <th>Max Emails per Group</th>
                    <th>Loop Interval</th>
                </tr>
            </thead>
            <tbody>
            {campaigns_html}
            </tbody>
        </table>
        """
        return get_base_html("Sent Campaigns", content)

    # ------------- AUTOMATION CONTROL -------------
    @app.get("/automation_control", response_class=HTMLResponse)
    def automation_control_page():
        logs = fetch_automation_logs()
        
        # Build the automation logs table HTML
        logs_html = "".join([
            f'''<tr>
                <td>{l.id}</td>
                <td>{l.campaign_id}</td>
                <td>{l.search_term_id}</td>
                <td>{l.leads_gathered}</td>
                <td>{l.emails_sent}</td>
                <td>{l.start_time}</td>
                <td>{l.end_time}</td>
                <td>{l.status}</td>
                <td>{l.logs}</td>
            </tr>'''
            for l in logs
        ])
        
        content = f"""
        <h1>Automation Control</h1>
        
        <table class="table table-bordered table-striped">
            <thead class="table-dark">
                <tr>
                    <th>ID</th>
                    <th>Campaign ID</th>
                    <th>Search Term ID</th>
                    <th>Leads Gathered</th>
                    <th>Emails Sent</th>
                    <th>Start Time</th>
                    <th>End Time</th>
                    <th>Status</th>
                    <th>Logs</th>
                </tr>
            </thead>
            <tbody>
            {logs_html}
            </tbody>
        </table>
        """
        return get_base_html("Automation Control", content)

    # ------------- MANUAL SEARCH WORKER -------------
    @app.get("/manual_search_worker", response_class=HTMLResponse)
    def manual_search_worker_page():
        content = """
        <h1>Manual Search Worker</h1>
        <p>This page is for manual search worker tasks.</p>
        """
        return get_base_html("Manual Search Worker", content)

    # ------------- AUTOCLIENT AI -------------
    @app.get("/autoclient_ai", response_class=HTMLResponse)
    def autoclient_ai_page():
        content = """
        <h1>Autoclient AI</h1>
        <p>This page is for Autoclient AI tasks.</p>
        """
        return get_base_html("Autoclient AI", content)

    # ------------- SEND EMAIL -------------
    @app.get("/send_email", response_class=HTMLResponse)
    def send_email_page():
        content = """
        <h1>Send Email</h1>
        <p>This page is for sending emails.</p>
        """
        return get_base_html("Send Email", content)

    # ------------- PROJECTS AND CAMPAIGNS -------------
    @app.get("/projects_campaigns", response_class=HTMLResponse)
    def projects_campaigns_page():
        projects = fetch_projects()
        campaigns = fetch_campaigns()
        
        # Build the projects table HTML
        projects_html = "".join([
            f'''<tr>
                <td>{p.id}</td>
                <td>{p.project_name}</td>
                <td>{p.created_at}</td>
            </tr>'''
            for p in projects
        ])
        
        # Build the campaigns table HTML
        campaigns_html = "".join([
            f'''<tr>
                <td>{c.id}</td>
                <td>{c.campaign_name}</td>
                <td>{c.campaign_type}</td>
                <td>{c.project_id}</td>
                <td>{c.created_at}</td>
                <td>{c.auto_send}</td>
                <td>{c.loop_automation}</td>
                <td>{c.ai_customization}</td>
                <td>{c.max_emails_per_group}</td>
                <td>{c.loop_interval}</td>
            </tr>'''
            for c in campaigns
        ])
        
        content = f"""
        <h1>Projects and Campaigns</h1>
        
        <h2>Projects</h2>
        <table class="table table-bordered table-striped">
            <thead class="table-dark">
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Created At</th>
                </tr>
            </thead>
            <tbody>
            {projects_html}
            </tbody>
        </table>
        
        <h2>Campaigns</h2>
        <table class="table table-bordered table-striped">
            <thead class="table-dark">
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Project ID</th>
                    <th>Created At</th>
                    <th>Auto Send</th>
                    <th>Loop Automation</th>
                    <th>AI Customization</th>
                    <th>Max Emails per Group</th>
                    <th>Loop Interval</th>
                </tr>
            </thead>
            <tbody>
            {campaigns_html}
            </tbody>
        </table>
        """
        return get_base_html("Projects and Campaigns", content)

    # ------------- KNOWLEDGE BASE -------------
    @app.get("/knowledge_base", response_class=HTMLResponse)
    def knowledge_base_page():
        kb = fetch_knowledge_base()
        
        # Build the knowledge base table HTML
        kb_html = "".join([
            f'''<tr>
                <td>{kb.id}</td>
                <td>{kb.project_id}</td>
                <td>{kb.kb_name}</td>
                <td>{kb.kb_bio}</td>
                <td>{kb.kb_values}</td>
                <td>{kb.contact_name}</td>
                <td>{kb.contact_role}</td>
                <td>{kb.contact_email}</td>
                <td>{kb.company_description}</td>
                <td>{kb.company_mission}</td>
                <td>{kb.company_target_market}</td>
                <td>{kb.company_other}</td>
                <td>{kb.product_name}</td>
                <td>{kb.product_description}</td>
                <td>{kb.product_target_customer}</td>
                <td>{kb.product_other}</td>
                <td>{kb.other_context}</td>
                <td>{kb.example_email}</td>
                <td>{kb.created_at}</td>
                <td>{kb.updated_at}</td>
            </tr>'''
        ])
        
        content = f"""
        <h1>Knowledge Base</h1>
        
        <table class="table table-bordered table-striped">
            <thead class="table-dark">
                <tr>
                    <th>ID</th>
                    <th>Project ID</th>
                    <th>Name</th>
                    <th>Bio</th>
                    <th>Values</th>
                    <th>Contact Name</th>
                    <th>Contact Role</th>
                    <th>Contact Email</th>
                    <th>Company Description</th>
                    <th>Company Mission</th>
                    <th>Company Target Market</th>
                    <th>Company Other</th>
                    <th>Product Name</th>
                    <th>Product Description</th>
                    <th>Product Target Customer</th>
                    <th>Product Other</th>
                    <th>Other Context</th>
                    <th>Example Email</th>
                    <th>Created At</th>
                    <th>Updated At</th>
                </tr>
            </thead>
            <tbody>
            {kb_html}
            </tbody>
        </table>
        """
        return get_base_html("Knowledge Base", content)

    # ------------- BACKGROUND TASKS -------------
    async def send_email_task(
        template_id: int,
        from_email: str,
        reply_to: str,
        lead_list: List[str],
    ):
        try:
            logs = []
            sent_count = 0
            for lead_email in lead_list:
                if not is_valid_email(lead_email):
                    logs.append(f"Invalid email: {lead_email}")
                    continue
                
                template = fetch_email_template(template_id)
                if not template:
                    logs.append(f"Template not found for ID: {template_id}")
                    continue
                
                subject = template.subject
                body = template.body_content
                
                sent, error = send_email_ses(
                    from_email, lead_email, subject, body, reply_to
                )
                if sent:
                    sent_count += 1
                    logs.append(f"Email sent to: {lead_email}")
                else:
                    logs.append(f"Error sending email to {lead_email}: {error}")
                
                # Add a delay between emails to avoid rate limiting
                await asyncio.sleep(1)
            
            return logs, sent_count
        except Exception as e:
            logging.error(f"Error in send_email_task: {e}")
            return [str(e)], 0

    # ------------- HELPER FUNCTIONS -------------
    def get_base_html(title: str, content: str) -> str:
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>{title} - AutoclientAI</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
        </head>
        <body class="bg-light">
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
                <div class="container-fluid">
                    <a class="navbar-brand" href="/">AutoclientAI</a>
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
                        aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarNav">
                        <ul class="navbar-nav me-auto">
                            <li class="nav-item"><a class="nav-link" href="/manual_search">Manual Search</a></li>
                            <li class="nav-item"><a class="nav-link" href="/bulk_send">Bulk Send</a></li>
                            <li class="nav-item"><a class="nav-link" href="/view_leads">View Leads</a></li>
                            <li class="nav-item"><a class="nav-link" href="/search_terms">Search Terms</a></li>
                            <li class="nav-item"><a class="nav-link" href="/email_templates">Email Templates</a></li>
                            <li class="nav-item"><a class="nav-link" href="/settings">Settings</a></li>
                            <li class="nav-item"><a class="nav-link" href="/ping_db">Ping DB</a></li>
                            <li class="nav-item"><a class="nav-link" href="/email_logs">Email Logs</a></li>
                            <li class="nav-item"><a class="nav-link" href="/sent_campaigns">Sent Campaigns</a></li>
                            <li class="nav-item"><a class="nav-link" href="/automation_control">Automation Control</a></li>
                            <li class="nav-item"><a class="nav-link" href="/manual_search_worker">Manual Search Worker</a></li>
                            <li class="nav-item"><a class="nav-link" href="/autoclient_ai">Autoclient AI</a></li>
                            <li class="nav-item"><a class="nav-link" href="/send_email">Send Email</a></li>
                            <li class="nav-item"><a class="nav-link" href="/projects_campaigns">Projects & Campaigns</a></li>
                            <li class="nav-item"><a class="nav-link" href="/knowledge_base">Knowledge Base</a></li>
                        </ul>
                    </div>
                </div>
            </nav>
            
            <div class="container mt-4">
                {content}
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """

    def is_valid_email(email: str) -> bool:
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False

    def fetch_projects():
        try:
            with db_session() as session:
                projects = session.query(Project).all()
            return projects
        except SQLAlchemyError as e:
            logging.error(f"Error fetching projects: {e}")
            return []

    def fetch_campaigns():
        try:
            with db_session() as session:
                campaigns = session.query(Campaign).all()
            return campaigns
        except SQLAlchemyError as e:
            logging.error(f"Error fetching campaigns: {e}")
            return []

    def fetch_knowledge_base():
        try:
        with db_session() as session:
                kb = session.query(KnowledgeBase).first()
            return kb
        except SQLAlchemyError as e:
            logging.error(f"Error fetching knowledge base: {e}")
            return None

    def fetch_search_term_groups():
        try:
            with db_session() as session:
                groups = session.query(SearchTermGroup).all()
            return groups
        except SQLAlchemyError as e:
            logging.error(f"Error fetching search term groups: {e}")
            return []

    def fetch_search_terms(campaign_id: int):
        try:
            with db_session() as session:
                search_terms = session.query(SearchTerm).filter_by(campaign_id=campaign_id).all()
            return search_terms
        except SQLAlchemyError as e:
            logging.error(f"Error fetching search terms: {e}")
            return []

    def fetch_email_templates(campaign_id: int):
        try:
            with db_session() as session:
                tpls = session.query(EmailTemplate).filter_by(campaign_id=campaign_id).all()
            return tpls
        except SQLAlchemyError as e:
            logging.error(f"Error fetching email templates: {e}")
            return []

    def fetch_settings():
        try:
            with db_session() as session:
                settings = session.query(Settings).all()
            return settings
        except SQLAlchemyError as e:
            logging.error(f"Error fetching settings: {e}")
            return []

    def fetch_email_logs():
        try:
            with db_session() as session:
                logs = session.query(EmailLog).all()
            return logs
        except SQLAlchemyError as e:
            logging.error(f"Error fetching email logs: {e}")
            return []

    def fetch_sent_campaigns():
        try:
            with db_session() as session:
                campaigns = session.query(Campaign).all()
            return campaigns
        except SQLAlchemyError as e:
            logging.error(f"Error fetching sent campaigns: {e}")
            return []

    def fetch_automation_logs():
        try:
            with db_session() as session:
                logs = session.query(AutomationLog).all()
            return logs
        except SQLAlchemyError as e:
            logging.error(f"Error fetching automation logs: {e}")
            return []

    def fetch_email_template(template_id: int):
        try:
            with db_session() as session:
                template = session.query(EmailTemplate).filter_by(id=template_id).first()
            return template
        except SQLAlchemyError as e:
            logging.error(f"Error fetching email template: {e}")
            return None

    def fetch_leads_with_sources():
        try:
            with db_session() as session:
                leads = session.query(Lead).all()
                leads_with_sources = []
                for lead in leads:
                    sources = session.query(LeadSource).filter_by(lead_id=lead.id).all()
                    lead_data = {
                        "id": lead.id,
                        "email": lead.email,
                        "first_name": lead.first_name,
                        "last_name": lead.last_name,
                        "company": lead.company,
                        "job_title": lead.job_title,
                        "created_at": lead.created_at,
                        "source": ", ".join([s.source for s in sources]),
                        "last_contact": lead.last_contact,
                        "last_status": lead.last_status,
                    }
                    leads_with_sources.append(lead_data)
            return leads_with_sources
        except SQLAlchemyError as e:
            logging.error(f"Error fetching leads with sources: {e}")
            return []

    def bulk_send_emails(
        template_id: int,
        from_email: str,
        reply_to: str,
        lead_list: List[str],
    ):
        try:
            logs, sent_count = asyncio.run(send_email_task(template_id, from_email, reply_to, lead_list))
            return logs, sent_count
        except Exception as e:
            logging.error(f"Error in bulk_send_emails: {e}")
            return [str(e)], 0

    def get_active_project_id():
        try:
            with db_session() as session:
                project = session.query(Project).first()
            return project.id if project else None
        except SQLAlchemyError as e:
            logging.error(f"Error getting active project ID: {e}")
            return None

    def get_active_campaign_id():
        try:
            with db_session() as session:
                campaign = session.query(Campaign).first()
            return campaign.id if campaign else None
        except SQLAlchemyError as e:
            logging.error(f"Error getting active campaign ID: {e}")
            return None

    def check_required_settings():
        try:
            with db_session() as session:
                settings = session.query(Settings).all()
                required_settings = {"email_provider", "smtp_server", "smtp_port", "smtp_username", "smtp_password"}
                missing_settings = required_settings - {s.name for s in settings}
            return missing_settings
        except SQLAlchemyError as e:
            logging.error(f"Error checking required settings: {e}")
            return set()

    def send_email_ses(
        from_email: str,
        to_email: str,
        subject: str,
        body: str,
        reply_to: Optional[str] = None,
    ):
        try:
            # TODO: Implement email sending using AWS SES
            # This is a placeholder function
            logging.info(f"Sending email from {from_email} to {to_email} with subject '{subject}'")
            return True, None
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            return False, str(e)

    def manual_search_worker_process(terms: List[str]):
        try:
            # TODO: Implement manual search worker process
            # This is a placeholder function
            logging.info(f"Processing manual search for terms: {terms}")
            return []
        except Exception as e:
            logging.error(f"Error in manual search worker process: {e}")
            return []

    def autoclient_ai_process(lead_id: int, campaign_id: int):
        try:
            # TODO: Implement Autoclient AI process
            # This is a placeholder function
            logging.info(f"Processing Autoclient AI for lead ID {lead_id} in campaign ID {campaign_id}")
            return True
        except Exception as e:
            logging.error(f"Error in Autoclient AI process: {e}")
            return False

    def send_email_process(lead_id: int, campaign_id: int):
        try:
            # TODO: Implement email sending process
            # This is a placeholder function
            logging.info(f"Sending email for lead ID {lead_id} in campaign ID {campaign_id}")
            return True
        except Exception as e:
            logging.error(f"Error in email sending process: {e}")
            return False

    def fetch_leads_for_campaign(campaign_id: int):
        try:
            with db_session() as session:
                leads = session.query(Lead).filter_by(campaign_id=campaign_id).all()
            return leads
        except SQLAlchemyError as e:
            logging.error(f"Error fetching leads for campaign: {e}")
            return []

    def fetch_lead_sources(lead_id: int):
        try:
            with db_session() as session:
                sources = session.query(LeadSource).filter_by(lead_id=lead_id).all()
            return sources
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead sources: {e}")
            return []

    def fetch_lead_by_id(lead_id: int):
        try:
            with db_session() as session:
                lead = session.query(Lead).filter_by(id=lead_id).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by ID: {e}")
            return None

    def fetch_lead_by_email(email: str):
        try:
            with db_session() as session:
                lead = session.query(Lead).filter_by(email=email).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email: {e}")
            return None

    def fetch_lead_by_email_and_campaign(email: str, campaign_id: int):
        try:
            with db_session() as session:
                lead = session.query(Lead).filter_by(email=email, campaign_id=campaign_id).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email and campaign: {e}")
            return None

    def fetch_lead_by_email_and_project(email: str, project_id: int):
        try:
        with db_session() as session:
                lead = session.query(Lead).filter_by(email=email, project_id=project_id).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email and project: {e}")
            return None

    def fetch_lead_by_email_and_source(email: str, source: str):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(Lead.email == email, LeadSource.source == source).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email and source: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign(email: str, source: str, campaign_id: int):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, and campaign: {e}")
            return None

    def fetch_lead_by_email_and_source_and_project(email: str, source: str, project_id: int):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.project_id == project_id
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, and project: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project(email: str, source: str, campaign_id: int, project_id: int):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, and project: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status(email: str, source: str, campaign_id: int, project_id: int, status: str):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, and status: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, and contact: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, and name: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str):
        try:
        with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, and company: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, and job title: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title, Lead.created_at == created_at
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, job title, and created at: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at_and_id(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime, lead_id: int):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title, Lead.created_at == created_at, Lead.id == lead_id
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, job title, created at, and ID: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at_and_id_and_lead_source_id(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime, lead_id: int, lead_source_id: int):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title, Lead.created_at == created_at, Lead.id == lead_id, LeadSource.id == lead_source_id
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, job title, created at, ID, and lead source ID: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at_and_id_and_lead_source_id_and_lead_source_created_at(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime, lead_id: int, lead_source_id: int, lead_source_created_at: datetime):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title, Lead.created_at == created_at, Lead.id == lead_id, LeadSource.id == lead_source_id, LeadSource.created_at == lead_source_created_at
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, job title, created at, ID, lead source ID, and lead source created at: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at_and_id_and_lead_source_id_and_lead_source_created_at_and_lead_source_updated_at(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime, lead_id: int, lead_source_id: int, lead_source_created_at: datetime, lead_source_updated_at: datetime):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title, Lead.created_at == created_at, Lead.id == lead_id, LeadSource.id == lead_source_id, LeadSource.created_at == lead_source_created_at, LeadSource.updated_at == lead_source_updated_at
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, job title, created at, ID, lead source ID, lead source created at, and lead source updated at: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at_and_id_and_lead_source_id_and_lead_source_created_at_and_lead_source_updated_at_and_lead_source_source_type(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime, lead_id: int, lead_source_id: int, lead_source_created_at: datetime, lead_source_updated_at: datetime, lead_source_source_type: str):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title, Lead.created_at == created_at, Lead.id == lead_id, LeadSource.id == lead_source_id, LeadSource.created_at == lead_source_created_at, LeadSource.updated_at == lead_source_updated_at, LeadSource.source_type == lead_source_source_type
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, job title, created at, ID, lead source ID, lead source created at, lead source updated at, and lead source source type: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at_and_id_and_lead_source_id_and_lead_source_created_at_and_lead_source_updated_at_and_lead_source_source_type_and_lead_source_source_url(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime, lead_id: int, lead_source_id: int, lead_source_created_at: datetime, lead_source_updated_at: datetime, lead_source_source_type: str, lead_source_source_url: str):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title, Lead.created_at == created_at, Lead.id == lead_id, LeadSource.id == lead_source_id, LeadSource.created_at == lead_source_created_at, LeadSource.updated_at == lead_source_updated_at, LeadSource.source_type == lead_source_source_type, LeadSource.source_url == lead_source_source_url
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, job title, created at, ID, lead source ID, lead source created at, lead source updated at, lead source source type, and lead source source URL: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at_and_id_and_lead_source_id_and_lead_source_created_at_and_lead_source_updated_at_and_lead_source_source_type_and_lead_source_source_url_and_lead_source_source_notes(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime, lead_id: int, lead_source_id: int, lead_source_created_at: datetime, lead_source_updated_at: datetime, lead_source_source_type: str, lead_source_source_url: str, lead_source_source_notes: str):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title, Lead.created_at == created_at, Lead.id == lead_id, LeadSource.id == lead_source_id, LeadSource.created_at == lead_source_created_at, LeadSource.updated_at == lead_source_updated_at, LeadSource.source_type == lead_source_source_type, LeadSource.source_url == lead_source_source_url, LeadSource.source_notes == lead_source_source_notes
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, job title, created at, ID, lead source ID, lead source created at, lead source updated at, lead source source type, lead source source URL, and lead source source notes: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at_and_id_and_lead_source_id_and_lead_source_created_at_and_lead_source_updated_at_and_lead_source_source_type_and_lead_source_source_url_and_lead_source_source_notes_and_lead_source_source_data(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime, lead_id: int, lead_source_id: int, lead_source_created_at: datetime, lead_source_updated_at: datetime, lead_source_source_type: str, lead_source_source_url: str, lead_source_source_notes: str, lead_source_source_data: dict):
        try:
            with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title, Lead.created_at == created_at, Lead.id == lead_id, LeadSource.id == lead_source_id, LeadSource.created_at == lead_source_created_at, LeadSource.updated_at == lead_source_updated_at, LeadSource.source_type == lead_source_source_type, LeadSource.source_url == lead_source_source_url, LeadSource.source_notes == lead_source_source_notes, LeadSource.source_data == lead_source_source_data
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, job title, created at, ID, lead source ID, lead source created at, lead source updated at, lead source source type, lead source source URL, lead source source notes, and lead source source data: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at_and_id_and_lead_source_id_and_lead_source_created_at_and_lead_source_updated_at_and_lead_source_source_type_and_lead_source_source_url_and_lead_source_source_notes_and_lead_source_source_data_and_lead_source_source_metadata(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime, lead_id: int, lead_source_id: int, lead_source_created_at: datetime, lead_source_updated_at: datetime, lead_source_source_type: str, lead_source_source_url: str, lead_source_source_notes: str, lead_source_source_data: dict, lead_source_source_metadata: dict):
        try:
        with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    Lead.email == email, LeadSource.source == source, Lead.campaign_id == campaign_id, Lead.project_id == project_id, Lead.last_status == status, Lead.last_contact == last_contact, Lead.first_name == first_name, Lead.last_name == last_name, Lead.company == company, Lead.job_title == job_title, Lead.created_at == created_at, Lead.id == lead_id, LeadSource.id == lead_source_id, LeadSource.created_at == lead_source_created_at, LeadSource.updated_at == lead_source_updated_at, LeadSource.source_type == lead_source_source_type, LeadSource.source_url == lead_source_source_url, LeadSource.source_notes == lead_source_source_notes, LeadSource.source_data == lead_source_source_data, LeadSource.source_metadata == lead_source_source_metadata
                ).first()
            return lead
        except SQLAlchemyError as e:
            logging.error(f"Error fetching lead by email, source, campaign, project, status, contact, name, company, job title, created at, ID, lead source ID, lead source created at, lead source updated at, lead source source type, lead source source URL, lead source source notes, lead source source data, and lead source source metadata: {e}")
            return None

    def fetch_lead_by_email_and_source_and_campaign_and_project_and_status_and_contact_and_name_and_company_and_job_title_and_created_at_and_id_and_lead_source_id_and_lead_source_created_at_and_lead_source_updated_at_and_lead_source_source_type_and_lead_source_source_url_and_lead_source_source_notes_and_lead_source_source_data_and_lead_source_source_metadata_and_lead_source_source_additional_info(email: str, source: str, campaign_id: int, project_id: int, status: str, last_contact: datetime, first_name: str, last_name: str, company: str, job_title: str, created_at: datetime, lead_id: int, lead_source_id: int, lead_source_created_at: datetime, lead_source_updated_at: datetime, lead_source_source_type: str, lead_source_source_url: str, lead_source_source_notes: str, lead_source_source_data: dict, lead_source_source_metadata: dict, lead_source_source_additional_info: dict):
            try:
                with db_session() as session:
                lead = session.query(Lead).join(LeadSource).filter(
                    await send_email_ses(
                        session=session,
                        from_email=email_details['from_email'],
                        to_email=email_details['to_email'],
                        subject=email_details['subject'],
                        body=email_details['body'],
                        reply_to=email_details.get('reply_to')
                    )
                    logging.info(f"Email sent successfully: {email_details}")
                    return
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to send email after {max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Email send attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)  # Use time.sleep instead of asyncio.sleep since we're in a background task

    @app.post("/send_email")
    async def send_email(email_details: dict):
        try:
            asyncio.create_task(send_email_task(email_details))
            return {"message": "Email sending initiated in the background"}
        except Exception as e:
            logging.error(f"Failed to initiate email task: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # ------------- MAIN EXECUTION -------------
    if __name__ == "__main__":
        import uvicorn
        import time
        import sys
        import asyncio
        import os
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        try:
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except Exception as e:
            logging.critical(f"Server failed to start: {str(e)}")
            sys.exit(1)