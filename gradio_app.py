import os
import json
import re
import logging
import asyncio
import time
import requests
import pandas as pd
import openai
import boto3
import uuid
import aiohttp
import urllib3
import random
import html
import smtplib
import gradio as gr
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import (
    func, create_engine, Column, BigInteger, Text, DateTime, 
    ForeignKey, Boolean, JSON, select, text, distinct, and_
)
from sqlalchemy.orm import (
    declarative_base, sessionmaker, relationship, Session, 
    joinedload
)
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import (
    retry, stop_after_attempt, wait_random_exponential, wait_fixed
)
from email_validator import validate_email, EmailNotValidError
from openai import OpenAI
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urlencode
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager

# Database configuration
load_dotenv()
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Database setup
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Database Models
class Project(Base):
    __tablename__ = 'projects'
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text, default="Default Project")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaigns = relationship("Campaign", back_populates="project")
    knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False)

# Context manager for database sessions
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

# Utility functions
def get_active_project_id() -> int:
    return 1  # Default project ID for Gradio app

def get_active_campaign_id() -> int:
    return 1  # Default campaign ID for Gradio app

def manual_search(session: Session, terms: List[str], num_results: int, 
                 ignore_previously_fetched: bool = True, 
                 optimize_english: bool = False,
                 optimize_spanish: bool = False,
                 shuffle_keywords_option: bool = False,
                 language: str = 'ES',
                 enable_email_sending: bool = True,
                 log_container = None,
                 from_email: Optional[str] = None,
                 reply_to: Optional[str] = None,
                 email_template: Optional[str] = None) -> Dict[str, Any]:
    [... Copy manual_search implementation ...]

class GradioAutoclientApp:
    def __init__(self):
        self.automation_status = False
        self.automation_logs = []
        self.total_leads_found = 0
        self.total_emails_sent = 0
        
        # Initialize database
        Base.metadata.create_all(bind=engine)
        
    def fetch_template_names(self) -> List[str]:
        with db_session() as session:
            templates = session.query(EmailTemplate).all()
            return [f"{t.id}: {t.template_name}" for t in templates]
    
    def fetch_email_settings_names(self) -> List[str]:
        with db_session() as session:
            settings = session.query(EmailSettings).all()
            return [f"{s.id}: {s.name} ({s.email})" for s in settings]
    
    def fetch_search_terms(self) -> List[str]:
        with db_session() as session:
            terms = session.query(SearchTerm).all()
            return [f"{t.id}: {t.term}" for t in terms]

    [Copy all the methods from the Streamlit app and convert them to work with Gradio]

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown("# AutoclientAI - Lead Generation Platform")
            
            # Manual Search Tab
            with gr.Tab("Manual Search"):
                with gr.Row():
                    with gr.Column():
                        search_terms = gr.Textbox(
                            label="Search Terms (one per line)",
                            lines=5,
                            placeholder="Enter search terms..."
                        )
                        num_results = gr.Slider(
                            label="Results per term",
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1
                        )
                        with gr.Row():
                            ignore_fetched = gr.Checkbox(label="Ignore fetched domains", value=True)
                            shuffle_keywords = gr.Checkbox(label="Shuffle Keywords", value=True)
                            optimize_english = gr.Checkbox(label="Optimize (English)", value=False)
                            optimize_spanish = gr.Checkbox(label="Optimize (Spanish)", value=False)
                        
                        language = gr.Dropdown(choices=["ES", "EN"], label="Language", value="ES")
                        enable_email = gr.Checkbox(label="Enable email sending", value=True)
                        
                        with gr.Group(visible=False) as email_group:
                            template_dropdown = gr.Dropdown(
                                choices=self.fetch_template_names(),
                                label="Email Template"
                            )
                            email_settings_dropdown = gr.Dropdown(
                                choices=self.fetch_email_settings_names(),
                                label="From Email"
                            )
                            reply_to = gr.Textbox(label="Reply To")
                        
                        search_btn = gr.Button("Search", variant="primary")
                    
                    with gr.Column():
                        results_table = gr.Dataframe(
                            headers=["Email", "URL", "Source", "Title"],
                            label="Search Results"
                        )
                        log_output = gr.Textbox(label="Logs", lines=10, interactive=False)
                        
                        with gr.Row():
                            export_btn = gr.Button("Export Results")
                            clear_btn = gr.Button("Clear Results")

            # Bulk Send Tab
            with gr.Tab("Bulk Send"):
                with gr.Row():
                    with gr.Column():
                        template_select = gr.Dropdown(
                            choices=self.fetch_template_names(),
                            label="Email Template"
                        )
                        email_setting_select = gr.Dropdown(
                            choices=self.fetch_email_settings_names(),
                            label="From Email"
                        )
                        reply_to_bulk = gr.Textbox(label="Reply To")
                        
                        send_option = gr.Radio(
                            choices=["All Leads", "Specific Email", "Leads from Search Terms", "Leads from Search Term Groups"],
                            label="Send to",
                            value="All Leads"
                        )
                        
                        with gr.Group() as specific_email_group:
                            specific_email = gr.Textbox(label="Enter email")
                        
                        with gr.Group() as search_terms_group:
                            selected_terms = gr.Dropdown(
                                choices=self.fetch_search_terms(),
                                label="Select Search Terms",
                                multiselect=True
                            )
                        
                        exclude_contacted = gr.Checkbox(label="Exclude Previously Contacted", value=True)
                        send_btn = gr.Button("Send Emails", variant="primary")
                    
                    with gr.Column():
                        preview = gr.HTML(label="Email Preview")
                        send_progress = gr.Progress()
                        send_logs = gr.Textbox(label="Sending Logs", lines=10, interactive=False)

            # View Leads Tab
            with gr.Tab("View Leads"):
                with gr.Row():
                    with gr.Column():
                        search_leads = gr.Textbox(label="Search leads")
                        leads_table = gr.Dataframe(
                            headers=["ID", "Email", "Name", "Company", "Source", "Last Contact"],
                            label="Leads"
                        )
                        refresh_leads = gr.Button("Refresh")
                    
                    with gr.Column():
                        lead_stats = gr.HTML(label="Lead Statistics")
                        lead_growth = gr.Plot(label="Lead Growth")
                        export_leads = gr.Button("Export Leads")

            # Search Terms Tab
            with gr.Tab("Search Terms"):
                with gr.Row():
                    with gr.Column():
                        new_term = gr.Textbox(label="New Search Term")
                        term_group = gr.Dropdown(
                            choices=self.fetch_search_term_groups(),
                            label="Assign to Group"
                        )
                        add_term_btn = gr.Button("Add Term")
                    
                    with gr.Column():
                        terms_table = gr.Dataframe(
                            headers=["Term", "Lead Count", "Email Count"],
                            label="Search Terms"
                        )
                        term_stats = gr.HTML(label="Term Statistics")
                        optimize_terms = gr.Button("Optimize Terms with AI")

            # Email Templates Tab
            with gr.Tab("Email Templates"):
                with gr.Row():
                    with gr.Column():
                        template_name = gr.Textbox(label="Template Name")
                        template_subject = gr.Textbox(label="Subject")
                        template_body = gr.TextArea(label="Body Content", lines=10)
                        is_ai_customizable = gr.Checkbox(label="AI Customizable")
                        template_language = gr.Dropdown(choices=["ES", "EN"], label="Language")
                        save_template_btn = gr.Button("Save Template")
                    
                    with gr.Column():
                        template_list = gr.Dropdown(
                            choices=self.fetch_template_names(),
                            label="Existing Templates"
                        )
                        template_preview = gr.HTML(label="Template Preview")
                        delete_template = gr.Button("Delete Template")

            # Projects & Campaigns Tab
            with gr.Tab("Projects & Campaigns"):
                with gr.Row():
                    with gr.Column():
                        project_name = gr.Textbox(label="Project Name")
                        add_project = gr.Button("Add Project")
                        project_list = gr.Dropdown(
                            choices=self.fetch_projects(),
                            label="Projects"
                        )
                    
                    with gr.Column():
                        campaign_name = gr.Textbox(label="Campaign Name")
                        campaign_type = gr.Dropdown(
                            choices=["Email", "LinkedIn", "Other"],
                            label="Campaign Type"
                        )
                        add_campaign = gr.Button("Add Campaign")
                        campaign_list = gr.Dropdown(
                            choices=self.fetch_campaigns(),
                            label="Campaigns"
                        )

            # Knowledge Base Tab
            with gr.Tab("Knowledge Base"):
                with gr.Row():
                    with gr.Column():
                        kb_name = gr.Textbox(label="Knowledge Base Name")
                        kb_bio = gr.TextArea(label="Bio", lines=3)
                        kb_values = gr.TextArea(label="Values", lines=3)
                        contact_info = gr.Group(label="Contact Information")
                        with contact_info:
                            contact_name = gr.Textbox(label="Contact Name")
                            contact_role = gr.Textbox(label="Contact Role")
                            contact_email = gr.Textbox(label="Contact Email")
                    
                    with gr.Column():
                        company_info = gr.Group(label="Company Information")
                        with company_info:
                            company_description = gr.TextArea(label="Description", lines=3)
                            company_mission = gr.TextArea(label="Mission", lines=3)
                            company_target_market = gr.TextArea(label="Target Market", lines=3)
                            company_other = gr.TextArea(label="Other Info", lines=3)
                        
                        product_info = gr.Group(label="Product Information")
                        with product_info:
                            product_name = gr.Textbox(label="Product Name")
                            product_description = gr.TextArea(label="Description", lines=3)
                            product_target_customer = gr.TextArea(label="Target Customer", lines=3)
                            product_other = gr.TextArea(label="Other Info", lines=3)
                        
                        save_kb = gr.Button("Save Knowledge Base")

            # AutoclientAI Tab
            with gr.Tab("AutoclientAI"):
                with gr.Row():
                    with gr.Column():
                        ai_prompt = gr.TextArea(label="AI Instructions", lines=5)
                        use_kb = gr.Checkbox(label="Use Knowledge Base")
                        generate_btn = gr.Button("Generate Content")
                    
                    with gr.Column():
                        ai_output = gr.HTML(label="Generated Content")
                        ai_logs = gr.Textbox(label="AI Logs", lines=5)

            # Automation Control Tab
            with gr.Tab("Automation Control"):
                with gr.Row():
                    with gr.Column():
                        auto_send = gr.Checkbox(label="Enable Auto-send")
                        loop_automation = gr.Checkbox(label="Loop Automation")
                        ai_customization = gr.Checkbox(label="AI Customization")
                        max_emails = gr.Slider(label="Max Emails per Group", minimum=1, maximum=1000, value=500)
                        loop_interval = gr.Slider(label="Loop Interval (minutes)", minimum=1, maximum=1440, value=60)
                    
                    with gr.Column():
                        automation_status = gr.HTML(label="Automation Status")
                        automation_logs = gr.Textbox(label="Automation Logs", lines=10)
                        start_stop = gr.Button("Start/Stop Automation")

            # Email Logs Tab
            with gr.Tab("Email Logs"):
                with gr.Row():
                    with gr.Column():
                        date_range = gr.DateRange(label="Date Range")
                        log_search = gr.Textbox(label="Search logs")
                        logs_table = gr.Dataframe(
                            headers=["Date", "Email", "Subject", "Status"],
                            label="Email Logs"
                        )
                    
                    with gr.Column():
                        email_stats = gr.HTML(label="Email Statistics")
                        success_rate = gr.Plot(label="Success Rate Over Time")
                        export_logs = gr.Button("Export Logs")

            # Settings Tab
            with gr.Tab("Settings"):
                with gr.Row():
                    with gr.Column():
                        openai_key = gr.Textbox(label="OpenAI API Key", type="password")
                        openai_base = gr.Textbox(label="OpenAI API Base URL")
                        openai_model = gr.Textbox(label="OpenAI Model")
                        save_general = gr.Button("Save General Settings")
                    
                    with gr.Column():
                        email_settings = gr.Group(label="Email Settings")
                        with email_settings:
                            setting_name = gr.Textbox(label="Setting Name")
                            email = gr.Textbox(label="Email")
                            provider = gr.Radio(choices=["smtp", "ses"], label="Provider")
                            
                            with gr.Group() as smtp_settings:
                                smtp_server = gr.Textbox(label="SMTP Server")
                                smtp_port = gr.Number(label="SMTP Port")
                                smtp_username = gr.Textbox(label="SMTP Username")
                                smtp_password = gr.Textbox(label="SMTP Password", type="password")
                            
                            with gr.Group() as ses_settings:
                                aws_access_key = gr.Textbox(label="AWS Access Key")
                                aws_secret_key = gr.Textbox(label="AWS Secret Key", type="password")
                                aws_region = gr.Textbox(label="AWS Region")
                            
                            save_email = gr.Button("Save Email Settings")

            # Sent Campaigns Tab
            with gr.Tab("Sent Campaigns"):
                with gr.Row():
                    with gr.Column():
                        campaign_filter = gr.Dropdown(
                            choices=self.fetch_campaigns(),
                            label="Filter by Campaign"
                        )
                        campaign_date = gr.DateRange(label="Date Range")
                        campaign_table = gr.Dataframe(
                            headers=["Campaign", "Sent", "Opened", "Clicked", "Success Rate"],
                            label="Campaign Performance"
                        )
                    
                    with gr.Column():
                        campaign_stats = gr.HTML(label="Campaign Statistics")
                        campaign_chart = gr.Plot(label="Campaign Performance Chart")
                        export_campaign = gr.Button("Export Campaign Data")

            # Event handlers
            search_btn.click(
                fn=self.perform_search,
                inputs=[search_terms, num_results, ignore_fetched, shuffle_keywords,
                       optimize_english, optimize_spanish, language],
                outputs=[results_table, log_output]
            )
            
            enable_email.change(
                fn=lambda x: gr.Group(visible=x),
                inputs=[enable_email],
                outputs=[email_group]
            )
            
            template_select.change(
                fn=self.preview_template,
                inputs=[template_select],
                outputs=[preview]
            )
            
            send_option.change(
                fn=self.update_send_options,
                inputs=[send_option],
                outputs=[specific_email_group, search_terms_group]
            )
            
            send_btn.click(
                fn=self.send_bulk_emails,
                inputs=[template_select, email_setting_select, reply_to_bulk,
                       send_option, specific_email, selected_terms],
                outputs=[send_logs]
            )
            
            # Add more event handlers for other tabs...

            return app

    def perform_search(self, search_terms: str, num_results: int, 
                      ignore_fetched: bool, shuffle_keywords: bool,
                      optimize_english: bool, optimize_spanish: bool,
                      language: str) -> tuple:
        terms = [term.strip() for term in search_terms.split('\n') if term.strip()]
        results = []
        logs = []
        
        with db_session() as session:
            for term in terms:
                try:
                    search_results = manual_search(
                        session, [term], num_results, ignore_fetched,
                        optimize_english, optimize_spanish, shuffle_keywords,
                        language, enable_email_sending=False
                    )
                    results.extend(search_results['results'])
                    logs.append(f"Found {len(search_results['results'])} results for '{term}'")
                except Exception as e:
                    logs.append(f"Error searching for '{term}': {str(e)}")
        
        return (
            pd.DataFrame(results)[["Email", "URL", "Lead Source", "Title"]],
            "\n".join(logs)
        )

    def save_template(self, name: str, subject: str, body: str, 
                     is_ai_customizable: bool, language: str) -> List[str]:
        with db_session() as session:
            template = EmailTemplate(
                template_name=name,
                subject=subject,
                body_content=body,
                is_ai_customizable=is_ai_customizable,
                language=language,
                campaign_id=1  # Default campaign
            )
            session.add(template)
            session.commit()
            return self.fetch_template_names()

    def preview_template(self, template_id: str) -> str:
        template_id = int(template_id.split(":")[0])
        with db_session() as session:
            template = session.query(EmailTemplate).get(template_id)
            if template:
                return wrap_email_body(template.body_content)
            return "<p>Template not found</p>"

    def update_send_options(self, send_option: str) -> tuple:
        return (
            gr.update(visible=send_option == "Specific Email"),
            gr.update(visible=send_option == "Leads from Search Terms")
        )

    def send_bulk_emails(self, template_id: str, from_email: str, 
                        reply_to: str, send_option: str,
                        specific_email: str, selected_terms: List[str]) -> str:
        template_id = int(template_id.split(":")[0])
        logs = []
        
        with db_session() as session:
            leads = fetch_leads(
                session, template_id, send_option,
                specific_email, selected_terms,
                exclude_previously_contacted=True
            )
            
            email_logs, sent_count = bulk_send_emails(
                session, template_id,
                from_email.split("(")[1].rstrip(")"),
                reply_to, leads
            )
            logs.extend(email_logs)
        
        return "\n".join(logs)

def create_app():
    app = GradioAutoclientApp()
    interface = app.create_interface()
    return interface

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)