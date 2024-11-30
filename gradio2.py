import os, re, logging, time, requests, pandas as pd, boto3, uuid, urllib3, smtplib, gradio as gr
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import (func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, Integer, Float, or_)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.exc import SQLAlchemyError
from email_validator import validate_email, EmailNotValidError
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import plotly.express as px

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
    description = Column(Text)
    project_type = Column(Text)
    status = Column(Text, default="Active")
    priority = Column(Integer, default=3)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaigns = relationship("Campaign", back_populates="project")
    knowledge_bases = relationship("KnowledgeBase", back_populates="project")

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
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    project = relationship("Project", back_populates="campaigns")
    tasks = relationship("AutomationTask", back_populates="campaign")

# Additional Models...

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

# Utility Functions
def is_valid_email_address(email):
    if not email:
        return False
    invalid_patterns = [
        r".*\.(png|jpg|jpeg|gif|css|js)$",
        r"^(nr|bootstrap|jquery|core|icon-|noreply)@.*",
        r"^(email|info|contact|support|hello|hola|hi|salutations|greetings|inquiries|questions)@.*",
        r"^email@email\.com$",
        r".*@example\.com$",
        r".*@.*\.(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$"
    ]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    if any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns):
        return False
    if any(email.lower().endswith(f"@{domain}") for domain in typo_domains):
        return False
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def wrap_email_content(content: str) -> str:
    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .button {{ display: inline-block; padding: 10px 20px; background-color: #007bff;
                      color: white; text-decoration: none; border-radius: 5px; }}
            .signature {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <div class="container">
            {content}
        </div>
    </body>
    </html>
    """

def send_email(session: Session, from_email: str, to_email: str, subject: str, body: str, reply_to: Optional[str] = None) -> tuple:
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings:
        logging.error(f"No email settings found for {from_email}")
        return None, None

    tracking_id = str(uuid.uuid4())
    tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
    wrapped_body = wrap_email_content(body)
    tracked_body = wrapped_body.replace('</body>', f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/></body>')

    soup = BeautifulSoup(tracked_body, 'html.parser')
    for a in soup.find_all('a', href=True):
        original_url = a['href']
        tracked_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'click', 'url': original_url})}"
        a['href'] = tracked_url
    tracked_body = str(soup)

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
                    'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                    'Body': {'Html': {'Data': tracked_body, 'Charset': 'UTF-8'}}
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
            msg.attach(MIMEText(tracked_body, 'html'))

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

class GradioAutoclientApp:
    def __init__(self):
        Base.metadata.create_all(bind=engine)
        self.stop_sending_flag = False

    def perform_search(self, search_terms, num_results, ignore_fetched, shuffle_keywords, optimize_english, optimize_spanish, language, enable_email=False, template=None, email_setting=None, reply_to=None):
        results, logs = [], []
        total_leads = 0
        with db_session() as session:
            terms = [term.strip() for term in search_terms.split('\n') if term.strip()]
            from_email = session.query(EmailSettings).get(int(email_setting.split(':')[0])).email if enable_email and email_setting else None
            for term in terms:
                optimized_term = self.optimize_search_term(term, optimize_english, optimize_spanish, shuffle_keywords)
                for url in google_search(optimized_term, num_results, lang=language):
                    try:
                        response = requests.get(url, timeout=10, verify=False, headers={'User-Agent': UserAgent().random})
                        soup = BeautifulSoup(response.text, 'html.parser')
                        emails = [email for email in extract_emails_from_html(response.text) if is_valid_email_address(email)]
                        for email in emails:
                            lead = self.save_lead(session, email, url, term)
                            if lead:
                                total_leads += 1
                                results.append(self.format_lead(lead, url, term))
                                if enable_email and from_email and template:
                                    self.send_campaign_email(session, from_email, email, template, reply_to)
                    except requests.RequestException as e:
                        logs.append(f"Error fetching {url}: {e}")
        logs.append(f"Total leads found: {total_leads}")
        return results, "\n".join(logs)

    def optimize_search_term(self, term, optimize_english, optimize_spanish, shuffle_keywords):
        if shuffle_keywords:
            term = shuffle_keywords_function(term)
        if optimize_english:
            term = optimize_term(term, 'english')
        if optimize_spanish:
            term = optimize_term(term, 'spanish')
        return term

    def save_lead(self, session, email, url, term_id):
        lead = session.query(Lead).filter_by(email=email).first()
        if not lead:
            lead = Lead(email=email, url=url, search_term_id=term_id)
            session.add(lead)
            session.commit()
        return lead

    def format_lead(self, lead, url, term):
        return {
            'Email': lead.email,
            'URL': url,
            'Lead Source': term,
            'Name': lead.first_name or "N/A",
            'Company': lead.company or "N/A",
            'Job Title': lead.job_title or "N/A"
        }

    def send_campaign_email(self, session, from_email, to_email, template_id, reply_to):
        template = session.query(EmailTemplate).get(int(template_id.split(":")[0]))
        if template:
            response, tracking_id = send_email(session, from_email, to_email, template.subject, template.body_content, reply_to)
            status = 'Sent' if response else 'Failed'
            EmailCampaign(
                lead_id=session.query(Lead).filter_by(email=to_email).first().id,
                template_id=template.id,
                status=status,
                sent_at=datetime.utcnow(),
                subject=template.subject,
                message_id=response['MessageId'] if response else None,
                email_body=template.body_content,
                created_at=datetime.utcnow(),
                open_count=0,
                click_count=0
            )
            session.commit()

    def create_ui(self):
        with gr.Blocks(title="Autoclient Email Crawler") as app:
            gr.Markdown("# Autoclient Email Crawler")
            with gr.Tabs():
                with gr.Tab("Manual Search"):
                    self.create_manual_search_tab()
                with gr.Tab("Email Templates"):
                    self.create_email_templates_tab()
                with gr.Tab("Bulk Email"):
                    self.create_bulk_email_tab()
                # Additional Tabs...

            return app

    def create_manual_search_tab(self):
        with gr.Row():
            with gr.Column(scale=2):
                search_terms = gr.Textbox(label="Search Terms", placeholder="One per line", lines=5)
                num_results = gr.Slider(label="Results Per Term", minimum=1, maximum=50, value=10, step=1)
            with gr.Column(scale=1):
                enable_email = gr.Checkbox(label="Enable Email Sending", value=False)
                ignore_fetched = gr.Checkbox(label="Ignore Fetched Domains", value=True)
                shuffle_keywords = gr.Checkbox(label="Shuffle Keywords", value=False)
                optimize_english = gr.Checkbox(label="Optimize (English)", value=False)
                optimize_spanish = gr.Checkbox(label="Optimize (Spanish)", value=False)
                language = gr.Dropdown(label="Language", choices=["EN", "ES"], value="ES")
        with gr.Group(visible=False) as email_group:
            with gr.Row():
                template_dropdown = gr.Dropdown(label="Email Template", choices=self.fetch_templates())
                email_settings_dropdown = gr.Dropdown(label="From Email", choices=self.fetch_email_settings())
            reply_to = gr.Textbox(label="Reply-To Email", placeholder="optional@example.com")
        search_btn = gr.Button("Search", variant="primary")
        results_table = gr.Dataframe(headers=["Email", "URL", "Name", "Company", "Job Title"], label="Results")
        log_output = gr.HTML(label="Logs")
        with gr.Row():
            export_btn = gr.Button("Export Results")
            clear_btn = gr.Button("Clear Results")
        enable_email.change(fn=lambda x: gr.Group(visible=x), inputs=[enable_email], outputs=[email_group])
        search_btn.click(fn=self.perform_search, inputs=[
            search_terms, num_results, ignore_fetched, shuffle_keywords,
            optimize_english, optimize_spanish, language, enable_email,
            template_dropdown, email_settings_dropdown, reply_to
        ], outputs=[results_table, log_output])
        export_btn.click(fn=self.export_to_csv, inputs=[results_table], outputs=[gr.File(label="Download CSV")])
        clear_btn.click(fn=lambda: (None, ""), inputs=[], outputs=[results_table, log_output])

    def create_email_templates_tab(self):
        with gr.Row():
            with gr.Column():
                template_name = gr.Textbox(label="Template Name")
                subject = gr.Textbox(label="Subject")
                body = gr.Textarea(label="Body Content", lines=10)
                is_ai_customizable = gr.Checkbox(label="AI Customizable", value=False)
                language = gr.Dropdown(label="Language", choices=["EN", "ES"], value="EN")
            with gr.Column():
                save_btn = gr.Button("Save Template", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
        save_btn.click(fn=self.save_template, inputs=[
            template_name, subject, body, is_ai_customizable, language
        ], outputs=[status])

    def create_bulk_email_tab(self):
        with gr.Row():
            with gr.Column(scale=2):
                template_select = gr.Dropdown(label="Select Template", choices=self.fetch_templates())
                email_setting = gr.Dropdown(label="From Email", choices=self.fetch_email_settings())
                reply_to = gr.Textbox(label="Reply-To Email", placeholder="optional@example.com")
                send_option = gr.Radio(
                    label="Send To",
                    choices=[
                        "All Leads",
                        "Specific Email",
                        "Search Terms",
                        "Groups"
                    ],
                    value="All Leads"
                )
                with gr.Group(visible=False) as specific_email_group:
                    specific_email = gr.Textbox(label="Specific Email")
                with gr.Group(visible=False) as search_terms_group:
                    search_terms = gr.Dropdown(label="Select Search Terms", choices=self.fetch_search_terms(), multiselect=True)
                send_option.change(fn=self.toggle_send_options, inputs=[send_option], outputs=[
                    specific_email_group, search_terms_group
                ])
                send_btn = gr.Button("Send Emails", variant="primary")
            with gr.Column(scale=2):
                results_table = gr.Dataframe(headers=["Email", "Status", "Sent At"], label="Sending Results")
                log_output = gr.HTML(label="Logs")
                with gr.Row():
                    export_btn = gr.Button("Export Logs")
                    stop_btn = gr.Button("Stop Sending")
        send_btn.click(fn=self.send_bulk_emails, inputs=[
            template_select, email_setting, reply_to, send_option,
            specific_email, search_terms
        ], outputs=[results_table, log_output])
        export_btn.click(fn=self.export_to_csv, inputs=[results_table], outputs=[gr.File(label="Download Logs")])
        stop_btn.click(fn=self.stop_sending_emails, inputs=[], outputs=[log_output])

    # Additional UI creation methods...

    def fetch_templates(self):
        with db_session() as session:
            templates = session.query(EmailTemplate).all()
            return [f"{t.id}: {t.template_name}" for t in templates]

    def fetch_email_settings(self):
        with db_session() as session:
            settings = session.query(EmailSettings).all()
            return [f"{s.id}: {s.name} ({s.email})" for s in settings]

    def fetch_search_terms(self):
        with db_session() as session:
            terms = session.query(SearchTerm).all()
            return [f"{t.id}: {t.term}" for t in terms]

    def save_template(self, name, subject, body, ai_customizable, language):
        with db_session() as session:
            try:
                template_id = create_or_update_email_template(session, name, subject, body, is_ai_customizable=ai_customizable, language=language)
                return f"Template '{name}' saved successfully with ID {template_id}."
            except ValueError as e:
                return f"Error: {e}"

    def export_to_csv(self, data):
        if not data:
            return None
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        return csv

    def toggle_send_options(self, option):
        return (
            gr.Group(visible=option == "Specific Email"),
            gr.Group(visible=option == "Search Terms")
        )

    def send_bulk_emails(self, template, email_setting, reply_to, send_option, specific_email, search_terms):
        logs, results = [], []
        with db_session() as session:
            template_obj = session.query(EmailTemplate).get(int(template.split(':')[0]))
            email_setting_obj = session.query(EmailSettings).get(int(email_setting.split(':')[0]))
            leads = []
            if send_option == "All Leads":
                leads = session.query(Lead).all()
            elif send_option == "Specific Email" and specific_email:
                lead = session.query(Lead).filter_by(email=specific_email).first()
                if lead:
                    leads.append(lead)
            elif send_option == "Search Terms" and search_terms:
                term_ids = [int(t.split(':')[0]) for t in search_terms]
                leads = session.query(Lead).filter(Lead.search_term_id.in_(term_ids)).all()
            # Additional options...
            for lead in leads:
                response, tracking_id = send_email(session, email_setting_obj.email, lead.email, template_obj.subject, template_obj.body_content, reply_to)
                status = 'Sent' if response else 'Failed'
                results.append({
                    'Email': lead.email,
                    'Status': status,
                    'Sent At': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                })
                logs.append(f"Email to {lead.email} - {status}")
        return results, "\n".join(logs)

    def stop_sending_emails(self):
        self.stop_sending_flag = True
        return "Email sending process has been stopped."

    def run(self):
        app = self.create_ui()
        app.launch(share=True)

def create_or_update_email_template(session, name, subject, body, template_id=None, is_ai_customizable=False, language='EN'):
    existing = session.query(EmailTemplate).filter_by(template_name=name).first()
    if existing and (not template_id or existing.id != int(template_id.split(':')[0])):
        raise ValueError("Template name already exists.")
    if template_id:
        template = session.query(EmailTemplate).get(int(template_id.split(':')[0]))
        if template:
            template.subject = subject
            template.body_content = body
            template.is_ai_customizable = is_ai_customizable
            template.language = language
    else:
        template = EmailTemplate(
            template_name=name,
            subject=subject,
            body_content=body,
            is_ai_customizable=is_ai_customizable,
            language=language,
            created_at=datetime.utcnow()
        )
        session.add(template)
    try:
        session.commit()
        return template.id
    except SQLAlchemyError as e:
        session.rollback()
        raise ValueError(f"Database error: {e}")

if __name__ == "__main__":
    app = GradioAutoclientApp()
    app.run()