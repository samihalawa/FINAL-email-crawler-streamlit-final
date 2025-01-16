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
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select, text, distinct, and_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from email_validator import validate_email, EmailNotValidError
from typing import List, Optional
from urllib.parse import urlparse, urlencode
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import signal
import subprocess
from fastapi import FastAPI, HTTPException, Depends, Request, Form, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any

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

engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal, Base = sessionmaker(bind=engine), declarative_base()

class Project(Base):
tablename = 'projects'
id = Column(BigInteger, primary_key=True)
project_name = Column(Text, default="Default Project")
created_at = Column(DateTime(timezone=True), server_default=func.now())
campaigns = relationship("Campaign", back_populates="project")
knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False)

class Campaign(Base):
tablename = 'campaigns'
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
tablename = 'campaign_leads'
id = Column(BigInteger, primary_key=True)
campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
lead_id = Column(BigInteger, ForeignKey('leads.id'))
status = Column(Text)
created_at = Column(DateTime(timezone=True), server_default=func.now())
lead = relationship("Lead", back_populates="campaign_leads")
campaign = relationship("Campaign", back_populates="campaign_leads")

class KnowledgeBase(Base):
tablename = 'knowledge_base'
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
content_copy download
 Use code with caution.
class Lead(Base):
tablename = 'leads'
id = Column(BigInteger, primary_key=True)
email = Column(Text, unique=True)
phone, first_name, last_name, company, job_title = [Column(Text) for _ in range(5)]
created_at = Column(DateTime(timezone=True), server_default=func.now())
campaign_leads = relationship("CampaignLead", back_populates="lead")
lead_sources = relationship("LeadSource", back_populates="lead")
email_campaigns = relationship("EmailCampaign", back_populates="lead")

class EmailTemplate(Base):
tablename = 'email_templates'
id = Column(BigInteger, primary_key=True)
campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
template_name, subject, body_content = Column(Text), Column(Text), Column(Text)
created_at = Column(DateTime(timezone=True), server_default=func.now())
is_ai_customizable = Column(Boolean, default=False)
language = Column(Text, default='ES')
campaign = relationship("Campaign")
email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
tablename = 'email_campaigns'
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
tablename = 'optimized_search_terms'
id = Column(BigInteger, primary_key=True)
original_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
term = Column(Text)
created_at = Column(DateTime(timezone=True), server_default=func.now())
original_term = relationship("SearchTerm", back_populates="optimized_terms")

class SearchTermEffectiveness(Base):
tablename = 'search_term_effectiveness'
id = Column(BigInteger, primary_key=True)
search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
total_results, valid_leads, irrelevant_leads, blogs_found, directories_found = [Column(BigInteger) for _ in range(5)]
created_at = Column(DateTime(timezone=True), server_default=func.now())
search_term = relationship("SearchTerm", back_populates="effectiveness")

class SearchTermGroup(Base):
tablename = 'search_term_groups'
id = Column(BigInteger, primary_key=True)
name, email_template, description = Column(Text), Column(Text), Column(Text)
created_at = Column(DateTime(timezone=True), server_default=func.now())
search_terms = relationship("SearchTerm", back_populates="group")

class SearchTerm(Base):
tablename = 'search_terms'
id = Column(BigInteger, primary_key=True)
group_id = Column(BigInteger, ForeignKey('search_term_groups.id'))
campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
term, category = Column(Text), Column(Text)
created_at = Column(DateTime(timezone=True), server_default=func.now())
language = Column(Text, default='ES')
group = relationship("SearchTermGroup", back_populates="search_terms")
campaign = relationship("Campaign", back_populates="search_terms")
optimized_terms = relationship("OptimizedSearchTerm", back_populates="original_term")
lead_sources = relationship("LeadSource", back_populates="search_term")
effectiveness = relationship("SearchTermEffectiveness", back_populates="search_term", uselist=False)

class LeadSource(Base):
tablename = 'lead_sources'
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
tablename = 'ai_request_logs'
id = Column(BigInteger, primary_key=True)
function_name, prompt, response, model_used = [Column(Text) for _ in range(4)]
created_at = Column(DateTime(timezone=True), server_default=func.now())
lead_id = Column(BigInteger, ForeignKey('leads.id'))
email_campaign_id = Column(BigInteger, ForeignKey('email_campaigns.id'))
lead = relationship("Lead")
email_campaign = relationship("EmailCampaign")

class AutomationLog(Base):
tablename = 'automation_logs'
id = Column(BigInteger, primary_key=True)
campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
leads_gathered, emails_sent = Column(BigInteger), Column(BigInteger)
start_time = Column(DateTime(timezone=True), server_default=func.now())
end_time = Column(DateTime(timezone=True))
status, logs = Column(Text), Column(JSON)
campaign = relationship("Campaign")
search_term = relationship("SearchTerm")

class Settings(Base):
tablename = 'settings'
id = Column(BigInteger, primary_key=True)
name = Column(Text, nullable=False)
setting_type = Column(Text, nullable=False)
value = Column(JSON, nullable=False)
created_at = Column(DateTime(timezone=True), server_default=func.now())
updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class EmailSettings(Base):
tablename = 'email_settings'
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

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
raise ValueError("DATABASE_URL not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

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

FastAPI setup
app = FastAPI()

origins = ["*"]

app.add_middleware(
CORSMiddleware,
allow_origins=origins,
allow_credentials=True,
allow_methods=[""],
allow_headers=[""],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class SettingsUpdate(BaseModel):
openai_api_key: str
openai_api_base: str
openai_model: str

class EmailSettingCreate(BaseModel):
name: str
email: str
provider: str
smtp_server: Optional[str] = None
smtp_port: Optional[int] = None
smtp_username: Optional[str] = None
smtp_password: Optional[str] = None
aws_access_key_id: Optional[str] = None
aws_secret_access_key: Optional[str] = None
aws_region: Optional[str] = None

class EmailSettingUpdate(BaseModel):
id: int
name: str
email: str
provider: str
smtp_server: Optional[str] = None
smtp_port: Optional[int] = None
smtp_username: Optional[str] = None
smtp_password: Optional[str] = None
aws_access_key_id: Optional[str] = None
aws_secret_access_key: Optional[str] = None
aws_region: Optional[str] = None

class SearchTermsInput(BaseModel):
terms: List[str]
num_results: int
optimize_english: bool
optimize_spanish: bool
shuffle_keywords: bool
language: str
enable_email_sending: bool
email_template: Optional[str] = None
email_setting_option: Optional[str] = None
reply_to: Optional[str] = None
ignore_previously_fetched: Optional[bool] = None

class EmailTemplateCreate(BaseModel):
template_name: str
subject: str
body_content: str
is_ai_customizable: Optional[bool] = False
language: Optional[str] = 'ES'

class EmailTemplateUpdate(BaseModel):
id: int
template_name: str
subject: str
body_content: str
is_ai_customizable: Optional[bool] = False
language: Optional[str] = 'ES'

class LeadUpdate(BaseModel):
id: int
email: str
first_name: Optional[str] = None
last_name: Optional[str] = None
company: Optional[str] = None
job_title: Optional[str] = None

class BulkSendInput(BaseModel):
template_id: int
from_email: str
reply_to: str
send_option: str
specific_email: Optional[str] = None
selected_terms: Optional[List[str]] = None
exclude_previously_contacted: Optional[bool] = None

class ProjectCreate(BaseModel):
project_name: str

class CampaignCreate(BaseModel):
campaign_name: str
project_id: int

class KnowledgeBaseCreate(BaseModel):
project_id: int
kb_name: Optional[str] = None
kb_bio: Optional[str] = None
kb_values: Optional[str] = None
contact_name: Optional[str] = None
contact_role: Optional[str] = None
contact_email: Optional[str] = None
company_description: Optional[str] = None
company_mission: Optional[str] = None
company_target_market: Optional[str] = None
company_other: Optional[str] = None
product_name: Optional[str] = None
product_description: Optional[str] = None
product_target_customer: Optional[str] = None
product_other: Optional[str] = None
other_context: Optional[str] = None
example_email: Optional[str] = None

class SearchTermGroupCreate(BaseModel):
name: str

class SearchTermGroupUpdate(BaseModel):
group_id: int
updated_terms: List[str]

class SearchTermCreate(BaseModel):
term: str
campaign_id: int
group_for_new_term: Optional[str] = None

class GroupedSearchTerm(BaseModel):
group_name: str
terms: List[str]

class SearchTermsGrouping(BaseModel):
grouped_terms: List[GroupedSearchTerm]
ungrouped_terms: List[str]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
return templates.TemplateResponse("index.html", {"request": request})

@app.get("/settings", response_model=Dict[str, Any])
async def get_settings(session: Session = Depends(db_session)):
general_settings = session.query(Settings).filter_by(setting_type='general').first()
email_settings = session.query(EmailSettings).all()

return {
    "general_settings": general_settings.value if general_settings else {},
    "email_settings": [{"id": setting.id, "name": setting.name, "email": setting.email, "provider": setting.provider, "smtp_server": setting.smtp_server, "smtp_port": setting.smtp_port, "aws_region": setting.aws_region} for setting in email_settings] if email_settings else [],
}
content_copy download
 Use code with caution.
@app.post("/settings", status_code=200)
async def update_settings(settings_update: SettingsUpdate, session: Session = Depends(db_session)):
general_settings = session.query(Settings).filter_by(setting_type='general').first() or Settings(name='General Settings', setting_type='general', value={})
general_settings.value = settings_update.dict()
session.add(general_settings)
return {"message": "General settings saved successfully!"}

@app.post("/email-settings", status_code=201)
async def create_email_setting(email_setting: EmailSettingCreate, session: Session = Depends(db_session)):
try:
new_setting = EmailSettings(**email_setting.dict())
session.add(new_setting)
session.commit()
return {"message": "Email setting saved successfully!"}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=str(e))

@app.put("/email-settings", status_code=200)
async def update_email_setting(email_setting_update: EmailSettingUpdate, session: Session = Depends(db_session)):
try:
setting = session.get(EmailSettings, email_setting_update.id)
if not setting:
raise HTTPException(status_code=404, detail="Email setting not found")
for key, value in email_setting_update.dict(exclude={'id'}).items():
setattr(setting, key, value)
session.commit()
return {"message": "Email setting updated successfully!"}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=str(e))

@app.delete("/email-settings/{setting_id}", status_code=200)
async def delete_email_setting(setting_id: int, session: Session = Depends(db_session)):
setting = session.get(EmailSettings, setting_id)
if setting:
session.delete(setting)
return {"message": f"Deleted {setting.name}"}
else:
raise HTTPException(status_code=404, detail="Email setting not found")

@app.post("/test-email", status_code=200)
async def test_email(test_email: str = Form(...), email_id: int = Form(...), session: Session = Depends(db_session)):
try:
setting = session.get(EmailSettings, email_id)
if not setting:
raise HTTPException(status_code=404, detail="Email setting not found")
response, _ = send_email_ses(
session,
setting.email,
test_email,
"Test Email from AutoclientAI",
"<p>This is a test email from your AutoclientAI email settings.</p>",
reply_to=setting.email
)
if response:
return {"message": f"Test email sent successfully to {test_email}"}
else:
raise HTTPException(status_code=500, detail="Failed to send test email")
except Exception as e:
raise HTTPException(status_code=500, detail=f"Error sending test email: {str(e)}")

@app.post("/manual-search", status_code=200)
async def perform_manual_search(search_input: SearchTermsInput, session: Session = Depends(db_session)):
if not search_input.terms:
raise HTTPException(status_code=400, detail="Enter at least one search term.")

if search_input.enable_email_sending:
    if not search_input.email_template:
        raise HTTPException(status_code=400, detail="No email templates available. Please create a template first.")
    if not search_input.email_setting_option:
      raise HTTPException(status_code=400, detail="No email settings available. Please add email settings first.")

    email_setting = session.query(EmailSettings).filter(EmailSettings.name == search_input.email_setting_option.split(" (")[0].strip()).first()
    if not email_setting:
        raise HTTPException(status_code=404, detail="Email setting not found.")
    from_email = email_setting.email

else:
    from_email = None

email_template = None
if search_input.email_template:
   template = session.query(EmailTemplate).filter(EmailTemplate.template_name == search_input.email_template.split(":")[1].strip()).first()
   if template:
     email_template = f"{template.id}: {template.template_name}"
   else:
    raise HTTPException(status_code=404, detail="Email template not found.")

results = manual_search(
    session, 
    search_input.terms, 
    search_input.num_results,
    search_input.ignore_previously_fetched if search_input.ignore_previously_fetched is not None else True,
    search_input.optimize_english, 
    search_input.optimize_spanish,
    search_input.shuffle_keywords,
    search_input.language, 
    search_input.enable_email_sending,
    None,
    from_email,
    search_input.reply_to,
    email_template
)
return {"total_leads": results['total_leads'], "results": results['results']}
content_copy download
 Use code with caution.
@app.get("/leads", response_model=List[Dict])
async def list_leads(session: Session = Depends(db_session)):
leads_df = fetch_leads_with_sources(session)
return leads_df.to_dict(orient='records') if not leads_df.empty else []

@app.put("/leads/{lead_id}", status_code=200)
async def update_lead_endpoint(lead_id: int, lead_data: LeadUpdate, session: Session = Depends(db_session)):
updated_data = {k: v for k, v in lead_data.dict().items() if k != 'id' and v is not None}
if update_lead(session, lead_id, updated_data):
return {"message": f"Updated lead: {lead_data.email}"}
else:
raise HTTPException(status_code=500, detail=f"Failed to update lead with id {lead_id}")

@app.delete("/leads/{lead_id}", status_code=200)
async def delete_lead_endpoint(lead_id: int, session: Session = Depends(db_session)):
if delete_lead_and_sources(session, lead_id):
return {"message": f"Deleted lead with id {lead_id}"}
else:
raise HTTPException(status_code=500, detail=f"Failed to delete lead with id {lead_id}")

@app.get("/search-terms", response_model=List[Dict])
async def get_search_terms(session: Session = Depends(db_session)):
search_terms_df = fetch_search_terms_with_lead_count(session)
return search_terms_df.to_dict(orient='records') if not search_terms_df.empty else []

@app.post("/search-terms", status_code=201)
async def add_new_search_term_endpoint(search_term: SearchTermCreate, session: Session = Depends(db_session)):
try:
add_new_search_term(session, search_term.term, search_term.campaign_id, search_term.group_for_new_term)
return {"message": f"Added: {search_term.term}"}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=f"Error adding search term: {str(e)}")

@app.get("/search-term-groups", response_model=List[Dict])
async def get_search_term_groups(session: Session = Depends(db_session)):
groups = session.query(SearchTermGroup).all()
return [{"id": group.id, "name": group.name} for group in groups]

@app.post("/search-term-groups", status_code=201)
async def create_search_term_group_endpoint(search_term_group: SearchTermGroupCreate, session: Session = Depends(db_session)):
try:
create_search_term_group(session, search_term_group.name)
return {"message": f"Created new group: {search_term_group.name}"}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=f"Error creating search term group: {str(e)}")

@app.delete("/search-term-groups/{group_id}", status_code=200)
async def delete_search_term_group_endpoint(group_id: int, session: Session = Depends(db_session)):
try:
delete_search_term_group(session, group_id)
return {"message": f"Deleted search term group with id {group_id}"}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=f"Error deleting search term group: {str(e)}")

@app.put("/search-term-groups", status_code=200)
async def update_search_term_group_endpoint(update_data: SearchTermGroupUpdate, session: Session = Depends(db_session)):
try:
update_search_term_group(session, update_data.group_id, update_data.updated_terms)
return {"message": "Group updated successfully"}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=f"Error updating search term group: {str(e)}")

@app.post("/search-terms/ai-group", response_model=Dict)
async def ai_grouping_search_terms(search_terms: SearchTermsGrouping, session: Session = Depends(db_session)):
try:
ungrouped_terms = session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
if search_terms.grouped_terms:
update_search_term_groups(session, {group.group_name: group.terms for group in search_terms.grouped_terms})
if search_terms.ungrouped_terms:
ungrouped_terms_from_request = [term.term for term in session.query(SearchTerm).filter(SearchTerm.term.in_(search_terms.ungrouped_terms))]
if ungrouped_terms_from_request:
grouped_terms = ai_group_search_terms(session, ungrouped_terms_from_request)
update_search_term_groups(session, grouped_terms)
return {"message": "Search terms have been grouped successfully!"}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=f"Error grouping search terms: {str(e)}")

@app.get("/email-templates", response_model=List[Dict])
async def get_email_templates(session: Session = Depends(db_session)):
templates = session.query(EmailTemplate).all()
return [{"id": t.id, "template_name": t.template_name, "subject": t.subject, "body_content": t.body_content, "is_ai_customizable": t.is_ai_customizable, "language": t.language} for t in templates]

@app.post("/email-templates", status_code=201)
async def create_email_template(template_data: EmailTemplateCreate, session: Session = Depends(db_session)):
try:
new_template = EmailTemplate(**template_data.dict(), campaign_id=get_active_campaign_id())
session.add(new_template)
session.commit()
return {"message": "Email template saved successfully!"}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=str(e))

@app.put("/email-templates", status_code=200)
async def update_email_template(template_data: EmailTemplateUpdate, session: Session = Depends(db_session)):
try:
template = session.get(EmailTemplate, template_data.id)
if not template:
raise HTTPException(status_code=404, detail="Email template not found")
for key, value in template_data.dict(exclude={'id'}).items():
setattr(template, key, value)
session.commit()
return {"message": "Email template updated successfully!"}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=str(e))

@app.delete("/email-templates/{template_id}", status_code=200)
async def delete_email_template(template_id: int, session: Session = Depends(db_session)):
try:
template = session.get(EmailTemplate, template_id)
if template:
session.delete(template)
session.commit()
return {"message": "Email template deleted successfully!"}
else:
raise HTTPException(status_code=404, detail="Email template not found")
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=str(e))

@app.post("/email-templates/ai-adjust", status_code=200)
async def adjust_email_template_with_ai(template_id: int = Form(...), ai_adjustment_prompt: str = Form(...), use_kb: bool = Form(...), session: Session = Depends(db_session)):
template = session.query(EmailTemplate).filter(EmailTemplate.id == template_id).first()
if not template:
raise HTTPException(status_code=404, detail="Email template not
Use code with caution.
Python
continue
raise HTTPException(status_code=404, detail="Email template not found")
kb_info = get_knowledge_base_info(session, get_active_project_id()) if use_kb else None
adjusted_template = generate_or_adjust_email_template(ai_adjustment_prompt, kb_info, current_template=template.body_content)
return {"subject": adjusted_template.get("subject", template.subject), "body": adjusted_template.get("body", template.body_content)}

@app.get("/email-preview", status_code=200)
async def email_preview(template_id: int, session: Session = Depends(db_session)):
template = session.query(EmailTemplate).filter_by(id=template_id).first()
if not template:
raise HTTPException(status_code=404, detail="Template not found")
wrapped_content = wrap_email_body(template.body_content)
return {"preview": wrapped_content}

@app.get("/email-logs", response_model=List[Dict])
async def get_email_logs(session: Session = Depends(db_session)):
logs_df = fetch_all_email_logs(session)
return logs_df.to_dict(orient='records') if not logs_df.empty else []

@app.post("/bulk-send", status_code=200)
async def bulk_send_emails_endpoint(bulk_send_input: BulkSendInput, session: Session = Depends(db_session)):
try:
template_id = bulk_send_input.template_id
from_email = bulk_send_input.from_email
reply_to = bulk_send_input.reply_to
send_option = bulk_send_input.send_option
specific_email = bulk_send_input.specific_email
selected_terms = bulk_send_input.selected_terms
exclude_previously_contacted = bulk_send_input.exclude_previously_contacted

if not from_email:
      raise HTTPException(status_code=400, detail="Email not found in settings. Please provide an email.")

    leads = fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted)
    if not leads:
        return {"message": "No leads found matching the selected criteria."}

    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    eligible_leads = [lead for lead in leads if lead.get('language', template.language) == template.language]

    logs, sent_count = bulk_send_emails(session, template_id, from_email, reply_to, eligible_leads)
    return {"message": f"Emails sent successfully to {sent_count} leads.", "logs": logs}
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
content_copy download
 Use code with caution.
@app.get("/projects", response_model=List[Dict])
async def list_projects(session: Session = Depends(db_session)):
projects = session.query(Project).all()
return [{"id": p.id, "project_name": p.project_name} for p in projects]

@app.post("/projects", status_code=201)
async def add_project_endpoint(project_data: ProjectCreate, session: Session = Depends(db_session)):
try:
new_project = Project(**project_data.dict(), created_at=datetime.utcnow())
session.add(new_project)
session.commit()
return {"message": f"Project '{project_data.project_name}' added successfully."}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=f"Error adding project: {str(e)}")

@app.get("/campaigns", response_model=List[Dict])
async def list_campaigns(session: Session = Depends(db_session)):
campaigns = session.query(Campaign).all()
return [{"id": c.id, "campaign_name": c.campaign_name, "project_id": c.project_id} for c in campaigns]

@app.post("/campaigns", status_code=201)
async def add_campaign_endpoint(campaign_data: CampaignCreate, session: Session = Depends(db_session)):
try:
new_campaign = Campaign(**campaign_data.dict(), created_at=datetime.utcnow())
session.add(new_campaign)
session.commit()
return {"message": f"Campaign '{campaign_data.campaign_name}' added to '{new_campaign.project_id}'."}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=f"Error adding campaign: {str(e)}")

@app.get("/knowledge-base", response_model=Optional[Dict])
async def get_knowledge_base(session: Session = Depends(db_session)):
project_id = get_active_project_id()
kb_info = get_knowledge_base_info(session, project_id)
return kb_info

@app.post("/knowledge-base", status_code=201)
async def create_or_update_knowledge_base(kb_data: KnowledgeBaseCreate, session: Session = Depends(db_session)):
try:
kb_entry = session.query(KnowledgeBase).filter_by(project_id=kb_data.project_id).first()
form_data = kb_data.dict()
if kb_entry:
for k, v in form_data.items():
setattr(kb_entry, k, v)
else:
new_kb = KnowledgeBase(**form_data, created_at=datetime.utcnow())
session.add(new_kb)
session.commit()
return {"message": "Knowledge Base saved successfully!", "kb_data": new_kb.to_dict() if not kb_entry else kb_entry.to_dict()}
except Exception as e:
session.rollback()
raise HTTPException(status_code=500, detail=f"Error saving knowledge base: {str(e)}")

@app.post("/start-automation", status_code=200)
async def start_automation(session: Session = Depends(db_session)):

try:
    base_terms = [term.term for term in session.query(SearchTerm).filter_by(campaign_id=get_active_campaign_id()).all()]
    if not base_terms:
        raise HTTPException(status_code=400, detail="No search terms found for the active campaign.")
    kb_info = get_knowledge_base_info(session, get_active_project_id())
    if not kb_info:
      raise HTTPException(status_code=400, detail="Knowledge base not found.")
      
    optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
    
    automation_log = AutomationLog(
        campaign_id=get_active_campaign_id(),
        status='running',
        start_time=datetime.utcnow(),
        logs=[],
        search_term_id=None,
        leads_gathered=0,
        emails_sent=0
    )
    session.add(automation_log)
    session.commit()

    # Start the worker process
    process = subprocess.Popen([
      'python', 'automated_search.py',
      str(automation_log.id)
    ])
    return {
         "message": "Automation started successfully",
         "automation_log_id": automation_log.id,
         "optimized_terms": optimized_terms
         }
except Exception as e:
    session.rollback()
    raise HTTPException(status_code=500, detail=f"Error starting automation: {str(e)}")
content_copy download
 Use code with caution.
@app.get("/automation-status/{automation_log_id}", status_code=200)
async def get_automation_status_endpoint(automation_log_id: int, session: Session = Depends(db_session)):
try:
status = get_automation_status(automation_log_id)
return status
except Exception as e:
raise HTTPException(status_code=500, detail=f"Error getting automation status: {str(e)}")

@app.get("/latest-logs/{automation_log_id}", response_model=List[Dict])
async def get_latest_logs_endpoint(automation_log_id: int, session: Session = Depends(db_session)):
try:
logs = get_latest_logs(automation_log_id)
return logs if logs else []
except Exception as e:
raise HTTPException(status_code=500, detail=f"Error getting automation logs: {str(e)}")

@app.post("/quick-scan", status_code=200)
async def perform_quick_scan_endpoint(session: Session = Depends(db_session)):
try:
res = perform_quick_scan(session)
return res
except Exception as e:
raise HTTPException(status_code=500, detail=f"Error performing quick scan: {str(e)}")

Utility Functions
def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
if not email_settings:
logging.error(f"No email settings found for {from_email}")
return None, None

tracking_id = str(uuid.uuid4())
tracking_pixel_url = f"https://autoclient-email-analytics.trigox.workers.dev/track?{urlencode({'id': tracking_id, 'type': 'open'})}"
wrapped_body = wrap_email_body(body)
tracked_body = wrapped_body.replace('</body>', f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none;"/></body>')

try:
    if email_settings.provider == 'ses':
        if ses_client is None:
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
                'Body': {'Html': {'Data': tracked_body, 'Charset': charset}}
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

        if not all([email_settings.smtp_server, email_settings.smtp_port, 
                   email_settings.smtp_username, email_settings.smtp_password]):
            raise ValueError("Incomplete SMTP settings")

        with smtplib.SMTP(email_settings.smtp_server, email_settings.smtp_port) as server:
            server.starttls()
            server.login(email_settings.smtp_username, email_settings.smtp_password)
            server.send_message(msg)
        return {'MessageId': f'smtp-{uuid.uuid4()}'}, tracking_id
    else:
        raise ValueError(f"Unknown email provider: {email_settings.provider}")
except Exception as e:
    logging.error(f"Error sending email: {str(e)}")
    raise
content_copy download
 Use code with caution.
def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
try:
lead = session.query(Lead).filter_by(email=lead_email).first()
if not lead:
logging.error(f"Lead with email {lead_email} not found.")
return

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
except Exception as e:
    logging.error(f"Error saving email campaign: {str(e)}")
    session.rollback()
content_copy download
 Use code with caution.
def update_log(log_container, message, level='info'):
icon = {'info': '🔵', 'success': '🟢', 'warning': '🟠', 'error': '🔴', 'email_sent': '🟣'}.get(level, '⚪')
log_entry = f"{icon} {message}"

# Simple console logging without HTML
print(f"{icon} {message.split('<')[0]}")
content_copy download
 Use code with caution.
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
pattern = r'^[\w.-]+@[\w.-]+.\w+$'
return re.match(pattern, email) is not None
def extract_emails_from_html(html_content):
pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+.[A-Z|a-z]{2,}\b'
return re.findall(pattern, html_content)

def extract_info_from_page(soup):
name_tag = soup.find('meta', {'name': 'author'})
name = name_tag['content'] if name_tag else ''

company_tag = soup.find('meta', {'property': 'og:site_name'})
company = company_tag['content'] if company_tag else ''

job_title_tag = soup.find('meta', {'name': 'job_title'})
job_title = job_title_tag['content'] if job_title_tag else ''

return name, company, job_title
content_copy download
 Use code with caution.
def manual_search(session, terms, num_results, ignore_previously_fetched=True, optimize_english=False, optimize_spanish=False, shuffle_keywords_option=False, language='ES', enable_email_sending=True, log_container=None, from_email=None, reply_to=None, email_template=None):
ua, results, total_leads, domains_processed = UserAgent(), [], 0, set()
for original_term in terms:
search_term_id = add_or_get_search_term(session, original_term, get_active_campaign_id())
search_term = shuffle_keywords(original_term) if shuffle_keywords_option else original_term
search_term = optimize_search_term(search_term, 'english' if optimize_english else 'spanish') if optimize_english or optimize_spanish else search_term
update_log(log_container, f"Searching for '{original_term}' (Used '{search_term}')")
for url in google_search(search_term, num_results, lang=language):
domain = get_domain_from_url(url)
if ignore_previously_fetched and domain in domains_processed:
update_log(log_container, f"Skipping Previously Fetched: {domain}", 'warning')
continue
update_log(log_container, f"Fetching: {url}")
try:
if not url.startswith(('http://', 'https://')):
url = 'http://' + url
response = requests.get(url, timeout=10, verify=False, headers={'User-Agent': ua.random})
response.raise_for_status()
html_content, soup = response.text, BeautifulSoup(response.text, 'html.parser')
emails = extract_emails_from_html(html_content)
update_log(log_container, f"Found {len(emails)} email(s) on {url}", 'success')
for email in filter(is_valid_email, emails):
if domain not in domains_processed:
name, company, job_title = extract_info_from_page(soup)
lead = save_lead(session, email=email, first_name=name, company=company, job_title=job_title, url=url, search_term_id=search_term_id, created_at=datetime.utcnow())
if lead:
total_leads += 1
results.append({
'Email': email, 'URL': url, 'Lead Source': original_term,
'Title': get_page_title(html_content), 'Description': get_page_description(html_content),
'Tags': [], 'Name': name, 'Company': company, 'Job Title': job_title,
'Search Term ID': search_term_id
})
update_log(log_container, f"Saved lead: {email}", 'success')
domains_processed.add(domain)
if enable_email_sending:
if not from_email or not email_template:
update_log(log_container, "Email sending is enabled but from_email or email_template is not provided", 'error')
return {"total_leads": total_leads, "results": results}

template = session.query(EmailTemplate).filter_by(id=int(email_template.split(":")[0])).first()
                            if not template:
                                update_log(log_container, "Email template not found", 'error')
                                return {"total_leads": total_leads, "results": results}

                            wrapped_content = wrap_email_body(template.body_content)
                            response, tracking_id = send_email_ses(session, from_email, email, template.subject, wrapped_content, reply_to=reply_to)
                            if response:
                                update_log(log_container, f"Sent email to: {email}", 'email_sent')
                                save_email_campaign(session, email, template.id, 'Sent', datetime.utcnow(), template.subject, response['MessageId'], wrapped_content)
                            else:
                                update_log(log_container, f"Failed to send email to: {email}", 'error')
                                save_email_campaign(session, email, template.id, 'Failed', datetime.utcnow(), template.subject, None, wrapped_content)
                            break
        except requests.RequestException as e:
            update_log(log_container, f"Error processing URL {url}: {str(e)}", 'error')
    
update_log(log_container, f"Total leads found: {total_leads}", 'info')
return {"total_leads": total_leads, "results": results}
content_copy download
 Use code with caution.
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
        return {
            "subject": "AI Generated Subject",
            "body": f"<p>{response}</p>"
        }
elif isinstance(response, dict):
    return response
else:
    return {"subject": "", "body": "<p>Failed to generate email content.</p>"}
content_copy download
 Use code with caution.
def fetch_leads_with_sources(session):
try:
query = session.query(Lead, func.string_agg(LeadSource.url, ', ').label('sources'), func.max(EmailCampaign.sent_at).label('last_contact'), func.string_agg(EmailCampaign.status, ', ').label('email_statuses')).outerjoin(LeadSource).outerjoin(EmailCampaign).group_by(Lead.id)
return pd.DataFrame([{**{k: getattr(lead, k) for k in ['id', 'email', 'first_name', 'last_name', 'company', 'job_title', 'created_at']}, 'Source': sources, 'Last Contact': last_contact, 'Last Email Status': email_statuses.split(', ')[-1] if email_statuses else 'Not Contacted'} for lead, sources, last_contact, email_statuses in query.all()])
except SQLAlchemyError as e:
logging.error(f"Database error in fetch_leads_with_sources: {str(e)}")
return pd.DataFrame()

def fetch_search_terms_with_lead_count(session):
query = session.query(SearchTerm.term, func.count(distinct(Lead.id)).label('lead_count'), func.count(distinct(EmailCampaign.id)).label('email_count')).join(LeadSource, SearchTerm.id == LeadSource.search_term_id).join(Lead, LeadSource.lead_id == Lead.id).outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id).group_by(SearchTerm.term)
return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

def add_search_term(session, term, campaign_id):
try:
new_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=datetime.utcnow())
session.add(new_term)
session.commit()
return new_term.id
except SQLAlchemyError as e:
session.rollback()
logging.error(f"Error adding search term: {str(e)}")
raise

def update_search_term_group(session, group_id, updated_terms):
try:
current_term_ids = set(int(term.split(":")[0]) for term in updated_terms)
existing_terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all()
for term in existing_terms:
term.group_id = None if term.id not in current_term_ids else group_id
for term_str in updated_terms:
term = session.query(SearchTerm).get(int(term_str.split(":")[0]))
if term: term.group_id = group_id
session.commit()
except Exception as e:
session.rollback()
logging.error(f"Error in update_search_term_group: {str(e)}")

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
try:
new_search_term = SearchTerm(term=new_term, campaign_id=campaign_id, created_at=datetime.utcnow(), group_id=int(group_for_new_term.split(":")[0]) if group_for_new_term != "None" else None)
session.add(new_search_term)
session.commit()
except Exception as e:
session.rollback()
logging.error(f"Error adding search term: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
existing_groups = session.query(SearchTermGroup).all()
prompt = f"Categorize these search terms into existing groups or suggest new ones:\n{', '.join([term.term for term in ungrouped_terms])}\n\nExisting groups: {', '.join([group.name for group in existing_groups])}\n\nRespond with a JSON object: {{group_name: [term1, term2, ...]}}"
messages = [{"role": "system", "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."}, {"role": "user", "content": prompt}]
response = openai_chat_completion(messages, function_name="ai_group_search_terms")
return response if isinstance(response, dict) else {}

def update_search_term_groups(session, grouped_terms):
for group_name, terms in grouped_terms.items():
group = session.query(SearchTermGroup).filter_by(name=group_name).first() or SearchTermGroup(name=group_name)
if not group.id: session.add(group); session.flush()
for term in terms:
search_term = session.query(SearchTerm).filter_by(term=term).first()
if search_term: search_term.group_id = group.id
session.commit()

def create_search_term_group(session, group_name):
try:
session.add(SearchTermGroup(name=group_name))
session.commit()
except Exception as e:
session.rollback()
logging.error(f"Error creating search term group: {str(e)}")

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

def ai_automation_loop(session, log_container, leads_container):
automation_logs, total_search_terms, total_emails_sent = [], 0, 0
while True:
try:
update_log(log_container, "Starting automation cycle")
kb_info = get_knowledge_base_info(session, get_active_project_id())
if not kb_info:
update_log(log_container, "Knowledge Base not found. Skipping cycle.", "warning")
time.sleep(3600)
continue
base_terms = [term.term for term in session.query(SearchTerm).filter_by(campaign_id=get_active_campaign_id()).all()]
optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
update_log(log_container, f"Optimized Search Terms: {', '.join(optimized_terms)}")

total_search_terms = len(optimized_terms)
        for idx, term in enumerate(optimized_terms):
            results = manual_search(session, [term], 10, ignore_previously_fetched=True)
            new_leads = []
            for res in results['results']:
                lead = save_lead(session, res['Email'], url=res['URL'])
                if lead:
                    new_leads.append((lead.id, lead.email))
            if new_leads:
                template = session.query(EmailTemplate).filter_by(campaign_id=get_active_campaign_id()).first()
                if template:
                    from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                    reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                    logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': email} for _, email in new_leads])
                    automation_logs.extend(logs)
                    total_emails_sent += sent_count
            update_log(log_container, f"New Leads Found: {', '.join([email for _, email in new_leads])}")
        update_log(log_container, f"Automation cycle completed. Total search terms: {total_search_terms}, Total emails sent: {total_emails_sent}", "success")
        time.sleep(3600)
    except Exception as e:
        update_log(log_container, f"Critical error in automation cycle: {str(e)}", "error")
        time.sleep(300)
content_copy download
 Use code with caution.
def openai_chat_completion(messages, temperature=0.7, function_name=None, lead_id=None, email_campaign_id=None):
with db_session() as session:
general_settings = session.query(Settings).filter_by(setting_type='general').first()
if not general_settings or 'openai_api_key' not in general_settings.value:
raise HTTPException(status_code=400, detail="OpenAI API key not set. Please configure it in the settings.")

client = OpenAI(api_key=general_settings.value['openai_api_key'])
    model = general_settings.value.get('openai_model', "gpt-4o-mini")

try:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    result = response.choices[0].message.content
    with db_session() as session:
        log_ai_request(session, function_name, messages, result, lead_id, email_campaign_id, model)
    
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return result
except Exception as e:
    with db_session() as session:
        log_ai_request(session, function_name, messages, str(e), lead_id, email_campaign_id, model)
    raise HTTPException(status_code=500, detail=f"Error in OpenAI API call: {str(e)}")
content_copy download
 Use code with caution.
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

def save_lead(session, email, first_name=None, last_name=None, company=None, job_title=None, phone=None, url=None, search_term_id=None, created_at=None):
try:
existing_lead = session.query(Lead).filter_by(email=email).first()
if existing_lead:
for attr in ['first_name', 'last_name', 'company', 'job_title', 'phone', 'created_at']:
if locals()[attr]:
setattr(existing_lead, attr, locals()[attr])
lead = existing_lead
else:
lead = Lead(
email=email,
first_name=first_name,
last_name=last_name,
company=company,
job_title=job_title,
phone=phone,
created_at=created_at or datetime.utcnow()
)
session.add(lead)

session.flush()

    lead_source = LeadSource(
        lead_id=lead.id, 
        url=url, 
        search_term_id=search_term_id
    )
    session.add(lead_source)

    campaign_lead = CampaignLead(
        campaign_id=get_active_campaign_id(), 
        lead_id=lead
content_copy download
 Use code with caution.
Use code with caution.
Python
continue
.id,
status="Not Contacted",
created_at=datetime.utcnow()
)
session.add(campaign_lead)

session.commit()
return lead

except Exception as e:
logging.error(f"Error saving lead: {str(e)}")
session.rollback()
return None
Use code with caution.
def save_lead_source(session, lead_id, search_term_id, url, http_status, scrape_duration, page_title=None, meta_description=None, content=None, tags=None, phone_numbers=None):
session.add(LeadSource(lead_id=lead_id, search_term_id=search_term_id, url=url, http_status=http_status, scrape_duration=scrape_duration, page_title=page_title or get_page_title(url), meta_description=meta_description or get_page_description(url), content=content or extract_visible_text(BeautifulSoup(requests.get(url).text, 'html.parser')), tags=tags, phone_numbers=phone_numbers))
session.commit()

def get_page_title(url):
try:
response = requests.get(url, timeout=10)
soup = BeautifulSoup(response.text, 'html.parser')
title = soup.title.string if soup.title else "No title found"
return title.strip()
except Exception as e:
logging.error(f"Error getting page title for {url}: {str(e)}")
return "Error fetching title"

def extract_visible_text(soup):
for script in soup(["script", "style"]):
script.extract()
text = soup.get_text()
lines = (line.strip() for line in text.splitlines())
chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
return ' '.join(chunk for chunk in chunks if chunk)

def log_search_term_effectiveness(session, term, total_results, valid_leads, blogs_found, directories_found):
session.add(SearchTermEffectiveness(term=term, total_results=total_results, valid_leads=valid_leads, irrelevant_leads=total_results - valid_leads, blogs_found=blogs_found, directories_found=directories_found))
session.commit()

get_active_project_id = lambda: 1
get_active_campaign_id = lambda: 1
set_active_project_id = lambda project_id: 1
set_active_campaign_id = lambda campaign_id: 1

def add_or_get_search_term(session, term, campaign_id, created_at=None):
search_term = session.query(SearchTerm).filter_by(term=term, campaign_id=campaign_id).first()
if not search_term:
search_term = SearchTerm(term=term, campaign_id=campaign_id, created_at=created_at or datetime.utcnow())
session.add(search_term)
session.commit()
session.refresh(search_term)
return search_term.id

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
content_copy download
 Use code with caution.
except Exception as e:
logging.error(f"Error fetching leads: {str(e)}")
return []
Use code with caution.
def get_domain_from_url(url): return urlparse(url).netloc

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

def is_valid_email(email):
try:
validate_email(email)
return True
except EmailNotValidError:
return False

def get_knowledge_base_info(session, project_id):
kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
return kb_info.to_dict() if kb_info else None

def shuffle_keywords(term):
words = term.split()
random.shuffle(words)
return ' '.join(words)

def get_page_description(html_content):
soup = BeautifulSoup(html_content, 'html.parser')
meta_desc = soup.find('meta', attrs={'name': 'description'})
return meta_desc['content'] if meta_desc else "No description found"

def is_valid_email(email):
if email is None: return False
invalid_patterns = [
r"..(png|jpg|jpeg|gif|css|js)
"
,
r
"
(
n
r
∣
b
o
o
t
s
t
r
a
p
∣
j
q
u
e
r
y
∣
c
o
r
e
∣
i
c
o
n
−
∣
n
o
r
e
p
l
y
)
@
.
∗
"
,
r
"
(
e
m
a
i
l
∣
i
n
f
o
∣
c
o
n
t
a
c
t
∣
s
u
p
p
o
r
t
∣
h
e
l
l
o
∣
h
o
l
a
∣
h
i
∣
s
a
l
u
t
a
t
i
o
n
s
∣
g
r
e
e
t
i
n
g
s
∣
i
n
q
u
i
r
i
e
s
∣
q
u
e
s
t
i
o
n
s
)
@
.
∗
"
,
r
"
e
m
a
i
l
@
e
m
a
i
l
c
˙
o
m
",r"
(
nr∣bootstrap∣jquery∣core∣icon−∣noreply)@.∗",r"
(
email∣info∣contact∣support∣hello∣hola∣hi∣salutations∣greetings∣inquiries∣questions)@.∗",r"
e
mail@email
c
˙
om
",
r".@example.com
"
,
r
"
.
∗
@
.
∗
(
˙
p
n
g
∣
j
p
g
∣
j
p
e
g
∣
g
i
f
∣
c
s
s
∣
j
s
∣
j
p
g
a
∣
P
M
∣
H
L
)
",r".∗@.∗
(
˙
​
png∣jpg∣jpeg∣gif∣css∣js∣jpga∣PM∣HL)
"
]
typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
if any(re.match(pattern, email, re.IGNORECASE) for pattern in invalid_patterns): return False
if any(email.lower().endswith(f"@{domain}") for domain in typo_domains): return False
try: validate_email(email); return True
except EmailNotValidError: return False

def remove_invalid_leads(session):
invalid_leads = session.query(Lead).filter(
Lead.email.op('')(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+.[a-zA-Z]{2,}
′
)
∣
L
e
a
d
.
e
m
a
i
l
.
o
p
(
′

′
)
(
r
′
.
∗
(
˙
p
n
g
∣
j
p
g
∣
j
p
e
g
∣
g
i
f
∣
c
s
s
∣
j
s
)
′
)∣Lead.email.op(
′

′
)(r
′
.∗
(
˙
​
png∣jpg∣jpeg∣gif∣css∣js)
') |
Lead.email.op('')(r'^(nr|bootstrap|jquery|core|icon-|noreply)@.*') |
Lead.email == 'email@email.com' |
Lead.email.like('%@example.com') |
Lead.email.op('')(r'.@..(png|jpg|jpeg|gif|css|js|jpga|PM|HL)$') |
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
Use code with caution.
def perform_quick_scan(session):
with db_session() as session:
terms = session.query(SearchTerm).order_by(func.random()).limit(3).all()
email_setting = fetch_email_settings(session)[0] if fetch_email_settings(session) else None
from_email = email_setting['email'] if email_setting else None
reply_to = from_email
email_template = session.query(EmailTemplate).first()
res = manual_search(session, [term.term for term in terms], 10, True, False, False, True, "EN", True, None, from_email, reply_to, f"{email_template.id}: {email_template.template_name}" if email_template else None)
return {"new_leads": len(res['results']), "terms_used": [term.term for term in terms]}

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
        pass
    if status_text:
        pass
    if results is not None:
        results.append({"Email": lead['Email'], "Status": status})

    if log_container:
        update_log(log_container, log_message)

except EmailNotValidError:
    log_message = f"❌ Invalid email address: {lead['Email']}"
    logs.append(log_message)
except Exception as e:
    error_message = f"Error sending email to {lead['Email']}: {str(e)}"
    logging.error(error_message)
    save_email_campaign(session, lead['Email'], template_id, 'failed', datetime.utcnow(), email_subject, f"error-{uuid.uuid4()}", email_content)
    logs.append(f"❌ Error sending email to: {lead['Email']} (Error: {str(e)})")
content_copy download
 Use code with caution.
return logs, sent_count
Use code with caution.
def wrap_email_body(body_content):
return f"""

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Email Template</title>
<style>
body {{
font-family: Arial, sans-serif;
line-height: 1.6;
color: #333;
max-width: 600px;
margin: 0 auto;
padding: 20px;
}}
</style>
</head>
<body>
{body_content}
</body>
</html>
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

def get_latest_logs(automation_log_id):
with db_session() as session:
log = session.query(AutomationLog).get(automation_log_id)
return log.logs if log else []

def safe_google_search(query, num_results=10, lang='es'):
try:
return list(google_search(query, num_results=num_results, lang=lang, stop=num_results))
except Exception as e:
logging.error(f"Google search error for '{query}': {str(e)}")
return []

def is_process_running(pid):
try:
os.kill(pid, 0)
return True
except (OSError, ProcessLookupError):
return False

def get_automation_status(automation_log_id):
try:
with db_session() as session:
log = session.query(AutomationLog).get(automation_log_id)
if not log:
return {'status': 'unknown', 'leads_gathered': 0, 'emails_sent': 0, 'latest_logs': []}
return {
'status': log.status,
'leads_gathered': log.leads_gathered or 0,
'emails_sent': log.emails_sent or 0,
'latest_logs': log.logs[-10:] if log.logs else []
}
except Exception as e:
logging.error(f"Error getting automation status: {e}")
return {'status': 'error', 'leads_gathered': 0, 'emails_sent': 0, 'latest_logs': []}

def update_automation_ui(automation_log_id, log_placeholder, metrics_placeholder):
try:
status = get_automation_status(automation_log_id)

with metrics_placeholder:
col1, col2 = st.columns(2)
col1.metric("Leads Found", status['leads_gathered'])
col2.metric("Emails Sent", status['emails_sent'])

with log_placeholder:
    if status['latest_logs']:
        log_text = "\n".join(
            f"{log.get('timestamp', '')}: {log.get('message', '')}" 
            for log in status['latest_logs']
            if isinstance(log, dict)
        )
        st.code(log_text)
    else:
        st.info("No logs available")
content_copy download
 Use code with caution.
except Exception as e:
st.error(f"Error updating UI: {e}")
Use code with caution.
def get_ai_response(prompt):
return openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100).choices[0].text.strip()

def fetch_email_settings(session):
try:
settings = session.query(EmailSettings).all()
return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
except Exception as e:
logging.error(f"Error fetching email settings: {e}")
return []

def generate_optimized_search_terms(session, base_terms, kb_info):
prompt = f"Optimize and expand these search terms for lead generation:\n{', '.join(base_terms)}\n\nConsider:\n1. Relevance to business and target market\n2. Potential for high-quality leads\n3. Variations and related terms\n4. Industry-specific jargon\n\nRespond with a JSON array of optimized terms."
response = openai_chat_completion([{"role": "system", "content": "You're an AI specializing in optimizing search terms for lead generation. Be concise and effective."}, {"role": "user", "content": prompt}], function_name="generate_optimized_search_terms")
return response.get('optimized_terms', base_terms) if isinstance(response, dict) else base_terms

if name == "main":
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
