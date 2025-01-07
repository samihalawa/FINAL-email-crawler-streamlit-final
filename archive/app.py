import os
import logging
from nicegui import ui
from fastapi import FastAPI
from dotenv import load_dotenv
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from datetime import datetime
from models import *
from contextlib import contextmanager
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME") 
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    logger.error("Missing required database environment variables")
    sys.exit(1)

# Create FastAPI app
app = FastAPI(title="Email Crawler")

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)

# Create session factory
Session = scoped_session(sessionmaker(bind=engine))

@contextmanager
def get_session():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        session.close()

# State management
class AppState:
    def __init__(self):
        self.current_project = None
        self.current_campaign = None
        self.dark_mode = False

state = AppState()

# Database queries
def get_total_leads() -> int:
    with get_session() as session:
        return session.query(func.count(Lead.id)).scalar() or 0

def get_active_campaigns() -> int:
    with get_session() as session:
        return session.query(func.count(Campaign.id)).filter(Campaign.auto_send == True).scalar() or 0

def get_total_emails() -> int:
    with get_session() as session:
        return session.query(func.count(EmailCampaign.id)).filter(EmailCampaign.status == 'sent').scalar() or 0

def get_success_rate() -> str:
    with get_session() as session:
        total = session.query(func.count(EmailCampaign.id)).filter(EmailCampaign.status == 'sent').scalar() or 0
        opened = session.query(func.count(EmailCampaign.id)).filter(EmailCampaign.opened_at.isnot(None)).scalar() or 0
        return f"{(opened/total*100 if total else 0):.1f}%"

def get_campaigns() -> list:
    with get_session() as session:
        campaigns = session.query(Campaign).all()
        return [{
            'campaign_name': c.campaign_name,
            'campaign_type': c.campaign_type,
            'auto_send': 'Active' if c.auto_send else 'Inactive',
            'created_at': c.created_at.strftime('%Y-%m-%d %H:%M')
        } for c in campaigns]

# UI Pages
@ui.page('/')
def main_page():
    with ui.header().classes('bg-blue-600 text-white'):
        ui.label('Email Crawler').classes('text-h6')
        
    with ui.left_drawer().classes('bg-blue-100'):
        ui.button('Dashboard', on_click=lambda: ui.open('/'))
        ui.button('Campaigns', on_click=lambda: ui.open('/campaigns'))
        ui.button('Settings', on_click=lambda: ui.open('/settings'))

    with ui.column().classes('w-full p-4'):
        ui.label('Dashboard').classes('text-h4 mb-4')
        
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('w-1/4'):
                ui.label('Total Leads')
                ui.label().bind_text_from(get_total_leads)
            
            with ui.card().classes('w-1/4'):
                ui.label('Active Campaigns')
                ui.label().bind_text_from(get_active_campaigns)
            
            with ui.card().classes('w-1/4'):
                ui.label('Emails Sent')
                ui.label().bind_text_from(get_total_emails)
            
            with ui.card().classes('w-1/4'):
                ui.label('Success Rate')
                ui.label().bind_text_from(get_success_rate)

@ui.page('/campaigns')
def campaigns_page():
    with ui.column().classes('w-full p-4'):
        ui.label('Campaigns').classes('text-h4 mb-4')
        
        with ui.row().classes('w-full gap-4 mb-4'):
            ui.button('New Campaign', on_click=create_campaign).classes('bg-blue-600 text-white')
            
        with ui.table({
            'columns': [
                {'name': 'name', 'label': 'Campaign Name', 'field': 'campaign_name'},
                {'name': 'type', 'label': 'Type', 'field': 'campaign_type'},
                {'name': 'status', 'label': 'Status', 'field': 'auto_send'},
                {'name': 'created', 'label': 'Created At', 'field': 'created_at'},
            ],
            'rows': get_campaigns()
        }).classes('w-full'):
            pass

async def create_campaign():
    with ui.dialog() as dialog, ui.card():
        ui.label('New Campaign').classes('text-h6 mb-4')
        name = ui.input('Campaign Name')
        campaign_type = ui.select(['Email', 'LinkedIn'], value='Email', label='Type')
        auto_send = ui.switch('Auto Send')
        
        def save():
            if not name.value:
                ui.notify('Campaign name required', type='error')
                return
                
            with get_session() as session:
                campaign = Campaign(
                    campaign_name=name.value,
                    campaign_type=campaign_type.value,
                    auto_send=auto_send.value
                )
                session.add(campaign)
            
            dialog.close()
            ui.notify('Campaign created')
            
        ui.button('Save', on_click=save).classes('bg-blue-600 text-white')
        ui.button('Cancel', on_click=dialog.close)

# Mount NiceGUI to FastAPI
ui.run_with(app, storage_secret=os.getenv('STORAGE_SECRET', 'your-secret-key'))