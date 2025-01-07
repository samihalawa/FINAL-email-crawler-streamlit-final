import os
from nicegui import ui, app
from dotenv import load_dotenv
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import plotly.express as px
from datetime import datetime
import pandas as pd
from models import *  # We'll keep your existing models
from typing import Optional
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import openai
from email_validator import validate_email, EmailNotValidError

# Load environment variables
load_dotenv()

# Database setup
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# State management
class AppState:
    def __init__(self):
        self.current_project: Optional[int] = None
        self.current_campaign: Optional[int] = None
        self.dark_mode = False

state = AppState()

# UI Components
@ui.page('/')
def main_page():
    with ui.header().classes('bg-blue-600 text-white'):
        ui.label('Email Crawler Dashboard').classes('text-h6')
        
    with ui.left_drawer().classes('bg-blue-100'):
        with ui.column():
            ui.button('Dashboard', on_click=lambda: ui.open('/'))
            ui.button('Projects', on_click=lambda: ui.open('/projects'))
            ui.button('Campaigns', on_click=lambda: ui.open('/campaigns'))
            ui.button('Knowledge Base', on_click=lambda: ui.open('/knowledge'))
            ui.button('Settings', on_click=lambda: ui.open('/settings'))

    with ui.column().classes('w-full p-4'):
        ui.label('Welcome to Email Crawler').classes('text-h4 mb-4')
        
        # Stats cards
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('w-1/4'):
                ui.label('Total Leads').classes('text-h6')
                ui.label('0').bind_text_from(get_total_leads)
            
            with ui.card().classes('w-1/4'):
                ui.label('Active Campaigns').classes('text-h6')
                ui.label('0').bind_text_from(get_active_campaigns)
            
            with ui.card().classes('w-1/4'):
                ui.label('Emails Sent').classes('text-h6')
                ui.label('0').bind_text_from(get_total_emails)
            
            with ui.card().classes('w-1/4'):
                ui.label('Success Rate').classes('text-h6')
                ui.label('0%').bind_text_from(get_success_rate)

# Database operations
def get_total_leads():
    with SessionLocal() as session:
        return session.query(func.count(Lead.id)).scalar()

def get_active_campaigns():
    with SessionLocal() as session:
        return session.query(func.count(Campaign.id)).filter(Campaign.auto_send == True).scalar()

def get_total_emails():
    with SessionLocal() as session:
        return session.query(func.count(EmailCampaign.id)).filter(EmailCampaign.status == 'sent').scalar()

def get_success_rate():
    with SessionLocal() as session:
        total = session.query(func.count(EmailCampaign.id)).filter(EmailCampaign.status == 'sent').scalar()
        opened = session.query(func.count(EmailCampaign.id)).filter(EmailCampaign.opened_at.isnot(None)).scalar()
        return f"{(opened/total*100 if total else 0):.1f}%"

# Campaign management
@ui.page('/campaigns')
def campaigns_page():
    with ui.column().classes('w-full p-4'):
        ui.label('Campaign Management').classes('text-h4 mb-4')
        
        with ui.row().classes('w-full gap-4 mb-4'):
            ui.button('New Campaign', on_click=create_campaign).classes('bg-blue-600 text-white')
            
        with ui.table({'columns': [
            {'name': 'name', 'label': 'Campaign Name', 'field': 'campaign_name'},
            {'name': 'type', 'label': 'Type', 'field': 'campaign_type'},
            {'name': 'status', 'label': 'Status', 'field': 'auto_send'},
            {'name': 'created', 'label': 'Created At', 'field': 'created_at'},
        ]}).classes('w-full') as table:
            table.add_rows(get_campaigns())

def get_campaigns():
    with SessionLocal() as session:
        campaigns = session.query(Campaign).all()
        return [{
            'campaign_name': c.campaign_name,
            'campaign_type': c.campaign_type,
            'auto_send': 'Active' if c.auto_send else 'Inactive',
            'created_at': c.created_at.strftime('%Y-%m-%d %H:%M')
        } for c in campaigns]

async def create_campaign():
    with ui.dialog() as dialog, ui.card():
        ui.label('Create New Campaign').classes('text-h6 mb-4')
        name = ui.input('Campaign Name')
        campaign_type = ui.select(['Email', 'LinkedIn'], value='Email', label='Campaign Type')
        auto_send = ui.switch('Auto Send')
        
        def save():
            with SessionLocal() as session:
                campaign = Campaign(
                    campaign_name=name.value,
                    campaign_type=campaign_type.value,
                    auto_send=auto_send.value
                )
                session.add(campaign)
                session.commit()
                dialog.close()
                ui.notify('Campaign created successfully')
                
        ui.button('Save', on_click=save).classes('bg-blue-600 text-white')
        ui.button('Cancel', on_click=dialog.close)

# Start the application
ui.run(title='Email Crawler', favicon='🚀', dark=state.dark_mode) 