from nicegui import ui, app
from sqlalchemy.orm import Session
from typing import Dict, List
import asyncio
from datetime import datetime
from formantic import ModelForm  # Auto form generation
from .models import Base, Project, Campaign, SearchTerm, Lead, LeadSource, SearchJob
from .worker import SearchWorker, AutomationWorker
from .utils import AIProcessor, GoogleSearcher

# Initialize workers and state
class AppState:
    def __init__(self):
        self.active_jobs: Dict[str, Dict] = {}
        self.automation_running: bool = False
        self.current_project_id: int = None
        self.current_campaign_id: int = None

state = AppState()

# Auto-generated forms for database models
class ProjectForm(ModelForm):
    class Meta:
        model = Project
        exclude = ['id', 'created_at']

class CampaignForm(ModelForm):
    class Meta:
        model = Campaign
        exclude = ['id', 'created_at']

class SearchTermForm(ModelForm):
    class Meta:
        model = SearchTerm
        exclude = ['id', 'created_at', 'last_used']

# Page layouts
@ui.page('/')
def home():
    with ui.tabs().classes('w-full') as tabs:
        ui.tab('Projects')
        ui.tab('Campaigns')
        ui.tab('Search')
        ui.tab('Results')
        ui.tab('Automation')
    
    with ui.tab_panels(tabs, value='Projects'):
        with ui.tab_panel('Projects'):
            projects_page()
        with ui.tab_panel('Campaigns'):
            campaigns_page()
        with ui.tab_panel('Search'):
            search_page()
        with ui.tab_panel('Results'):
            results_page()
        with ui.tab_panel('Automation'):
            automation_page()

def projects_page():
    """Auto-generated project management page"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Projects').classes('text-2xl mb-4')
        
        # Project form
        form = ProjectForm()
        form.render()  # Auto-generates form from model
        
        async def save_project():
            with Session() as session:
                project = form.save(session)
                session.commit()
                ui.notify(f'Project {project.project_name} saved!')
                await load_projects()
        
        ui.button('Save Project', on_click=save_project)
        
        # Projects table
        async def load_projects():
            with Session() as session:
                projects = session.query(Project).all()
                table.rows = [{
                    'id': p.id,
                    'name': p.project_name,
                    'campaigns': len(p.campaigns),
                    'created': p.created_at.strftime('%Y-%m-%d')
                } for p in projects]
        
        table = ui.table(
            columns=[
                {'name': 'id', 'label': 'ID', 'field': 'id'},
                {'name': 'name', 'label': 'Name', 'field': 'name'},
                {'name': 'campaigns', 'label': 'Campaigns', 'field': 'campaigns'},
                {'name': 'created', 'label': 'Created', 'field': 'created'}
            ],
            rows=[],
            row_key='id'
        ).classes('w-full')
        
        ui.button('Refresh', on_click=load_projects)

def campaigns_page():
    """Auto-generated campaign management page"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Campaigns').classes('text-2xl mb-4')
        
        form = CampaignForm()
        form.render()
        
        async def save_campaign():
            with Session() as session:
                campaign = form.save(session)
                session.commit()
                ui.notify(f'Campaign {campaign.campaign_name} saved!')
                await load_campaigns()
        
        ui.button('Save Campaign', on_click=save_campaign)
        
        async def load_campaigns():
            with Session() as session:
                campaigns = session.query(Campaign).all()
                table.rows = [{
                    'id': c.id,
                    'name': c.campaign_name,
                    'project': c.project.project_name,
                    'auto_send': c.auto_send,
                    'created': c.created_at.strftime('%Y-%m-%d')
                } for c in campaigns]
        
        table = ui.table(
            columns=[
                {'name': 'id', 'label': 'ID', 'field': 'id'},
                {'name': 'name', 'label': 'Name', 'field': 'name'},
                {'name': 'project', 'label': 'Project', 'field': 'project'},
                {'name': 'auto_send', 'label': 'Auto Send', 'field': 'auto_send'},
                {'name': 'created', 'label': 'Created', 'field': 'created'}
            ],
            rows=[],
            row_key='id'
        ).classes('w-full')
        
        ui.button('Refresh', on_click=load_campaigns)

def search_page():
    """Search interface"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Search').classes('text-2xl mb-4')
        
        terms_area = ui.textarea('Enter search terms (one per line)').classes('w-full')
        progress = ui.progress().classes('w-full')
        
        async def start_search():
            terms = terms_area.value.split('\n')
            with Session() as session:
                worker = SearchWorker(session)
                job = await worker.create_job(terms)
                
                # Update progress in UI
                while job.id in worker.active_jobs:
                    job_status = worker.active_jobs[str(job.id)]
                    progress.value = job_status['progress']
                    if job_status['status'] in ['completed', 'failed']:
                        break
                    await asyncio.sleep(1)
        
        ui.button('Start Search', on_click=start_search)

def results_page():
    """Results display"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Results').classes('text-2xl mb-4')
        
        async def load_results():
            with Session() as session:
                leads = session.query(Lead).all()
                table.rows = [{
                    'id': l.id,
                    'email': l.email,
                    'company': l.company,
                    'status': l.status,
                    'created': l.created_at.strftime('%Y-%m-%d')
                } for l in leads]
        
        table = ui.table(
            columns=[
                {'name': 'id', 'label': 'ID', 'field': 'id'},
                {'name': 'email', 'label': 'Email', 'field': 'email'},
                {'name': 'company', 'label': 'Company', 'field': 'company'},
                {'name': 'status', 'label': 'Status', 'field': 'status'},
                {'name': 'created', 'label': 'Created', 'field': 'created'}
            ],
            rows=[],
            row_key='id'
        ).classes('w-full')
        
        ui.button('Refresh', on_click=load_results)

def automation_page():
    """Automation controls"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Automation').classes('text-2xl mb-4')
        
        async def toggle_automation():
            state.automation_running = not state.automation_running
            if state.automation_running:
                with Session() as session:
                    worker = SearchWorker(session)
                    automation = AutomationWorker(session, worker)
                    asyncio.create_task(automation.start())
                auto_button.text = 'Stop Automation'
            else:
                auto_button.text = 'Start Automation'
        
        auto_button = ui.button(
            'Start Automation' if not state.automation_running else 'Stop Automation',
            on_click=toggle_automation
        )

# Start the app
ui.run(
    title='Email Crawler',
    favicon='🔍',
    dark=True,
    reload=False
)
