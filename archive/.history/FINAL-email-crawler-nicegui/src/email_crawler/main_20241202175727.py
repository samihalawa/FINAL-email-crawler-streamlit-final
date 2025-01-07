from nicegui import ui, app
from sqlalchemy.orm import Session, joinedload
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import pandas as pd
import logging
import json
from formantic import ModelForm
from .models import (
    Base, Project, Campaign, SearchTerm, Lead, LeadSource, SearchJob,
    EmailTemplate, EmailSettings, KnowledgeBase, EmailCampaign,
    SearchTermGroup, AIRequestLog, AutomationLog
)
from .worker import SearchWorker, AutomationWorker
from .utils import AIProcessor, GoogleSearcher, EmailSender

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app state
class AppState:
    def __init__(self):
        self.active_jobs: Dict[str, Dict] = {}
        self.automation_running: bool = False
        self.current_project_id: Optional[int] = None
        self.current_campaign_id: Optional[int] = None
        self.search_terms: List[str] = []
        self.optimized_terms: List[str] = []
        self.automation_logs: List[str] = []
        self.total_leads_found: int = 0
        self.total_emails_sent: int = 0

state = AppState()

# Auto-generated forms
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

class EmailTemplateForm(ModelForm):
    class Meta:
        model = EmailTemplate
        exclude = ['id', 'created_at']

class KnowledgeBaseForm(ModelForm):
    class Meta:
        model = KnowledgeBase
        exclude = ['id', 'created_at', 'updated_at']

# Main navigation
@ui.page('/')
def home():
    with ui.header().classes('w-full'):
        ui.label('AutoclientAI').classes('text-2xl font-bold')
        with ui.row():
            ui.button('Projects', on_click=lambda: tabs.set_value('Projects'))
            ui.button('Campaigns', on_click=lambda: tabs.set_value('Campaigns'))
            ui.button('Search', on_click=lambda: tabs.set_value('Search'))
            ui.button('Results', on_click=lambda: tabs.set_value('Results'))
            ui.button('Email Templates', on_click=lambda: tabs.set_value('Templates'))
            ui.button('Knowledge Base', on_click=lambda: tabs.set_value('Knowledge'))
            ui.button('Settings', on_click=lambda: tabs.set_value('Settings'))
    
    with ui.tabs().classes('w-full') as tabs:
        ui.tab('Projects')
        ui.tab('Campaigns')
        ui.tab('Search')
        ui.tab('Results')
        ui.tab('Templates')
        ui.tab('Knowledge')
        ui.tab('Settings')
    
    with ui.tab_panels(tabs, value='Projects'):
        with ui.tab_panel('Projects'):
            projects_page()
        with ui.tab_panel('Campaigns'):
            campaigns_page()
        with ui.tab_panel('Search'):
            search_page()
        with ui.tab_panel('Results'):
            results_page()
        with ui.tab_panel('Templates'):
            email_templates_page()
        with ui.tab_panel('Knowledge'):
            knowledge_base_page()
        with ui.tab_panel('Settings'):
            settings_page()

def projects_page():
    """Project management page"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Projects').classes('text-2xl mb-4')
        
        # Project form
        with ui.card().classes('w-full'):
            form = ProjectForm()
            form.render()
            
            async def save_project():
                with Session() as session:
                    project = form.save(session)
                    session.commit()
                    ui.notify(f'Project {project.project_name} saved!')
                    await load_projects()
            
            ui.button('Save Project', on_click=save_project).classes('mt-4')
        
        # Projects table
        async def load_projects():
            with Session() as session:
                projects = session.query(Project).all()
                table.rows = [{
                    'id': p.id,
                    'name': p.project_name,
                    'campaigns': len(p.campaigns),
                    'created': p.created_at.strftime('%Y-%m-%d'),
                    'actions': ui.button('Select', on_click=lambda p=p: set_active_project(p.id))
                } for p in projects]
        
        table = ui.table(
            columns=[
                {'name': 'id', 'label': 'ID', 'field': 'id'},
                {'name': 'name', 'label': 'Name', 'field': 'name'},
                {'name': 'campaigns', 'label': 'Campaigns', 'field': 'campaigns'},
                {'name': 'created', 'label': 'Created', 'field': 'created'},
                {'name': 'actions', 'label': 'Actions', 'field': 'actions'}
            ],
            rows=[],
            row_key='id'
        ).classes('w-full mt-4')
        
        ui.button('Refresh', on_click=load_projects).classes('mt-4')

def campaigns_page():
    """Campaign management page"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Campaigns').classes('text-2xl mb-4')
        
        if not state.current_project_id:
            ui.label('Please select a project first').classes('text-red-500')
            return
        
        with ui.card().classes('w-full'):
            form = CampaignForm()
            form.render()
            
            async def save_campaign():
                with Session() as session:
                    campaign = form.save(session)
                    campaign.project_id = state.current_project_id
                    session.commit()
                    ui.notify(f'Campaign {campaign.campaign_name} saved!')
                    await load_campaigns()
            
            ui.button('Save Campaign', on_click=save_campaign).classes('mt-4')
        
        async def load_campaigns():
            with Session() as session:
                campaigns = session.query(Campaign).filter_by(project_id=state.current_project_id).all()
                table.rows = [{
                    'id': c.id,
                    'name': c.campaign_name,
                    'auto_send': c.auto_send,
                    'created': c.created_at.strftime('%Y-%m-%d'),
                    'actions': ui.button('Select', on_click=lambda c=c: set_active_campaign(c.id))
                } for c in campaigns]
        
        table = ui.table(
            columns=[
                {'name': 'id', 'label': 'ID', 'field': 'id'},
                {'name': 'name', 'label': 'Name', 'field': 'name'},
                {'name': 'auto_send', 'label': 'Auto Send', 'field': 'auto_send'},
                {'name': 'created', 'label': 'Created', 'field': 'created'},
                {'name': 'actions', 'label': 'Actions', 'field': 'actions'}
            ],
            rows=[],
            row_key='id'
        ).classes('w-full mt-4')
        
        ui.button('Refresh', on_click=load_campaigns).classes('mt-4')

def search_page():
    """Search interface"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Search').classes('text-2xl mb-4')
        
        if not state.current_campaign_id:
            ui.label('Please select a campaign first').classes('text-red-500')
            return
        
        with ui.card().classes('w-full'):
            terms_area = ui.textarea('Enter search terms (one per line)').classes('w-full')
            
            with ui.row():
                num_results = ui.number('Results per term', value=10, min=1, max=100)
                language = ui.select('Language', options=['ES', 'EN'], value='ES')
            
            with ui.row():
                ui.checkbox('Enable email sending', value=True).bind_value(state, 'enable_email_sending')
                ui.checkbox('Ignore fetched domains', value=True).bind_value(state, 'ignore_previously_fetched')
                ui.checkbox('Shuffle Keywords', value=True).bind_value(state, 'shuffle_keywords_option')
            
            progress = ui.progress().classes('w-full')
            log_area = ui.textarea('Logs', readonly=True).classes('w-full h-40')
            
            async def start_search():
                terms = terms_area.value.split('\n')
                with Session() as session:
                    worker = SearchWorker(session)
                    job = await worker.create_job(
                        terms=terms,
                        campaign_id=state.current_campaign_id,
                        settings={
                            'num_results': num_results.value,
                            'language': language.value,
                            'enable_email_sending': state.enable_email_sending,
                            'ignore_previously_fetched': state.ignore_previously_fetched,
                            'shuffle_keywords_option': state.shuffle_keywords_option
                        }
                    )
                    
                    while job.id in worker.active_jobs:
                        status = worker.active_jobs[str(job.id)]
                        progress.value = status['progress']
                        log_area.value += f"\n{status['message']}"
                        if status['status'] in ['completed', 'failed']:
                            break
                        await asyncio.sleep(1)
            
            ui.button('Start Search', on_click=start_search).classes('mt-4')

def results_page():
    """Results display"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Results').classes('text-2xl mb-4')
        
        if not state.current_campaign_id:
            ui.label('Please select a campaign first').classes('text-red-500')
            return
        
        async def load_results():
            with Session() as session:
                leads = (session.query(Lead)
                        .join(LeadSource)
                        .join(SearchTerm)
                        .filter(SearchTerm.campaign_id == state.current_campaign_id)
                        .all())
                
                table.rows = [{
                    'id': l.id,
                    'email': l.email,
                    'company': l.company,
                    'status': l.status,
                    'created': l.created_at.strftime('%Y-%m-%d'),
                    'actions': ui.button('View', on_click=lambda l=l: view_lead_details(l.id))
                } for l in leads]
        
        table = ui.table(
            columns=[
                {'name': 'id', 'label': 'ID', 'field': 'id'},
                {'name': 'email', 'label': 'Email', 'field': 'email'},
                {'name': 'company', 'label': 'Company', 'field': 'company'},
                {'name': 'status', 'label': 'Status', 'field': 'status'},
                {'name': 'created', 'label': 'Created', 'field': 'created'},
                {'name': 'actions', 'label': 'Actions', 'field': 'actions'}
            ],
            rows=[],
            row_key='id'
        ).classes('w-full')
        
        ui.button('Refresh', on_click=load_results).classes('mt-4')
        
        # Export functionality
        async def export_results():
            with Session() as session:
                leads = (session.query(Lead)
                        .join(LeadSource)
                        .join(SearchTerm)
                        .filter(SearchTerm.campaign_id == state.current_campaign_id)
                        .all())
                
                df = pd.DataFrame([{
                    'ID': l.id,
                    'Email': l.email,
                    'Company': l.company,
                    'Status': l.status,
                    'Created': l.created_at.strftime('%Y-%m-%d')
                } for l in leads])
                
                df.to_csv('leads_export.csv', index=False)
                ui.download('leads_export.csv')
        
        ui.button('Export to CSV', on_click=export_results).classes('mt-4')

def email_templates_page():
    """Email template management"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Email Templates').classes('text-2xl mb-4')
        
        if not state.current_campaign_id:
            ui.label('Please select a campaign first').classes('text-red-500')
            return
        
        with ui.card().classes('w-full'):
            form = EmailTemplateForm()
            form.render()
            
            async def save_template():
                with Session() as session:
                    template = form.save(session)
                    template.campaign_id = state.current_campaign_id
                    session.commit()
                    ui.notify('Template saved successfully!')
                    await load_templates()
            
            ui.button('Save Template', on_click=save_template).classes('mt-4')
        
        async def load_templates():
            with Session() as session:
                templates = session.query(EmailTemplate).filter_by(campaign_id=state.current_campaign_id).all()
                templates_container.clear()
                
                for template in templates:
                    with templates_container:
                        with ui.card().classes('w-full mb-4'):
                            ui.label(template.template_name).classes('text-xl font-bold')
                            ui.label(f'Subject: {template.subject}')
                            ui.markdown(template.body_content)
                            
                            with ui.row():
                                ui.button('Edit', on_click=lambda t=template: edit_template(t.id))
                                ui.button('Delete', on_click=lambda t=template: delete_template(t.id))
        
        templates_container = ui.column().classes('w-full mt-4')
        ui.button('Refresh', on_click=load_templates).classes('mt-4')

def knowledge_base_page():
    """Knowledge base management"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Knowledge Base').classes('text-2xl mb-4')
        
        if not state.current_project_id:
            ui.label('Please select a project first').classes('text-red-500')
            return
        
        with ui.card().classes('w-full'):
            form = KnowledgeBaseForm()
            form.render()
            
            async def save_kb():
                with Session() as session:
                    kb = form.save(session)
                    kb.project_id = state.current_project_id
                    session.commit()
                    ui.notify('Knowledge Base saved successfully!')
            
            ui.button('Save Knowledge Base', on_click=save_kb).classes('mt-4')

def settings_page():
    """Settings management"""
    with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
        ui.label('Settings').classes('text-2xl mb-4')
        
        with ui.card().classes('w-full'):
            ui.label('Email Settings').classes('text-xl mb-2')
            
            email_provider = ui.select('Email Provider', options=['smtp', 'ses'])
            email_name = ui.input('Name')
            email_address = ui.input('Email Address')
            
            with ui.column().bind_visibility_from(email_provider, lambda v: v == 'smtp'):
                smtp_server = ui.input('SMTP Server')
                smtp_port = ui.number('SMTP Port', value=587)
                smtp_username = ui.input('SMTP Username')
                smtp_password = ui.input('SMTP Password', password=True)
            
            with ui.column().bind_visibility_from(email_provider, lambda v: v == 'ses'):
                aws_access_key = ui.input('AWS Access Key')
                aws_secret_key = ui.input('AWS Secret Key', password=True)
                aws_region = ui.input('AWS Region')
            
            async def save_email_settings():
                with Session() as session:
                    settings = EmailSettings(
                        name=email_name.value,
                        email=email_address.value,
                        provider=email_provider.value,
                        smtp_server=smtp_server.value if email_provider.value == 'smtp' else None,
                        smtp_port=smtp_port.value if email_provider.value == 'smtp' else None,
                        smtp_username=smtp_username.value if email_provider.value == 'smtp' else None,
                        smtp_password=smtp_password.value if email_provider.value == 'smtp' else None,
                        aws_access_key_id=aws_access_key.value if email_provider.value == 'ses' else None,
                        aws_secret_access_key=aws_secret_key.value if email_provider.value == 'ses' else None,
                        aws_region=aws_region.value if email_provider.value == 'ses' else None
                    )
                    session.add(settings)
                    session.commit()
                    ui.notify('Email settings saved successfully!')
            
            ui.button('Save Settings', on_click=save_email_settings).classes('mt-4')

# Helper functions
def set_active_project(project_id: int):
    state.current_project_id = project_id
    ui.notify(f'Active project set to ID: {project_id}')

def set_active_campaign(campaign_id: int):
    state.current_campaign_id = campaign_id
    ui.notify(f'Active campaign set to ID: {campaign_id}')

async def view_lead_details(lead_id: int):
    with Session() as session:
        lead = session.query(Lead).options(
            joinedload(Lead.lead_sources),
            joinedload(Lead.email_campaigns)
        ).filter_by(id=lead_id).first()
        
        if not lead:
            ui.notify('Lead not found', type='error')
            return
        
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Lead Details: {lead.email}').classes('text-xl font-bold')
            
            with ui.grid(columns=2).classes('gap-4'):
                ui.label('Company:')
                ui.label(lead.company or 'N/A')
                
                ui.label('Status:')
                ui.label(lead.status or 'N/A')
                
                ui.label('Created:')
                ui.label(lead.created_at.strftime('%Y-%m-%d %H:%M:%S'))
            
            ui.label('Sources:').classes('font-bold mt-4')
            for source in lead.lead_sources:
                ui.link(source.url, target='_blank')
            
            ui.label('Email Campaigns:').classes('font-bold mt-4')
            for campaign in lead.email_campaigns:
                ui.label(f'Sent: {campaign.sent_at.strftime("%Y-%m-%d %H:%M:%S") if campaign.sent_at else "Not sent"}')
                ui.label(f'Status: {campaign.status}')
            
            ui.button('Close', on_click=dialog.close).classes('mt-4')
        
        dialog.open()

async def edit_template(template_id: int):
    with Session() as session:
        template = session.query(EmailTemplate).get(template_id)
        
        if not template:
            ui.notify('Template not found', type='error')
            return
        
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Edit Template: {template.template_name}').classes('text-xl font-bold')
            
            name = ui.input('Template Name', value=template.template_name)
            subject = ui.input('Subject', value=template.subject)
            body = ui.textarea('Body', value=template.body_content).classes('w-full h-40')
            
            async def save_changes():
                template.template_name = name.value
                template.subject = subject.value
                template.body_content = body.value
                session.commit()
                ui.notify('Template updated successfully!')
                dialog.close()
            
            ui.button('Save Changes', on_click=save_changes).classes('mt-4')
            ui.button('Cancel', on_click=dialog.close).classes('mt-4')
        
        dialog.open()

async def delete_template(template_id: int):
    with Session() as session:
        template = session.query(EmailTemplate).get(template_id)
        
        if not template:
            ui.notify('Template not found', type='error')
            return
        
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Delete Template: {template.template_name}?').classes('text-xl font-bold')
            ui.label('This action cannot be undone.')
            
            async def confirm_delete():
                session.delete(template)
                session.commit()
                ui.notify('Template deleted successfully!')
                dialog.close()
            
            with ui.row():
                ui.button('Delete', on_click=confirm_delete).classes('bg-red-500')
                ui.button('Cancel', on_click=dialog.close)
        
        dialog.open()

# Start the app
ui.run(
    title='AutoclientAI',
    favicon='🔍',
    dark=True,
    reload=False,
    port=8080
)
