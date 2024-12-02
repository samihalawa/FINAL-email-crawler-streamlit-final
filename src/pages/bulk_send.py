from nicegui import ui
from typing import List, Optional
from datetime import datetime
import asyncio

from core.database import get_db, EmailTemplate, Lead, EmailCampaign
from core.auth import auth_handler
from services.email import EmailService
from services.ai import AIService

email_service = EmailService()
ai_service = AIService()

@ui.page('/bulk-send')
@auth_handler.require_auth
async def page():
    # State management
    selected_leads: List[Lead] = []
    selected_template: Optional[EmailTemplate] = None
    progress = None
    
    async def load_templates():
        async with get_db() as session:
            templates = await session.execute(
                "SELECT id, name, subject FROM email_templates ORDER BY created_at DESC"
            )
            return [{"id": t.id, "name": t.name, "subject": t.subject} for t in templates]
    
    async def load_leads():
        async with get_db() as session:
            leads = await session.execute(
                """
                SELECT l.id, l.email, l.name, l.company, l.last_contacted 
                FROM leads l 
                LEFT JOIN email_campaigns ec ON l.id = ec.lead_id 
                WHERE ec.id IS NULL OR ec.sent_at < NOW() - INTERVAL '7 days'
                ORDER BY l.created_at DESC
                """
            )
            return [dict(l) for l in leads]
    
    async def preview_template(template_id: int):
        async with get_db() as session:
            template = await session.get(EmailTemplate, template_id)
            if template:
                with preview_dialog:
                    preview_dialog.clear()
                    ui.label('Template Preview').classes('text-h6')
                    ui.label(f'Subject: {template.subject}').classes('text-bold')
                    ui.html(template.body_content).classes('q-pa-md')
                preview_dialog.open()
    
    async def optimize_template(template_id: int, lead_id: int):
        async with get_db() as session:
            template = await session.get(EmailTemplate, template_id)
            lead = await session.get(Lead, lead_id)
            kb = await session.get_knowledge_base()  # Implement this helper
            
            if template and lead and kb:
                optimized = await ai_service.optimize_email_template(template, lead, kb)
                
                with optimization_dialog:
                    optimization_dialog.clear()
                    ui.label('AI-Optimized Template').classes('text-h6')
                    ui.label(f'Original Subject: {template.subject}').classes('text-bold')
                    ui.label(f'Optimized Subject: {optimized.subject}').classes('text-bold text-primary')
                    
                    with ui.row().classes('w-full'):
                        with ui.column().classes('w-1/2'):
                            ui.label('Original Content').classes('text-bold')
                            ui.html(template.body_content).classes('q-pa-md')
                        with ui.column().classes('w-1/2'):
                            ui.label('Optimized Content').classes('text-bold')
                            ui.html(optimized.body).classes('q-pa-md')
                    
                    ui.label('Personalization Strategy').classes('text-bold')
                    for key, value in optimized.personalization_strategy.items():
                        ui.label(f'{key}: {value}').classes('text-body2')
                    
                    with ui.row().classes('w-full justify-end'):
                        ui.button('Apply Changes', on_click=lambda: apply_optimization(
                            template_id, optimized
                        )).props('primary')
                        ui.button('Cancel', on_click=optimization_dialog.close).props('flat')
                
                optimization_dialog.open()
    
    async def apply_optimization(template_id: int, optimized_content: dict):
        async with get_db() as session:
            template = await session.get(EmailTemplate, template_id)
            if template:
                template.subject = optimized_content.subject
                template.body_content = optimized_content.body
                await session.commit()
                optimization_dialog.close()
                ui.notify('Template updated successfully')
    
    async def start_bulk_send():
        if not selected_leads or not selected_template:
            ui.notify('Please select leads and template', type='warning')
            return
        
        nonlocal progress
        progress = ui.progress()
        
        async def update_progress(current_progress: float, results: dict):
            progress.set_value(current_progress)
            with stats_container:
                stats_container.clear()
                ui.label(f'Sent: {results["sent"]}').classes('text-positive')
                ui.label(f'Failed: {results["failed"]}').classes('text-negative')
                if results["errors"]:
                    with ui.expansion('Show Errors').classes('w-full'):
                        for error in results["errors"]:
                            ui.label(f'{error["email"]}: {error["error"]}').classes('text-body2 text-negative')
        
        results = await email_service.send_bulk_emails(
            selected_leads,
            selected_template,
            from_email.value,
            reply_to.value,
            provider.value,
            update_progress
        )
        
        ui.notify(
            f'Bulk send completed. {results["sent"]} sent, {results["failed"]} failed',
            type='positive' if results["failed"] == 0 else 'warning'
        )
    
    # UI Layout
    ui.label('Bulk Email Sender').classes('text-h4 q-ma-md')
    
    with ui.tabs().classes('w-full') as tabs:
        ui.tab('Template Selection')
        ui.tab('Lead Selection')
        ui.tab('Send Configuration')
    
    with ui.tab_panels(tabs).classes('w-full'):
        # Template Selection Panel
        with ui.tab_panel('Template Selection'):
            templates = await load_templates()
            
            with ui.card().classes('w-full'):
                template_select = ui.select(
                    label='Select Template',
                    options=[{'label': t['name'], 'value': t['id']} for t in templates]
                ).classes('w-full')
                
                with ui.row().classes('w-full justify-between'):
                    ui.button(
                        'Preview',
                        on_click=lambda: preview_template(template_select.value)
                    ).props('outlined')
                    ui.button(
                        'Optimize',
                        on_click=lambda: optimize_template(template_select.value, None)
                    ).props('primary')
        
        # Lead Selection Panel
        with ui.tab_panel('Lead Selection'):
            leads = await load_leads()
            
            with ui.card().classes('w-full'):
                with ui.table().props('rows-per-page-options=[10,20,50]').classes('w-full') as table:
                    ui.table_column('Select', 'select')
                    ui.table_column('Email', 'email')
                    ui.table_column('Name', 'name')
                    ui.table_column('Company', 'company')
                    ui.table_column('Last Contacted', 'last_contacted')
                    
                    rows = []
                    for lead in leads:
                        row = {
                            'select': ui.checkbox(on_change=lambda l=lead: toggle_lead(l)),
                            'email': lead['email'],
                            'name': lead['name'] or '',
                            'company': lead['company'] or '',
                            'last_contacted': lead['last_contacted'].strftime('%Y-%m-%d') if lead['last_contacted'] else 'Never'
                        }
                        rows.append(row)
                    
                    ui.table_rows(rows)
        
        # Send Configuration Panel
        with ui.tab_panel('Send Configuration'):
            with ui.card().classes('w-full'):
                with ui.column().classes('w-full q-pa-md'):
                    provider = ui.select(
                        'Email Provider',
                        options=[
                            {'label': 'SMTP', 'value': 'smtp'},
                            {'label': 'AWS SES', 'value': 'ses'}
                        ],
                        value='smtp'
                    )
                    
                    from_email = ui.input('From Email').props('outlined')
                    reply_to = ui.input('Reply-To Email').props('outlined')
                    
                    ui.button(
                        'Start Bulk Send',
                        on_click=start_bulk_send
                    ).props('primary').classes('q-mt-md')
                    
                    stats_container = ui.column().classes('w-full q-pa-md')
    
    # Dialogs
    preview_dialog = ui.dialog()
    optimization_dialog = ui.dialog()
    
    def toggle_lead(lead):
        if lead in selected_leads:
            selected_leads.remove(lead)
        else:
            selected_leads.append(lead) 