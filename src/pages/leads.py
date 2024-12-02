from nicegui import ui
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import plotly.express as px

from core.database import get_db, Lead, EmailCampaign
from core.auth import auth_handler
from services.ai import AIService

ai_service = AIService()

@ui.page('/leads')
@auth_handler.require_auth
async def page():
    # State management
    selected_leads: List[int] = []
    filter_conditions: Dict[str, Any] = {}
    
    async def load_leads():
        """Load leads with filters"""
        query = """
            SELECT 
                l.*,
                c.name as campaign_name,
                COUNT(DISTINCT ec.id) as emails_sent,
                MAX(ec.sent_at) as last_contacted,
                COUNT(DISTINCT CASE WHEN ec.opened_at IS NOT NULL THEN ec.id END) as emails_opened,
                COUNT(DISTINCT CASE WHEN ec.replied_at IS NOT NULL THEN ec.id END) as emails_replied
            FROM leads l
            LEFT JOIN campaigns c ON l.campaign_id = c.id
            LEFT JOIN email_campaigns ec ON l.id = ec.lead_id
        """
        
        conditions = []
        params = {}
        
        if filter_conditions.get('campaign_id'):
            conditions.append("l.campaign_id = :campaign_id")
            params['campaign_id'] = filter_conditions['campaign_id']
        
        if filter_conditions.get('status'):
            if filter_conditions['status'] == 'contacted':
                conditions.append("ec.id IS NOT NULL")
            elif filter_conditions['status'] == 'not_contacted':
                conditions.append("ec.id IS NULL")
            elif filter_conditions['status'] == 'replied':
                conditions.append("ec.replied_at IS NOT NULL")
        
        if filter_conditions.get('search'):
            search_term = f"%{filter_conditions['search']}%"
            conditions.append("(l.email LIKE :search OR l.name LIKE :search OR l.company LIKE :search)")
            params['search'] = search_term
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " GROUP BY l.id ORDER BY l.created_at DESC"
        
        async with get_db() as session:
            results = await session.execute(query, params)
            return [dict(r) for r in results]
    
    async def validate_leads():
        """Validate selected leads with AI"""
        if not selected_leads:
            ui.notify('Please select leads to validate', type='warning')
            return
        
        async with get_db() as session:
            leads = []
            for lead_id in selected_leads:
                lead = await session.get(Lead, lead_id)
                if lead:
                    leads.append(lead)
            
            with ui.loading('Validating leads...'):
                results = []
                for lead in leads:
                    validation = await ai_service.validate_lead_quality(
                        lead,
                        {
                            'min_company_size': 50,
                            'target_positions': ['CEO', 'CTO', 'Director', 'Manager'],
                            'required_fields': ['email', 'company']
                        }
                    )
                    results.append({
                        'lead': lead,
                        'validation': validation
                    })
                
                with validation_dialog:
                    validation_dialog.clear()
                    ui.label('Lead Validation Results').classes('text-h6')
                    
                    for result in results:
                        with ui.card().classes('w-full q-ma-sm'):
                            ui.label(f"Lead: {result['lead'].email}").classes('text-bold')
                            ui.label(f"Quality Score: {result['validation'].get('quality_score', 0)}%")
                            
                            if result['validation'].get('matches'):
                                with ui.expansion('Matches', value=True):
                                    for match in result['validation']['matches']:
                                        ui.label(f"✓ {match}").classes('text-positive')
                            
                            if result['validation'].get('mismatches'):
                                with ui.expansion('Issues'):
                                    for mismatch in result['validation']['mismatches']:
                                        ui.label(f"✗ {mismatch}").classes('text-negative')
                            
                            if result['validation'].get('recommendations'):
                                with ui.expansion('Recommendations'):
                                    for rec in result['validation']['recommendations']:
                                        ui.label(f"• {rec}")
                
                validation_dialog.open()
    
    async def export_leads():
        """Export selected leads"""
        if not selected_leads:
            ui.notify('Please select leads to export', type='warning')
            return
        
        async with get_db() as session:
            leads = []
            for lead_id in selected_leads:
                lead = await session.get(Lead, lead_id)
                if lead:
                    leads.append({
                        'Email': lead.email,
                        'Name': lead.name,
                        'Company': lead.company,
                        'Position': lead.position,
                        'Source': lead.source_url,
                        'Created': lead.created_at.strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            if leads:
                df = pd.DataFrame(leads)
                csv = df.to_csv(index=False)
                ui.download(csv, 'leads_export.csv')
    
    async def delete_leads():
        """Delete selected leads"""
        if not selected_leads:
            ui.notify('Please select leads to delete', type='warning')
            return
        
        async with get_db() as session:
            for lead_id in selected_leads:
                lead = await session.get(Lead, lead_id)
                if lead:
                    await session.delete(lead)
            await session.commit()
            ui.notify('Leads deleted successfully')
            await refresh_leads()
    
    def update_lead_table(leads: List[Dict]):
        """Update the lead table display"""
        with lead_table:
            lead_table.clear()
            
            with ui.table().props('rows-per-page-options=[10,20,50]').classes('w-full'):
                ui.table_column('Select', 'select')
                ui.table_column('Email', 'email')
                ui.table_column('Name', 'name')
                ui.table_column('Company', 'company')
                ui.table_column('Campaign', 'campaign_name')
                ui.table_column('Emails Sent', 'emails_sent')
                ui.table_column('Last Contacted', 'last_contacted')
                ui.table_column('Status', 'status')
                ui.table_column('Actions', 'actions')
                
                rows = []
                for lead in leads:
                    status = 'Not Contacted'
                    if lead['emails_replied'] > 0:
                        status = 'Replied'
                    elif lead['emails_opened'] > 0:
                        status = 'Opened'
                    elif lead['emails_sent'] > 0:
                        status = 'Contacted'
                    
                    row = {
                        'select': ui.checkbox(
                            value=lead['id'] in selected_leads,
                            on_change=lambda e, l=lead: toggle_lead(l['id'], e.value)
                        ),
                        'email': lead['email'],
                        'name': lead['name'] or '',
                        'company': lead['company'] or '',
                        'campaign_name': lead['campaign_name'],
                        'emails_sent': lead['emails_sent'],
                        'last_contacted': lead['last_contacted'].strftime('%Y-%m-%d %H:%M') if lead['last_contacted'] else 'Never',
                        'status': status,
                        'actions': ui.button(
                            'View History',
                            on_click=lambda l=lead: show_lead_history(l['id'])
                        ).props('outline')
                    }
                    rows.append(row)
                
                ui.table_rows(rows)
    
    def toggle_lead(lead_id: int, selected: bool):
        """Toggle lead selection"""
        if selected and lead_id not in selected_leads:
            selected_leads.append(lead_id)
        elif not selected and lead_id in selected_leads:
            selected_leads.remove(lead_id)
    
    async def show_lead_history(lead_id: int):
        """Show lead interaction history"""
        async with get_db() as session:
            lead = await session.get(Lead, lead_id)
            if not lead:
                ui.notify('Lead not found', type='negative')
                return
            
            campaigns = await session.execute("""
                SELECT 
                    ec.*,
                    et.name as template_name,
                    et.subject
                FROM email_campaigns ec
                JOIN email_templates et ON ec.template_id = et.id
                WHERE ec.lead_id = :lead_id
                ORDER BY ec.sent_at DESC
            """, {'lead_id': lead_id})
            
            with history_dialog:
                history_dialog.clear()
                ui.label(f"Lead History: {lead.email}").classes('text-h6')
                
                with ui.tabs().classes('w-full') as tabs:
                    ui.tab('Overview')
                    ui.tab('Email History')
                    ui.tab('Analytics')
                
                with ui.tab_panels(tabs).classes('w-full'):
                    # Overview Panel
                    with ui.tab_panel('Overview'):
                        with ui.row().classes('w-full'):
                            with ui.card().classes('w-1/2'):
                                ui.label('Lead Information').classes('text-h6')
                                ui.label(f"Name: {lead.name or 'N/A'}")
                                ui.label(f"Company: {lead.company or 'N/A'}")
                                ui.label(f"Position: {lead.position or 'N/A'}")
                                ui.label(f"Source: {lead.source_url}")
                                ui.label(f"Created: {lead.created_at.strftime('%Y-%m-%d %H:%M')}")
                            
                            with ui.card().classes('w-1/2'):
                                ui.label('Engagement Summary').classes('text-h6')
                                campaigns_list = [dict(c) for c in campaigns]
                                total_sent = len(campaigns_list)
                                total_opened = sum(1 for c in campaigns_list if c['opened_at'])
                                total_replied = sum(1 for c in campaigns_list if c['replied_at'])
                                
                                ui.label(f"Total Emails Sent: {total_sent}")
                                ui.label(f"Open Rate: {(total_opened/total_sent*100 if total_sent > 0 else 0):.1f}%")
                                ui.label(f"Reply Rate: {(total_replied/total_sent*100 if total_sent > 0 else 0):.1f}%")
                    
                    # Email History Panel
                    with ui.tab_panel('Email History'):
                        for campaign in campaigns_list:
                            with ui.card().classes('w-full q-ma-sm'):
                                with ui.row().classes('w-full justify-between'):
                                    ui.label(f"Template: {campaign['template_name']}").classes('text-bold')
                                    ui.label(f"Sent: {campaign['sent_at'].strftime('%Y-%m-%d %H:%M')}")
                                
                                ui.label(f"Subject: {campaign['subject']}")
                                
                                with ui.row().classes('w-full'):
                                    if campaign['opened_at']:
                                        ui.label(f"Opened: {campaign['opened_at'].strftime('%Y-%m-%d %H:%M')}").classes('text-positive')
                                    if campaign['replied_at']:
                                        ui.label(f"Replied: {campaign['replied_at'].strftime('%Y-%m-%d %H:%M')}").classes('text-positive')
                    
                    # Analytics Panel
                    with ui.tab_panel('Analytics'):
                        # Create engagement timeline
                        events = []
                        for campaign in campaigns_list:
                            events.append({
                                'date': campaign['sent_at'],
                                'event': 'Email Sent',
                                'template': campaign['template_name']
                            })
                            if campaign['opened_at']:
                                events.append({
                                    'date': campaign['opened_at'],
                                    'event': 'Email Opened',
                                    'template': campaign['template_name']
                                })
                            if campaign['replied_at']:
                                events.append({
                                    'date': campaign['replied_at'],
                                    'event': 'Email Replied',
                                    'template': campaign['template_name']
                                })
                        
                        df_events = pd.DataFrame(events)
                        if not df_events.empty:
                            fig = px.timeline(
                                df_events,
                                x_start='date',
                                y='template',
                                color='event',
                                title='Engagement Timeline'
                            )
                            ui.plotly(fig).classes('w-full h-64')
                
                history_dialog.open()
    
    async def refresh_leads():
        """Refresh leads list"""
        leads = await load_leads()
        update_lead_table(leads)
    
    # UI Layout
    ui.label('Lead Management').classes('text-h4 q-ma-md')
    
    # Filters
    with ui.card().classes('w-full q-ma-md'):
        with ui.row().classes('w-full items-center'):
            ui.input(
                'Search',
                on_change=lambda e: filter_conditions.update({'search': e.value})
            ).classes('w-1/3')
            
            ui.select(
                'Campaign',
                options=await load_campaigns(),  # Implement this helper
                on_change=lambda e: filter_conditions.update({'campaign_id': e.value})
            ).classes('w-1/3')
            
            ui.select(
                'Status',
                options=[
                    {'label': 'All', 'value': ''},
                    {'label': 'Not Contacted', 'value': 'not_contacted'},
                    {'label': 'Contacted', 'value': 'contacted'},
                    {'label': 'Replied', 'value': 'replied'}
                ],
                on_change=lambda e: filter_conditions.update({'status': e.value})
            ).classes('w-1/3')
        
        with ui.row().classes('w-full justify-end'):
            ui.button('Apply Filters', on_click=refresh_leads).props('primary')
            ui.button('Reset', on_click=lambda: filter_conditions.clear() or refresh_leads()).props('outline')
    
    # Action Buttons
    with ui.row().classes('w-full justify-between q-ma-md'):
        with ui.row():
            ui.button('Validate Selected', on_click=validate_leads).props('secondary')
            ui.button('Export Selected', on_click=export_leads).props('secondary')
            ui.button('Delete Selected', on_click=delete_leads).props('negative outline')
        
        ui.button('Refresh', on_click=refresh_leads).props('outline')
    
    # Lead Table
    lead_table = ui.column().classes('w-full')
    
    # Dialogs
    validation_dialog = ui.dialog()
    history_dialog = ui.dialog()
    
    # Load initial data
    await refresh_leads() 