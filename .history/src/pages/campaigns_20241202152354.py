from nicegui import ui
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from core.database import get_db, Campaign, Lead, EmailCampaign, SearchProcess
from core.auth import auth_handler
from services.ai import AIService

ai_service = AIService()

@ui.page('/campaigns')
@auth_handler.require_auth
async def page():
    current_campaign: Optional[Campaign] = None
    
    async def load_campaigns():
        """Load all campaigns with basic stats"""
        async with get_db() as session:
            campaigns = await session.execute("""
                SELECT 
                    c.*,
                    COUNT(DISTINCT l.id) as total_leads,
                    COUNT(DISTINCT CASE WHEN ec.status = 'sent' THEN ec.id END) as emails_sent,
                    COUNT(DISTINCT CASE WHEN ec.opened_at IS NOT NULL THEN ec.id END) as emails_opened,
                    COUNT(DISTINCT CASE WHEN ec.replied_at IS NOT NULL THEN ec.id END) as emails_replied
                FROM campaigns c
                LEFT JOIN leads l ON l.campaign_id = c.id
                LEFT JOIN email_campaigns ec ON ec.lead_id = l.id
                GROUP BY c.id
                ORDER BY c.created_at DESC
            """)
            return [dict(c) for c in campaigns]
    
    async def load_campaign_details(campaign_id: int):
        """Load detailed campaign information"""
        async with get_db() as session:
            # Basic info
            campaign = await session.get(Campaign, campaign_id)
            if not campaign:
                return None
            
            # Get time series data
            leads_over_time = await session.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM leads
                WHERE campaign_id = :campaign_id
                GROUP BY DATE(created_at)
                ORDER BY date
            """, {'campaign_id': campaign_id})
            
            emails_over_time = await session.execute("""
                SELECT 
                    DATE(sent_at) as date,
                    COUNT(*) as sent,
                    COUNT(CASE WHEN opened_at IS NOT NULL THEN 1 END) as opened,
                    COUNT(CASE WHEN replied_at IS NOT NULL THEN 1 END) as replied
                FROM email_campaigns ec
                JOIN leads l ON ec.lead_id = l.id
                WHERE l.campaign_id = :campaign_id
                GROUP BY DATE(sent_at)
                ORDER BY date
            """, {'campaign_id': campaign_id})
            
            return {
                'campaign': dict(campaign),
                'leads_over_time': [dict(r) for r in leads_over_time],
                'emails_over_time': [dict(r) for r in emails_over_time]
            }
    
    async def save_campaign():
        """Save campaign details"""
        async with get_db() as session:
            if current_campaign and current_campaign.id:
                campaign = await session.get(Campaign, current_campaign.id)
                campaign.name = name_input.value
                campaign.status = status_select.value
                campaign.settings = settings_dict
            else:
                campaign = Campaign(
                    name=name_input.value,
                    project_id=project_select.value,
                    status='active',
                    settings=settings_dict,
                    created_at=datetime.utcnow()
                )
                session.add(campaign)
            await session.commit()
            ui.notify('Campaign saved successfully')
            await refresh_campaign_list()
    
    async def delete_campaign(campaign_id: int):
        """Delete campaign and related data"""
        async with get_db() as session:
            campaign = await session.get(Campaign, campaign_id)
            if campaign:
                # Delete related data
                await session.execute(
                    "DELETE FROM email_campaigns WHERE lead_id IN "
                    "(SELECT id FROM leads WHERE campaign_id = :campaign_id)",
                    {'campaign_id': campaign_id}
                )
                await session.execute(
                    "DELETE FROM leads WHERE campaign_id = :campaign_id",
                    {'campaign_id': campaign_id}
                )
                await session.execute(
                    "DELETE FROM search_processes WHERE campaign_id = :campaign_id",
                    {'campaign_id': campaign_id}
                )
                await session.delete(campaign)
                await session.commit()
                ui.notify('Campaign deleted successfully')
                await refresh_campaign_list()
    
    def update_campaign_list(campaigns):
        """Update campaign list display"""
        with campaign_list:
            campaign_list.clear()
            
            for campaign in campaigns:
                with ui.card().classes('w-full q-ma-sm'):
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(campaign['name']).classes('text-h6')
                        ui.label(f"Status: {campaign['status']}").classes(
                            'text-positive' if campaign['status'] == 'active' else 'text-warning'
                        )
                    
                    with ui.row().classes('w-full justify-between'):
                        ui.label(f"Leads: {campaign['total_leads']}")
                        ui.label(f"Emails Sent: {campaign['emails_sent']}")
                        ui.label(f"Open Rate: {(campaign['emails_opened'] / campaign['emails_sent'] * 100 if campaign['emails_sent'] > 0 else 0):.1f}%")
                        ui.label(f"Reply Rate: {(campaign['emails_replied'] / campaign['emails_sent'] * 100 if campaign['emails_sent'] > 0 else 0):.1f}%")
                    
                    with ui.row().classes('w-full justify-end'):
                        ui.button(
                            'View Details',
                            on_click=lambda c=campaign: show_campaign_details(c['id'])
                        ).props('outline')
                        ui.button(
                            'Edit',
                            on_click=lambda c=campaign: load_campaign(c)
                        ).props('flat')
                        ui.button(
                            'Delete',
                            on_click=lambda c=campaign: delete_campaign(c['id'])
                        ).props('flat negative')
    
    async def show_campaign_details(campaign_id: int):
        """Show detailed campaign analytics"""
        details = await load_campaign_details(campaign_id)
        if not details:
            ui.notify('Campaign not found', type='negative')
            return
        
        with details_dialog:
            details_dialog.clear()
            ui.label(f"Campaign: {details['campaign']['name']}").classes('text-h6')
            
            with ui.tabs().classes('w-full') as tabs:
                ui.tab('Overview')
                ui.tab('Lead Analytics')
                ui.tab('Email Performance')
                ui.tab('Search Terms')
            
            with ui.tab_panels(tabs).classes('w-full'):
                # Overview Panel
                with ui.tab_panel('Overview'):
                    with ui.row().classes('w-full justify-between'):
                        with ui.card().classes('w-1/3'):
                            ui.label('Lead Generation').classes('text-h6')
                            ui.label(f"Total Leads: {details['campaign']['total_leads']}")
                            ui.label(f"Cost per Lead: ${details['campaign'].get('cost_per_lead', 0):.2f}")
                        
                        with ui.card().classes('w-1/3'):
                            ui.label('Email Performance').classes('text-h6')
                            ui.label(f"Emails Sent: {details['campaign']['emails_sent']}")
                            ui.label(f"Open Rate: {details['campaign'].get('open_rate', 0):.1f}%")
                            ui.label(f"Reply Rate: {details['campaign'].get('reply_rate', 0):.1f}%")
                        
                        with ui.card().classes('w-1/3'):
                            ui.label('Campaign Status').classes('text-h6')
                            ui.label(f"Status: {details['campaign']['status']}")
                            ui.label(f"Duration: {(datetime.utcnow() - details['campaign']['created_at']).days} days")
                
                # Lead Analytics Panel
                with ui.tab_panel('Lead Analytics'):
                    # Leads over time graph
                    df_leads = pd.DataFrame(details['leads_over_time'])
                    fig_leads = px.line(
                        df_leads,
                        x='date',
                        y='count',
                        title='Leads Generated Over Time'
                    )
                    ui.plotly(fig_leads).classes('w-full h-64')
                    
                    # Lead sources breakdown
                    lead_sources = pd.DataFrame({
                        'source': ['Search', 'Manual', 'Import'],
                        'count': [150, 30, 20]  # Example data
                    })
                    fig_sources = px.pie(
                        lead_sources,
                        values='count',
                        names='source',
                        title='Lead Sources'
                    )
                    ui.plotly(fig_sources).classes('w-full h-64')
                
                # Email Performance Panel
                with ui.tab_panel('Email Performance'):
                    # Email metrics over time
                    df_emails = pd.DataFrame(details['emails_over_time'])
                    fig_emails = go.Figure()
                    fig_emails.add_trace(go.Scatter(
                        x=df_emails['date'],
                        y=df_emails['sent'],
                        name='Sent'
                    ))
                    fig_emails.add_trace(go.Scatter(
                        x=df_emails['date'],
                        y=df_emails['opened'],
                        name='Opened'
                    ))
                    fig_emails.add_trace(go.Scatter(
                        x=df_emails['date'],
                        y=df_emails['replied'],
                        name='Replied'
                    ))
                    fig_emails.update_layout(title='Email Performance Over Time')
                    ui.plotly(fig_emails).classes('w-full h-64')
                
                # Search Terms Panel
                with ui.tab_panel('Search Terms'):
                    search_terms = await load_search_terms(campaign_id)
                    with ui.table().props('rows-per-page-options=[10,20,50]'):
                        ui.table_column('Term', 'term')
                        ui.table_column('Total Results', 'total_results')
                        ui.table_column('Valid Leads', 'valid_leads')
                        ui.table_column('Success Rate', 'success_rate')
                        
                        rows = []
                        for term in search_terms:
                            success_rate = term['valid_leads'] / term['total_results'] * 100 if term['total_results'] > 0 else 0
                            rows.append({
                                'term': term['term'],
                                'total_results': term['total_results'],
                                'valid_leads': term['valid_leads'],
                                'success_rate': f"{success_rate:.1f}%"
                            })
                        
                        ui.table_rows(rows)
        
        details_dialog.open()
    
    def load_campaign(campaign: Dict):
        """Load campaign into editor"""
        nonlocal current_campaign
        current_campaign = campaign
        name_input.value = campaign['name']
        status_select.value = campaign['status']
        project_select.value = campaign['project_id']
        settings_dict.update(campaign['settings'])
        editor_card.set_visibility(True)
    
    async def refresh_campaign_list():
        """Refresh campaign list"""
        campaigns = await load_campaigns()
        update_campaign_list(campaigns)
    
    # UI Layout
    ui.label('Campaign Management').classes('text-h4 q-ma-md')
    
    with ui.row().classes('w-full justify-between q-ma-md'):
        ui.button(
            'New Campaign',
            on_click=lambda: editor_card.set_visibility(True)
        ).props('primary')
        ui.button(
            'Refresh',
            on_click=refresh_campaign_list
        ).props('outline')
    
    # Campaign List
    campaign_list = ui.column().classes('w-full')
    
    # Campaign Editor
    with ui.card().classes('w-full q-ma-md') as editor_card:
        editor_card.set_visibility(False)
        
        ui.label('Campaign Editor').classes('text-h6')
        
        name_input = ui.input('Campaign Name').classes('w-full')
        project_select = ui.select(
            'Project',
            options=await load_projects()  # Implement this helper
        ).classes('w-full')
        status_select = ui.select(
            'Status',
            options=[
                {'label': 'Active', 'value': 'active'},
                {'label': 'Paused', 'value': 'paused'},
                {'label': 'Completed', 'value': 'completed'}
            ]
        ).classes('w-full')
        
        # Campaign Settings
        settings_dict = {}
        with ui.expansion('Settings', value=True):
            ui.number('Daily Lead Limit', value=100, on_change=lambda e: settings_dict.update({'daily_lead_limit': e.value}))
            ui.number('Email Delay (seconds)', value=5, on_change=lambda e: settings_dict.update({'email_delay': e.value}))
            ui.checkbox('Auto-optimize Templates', value=True, on_change=lambda e: settings_dict.update({'auto_optimize': e.value}))
            ui.checkbox('Auto-pause on High Bounce', value=True, on_change=lambda e: settings_dict.update({'auto_pause': e.value}))
        
        with ui.row().classes('w-full justify-end'):
            ui.button('Save', on_click=save_campaign).props('primary')
            ui.button('Cancel', on_click=lambda: editor_card.set_visibility(False)).props('flat')
    
    # Dialogs
    details_dialog = ui.dialog()
    
    # Load initial data
    await refresh_campaign_list() 