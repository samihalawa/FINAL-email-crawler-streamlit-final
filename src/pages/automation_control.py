from nicegui import ui
from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import plotly.graph_objects as go

from core.database import get_db, Campaign, SearchProcess, EmailCampaign
from core.auth import auth_handler
from core.background import task_manager
from services.ai import AIService
from services.search import SearchService
from services.email import EmailService

ai_service = AIService()
search_service = SearchService()
email_service = EmailService()

@ui.page('/automation-control')
@auth_handler.require_auth
async def page():
    # State management
    active_processes: Dict[str, Any] = {}
    stats: Dict[str, Any] = {}
    
    async def load_active_processes():
        """Load all active automation processes"""
        async with get_db() as session:
            processes = await session.execute(
                """
                SELECT sp.*, c.name as campaign_name
                FROM search_processes sp
                JOIN campaigns c ON sp.campaign_id = c.id
                WHERE sp.status IN ('running', 'pending')
                ORDER BY sp.created_at DESC
                """
            )
            return [dict(p) for p in processes]
    
    async def load_campaign_stats():
        """Load campaign statistics"""
        async with get_db() as session:
            stats = await session.execute(
                """
                SELECT 
                    c.id,
                    c.name,
                    COUNT(DISTINCT l.id) as total_leads,
                    COUNT(DISTINCT CASE WHEN ec.status = 'sent' THEN ec.id END) as emails_sent,
                    COUNT(DISTINCT CASE WHEN ec.opened_at IS NOT NULL THEN ec.id END) as emails_opened,
                    COUNT(DISTINCT CASE WHEN ec.replied_at IS NOT NULL THEN ec.id END) as emails_replied
                FROM campaigns c
                LEFT JOIN leads l ON l.campaign_id = c.id
                LEFT JOIN email_campaigns ec ON ec.lead_id = l.id
                GROUP BY c.id, c.name
                """
            )
            return [dict(s) for s in stats]
    
    def update_process_display():
        """Update the process monitoring display"""
        with process_container:
            process_container.clear()
            
            for process in active_processes:
                with ui.card().classes('w-full q-ma-sm'):
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(f"Campaign: {process['campaign_name']}").classes('text-h6')
                        ui.label(f"Status: {process['status']}").classes(
                            'text-positive' if process['status'] == 'running' else 'text-warning'
                        )
                    
                    with ui.row().classes('w-full'):
                        ui.label(f"Started: {process['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                        ui.label(f"Search Terms: {len(process['search_terms'])}")
                        ui.label(f"Found Leads: {process.get('total_leads', 0)}")
                    
                    if process['status'] == 'running':
                        ui.progress(value=process.get('progress', 0)).classes('w-full')
                        ui.button(
                            'Pause',
                            on_click=lambda p=process: pause_process(p['id'])
                        ).props('warning outline')
                    else:
                        ui.button(
                            'Resume',
                            on_click=lambda p=process: resume_process(p['id'])
                        ).props('positive outline')
                    
                    ui.button(
                        'Stop',
                        on_click=lambda p=process: stop_process(p['id'])
                    ).props('negative outline')
    
    def update_stats_display():
        """Update the statistics display"""
        with stats_container:
            stats_container.clear()
            
            # Create metrics cards
            with ui.row().classes('w-full justify-between'):
                for campaign in stats:
                    with ui.card().classes('w-1/3 q-ma-sm'):
                        ui.label(campaign['name']).classes('text-h6')
                        ui.label(f"Leads: {campaign['total_leads']}")
                        ui.label(f"Emails Sent: {campaign['emails_sent']}")
                        ui.label(f"Open Rate: {(campaign['emails_opened'] / campaign['emails_sent'] * 100 if campaign['emails_sent'] > 0 else 0):.1f}%")
                        ui.label(f"Reply Rate: {(campaign['emails_replied'] / campaign['emails_sent'] * 100 if campaign['emails_sent'] > 0 else 0):.1f}%")
            
            # Create performance graph
            fig = go.Figure()
            for campaign in stats:
                fig.add_trace(go.Scatter(
                    name=campaign['name'],
                    x=['Leads', 'Sent', 'Opened', 'Replied'],
                    y=[
                        campaign['total_leads'],
                        campaign['emails_sent'],
                        campaign['emails_opened'],
                        campaign['emails_replied']
                    ],
                    mode='lines+markers'
                ))
            
            ui.plotly(fig).classes('w-full h-64')
    
    async def start_automation():
        """Start new automation process"""
        campaign_id = campaign_select.value
        if not campaign_id:
            ui.notify('Please select a campaign', type='warning')
            return
        
        async with get_db() as session:
            campaign = await session.get(Campaign, campaign_id)
            if not campaign:
                ui.notify('Campaign not found', type='negative')
                return
            
            # Get search terms from knowledge base
            kb = await session.get_knowledge_base(campaign.project_id)
            if not kb:
                ui.notify('Knowledge base not found', type='negative')
                return
            
            # Generate search strategy
            strategy = await ai_service.generate_search_strategy(kb)
            
            # Create search process
            process = SearchProcess(
                campaign_id=campaign_id,
                search_terms=strategy.search_terms,
                status='pending',
                settings={
                    'target_audience': strategy.target_audience,
                    'rationale': strategy.rationale
                }
            )
            session.add(process)
            await session.commit()
            
            # Start background task
            task_id = await task_manager.add_task(
                search_service.perform_search,
                search_terms=strategy.search_terms,
                process_id=process.id
            )
            
            ui.notify('Automation process started successfully')
            await load_data()
    
    async def pause_process(process_id: int):
        """Pause automation process"""
        async with get_db() as session:
            process = await session.get(SearchProcess, process_id)
            if process:
                process.status = 'paused'
                await session.commit()
                ui.notify('Process paused successfully')
                await load_data()
    
    async def resume_process(process_id: int):
        """Resume automation process"""
        async with get_db() as session:
            process = await session.get(SearchProcess, process_id)
            if process:
                process.status = 'running'
                await session.commit()
                ui.notify('Process resumed successfully')
                await load_data()
    
    async def stop_process(process_id: int):
        """Stop automation process"""
        async with get_db() as session:
            process = await session.get(SearchProcess, process_id)
            if process:
                process.status = 'stopped'
                await session.commit()
                ui.notify('Process stopped successfully')
                await load_data()
    
    async def load_data():
        """Load all necessary data"""
        active_processes = await load_active_processes()
        stats = await load_campaign_stats()
        update_process_display()
        update_stats_display()
    
    # UI Layout
    ui.label('Automation Control Panel').classes('text-h4 q-ma-md')
    
    with ui.tabs().classes('w-full') as tabs:
        ui.tab('Process Monitor')
        ui.tab('Performance Analytics')
        ui.tab('Settings')
    
    with ui.tab_panels(tabs).classes('w-full'):
        # Process Monitor Panel
        with ui.tab_panel('Process Monitor'):
            with ui.card().classes('w-full q-ma-md'):
                ui.label('Start New Automation').classes('text-h6')
                campaign_select = ui.select(
                    'Select Campaign',
                    options=await load_campaigns()  # Implement this helper
                ).classes('w-full')
                ui.button(
                    'Start Automation',
                    on_click=start_automation
                ).props('primary').classes('q-mt-md')
            
            process_container = ui.column().classes('w-full')
        
        # Performance Analytics Panel
        with ui.tab_panel('Performance Analytics'):
            stats_container = ui.column().classes('w-full')
        
        # Settings Panel
        with ui.tab_panel('Settings'):
            with ui.card().classes('w-full q-ma-md'):
                ui.label('Automation Settings').classes('text-h6')
                
                ui.number('Max Concurrent Processes', value=5)
                ui.number('Search Delay (seconds)', value=2)
                ui.number('Email Delay (seconds)', value=5)
                ui.checkbox('Enable AI Optimization', value=True)
                ui.checkbox('Auto-pause on High Bounce Rate', value=True)
                
                ui.button('Save Settings').props('primary').classes('q-mt-md')
    
    # Start data refresh loop
    async def refresh_data():
        while True:
            await load_data()
            await asyncio.sleep(30)  # Refresh every 30 seconds
    
    asyncio.create_task(refresh_data())
    
    # Initial data load
    await load_data() 