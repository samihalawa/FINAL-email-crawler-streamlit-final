from nicegui import ui
from typing import Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px

from core.database import get_db, EmailLog
from core.auth import auth_handler
from services.ai import AIService

ai_service = AIService()

@ui.page('/email-logs')
@auth_handler.require_auth
async def page():
    # State management
    logs: List[EmailLog] = []
    filter_options: Dict[str, Any] = {
        'project_id': None,
        'campaign_id': None,
        'status': None,
        'date_range': 'last_7_days',
        'search_query': ''
    }
    
    async def load_logs():
        """Load email logs with filters"""
        async with get_db() as session:
            query = "SELECT * FROM email_logs WHERE 1=1"
            params = {}
            
            # Apply filters
            if filter_options['project_id']:
                query += " AND project_id = :project_id"
                params['project_id'] = filter_options['project_id']
            
            if filter_options['campaign_id']:
                query += " AND campaign_id = :campaign_id"
                params['campaign_id'] = filter_options['campaign_id']
            
            if filter_options['status']:
                query += " AND status = :status"
                params['status'] = filter_options['status']
            
            # Date range
            if filter_options['date_range']:
                end_date = datetime.utcnow()
                if filter_options['date_range'] == 'last_24h':
                    start_date = end_date - timedelta(days=1)
                elif filter_options['date_range'] == 'last_7_days':
                    start_date = end_date - timedelta(days=7)
                elif filter_options['date_range'] == 'last_30_days':
                    start_date = end_date - timedelta(days=30)
                elif filter_options['date_range'] == 'last_90_days':
                    start_date = end_date - timedelta(days=90)
                
                query += " AND created_at BETWEEN :start_date AND :end_date"
                params.update({
                    'start_date': start_date,
                    'end_date': end_date
                })
            
            # Search query
            if filter_options['search_query']:
                query += " AND (recipient_email LIKE :search OR subject LIKE :search OR message_id LIKE :search)"
                params['search'] = f"%{filter_options['search_query']}%"
            
            # Order by date
            query += " ORDER BY created_at DESC LIMIT 1000"
            
            results = await session.execute(query, params)
            logs.clear()
            logs.extend(results.scalars().all())
            
            update_logs_display()
            update_analytics()
    
    async def get_projects():
        """Get list of projects"""
        async with get_db() as session:
            results = await session.execute("SELECT id, name FROM projects ORDER BY name")
            return [(row.id, row.name) for row in results]
    
    async def get_campaigns(project_id: int = None):
        """Get list of campaigns"""
        async with get_db() as session:
            query = "SELECT id, name FROM campaigns"
            if project_id:
                query += " WHERE project_id = :project_id"
            query += " ORDER BY name"
            
            results = await session.execute(query, {'project_id': project_id} if project_id else {})
            return [(row.id, row.name) for row in results]
    
    def update_logs_display():
        """Update logs table"""
        with logs_container:
            logs_container.clear()
            
            # Create table
            columns = [
                {'name': 'date', 'label': 'Date', 'field': 'created_at', 'sortable': True},
                {'name': 'project', 'label': 'Project', 'field': 'project_name', 'sortable': True},
                {'name': 'campaign', 'label': 'Campaign', 'field': 'campaign_name', 'sortable': True},
                {'name': 'recipient', 'label': 'Recipient', 'field': 'recipient_email', 'sortable': True},
                {'name': 'subject', 'label': 'Subject', 'field': 'subject'},
                {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True},
                {'name': 'opens', 'label': 'Opens', 'field': 'open_count', 'sortable': True},
                {'name': 'clicks', 'label': 'Clicks', 'field': 'click_count', 'sortable': True},
                {'name': 'message_id', 'label': 'Message ID', 'field': 'message_id'}
            ]
            
            rows = [{
                'created_at': log.created_at.strftime('%Y-%m-%d %H:%M'),
                'project_name': log.project.name if log.project else '',
                'campaign_name': log.campaign.name if log.campaign else '',
                'recipient_email': log.recipient_email,
                'subject': log.subject,
                'status': log.status,
                'open_count': log.open_count,
                'click_count': log.click_count,
                'message_id': log.message_id
            } for log in logs]
            
            ui.table(
                columns=columns,
                rows=rows,
                row_key='message_id',
                pagination={'rowsPerPage': 20}
            ).classes('w-full')
    
    def update_analytics():
        """Update analytics charts"""
        with analytics_container:
            analytics_container.clear()
            
            if not logs:
                ui.label('No data available for the selected filters')
                return
            
            # Convert logs to pandas DataFrame for analysis
            df = pd.DataFrame([{
                'date': log.created_at.date(),
                'status': log.status,
                'project': log.project.name if log.project else 'Unknown',
                'campaign': log.campaign.name if log.campaign else 'Unknown',
                'opens': log.open_count,
                'clicks': log.click_count
            } for log in logs])
            
            with ui.grid(columns=2).classes('w-full gap-4'):
                # Daily Email Volume
                daily_volume = df.groupby('date').size().reset_index()
                daily_volume.columns = ['date', 'count']
                fig = px.line(
                    daily_volume,
                    x='date',
                    y='count',
                    title='Daily Email Volume'
                )
                ui.plotly(fig).classes('w-full')
                
                # Status Distribution
                status_dist = df['status'].value_counts()
                fig = px.pie(
                    values=status_dist.values,
                    names=status_dist.index,
                    title='Email Status Distribution'
                )
                ui.plotly(fig).classes('w-full')
                
                # Open Rates by Project
                project_stats = df.groupby('project').agg({
                    'opens': 'sum',
                    'status': 'count'
                }).reset_index()
                project_stats['open_rate'] = project_stats['opens'] / project_stats['status']
                fig = px.bar(
                    project_stats,
                    x='project',
                    y='open_rate',
                    title='Open Rates by Project'
                )
                ui.plotly(fig).classes('w-full')
                
                # Click Rates by Campaign
                campaign_stats = df.groupby('campaign').agg({
                    'clicks': 'sum',
                    'status': 'count'
                }).reset_index()
                campaign_stats['click_rate'] = campaign_stats['clicks'] / campaign_stats['status']
                fig = px.bar(
                    campaign_stats,
                    x='campaign',
                    y='click_rate',
                    title='Click Rates by Campaign'
                )
                ui.plotly(fig).classes('w-full')
    
    async def export_logs():
        """Export logs to CSV"""
        if not logs:
            ui.notify('No data to export')
            return
        
        # Convert logs to DataFrame
        df = pd.DataFrame([{
            'date': log.created_at,
            'project': log.project.name if log.project else 'Unknown',
            'campaign': log.campaign.name if log.campaign else 'Unknown',
            'recipient': log.recipient_email,
            'subject': log.subject,
            'status': log.status,
            'opens': log.open_count,
            'clicks': log.click_count,
            'message_id': log.message_id,
            'bounce_type': log.bounce_type,
            'bounce_detail': log.bounce_detail,
            'complaint_type': log.complaint_type,
            'complaint_detail': log.complaint_detail
        } for log in logs])
        
        # Save to CSV
        filename = f'email_logs_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filename, index=False)
        ui.notify(f'Logs exported to {filename}')
    
    async def analyze_performance():
        """AI-powered performance analysis"""
        if not logs:
            ui.notify('No data available for analysis')
            return
        
        with ui.loading('Analyzing performance...'):
            # Prepare data for analysis
            data = {
                'total_emails': len(logs),
                'delivery_rate': sum(1 for log in logs if log.status == 'delivered') / len(logs),
                'open_rate': sum(1 for log in logs if log.open_count > 0) / len(logs),
                'click_rate': sum(1 for log in logs if log.click_count > 0) / len(logs),
                'bounce_rate': sum(1 for log in logs if log.status == 'bounced') / len(logs),
                'complaint_rate': sum(1 for log in logs if log.status == 'complaint') / len(logs)
            }
            
            # Get AI analysis
            analysis = await ai_service.analyze_email_performance(data)
            
            # Show analysis in dialog
            with ui.dialog() as dialog, ui.card():
                ui.label('Performance Analysis').classes('text-h6')
                ui.markdown(analysis)
                ui.button('Close', on_click=dialog.close).props('flat')
            dialog.open()
    
    # UI Layout
    ui.label('Email Logs').classes('text-h4 q-ma-md')
    
    # Filters
    with ui.card().classes('w-full q-ma-md'):
        with ui.grid(columns=4).classes('w-full gap-4'):
            # Project filter
            ui.select(
                'Project',
                options=await get_projects(),
                value=filter_options['project_id'],
                on_change=lambda e: filter_options.update({'project_id': e.value})
            )
            
            # Campaign filter
            ui.select(
                'Campaign',
                options=await get_campaigns(filter_options['project_id']),
                value=filter_options['campaign_id'],
                on_change=lambda e: filter_options.update({'campaign_id': e.value})
            )
            
            # Status filter
            ui.select(
                'Status',
                options=[
                    ('', 'All'),
                    ('sent', 'Sent'),
                    ('delivered', 'Delivered'),
                    ('opened', 'Opened'),
                    ('clicked', 'Clicked'),
                    ('bounced', 'Bounced'),
                    ('complaint', 'Complaint')
                ],
                value=filter_options['status'],
                on_change=lambda e: filter_options.update({'status': e.value})
            )
            
            # Date range filter
            ui.select(
                'Date Range',
                options=[
                    ('last_24h', 'Last 24 Hours'),
                    ('last_7_days', 'Last 7 Days'),
                    ('last_30_days', 'Last 30 Days'),
                    ('last_90_days', 'Last 90 Days')
                ],
                value=filter_options['date_range'],
                on_change=lambda e: filter_options.update({'date_range': e.value})
            )
        
        # Search
        with ui.row().classes('w-full items-center'):
            ui.input(
                placeholder='Search by email, subject, or message ID',
                value=filter_options['search_query'],
                on_change=lambda e: filter_options.update({'search_query': e.value})
            ).classes('w-full')
            ui.button('Search', on_click=load_logs).props('primary')
            ui.button('Reset', on_click=lambda: filter_options.clear() or load_logs()).props('outline')
    
    # Action buttons
    with ui.row().classes('w-full justify-end q-ma-md'):
        ui.button('Export to CSV', on_click=export_logs).props('outline')
        ui.button('Analyze Performance', on_click=analyze_performance).props('primary')
    
    # Tabs for Logs and Analytics
    with ui.tabs().classes('w-full') as tabs:
        ui.tab('Logs')
        ui.tab('Analytics')
    
    with ui.tab_panels(tabs).classes('w-full'):
        # Logs panel
        with ui.tab_panel('Logs'):
            logs_container = ui.column().classes('w-full')
        
        # Analytics panel
        with ui.tab_panel('Analytics'):
            analytics_container = ui.column().classes('w-full')
    
    # Load initial data
    await load_logs() 