from nicegui import ui
from typing import Dict, Any, List
from datetime import datetime
import asyncio

from core.database import get_db, Project, Campaign
from core.auth import auth_handler
from core.background import task_manager
from services.ai import AIService
from services.search import SearchService

ai_service = AIService()
search_service = SearchService()

@ui.page('/projects')
@auth_handler.require_auth
async def page():
    # State management
    projects: List[Project] = []
    selected_project: Dict[str, Any] = {}
    
    async def load_projects():
        """Load all projects"""
        async with get_db() as session:
            projects.clear()
            results = await session.execute("SELECT * FROM projects ORDER BY created_at DESC")
            projects.extend(results.scalars().all())
            update_projects_display()
    
    async def create_project():
        """Create a new project"""
        name = await ui.ask('Enter project name:')
        if name:
            async with get_db() as session:
                project = Project(
                    name=name,
                    description='',
                    status='active',
                    created_at=datetime.utcnow(),
                    settings={
                        'search_terms': [],
                        'excluded_domains': [],
                        'target_industries': [],
                        'email_templates': [],
                        'daily_limits': {
                            'leads': 100,
                            'emails': 200
                        },
                        'automation': {
                            'enabled': False,
                            'schedule': '0 9 * * 1-5',  # Mon-Fri 9am
                            'auto_pause_threshold': 0.1  # 10% bounce rate
                        }
                    }
                )
                session.add(project)
                await session.commit()
                
                projects.append(project)
                update_projects_display()
                ui.notify(f'Project "{name}" created successfully')
    
    async def delete_project(project_id: int):
        """Delete a project"""
        if await ui.ask(f'Are you sure you want to delete this project? This action cannot be undone.'):
            async with get_db() as session:
                project = await session.get(Project, project_id)
                if project:
                    await session.delete(project)
                    await session.commit()
                    projects.remove(project)
                    update_projects_display()
                    ui.notify('Project deleted successfully')
    
    async def update_project(project_id: int):
        """Update project settings"""
        async with get_db() as session:
            project = await session.get(Project, project_id)
            if project:
                project.settings.update(selected_project)
                project.updated_at = datetime.utcnow()
                await session.commit()
                ui.notify('Project settings updated successfully')
    
    async def start_automation(project_id: int):
        """Start project automation"""
        async with get_db() as session:
            project = await session.get(Project, project_id)
            if project and not project.settings['automation']['enabled']:
                project.settings['automation']['enabled'] = True
                await session.commit()
                
                # Start background tasks
                task_manager.start_project_tasks(project_id)
                ui.notify('Project automation started')
                update_projects_display()
    
    async def stop_automation(project_id: int):
        """Stop project automation"""
        async with get_db() as session:
            project = await session.get(Project, project_id)
            if project and project.settings['automation']['enabled']:
                project.settings['automation']['enabled'] = False
                await session.commit()
                
                # Stop background tasks
                task_manager.stop_project_tasks(project_id)
                ui.notify('Project automation stopped')
                update_projects_display()
    
    async def get_project_stats(project_id: int) -> Dict[str, Any]:
        """Get project statistics"""
        async with get_db() as session:
            stats = {
                'total_leads': 0,
                'total_emails_sent': 0,
                'response_rate': 0,
                'bounce_rate': 0,
                'active_campaigns': 0
            }
            
            # Get campaign stats
            results = await session.execute(
                "SELECT COUNT(*) as count, SUM(emails_sent) as sent, "
                "SUM(responses) as responses, SUM(bounces) as bounces "
                "FROM campaigns WHERE project_id = :project_id",
                {'project_id': project_id}
            )
            row = results.first()
            if row:
                stats['active_campaigns'] = row.count
                stats['total_emails_sent'] = row.sent or 0
                if stats['total_emails_sent'] > 0:
                    stats['response_rate'] = (row.responses or 0) / stats['total_emails_sent']
                    stats['bounce_rate'] = (row.bounces or 0) / stats['total_emails_sent']
            
            # Get leads count
            results = await session.execute(
                "SELECT COUNT(*) FROM leads WHERE project_id = :project_id",
                {'project_id': project_id}
            )
            stats['total_leads'] = results.scalar()
            
            return stats
    
    def update_projects_display():
        """Update projects UI"""
        with projects_container:
            projects_container.clear()
            
            # Projects Grid
            with ui.grid(columns=3).classes('w-full gap-4'):
                for project in projects:
                    with ui.card().classes('w-full'):
                        with ui.row().classes('w-full items-center justify-between'):
                            ui.label(project.name).classes('text-h6')
                            ui.badge(project.status).props(f'color={get_status_color(project.status)}')
                        
                        ui.label(f'Created: {project.created_at.strftime("%Y-%m-%d")}')
                        
                        if project.description:
                            ui.label(project.description).classes('text-caption')
                        
                        # Quick stats
                        asyncio.create_task(display_project_stats(project.id))
                        
                        # Action buttons
                        with ui.row().classes('w-full justify-end'):
                            ui.button(
                                'Settings',
                                on_click=lambda p=project: open_settings(p)
                            ).props('outline')
                            
                            if project.settings['automation']['enabled']:
                                ui.button(
                                    'Stop Automation',
                                    on_click=lambda id=project.id: stop_automation(id)
                                ).props('negative')
                            else:
                                ui.button(
                                    'Start Automation',
                                    on_click=lambda id=project.id: start_automation(id)
                                ).props('positive')
                            
                            ui.button(
                                'Delete',
                                on_click=lambda id=project.id: delete_project(id)
                            ).props('flat negative')
    
    async def display_project_stats(project_id: int):
        """Display project statistics"""
        stats = await get_project_stats(project_id)
        with ui.row().classes('w-full justify-between text-caption'):
            ui.label(f'Leads: {stats["total_leads"]}')
            ui.label(f'Emails: {stats["total_emails_sent"]}')
            ui.label(f'Response Rate: {stats["response_rate"]:.1%}')
    
    def get_status_color(status: str) -> str:
        """Get color for status badge"""
        return {
            'active': 'positive',
            'paused': 'warning',
            'completed': 'info',
            'error': 'negative'
        }.get(status, 'grey')
    
    def open_settings(project: Project):
        """Open project settings dialog"""
        selected_project.clear()
        selected_project.update(project.settings)
        
        with ui.dialog() as dialog, ui.card().classes('w-160'):
            ui.label(f'Settings: {project.name}').classes('text-h6 q-mb-md')
            
            with ui.tabs().classes('w-full') as tabs:
                ui.tab('General')
                ui.tab('Search')
                ui.tab('Automation')
                ui.tab('Templates')
            
            with ui.tab_panels(tabs).classes('w-full'):
                # General Settings
                with ui.tab_panel('General'):
                    ui.input(
                        'Project Name',
                        value=project.name,
                        on_change=lambda e: setattr(project, 'name', e.value)
                    ).classes('w-full')
                    ui.textarea(
                        'Description',
                        value=project.description,
                        on_change=lambda e: setattr(project, 'description', e.value)
                    ).classes('w-full')
                
                # Search Settings
                with ui.tab_panel('Search'):
                    with ui.expansion('Search Terms', value=True):
                        ui.textarea(
                            'One term per line',
                            value='\n'.join(selected_project['search_terms']),
                            on_change=lambda e: selected_project.update({
                                'search_terms': [t.strip() for t in e.value.split('\n') if t.strip()]
                            })
                        ).classes('w-full')
                    
                    with ui.expansion('Excluded Domains', value=True):
                        ui.textarea(
                            'One domain per line',
                            value='\n'.join(selected_project['excluded_domains']),
                            on_change=lambda e: selected_project.update({
                                'excluded_domains': [d.strip() for d in e.value.split('\n') if d.strip()]
                            })
                        ).classes('w-full')
                    
                    with ui.expansion('Target Industries', value=True):
                        ui.textarea(
                            'One industry per line',
                            value='\n'.join(selected_project['target_industries']),
                            on_change=lambda e: selected_project.update({
                                'target_industries': [i.strip() for i in e.value.split('\n') if i.strip()]
                            })
                        ).classes('w-full')
                
                # Automation Settings
                with ui.tab_panel('Automation'):
                    ui.number(
                        'Daily Lead Limit',
                        value=selected_project['daily_limits']['leads'],
                        on_change=lambda e: selected_project['daily_limits'].update({'leads': e.value})
                    )
                    ui.number(
                        'Daily Email Limit',
                        value=selected_project['daily_limits']['emails'],
                        on_change=lambda e: selected_project['daily_limits'].update({'emails': e.value})
                    )
                    ui.input(
                        'Schedule (cron)',
                        value=selected_project['automation']['schedule'],
                        on_change=lambda e: selected_project['automation'].update({'schedule': e.value})
                    ).classes('w-full')
                    ui.number(
                        'Auto-pause Bounce Rate',
                        value=selected_project['automation']['auto_pause_threshold'],
                        min=0,
                        max=1,
                        step=0.01,
                        on_change=lambda e: selected_project['automation'].update({'auto_pause_threshold': e.value})
                    )
                
                # Email Templates
                with ui.tab_panel('Templates'):
                    for i, template in enumerate(selected_project['email_templates']):
                        with ui.card().classes('w-full q-ma-sm'):
                            ui.input(
                                'Subject',
                                value=template['subject'],
                                on_change=lambda e, idx=i: selected_project['email_templates'][idx].update({'subject': e.value})
                            ).classes('w-full')
                            ui.textarea(
                                'Body',
                                value=template['body'],
                                on_change=lambda e, idx=i: selected_project['email_templates'][idx].update({'body': e.value})
                            ).classes('w-full')
                            ui.button(
                                'Delete',
                                on_click=lambda idx=i: selected_project['email_templates'].pop(idx)
                            ).props('flat negative')
                    
                    ui.button('Add Template', on_click=lambda: selected_project['email_templates'].append({
                        'subject': '',
                        'body': ''
                    })).props('outline')
            
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                ui.button('Save', on_click=lambda: save_settings(project.id, dialog)).props('primary')
    
    async def save_settings(project_id: int, dialog):
        """Save project settings"""
        await update_project(project_id)
        dialog.close()
        await load_projects()
    
    # UI Layout
    with ui.row().classes('w-full items-center justify-between q-ma-md'):
        ui.label('Projects').classes('text-h4')
        ui.button('New Project', on_click=create_project).props('primary')
    
    # Projects Container
    projects_container = ui.column().classes('w-full')
    
    # Load initial projects
    await load_projects() 