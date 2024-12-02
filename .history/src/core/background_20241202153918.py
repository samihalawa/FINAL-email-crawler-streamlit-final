import asyncio
from typing import Dict, Any, List, Set
import aiojobs
from datetime import datetime
import json

from core.database import get_db
from core.logging import app_logger
from services.search import SearchService
from services.email import EmailService
from services.ai import AIService

search_service = SearchService()
email_service = EmailService()
ai_service = AIService()

class TaskManager:
    """Optimized background task manager with parallel processing"""
    
    def __init__(self):
        self.scheduler = None
        self.active_tasks: Dict[int, Set[str]] = {}  # project_id -> set of task_ids
        self.task_queues: Dict[int, asyncio.Queue] = {}  # project_id -> task queue
        self.workers: Dict[int, List[asyncio.Task]] = {}  # project_id -> list of worker tasks
        
    async def start(self):
        """Initialize the task manager"""
        self.scheduler = await aiojobs.create_scheduler(
            close_timeout=10.0,
            limit=100,  # Max concurrent jobs
            pending_limit=200  # Max pending jobs
        )
        
        # Start monitoring task
        asyncio.create_task(self.monitor_tasks())
    
    async def stop(self):
        """Shutdown the task manager"""
        if self.scheduler:
            await self.scheduler.close()
            
        # Cancel all workers
        for workers in self.workers.values():
            for worker in workers:
                if not worker.done():
                    worker.cancel()
    
    async def start_project_tasks(self, project_id: int):
        """Start background tasks for a project"""
        if project_id in self.active_tasks:
            return
        
        self.active_tasks[project_id] = set()
        self.task_queues[project_id] = asyncio.Queue()
        
        # Start workers for this project
        self.workers[project_id] = [
            asyncio.create_task(self.task_worker(project_id))
            for _ in range(5)  # Number of workers per project
        ]
        
        # Schedule initial tasks
        await self.schedule_project_tasks(project_id)
    
    async def stop_project_tasks(self, project_id: int):
        """Stop background tasks for a project"""
        if project_id not in self.active_tasks:
            return
        
        # Cancel workers
        if project_id in self.workers:
            for worker in self.workers[project_id]:
                if not worker.done():
                    worker.cancel()
            del self.workers[project_id]
        
        # Clear task queue
        if project_id in self.task_queues:
            while not self.task_queues[project_id].empty():
                try:
                    self.task_queues[project_id].get_nowait()
                except:
                    pass
            del self.task_queues[project_id]
        
        # Clear active tasks
        del self.active_tasks[project_id]
    
    async def schedule_project_tasks(self, project_id: int):
        """Schedule tasks for a project"""
        async with get_db() as session:
            project = await session.get_project(project_id)
            if not project or not project.settings['automation']['enabled']:
                return
            
            # Schedule search tasks
            for term in project.settings['search_terms']:
                await self.task_queues[project_id].put({
                    'type': 'search',
                    'data': {
                        'term': term,
                        'excluded_domains': project.settings['excluded_domains']
                    }
                })
            
            # Schedule email tasks for existing leads
            results = await session.execute(
                "SELECT * FROM leads WHERE project_id = :project_id AND status = 'pending'",
                {'project_id': project_id}
            )
            leads = results.scalars().all()
            
            for lead in leads:
                await self.task_queues[project_id].put({
                    'type': 'email',
                    'data': {
                        'lead_id': lead.id,
                        'templates': project.settings['email_templates']
                    }
                })
    
    async def task_worker(self, project_id: int):
        """Worker process for executing tasks"""
        while True:
            try:
                # Get task from queue
                task = await self.task_queues[project_id].get()
                
                # Execute task based on type
                if task['type'] == 'search':
                    await self.execute_search_task(project_id, task['data'])
                elif task['type'] == 'email':
                    await self.execute_email_task(project_id, task['data'])
                
                self.task_queues[project_id].task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                app_logger.error(f"Task error in project {project_id}", exc_info=e)
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def execute_search_task(self, project_id: int, data: Dict[str, Any]):
        """Execute a search task"""
        try:
            # Perform search
            results = await search_service.search(
                query=data['term'],
                excluded_domains=data['excluded_domains']
            )
            
            # Process results
            async with get_db() as session:
                for result in results:
                    # Check if lead already exists
                    existing = await session.execute(
                        "SELECT id FROM leads WHERE email = :email AND project_id = :project_id",
                        {'email': result['email'], 'project_id': project_id}
                    )
                    if existing.scalar():
                        continue
                    
                    # Create new lead
                    lead = Lead(
                        project_id=project_id,
                        email=result['email'],
                        first_name=result['first_name'],
                        last_name=result['last_name'],
                        company=result['company'],
                        position=result['position'],
                        linkedin_url=result['linkedin_url'],
                        website=result['website'],
                        source=data['term'],
                        status='pending',
                        created_at=datetime.utcnow()
                    )
                    session.add(lead)
                
                await session.commit()
            
        except Exception as e:
            app_logger.error(f"Search task error in project {project_id}", exc_info=e)
    
    async def execute_email_task(self, project_id: int, data: Dict[str, Any]):
        """Execute an email task"""
        try:
            async with get_db() as session:
                lead = await session.get_lead(data['lead_id'])
                if not lead or lead.status != 'pending':
                    return
                
                # Select best template using AI
                template = await ai_service.select_best_template(
                    lead=lead,
                    templates=data['templates']
                )
                
                # Personalize template
                subject, body = await ai_service.personalize_template(
                    template=template,
                    lead=lead
                )
                
                # Send email
                success = await email_service.send_email(
                    to_email=lead.email,
                    subject=subject,
                    body_html=body,
                    lead_id=lead.id,
                    project_id=project_id
                )
                
                if success:
                    lead.status = 'contacted'
                    lead.last_contacted = datetime.utcnow()
                    await session.commit()
                
        except Exception as e:
            app_logger.error(f"Email task error in project {project_id}", exc_info=e)
    
    async def monitor_tasks(self):
        """Monitor task execution and reschedule as needed"""
        while True:
            try:
                for project_id in list(self.active_tasks.keys()):
                    # Check project status
                    async with get_db() as session:
                        project = await session.get_project(project_id)
                        if not project or not project.settings['automation']['enabled']:
                            await self.stop_project_tasks(project_id)
                            continue
                        
                        # Check if we need to schedule more tasks
                        if self.task_queues[project_id].empty():
                            await self.schedule_project_tasks(project_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                app_logger.error("Task monitor error", exc_info=e)
                await asyncio.sleep(5)

# Global task manager instance
task_manager = TaskManager()