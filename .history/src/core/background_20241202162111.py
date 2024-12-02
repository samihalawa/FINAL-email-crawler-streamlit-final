import asyncio
from typing import Dict, Any, List, Set, Optional
import aiojobs
from datetime import datetime, timedelta
import json
import random
from sqlalchemy import select, and_, or_, func

from core.database import (
    get_db, Project, Campaign, SearchTerm, SearchTermGroup, Lead, 
    EmailTemplate, EmailCampaign, KnowledgeBase, AutomationTask,
    TaskStatus, LeadStatus
)
from core.logging import app_logger
from services.search import SearchService
from services.email import EmailService
from services.ai import AIService

search_service = SearchService()
email_service = EmailService()
ai_service = AIService()

class TaskPrioritizer:
    """Handles task prioritization and scheduling"""
    
    @staticmethod
    async def calculate_task_priority(task: AutomationTask, session) -> int:
        base_priority = task.priority
        
        # Age factor: older tasks get higher priority
        age_hours = (datetime.utcnow() - task.created_at).total_seconds() / 3600
        age_factor = min(int(age_hours / 24), 5)  # Cap at 5
        
        # Retry factor: more retries = lower priority
        retry_factor = max(5 - task.retry_count, 0)
        
        # Project factor: based on project settings and performance
        project = await session.get(Project, task.project_id)
        if project and project.settings['automation']['enabled']:
            project_factor = 2
        else:
            project_factor = 0
        
        return base_priority + age_factor + retry_factor + project_factor

    @staticmethod
    async def get_next_tasks(session, limit: int = 10) -> List[AutomationTask]:
        """Get next batch of tasks to execute"""
        query = select(AutomationTask).where(
            and_(
                AutomationTask.status.in_([TaskStatus.PENDING, TaskStatus.FAILED]),
                or_(
                    AutomationTask.next_retry.is_(None),
                    AutomationTask.next_retry <= datetime.utcnow()
                )
            )
        ).order_by(AutomationTask.priority.desc()).limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()

class ResourceManager:
    """Manages system resources and enforces limits"""
    
    def __init__(self):
        self.active_tasks: Dict[str, int] = {}
        self.rate_limits: Dict[int, Dict] = {}  # project_id -> limits
    
    async def can_start_task(self, project_id: int, task_type: str) -> bool:
        """Check if a new task can be started"""
        if project_id not in self.rate_limits:
            async with get_db() as session:
                project = await session.get(Project, project_id)
                if not project:
                    return False
                self.rate_limits[project_id] = project.settings
        
        limits = self.rate_limits[project_id]
        current = self.active_tasks.get(f"{project_id}:{task_type}", 0)
        
        if task_type == "email":
            return current < limits['email']['daily_limit']
        elif task_type == "search":
            return current < limits['automation']['max_concurrent_tasks']
        
        return True
    
    def start_task(self, project_id: int, task_type: str):
        """Record task start"""
        key = f"{project_id}:{task_type}"
        self.active_tasks[key] = self.active_tasks.get(key, 0) + 1
    
    def end_task(self, project_id: int, task_type: str):
        """Record task end"""
        key = f"{project_id}:{task_type}"
        if key in self.active_tasks and self.active_tasks[key] > 0:
            self.active_tasks[key] -= 1

class TaskManager:
    """Enhanced background task manager with autonomous capabilities"""
    
    def __init__(self):
        self.scheduler = None
        self.resource_manager = ResourceManager()
        self.prioritizer = TaskPrioritizer()
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Initialize the task manager"""
        self.scheduler = await aiojobs.create_scheduler(
            close_timeout=10.0,
            limit=100,
            pending_limit=200
        )
        self.is_running = True
        
        # Start core workers
        self.worker_tasks = [
            asyncio.create_task(self.task_processor()),
            asyncio.create_task(self.maintenance_worker()),
            asyncio.create_task(self.analytics_worker())
        ]
    
    async def stop(self):
        """Shutdown the task manager"""
        self.is_running = False
        
        if self.scheduler:
            await self.scheduler.close()
        
        # Cancel all workers
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
    
    async def task_processor(self):
        """Main task processing loop"""
        while self.is_running:
            try:
                async with get_db() as session:
                    # Get next batch of tasks
                    tasks = await self.prioritizer.get_next_tasks(session)
                    
                    for task in tasks:
                        if await self.resource_manager.can_start_task(task.project_id, task.task_type):
                            # Update task status
                            task.status = TaskStatus.RUNNING
                            task.started_at = datetime.utcnow()
                            await session.commit()
                            
                            # Execute task
                            try:
                                self.resource_manager.start_task(task.project_id, task.task_type)
                                result = await self.execute_task(task)
                                
                                # Update task completion
                                task.status = TaskStatus.COMPLETED
                                task.completed_at = datetime.utcnow()
                                task.result = result
                                await session.commit()
                                
                            except Exception as e:
                                app_logger.error(f"Task execution error: {str(e)}", exc_info=e)
                                task.status = TaskStatus.FAILED
                                task.error = str(e)
                                task.retry_count += 1
                                task.next_retry = datetime.utcnow() + timedelta(
                                    minutes=min(task.retry_count * 5, 60)
                                )
                                await session.commit()
                            
                            finally:
                                self.resource_manager.end_task(task.project_id, task.task_type)
                
                await asyncio.sleep(1)  # Prevent tight loop
                
            except Exception as e:
                app_logger.error(f"Task processor error: {str(e)}", exc_info=e)
                await asyncio.sleep(5)  # Longer delay on error
    
    async def maintenance_worker(self):
        """Handles system maintenance tasks"""
        while self.is_running:
            try:
                async with get_db() as session:
                    # Clean up stale tasks
                    stale_time = datetime.utcnow() - timedelta(hours=1)
                    stale_tasks = await session.execute(
                        select(AutomationTask).where(
                            and_(
                                AutomationTask.status == TaskStatus.RUNNING,
                                AutomationTask.started_at <= stale_time
                            )
                        )
                    )
                    
                    for task in stale_tasks.scalars():
                        task.status = TaskStatus.FAILED
                        task.error = "Task timed out"
                        task.retry_count += 1
                        task.next_retry = datetime.utcnow() + timedelta(minutes=5)
                    
                    await session.commit()
                    
                    # Update rate limits cache
                    projects = await session.execute(select(Project))
                    for project in projects.scalars():
                        self.resource_manager.rate_limits[project.id] = project.settings
                
                await asyncio.sleep(60)  # Run maintenance every minute
                
            except Exception as e:
                app_logger.error(f"Maintenance worker error: {str(e)}", exc_info=e)
                await asyncio.sleep(5)
    
    async def analytics_worker(self):
        """Handles analytics and optimization tasks"""
        while self.is_running:
            try:
                async with get_db() as session:
                    # Update campaign statistics
                    campaigns = await session.execute(
                        select(Campaign).where(Campaign.status == 'active')
                    )
                    
                    for campaign in campaigns.scalars():
                        # Calculate lead statistics
                        lead_stats = await session.execute(
                            select(
                                func.count(Lead.id).label('total'),
                                func.count(Lead.id).filter(Lead.status == LeadStatus.QUALIFIED).label('qualified'),
                                func.avg(Lead.quality_score).label('avg_quality')
                            ).where(Lead.campaign_id == campaign.id)
                        )
                        stats = lead_stats.first()
                        
                        # Calculate email statistics
                        email_stats = await session.execute(
                            select(
                                func.count(EmailCampaign.id).label('sent'),
                                func.count(EmailCampaign.id).filter(EmailCampaign.opened_at.isnot(None)).label('opened'),
                                func.count(EmailCampaign.id).filter(EmailCampaign.replied_at.isnot(None)).label('replied')
                            ).where(EmailCampaign.lead_id.in_(
                                select(Lead.id).where(Lead.campaign_id == campaign.id)
                            ))
                        )
                        email_stats = email_stats.first()
                        
                        # Update campaign stats
                        campaign.stats.update({
                            'total_leads': stats.total,
                            'qualified_leads': stats.qualified,
                            'emails_sent': email_stats.sent,
                            'emails_opened': email_stats.opened,
                            'emails_replied': email_stats.replied,
                            'conversion_rate': email_stats.replied / email_stats.sent if email_stats.sent > 0 else 0,
                            'avg_quality_score': stats.avg_quality or 0
                        })
                        
                        # Auto-optimize if enabled
                        if campaign.project.settings['automation']['ai_optimization']:
                            await self.optimize_campaign(campaign, session)
                    
                    await session.commit()
                
                await asyncio.sleep(300)  # Run analytics every 5 minutes
                
            except Exception as e:
                app_logger.error(f"Analytics worker error: {str(e)}", exc_info=e)
                await asyncio.sleep(5)
    
    async def execute_task(self, task: AutomationTask) -> Dict[str, Any]:
        """Execute a specific task"""
        if task.task_type == "search":
            return await self.execute_search_task(task)
        elif task.task_type == "email":
            return await self.execute_email_task(task)
        elif task.task_type == "optimize":
            return await self.execute_optimization_task(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def execute_search_task(self, task: AutomationTask) -> Dict[str, Any]:
        """Execute a search task"""
        params = task.params
        results = await search_service.perform_search(
            search_terms=params['search_terms'],
            excluded_domains=params.get('excluded_domains', []),
            max_results=params.get('max_results', 50)
        )
        
        async with get_db() as session:
            # Process and store results
            for result in results:
                lead = Lead(
                    email=result['email'],
                    name=result.get('name'),
                    company=result.get('company'),
                    position=result.get('position'),
                    source_url=result['url'],
                    campaign_id=params['campaign_id'],
                    quality_score=await ai_service.calculate_lead_quality(result)
                )
                session.add(lead)
            
            await session.commit()
        
        return {
            'total_results': len(results),
            'processed_at': datetime.utcnow().isoformat()
        }
    
    async def execute_email_task(self, task: AutomationTask) -> Dict[str, Any]:
        """Execute an email task"""
        params = task.params
        
        async with get_db() as session:
            lead = await session.get(Lead, params['lead_id'])
            template = await session.get(EmailTemplate, params['template_id'])
            
            if not lead or not template:
                raise ValueError("Lead or template not found")
            
            # Customize email content
            customized_content = await ai_service.customize_email(
                template.body_content,
                lead=lead,
                template_vars=template.variables
            )
            
            # Send email
            result = await email_service.send_email(
                to_email=lead.email,
                subject=template.subject,
                content=customized_content,
                tracking_enabled=True
            )
            
            # Record email campaign
            campaign = EmailCampaign(
                lead_id=lead.id,
                template_id=template.id,
                status='sent',
                customized_content=customized_content,
                sent_at=datetime.utcnow()
            )
            session.add(campaign)
            
            # Update lead status
            lead.status = LeadStatus.CONTACTED
            lead.last_contacted = datetime.utcnow()
            
            await session.commit()
        
        return {
            'email_sent': True,
            'sent_at': datetime.utcnow().isoformat(),
            'tracking_id': result['tracking_id']
        }
    
    async def optimize_campaign(self, campaign: Campaign, session) -> None:
        """Optimize campaign performance using AI"""
        # Analyze performance
        if campaign.stats['emails_sent'] > 100:  # Enough data for optimization
            # Get best performing templates
            templates = await session.execute(
                select(EmailTemplate)
                .where(EmailTemplate.campaign_id == campaign.id)
                .order_by(EmailTemplate.performance_stats['conversion_rate'].desc())
            )
            best_templates = templates.scalars().all()[:3]
            
            # Get best performing search terms
            terms = await session.execute(
                select(SearchTerm)
                .where(SearchTerm.campaign_id == campaign.id)
                .order_by(SearchTerm.success_rate.desc())
            )
            best_terms = terms.scalars().all()[:5]
            
            # Generate optimization suggestions
            suggestions = await ai_service.generate_optimization_suggestions(
                campaign=campaign,
                best_templates=best_templates,
                best_terms=best_terms
            )
            
            # Apply optimizations
            if suggestions.get('new_search_terms'):
                for term in suggestions['new_search_terms']:
                    new_term = SearchTerm(
                        term=term,
                        campaign_id=campaign.id,
                        priority=1
                    )
                    session.add(new_term)
            
            if suggestions.get('template_improvements'):
                for template_id, improvements in suggestions['template_improvements'].items():
                    template = await session.get(EmailTemplate, template_id)
                    if template:
                        template.body_content = improvements['content']
                        template.subject = improvements['subject']
            
            # Update campaign settings
            campaign.settings['optimization_history'] = campaign.settings.get('optimization_history', [])
            campaign.settings['optimization_history'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'changes': suggestions
            })
            
            await session.commit()

task_manager = TaskManager()