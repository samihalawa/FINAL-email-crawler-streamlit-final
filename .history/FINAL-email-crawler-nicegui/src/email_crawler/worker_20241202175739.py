import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from .models import SearchJob, SearchTerm, Campaign
from .utils import GoogleSearcher, LeadProcessor, AIProcessor
import logging

logger = logging.getLogger(__name__)

class SearchWorker:
    def __init__(self, session: Session, ai_processor: Optional[AIProcessor] = None):
        self.session = session
        self.searcher = GoogleSearcher()
        self.ai_processor = ai_processor
        self.active_jobs: Dict[str, Dict] = {}

    async def create_job(self, terms: List[str], campaign_id: Optional[int] = None) -> SearchJob:
        """Create a new search job"""
        job = SearchJob(
            status="pending",
            terms=terms,
            created_at=datetime.utcnow()
        )
        self.session.add(job)
        self.session.commit()
        
        # Store in memory for progress tracking
        self.active_jobs[str(job.id)] = {
            'status': 'pending',
            'progress': 0,
            'total_terms': len(terms),
            'processed_terms': 0
        }
        
        return job

    async def run_job(self, job_id: int):
        """Execute a search job"""
        try:
            job = self.session.query(SearchJob).get(job_id)
            if not job:
                return
            
            job.status = "running"
            self.session.commit()
            
            results = []
            for i, term in enumerate(job.terms):
                # Optimize term if AI processor available
                if self.ai_processor:
                    optimized_terms = await self.ai_processor.optimize_search_terms([term], "")
                    term = optimized_terms[0] if optimized_terms else term
                
                # Execute search
                search_results = await self.searcher.search(term)
                
                # Process and save results
                search_term = SearchTerm(term=term)
                self.session.add(search_term)
                self.session.flush()
                
                await LeadProcessor.process_search_results(
                    self.session, 
                    search_results, 
                    search_term.id
                )
                
                results.extend(search_results)
                
                # Update progress
                self.active_jobs[str(job_id)]['processed_terms'] = i + 1
                self.active_jobs[str(job_id)]['progress'] = ((i + 1) / len(job.terms)) * 100
            
            # Update job status
            job.status = "completed"
            job.results = results
            job.completed_at = datetime.utcnow()
            self.session.commit()
            
        except Exception as e:
            logger.error(f"Error in search job {job_id}: {str(e)}")
            job.status = "failed"
            job.error = str(e)
            self.session.commit()

class AutomationWorker:
    def __init__(self, session: Session, search_worker: SearchWorker):
        self.session = session
        self.search_worker = search_worker
        self.running = False

    async def start(self):
        """Start the automation loop"""
        self.running = True
        while self.running:
            try:
                # Get active campaigns with auto_send enabled
                campaigns = self.session.query(Campaign).filter_by(auto_send=True).all()
                
                for campaign in campaigns:
                    # Get pending search terms
                    terms = [
                        term.term for term in 
                        self.session.query(SearchTerm)
                        .filter_by(campaign_id=campaign.id, is_active=True)
                        .filter(SearchTerm.last_used.is_(None))
                        .all()
                    ]
                    
                    if terms:
                        # Create and run search job
                        job = await self.search_worker.create_job(terms, campaign.id)
                        await self.search_worker.run_job(job.id)
                
                # Wait before next iteration
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in automation loop: {str(e)}")
                await asyncio.sleep(300)  # 5 minutes on error

    def stop(self):
        """Stop the automation loop"""
        self.running = False
