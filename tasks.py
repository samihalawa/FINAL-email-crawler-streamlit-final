from celery_config import celery_app
from database import db_session
from models import SearchTerm
from tenacity import retry, stop_after_attempt
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3)
def background_search_task(self, search_terms, num_results, search_config):
    try:
        results = []
        with db_session() as session:
            total_terms = len(search_terms)
            for idx, term in enumerate(search_terms):
                try:
                    term_results = manual_search(
                        session=session,
                        search_terms=[term],
                        num_results=num_results,
                        **search_config
                    )
                    results.extend(term_results['results'])
                    
                    # Update progress
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'current': idx + 1,
                            'total': total_terms,
                            'term': term,
                            'latest_results': term_results['results']
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing term {term}: {str(e)}")
                    continue
                    
        return {'status': 'completed', 'results': results}
    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        self.retry(exc=e)

@celery_app.task
def automation_loop_task(search_config):
    """Background task for continuous automation"""
    with db_session() as session:
        kb_info = get_knowledge_base_info(session, get_active_project_id())
        if not kb_info:
            return {'status': 'error', 'message': 'Knowledge Base not found'}
            
        base_terms = [term.term for term in session.query(SearchTerm)
                     .filter_by(project_id=get_active_project_id()).all()]
        optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
        
        results = []
        for term in optimized_terms:
            search_results = manual_search(session, [term], 10, **search_config)
            results.extend(search_results['results'])
            
        return {
            'status': 'completed',
            'results': results,
            'terms_processed': len(optimized_terms)
        } 