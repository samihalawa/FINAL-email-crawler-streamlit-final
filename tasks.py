from celery_config import celery_app
from streamlit_app import (
    manual_search, db_session, get_knowledge_base_info,
    get_active_project_id, generate_optimized_search_terms,
    save_lead, bulk_send_emails, EmailTemplate
)

@celery_app.task(bind=True)
def background_search_task(self, search_terms, num_results, search_config):
    """Background task for performing searches"""
    results = []
    with db_session() as session:
        for term in search_terms:
            term_results = manual_search(
                session=session,
                search_terms=[term],
                num_results=num_results,
                **search_config
            )
            results.extend(term_results['results'])
            
            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': len(results),
                    'term': term,
                    'latest_results': term_results['results']
                }
            )
    return {'status': 'completed', 'results': results}

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