import threading
import time
import logging
from datetime import datetime
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchWorkerThread(threading.Thread):
    def __init__(self, engine, SessionLocal):
        super().__init__()
        self.engine = engine
        self.SessionLocal = SessionLocal
        self.daemon = True  # Thread will exit when main program exits
        self._stop_event = threading.Event()
        
    def stop(self):
        self._stop_event.set()
        
    def stopped(self):
        return self._stop_event.is_set()
        
    @contextmanager
    def session_scope(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
            
    def run(self):
        logger.info("Starting search worker thread...")
        while not self.stopped():
            try:
                with self.session_scope() as session:
                    # Get pending search processes
                    pending_processes = session.query(SearchProcess).filter(
                        SearchProcess.status == 'pending'
                    ).all()
                    
                    for process in pending_processes:
                        if self.stopped():
                            break
                            
                        try:
                            logger.info(f"Processing search process {process.id}")
                            process.status = 'running'
                            process.started_at = datetime.utcnow()
                            session.commit()
                            
                            # Execute the search
                            self.execute_search(session, process)
                            
                            process.status = 'completed'
                            process.completed_at = datetime.utcnow()
                            session.commit()
                            
                        except Exception as e:
                            logger.error(f"Error processing search {process.id}: {str(e)}")
                            process.status = 'failed'
                            process.error_message = str(e)
                            process.completed_at = datetime.utcnow()
                            session.commit()
                            
            except Exception as e:
                logger.error(f"Worker thread error: {str(e)}")
                
            time.sleep(5)  # Check for new tasks every 5 seconds
            
    def execute_search(self, session, process):
        from streamlit_app_BACKGROUND_PROCESS_ADDED import manual_search
        
        results = manual_search(
            session,
            process.search_terms,
            process.num_results,
            process.settings.get('ignore_previously_fetched', True),
            process.settings.get('optimize_english', False),
            process.settings.get('optimize_spanish', False),
            process.settings.get('shuffle_keywords_option', True),
            process.settings.get('language', 'ES'),
            process.settings.get('enable_email_sending', True),
            None,  # log_container not needed for background
            process.settings.get('from_email'),
            process.settings.get('reply_to'),
            process.settings.get('email_template'),
            process.id
        )
        
        process.total_leads_found = results.get('total_leads', 0) 