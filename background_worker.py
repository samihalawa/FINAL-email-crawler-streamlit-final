import os, json, logging, time
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit_app import manual_search, Base, SessionLocal
import multiprocessing
import queue
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='background_worker.log'
)

class SearchJob:
    def __init__(self, terms, num_results, settings):
        self.terms = terms
        self.num_results = num_results
        self.settings = settings
        self.status = "pending"
        self.results = []
        self.job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"
        self.created_at = datetime.now()

class BackgroundWorker:
    def __init__(self):
        self.jobs = {}
        self.active = True
        self.job_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.worker_process = None

    def start(self):
        self.worker_process = multiprocessing.Process(target=self._process_jobs)
        self.worker_process.start()
        logging.info("Background worker started")

    def stop(self):
        self.active = False
        if self.worker_process:
            self.worker_process.terminate()
            self.worker_process.join()
        logging.info("Background worker stopped")

    def add_job(self, terms, num_results, settings):
        job = SearchJob(terms, num_results, settings)
        self.jobs[job.job_id] = job
        self.job_queue.put(job)
        logging.info(f"Added new job: {job.job_id}")
        return job.job_id

    def get_job_status(self, job_id):
        job = self.jobs.get(job_id)
        if not job:
            return None
        return {
            "status": job.status,
            "results": job.results,
            "created_at": job.created_at.isoformat()
        }

    def _process_jobs(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        while self.active:
            try:
                # Try to get a job from the queue
                try:
                    job = self.job_queue.get(timeout=1)
                except queue.Empty:
                    continue

                logging.info(f"Processing job: {job.job_id}")
                job.status = "running"

                # Create a new database session
                session = SessionLocal()
                try:
                    # Run the manual search
                    results = manual_search(
                        session=session,
                        terms=job.terms,
                        num_results=job.num_results,
                        ignore_previously_fetched=job.settings.get('ignore_previously_fetched', True),
                        optimize_english=job.settings.get('optimize_english', False),
                        optimize_spanish=job.settings.get('optimize_spanish', False),
                        shuffle_keywords_option=job.settings.get('shuffle_keywords_option', False),
                        language=job.settings.get('language', 'ES'),
                        enable_email_sending=job.settings.get('enable_email_sending', True),
                        from_email=job.settings.get('from_email'),
                        reply_to=job.settings.get('reply_to'),
                        email_template=job.settings.get('email_template')
                    )
                    
                    job.results = results
                    job.status = "completed"
                    logging.info(f"Job completed: {job.job_id}")
                    
                except Exception as e:
                    job.status = "failed"
                    job.results = {"error": str(e)}
                    logging.error(f"Job failed: {job.job_id} - {str(e)}")
                finally:
                    session.close()

            except Exception as e:
                logging.error(f"Worker error: {str(e)}")
                time.sleep(1)

# Global worker instance
worker = BackgroundWorker()

def start_worker():
    worker.start()

def stop_worker():
    worker.stop()

if __name__ == "__main__":
    start_worker() 