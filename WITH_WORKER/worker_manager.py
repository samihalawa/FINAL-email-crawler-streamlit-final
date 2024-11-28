import os
import time
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
import multiprocessing
from worker import process_task

# Database setup
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def process_pending_tasks():
    """Process pending tasks in the background"""
    while True:
        try:
            with SessionLocal() as session:
                # Get pending tasks or tasks that have been running for too long
                tasks = session.query(WorkerTask).filter(
                    or_(
                        WorkerTask.status == 'pending',
                        # Tasks running for more than 1 hour are considered stuck
                        (WorkerTask.status == 'running') & 
                        (WorkerTask.started_at < datetime.utcnow() - timedelta(hours=1))
                    )
                ).all()
                
                for task in tasks:
                    # Start a new process for each task
                    process = multiprocessing.Process(target=process_task, args=(task.id,))
                    process.start()
            
            # Sleep for a bit before checking for new tasks
            time.sleep(5)
            
        except Exception as e:
            logging.error(f"Error in worker manager: {str(e)}")
            time.sleep(30)

if __name__ == '__main__':
    # Start the worker manager
    process_pending_tasks() 