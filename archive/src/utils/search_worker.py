import os, json, logging, requests, random, uuid, time, gradio as gr
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from googlesearch import search as google_search
import threading

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class WorkerTask(Base):
    __tablename__ = 'worker_tasks'
    
    id = Column(Integer, primary_key=True)
    task_type = Column(String)  # 'search', 'bulk_send', 'ai_automation'
    status = Column(String)     # 'pending', 'running', 'completed', 'failed'
    params = Column(JSON)       # Task parameters
    results = Column(JSON)      # Task results
    logs = Column(JSON)         # Task logs
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error = Column(String)

Base.metadata.create_all(engine)

# Worker Functions
def process_task(task_id: int):
    """Main worker function that processes any type of task"""
    with SessionLocal() as session:
        task = session.query(WorkerTask).get(task_id)
        if not task:
            logging.error(f"Task {task_id} not found")
            return
        
        try:
            task.status = 'running'
            task.started_at = datetime.utcnow()
            session.commit()
            
            if task.task_type == 'search':
                results = handle_search_task(task, session)
            elif task.task_type == 'bulk_send':
                results = handle_bulk_send_task(task, session)
            elif task.task_type == 'ai_automation':
                results = handle_ai_automation_task(task, session)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.status = 'completed'
            task.results = results
            task.completed_at = datetime.utcnow()
            
        except Exception as e:
            task.status = 'failed'
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            logging.error(f"Task {task_id} failed: {str(e)}")
        
        finally:
            session.commit()

def handle_search_task(task, session):
    """Handle manual search task"""
    params = task.params
    results = []
    total_leads = 0
    
    for term in params['search_terms']:
        update_task_log(session, task.id, f"Processing term: {term}")
        try:
            # Perform Google search
            for url in google_search(term, params['num_results'], lang=params['language']):
                # Process URL and extract leads
                # ... (your existing search logic)
                pass
                
        except Exception as e:
            update_task_log(session, task.id, f"Error processing term {term}: {str(e)}", 'error')
    
    return {
        'total_leads': total_leads,
        'results': results
    }

def handle_bulk_send_task(task, session):
    """Handle bulk email sending task"""
    params = task.params
    results = []
    sent_count = 0
    
    # Your bulk send logic here
    
    return {
        'sent_count': sent_count,
        'results': results
    }

def handle_ai_automation_task(task, session):
    """Handle AI automation task"""
    params = task.params
    
    # Your AI automation logic here
    
    return {
        'automation_results': 'completed'
    }

def update_task_log(session, task_id, message, level='info'):
    """Update task logs"""
    task = session.query(WorkerTask).get(task_id)
    if task:
        if not task.logs:
            task.logs = []
        task.logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message
        })
        session.commit()

# Worker process
def run_worker():
    """Main worker process that continuously checks for new tasks"""
    logging.info("Starting worker process")
    
    while True:
        try:
            with SessionLocal() as session:
                pending_task = session.query(WorkerTask).filter_by(
                    status='pending'
                ).order_by(WorkerTask.created_at).first()
                
                if pending_task:
                    logging.info(f"Processing task {pending_task.id}")
                    process_task(pending_task.id)
                else:
                    time.sleep(5)
                    
        except Exception as e:
            logging.error(f"Worker error: {str(e)}")
            time.sleep(5)

# API Functions
def create_task(task_type: str, params: dict) -> int:
    """Create a new task and return its ID"""
    with SessionLocal() as session:
        task = WorkerTask(
            task_type=task_type,
            status='pending',
            params=params
        )
        session.add(task)
        session.commit()
        return task.id

def get_task_status(task_id: int) -> str:
    """Get the status of any task"""
    with SessionLocal() as session:
        task = session.query(WorkerTask).get(task_id)
        if not task:
            return json.dumps({"error": "Task not found"})
        
        return json.dumps({
            "task_id": task.id,
            "type": task.task_type,
            "status": task.status,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "logs": task.logs,
            "results": task.results,
            "error": task.error
        })

# Gradio API Interface
def start_search_process(
    search_terms: str,
    num_results: int = 10,
    ignore_previously_fetched: bool = True,
    optimize_english: bool = False,
    optimize_spanish: bool = False,
    shuffle_keywords: bool = True,
    language: str = "ES",
    enable_email_sending: bool = False,
    from_email: str = None,
    reply_to: str = None,
    email_template: str = None
) -> str:
    """Start a new search task"""
    params = {
        "search_terms": [t.strip() for t in search_terms.split(',')],
        "num_results": num_results,
        "ignore_previously_fetched": ignore_previously_fetched,
        "optimize_english": optimize_english,
        "optimize_spanish": optimize_spanish,
        "shuffle_keywords": shuffle_keywords,
        "language": language,
        "enable_email_sending": enable_email_sending,
        "from_email": from_email,
        "reply_to": reply_to,
        "email_template": email_template
    }
    
    task_id = create_task('search', params)
    
    return json.dumps({
        "task_id": task_id,
        "status": "pending",
        "message": "Search task created successfully"
    })

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# AutoClient.ai API")
    
    with gr.Tab("Manual Search"):
        with gr.Row():
            search_terms_input = gr.Textbox(label="Search Terms (comma-separated)")
            num_results_input = gr.Number(label="Number of Results", value=10)
        
        with gr.Row():
            ignore_fetched = gr.Checkbox(label="Ignore Previously Fetched", value=True)
            optimize_english = gr.Checkbox(label="Optimize English")
            optimize_spanish = gr.Checkbox(label="Optimize Spanish")
            shuffle_keywords = gr.Checkbox(label="Shuffle Keywords", value=True)
            
        with gr.Row():
            language = gr.Dropdown(choices=["ES", "EN"], label="Language", value="ES")
            enable_email = gr.Checkbox(label="Enable Email Sending")
            
        with gr.Row():
            from_email_input = gr.Textbox(label="From Email")
            reply_to_input = gr.Textbox(label="Reply To")
            template_input = gr.Textbox(label="Email Template ID")
            
        start_btn = gr.Button("Start Search")
        output = gr.JSON(label="Result")
        
        start_btn.click(
            start_search_process,
            inputs=[
                search_terms_input, num_results_input,
                ignore_fetched, optimize_english, optimize_spanish,
                shuffle_keywords, language, enable_email,
                from_email_input, reply_to_input, template_input
            ],
            outputs=output
        )
    
    with gr.Tab("Task Status"):
        task_id_input = gr.Number(label="Task ID")
        status_btn = gr.Button("Get Status")
        status_output = gr.JSON(label="Task Status")
        
        status_btn.click(
            get_task_status,
            inputs=task_id_input,
            outputs=status_output
        )

if __name__ == "__main__":
    # Start worker thread
    worker_thread = threading.Thread(target=run_worker)
    worker_thread.daemon = True
    worker_thread.start()
    
    # Start Gradio app
    app.launch(server_name="0.0.0.0", server_port=7860) 