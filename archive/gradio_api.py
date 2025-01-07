import gradio as gr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from search_worker import WorkerTask
import json
import os
import threading

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

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
    
    with gr.Tab("Process Status"):
        process_id_input = gr.Number(label="Process ID")
        status_btn = gr.Button("Get Status")
        status_output = gr.JSON(label="Process Status")
        
        status_btn.click(
            get_task_status,
            inputs=process_id_input,
            outputs=status_output
        )
    
    with gr.Tab("Bulk Send"):
        template_id_input = gr.Number(label="Template ID")
        bulk_from_email = gr.Textbox(label="From Email")
        bulk_reply_to = gr.Textbox(label="Reply To")
        leads_json_input = gr.Textbox(label="Leads JSON")
        settings_json_input = gr.Textbox(label="Settings JSON")
        
        bulk_send_btn = gr.Button("Start Bulk Send")
        bulk_send_output = gr.JSON(label="Result")
        
        bulk_send_btn.click(
            start_bulk_send,
            inputs=[
                template_id_input, bulk_from_email,
                bulk_reply_to, leads_json_input,
                settings_json_input
            ],
            outputs=bulk_send_output
        )
    
    with gr.Tab("AI Automation"):
        project_id_input = gr.Number(label="Project ID")
        ai_settings_input = gr.Textbox(label="Settings JSON")
        
        ai_start_btn = gr.Button("Start AI Automation")
        ai_output = gr.JSON(label="Result")
        
        ai_start_btn.click(
            start_ai_automation,
            inputs=[project_id_input, ai_settings_input],
            outputs=ai_output
        )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860) 