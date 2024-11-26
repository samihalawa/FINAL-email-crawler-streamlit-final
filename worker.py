import gradio as gr
import os, json, time, asyncio, torch
from datetime import datetime
from sqlalchemy import create_engine, func, distinct
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
import numpy as np

# Load environment variables
load_dotenv()

# Database connection
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

class AIAgent:
    """AI agent for task optimization and decision making"""
    def __init__(self, model_name: str = "Qwen/Qwen-7B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def load_model(self):
        """Lazy load model when needed"""
        if not self.model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            
    async def analyze_search_results(self, results: List[Dict]) -> Dict:
        """Analyze search results for quality and relevance"""
        await self.load_model()
        # Implementation for search results analysis
        return {
            "quality_score": 0.8,
            "relevance_score": 0.7,
            "recommendations": []
        }
        
    async def optimize_search_terms(self, terms: List[str], results: List[Dict]) -> List[str]:
        """Optimize search terms based on results"""
        await self.load_model()
        # Implementation for search term optimization
        return terms
        
    async def detect_email_quality(self, emails: List[str]) -> List[bool]:
        """Detect email quality and relevance"""
        await self.load_model()
        # Implementation for email quality detection
        return [True] * len(emails)

class TaskManager:
    """Manages different types of background tasks"""
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.ai_agent = AIAgent()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    async def create_task(self, task_type: str, params: Dict) -> str:
        """Create a new task"""
        task_id = f"{task_type}-{time.time()}"
        self.tasks[task_id] = {
            "type": task_type,
            "status": "pending",
            "params": params,
            "progress": 0,
            "results": None,
            "logs": [],
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        return task_id
        
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status and progress"""
        return self.tasks.get(task_id)
        
    async def process_task(self, task_id: str):
        """Process a task based on its type"""
        task = self.tasks[task_id]
        task["status"] = "running"
        
        try:
            if task["type"] == "lead_search":
                await self._process_lead_search(task)
            elif task["type"] == "email_campaign":
                await self._process_email_campaign(task)
            elif task["type"] == "lead_analysis":
                await self._process_lead_analysis(task)
            elif task["type"] == "database_optimization":
                await self._process_database_optimization(task)
                
            task["status"] = "completed"
        except Exception as e:
            task["status"] = "error"
            task["error"] = str(e)
            
        task["updated_at"] = datetime.now()
        
    async def _process_lead_search(self, task: Dict):
        """Process lead search task with AI optimization"""
        params = task["params"]
        with SessionLocal() as session:
            from streamlit_app import manual_search
            
            # Initial search
            results = []
            for idx, term in enumerate(params["search_terms"]):
                search_results = manual_search(
                    session=session,
                    terms=[term],
                    num_results=params["num_results"],
                    **params["settings"]
                )
                results.extend(search_results.get("results", []))
                
                # Update progress
                task["progress"] = ((idx + 1) / len(params["search_terms"])) * 100
                task["logs"].append(f"Processed term: {term}")
                
            # AI analysis and optimization
            analysis = await self.ai_agent.analyze_search_results(results)
            if analysis["quality_score"] < 0.6:
                # Optimize search terms and retry
                optimized_terms = await self.ai_agent.optimize_search_terms(
                    params["search_terms"], 
                    results
                )
                task["logs"].append("Low quality results, retrying with optimized terms")
                
                # Perform search with optimized terms
                for term in optimized_terms:
                    additional_results = manual_search(
                        session=session,
                        terms=[term],
                        num_results=params["num_results"],
                        **params["settings"]
                    )
                    results.extend(additional_results.get("results", []))
            
            task["results"] = results
            
    async def _process_email_campaign(self, task: Dict):
        """Process email campaign task with AI optimization"""
        params = task["params"]
        with SessionLocal() as session:
            from streamlit_app import send_email_ses, save_email_campaign
            
            # AI quality check for emails
            emails = [lead["email"] for lead in params["leads"]]
            quality_checks = await self.ai_agent.detect_email_quality(emails)
            
            results = []
            for idx, (lead, is_quality) in enumerate(zip(params["leads"], quality_checks)):
                if is_quality:
                    response = send_email_ses(
                        session=session,
                        from_email=params["from_email"],
                        to_email=lead["email"],
                        subject=params["subject"],
                        body=params["body"],
                        reply_to=params.get("reply_to")
                    )
                    
                    if response:
                        save_email_campaign(
                            session=session,
                            lead_email=lead["email"],
                            template_id=params["template_id"],
                            status="sent",
                            sent_at=datetime.utcnow(),
                            subject=params["subject"],
                            message_id=response["MessageId"],
                            email_body=params["body"]
                        )
                        results.append({"email": lead["email"], "status": "sent"})
                    else:
                        results.append({"email": lead["email"], "status": "failed"})
                else:
                    results.append({"email": lead["email"], "status": "skipped_low_quality"})
                    
                task["progress"] = ((idx + 1) / len(params["leads"])) * 100
                
            task["results"] = results
            
    async def _process_lead_analysis(self, task: Dict):
        """Analyze leads for quality and optimization"""
        with SessionLocal() as session:
            # Implementation for lead analysis
            pass
            
    async def _process_database_optimization(self, task: Dict):
        """Optimize database and clean up low-quality data"""
        with SessionLocal() as session:
            # Implementation for database optimization
            pass

# Initialize task manager
task_manager = TaskManager()

def submit_task(task_type: str, params_json: str) -> str:
    """Submit a new task"""
    try:
        params = json.loads(params_json)
        task_id = asyncio.run(task_manager.create_task(task_type, params))
        # Start processing in background
        asyncio.create_task(task_manager.process_task(task_id))
        return task_id
    except Exception as e:
        return f"Error: {str(e)}"

def check_task_status(task_id: str) -> tuple:
    """Check task status"""
    if not task_id:
        return "waiting", "0%", "No task running"
        
    task = task_manager.get_task_status(task_id)
    if not task:
        return "not_found", "0%", "Task not found"
        
    return (
        task["status"],
        f"{task['progress']:.1f}%",
        "\n".join(task["logs"][-10:])
    )

def get_task_results(task_id: str) -> str:
    """Get task results"""
    task = task_manager.get_task_status(task_id)
    if not task or not task.get("results"):
        return "No results found"
    return json.dumps(task["results"], indent=2)

# Create Gradio interface
with gr.Blocks(title="AI-Powered Task Worker", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI-Powered Task Worker")
    
    with gr.Tab("Lead Search"):
        with gr.Row():
            with gr.Column():
                search_terms = gr.Textbox(
                    label="Search Terms (one per line)",
                    lines=5,
                    placeholder="Enter search terms..."
                )
                num_results = gr.Number(
                    label="Results per term",
                    value=10,
                    minimum=1,
                    maximum=50000
                )
                search_settings = gr.JSON(
                    label="Search Settings",
                    value={
                        "ignore_previously_fetched": True,
                        "optimize_english": False,
                        "optimize_spanish": False,
                        "shuffle_keywords": True,
                        "language": "ES",
                        "enable_email_sending": False
                    }
                )
                search_btn = gr.Button("Start Search", variant="primary")
            
            with gr.Column():
                search_task_id = gr.Textbox(label="Task ID", interactive=False)
                search_status = gr.Textbox(label="Status", interactive=False)
                search_progress = gr.Textbox(label="Progress", interactive=False)
                search_logs = gr.Textbox(label="Logs", lines=10, interactive=False)
                search_results = gr.JSON(label="Results")
    
    with gr.Tab("Email Campaign"):
        with gr.Row():
            with gr.Column():
                campaign_params = gr.JSON(
                    label="Campaign Parameters",
                    value={
                        "template_id": None,
                        "from_email": "",
                        "reply_to": "",
                        "leads": []
                    }
                )
                campaign_btn = gr.Button("Start Campaign", variant="primary")
            
            with gr.Column():
                campaign_task_id = gr.Textbox(label="Task ID", interactive=False)
                campaign_status = gr.Textbox(label="Status", interactive=False)
                campaign_progress = gr.Textbox(label="Progress", interactive=False)
                campaign_logs = gr.Textbox(label="Logs", lines=10, interactive=False)
                campaign_results = gr.JSON(label="Results")
    
    with gr.Tab("Analysis & Optimization"):
        with gr.Row():
            with gr.Column():
                analysis_type = gr.Radio(
                    label="Analysis Type",
                    choices=["Lead Quality", "Database Optimization"],
                    value="Lead Quality"
                )
                analysis_params = gr.JSON(label="Analysis Parameters", value={})
                analysis_btn = gr.Button("Start Analysis", variant="primary")
            
            with gr.Column():
                analysis_task_id = gr.Textbox(label="Task ID", interactive=False)
                analysis_status = gr.Textbox(label="Status", interactive=False)
                analysis_progress = gr.Textbox(label="Progress", interactive=False)
                analysis_logs = gr.Textbox(label="Logs", lines=10, interactive=False)
                analysis_results = gr.JSON(label="Results")
    
    # Event handlers for Lead Search
    search_btn.click(
        lambda terms, num, settings: submit_task("lead_search", json.dumps({
            "search_terms": terms.split("\n"),
            "num_results": num,
            "settings": json.loads(settings)
        })),
        inputs=[search_terms, num_results, search_settings],
        outputs=[search_task_id]
    )
    
    demo.load(
        check_task_status,
        inputs=[search_task_id],
        outputs=[search_status, search_progress, search_logs],
        every=1
    )
    
    demo.load(
        get_task_results,
        inputs=[search_task_id],
        outputs=[search_results],
        every=1
    )
    
    # Event handlers for Email Campaign
    campaign_btn.click(
        lambda params: submit_task("email_campaign", params),
        inputs=[campaign_params],
        outputs=[campaign_task_id]
    )
    
    demo.load(
        check_task_status,
        inputs=[campaign_task_id],
        outputs=[campaign_status, campaign_progress, campaign_logs],
        every=1
    )
    
    demo.load(
        get_task_results,
        inputs=[campaign_task_id],
        outputs=[campaign_results],
        every=1
    )
    
    # Event handlers for Analysis
    analysis_btn.click(
        lambda type, params: submit_task(
            "lead_analysis" if type == "Lead Quality" else "database_optimization",
            params
        ),
        inputs=[analysis_type, analysis_params],
        outputs=[analysis_task_id]
    )
    
    demo.load(
        check_task_status,
        inputs=[analysis_task_id],
        outputs=[analysis_status, analysis_progress, analysis_logs],
        every=1
    )
    
    demo.load(
        get_task_results,
        inputs=[analysis_task_id],
        outputs=[analysis_results],
        every=1
    )

if __name__ == "__main__":
    # Launch with queue enabled for background processing
    demo.queue(concurrency_count=3).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )