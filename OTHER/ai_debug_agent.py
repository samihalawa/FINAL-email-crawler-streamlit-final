# pip install streamlit torch transformers huggingface_hub pyyaml
import ast
import hashlib
import json
import logging
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
import pytest
import streamlit as st
import plotly.express as px
from pydantic import BaseModel, Field, ValidationError
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from loguru import logger
from streamlit_aggrid import AgGrid, GridOptionsBuilder
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.agents import Tool, AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import semantic_kernel as sk
from semantic_kernel.planning.sequential_planner import SequentialPlanner
from continuedev import ContinueSDK
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import hf_hub_download
from functools import lru_cache
import yaml
from transformers import pipeline
import pandas as pd
import requests
import atexit
import gc
from concurrent.futures import TimeoutError
from threading import Lock
import shutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from continuedev.core.main import Continue
from continuedev.libs.analysis import CodeAnalyzer as ContinueAnalyzer
from continuedev.libs.util.logging import logger as continue_logger
from streamlit_autorefresh import st_autorefresh
from streamlit_ace import st_ace
from streamlit_timeline import timeline
from streamlit_elements import elements, mui, html
from streamlit_card import card
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_custom_notification_box import custom_notification_box
from streamlit_lottie import st_lottie

# Configure Loguru to work with Streamlit
logger.remove()
logger.add(lambda msg: st.session_state.log_messages.append(msg), level="INFO")

# Initialize session state for logs
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

# Data Classes
@dataclass
class CodeIssue:
    file_path: str
    line_number: int
    issue_type: str
    description: str
    suggested_fix: str
    confidence: float

# Pydantic Models for Configuration
class AgentConfig(BaseModel):
    api_key: str = Field(..., env="OPENAI_API_KEY")
    project_path: Path
    model: str = "gpt-4o-mini"
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0)
    max_retries: int = Field(3, ge=0)
    cache_timeout: int = Field(3600, ge=0)  # in seconds

# AI Debug Agent Class
class AIDebugAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        with console.status("[bold green]Initializing AI Debug Agent..."):
            self.setup_components()
            self.setup_tools()
            self.setup_memory()
            self.setup_agent()
            self.issue_history: List[CodeIssue] = []
            self._analysis_lock = asyncio.Lock()
            console.log("[bold green]✓[/] Agent initialized successfully")

    def setup_components(self):
        """Initialize core components"""
        self.llm = ChatOpenAI(
            temperature=0.2,
            model=self.config.model,
            openai_api_key=self.config.api_key
        )
        
        # Setup semantic kernel
        self.kernel = sk.Kernel()
        self.kernel.add_chat_service("code_analysis", self.llm)
        self.planner = SequentialPlanner(self.kernel)

    def setup_tools(self):
        """Initialize specialized code tools"""
        self.tools = [
            Tool(
                name="static_analysis",
                func=self._static_analysis,
                description="Perform static code analysis"
            ),
            Tool(
                name="ai_analysis",
                func=self._ai_analysis,
                description="Perform AI-powered code analysis"
            ),
            Tool(
                name="apply_fix",
                func=self._apply_fix,
                description="Apply fixes to code issues"
            ),
            Tool(
                name="run_tests",
                func=self.run_tests,
                description="Run project tests"
            )
        ]

    def setup_memory(self):
        """Initialize memory systems"""
        self.memory = ConversationBufferMemory(
            memory_key="code_history",
            return_messages=True
        )
        
        # Vector store for code snippets
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.config.api_key)
        self.code_memory = FAISS.from_texts(
            [],
            self.embeddings,
            metadatas=[{"type": "code_snippet"}]
        )
        
        # Analysis cache
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}

    def setup_agent(self):
        """Initialize the agent with all components"""
        self.agent = OpenAIFunctionsAgent(
            llm=self.llm,
            tools=self.tools,
            memory=self.memory
        )
        
        self.executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3
        )

    async def analyze_code(self, file_path: str) -> List[CodeIssue]:
        """Optimized code analysis"""
        async with self._analysis_lock:
            progress = AnalysisProgress()
            
            try:
                # Check cache
                file_hash = self.get_file_hash(file_path)
                cache_key = f"analysis_{file_hash}"
                
                if cache_key in self.analysis_cache:
                    cache_entry = self.analysis_cache[cache_key]
                    if time.time() - cache_entry['timestamp'] < self.config.cache_timeout:
                        progress.update(1.0, 'complete', {
                            'Source': 'Cache',
                            'Issues': len(cache_entry['issues'])
                        })
                        return cache_entry['issues']

                with open(file_path, 'r') as f:
                    code = f.read()

                # Static analysis (20%)
                progress.update(0.2, 'static')
                static_issues = await self._static_analysis(code)
                
                # AI analysis (50%)
                progress.update(0.5, 'ai', {
                    'Issues Found': len(static_issues)
                })
                ai_issues = await self._ai_analysis(code)
                
                # Continue analysis (80%)
                progress.update(0.8, 'continue', {
                    'Total Issues': len(static_issues) + len(ai_issues)
                })
                continue_issues = await self.analyzer._continue_analysis(code)
                
                # Combine results (90%)
                progress.update(0.9, 'combining')
                issues = self._combine_issues(
                    static_issues, 
                    ai_issues,
                    continue_issues
                )
                
                # Cache results
                self.analysis_cache[cache_key] = {
                    'issues': issues,
                    'timestamp': time.time()
                }
                
                # Complete
                progress.update(1.0, 'complete', {
                    'Total Issues': len(issues),
                    'Confidence': f"{self.get_average_confidence(issues):.1%}"
                })
                
                return issues

            except Exception as e:
                progress.complete(success=False)
                self.log_error(f"Analysis failed: {str(e)}")
                return []

    def _combine_issues(self, static_issues: List[Dict], ai_issues: List[Dict]) -> List[CodeIssue]:
        """Combine and deduplicate issues from different sources"""
        combined = []
        seen = set()
        
        for issue in static_issues + ai_issues:
            key = f"{issue['line_number']}:{issue['description']}"
            if key not in seen:
                seen.add(key)
                combined.append(CodeIssue(**issue))
        
        return combined

    async def fix_issues(self, issues: List[CodeIssue]) -> bool:
        """Enhanced issue fixing with validation"""
        try:
            for issue in issues:
                if issue.confidence >= self.config.confidence_threshold:
                    # Create fix plan
                    plan = await self.planner.create_plan(
                        f"Fix issue in {issue.file_path} while maintaining code integrity"
                    )

                    # Apply fix
                    success = await self.executor.arun(
                        input={
                            "issue": issue,
                            "plan": plan
                        }
                    )

                    if success:
                        self.issue_history.append(issue)
                        logger.success(f"Fixed issue at line {issue.line_number}")
                    else:
                        self.log_error(f"Failed to fix issue at line {issue.line_number}")
                        return False

            # Validate all changes
            if await self._validate_changes(issues[0].file_path):
                return True
            
            # Rollback if validation fails
            await self._rollback_changes(issues[0].file_path)
            return False

        except Exception as e:
            self.log_error(f"Error fixing issues: {str(e)}")
            return False

    def run_tests(self) -> Tuple[bool, str]:
        """Run project tests"""
        try:
            result = pytest.main([str(self.project_path)])
            if result == 0:
                return True, "Tests passed successfully."
            else:
                return False, "Tests failed. Please check the test logs."
        except Exception as e:
            return False, f"Failed to run tests: {str(e)}"

    @lru_cache(maxsize=100)
    def get_file_hash(self, file_path: str) -> str:
        """Optimized file hash computation with caching"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def log_error(self, message: str):
        """Log error messages to the UI"""
        logger.error(message)
        st.error(message)

    async def _validate_changes(self, file_path: str) -> bool:
        """Validate code changes"""
        try:
            # Basic syntax check
            with open(file_path, 'r') as f:
                code = f.read()
            ast.parse(code)
            
            # Run tests if available
            success, _ = await self.run_tests()
            return success
            
        except SyntaxError:
            return False
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

    async def _rollback_changes(self, file_path: str):
        """Rollback changes if validation fails"""
        backup_path = f"{file_path}.backup"
        try:
            if Path(backup_path).exists():
                with open(backup_path, 'r') as f:
                    original_code = f.read()
                with open(file_path, 'w') as f:
                    f.write(original_code)
                logger.info("Changes rolled back successfully")
        except Exception as e:
            logger.error(f"Failed to rollback changes: {str(e)}")

    async def _static_analysis(self, code: str) -> List[Dict]:
        """Perform static code analysis"""
        issues = []
        try:
            tree = ast.parse(code)
            analyzer = StaticAnalyzer()
            analyzer.visit(tree)
            issues.extend(analyzer.issues)
        except Exception as e:
            logger.error(f"Static analysis failed: {str(e)}")
        return issues

    async def _ai_analysis(self, code: str) -> List[Dict]:
        """Perform AI-powered code analysis"""
        try:
            response = await self.llm.agenerate([code])
            return self._parse_ai_response(response)
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            return []

    async def batch_analyze_files(self, files: List[str], batch_size: int = 3) -> Dict[str, List[CodeIssue]]:
        """Process multiple files in batches with progress tracking"""
        results = {}
        total_files = len(files)
        progress = ProgressManager()
        
        try:
            for i in range(0, total_files, batch_size):
                batch = files[i:i + batch_size]
                progress.update_progress(
                    i/total_files,
                    f"Analyzing batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1}"
                )
                
                # Process batch concurrently
                batch_results = await asyncio.gather(
                    *[self.analyze_code(file) for file in batch],
                    return_exceptions=True
                )
                
                # Store results
                for file, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to analyze {file}: {str(result)}")
                        results[file] = []
                    else:
                        results[file] = result
                
            progress.complete(success=True)
            return results
            
        except Exception as e:
            progress.complete(success=False)
            logger.error(f"Batch analysis failed: {str(e)}")
            return {}

# File Change Handler Class
class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, agent: AIDebugAgent):
        super().__init__()
        self.agent = agent
        self._change_queue = asyncio.Queue()

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            asyncio.create_task(self.process_file_change(event.src_path))

    async def process_file_change(self, file_path: str):
        """Enhanced file change processing"""
        st.session_state.current_file = file_path
        
        with st.spinner(f"Analyzing {Path(file_path).name}..."):
            issues = await self.agent.analyze_code(file_path)
            
            if issues:
                st.info(f"Detected {len(issues)} issues in {Path(file_path).name}")
                if await self.agent.fix_issues(issues):
                    st.success(f"All issues in {Path(file_path).name} have been fixed.")
                    success, test_result = await self.agent.run_tests()
                    if success:
                        st.success("✅ All tests passed after fixes.")
                    else:
                        st.warning(f"⚠️ Tests failed after fixes: {test_result}")
            else:
                st.success(f"No issues found in {Path(file_path).name}.")

# UI Creation Functions
def create_ui() -> AgentConfig:
    """Create optimized UI layout and return validated AgentConfig"""
    st.set_page_config(
        page_title="AI Debug Agent",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.image("logo.png", width=100)  # Replace with your logo
        st.title("AI Debug Agent")
        
        # Project Settings
        with st.expander("🔧 Project Settings", expanded=True):
            api_key = st.text_input(
                "🔑 OpenAI API Key", 
                type="password",
                help="Your OpenAI API key for code analysis"
            )
            
            project_path_input = st.text_input(
                "📁 Project Path",
                help="Path to your project directory"
            )
            project_path = Path(project_path_input) if project_path_input else Path.cwd()
        
        # Advanced Settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            model = st.selectbox(
                "🤖 AI Model",
                ["gpt-4-mini", "gpt-3.5-turbo"],
                index=0,
                help="Select AI model for code analysis"
            )
            
            confidence_threshold = st.slider(
                "🔍 Auto-fix Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Minimum confidence level for automatic fixes"
            )
    
    # Validate Configuration
    try:
        config = AgentConfig(
            api_key=api_key,
            project_path=project_path,
            model=model,
            confidence_threshold=confidence_threshold
        )
    except ValidationError as ve:
        st.sidebar.error(f"Configuration Error: {ve}")
        return None

    return config

def create_dashboard(agent: AIDebugAgent):
    """Create real-time monitoring dashboard"""
    st.header("🖥️ Real-Time Monitoring Dashboard")
    
    # Status Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📂 Files Monitored", len(agent.analysis_cache))
    with col2:
        st.metric("🐞 Issues Fixed", len(agent.issue_history))
    with col3:
        st.metric("⏰ Cache Timeout", f"{agent.config.cache_timeout} seconds")
    
    add_vertical_space(2)
    
    # Main Content Area with Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Analytics", "🐞 Issues", "📝 Logs"])
    
    with tab1:
        create_analytics_panel(agent)
    
    with tab2:
        create_issues_panel(agent)
    
    with tab3:
        create_logs_panel()

def create_analytics_panel(agent: AIDebugAgent):
    """Create analytics dashboard"""
    st.subheader("📊 Analytics")
    if not agent.issue_history:
        st.info("No issues detected yet.")
        return
    
    # Issue Types Distribution
    issue_types = {}
    for issue in agent.issue_history:
        issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
    
    fig1 = px.pie(
        names=list(issue_types.keys()),
        values=list(issue_types.values()),
        title='Issue Types Distribution'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    add_vertical_space(1)
    
    # Confidence Level Distribution
    confidence_levels = [issue.confidence for issue in agent.issue_history]
    fig2 = px.histogram(
        confidence_levels,
        nbins=10,
        title='Confidence Level Distribution',
        labels={'value': 'Confidence Level'},
        range_x=[0,1]
    )
    st.plotly_chart(fig2, use_container_width=True)

def create_issues_panel(agent: AIDebugAgent):
    """Create interactive issues panel using AgGrid"""
    st.subheader("🐞 Detected Issues")
    if not agent.issue_history:
        st.info("No issues detected yet.")
        return
    
    # Prepare data for AgGrid
    issues_data = [
        {
            "File Path": issue.file_path,
            "Line Number": issue.line_number,
            "Issue Type": issue.issue_type,
            "Description": issue.description,
            "Suggested Fix": issue.suggested_fix,
            "Confidence (%)": f"{issue.confidence * 100:.2f}"
        }
        for issue in agent.issue_history
    ]
    
    gb = GridOptionsBuilder.from_dataframe(pd.DataFrame(issues_data))
    gb.configure_pagination(paginationAutoPageSize=True)  # Add pagination
    gb.configure_side_bar()
    gb.configure_default_column(editable=False)
    gridOptions = gb.build()
    
    AgGrid(
        pd.DataFrame(issues_data),
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        height=400,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True
    )

def create_logs_panel():
    """Create logs panel to display log messages"""
    st.subheader("📝 Logs")
    if st.session_state.log_messages:
        logs = "\n".join(st.session_state.log_messages)
        st.text_area("Application Logs", value=logs, height=300)
    else:
        st.info("No logs to display yet.")

# Define available models
AVAILABLE_MODELS = {
    "StarCoder-Mini": {
        "name": "bigcode/starcoderbase-1b",
        "description": "Lightweight code model (1B parameters)",
        "memory_required": "4GB"
    },
    "CodeGen-Mono": {
        "name": "Salesforce/codegen-350M-mono",
        "description": "Very lightweight model (350M parameters)",
        "memory_required": "2GB"
    },
    "SantaCoder": {
        "name": "bigcode/santacoder",
        "description": "Balanced model (1.1B parameters)",
        "memory_required": "4GB"
    }
}

@dataclass
class CodeAnalysisResult:
    issues: List[Dict]
    suggestions: List[str]
    confidence: float

class ModelManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
            return cls._instance

    @st.cache_resource(show_spinner=False)
    def load_model(self, model_name: str, device: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Load and cache the model with optimizations"""
        try:
            memory_manager = MemoryManager()
            
            with st.spinner(f"Loading {model_name}..."):
                # Check available memory
                if not memory_manager.check_memory():
                    raise RuntimeError("Insufficient memory")

                # Load tokenizer with safety timeout
                tokenizer = self._load_with_timeout(
                    lambda: AutoTokenizer.from_pretrained(
                        model_name,
                        use_auth_token=st.secrets.get("HF_TOKEN"),
                        use_fast=True
                    ),
                    timeout=30
                )

                # Load model with optimizations
                model = self._load_with_timeout(
                    lambda: AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        use_auth_token=st.secrets.get("HF_TOKEN"),
                        revision="main",
                        offload_folder="offload",
                        quantization_config=None if device == "cuda" else "8bit"
                    ),
                    timeout=120
                )

                return model, tokenizer

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            st.error(f"Failed to load model: {str(e)}")
            return None, None

    @staticmethod
    def _load_with_timeout(func, timeout: int):
        """Execute function with timeout"""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                raise RuntimeError(f"Operation timed out after {timeout} seconds")

class CodeAnalyzer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.setup_pipeline()
        self._cache = {}
        self._cache_lock = Lock()
        
        # Initialize Continue features using existing model
        self.continue_analyzer = ContinueAnalyzer(
            model=self.model,
            tokenizer=self.tokenizer
        )

    async def analyze_code(self, code: str) -> CodeAnalysisResult:
        """Enhanced analysis using existing components"""
        try:
            code_hash = hashlib.md5(code.encode()).hexdigest()
            
            # Check cache
            with self._cache_lock:
                if code_hash in self._cache:
                    return self._cache[code_hash]

            # Parallel analysis using existing tools
            results = await asyncio.gather(
                self._static_analysis(code),
                self._ai_analysis(code),
                self._continue_analysis(code)
            )
            
            # Combine results
            combined_result = self._combine_analysis_results(*results)
            
            # Cache results
            with self._cache_lock:
                self._cache[code_hash] = combined_result
                if len(self._cache) > 100:
                    self._cache.pop(next(iter(self._cache)))
            
            return combined_result

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return CodeAnalysisResult([], [], 0.0)

    async def _continue_analysis(self, code: str) -> Dict:
        """Analyze using Continue features"""
        try:
            return await self.continue_analyzer.analyze(
                code,
                include_fixes=True,
                include_metrics=True
            )
        except Exception as e:
            logger.error(f"Continue analysis failed: {str(e)}")
            return {}

    def _combine_analysis_results(self, static_results, ai_results, continue_results):
        """Combine results from different analyzers"""
        all_issues = []
        all_suggestions = []
        confidence_scores = []

        # Add static analysis results
        if static_results:
            all_issues.extend(static_results.get('issues', []))
            confidence_scores.append(0.9)  # High confidence for static analysis

        # Add AI analysis results
        if ai_results:
            all_issues.extend(ai_results.get('issues', []))
            all_suggestions.extend(ai_results.get('suggestions', []))
            confidence_scores.append(float(ai_results.get('confidence', 0.7)))

        # Add Continue analysis results
        if continue_results:
            all_issues.extend(continue_results.get('issues', []))
            all_suggestions.extend(continue_results.get('suggestions', []))
            confidence_scores.append(float(continue_results.get('confidence', 0.8)))

        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        return CodeAnalysisResult(
            issues=self._deduplicate_issues(all_issues),
            suggestions=list(set(all_suggestions)),
            confidence=avg_confidence
        )

    def _deduplicate_issues(self, issues: List[Dict]) -> List[Dict]:
        """Remove duplicate issues based on similarity"""
        unique_issues = []
        seen_descriptions = set()

        for issue in issues:
            desc_hash = hashlib.md5(
                issue['description'].lower().encode()
            ).hexdigest()
            
            if desc_hash not in seen_descriptions:
                seen_descriptions.add(desc_hash)
                unique_issues.append(issue)

        return unique_issues

    def _create_analysis_prompt(self, code: str) -> str:
        """Create a detailed analysis prompt"""
        return f"""
        Analyze this Python code for issues and improvements:
        ```python
        {code}
        ```
        Focus on:
        1. Security vulnerabilities
        2. Performance optimizations
        3. Code quality issues
        4. Best practices
        5. Potential bugs

        Format response as JSON with:
        - issues: list of found issues
        - suggestions: list of improvements
        - confidence: float between 0-1
        """

    def _parse_result(self, result: str) -> CodeAnalysisResult:
        """Parse model output into CodeAnalysisResult"""
        try:
            # Extract JSON from the generated text
            json_str = result[result.find('{'):result.rfind('}')+1]
            data = json.loads(json_str)
            
            return CodeAnalysisResult(
                issues=data.get('issues', []),
                suggestions=data.get('suggestions', []),
                confidence=float(data.get('confidence', 0.5))
            )
        except Exception as e:
            logger.error(f"Failed to parse model output: {str(e)}")
            return CodeAnalysisResult([], [], 0.0)

    def setup_pipeline(self):
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=1024,
            temperature=0.2
        )

def create_sidebar():
    """Create the sidebar UI"""
    st.sidebar.title("⚙️ Settings")
    
    # Model Selection
    st.sidebar.header("Model Configuration")
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(AVAILABLE_MODELS.keys()),
        help="Choose a code analysis model"
    )
    
    # Show model details
    model_info = AVAILABLE_MODELS[selected_model]
    st.sidebar.info(
        f"""
        **Model Details:**
        - Size: {model_info['memory_required']}
        - {model_info['description']}
        """
    )
    
    # Device Selection
    device = st.sidebar.radio(
        "Device",
        options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        help="Select processing device"
    )
    
    # Analysis Settings
    st.sidebar.header("Analysis Settings")
    settings = {
        "confidence_threshold": st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Minimum confidence for issue reporting"
        ),
        "max_issues": st.sidebar.number_input(
            "Max Issues",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of issues to report"
        )
    }
    
    return selected_model, device, settings

def create_main_ui():
    """Enhanced main UI with debugging features"""
    st.title("🚀 AI Code Debug Agent")
    
    # Debug Controls
    with st.expander("🔧 Debug Controls", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Continue"):
                st.session_state.debug_session.resume()
        
        with col2:
            if st.button("⏸️ Pause"):
                st.session_state.debug_session.pause()
        
        with col3:
            if st.button("⏭️ Step Over"):
                st.session_state.debug_session.step()
    
    # Add watch expression
    watch_expr = st.text_input("Add Watch Expression")
    if watch_expr:
        st.session_state.debug_session.add_watch(watch_expr)
    
    # Show debug information
    if hasattr(st.session_state, 'debug_session'):
        DebugUI.create_debug_panel(st.session_state.debug_session)

def display_analysis_results(results: CodeAnalysisResult):
    """Display analysis results in a structured way"""
    st.header("📊 Analysis Results")
    
    # Issues Tab
    tab1, tab2, tab3 = st.tabs(["🐛 Issues", "💡 Suggestions", "📈 Stats"])
    
    with tab1:
        for idx, issue in enumerate(results.issues):
            with st.expander(f"Issue #{idx+1}: {issue['type']}", expanded=idx == 0):
                st.markdown(f"**Description:** {issue['description']}")
                if 'code' in issue:
                    st.code(issue['code'], language="python")
                if 'fix' in issue:
                    st.markdown("**Suggested Fix:**")
                    st.code(issue['fix'], language="python")
                    if st.button(f"Apply Fix #{idx+1}"):
                        apply_fix(issue['fix'])
    
    with tab2:
        for idx, suggestion in enumerate(results.suggestions):
            st.markdown(f"**Suggestion {idx+1}:**")
            st.markdown(suggestion)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Issues Found", len(results.issues))
        with col2:
            st.metric("Confidence Score", f"{results.confidence:.2%}")

# Add cleanup handler
def cleanup_resources():
    """Clean up resources when application exits"""
    try:
        if hasattr(st.session_state, 'model'):
            del st.session_state.model
        if hasattr(st.session_state, 'analyzer'):
            del st.session_state.analyzer
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")

atexit.register(cleanup_resources)

# Add memory management
class MemoryManager:
    def __init__(self):
        self._lock = Lock()
        self._max_memory = 0.9  # 90% memory threshold

    def check_memory(self) -> bool:
        try:
            with self._lock:
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated()
                    memory_reserved = torch.cuda.memory_reserved()
                    if memory_allocated / memory_reserved > self._max_memory:
                        torch.cuda.empty_cache()
                        gc.collect()
                return True
        except Exception as e:
            logger.error(f"Memory check failed: {str(e)}")
            return False

# Main Function
def main():
    """Enhanced main function with modern UI"""
    # Page config
    st.set_page_config(
        page_title="AI Debug Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Create enhanced UI
    create_enhanced_ui()
    
    if not st.session_state.debug_started:
        show_welcome_screen()
    else:
        show_debug_session()

def show_welcome_screen():
    """Show welcome screen with setup options"""
    with st.container():
        st.markdown(
            """
            <div style='text-align: center'>
                <h1>🚀 Welcome to AI Debug Agent</h1>
                <p>Intelligent code analysis and debugging powered by AI</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Project setup card
        with st.expander("🔧 Project Setup", expanded=True):
            setup_project()

def show_debug_session():
    """Show active debugging session"""
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔍 Analysis",
        "💻 Code",
        "📝 Logs"
    ])
    
    with tab1:
        create_analysis_dashboard(st.session_state.agent)
        show_analysis_timeline(st.session_state.agent)
    
    with tab2:
        if st.session_state.current_file:
            show_file_analysis()
    
    with tab3:
        code = create_code_editor()
        if st.button("Analyze Code"):
            analyze_code(code)
    
    with tab4:
        show_enhanced_logs()

def show_enhanced_logs():
    """Show enhanced logs with filtering and search"""
    st.subheader("📝 Application Logs")
    
    # Log filters
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Search logs")
    with col2:
        log_level = st.selectbox(
            "Log Level",
            ["ALL", "INFO", "WARNING", "ERROR"]
        )
    
    # Filter logs
    filtered_logs = filter_logs(
        st.session_state.log_messages,
        search,
        log_level
    )
    
    # Show logs with syntax highlighting
    for log in filtered_logs:
        st.code(log, language="python")

def apply_fix(fix_code: str):
    """Apply code fix with safety checks"""
    if 'current_file' not in st.session_state:
        st.error("No file selected for fixing")
        return

    file_path = st.session_state.current_file
    backup_path = f"{file_path}.backup"
    
    try:
        # Create backup first
        shutil.copy2(file_path, backup_path)
        
        # Validate fix_code
        try:
            ast.parse(fix_code)
        except SyntaxError:
            raise ValueError("Invalid Python syntax in fix")

        # Apply fix
        with open(file_path, 'w') as f:
            f.write(fix_code)

        # Verify file was written correctly
        with open(file_path, 'r') as f:
            if f.read() != fix_code:
                raise ValueError("File verification failed")

        st.success("Fix applied successfully!")
        
    except Exception as e:
        logger.error(f"Failed to apply fix: {str(e)}")
        st.error(f"Failed to apply fix: {str(e)}")
        
        # Restore backup
        try:
            if Path(backup_path).exists():
                shutil.copy2(backup_path, file_path)
                st.info("Original file restored from backup")
        except Exception as restore_error:
            logger.error(f"Failed to restore backup: {str(restore_error)}")
            st.error("Failed to restore backup. Please check the file manually.")
    
    finally:
        # Cleanup backup
        try:
            if Path(backup_path).exists():
                Path(backup_path).unlink()
        except Exception as e:
            logger.error(f"Failed to cleanup backup: {str(e)}")

class EnhancedDebugAgent(AIDebugAgent):
    """Enhanced debug agent with advanced ContinueSDK features"""
    
    async def setup_continue_sdk(self):
        """Initialize enhanced ContinueSDK components"""
        self.continue_sdk = Continue(
            config={
                "models": {
                    "default": self.config.model,
                    "fallback": "codegen-350M-mono"
                },
                "features": {
                    "auto_complete": True,
                    "real_time_suggestions": True,
                    "semantic_search": True,
                    "code_explanation": True,
                    "test_generation": True,
                    "performance_analysis": True
                },
                "ui": {
                    "show_progress": True,
                    "show_suggestions": True,
                    "theme": "dark"
                }
            }
        )
        
        # Setup enhanced features
        await self.setup_enhanced_features()
    
    async def setup_enhanced_features(self):
        """Setup additional ContinueSDK features"""
        # Code explanation
        self.explanation_engine = await self.continue_sdk.create_explanation_engine()
        
        # Test generation
        self.test_generator = await self.continue_sdk.create_test_generator()
        
        # Performance analyzer
        self.perf_analyzer = await self.continue_sdk.create_performance_analyzer()
        
        # Real-time suggestions
        self.suggestion_engine = await self.continue_sdk.create_suggestion_engine(
            update_interval=2.0
        )
    
    async def get_code_explanation(self, code: str) -> str:
        """Get detailed code explanation"""
        return await self.explanation_engine.explain(code)
    
    async def generate_tests(self, code: str) -> List[str]:
        """Generate unit tests"""
        return await self.test_generator.generate(code)
    
    async def analyze_performance(self, code: str) -> Dict:
        """Analyze code performance"""
        return await self.perf_analyzer.analyze(code)

class DebugSession:
    def __init__(self):
        self.breakpoints = set()
        self.watch_expressions = {}
        self.call_stack = []
        self.variables = {}

    def add_breakpoint(self, file: str, line: int):
        self.breakpoints.add((file, line))

    def add_watch(self, expression: str):
        self.watch_expressions[expression] = None

    def update_context(self, frame):
        """Update debugging context"""
        self.variables = {
            name: str(value)
            for name, value in frame.f_locals.items()
        }
        self.call_stack = self._get_call_stack(frame)

    def _get_call_stack(self, frame):
        stack = []
        while frame:
            stack.append({
                'file': frame.f_code.co_filename,
                'function': frame.f_code.co_name,
                'line': frame.f_lineno
            })
            frame = frame.f_back
        return stack

class DebugUI:
    """Debug UI components"""
    
    def create_debug_panel(debug_session: DebugSession):
        st.subheader("🔍 Debug Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(" Breakpoints")
            for file, line in debug_session.breakpoints:
                st.code(f"{file}:{line}")
        
        with col2:
            st.write("👀 Watch Expressions")
            for expr, value in debug_session.watch_expressions.items():
                st.code(f"{expr} = {value}")
        
        st.write("📚 Call Stack")
        for frame in debug_session.call_stack:
            st.code(
                f"{frame['file']}:{frame['line']} in {frame['function']}"
            )
        
        st.write("🔤 Variables")
        st.json(debug_session.variables)

def create_enhanced_ui():
    """Create enhanced UI with modern components"""
    # Auto refresh for real-time updates
    count = st_autorefresh(interval=2000, limit=100)
    
    # Modern header
    colored_header(
        label="🤖 AI Code Debug Agent",
        description="Intelligent code analysis and debugging",
        color_name="blue-70"
    )

    # Dashboard layout
    with elements("dashboard"):
        # Layout
        with mui.Box(sx={"display": "flex", "flexDirection": "row", "p": 1}):
            # Sidebar
            with mui.Paper(sx={"width": 240, "mr": 2}):
                mui.Typography("Debug Controls", variant="h6")
                
            # Main content
            with mui.Paper(sx={"flex": 1}):
                mui.Typography("Analysis Results", variant="h6")

def create_analysis_dashboard(agent: AIDebugAgent):
    """Create enhanced real-time analysis dashboard"""
    # Header with animation
    st.markdown(
        """
        <div style='text-align: center'>
            <h1>🔍 Analysis Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Metrics with animations
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card(
            "Files Analyzed",
            len(agent.analysis_cache),
            delta=f"+{len(agent.analysis_cache) - agent.last_cache_size}",
            animation_url="https://assets8.lottiefiles.com/packages/lf20_sz6aib9w.json"
        )
    
    with col2:
        metric_card(
            "Issues Fixed",
            len(agent.issue_history),
            delta=f"+{len(agent.issue_history) - agent.last_issues_count}",
            animation_url="https://assets8.lottiefiles.com/packages/lf20_yg2gpwv5.json"
        )
    
    # Analysis Timeline
    st.subheader("📊 Analysis Timeline")
    timeline_data = create_timeline_data(agent.analysis_events)
    show_interactive_timeline(timeline_data)
    
    # Real-time Metrics
    show_realtime_metrics(agent)
    
    # Issue Distribution
    show_issue_distribution(agent.issue_history)

def metric_card(title: str, value: Any, delta: str = None, animation_url: str = None):
    """Create animated metric card"""
    if animation_url:
        animation = load_lottie_url(animation_url)
        if animation:
            st_lottie(animation, height=100, key=f"metric_{title}")
    
    st.metric(
        title,
        value,
        delta=delta,
        delta_color="normal"
    )

def show_interactive_timeline(data: List[Dict]):
    """Show interactive timeline with plotly"""
    fig = px.timeline(
        data,
        x_start="start",
        x_end="end",
        y="task",
        color="status",
        hover_data=["description"]
    )
    
    fig.update_layout(
        showlegend=True,
        height=300,
        title="Analysis Progress Timeline"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_code_editor():
    """Create interactive code editor"""
    code = st_ace(
        value="# Enter your code here",
        language="python",
        theme="monokai",
        font_size=14,
        show_gutter=True,
        show_print_margin=True,
        wrap=True,
        auto_update=True,
        key="code_editor"
    )
    
    return code

def show_analysis_timeline(agent: AIDebugAgent):
    """Show analysis progress timeline"""
    events = [
        {
            "start": event["timestamp"],
            "content": event["description"],
            "group": event["type"],
            "className": event["status"]
        }
        for event in agent.analysis_events
    ]
    
    timeline(events, groups=["Analysis", "Fix", "Test"])

def create_issue_card(issue: CodeIssue):
    """Create a modern card for each issue"""
    return card(
        title=f"Issue: {issue.issue_type}",
        text=issue.description,
        image="",
        styles={
            "card": {
                "background-color": "#1b2838",
                "border-radius": "10px",
                "padding": "10px",
                "margin": "10px"
            }
        }
    )

def load_lottie_url(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

class ProgressManager:
    """Enhanced progress visualization"""
    def __init__(self):
        self.progress_bar = st.progress(0)
        self.status = st.empty()
        self.animation = st.empty()
        
        # Load analysis animation
        self.analysis_animation = load_lottie_url(
            "https://assets5.lottiefiles.com/packages/lf20_qwATHw.json"
        )
    
    def update_progress(self, progress: float, status: str, show_animation: bool = True):
        """Update progress with animation"""
        self.progress_bar.progress(progress)
        self.status.markdown(f"**{status}**")
        
        if show_animation and self.analysis_animation:
            with self.animation:
                st_lottie(
                    self.analysis_animation,
                    speed=1,
                    height=100,
                    key=f"progress_{time.time()}"
                )
    
    def complete(self, success: bool = True):
        """Show completion status"""
        if success:
            custom_notification_box(
                icon='✅',
                textDisplay='Analysis completed successfully!',
                externalLink='',
                styles="{background-color: '#0f3d3e'}"
            )
        else:
            custom_notification_box(
                icon='⚠️',
                textDisplay='Analysis completed with issues',
                externalLink='',
                styles="{background-color: '#3d0f0f'}"
            )
        
        self.progress_bar.empty()
        self.status.empty()
        self.animation.empty()

class AnalysisProgress:
    def __init__(self):
        self.progress_bar = st.progress(0)
        self.status = st.empty()
        self.metrics = st.empty()
        self.start_time = time.time()

    def update(self, progress: float, phase: str, metrics: Dict = None):
        """Update progress with metrics"""
        self.progress_bar.progress(progress)
        
        # Show current phase with emoji
        emoji = {
            'static': '🔍',
            'ai': '🤖',
            'continue': '🚀',
            'combining': '🔄',
            'complete': '✅'
        }.get(phase, '⚡')
        
        self.status.markdown(f"**{emoji} {phase.title()}**")
        
        if metrics:
            # Show real-time metrics
            cols = self.metrics.columns(len(metrics))
            for col, (key, value) in zip(cols, metrics.items()):
                col.metric(key, value)

    def complete(self, success: bool = True):
        """Show completion with timing"""
        duration = time.time() - self.start_time
        
        if success:
            st.success(f"✅ Analysis completed in {duration:.2f}s")
        else:
            st.error("❌ Analysis failed")
        
        self.progress_bar.empty()
        self.status.empty()
        self.metrics.empty()

if __name__ == "__main__":
    main()