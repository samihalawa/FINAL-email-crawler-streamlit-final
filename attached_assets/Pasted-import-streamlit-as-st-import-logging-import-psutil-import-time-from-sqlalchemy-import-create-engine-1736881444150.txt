import streamlit as st
import logging
import psutil
import time
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import importlib
import glob
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_database_connection() -> Dict:
    """Check database connectivity and health metrics"""
    try:
        load_dotenv()
        DATABASE_URL = os.getenv('DATABASE_URL')
        if not DATABASE_URL:
            return {"status": False, "error": "DATABASE_URL not found"}

        start_time = time.time()
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            # Basic connectivity check
            connection.execute(text("SELECT 1"))

            # Get database size and stats
            db_stats = connection.execute(text("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as db_size,
                       (SELECT count(*) FROM pg_stat_activity) as active_connections
            """))
            stats = db_stats.fetchone()

            response_time = time.time() - start_time

            return {
                "status": True,
                "response_time": f"{response_time:.2f}s",
                "db_size": stats[0],
                "active_connections": stats[1]
            }
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        return {"status": False, "error": str(e)}

def check_page_imports(page_path: str) -> Dict:
    """Check if a page's imports are working and measure load time"""
    try:
        start_time = time.time()
        spec = importlib.util.spec_from_file_location(
            os.path.basename(page_path).replace(".py", ""),
            page_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        load_time = time.time() - start_time

        return {
            "status": True,
            "load_time": f"{load_time:.2f}s"
        }
    except Exception as e:
        logging.error(f"Import error in {page_path}: {str(e)}")
        return {"status": False, "error": str(e)}

def get_system_metrics() -> Dict:
    """Get system resource usage metrics"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')

        return {
            "status": True,
            "memory_used": f"{memory.percent}%",
            "cpu_usage": f"{cpu_percent}%",
            "disk_usage": f"{disk.percent}%"
        }
    except Exception as e:
        return {"status": False, "error": str(e)}

def check_background_processes() -> Dict:
    """Check status of background processes"""
    try:
        # Check for search worker process
        search_pid_file = '.search_pid'
        search_process_running = os.path.exists(search_pid_file)
        if search_process_running:
            with open(search_pid_file, 'r') as f:
                pid = int(f.read().strip())
                search_process_running = psutil.pid_exists(pid)

        return {
            "status": True,
            "search_worker": "Running" if search_process_running else "Stopped"
        }
    except Exception as e:
        return {"status": False, "error": str(e)}

def run_health_checks() -> Dict:
    """Run all health checks and return results"""
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "database": check_database_connection(),
        "system": get_system_metrics(),
        "background_processes": check_background_processes(),
        "pages": {}
    }

    # Check all pages
    for page in sorted(glob.glob("pages/[0-9]*.py")):
        results["pages"][os.path.basename(page)] = check_page_imports(page)

    return results

def display_health_dashboard():
    """Display health check results in Streamlit"""
    st.title("🏥 System Health Dashboard")

    # Add auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.empty()
        st.rerun()

    if st.button("Run Health Check") or auto_refresh:
        results = run_health_checks()

        # Display system metrics
        st.subheader("System Resources")
        system = results["system"]
        if system["status"]:
            cols = st.columns(3)
            cols[0].metric("Memory Usage", system["memory_used"])
            cols[1].metric("CPU Usage", system["cpu_usage"])
            cols[2].metric("Disk Usage", system["disk_usage"])
        else:
            st.error(f"System metrics error: {system.get('error', 'Unknown error')}")

        # Display database status
        st.subheader("Database Status")
        db_status = results["database"]
        if db_status["status"]:
            cols = st.columns(3)
            cols[0].metric("Response Time", db_status["response_time"])
            cols[1].metric("Database Size", db_status["db_size"])
            cols[2].metric("Active Connections", str(db_status["active_connections"]))
        else:
            st.error(f"Database connection failed: {db_status.get('error', 'Unknown error')}")

        # Display background processes
        st.subheader("Background Processes")
        bg_processes = results["background_processes"]
        if bg_processes["status"]:
            st.info(f"Search Worker: {bg_processes['search_worker']}")
        else:
            st.error(f"Process check error: {bg_processes.get('error', 'Unknown error')}")

        # Display page status
        st.subheader("Page Status")
        for page, status in results["pages"].items():
            if status["status"]:
                st.success(f"{page}: OK (Load time: {status['load_time']})")
            else:
                st.error(f"{page}: Failed - {status.get('error', 'Unknown error')}")

        # Display timestamp
        st.caption(f"Last checked: {results['timestamp']}")

if __name__ == "__main__":
    display_health_dashboard()