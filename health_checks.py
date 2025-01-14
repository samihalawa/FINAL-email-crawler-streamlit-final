import streamlit as st
import logging
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

def check_database_connection() -> Dict[str, bool]:
    """Check database connectivity"""
    try:
        load_dotenv()
        DATABASE_URL = os.getenv('DATABASE_URL')
        if not DATABASE_URL:
            return {"status": False, "error": "DATABASE_URL not found"}
        
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return {"status": True}
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        return {"status": False, "error": str(e)}

def get_all_pages() -> List[str]:
    """Get all Streamlit pages"""
    pages = glob.glob("pages/[0-9]*.py")
    return sorted(pages)

def check_page_imports(page_path: str) -> Dict[str, bool]:
    """Check if a page's imports are working"""
    try:
        spec = importlib.util.spec_from_file_location(
            os.path.basename(page_path).replace(".py", ""),
            page_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {"status": True}
    except Exception as e:
        logging.error(f"Import error in {page_path}: {str(e)}")
        return {"status": False, "error": str(e)}

def run_health_checks() -> Dict:
    """Run all health checks and return results"""
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "database": check_database_connection(),
        "pages": {}
    }
    
    # Check all pages
    for page in get_all_pages():
        results["pages"][os.path.basename(page)] = check_page_imports(page)
    
    return results

def display_health_dashboard():
    """Display health check results in Streamlit"""
    st.title("🏥 System Health Dashboard")
    
    if st.button("Run Health Check"):
        results = run_health_checks()
        
        # Display database status
        st.subheader("Database Status")
        db_status = results["database"]
        if db_status["status"]:
            st.success("Database connection: OK")
        else:
            st.error(f"Database connection failed: {db_status.get('error', 'Unknown error')}")
        
        # Display page status
        st.subheader("Page Status")
        for page, status in results["pages"].items():
            if status["status"]:
                st.success(f"{page}: OK")
            else:
                st.error(f"{page}: Failed - {status.get('error', 'Unknown error')}")
        
        # Display timestamp
        st.text(f"Last checked: {results['timestamp']}")

if __name__ == "__main__":
    display_health_dashboard()
