import streamlit as st
import psutil
import time
from datetime import datetime
import importlib.machinery
import glob
import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from health_checks import (
    check_database_connection,
    check_page_imports,
    get_system_metrics,
    check_background_processes,
    run_health_checks
)

def main():
    """Display health check results in Streamlit"""
    st.title("🏥 System Health Dashboard")

    # Add auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False, key="health_refresh")
    if auto_refresh:
        st.empty()
        st.rerun()

    if st.button("Run Health Check", key="run_health_check") or auto_refresh:
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
    main()
