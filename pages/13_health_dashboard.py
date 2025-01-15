import streamlit as st
from health_checks import (
    display_health_dashboard, 
    check_database_connection,
    get_system_metrics,
    check_background_processes,
    run_health_checks
)

# Configure page
st.set_page_config(page_title="Health Dashboard", page_icon="🏥")

def main():
    """Main function to display the health dashboard"""
    display_health_dashboard()

if __name__ == "__main__":
    main()
import streamlit as st
from health_checks import display_health_dashboard

st.title("🏥 System Health Dashboard")
display_health_dashboard()
