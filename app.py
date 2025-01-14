import streamlit as st
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    st.error("Database configuration missing!")
    st.stop()

# Initialize database connection
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Test connection
    with engine.connect() as conn:
        st.sidebar.success("Database connected!")
except Exception as e:
    st.error(f"Database connection failed: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Lead Management System",
    page_icon="🎯",
    layout="wide"
)

# Main navigation
st.title("Lead Management System")

# Welcome message
st.write("""
Welcome to the Lead Management System. Use the sidebar navigation to:
- Search for leads manually
- View and manage existing leads
- Configure search terms
- Adjust system settings
""")

# Display some basic stats
try:
    with SessionLocal() as session:
        from streamlit_app import Lead, Campaign, SearchTerm
        
        total_leads = session.query(Lead).count()
        total_campaigns = session.query(Campaign).count()
        total_search_terms = session.query(SearchTerm).count()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Leads", total_leads)
        with col2:
            st.metric("Active Campaigns", total_campaigns)
        with col3:
            st.metric("Search Terms", total_search_terms)
except Exception as e:
    st.error(f"Error loading statistics: {str(e)}")
