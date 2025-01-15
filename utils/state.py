import streamlit as st
from models import Project, Campaign
from utils.db import SessionLocal

def init_session_state():
    """Initialize or ensure default session state"""
    if 'current_project_id' not in st.session_state:
        st.session_state.current_project_id = 1
    if 'current_campaign_id' not in st.session_state:
        st.session_state.current_campaign_id = 1

def ensure_defaults():
    """Ensure default project and campaign exist"""
    with SessionLocal() as session:
        # Check/create default project
        default_project = session.query(Project).get(1)
        if not default_project:
            default_project = Project(
                id=1,
                project_name="Default Project"
            )
            session.add(default_project)
            
            # Create default campaign
            default_campaign = Campaign(
                id=1,
                campaign_name="Default Campaign",
                project_id=1,
                campaign_type="Email"
            )
            session.add(default_campaign)
            session.commit() 