
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from models import Project, Campaign, KnowledgeBase, Base

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("🚀 Projects & Campaigns")

try:
    with SessionLocal() as session:
        # Project Management
        st.subheader("Projects")
        
        # Display existing projects
        projects = session.query(
            Project,
            func.count(Campaign.id).label('campaign_count')
        ).outerjoin(Campaign).group_by(Project.id).all()
        
        if projects:
            project_data = []
            for project, campaign_count in projects:
                kb = session.query(KnowledgeBase).filter_by(project_id=project.id).first()
                project_data.append({
                    'ID': project.id,
                    'Name': project.project_name,
                    'Campaigns': campaign_count,
                    'Has KB': 'Yes' if kb else 'No',
                    'Created': project.created_at.strftime('%Y-%m-%d %H:%M')
                })
            
            df = pd.DataFrame(project_data)
            st.dataframe(df, use_container_width=True)
            
            # Project selection
            selected_project_name = st.selectbox(
                "Select Active Project",
                options=[p.project_name for p, _ in projects],
                index=0 if 'current_project_id' not in st.session_state else 
                      next((i for i, (p, _) in enumerate(projects) 
                           if p.id == st.session_state.get('current_project_id')), 0)
            )
            
            selected_project = next((p for p, _ in projects if p.project_name == selected_project_name), None)
            if selected_project:
                st.session_state['current_project_id'] = selected_project.id
                st.success(f"Active project set to: {selected_project_name}")
        
        # Add new project
        with st.form("new_project"):
            st.subheader("Add New Project")
            project_name = st.text_input("Project Name")
            
            if st.form_submit_button("Create Project"):
                if project_name:
                    try:
                        new_project = Project(project_name=project_name)
                        session.add(new_project)
                        session.commit()
                        st.success("Project created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating project: {str(e)}")
                        session.rollback()
                else:
                    st.warning("Please enter a project name")
        
        # Campaign Management for selected project
        if 'current_project_id' in st.session_state:
            st.subheader("Project Campaigns")
            campaigns = session.query(Campaign).filter_by(project_id=st.session_state['current_project_id']).all()
            
            if campaigns:
                campaign_data = [{
                    'ID': c.id,
                    'Name': c.campaign_name,
                    'Type': c.campaign_type,
                    'Auto Send': '✓' if c.auto_send else '✗',
                    'AI Custom': '✓' if c.ai_customization else '✗',
                    'Created': c.created_at.strftime('%Y-%m-%d %H:%M')
                } for c in campaigns]
                st.dataframe(pd.DataFrame(campaign_data), use_container_width=True)
            
            # Add new campaign
            with st.form("new_campaign"):
                st.subheader("Add New Campaign")
                campaign_name = st.text_input("Campaign Name")
                campaign_type = st.selectbox("Campaign Type", ["Email", "LinkedIn", "Other"])
                auto_send = st.checkbox("Enable Auto Send")
                ai_customization = st.checkbox("Enable AI Customization")
                
                if st.form_submit_button("Create Campaign"):
                    if campaign_name:
                        try:
                            new_campaign = Campaign(
                                campaign_name=campaign_name,
                                campaign_type=campaign_type,
                                project_id=st.session_state['current_project_id'],
                                auto_send=auto_send,
                                ai_customization=ai_customization
                            )
                            session.add(new_campaign)
                            session.commit()
                            st.success("Campaign created successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error creating campaign: {str(e)}")
                            session.rollback()
                    else:
                        st.warning("Please enter a campaign name")

except Exception as e:
    st.error(f"Error in Projects & Campaigns: {str(e)}")
