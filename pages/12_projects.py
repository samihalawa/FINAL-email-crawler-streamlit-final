import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("🚀 Projects & Campaigns")

try:
    with SessionLocal() as session:
        from streamlit_app import Project, Campaign, KnowledgeBase
        
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
                    'Created': project.created_at
                })
            
            st.dataframe(pd.DataFrame(project_data))
        
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
        
        # Campaign Management
        st.subheader("Campaigns")
        
        # Select project for campaign
        selected_project = st.selectbox(
            "Select Project",
            options=[(p.id, p.project_name) for p, _ in projects],
            format_func=lambda x: x[1]
        )
        
        if selected_project:
            # Display project's campaigns
            campaigns = session.query(Campaign).filter_by(project_id=selected_project[0]).all()
            if campaigns:
                campaign_data = []
                for campaign in campaigns:
                    campaign_data.append({
                        'ID': campaign.id,
                        'Name': campaign.campaign_name,
                        'Type': campaign.campaign_type,
                        'Auto Send': campaign.auto_send,
                        'AI Customization': campaign.ai_customization,
                        'Created': campaign.created_at
                    })
                st.dataframe(pd.DataFrame(campaign_data))
            
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
                                project_id=selected_project[0],
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
