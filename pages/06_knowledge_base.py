
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from models import Project, KnowledgeBase
from datetime import datetime

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("📚 Knowledge Base")

try:
    with SessionLocal() as session:
        # Get or create knowledge base for current project
        project_id = st.session_state.get('current_project_id', 1)
        kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
        project = session.query(Project).get(project_id)
        
        if not project:
            st.warning("Please select a project first")
        else:
            if not kb:
                kb = KnowledgeBase(project_id=project_id)
                session.add(kb)
                session.commit()
            
            # Knowledge Base Form
            with st.form("knowledge_base"):
                col1, col2 = st.columns(2)
                
                with col1:
                    kb_name = st.text_input("Knowledge Base Name", value=kb.kb_name or "")
                    kb_bio = st.text_area("Bio", value=kb.kb_bio or "")
                    kb_values = st.text_area("Values", value=kb.kb_values or "")
                    
                    st.subheader("Contact Information")
                    contact_name = st.text_input("Contact Name", value=kb.contact_name or "")
                    contact_role = st.text_input("Contact Role", value=kb.contact_role or "")
                    contact_email = st.text_input("Contact Email", value=kb.contact_email or "")
                
                with col2:
                    st.subheader("Company Information")
                    company_description = st.text_area("Company Description", value=kb.company_description or "")
                    company_mission = st.text_area("Company Mission", value=kb.company_mission or "")
                    company_target_market = st.text_area("Target Market", value=kb.company_target_market or "")
                    company_other = st.text_area("Other Company Info", value=kb.company_other or "")
                    
                    st.subheader("Product Information")
                    product_name = st.text_input("Product Name", value=kb.product_name or "")
                    product_description = st.text_area("Product Description", value=kb.product_description or "")
                    product_target_customer = st.text_area("Target Customer", value=kb.product_target_customer or "")
                    product_other = st.text_area("Other Product Info", value=kb.product_other or "")
                
                example_email = st.text_area("Example Email", value=kb.example_email or "", height=200)
                other_context = st.text_area("Other Context", value=kb.other_context or "")
                
                if st.form_submit_button("Save Knowledge Base"):
                    try:
                        # Update knowledge base
                        kb.kb_name = kb_name
                        kb.kb_bio = kb_bio
                        kb.kb_values = kb_values
                        kb.contact_name = contact_name
                        kb.contact_role = contact_role
                        kb.contact_email = contact_email
                        kb.company_description = company_description
                        kb.company_mission = company_mission
                        kb.company_target_market = company_target_market
                        kb.company_other = company_other
                        kb.product_name = product_name
                        kb.product_description = product_description
                        kb.product_target_customer = product_target_customer
                        kb.product_other = product_other
                        kb.example_email = example_email
                        kb.other_context = other_context
                        kb.updated_at = datetime.utcnow()
                        
                        session.commit()
                        st.success("Knowledge base updated successfully!")
                    except Exception as e:
                        st.error(f"Error updating knowledge base: {str(e)}")
                        session.rollback()

except Exception as e:
    st.error(f"Error loading knowledge base: {str(e)}")
