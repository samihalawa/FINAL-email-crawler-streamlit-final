import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("✉️ Email Templates")

try:
    with SessionLocal() as session:
        from models import EmailTemplate
        
        # Display existing templates
        st.subheader("Existing Templates")
        templates = session.query(EmailTemplate).all()
        if templates:
            template_data = []
            for template in templates:
                template_data.append({
                    'ID': template.id,
                    'Name': template.template_name,
                    'Subject': template.subject,
                    'Language': template.language,
                    'AI Customizable': template.is_ai_customizable
                })
            df = pd.DataFrame(template_data)
            st.dataframe(df)
        else:
            st.info("No email templates found")
        
        # Add new template
        st.subheader("Add New Template")
        with st.form("new_template"):
            template_name = st.text_input("Template Name")
            subject = st.text_input("Subject")
            body = st.text_area("Body Content", height=300)
            language = st.selectbox("Language", ["ES", "EN"])
            is_ai_customizable = st.checkbox("AI Customizable")
            campaign_id = 1  # Default campaign ID
            
            if st.form_submit_button("Add Template"):
                if template_name and subject and body:
                    try:
                        new_template = EmailTemplate(
                            template_name=template_name,
                            subject=subject,
                            body_content=body,
                            language=language,
                            is_ai_customizable=is_ai_customizable,
                            campaign_id=campaign_id
                        )
                        session.add(new_template)
                        session.commit()
                        st.success("Template added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding template: {str(e)}")
                        session.rollback()
                else:
                    st.warning("Please fill in all required fields")

except Exception as e:
    st.error(f"Error loading email templates: {str(e)}")
