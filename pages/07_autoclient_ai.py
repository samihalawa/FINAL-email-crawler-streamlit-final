import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from models import KnowledgeBase, EmailTemplate, Campaign

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def generate_ai_template(prompt, kb, current_template=None):
    """Generate AI template with proper error handling"""
    try:
        # Placeholder for AI template generation logic
        # This would integrate with OpenAI or similar service
        return {
            "subject": "Sample AI Generated Subject",
            "body": "Sample AI Generated Body"
        }
    except Exception as e:
        st.error(f"Error generating template: {str(e)}")
        return None

def main():
    st.title("🤖 AutoclientAI")

    try:
        with SessionLocal() as session:
            # Get knowledge base info
            project_id = st.session_state.get('current_project_id', 1)
            kb = session.query(KnowledgeBase).filter_by(project_id=project_id).first()

            if not kb:
                st.warning("Please set up your Knowledge Base first")
            else:
                st.subheader("Email Template Generation")

                with st.form("generate_template"):
                    prompt = st.text_area(
                        "Describe what kind of email template you want to generate", 
                        help="Provide details about the tone, purpose, and key points to include."
                    )

                    campaign_id = st.session_state.get('current_campaign_id', 1)
                    templates = session.query(EmailTemplate).filter_by(campaign_id=campaign_id).all()
                    template_to_adjust = st.selectbox(
                        "Select template to adjust (optional)",
                        options=[(None, "Create New Template")] + 
                                [(t.id, t.template_name) for t in templates],
                        format_func=lambda x: x[1]
                    )

                    if st.form_submit_button("Generate Template"):
                        if 'current_project_id' not in st.session_state:
                            st.error("Please select a project first")
                        else:
                            current_template = None
                            if template_to_adjust[0]:
                                template = session.query(EmailTemplate).get(template_to_adjust[0])
                                if template:
                                    current_template = {
                                        "subject": template.subject,
                                        "body": template.body_content
                                    }

                            result = generate_ai_template(prompt, kb, current_template)

                            if result:
                                st.subheader("Generated Template")
                                st.text("Subject:")
                                st.write(result["subject"])
                                st.text("Body:")
                                st.markdown(result["body"], unsafe_allow_html=True)

                                if st.button("Save as New Template"):
                                    new_template = EmailTemplate(
                                        template_name=f"AI Generated - {result['subject'][:30]}",
                                        subject=result["subject"],
                                        body_content=result["body"],
                                        campaign_id=campaign_id,
                                        is_ai_customizable=True
                                    )
                                    session.add(new_template)
                                    session.commit()
                                    st.success("Template saved successfully!")

    except Exception as e:
        st.error(f"Error in AutoclientAI: {str(e)}")

if __name__ == "__main__":
    main()