import streamlit as st
from streamlit_app import *
import time
from datetime import datetime, timedelta
import os

def initialize_ai_settings():
    with db_session() as session:
        existing_settings = session.query(Settings).filter_by(name='openai_settings', setting_type='ai').first()
        if not existing_settings:
            ai_settings = Settings(
                name='openai_settings',
                setting_type='ai',
                value={
<<<<<<< HEAD
                    'model': os.getenv('OPENAI_MODEL'),
                    'api_base': os.getenv('OPENAI_API_BASE')
=======
                    'model': 'Qwen/Qwen2.5-Coder-32B-Instruct',
                    'api_base': 'https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1'
>>>>>>> d95791f (Remove secret from test_automation.py)
                }
            )
            session.add(ai_settings)
            session.commit()
            print("AI settings added to database")
        else:
            print("AI settings already exist in database")

def test_manual_search():
    print("Testing Manual Search Page...")
    with db_session() as session:
        search_terms = ["software engineer spain", "tech lead barcelona", "developer madrid"]
        st.session_state.current_search_terms = search_terms
        email_settings = session.query(EmailSettings).first()
        template = session.query(EmailTemplate).first()
        if not email_settings or not template:
            print("Warning: No email settings or template found")
            return
        results = manual_search(
            session=session,
            terms=search_terms,
            num_results=10,
            ignore_previously_fetched=True,
            optimize_english=False,
            optimize_spanish=True,
            shuffle_keywords_option=True,
            language='ES',
            enable_email_sending=True,
            log_container=st.empty(),
            from_email=email_settings.email,
            reply_to=email_settings.email,
            email_template=f"{template.id}: {template.template_name}"
        )
        print(f"Manual Search Results: {len(results.get('results', []))} leads found")

def test_automation():
    print("Testing Automation...")
    automation_settings = {
        'max_leads_per_cycle': 50,
        'results_per_search': 10,
        'cycle_interval': 1,
        'auto_email': True,
        'optimize_english': False,
        'optimize_spanish': True,
        'language': 'ES'
    }
    st.session_state.update({
        'automation_status': True,
        'automation_logs': [],
        'total_leads_found': 0,
        'total_emails_sent': 0,
        'automation_start_time': datetime.utcnow(),
        'automation_active': True
    })
    with db_session() as session:
        cycle_results = run_automation_cycle(
            session=session,
            settings=automation_settings,
            log_container=st.empty()
        )
        print(f"Automation Results:")
        print(f"- New Leads: {len(cycle_results.get('new_leads', []))}")
        print(f"- Emails Sent: {cycle_results.get('emails_sent', 0)}")
        print(f"- Search Terms Used: {len(cycle_results.get('search_terms_used', []))}")
        st.session_state.automation_status = False
        st.session_state.automation_active = False

def main():
    print("Starting tests...")
    initialize_settings()
    initialize_ai_settings()
    with db_session() as session:
        if not session.query(Project).first():
            project = Project(project_name="Test Project")
            session.add(project)
            session.commit()
            campaign = Campaign(
                campaign_name="Test Campaign",
                project_id=project.id
            )
            session.add(campaign)
            session.commit()
            st.session_state.active_project_id = project.id
            st.session_state.active_campaign_id = campaign.id
    test_manual_search()
    time.sleep(2)
    test_automation()
    print("Tests completed!")

if __name__ == "__main__":
    main() 