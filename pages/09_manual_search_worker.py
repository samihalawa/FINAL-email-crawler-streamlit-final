import streamlit as st
import subprocess
import signal
import os
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
from models import AutomationLog, Campaign, SearchTerm
import time

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("⚙️ Manual Search Worker")

def display_logs(automation_log_id):
    placeholder = st.empty()
    while True:
        with SessionLocal() as session:
            log = session.query(AutomationLog).get(automation_log_id)
            if log:
                with placeholder.container():
                    st.write(f"Status: {log.status}")
                    st.write(f"Leads gathered: {log.leads_gathered}")
                    if log.logs:
                        for entry in log.logs:
                            if isinstance(entry, dict):
                                st.write(f"[{entry.get('timestamp', '')}] {entry.get('message', '')}")
                if log.status in ['completed', 'failed', 'stopped']:
                    break
        time.sleep(1)

try:
    with SessionLocal() as session:
        campaign_id = st.session_state.get('current_campaign_id', 1)
        campaign = session.get(Campaign, campaign_id)

        if not campaign:
            st.warning("Please select a campaign first")
        else:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Start Search"):
                    try:
                        search_terms = session.query(SearchTerm).filter_by(campaign_id=campaign_id).all()
                        if not search_terms:
                            st.warning("No search terms found for this campaign")
                        else:
                            new_log = AutomationLog(
                                campaign_id=campaign_id,
                                start_time=datetime.utcnow(),
                                status='running',
                                logs=[]
                            )
                            session.add(new_log)
                            session.commit()

                            process = subprocess.Popen(
                                ['python', 'automated_search.py', str(new_log.id)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )

                            with open('.search_pid', 'w') as f:
                                f.write(str(process.pid))

                            st.success(f"Search started! Automation Log ID: {new_log.id}")
                            display_logs(new_log.id)

                    except Exception as e:
                        st.error(f"Error starting search: {str(e)}")

            with col2:
                if st.button("Stop Search"):
                    if os.path.exists('.search_pid'):
                        with open('.search_pid', 'r') as f:
                            pid = int(f.read().strip())
                        try:
                            os.kill(pid, signal.SIGTERM)
                            st.success("Search process stopped")
                            os.remove('.search_pid')
                        except ProcessLookupError:
                            st.warning("Search process already stopped")
                        except Exception as e:
                            st.error(f"Error stopping process: {str(e)}")
                    else:
                        st.warning("No running search process found")

except Exception as e:
    st.error(f"Error in Manual Search Worker: {str(e)}")