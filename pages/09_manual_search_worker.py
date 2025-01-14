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

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("⚙️ Manual Search Worker")

try:
    with SessionLocal() as session:
        campaign_id = st.session_state.get('current_campaign_id', 1)
        campaign = session.query(Campaign).get(campaign_id)

        if not campaign:
            st.warning("Please select a campaign first")
        else:
            # Create new automation log if starting search
            if st.button("Start Search"):
                try:
                    # Get search terms for campaign
                    search_terms = session.query(SearchTerm).filter_by(campaign_id=campaign_id).all()
                    if not search_terms:
                        st.warning("No search terms found for this campaign")
                    else:
                        # Create automation log
                        new_log = AutomationLog(
                            campaign_id=campaign_id,
                            start_time=datetime.utcnow(),
                            status='running',
                            logs=[]
                        )
                        session.add(new_log)
                        session.commit()

                        # Start search process
                        search_settings = {
                            'search_terms': [term.term for term in search_terms],
                            'num_results': 10,
                            'ignore_previously_fetched': True,
                            'optimize_english': False,
                            'optimize_spanish': False,
                            'shuffle_keywords_option': False,
                            'language': 'ES',
                            'enable_email_sending': False
                        }

                        new_log.logs.append({
                            'timestamp': datetime.utcnow().isoformat(),
                            'level': 'info',
                            'message': 'Starting search process',
                            'search_settings': search_settings
                        })
                        session.commit()

                        try:
                            process = subprocess.Popen(
                                ['python', 'automated_search.py', str(new_log.id)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            st.success(f"Search started! Automation Log ID: {new_log.id}")

                            # Save process ID
                            with open('.search_pid', 'w') as f:
                                f.write(str(process.pid))
                        except Exception as e:
                            new_log.status = 'failed'
                            new_log.logs.append({
                                'timestamp': datetime.utcnow().isoformat(),
                                'level': 'error',
                                'message': f'Failed to start search: {str(e)}'
                            })
                            session.commit()
                            st.error(f"Failed to start search: {str(e)}")
                except Exception as e:
                    st.error(f"Error starting search: {str(e)}")

            # Stop search if running
            if st.button("Stop Search"):
                try:
                    if os.path.exists('.search_pid'):
                        with open('.search_pid', 'r') as f:
                            pid = int(f.read().strip())
                        try:
                            os.kill(pid, signal.SIGTERM)
                            st.success("Search process stopped")
                        except ProcessLookupError:
                            st.warning("Search process already stopped")
                        os.remove('.search_pid')
                    else:
                        st.warning("No running search process found")
                except Exception as e:
                    st.error(f"Error stopping search: {str(e)}")

            # Display current automation log
            st.subheader("Current Automation Log")
            current_log = session.query(AutomationLog).filter_by(
                campaign_id=campaign_id,
                status='running'
            ).first()

            if current_log:
                st.write(f"Log ID: {current_log.id}")
                st.write(f"Status: {current_log.status}")
                st.write(f"Start Time: {current_log.start_time}")
                st.write(f"Leads Gathered: {current_log.leads_gathered}")

                if current_log.logs:
                    st.subheader("Search Logs")
                    for log in current_log.logs:
                        if isinstance(log, dict):
                            st.text(f"[{log.get('timestamp', 'Unknown Time')}] {log.get('message', '')}")

except Exception as e:
    st.error(f"Error in Manual Search Worker: {str(e)}")