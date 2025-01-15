import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
from models import AutomationLog, Campaign

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("⚙️ Automation Control")

try:
    with SessionLocal() as session:
        # Display Active Automations
        st.subheader("Active Automations")
        active_logs = session.query(AutomationLog).filter(
            AutomationLog.status.in_(['running', 'paused'])
        ).all()

        if active_logs:
            active_data = []
            for log in active_logs:
                campaign = session.get(Campaign, log.campaign_id) if log.campaign_id else None
                active_data.append({
                    'ID': log.id,
                    'Campaign': campaign.campaign_name if campaign else 'Unknown',
                    'Status': log.status,
                    'Start Time': log.start_time,
                    'Leads Gathered': log.leads_gathered,
                    'Emails Sent': log.emails_sent
                })
            st.dataframe(pd.DataFrame(active_data))
        else:
            st.info("No active automations")

        # Automation History
        st.subheader("Automation History")
        completed_logs = session.query(AutomationLog).filter(
            AutomationLog.status.in_(['completed', 'error'])
        ).order_by(AutomationLog.start_time.desc()).limit(10).all()

        if completed_logs:
            history_data = []
            for log in completed_logs:
                campaign = session.get(Campaign, log.campaign_id)
                history_data.append({
                    'ID': log.id,
                    'Campaign': campaign.campaign_name if campaign else 'Unknown',
                    'Status': log.status,
                    'Start Time': log.start_time,
                    'End Time': log.end_time,
                    'Leads Gathered': log.leads_gathered,
                    'Emails Sent': log.emails_sent
                })
            st.dataframe(pd.DataFrame(history_data))
        else:
            st.info("No automation history")

        # View Automation Logs
        st.subheader("View Automation Logs")
        log_id = st.number_input("Enter Automation Log ID", min_value=1, value=1)
        if st.button("View Logs"):
            automation_log = session.query(AutomationLog).get(log_id)
            if automation_log and automation_log.logs:
                for log_entry in automation_log.logs:
                    if isinstance(log_entry, dict):
                        st.text(f"[{log_entry.get('timestamp', 'Unknown Time')}] {log_entry.get('message', '')}")
            else:
                st.info("No logs found for this automation")

except Exception as e:
    st.error(f"Error in Automation Control: {str(e)}")