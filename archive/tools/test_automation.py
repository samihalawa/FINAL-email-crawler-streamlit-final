import streamlit as st
from RELEASE_streamlit_app import db_session, ai_automation_loop, init_db
import time

st.title("Automation Test")

# Initialize session state
if 'automation_status' not in st.session_state:
    st.session_state.automation_status = False
if 'automation_logs' not in st.session_state:
    st.session_state.automation_logs = []

# Create containers for output
log_container = st.empty()
leads_container = st.empty()
status_container = st.empty()

# Create a log container for output
class LogContainer:
    def info(self, msg): 
        st.write(f'INFO: {msg}')
        print(f'INFO: {msg}')
    
    def warning(self, msg): 
        st.warning(msg)
        print(f'WARNING: {msg}')
    
    def error(self, msg): 
        st.error(msg)
        print(f'ERROR: {msg}')

# Create a leads container for output
class LeadsContainer:
    def text_area(self, title, content, height=None):
        st.text_area(title, content, height=height)
        print(f'{title}:\n{content}')

# Initialize DB
init_db()

# Simulate clicking Start Automation button
if st.button("Start Automation"):
    st.session_state.automation_status = True
    st.session_state.automation_logs = []
    
    status_container.write("Automation Status: Running")
    
    # Start automation loop
    with db_session() as session:
        ai_automation_loop(session, LogContainer(), LeadsContainer())
else:
    status_container.write("Click Start Automation to begin") 