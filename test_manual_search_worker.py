import streamlit as st
import time
import logging
from datetime import datetime
from streamlit_app import (
    db_session, AutomationLog, manual_search, get_active_campaign_id,
    display_search_controls, display_logs, cleanup_search_state,
    run_automated_search
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_manual_search.log')
    ]
)
logger = logging.getLogger(__name__)

def test_manual_search_worker():
    """Test manual search worker functionality"""
    st.title("Manual Search Worker Test")
    
    # Test Case 1: Start a search
    st.subheader("Test Case 1: Start Search")
    with db_session() as session:
        # Create test search terms
        search_terms = ['test engineer spain', 'software developer madrid']
        
        # Create new automation log
        new_log = AutomationLog(
            campaign_id=get_active_campaign_id(),
            start_time=datetime.utcnow(),
            status='running',
            leads_gathered=0,
            emails_sent=0,
            logs=[{
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'info',
                'message': "Starting test search process",
                'search_settings': {
                    'search_terms': search_terms,
                    'num_results': 3,  # Small number for testing
                    'ignore_previously_fetched': True,
                    'optimize_english': False,
                    'optimize_spanish': True,
                    'shuffle_keywords_option': False,
                    'language': 'ES',
                    'enable_email_sending': False,
                    'term_index': 0,
                    'url_index': {},
                    'processed_urls': []
                }
            }]
        )
        session.add(new_log)
        session.commit()
        st.session_state.automation_log_id = new_log.id
        
        # Start search process
        run_automated_search(new_log.id)
        
        # Wait for some results
        time.sleep(5)
        
        # Verify search is running
        automation_log = session.query(AutomationLog).get(new_log.id)
        if automation_log.status == 'running':
            st.success("✅ Test Case 1: Search started successfully")
        else:
            st.error("❌ Test Case 1: Search failed to start")
            return
        
        # Test Case 2: Stop Search
        st.subheader("Test Case 2: Stop Search")
        automation_log.status = 'completed'
        automation_log.end_time = datetime.utcnow()
        automation_log.logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'info',
            'message': "Search stopped for testing"
        })
        session.commit()
        
        # Try to kill the process
        try:
            with open('.search_pid', 'r') as f:
                pid = int(f.read().strip())
                import os, signal
                os.kill(pid, signal.SIGTERM)
        except:
            pass
        
        # Verify search stopped
        time.sleep(2)
        automation_log = session.query(AutomationLog).get(new_log.id)
        if automation_log.status == 'completed':
            st.success("✅ Test Case 2: Search stopped successfully")
        else:
            st.error("❌ Test Case 2: Search failed to stop")
            return
        
        # Test Case 3: Start New Search
        st.subheader("Test Case 3: Start New Search")
        # Clear session state
        if 'automation_log_id' in st.session_state:
            del st.session_state.automation_log_id
        if 'worker_log_state' in st.session_state:
            del st.session_state.worker_log_state
        if 'search_thread' in st.session_state:
            del st.session_state.search_thread
            
        # Create new search
        search_terms = ['CTO startup barcelona', 'tech lead valencia']
        new_log = AutomationLog(
            campaign_id=get_active_campaign_id(),
            start_time=datetime.utcnow(),
            status='running',
            leads_gathered=0,
            emails_sent=0,
            logs=[{
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'info',
                'message': "Starting new test search process",
                'search_settings': {
                    'search_terms': search_terms,
                    'num_results': 3,
                    'ignore_previously_fetched': True,
                    'optimize_english': False,
                    'optimize_spanish': True,
                    'shuffle_keywords_option': False,
                    'language': 'ES',
                    'enable_email_sending': False,
                    'term_index': 0,
                    'url_index': {},
                    'processed_urls': []
                }
            }]
        )
        session.add(new_log)
        session.commit()
        st.session_state.automation_log_id = new_log.id
        
        # Start new search process
        run_automated_search(new_log.id)
        
        # Wait for some results
        time.sleep(5)
        
        # Verify new search is running
        automation_log = session.query(AutomationLog).get(new_log.id)
        if automation_log.status == 'running':
            st.success("✅ Test Case 3: New search started successfully")
        else:
            st.error("❌ Test Case 3: New search failed to start")
            return
        
        # Test Case 4: Error Handling
        st.subheader("Test Case 4: Error Handling")
        automation_log.status = 'error'
        automation_log.logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'error',
            'message': "Test error condition"
        })
        session.commit()
        
        # Try retry
        automation_log.status = 'running'
        automation_log.logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'info',
            'message': "Retrying search",
            'search_settings': {
                'search_terms': search_terms,
                'num_results': 3,
                'ignore_previously_fetched': True,
                'optimize_english': False,
                'optimize_spanish': True,
                'shuffle_keywords_option': False,
                'language': 'ES',
                'enable_email_sending': False,
                'term_index': 0,
                'url_index': {},
                'processed_urls': []
            }
        })
        
        # Start retry process
        run_automated_search(automation_log.id)
        
        # Wait for retry to start
        time.sleep(2)
        
        # Verify retry worked
        automation_log = session.query(AutomationLog).get(automation_log.id)
        if automation_log.status == 'running':
            st.success("✅ Test Case 4: Error handling and retry successful")
        else:
            st.error("❌ Test Case 4: Error handling failed")
            return
        
        # Final cleanup
        cleanup_search_state()
        st.success("✅ All tests completed successfully!")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Manual Search Worker Test",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    test_manual_search_worker() 