import streamlit as st
import time
import logging
from datetime import datetime
from streamlit_app import (
    db_session, AutomationLog, manual_search, get_active_campaign_id,
    display_search_controls, display_logs, cleanup_search_state
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('automated_search.log')
    ]
)
logger = logging.getLogger(__name__)

def run_background_search(session, automation_log):
    """Run the search process in background"""
    logger.info("Starting background search process")
    
    # Store the automation log ID for new sessions
    automation_log_id = automation_log.id
    
    try:
        # Define search terms for Spain
        search_terms = [
            'CTO startup españa',
            'director tecnología madrid', 
            'tech lead barcelona',
            'VP engineering valencia',
            'head of engineering malaga'
        ]
        
        # Create a placeholder for logs
        log_placeholder = st.empty()
        
        for term in search_terms:
            # Create a new session for this iteration
            with db_session() as thread_session:
                try:
                    # Get fresh automation log instance
                    automation_log = thread_session.query(AutomationLog).get(automation_log_id)
                    if not automation_log or automation_log.status != 'running':
                        logger.info("Search stopped")
                        return
                    
                    logger.info(f"Searching term: {term}")
                    results = manual_search(
                        session=thread_session,
                        terms=[term],
                        num_results=5,
                        ignore_previously_fetched=True,
                        optimize_english=False,
                        optimize_spanish=True,
                        shuffle_keywords_option=True,
                        language='ES',
                        enable_email_sending=False,
                        log_container=log_placeholder
                    )
                    
                    # Update automation log with results
                    if results.get('results'):
                        automation_log.leads_gathered = (automation_log.leads_gathered or 0) + len(results['results'])
                        automation_log.logs.extend([{
                            'timestamp': datetime.utcnow().isoformat(),
                            'level': 'success',
                            'message': f"Found lead: {result['Email']} ({result.get('Company', 'Unknown')})"
                        } for result in results['results']])
                        
                        # Check if target reached
                        if automation_log.leads_gathered >= st.session_state.current_target_leads:
                            automation_log.status = 'completed'
                            automation_log.end_time = datetime.utcnow()
                            automation_log.logs.append({
                                'timestamp': datetime.utcnow().isoformat(),
                                'level': 'success',
                                'message': f'Search completed - found {automation_log.leads_gathered} leads'
                            })
                            thread_session.commit()
                            return
                    
                    # Log any errors
                    if results.get('errors'):
                        for error in results['errors']:
                            logger.error(f"Search error: {error}")
                            automation_log.logs.append({
                                'timestamp': datetime.utcnow().isoformat(),
                                'level': 'error',
                                'message': f"Error: {error}"
                            })
                    
                    thread_session.commit()
                    
                except Exception as e:
                    logger.error(f"Error processing term {term}: {str(e)}")
                    try:
                        automation_log.logs.append({
                            'timestamp': datetime.utcnow().isoformat(),
                            'level': 'error',
                            'message': f"Error processing term {term}: {str(e)}"
                        })
                        thread_session.commit()
                    except:
                        pass
                    continue
                
            time.sleep(2)  # Small delay between searches
            
        # Final status update
        with db_session() as thread_session:
            try:
                automation_log = thread_session.query(AutomationLog).get(automation_log_id)
                if automation_log and automation_log.status == 'running':
                    automation_log.status = 'completed'
                    automation_log.end_time = datetime.utcnow()
                    automation_log.logs.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'level': 'success',
                        'message': 'Search completed successfully'
                    })
                    thread_session.commit()
            except Exception as e:
                logger.error(f"Error updating final status: {str(e)}")
            
    except Exception as e:
        logger.error(f"Critical error in search process: {str(e)}")
        with db_session() as thread_session:
            try:
                automation_log = thread_session.query(AutomationLog).get(automation_log_id)
                if automation_log:
                    automation_log.status = 'error'
                    automation_log.logs.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'level': 'error',
                        'message': f"Critical error: {str(e)}"
                    })
                    thread_session.commit()
            except Exception as commit_error:
                logger.error(f"Error logging critical error: {str(commit_error)}")

def test_manual_search_worker():
    st.title("Manual Search Worker Test")
    logger.info("Starting Manual Search Worker test page")

    # Initialize session state
    if 'worker_log_state' not in st.session_state:
        st.session_state.worker_log_state = {
            'buffer': [],
            'last_count': 0,
            'last_update': time.time(),
            'update_counter': 0,
            'auto_scroll': True
        }

    if 'current_target_leads' not in st.session_state:
        st.session_state.current_target_leads = 3
        
    if 'search_thread' not in st.session_state:
        st.session_state.search_thread = None

    with db_session() as session:
        # Check for active automation
        if 'automation_log_id' in st.session_state:
            automation_log = session.query(AutomationLog).get(st.session_state.automation_log_id)
            if automation_log:
                logger.info(f"Found active automation log: {automation_log.id}")
                
                # Check if thread is still alive
                if st.session_state.search_thread and not st.session_state.search_thread.is_alive():
                    logger.info("Search thread completed")
                    st.session_state.search_thread = None
                
                display_search_controls(automation_log)
                
                # Add log controls
                col1, col2 = st.columns([3, 1])
                with col1:
                    log_filter = st.selectbox(
                        "Filter logs",
                        ["all", "error", "success", "email", "search"],
                        key="worker_log_filter"
                    )
                with col2:
                    st.checkbox("Auto-scroll", value=True, key="worker_auto_scroll")

                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Leads Found", automation_log.leads_gathered or 0)
                col2.metric("Target Leads", st.session_state.current_target_leads)
                col3.metric("Success Rate", f"{(automation_log.leads_gathered/st.session_state.current_target_leads*100 if automation_log.leads_gathered else 0):.1f}%")

                # Update logs with improved display
                display_logs(
                    log_container=st.empty(),
                    logs=automation_log.logs,
                    selected_filter=st.session_state.get('worker_log_filter', 'all'),
                    auto_scroll=st.session_state.get('worker_auto_scroll', True)
                )

                # Check if we need to restart with new target
                if automation_log.status == 'completed':
                    if automation_log.leads_gathered >= st.session_state.current_target_leads:
                        if st.session_state.current_target_leads == 3:
                            if st.button("Start New Search (5 Leads)"):
                                st.session_state.current_target_leads = 5
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
                                        'message': f"Starting search for {st.session_state.current_target_leads} leads"
                                    }]
                                )
                                session.add(new_log)
                                session.commit()
                                st.session_state.automation_log_id = new_log.id
                                
                                # Start the search process
                                import threading
                                thread = threading.Thread(target=run_background_search, args=(session, new_log))
                                thread.daemon = True
                                thread.start()
                                st.session_state.search_thread = thread
                                logger.info("Started new search process for 5 leads")
                                st.rerun()
        else:
            # Create new automation log
            logger.info("Creating new automation log")
            new_log = AutomationLog(
                campaign_id=get_active_campaign_id(),
                start_time=datetime.utcnow(),
                status='running',
                leads_gathered=0,
                emails_sent=0,
                logs=[{
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': 'info',
                    'message': f"Starting search for {st.session_state.current_target_leads} leads"
                }]
            )
            session.add(new_log)
            session.commit()
            st.session_state.automation_log_id = new_log.id
            
            # Start the search process
            import threading
            thread = threading.Thread(target=run_background_search, args=(session, new_log))
            thread.daemon = True
            thread.start()
            st.session_state.search_thread = thread
            logger.info("Started background search process")
            st.rerun()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Manual Search Worker Test",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    test_manual_search_worker() 