import logging, uuid, json, argparse, os, signal
from datetime import datetime
from models import EmailTemplate, EmailSettings, AutomationLog, SearchTerm
from app import db_session, manual_search, get_active_campaign_id, get_active_project_id
from typing import Optional, Dict
from sqlalchemy import create_engine, func, distinct #Added missing imports
from sqlalchemy.orm import sessionmaker, aliased #Added missing imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def save_pid():
    """Save current process ID to file"""
    pid = os.getpid()
    logging.info(f"Saving PID: {pid}")
    print(f"Saving PID: {pid}")

    with open('.search_pid', 'w') as f:
        f.write(str(os.getpid()))

def cleanup_pid():
    """Clean up PID file"""
    try:
        os.remove('.search_pid')
    except:
        logging.error("Error removing .search_pid file")

def signal_handler(signum, frame):
    """Handle termination signals"""
    logging.info(f"Signal {signum} received. Cleaning up...")
    print(f"Signal {signum} received. Cleaning up...")
    cleanup_pid()
    exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("automation_log_id", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Save PID
    save_pid()

    try:
        with db_session() as session:
            automation_log = session.query(AutomationLog).get(args.automation_log_id)
            if automation_log is None:
                logging.error("Automation log is None")
                return

            if not automation_log:
                logging.error(f"No automation log found with ID {args.automation_log_id}")
                return

            search_state = SearchState.load_from_db(session, args.automation_log_id)
            log_container = LogContainer(session, args.automation_log_id)

            try:
                # Get search settings from logs
                search_settings = {}
                if automation_log.logs:
                    for log in reversed(automation_log.logs):
                        if isinstance(log, dict) and 'search_settings' in log:
                            search_settings = log['search_settings']
                            break

                search_terms = search_settings.get('search_terms', [])
                if not search_terms:
                    logging.warning("No search terms found in search settings.")
                    return

                for i, term in enumerate(search_terms[search_state.current_term_index:], search_state.current_term_index):
                    if automation_log.status != 'running':
                        logging.info("Search paused or stopped")
                        break

                    search_state.current_term_index = i
                    log_container.markdown(f"Processing search term: {term}")

                    # Perform the actual search
                    results = manual_search(
                        session=session,
                        terms=[term],
                        num_results=search_settings.get('num_results', 10),
                        ignore_previously_fetched=search_settings.get('ignore_previously_fetched', True),
                        optimize_english=search_settings.get('optimize_english', False),
                        optimize_spanish=search_settings.get('optimize_spanish', False),
                        shuffle_keywords_option=search_settings.get('shuffle_keywords_option', False),
                        language=search_settings.get('language', 'ES'),
                        enable_email_sending=search_settings.get('enable_email_sending', False),
                        log_container=log_container,
                        from_email=search_settings.get('from_email'),
                        reply_to=search_settings.get('reply_to'),
                        email_template=search_settings.get('email_template')
                    )

                    # Update metrics
                    if results and 'results' in results:
                        automation_log.leads_gathered = (automation_log.leads_gathered or 0) + len(results['results'])
                        session.commit()

                    # Save state after each term
                    search_state.save_to_db(session)

                if automation_log.status == 'running':
                    automation_log.status = 'completed'
                    automation_log.end_time = datetime.utcnow()
                    automation_log.logs.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'level': 'success',
                        'message': 'Search completed successfully'
                    })
                    session.commit()

            except Exception as e:
                logging.error(f"Error in search process: {str(e)}")
                automation_log.status = 'error'
                automation_log.logs.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': 'error',
                    'message': f"Error: {str(e)}"
                })
                session.commit()
    finally:
        cleanup_pid()

if __name__ == "__main__":
    main() 

class LogContainer:
    def __init__(self, session, automation_log_id):
        self.session = session
        self.automation_log_id = automation_log_id

    def markdown(self, text, unsafe_allow_html=False):
        print(text.replace('<br>', '\n'))

        logging.info(text.replace('<br>', '\n'))
        # Log to database
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'info',
            'message': text
        }
        automation_log = self.session.query(AutomationLog).get(self.automation_log_id)
        if automation_log:
            if automation_log.logs:
                automation_log.logs.append(log_entry)
            else:
                automation_log.logs = [log_entry]
            self.session.commit()

class SearchState:
    def __init__(self, automation_log_id: int):
        self.automation_log_id = automation_log_id
        self.current_term_index: int = 0
        self.current_url_index: Dict[str, int] = {}  # term -> last processed URL index
        self.processed_urls: set = set()

    @classmethod
    def load_from_db(cls, session, automation_log_id: int) -> 'SearchState':
        automation_log = session.query(AutomationLog).get(automation_log_id)
        state = cls(automation_log_id)
        if automation_log.logs:
            # Look for the most recent search_settings entry
            for log in reversed(automation_log.logs):
                if isinstance(log, dict) and 'search_settings' in log:
                    saved_state = log['search_settings']
                    state.current_term_index = saved_state.get('term_index', 0)
                    state.current_url_index = saved_state.get('url_index', {})
                    state.processed_urls = set(saved_state.get('processed_urls', []))
                    break
        return state

    def save_to_db(self, session):
        automation_log = session.query(AutomationLog).get(self.automation_log_id)
        if not automation_log.logs:
            automation_log.logs = []
        automation_log.logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'info',
            'message': 'Search state updated',
            'search_settings': {
                'term_index': self.current_term_index,
                'url_index': self.current_url_index,
                'processed_urls': list(self.processed_urls)
            }
        })
        session.commit()