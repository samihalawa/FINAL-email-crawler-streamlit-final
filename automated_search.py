import logging, uuid, json, argparse, os, signal
from datetime import datetime
from models import EmailTemplate, EmailSettings, AutomationLog, SearchTerm
from sqlalchemy import create_engine, func, distinct
from sqlalchemy.orm import sessionmaker, aliased
from app import manual_search, SessionLocal

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
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Save PID
    save_pid()

    try:
        session = SessionLocal()
        try:
            # Create a default automation log if no ID is provided
            automation_log = AutomationLog(
                start_time=datetime.utcnow(),
                status='running',
                logs=[{
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': 'info',
                    'message': 'Search worker started'
                }]
            )
            session.add(automation_log)
            session.commit()

            logging.info(f"Created new automation log with ID: {automation_log.id}")

            # Run until stopped
            while True:
                if automation_log.status != 'running':
                    logging.info("Search worker stopped")
                    break

                # Sleep to prevent high CPU usage
                signal.pause()

        except Exception as e:
            logging.error(f"Error in search process: {str(e)}")
            if automation_log:
                automation_log.status = 'error'
                automation_log.logs.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': 'error',
                    'message': f"Error: {str(e)}"
                })
                session.commit()
        finally:
            session.close()
    finally:
        cleanup_pid()

if __name__ == "__main__":
    main()

class LogContainer:
    def __init__(self, session, automation_log_id):
        self.session = session
        self.automation_log_id = automation_log_id

    def write(self, text):
        print(text)
        logging.info(text)
        # Log to database
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'info',
            'message': text
        }
        automation_log = self.session.query(AutomationLog).get(self.automation_log_id)
        if automation_log:
            if not automation_log.logs:
                automation_log.logs = []
            automation_log.logs.append(log_entry)
            self.session.commit()

class SearchState:
    def __init__(self, automation_log_id):
        self.automation_log_id = automation_log_id
        self.current_term_index = 0
        self.current_url_index = {}  # term -> last processed URL index
        self.processed_urls = set()

    @classmethod
    def load_from_db(cls, session, automation_log_id):
        automation_log = session.query(AutomationLog).get(automation_log_id)
        state = cls(automation_log_id)
        if automation_log and automation_log.logs:
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
        if automation_log:
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