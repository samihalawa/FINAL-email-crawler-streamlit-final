import logging, uuid, json, argparse
from datetime import datetime
from streamlit_app import db_session, manual_search, get_active_campaign_id, get_active_project_id, EmailTemplate, EmailSettings, AutomationLog, SearchTerm
from typing import Optional, Dict

# Configure logging (remove file handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class LogContainer:
    def __init__(self, session, automation_log_id):
        self.session = session
        self.automation_log_id = automation_log_id

    def markdown(self, text, unsafe_allow_html=False):
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
        if automation_log.settings and 'search_state' in automation_log.settings:
            saved_state = automation_log.settings['search_state']
            state.current_term_index = saved_state.get('term_index', 0)
            state.current_url_index = saved_state.get('url_index', {})
            state.processed_urls = set(saved_state.get('processed_urls', []))
        return state

    def save_to_db(self, session):
        automation_log = session.query(AutomationLog).get(self.automation_log_id)
        if not automation_log.settings:
            automation_log.settings = {}
        automation_log.settings['search_state'] = {
            'term_index': self.current_term_index,
            'url_index': self.current_url_index,
            'processed_urls': list(self.processed_urls)
        }
        session.commit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("automation_log_id", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with db_session() as session:
        automation_log = session.query(AutomationLog).get(args.automation_log_id)
        if not automation_log:
            logging.error(f"No automation log found with ID {args.automation_log_id}")
            return

        search_state = SearchState.load_from_db(session, args.automation_log_id)
        
        try:
            search_terms = automation_log.search_terms
            for i, term in enumerate(search_terms[search_state.current_term_index:], search_state.current_term_index):
                if automation_log.status != 'running':
                    logging.info("Search paused or stopped")
                    break

                search_state.current_term_index = i
                start_url_index = search_state.current_url_index.get(term, 0)
                
                for url_index, url in enumerate(google_search(term, automation_log.settings['num_results']), start_url_index):
                    if url in search_state.processed_urls:
                        continue

                    if automation_log.status != 'running':
                        search_state.current_url_index[term] = url_index
                        search_state.save_to_db(session)
                        logging.info("Search paused or stopped")
                        break

                    # Process URL and save leads
                    process_url(session, url, automation_log, search_state)
                    search_state.processed_urls.add(url)
                    search_state.save_to_db(session)

            if automation_log.status == 'running':
                automation_log.status = 'completed'
                automation_log.end_time = datetime.utcnow()
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

def process_url(session, url, automation_log, search_state):
    # Your existing URL processing code...
    pass

if __name__ == "__main__":
    main() 