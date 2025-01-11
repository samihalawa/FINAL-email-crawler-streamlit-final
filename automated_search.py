import logging
from streamlit_app import db_session, manual_search, get_active_campaign_id, get_active_project_id, EmailTemplate, EmailSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('automated_search.log')
    ]
)

class LogContainer:
    def markdown(self, text, unsafe_allow_html=False):
        logging.info(text.replace('<br>', '\n'))

def main():
    with db_session() as session:
        # Get email settings
        email_settings = session.query(EmailSettings).first()
        if not email_settings:
            logging.error("No email settings found. Please configure email settings first.")
            return

        # Get email template
        template = session.query(EmailTemplate).first()
        if not template:
            logging.error("No email template found. Please create an email template first.")
            return

        search_terms = [
            "vacation rental property manager barcelona",
            "property management company barcelona",
            "airbnb management barcelona",
            "holiday rental management barcelona",
            "short term rental manager barcelona"
        ]

        log_container = LogContainer()
        results = manual_search(
            session=session,
            terms=search_terms,
            num_results=10,
            ignore_previously_fetched=True,
            optimize_english=False,
            optimize_spanish=False,
            shuffle_keywords_option=False,
            language='ES',
            enable_email_sending=True,
            log_container=log_container,
            from_email=email_settings.email,
            reply_to=email_settings.email,
            email_template=f"{template.id}:{template.template_name}"
        )

        logging.info(f"Total leads found: {results['total_leads']}")
        for result in results['results']:
            logging.info(f"Lead: {result['Email']} from {result['URL']}")

if __name__ == "__main__":
    main() 