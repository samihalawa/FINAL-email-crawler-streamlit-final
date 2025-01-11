import logging
from streamlit_app import manual_search, db_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_manual_search():
    logger.info("Starting manual search test...")
    with db_session() as session:
        logger.info("Created database session")
        try:
            results = manual_search(
                session=session,
                search_terms=['software engineer'],
                num_results=2,
                ignore_previously_fetched=True,
                optimize_english=False,
                optimize_spanish=False,
                shuffle_keywords=False,
                language='ES',
                enable_email_sending=True
            )
            logger.info(f"Search completed. Results: {results}")
        except Exception as e:
            logger.error(f"Error during search: {str(e)}", exc_info=True)

if __name__ == '__main__':
    test_manual_search() 