from RELEASE_streamlit_app import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_search():
    DATABASE_URL = "postgresql://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    
    with Session() as session:
        try:
            # Get active campaign and template
            campaign = session.query(Campaign).filter_by(id=1).first()
            template = session.query(EmailTemplate).first()
            
            if not template:
                logger.error("No email template found!")
                return
                
            logger.info(f"Using template: {template.template_name}")
            
            # Get search terms
            search_terms = session.query(SearchTerm).filter_by(campaign_id=1).all()
            logger.info(f"Found {len(search_terms)} search terms")
            
            total_leads = 0
            for term in search_terms:
                logger.info(f"Processing term: {term.term}")
                try:
                    # Use the original manual_search function with email sending enabled
                    results = manual_search(
                        session=session,
                        search_term=term.term,
                        num_results=10,
                        enable_email_sending=True,
                        from_email="sami@samihalawa.com",
                        email_template=f"{template.id}:{template.template_name}",
                        reply_to="sami@samihalawa.com"
                    )
                    
                    leads_found = results['total_leads']
                    total_leads += leads_found
                    logger.info(f"Found {leads_found} leads for term: {term.term}")
                    
                    if results['results']:
                        for lead in results['results']:
                            logger.info(f"Lead found: {lead['Email']} from {lead['URL']}")
                            
                except Exception as e:
                    logger.error(f"Error processing term {term.term}: {str(e)}")
                    continue
            
            logger.info(f"Search completed. Total leads found: {total_leads}")
            
        except Exception as e:
            logger.error(f"Database error: {str(e)}")

if __name__ == "__main__":
    run_search() 