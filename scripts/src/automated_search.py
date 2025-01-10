from datetime import datetime
import logging
import streamlit as st
from streamlit_app import (
    db_session,
    EmailTemplate,
    EmailSettings,
    save_lead,
    send_email_ses,
    get_domain_from_url,
    manual_search,
    is_valid_email,
    extract_emails_from_html
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('automated_search.log')
    ]
)

class MockContainer:
    def markdown(self, text, unsafe_allow_html=False):
        logging.info(text.replace('<br>', '\n'))

def main():
    # Create mock container for logging
    log_container = MockContainer()
    
    with db_session() as session:
        # Get all active email settings
        settings = session.query(EmailSettings).all()
        if not settings:
            logging.error("No email settings found")
            return
        
        # Get all templates
        templates = session.query(EmailTemplate).all()
        if not templates:
            logging.error("No email templates found")
            return
        
        # Select first available settings and template
        email_setting = settings[0]
        template = templates[0]
        
        # Store needed values
        from_email = email_setting.email
        domains_processed = set()
    
        # Search terms focused on vacation rentals and property management
        search_terms = [
            # Spanish terms
            "gestor pisos turisticos barcelona contacto",
            "administrador apartamentos airbnb madrid correo",
            "empresa gestion alquiler vacacional malaga",
            "agencia alquiler turistico valencia email",
            "propietario apartamento turistico ibiza",
            "anfitrion airbnb sevilla contacto",
            # English terms
            "vacation rental manager spain contact",
            "holiday apartment owner mallorca email",
            "property management company costa del sol",
            "airbnb property manager barcelona"
        ]
        
        total_leads = 0
        for term in search_terms:
            logging.info(f"\n{'='*50}\nSearching for term: {term}")
            
            try:
                # Use the manual_search function with mock container
                search_results = manual_search(
                    session=session,
                    terms=[term],
                    num_results=5,  # Reduced number for testing
                    ignore_previously_fetched=True,
                    optimize_spanish='spain' in term.lower() or any(city in term.lower() for city in ['barcelona', 'madrid', 'malaga', 'valencia', 'sevilla', 'ibiza', 'mallorca']),
                    language='es' if any(word in term.lower() for word in ['correo', 'contacto', 'gestor', 'propietario', 'empresa']) else 'en',
                    enable_email_sending=True,
                    from_email=from_email,
                    reply_to=from_email,
                    email_template=f"{template.id}: {template.template_name}",
                    log_container=log_container
                )
                
                if search_results and search_results.get('results'):
                    leads_found = len(search_results['results'])
                    total_leads += leads_found
                    logging.info(f"Found {leads_found} leads for term: {term}")
                    
                    # Log each found lead with more details
                    for lead in search_results['results']:
                        domain = get_domain_from_url(lead['URL'])
                        if domain not in domains_processed:
                            domains_processed.add(domain)
                            logging.info(f"New lead found:")
                            logging.info(f"  Email: {lead['Email']}")
                            logging.info(f"  URL: {lead['URL']}")
                            logging.info(f"  Domain: {domain}")
                            if 'Company' in lead:
                                logging.info(f"  Company: {lead['Company']}")
                else:
                    logging.warning(f"No leads found for term: {term}")
                    
            except Exception as e:
                logging.error(f"Search error ({term}): {str(e)}")
                continue
            
            # Add a small delay between searches
            import time
            time.sleep(2)
        
        logging.info(f"\n{'='*50}")
        logging.info(f"Search completed. Total leads found: {total_leads}")
        logging.info(f"Unique domains processed: {len(domains_processed)}")

if __name__ == "__main__":
    main() 
    main() 