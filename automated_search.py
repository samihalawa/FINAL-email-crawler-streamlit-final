from datetime import datetime
from streamlit_app import (
    db_session,
    EmailTemplate,
    EmailSettings,
    google_search,
    save_lead,
    send_email_ses,
    get_domain_from_url
)

# Global variables
log_messages = []
all_results = []
domains_processed = set()

def log_update(message, level='info'):
    """Log a message and add it to the log messages list"""
    log_messages.append(f"[{level.upper()}] {message}")
    print(f"[{level.upper()}] {message}")

if __name__ == "__main__":
    with db_session() as session:
        # Get template and email settings
        template = session.query(EmailTemplate).filter(
            EmailTemplate.subject.ilike('%telefonillo%')
        ).first()
        if not template:
            print("No template found with 'telefonillo' in subject")
            exit(1)
        
        settings = session.query(EmailSettings).filter_by(is_active=True).first()
        if not settings:
            print("No active email settings found")
            exit(1)
        
        # Store needed values
        template_id = template.id
        template_subject = template.subject
        template_body = template.body_content
        from_email = settings.email
    
        # Airbnb related search terms
        search_terms = [
            "airbnb host barcelona email",
            "propietario apartamento turistico madrid correo",
            "alquiler vacacional malaga contacto",
            "anfitrion airbnb valencia email",
            "gestor pisos turisticos sevilla",
            "administrador apartamentos airbnb barcelona",
            "alquiler temporada costa brava propietario",
            "vacation rental owner mallorca email",
            "holiday apartment manager ibiza contact"
        ]
        
        for term in search_terms:
            log_update(f"Searching for term: {term}")
            
            try:
                urls = google_search(term, num_results=10, lang='ES')
                log_update(f"Found {len(urls)} URLs for term: {term}")
                
                for url in urls:
                    domain = get_domain_from_url(url)
                    if domain in domains_processed:
                        log_update(f"Skipping domain {domain}: Already processed", level='warning')
                        continue
                        
                    domains_processed.add(domain)
                    
                    try:
                        lead = save_lead(session, url=url, search_term=term)
                        if lead:
                            result = {
                                'Email': lead.email,
                                'Company': lead.company,
                                'Source': url,
                                'Term': term
                            }
                            all_results.append(result)
                            log_update(f"Found lead: {lead.email}")
                            
                            if template_id and from_email:
                                try:
                                    response, tracking_id = send_email_ses(
                                        session,
                                        from_email,
                                        lead.email,
                                        template_subject,
                                        template_body,
                                        reply_to=from_email
                                    )
                                    log_update(f"Email sent to: {lead.email}")
                                except Exception as e:
                                    log_update(f"Error sending email to {lead.email}: {str(e)}", level='error')
                    
                    except Exception as e:
                        log_update(f"Error processing URL {url}: {str(e)}", level='error')
                        continue
                    
            except Exception as e:
                log_update(f"Error processing term {term}: {str(e)}", level='error')
                continue
    
        # Print logs
        print("\nSearch Logs:")
        for log in log_messages:
            print(log)
            
        # Print results
        print("\nSearch Results:")
        for result in all_results:
            print(f"Found: {result.get('Email')} from {result.get('Company')}") 