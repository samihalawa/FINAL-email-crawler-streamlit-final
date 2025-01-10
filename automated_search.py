from datetime import datetime
from streamlit_app import (
    db_session,
    EmailTemplate,
    EmailSettings,
    save_lead,
    send_email_ses,
    get_domain_from_url
)
from googlesearch import search

def log_update(message, level='info'):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [{level.upper()}] {message}")

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
        from_email = settings.email
        domains_processed = set()
    
        # Search terms
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
                # Use googlesearch-python package directly
                for url in search(term, num=10, lang='es'):
                    domain = get_domain_from_url(url)
                    if domain in domains_processed:
                        log_update(f"Skipping domain {domain}: Already processed", 'warning')
                        continue
                        
                    domains_processed.add(domain)
                    
                    try:
                        lead = save_lead(session, url=url, search_term=term)
                        if lead and lead.email:
                            log_update(f"Found lead: {lead.email}", 'success')
                            
                            try:
                                response, _ = send_email_ses(
                                    session,
                                    from_email,
                                    lead.email,
                                    template.subject,
                                    template.body_content,
                                    reply_to=from_email
                                )
                                log_update(f"Email sent to: {lead.email}", 'success')
                            except Exception as e:
                                log_update(f"Email error ({lead.email}): {str(e)}", 'error')
                    
                    except Exception as e:
                        log_update(f"URL error ({url}): {str(e)}", 'error')
                        continue
                    
            except Exception as e:
                log_update(f"Search error ({term}): {str(e)}", 'error')
                continue 