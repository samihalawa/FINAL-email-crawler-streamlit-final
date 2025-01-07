import os
from RELEASE_streamlit_app import db_session, get_knowledge_base_info, manual_search, save_lead, bulk_send_emails
from RELEASE_streamlit_app import get_active_project_id, SearchTerm, EmailTemplate

# Set database URL from environment
os.environ['DATABASE_URL'] = 'postgresql://postgres:postgres@db.qvtqvqwwkxrwqokvhbzz.supabase.co:5432/postgres'

print("Starting automation simulation...")

with db_session() as session:
    # Get knowledge base info
    kb_info = get_knowledge_base_info(session, get_active_project_id())
    if not kb_info:
        print("No knowledge base found")
        exit()
    
    print("Found knowledge base")
    
    # Get search terms
    base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
    if not base_terms:
        print("No search terms found")
        exit()
    
    print(f"Found {len(base_terms)} search terms")
    
    # Process each term
    for term in base_terms:
        print(f"\nProcessing term: {term}")
        results = manual_search(session, [term], 10, ignore_previously_fetched=True)
        
        if results and 'results' in results:
            new_leads = []
            for res in results['results']:
                lead = save_lead(session, res['Email'], url=res.get('URL'))
                if lead:
                    new_leads.append((lead.id, lead.email))
            
            if new_leads:
                template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                if template:
                    from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                    reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                    print(f"Sending emails to {len(new_leads)} leads")
                    logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': email} for _, email in new_leads])
                    print(f"Sent {sent_count} emails") 