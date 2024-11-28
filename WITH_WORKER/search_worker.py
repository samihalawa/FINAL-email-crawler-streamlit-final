import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import random
import json

def run_search_process(process_id, database_url, search_terms, num_results, ignore_previously_fetched, 
                      optimize_english, optimize_spanish, shuffle_keywords, language,
                      enable_email_sending, from_email=None, reply_to=None, email_template=None):
    """Run the search process in the background"""
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(bind=engine)

    def update_log(message, level='info'):
        with SessionLocal() as session:
            process = session.query(SearchProcess).get(process_id)
            if process:
                if not process.logs:
                    process.logs = []
                process.logs.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'message': message,
                    'level': level
                })
                session.commit()

    try:
        update_log("Starting search process", "info")
        
        # Initialize variables
        total_leads_found = 0
        total_emails_sent = 0
        results = []
        
        # Shuffle search terms if requested
        if shuffle_keywords:
            random.shuffle(search_terms)
        
        # Process each search term
        for i, term in enumerate(search_terms):
            update_log(f"Processing search term {i+1}/{len(search_terms)}: {term}", "info")
            
            # Optimize search term if requested
            if optimize_english or optimize_spanish:
                optimized_term = optimize_search_term(term, language)
                update_log(f"Optimized term: {optimized_term}", "info")
                term = optimized_term
            
            # Perform the search
            try:
                term_results = perform_search(term, num_results, ignore_previously_fetched)
                results.extend(term_results['results'])
                total_leads_found += len(term_results['results'])
                
                update_log(f"Found {len(term_results['results'])} leads for term: {term}", "success")
                
                # Send emails if enabled
                if enable_email_sending and term_results['results']:
                    for lead in term_results['results']:
                        try:
                            send_email(
                                from_email=from_email,
                                to_email=lead['Email'],
                                reply_to=reply_to,
                                template_id=email_template,
                                lead_data=lead
                            )
                            total_emails_sent += 1
                            update_log(f"Sent email to {lead['Email']}", "email_sent")
                        except Exception as e:
                            update_log(f"Failed to send email to {lead['Email']}: {str(e)}", "error")
            
            except Exception as e:
                update_log(f"Error processing term {term}: {str(e)}", "error")
        
        # Update process status
        with SessionLocal() as session:
            process = session.query(SearchProcess).get(process_id)
            if process:
                process.status = 'completed'
                process.end_time = datetime.utcnow()
                process.results = results
                session.commit()
        
        update_log(f"Search process completed. Found {total_leads_found} leads, sent {total_emails_sent} emails.", "success")
        
    except Exception as e:
        update_log(f"Fatal error in search process: {str(e)}", "error")
        # Update process status to failed
        with SessionLocal() as session:
            process = session.query(SearchProcess).get(process_id)
            if process:
                process.status = 'failed'
                process.end_time = datetime.utcnow()
                process.error = str(e)
                session.commit() 