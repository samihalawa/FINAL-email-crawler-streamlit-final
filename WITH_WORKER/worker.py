import os
import logging
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import random
import json
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import re
from fake_useragent import UserAgent
from googlesearch import search as google_search

# Database setup
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def update_task_log(session, task_id, message, level='info'):
    """Update task logs"""
    task = session.query(WorkerTask).get(task_id)
    if task:
        if not task.logs:
            task.logs = []
        task.logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'message': message,
            'level': level
        })
        session.commit()

def process_search_task(task_id):
    """Process a search task"""
    with SessionLocal() as session:
        task = session.query(WorkerTask).get(task_id)
        if not task:
            return
        
        try:
            # Update task status
            task.status = 'running'
            task.started_at = datetime.utcnow()
            session.commit()
            
            params = task.params
            update_task_log(session, task_id, "Starting search process")
            
            # Initialize variables
            total_leads = 0
            results = []
            
            # Process each search term
            search_terms = params.get('search_terms', [])
            if params.get('shuffle_keywords', False):
                random.shuffle(search_terms)
            
            for term in search_terms:
                try:
                    # Perform search
                    update_task_log(session, task_id, f"Searching for: {term}")
                    
                    for url in google_search(term, params.get('num_results', 10)):
                        try:
                            # Process URL
                            domain = urlparse(url).netloc
                            
                            # Skip if already processed and ignore_previously_fetched is True
                            if params.get('ignore_previously_fetched', True):
                                existing = session.query(LeadSource).filter_by(domain=domain).first()
                                if existing:
                                    update_task_log(session, task_id, f"Skipping previously fetched domain: {domain}", 'warning')
                                    continue
                            
                            # Fetch and process page
                            response = requests.get(url, headers={'User-Agent': UserAgent().random}, timeout=10)
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # Extract emails
                            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', response.text)
                            
                            if emails:
                                update_task_log(session, task_id, f"Found {len(emails)} email(s) on {url}", 'success')
                                
                                for email in emails:
                                    # Save lead
                                    lead = save_lead(session, email, url=url)
                                    if lead:
                                        total_leads += 1
                                        results.append({
                                            'Email': email,
                                            'URL': url,
                                            'Domain': domain,
                                            'Search Term': term
                                        })
                                        
                                        # Send email if enabled
                                        if params.get('enable_email_sending', False):
                                            try:
                                                send_email(
                                                    session,
                                                    params.get('from_email'),
                                                    email,
                                                    params.get('email_template'),
                                                    params.get('reply_to')
                                                )
                                                update_task_log(session, task_id, f"Sent email to {email}", 'email_sent')
                                            except Exception as e:
                                                update_task_log(session, task_id, f"Failed to send email to {email}: {str(e)}", 'error')
                            
                        except Exception as e:
                            update_task_log(session, task_id, f"Error processing URL {url}: {str(e)}", 'error')
                            continue
                
                except Exception as e:
                    update_task_log(session, task_id, f"Error processing term {term}: {str(e)}", 'error')
                    continue
            
            # Update task with results
            task.status = 'completed'
            task.completed_at = datetime.utcnow()
            task.results = {
                'total_leads': total_leads,
                'results': results
            }
            session.commit()
            
            update_task_log(session, task_id, f"Search completed. Found {total_leads} leads.")
            
        except Exception as e:
            task.status = 'failed'
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            session.commit()
            update_task_log(session, task_id, f"Task failed: {str(e)}", 'error')

def process_task(task_id):
    """Process any type of task"""
    with SessionLocal() as session:
        task = session.query(WorkerTask).get(task_id)
        if not task:
            return
        
        if task.task_type == 'search':
            process_search_task(task_id)
        # Add other task types here as needed 