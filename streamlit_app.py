import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime
from bs4 import BeautifulSoup
from googlesearch import search
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import openai
import logging
import json
import re
import os
import sqlite3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor
import time
import threading

# Configuration
AWS_ACCESS_KEY_ID = "AKIASO2XOMEGIVD422N7"
AWS_SECRET_ACCESS_KEY = "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"
REGION_NAME = "us-east-1"

openai.api_key = os.getenv("OPENAI_API_KEY", "default_openai_api_key")
openai.api_base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
openai_model = "gpt-3.5-turbo"

# SQLite configuration
sqlite_db_path = "autoclient.db"

# Ensure the database file exists
try:
    if not os.path.exists(sqlite_db_path):
        open(sqlite_db_path, 'w').close()
except IOError as e:
    logging.error(f"Failed to create database file: {e}")
    raise

# Initialize AWS SES client
try:
    ses_client = boto3.client('ses',
                              aws_access_key_id=AWS_ACCESS_KEY_ID,
                              aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                              region_name=REGION_NAME)
except (NoCredentialsError, PartialCredentialsError) as e:
    logging.error(f"AWS SES client initialization failed: {e}")
    raise

# SQLite connection
def get_db_connection():
    try:
        return sqlite3.connect(sqlite_db_path)
    except sqlite3.Error as e:
        logging.error(f"Database connection failed: {e}")
        raise

# HTTP session with retry strategy
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Setup logging
try:
    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
except IOError as e:
    print(f"Error setting up logging: {e}")
    raise

# Input validation functions
def validate_name(name):
    if not name or not name.strip():
        raise ValueError("Name cannot be empty or just whitespace")
    if len(name) > 100:
        raise ValueError("Name is too long (max 100 characters)")
    return name.strip()

def validate_email(email):
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        raise ValueError("Invalid email address")
    return email

def validate_campaign_type(campaign_type):
    valid_types = ["Email", "SMS"]
    if campaign_type not in valid_types:
        raise ValueError(f"Invalid campaign type. Must be one of {valid_types}")
    return campaign_type

def validate_id(id_value, id_type):
    try:
        id_int = int(id_value.split(':')[0] if ':' in str(id_value) else id_value)
        if id_int <= 0:
            raise ValueError
        return id_int
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid {id_type} ID")

def validate_status(status, valid_statuses):
    if status not in valid_statuses:
        raise ValueError(f"Invalid status. Must be one of {valid_statuses}")
    return status

def validate_num_results(num_results):
    if not isinstance(num_results, int) or num_results < 0:
        raise ValueError("Invalid number of results")
    return num_results

# Initialize database
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS campaigns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_name TEXT NOT NULL,
        project_id INTEGER,
        campaign_type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects (id)
    );

    CREATE TABLE IF NOT EXISTS message_templates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        template_name TEXT NOT NULL,
        subject TEXT,
        body_content TEXT NOT NULL,
        campaign_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
    );

    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        phone TEXT,
        first_name TEXT,
        last_name TEXT,
        company TEXT,
        job_title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS lead_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id INTEGER,
        search_term_id INTEGER,
        url TEXT,
        page_title TEXT,
        meta_description TEXT,
        http_status INTEGER,
        scrape_duration TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (lead_id) REFERENCES leads (id),
        FOREIGN KEY (search_term_id) REFERENCES search_terms (id)
    );

    CREATE TABLE IF NOT EXISTS campaign_leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER,
        lead_id INTEGER,
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (campaign_id) REFERENCES campaigns (id),
        FOREIGN KEY (lead_id) REFERENCES leads (id)
    );

    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER,
        lead_id INTEGER,
        template_id INTEGER,
        customized_subject TEXT,
        customized_content TEXT,
        sent_at TIMESTAMP,
        status TEXT DEFAULT 'pending',
        engagement_data TEXT,
        message_id TEXT,
        FOREIGN KEY (campaign_id) REFERENCES campaigns (id),
        FOREIGN KEY (lead_id) REFERENCES leads (id),
        FOREIGN KEY (template_id) REFERENCES message_templates (id)
    );

    CREATE TABLE IF NOT EXISTS search_terms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        term TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        processed_leads INTEGER DEFAULT 0,
        last_processed_at TIMESTAMP,
        campaign_id INTEGER,
        FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
    );
    ''')
    conn.commit()
    conn.close()
    logging.info("Database initialized successfully!")

# Call this at the start of your script
init_db()

def alter_messages_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if the column exists
        cursor.execute("PRAGMA table_info(messages)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'message_id' not in columns:
            cursor.execute("ALTER TABLE messages ADD COLUMN message_id TEXT")
            conn.commit()
            logging.info("Added message_id column to messages table")
        else:
            logging.info("message_id column already exists in messages table")
    except sqlite3.Error as e:
        logging.error(f"Error altering messages table: {e}")
    finally:
        conn.close()

alter_messages_table()

# Function to create a new project
def create_project(project_name):
    project_name = validate_name(project_name)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO projects (project_name) VALUES (?)", (project_name,))
    project_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return project_id

# Function to create a new campaign
def create_campaign(campaign_name, project_id, campaign_type):
    campaign_name = validate_name(campaign_name)
    project_id = validate_id(project_id, "project")
    campaign_type = validate_campaign_type(campaign_type)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO campaigns (campaign_name, project_id, campaign_type) VALUES (?, ?, ?)",
                   (campaign_name, project_id, campaign_type))
    campaign_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return campaign_id

# Function to create a new message template
def create_message_template(template_name, subject, body_content, campaign_id):
    template_name = validate_name(template_name)
    subject = validate_name(subject)
    campaign_id = validate_id(campaign_id, "campaign")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO message_templates (template_name, subject, body_content, campaign_id)
        VALUES (?, ?, ?, ?)
    """, (template_name, subject, body_content, campaign_id))
    template_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return template_id

# Function to add a new search term
def add_search_term(term, campaign_id):
    term = validate_name(term)
    campaign_id = validate_id(campaign_id, "campaign")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if the term already exists for this campaign
    cursor.execute("SELECT id FROM search_terms WHERE term = ? AND campaign_id = ?", (term, campaign_id))
    existing_term = cursor.fetchone()
    
    if existing_term:
        term_id = existing_term[0]
        print(f"Search term '{term}' already exists for this campaign.")
    else:
        cursor.execute("INSERT INTO search_terms (term, campaign_id) VALUES (?, ?)", (term, campaign_id))
        term_id = cursor.lastrowid
        print(f"New search term '{term}' added to the database.")
    
    conn.commit()
    conn.close()
    return term_id

# Function to fetch search terms
def fetch_search_terms(campaign_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if campaign_id:
        campaign_id = validate_id(campaign_id, "campaign")
        query = '''
        SELECT st.id, st.term, COUNT(DISTINCT ls.lead_id) as processed_leads, st.status 
        FROM search_terms st
        LEFT JOIN lead_sources ls ON st.id = ls.search_term_id
        WHERE st.campaign_id = ?
        GROUP BY st.id
        '''
        cursor.execute(query, (campaign_id,))
    else:
        query = '''
        SELECT st.id, st.term, COUNT(DISTINCT ls.lead_id) as processed_leads, st.status 
        FROM search_terms st
        LEFT JOIN lead_sources ls ON st.id = ls.search_term_id
        GROUP BY st.id
        '''
        cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["ID", "Search Term", "Leads Fetched", "Status"])

# Function to update search term status
def update_search_term_status(search_term_id, new_status, processed_leads=None):
    search_term_id = validate_id(search_term_id, "search term")
    new_status = validate_status(new_status, ["pending", "completed"])
    conn = get_db_connection()
    cursor = conn.cursor()
    if processed_leads is not None:
        processed_leads = validate_num_results(processed_leads)
        cursor.execute("""
            UPDATE search_terms
            SET status = ?, processed_leads = ?, last_processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_status, processed_leads, search_term_id))
    else:
        cursor.execute("""
            UPDATE search_terms
            SET status = ?, last_processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_status, search_term_id))
    conn.commit()
    conn.close()

# Update the save_lead function
def save_lead(email, phone, first_name, last_name, company, job_title):
    email = validate_email(email)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if the email already exists
    cursor.execute("SELECT id FROM leads WHERE email = ?", (email,))
    existing_lead = cursor.fetchone()
    
    if existing_lead:
        lead_id = existing_lead[0]
        logging.info(f"Existing lead found for email {email} with ID {lead_id}")
        # Update existing lead information if provided
        if any([phone, first_name, last_name, company, job_title]):
            cursor.execute("""
                UPDATE leads
                SET phone = COALESCE(?, phone),
                    first_name = COALESCE(?, first_name),
                    last_name = COALESCE(?, last_name),
                    company = COALESCE(?, company),
                    job_title = COALESCE(?, job_title)
                WHERE id = ?
            """, (phone, first_name, last_name, company, job_title, lead_id))
            logging.info(f"Updated existing lead with ID {lead_id}")
    else:
        cursor.execute("""
            INSERT INTO leads (email, phone, first_name, last_name, company, job_title)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (email, phone, first_name, last_name, company, job_title))
        lead_id = cursor.lastrowid
        logging.info(f"New lead created for email {email} with ID {lead_id}")
    
    conn.commit()
    conn.close()
    return lead_id

# Function to save lead source
def save_lead_source(lead_id, search_term_id, url, page_title, meta_description, http_status, scrape_duration):
    lead_id = validate_id(lead_id, "lead")
    search_term_id = validate_id(search_term_id, "search term")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO lead_sources (lead_id, search_term_id, url, page_title, meta_description, http_status, scrape_duration)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (lead_id, search_term_id, url, page_title, meta_description, http_status, scrape_duration))
    conn.commit()
    conn.close()

# Function to add a lead to a campaign
def add_lead_to_campaign(campaign_id, lead_id):
    campaign_id = validate_id(campaign_id, "campaign")
    lead_id = validate_id(lead_id, "lead")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO campaign_leads (campaign_id, lead_id) VALUES (?, ?)",
                   (campaign_id, lead_id))
    conn.commit()
    conn.close()

# Function to create a new message
def create_message(campaign_id, lead_id, template_id, customized_subject, customized_content):
    campaign_id = validate_id(campaign_id, "campaign")
    lead_id = validate_id(lead_id, "lead")
    template_id = validate_id(template_id, "template")
    customized_subject = validate_name(customized_subject)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (campaign_id, lead_id, template_id, customized_subject, customized_content)
        VALUES (?, ?, ?, ?, ?)
    """, (campaign_id, lead_id, template_id, customized_subject, customized_content))
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return message_id

# Function to update message status
def update_message_status(message_id, status, sent_at=None):
    message_id = validate_id(message_id, "message")
    status = validate_status(status, ["pending", "sent", "failed"])
    conn = get_db_connection()
    cursor = conn.cursor()
    if sent_at:
        cursor.execute("UPDATE messages SET status = ?, sent_at = ? WHERE id = ?",
                       (status, sent_at, message_id))
    else:
        cursor.execute("UPDATE messages SET status = ? WHERE id = ?",
                       (status, message_id))
    conn.commit()
    conn.close()

# Function to fetch message templates
def fetch_message_templates(campaign_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if campaign_id:
        campaign_id = validate_id(campaign_id, "campaign")
        cursor.execute('SELECT id, template_name FROM message_templates WHERE campaign_id = ?', (campaign_id,))
    else:
        cursor.execute('SELECT id, template_name FROM message_templates')
    rows = cursor.fetchall()
    conn.close()
    return [f"{row[0]}: {row[1]}" for row in rows]

# Function to fetch projects
def fetch_projects():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, project_name FROM projects')
    rows = cursor.fetchall()
    conn.close()
    return [f"{row[0]}: {row[1]}" for row in rows]

# Function to fetch campaigns
def fetch_campaigns():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, campaign_name FROM campaigns')
    campaigns = cursor.fetchall()
    conn.close()
    return [f"{campaign[0]}: {campaign[1]}" for campaign in campaigns]

# Updated bulk search function
async def bulk_search(num_results):
    total_leads = 0
    all_results = []

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, term FROM search_terms')
    search_terms = cursor.fetchall()
    conn.close()

    logs = []
    for term_id, term in search_terms:
        leads_found = 0
        try:
            search_urls = list(search(term, num=num_results, stop=num_results, pause=2))
            for url in search_urls:
                if leads_found >= num_results:
                    break
                try:
                    response = session.get(url, timeout=10)
                    response.encoding = 'utf-8'
                    soup = BeautifulSoup(response.text, 'html.parser')
                    emails = find_emails(response.text)
                    
                    for email in emails:
                        if is_valid_email(email):
                            lead_id = save_lead(email, None, None, None, None, None)
                            save_lead_source(lead_id, term_id, url, soup.title.string if soup.title else None, 
                                             soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else None, 
                                             response.status_code, str(response.elapsed))
                            all_results.append([email, url, term])
                            leads_found += 1
                            total_leads += 1
                            
                            if leads_found >= num_results:
                                break
                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")

            logs.append(f"Processed {leads_found} leads for term '{term}'")
        except Exception as e:
            logging.error(f"Error performing search for term '{term}': {e}")

        update_search_term_status(term_id, 'completed', leads_found)

    logs.append(f"Bulk search completed. Total new leads found: {total_leads}")
    return logs, all_results

# Function to get email preview
def get_email_preview(template_id, from_email, reply_to):
    template_id = validate_id(template_id, "template")
    from_email = validate_email(from_email)
    reply_to = validate_email(reply_to)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT subject, body_content FROM message_templates WHERE id = ?', (template_id,))
    template = cursor.fetchone()
    conn.close()

    if template:
        subject, body_content = template
        preview = f"""
        <h3>Email Preview</h3>
        <strong>Subject:</strong> {subject}<br>
        <strong>From:</strong> {from_email}<br>
        <strong>Reply-To:</strong> {reply_to}<br>
        <hr>
        <h4>Body:</h4>
        <iframe srcdoc="{body_content.replace('"', '&quot;')}" width="100%" height="600" style="border: 1px solid #ccc;"></iframe>
        """
        return preview
    else:
        return "<p>Template not found</p>"

# Add this function to fetch sent messages
def fetch_sent_messages():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT m.id, l.email, mt.template_name, m.customized_subject, m.sent_at, m.status
    FROM messages m
    JOIN leads l ON m.lead_id = l.id
    JOIN message_templates mt ON m.template_id = mt.id
    ORDER BY m.sent_at DESC
    ''')
    messages = cursor.fetchall()
    conn.close()
    return pd.DataFrame(messages, columns=["ID", "Email", "Template", "Subject", "Sent At", "Status"])

# Update the bulk_send function
async def bulk_send(template_id, from_email, reply_to):
    template_id = validate_id(template_id, "template")
    from_email = validate_email(from_email)
    reply_to = validate_email(reply_to)

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Fetch the template
    cursor.execute('SELECT template_name, subject, body_content FROM message_templates WHERE id = ?', (template_id,))
    template = cursor.fetchone()
    if not template:
        conn.close()
        return [f"Template not found"]

    template_name, subject, body_content = template

    # Fetch all leads that haven't been sent this template
    cursor.execute('''
        SELECT DISTINCT l.id, l.email 
        FROM leads l
        LEFT JOIN messages m ON l.id = m.lead_id AND m.template_id = ?
        WHERE m.id IS NULL OR m.status = 'failed'
    ''', (template_id,))
    leads = cursor.fetchall()
    
    conn.close()

    total_leads = len(leads)
    logs = []
    logs.append(f"Preparing to send emails to {total_leads} leads")
    logs.append(f"Template Name: {template_name}")
    logs.append(f"Template ID: {template_id}")
    logs.append(f"Subject: {subject}")
    logs.append(f"From: {from_email}")
    logs.append(f"Reply-To: {reply_to}")
    logs.append("---")

    if total_leads == 0:
        logs.append("No leads found to send emails to. Please check if there are leads in the database and if they have already been sent this template.")
        return logs

    total_sent = 0
    for index, (lead_id, email) in enumerate(leads, 1):
        try:
            response = ses_client.send_email(
                Source=from_email,
                Destination={
                    'ToAddresses': [email],
                },
                Message={
                    'Subject': {
                        'Data': subject,
                        'Charset': 'UTF-8'
                    },
                    'Body': {
                        'Html': {
                            'Data': body_content,
                            'Charset': 'UTF-8'
                        }
                    }
                },
                ReplyToAddresses=[reply_to]
            )
            message_id = response['MessageId']
            save_message(lead_id, template_id, 'sent', datetime.now(), subject, message_id)
            total_sent += 1
            logs.append(f"[{index}/{total_leads}] Sent email to {email} - Subject: {subject} - MessageId: {message_id}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logging.error(f"Failed to send email to {email}: {error_code} - {error_message}")
            save_message(lead_id, template_id, 'failed', None, subject)
            logs.append(f"[{index}/{total_leads}] Failed to send email to {email}: {error_code} - {error_message}")
        except Exception as e:
            logging.error(f"Unexpected error sending email to {email}: {str(e)}")
            save_message(lead_id, template_id, 'failed', None, subject)
            logs.append(f"[{index}/{total_leads}] Unexpected error sending email to {email}: {str(e)}")
        
        await asyncio.sleep(0.1)  # Small delay to allow UI updates

    logs.append("---")
    logs.append(f"Bulk send completed. Total emails sent: {total_sent}/{total_leads}")
    return logs

# Update the save_message function to include the subject
def save_message(lead_id, template_id, status, sent_at=None, subject=None, message_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if sent_at:
        cursor.execute("""
            INSERT INTO messages (lead_id, template_id, status, sent_at, customized_subject, message_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (lead_id, template_id, status, sent_at, subject, message_id))
    else:
        cursor.execute("""
            INSERT INTO messages (lead_id, template_id, status, customized_subject, message_id)
            VALUES (?, ?, ?, ?, ?)
        """, (lead_id, template_id, status, subject, message_id))
    conn.commit()
    conn.close()

# Function to sanitize HTML content
def sanitize_html(content):
    return re.sub('<[^<]+?>', '', content)

# Function to find valid emails in HTML text
def find_emails(html_text):
    email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[^@]+\.[A-Z|a-z]{2,7}\b')
    all_emails = set(email_regex.findall(html_text))
    valid_emails = {email for email in all_emails if is_valid_email(email)}

    unique_emails = {}
    for email in valid_emails:
        domain = email.split('@')[1]
        if domain not in unique_emails:
            unique_emails[domain] = email

    return set(unique_emails.values())

# Function to validate email address
def is_valid_email(email):
    invalid_patterns = [
        r'\.png', r'\.jpg', r'\.jpeg', r'\.gif', r'\.bmp', r'^no-reply@',
        r'^prueba@', r'^\d+[a-z]*@'
    ]
    typo_domains = ["gmil.com", "gmal.com", "gmaill.com", "gnail.com"]
    if len(email) < 6 or len(email) > 254:
        return False
    for pattern in invalid_patterns:
        if re.search(pattern, email, re.IGNORECASE):
            return False
    domain = email.split('@')[1]
    if domain in typo_domains or not re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", domain):
        return False
    return True

# Function to refresh search terms
def refresh_search_terms(campaign_id):
    return df_to_list(fetch_search_terms(campaign_id))

# Function to convert DataFrame to list of lists
def df_to_list(df):
    return df.values.tolist()

# Add this function before the Gradio interface definition
def manual_search(term, num_results, search_term_id):
    results = []
    try:
        print(f"Starting search for term: {term}")
        search_urls = list(search(term, num=num_results, stop=num_results, pause=2))
        print(f"Found {len(search_urls)} URLs")
        for url in search_urls:
            try:
                print(f"Processing URL: {url}")
                response = session.get(url, timeout=10)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                emails = find_emails(response.text)
                print(f"Found {len(emails)} emails on {url}")
                
                for email in emails:
                    if is_valid_email(email):
                        results.append([email, url])
                        print(f"Valid email found: {email}")
                        
                        # Save the lead and lead source to the database
                        lead_id = save_lead(email, None, None, None, None, None)
                        save_lead_source(lead_id, search_term_id, url, soup.title.string if soup.title else None, 
                                         soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else None, 
                                         response.status_code, str(response.elapsed))
                
                if len(results) >= num_results:
                    break
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
                print(f"Error processing {url}: {e}")
    except Exception as e:
        logging.error(f"Error in manual search: {e}")
        print(f"Error in manual search: {e}")
    
    print(f"Search completed. Found {len(results)} results.")
    return results[:num_results]

# Update the update_search_term_status function to handle term as string
def update_search_term_status(search_term_id, new_status, processed_leads=None):
    new_status = validate_status(new_status, ["pending", "completed"])
    conn = get_db_connection()
    cursor = conn.cursor()
    if processed_leads is not None:
        processed_leads = validate_num_results(processed_leads)
        cursor.execute("""
            UPDATE search_terms
            SET status = ?, processed_leads = ?, last_processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_status, processed_leads, search_term_id))
    else:
        cursor.execute("""
            UPDATE search_terms
            SET status = ?, last_processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_status, search_term_id))
    conn.commit()
    conn.close()

# Update the manual_search_wrapper function
def manual_search_wrapper(term, num_results, campaign_id):
    print(f"Manual search triggered with term: {term}, num_results: {num_results}")
    
    # Save the search term and get its ID
    search_term_id = add_search_term(term, campaign_id)
    
    # Perform the search
    results = manual_search(term, num_results, search_term_id)
    
    # Update the search term status
    update_search_term_status(search_term_id, 'completed', len(results))
    
    return results

# Function to fetch leads
def fetch_leads():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT l.id, l.email, l.first_name, l.last_name, l.company, l.job_title, st.term as search_term, ls.url as source_url
    FROM leads l
    JOIN lead_sources ls ON l.id = ls.lead_id
    JOIN search_terms st ON ls.search_term_id = st.id
    ORDER BY l.id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["ID", "Email", "First Name", "Last Name", "Company", "Job Title", "Search Term", "Source URL"])

# Add this function to fetch search terms for a specific campaign
def fetch_search_terms_for_campaign(campaign_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, term FROM search_terms WHERE campaign_id = ?', (campaign_id,))
    terms = cursor.fetchall()
    conn.close()
    return [{"name": f"{term[0]}: {term[1]}", "value": str(term[0])} for term in terms]

# Update the manual_search_multiple function
def manual_search_multiple(terms, num_results, campaign_id):
    all_results = []
    for term in terms:
        if term.strip():  # Only process non-empty terms
            results = manual_search_wrapper(term.strip(), num_results, campaign_id)
            all_results.extend(results)
    return all_results

# New function to get least searched terms
def get_least_searched_terms(n):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT term
    FROM search_terms
    ORDER BY processed_leads ASC
    LIMIT ?
    ''', (n,))
    terms = [row[0] for row in cursor.fetchall()]
    conn.close()
    return terms

# Streamlit app
def main():
    st.title("AUTOCLIENT")

    # Sidebar for navigation with radio buttons
    st.sidebar.title("Navigation")
    pages = [
        "Manual Search",
        "Bulk Search",
        "Bulk Send",
        "View Leads",
        "Search Terms",
        "Message Templates",
        "View Sent Messages",
        "Projects & Campaigns"
    ]
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Manual Search"

    st.session_state.current_page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.current_page))

    # Display the selected page
    if st.session_state.current_page == "Manual Search":
        manual_search_page()
    elif st.session_state.current_page == "Bulk Search":
        bulk_search_page()
    elif st.session_state.current_page == "Bulk Send":
        bulk_send_page()
    elif st.session_state.current_page == "View Leads":
        view_leads()
    elif st.session_state.current_page == "Search Terms":
        search_terms()
    elif st.session_state.current_page == "Message Templates":
        message_templates()
    elif st.session_state.current_page == "View Sent Messages":
        view_sent_messages()
    elif st.session_state.current_page == "Projects & Campaigns":
        projects_and_campaigns()

def display_result_card(result, index):
    with st.container():
        st.markdown(f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; margin-bottom:10px;">
            <h3 style="color:#1E90FF;">Result {index + 1}</h3>
            <p><strong>Email:</strong> {result[0]}</p>
            <p><strong>Source:</strong> <a href="{result[1]}" target="_blank">{result[1][:50]}...</a></p>
        </div>
        """, unsafe_allow_html=True)

def manual_search_page():
    st.header("Manual Search")
    
    tab1, tab2 = st.tabs(["Single Term Search", "Multiple Terms Search"])
    
    with tab1:
        with st.form(key="single_search_form"):
            search_term = st.text_input("Search Term")
            num_results = st.slider("Number of Results", min_value=10, max_value=200, value=30, step=10)
            campaign_id = st.selectbox("Select Campaign", options=fetch_campaigns())
            submit_button = st.form_submit_button(label="Search")
        
        if submit_button:
            st.session_state.single_search_started = True
            st.session_state.single_search_results = []
            st.session_state.single_search_logs = []
            st.session_state.single_search_progress = 0
        
        if 'single_search_started' in st.session_state and st.session_state.single_search_started:
            results_container = st.empty()
            progress_bar = st.progress(st.session_state.single_search_progress)
            status_text = st.empty()
            log_container = st.empty()
            
            if len(st.session_state.single_search_results) < num_results:
                with st.spinner("Searching..."):
                    new_results = manual_search_wrapper(search_term, num_results - len(st.session_state.single_search_results), campaign_id.split(":")[0])
                    st.session_state.single_search_results.extend(new_results)
                    st.session_state.single_search_logs.extend([f"Found result: {result[0]} from {result[1]}" for result in new_results])
                    st.session_state.single_search_progress = len(st.session_state.single_search_results) / num_results
            
            progress_bar.progress(st.session_state.single_search_progress)
            status_text.text(f"Found {len(st.session_state.single_search_results)} results...")
            log_container.text_area("Search Logs", "\n".join(st.session_state.single_search_logs), height=200)
            
            with results_container.container():
                for j, res in enumerate(st.session_state.single_search_results):
                    display_result_card(res, j)
            
            if len(st.session_state.single_search_results) >= num_results:
                st.success(f"Search completed! Found {len(st.session_state.single_search_results)} results.")
                st.session_state.single_search_started = False
            
            # Display statistics
            st.subheader("Search Statistics")
            st.metric("Total Results Found", len(st.session_state.single_search_results))
            st.metric("Unique Domains", len(set(result[0].split('@')[1] for result in st.session_state.single_search_results)))

    with tab2:
        with st.form(key="multi_search_form"):
            search_terms = [st.text_input(f"Search Term {i+1}") for i in range(4)]
            num_results_multiple = st.slider("Number of Results per Term", min_value=10, max_value=200, value=30, step=10)
            campaign_id_multiple = st.selectbox("Select Campaign for Multiple Search", options=fetch_campaigns(), key="multi_campaign")
            
            col1, col2 = st.columns(2)
            with col1:
                submit_button = st.form_submit_button(label="Search All Terms")
            with col2:
                fill_button = st.form_submit_button(label="Fill with Least Searched Terms")
        
        if submit_button:
            st.session_state.multi_search_started = True
            st.session_state.multi_search_results = []
            st.session_state.multi_search_logs = []
            st.session_state.multi_search_progress = 0
        
        if 'multi_search_started' in st.session_state and st.session_state.multi_search_started:
            results_container = st.empty()
            progress_bar = st.progress(st.session_state.multi_search_progress)
            status_text = st.empty()
            log_container = st.empty()
            
            with st.spinner("Searching..."):
                all_results = []
                logs = []
                for term_index, term in enumerate(search_terms):
                    if term.strip():
                        status_text.text(f"Searching term {term_index + 1} of {len(search_terms)}: {term}")
                        term_results = manual_search_wrapper(term, num_results_multiple, campaign_id_multiple.split(":")[0])
                        for i, result in enumerate(term_results):
                            all_results.append(result)
                            progress = (term_index * num_results_multiple + i + 1) / (len(search_terms) * num_results_multiple)
                            progress_bar.progress(progress)
                            logs.append(f"Term {term_index + 1}: Found result {i + 1}: {result[0]} from {result[1]}")
                            log_container.text_area("Search Logs", "\n".join(logs), height=200)
                            
                            with results_container.container():
                                for j, res in enumerate(all_results):
                                    display_result_card(res, j)
                            time.sleep(0.1)  # Add a small delay for animation effect
            
            st.success(f"Found {len(all_results)} results across all terms!")
            
            # Display statistics
            st.subheader("Search Statistics")
            st.metric("Total Results Found", len(all_results))
            st.metric("Unique Domains", len(set(result[0].split('@')[1] for result in all_results)))
            st.metric("Search Terms Processed", len([term for term in search_terms if term.strip()]))
        
        if fill_button:
            least_searched = get_least_searched_terms(4)
            for i, term in enumerate(least_searched):
                st.session_state[f"search_term_{i+1}"] = term
            st.rerun()

def bulk_search_page():
    st.header("Bulk Search")
    
    with st.form(key="bulk_search_form"):
        num_results = st.slider("Results per term", min_value=10, max_value=200, value=30, step=10)
        submit_button = st.form_submit_button(label="Start Bulk Search")
    
    if submit_button:
        st.session_state.bulk_search_started = True
        st.session_state.bulk_search_results = []
        st.session_state.bulk_search_logs = []
        st.session_state.bulk_search_progress = 0
        st.session_state.bulk_search_terms = fetch_search_terms()
    
    if 'bulk_search_started' in st.session_state and st.session_state.bulk_search_started:
        progress_bar = st.progress(st.session_state.bulk_search_progress)
        status_text = st.empty()
        log_container = st.empty()
        results_container = st.empty()
        
        with st.spinner("Performing bulk search..."):
            total_terms = len(st.session_state.bulk_search_terms)
            
            for term_index, term_row in enumerate(st.session_state.bulk_search_terms.itertuples()):
                if term_index < len(st.session_state.bulk_search_results) // num_results:
                    continue
                
                term = term_row.Term
                status_text.text(f"Searching term {term_index + 1} of {total_terms}: {term}")
                term_results = manual_search_wrapper(term, num_results, term_row.ID)
                
                st.session_state.bulk_search_results.extend(term_results)
                st.session_state.bulk_search_logs.extend([f"Term {term_index + 1}: Found result {i + 1}: {result[0]} from {result[1]}" for i, result in enumerate(term_results)])
                st.session_state.bulk_search_progress = (term_index + 1) / total_terms
                
                progress_bar.progress(st.session_state.bulk_search_progress)
                log_container.text_area("Search Logs", "\n".join(st.session_state.bulk_search_logs), height=200)
                
                with results_container.container():
                    for j, res in enumerate(st.session_state.bulk_search_results):
                        display_result_card(res, j)
        
        if st.session_state.bulk_search_progress >= 1:
            st.success(f"Bulk search completed! Found {len(st.session_state.bulk_search_results)} results.")
            st.session_state.bulk_search_started = False
        
        # Display statistics
        st.subheader("Bulk Search Statistics")
        st.metric("Total Results Found", len(st.session_state.bulk_search_results))
        st.metric("Unique Domains", len(set(result[0].split('@')[1] for result in st.session_state.bulk_search_results)))
        st.metric("Search Terms Processed", total_terms)

def bulk_send_page():
    st.header("Bulk Send")
    
    templates = fetch_message_templates()
    if not templates:
        st.warning("No message templates found. Please create a template first.")
        return

    with st.form(key="bulk_send_form"):
        template_id = st.selectbox("Select Message Template", options=templates)
        from_email = st.text_input("From Email", value="Sami Halawa <hello@indosy.com>")
        reply_to = st.text_input("Reply To", value="eugproductions@gmail.com")
        
        col1, col2 = st.columns(2)
        with col1:
            preview_button = st.form_submit_button(label="Preview Email")
        with col2:
            send_button = st.form_submit_button(label="Start Bulk Send")
    
    if preview_button:
        preview = get_email_preview(template_id.split(":")[0], from_email, reply_to)
        st.code(preview, language="html")
    
    if send_button:
        st.session_state.bulk_send_started = True
        st.session_state.bulk_send_logs = []
        st.session_state.bulk_send_progress = 0
    
    if 'bulk_send_started' in st.session_state and st.session_state.bulk_send_started:
        progress_bar = st.progress(st.session_state.bulk_send_progress)
        status_text = st.empty()
        log_container = st.empty()
        
        if not st.session_state.bulk_send_logs:
            with st.spinner("Sending emails..."):
                logs = asyncio.run(bulk_send(template_id.split(":")[0], from_email, reply_to))
                st.session_state.bulk_send_logs = logs
        
        for i, log in enumerate(st.session_state.bulk_send_logs):
            progress = (i + 1) / len(st.session_state.bulk_send_logs)
            if progress > st.session_state.bulk_send_progress:
                st.session_state.bulk_send_progress = progress
                progress_bar.progress(st.session_state.bulk_send_progress)
            status_text.text(f"Displaying log {i + 1} of {len(st.session_state.bulk_send_logs)}")
            log_container.text(log)
        
        if st.session_state.bulk_send_progress >= 1:
            st.success("Bulk send process completed!")
            st.session_state.bulk_send_started = False

        # Display a summary of the bulk send operation
        st.subheader("Bulk Send Summary")
        total_sent = sum(1 for log in st.session_state.bulk_send_logs if "Sent email to" in log)
        total_failed = sum(1 for log in st.session_state.bulk_send_logs if "Failed to send email to" in log)
        st.metric("Total Emails Sent", total_sent)
        st.metric("Total Emails Failed", total_failed)

        if total_sent == 0:
            st.warning("No emails were sent. Please check the logs for more information.")

def view_leads():
    st.header("View Leads")
    
    if st.button("Refresh Leads"):
        st.session_state.leads = fetch_leads()
    
    if 'leads' not in st.session_state:
        st.session_state.leads = fetch_leads()
    
    st.dataframe(st.session_state.leads, use_container_width=True)

def search_terms():
    st.header("Search Terms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add Search Term")
        with st.form(key="add_search_term_form"):
            search_term = st.text_input("Search Term")
            campaign_id = st.selectbox("Select Campaign", options=fetch_campaigns())
            submit_button = st.form_submit_button(label="Add Search Term")
        
        if submit_button:
            term_id = add_search_term(search_term, campaign_id.split(":")[0])
            st.success(f"Search term added with ID: {term_id}")
            st.session_state.search_terms = fetch_search_terms()

    with col2:
        st.subheader("Existing Search Terms")
        if st.button("Refresh Search Terms"):
            st.session_state.search_terms = fetch_search_terms()
        
        if 'search_terms' not in st.session_state:
            st.session_state.search_terms = fetch_search_terms()
        
        for index, row in st.session_state.search_terms.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{row['ID']}: {row['Search Term']} (Leads: {row['Leads Fetched']})")
            with col2:
                if st.button("Delete", key=f"delete_{row['ID']}"):
                    st.session_state.confirm_delete = row['ID']
                    st.session_state.leads_to_delete = delete_search_term_and_leads(row['ID'])
        
        if 'confirm_delete' in st.session_state:
            st.warning(f"Are you sure you want to delete search term {st.session_state.confirm_delete} and its related leads?")
            st.write("Leads to be deleted:")
            for lead in st.session_state.leads_to_delete:
                st.write(f"- {lead[1]}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm Delete"):
                    delete_search_term_and_leads(st.session_state.confirm_delete)
                    del st.session_state.confirm_delete
                    del st.session_state.leads_to_delete
                    st.session_state.search_terms = fetch_search_terms()
                    st.success("Search term and related leads deleted successfully.")
            with col2:
                if st.button("Cancel"):
                    del st.session_state.confirm_delete
                    del st.session_state.leads_to_delete

def delete_search_term_and_leads(search_term_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get the leads associated with this search term
        cursor.execute("""
            SELECT DISTINCT l.id, l.email
            FROM leads l
            JOIN lead_sources ls ON l.id = ls.lead_id
            WHERE ls.search_term_id = ?
        """, (search_term_id,))
        leads_to_delete = cursor.fetchall()

        # Delete the lead sources
        cursor.execute("DELETE FROM lead_sources WHERE search_term_id = ?", (search_term_id,))

        # Delete the leads
        lead_ids = [lead[0] for lead in leads_to_delete]
        cursor.executemany("DELETE FROM leads WHERE id = ?", [(id,) for id in lead_ids])

        # Delete the search term
        cursor.execute("DELETE FROM search_terms WHERE id = ?", (search_term_id,))

        conn.commit()
        return leads_to_delete
    except sqlite3.Error as e:
        conn.rollback()
        logging.error(f"Error deleting search term and leads: {e}")
        raise
    finally:
        conn.close()

def message_templates():
    st.header("Message Templates")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add Message Template")
        with st.form(key="add_message_template_form"):
            template_name = st.text_input("Template Name")
            subject = st.text_input("Subject")
            body_content = st.text_area("Body Content (HTML)", height=400)
            campaign_id = st.selectbox("Select Campaign", options=fetch_campaigns())
            submit_button = st.form_submit_button(label="Add Message Template")
        
        if submit_button:
            template_id = create_message_template(template_name, subject, body_content, campaign_id.split(":")[0])
            st.success(f"Message template added with ID: {template_id}")

    with col2:
        st.subheader("Existing Message Templates")
        if st.button("Refresh Message Templates"):
            st.session_state.message_templates = fetch_message_templates()
        
        if 'message_templates' not in st.session_state:
            st.session_state.message_templates = fetch_message_templates()
        
        st.dataframe(pd.DataFrame(st.session_state.message_templates, columns=["Template"]), use_container_width=True)

def view_sent_messages():
    st.header("View Sent Messages")
    
    if st.button("Refresh Sent Messages"):
        st.session_state.sent_messages = fetch_sent_messages()
    
    if 'sent_messages' not in st.session_state:
        st.session_state.sent_messages = fetch_sent_messages()
    
    st.dataframe(st.session_state.sent_messages, use_container_width=True)

def projects_and_campaigns():
    st.header("Projects & Campaigns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add Project")
        with st.form(key="add_project_form"):
            project_name = st.text_input("Project Name")
            submit_button = st.form_submit_button(label="Add Project")
        
        if submit_button:
            project_id = create_project(project_name)
            st.success(f"Project added with ID: {project_id}")

    with col2:
        st.subheader("Add Campaign")
        with st.form(key="add_campaign_form"):
            campaign_name = st.text_input("Campaign Name")
            project_id = st.selectbox("Select Project", options=fetch_projects())
            campaign_type = st.selectbox("Campaign Type", options=["Email", "SMS"])
            submit_button = st.form_submit_button(label="Add Campaign")
        
        if submit_button:
            campaign_id = create_campaign(campaign_name, project_id.split(":")[0], campaign_type)
            st.success(f"Campaign added with ID: {campaign_id}")

if __name__ == "__main__":
    main()