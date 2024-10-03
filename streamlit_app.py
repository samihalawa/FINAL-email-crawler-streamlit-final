import os
import re
import sqlite3
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from googlesearch import search
import gradio as gr
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import openai
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import json
import asyncio

# Configuration
AWS_ACCESS_KEY_ID = "AKIASO2XOMEGIVD422N7"
AWS_SECRET_ACCESS_KEY = "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"
REGION_NAME = "us-east-1"

openai.api_key = os.getenv("OPENAI_API_KEY", "default_openai_api_key")
openai.api_base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
openai_model = "mistral"

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
    body_content = sanitize_html(body_content)
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
    else:
        cursor.execute("""
            INSERT INTO leads (email, phone, first_name, last_name, company, job_title)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (email, phone, first_name, last_name, company, job_title))
        lead_id = cursor.lastrowid
    
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
    customized_content = sanitize_html(customized_content)
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
async def bulk_search(num_results, progress=gr.Progress()):
    num_results = validate_num_results(num_results)
    total_leads = 0
    all_results = []

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, term FROM search_terms')
    search_terms = cursor.fetchall()
    conn.close()

    for term_id, term in progress.tqdm(search_terms, desc="Processing search terms"):
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

            yield f"Processed {leads_found} leads for term '{term}'"
        except Exception as e:
            logging.error(f"Error performing search for term '{term}': {e}")

        update_search_term_status(term_id, 'completed', leads_found)

    yield f"Bulk search completed. Total new leads found: {total_leads}"
    yield all_results

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
        preview = f"Subject: {subject}\n\nFrom: {from_email}\nReply-To: {reply_to}\n\nBody:\n{body_content}"
        return preview
    else:
        return "Template not found"

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
async def bulk_send(template_id, from_email, reply_to, progress=gr.Progress()):
    template_id = validate_id(template_id, "template")
    from_email = validate_email(from_email)
    reply_to = validate_email(reply_to)

    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Fetch the template
        cursor.execute('SELECT template_name, subject, body_content FROM message_templates WHERE id = ?', (template_id,))
        template = cursor.fetchone()
        if not template:
            yield "Template not found"
            return

        template_name, subject, body_content = template

        # Fetch all leads that haven't been sent this template
        cursor.execute('''
            SELECT l.id, l.email 
            FROM leads l
            WHERE l.id NOT IN (
                SELECT m.lead_id 
                FROM messages m 
                WHERE m.template_id = ? AND m.status = 'sent'
            )
        ''', (template_id,))
        leads = cursor.fetchall()

        total_leads = len(leads)
        yield f"Preparing to send emails to {total_leads} leads"
        yield f"Template Name: {template_name}"
        yield f"Template ID: {template_id}"
        yield f"Subject: {subject}"
        yield f"From: {from_email}"
        yield f"Reply-To: {reply_to}"
        yield "---"

        total_sent = 0
        for index, (lead_id, email) in enumerate(progress.tqdm(leads, desc="Sending emails"), 1):
            # Check if the email has already been sent for this template
            cursor.execute('''
                SELECT id FROM messages 
                WHERE lead_id = ? AND template_id = ? AND status = 'sent'
            ''', (lead_id, template_id))
            if cursor.fetchone():
                yield f"[{index}/{total_leads}] Skipped email to {email} - Already sent"
                continue

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
                save_message(cursor, lead_id, template_id, 'sent', datetime.now(), subject, message_id)
                total_sent += 1
                yield f"[{index}/{total_leads}] Sent email to {email} - Subject: {subject} - MessageId: {message_id}"
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                logging.error(f"Failed to send email to {email}: {error_code} - {error_message}")
                save_message(cursor, lead_id, template_id, 'failed', None, subject)
                yield f"[{index}/{total_leads}] Failed to send email to {email}: {error_code} - {error_message}"
            
            conn.commit()  # Commit after each message
            await asyncio.sleep(0.1)  # Small delay to allow UI updates

        yield "---"
        yield f"Bulk send completed. Total emails sent: {total_sent}/{total_leads}"

    finally:
        conn.close()

# Update the save_message function to accept a cursor
def save_message(cursor, lead_id, template_id, status, sent_at=None, subject=None, message_id=None):
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

# Update the Gradio interface
with gr.Blocks() as gradio_app:
    # ... (other tabs remain unchanged)

    with gr.Tab("Bulk Send"):
        template_id = gr.Dropdown(choices=fetch_message_templates(), label="Select Message Template")
        from_email = gr.Textbox(label="From Email", value="Sami Halawa <hello@indosy.com>")
        reply_to = gr.Textbox(label="Reply To", value="eugproductions@gmail.com")
        preview_button = gr.Button("Preview Email")
        email_preview = gr.HTML(label="Email Preview")
        bulk_send_button = gr.Button("Bulk Send")
        bulk_send_log = gr.Textbox(label="Bulk Send Log", interactive=False, lines=20)

    # ... (other tabs remain unchanged)

    # Update the bulk send button action
    async def bulk_send_wrapper(template_id, from_email, reply_to):
        logs = []
        async for log in bulk_send(template_id, from_email, reply_to):
            logs.append(log)
            yield "\n".join(logs)

    bulk_send_button.click(
        bulk_send_wrapper,
        inputs=[template_id, from_email, reply_to],
        outputs=[bulk_send_log],
        api_name="bulk_send"
    )

    # ... (other button actions remain unchanged)

with gr.Blocks() as gradio_app:
    gr.Markdown("# Email Campaign Management System")

    with gr.Tab("Projects and Campaigns"):
        with gr.Row():
            with gr.Column():
                project_name = gr.Textbox(label="Project Name")
                create_project_btn = gr.Button("Create Project")
                project_status = gr.Textbox(label="Project Status", interactive=False)
            with gr.Column():
                campaign_name = gr.Textbox(label="Campaign Name")
                project_id = gr.Dropdown(label="Project", choices=fetch_projects())
                campaign_type = gr.Radio(["Email", "SMS"], label="Campaign Type")
                create_campaign_btn = gr.Button("Create Campaign")
                campaign_status = gr.Textbox(label="Campaign Status", interactive=False)

    with gr.Tab("Message Templates"):
        with gr.Row():
            with gr.Column():
                template_name = gr.Textbox(label="Template Name")
                subject = gr.Textbox(label="Subject")
                body_content = gr.Code(language="html", label="Body Content")
                campaign_id_for_template = gr.Dropdown(label="Campaign", choices=fetch_campaigns())
                create_template_btn = gr.Button("Create Template")
            with gr.Column():
                template_status = gr.Textbox(label="Template Status", interactive=False)
                template_preview = gr.HTML(label="Template Preview")

    with gr.Tab("Search Terms"):
        with gr.Row():
            with gr.Column():
                search_term = gr.Textbox(label="Search Term")
                campaign_id_for_search = gr.Dropdown(label="Campaign", choices=fetch_campaigns())
                add_term_btn = gr.Button("Add Search Term")
            with gr.Column():
                search_term_status = gr.Textbox(label="Search Term Status", interactive=False)
        refresh_search_terms_btn = gr.Button("Refresh Search Terms")
        search_term_list = gr.Dataframe(df_to_list(fetch_search_terms()), headers=["ID", "Search Term", "Leads Fetched", "Status"])

    with gr.Tab("Bulk Search"):
        num_results_bulk = gr.Slider(minimum=10, maximum=500, value=120, step=10, label="Results per term")
        bulk_search_button = gr.Button("Bulk Search")
        bulk_search_results = gr.Dataframe(headers=["Email", "Source URL", "Search Term"])
        bulk_search_log = gr.TextArea(label="Bulk Search Log", interactive=False)

    with gr.Tab("Bulk Send"):
        template_id = gr.Dropdown(choices=fetch_message_templates(), label="Select Message Template")
        from_email = gr.Textbox(label="From Email", value="Sami Halawa <hello@indosy.com>")
        reply_to = gr.Textbox(label="Reply To", value="eugproductions@gmail.com")
        preview_button = gr.Button("Preview Email")
        email_preview = gr.HTML(label="Email Preview")
        bulk_send_button = gr.Button("Bulk Send")
        bulk_send_log = gr.Textbox(label="Bulk Send Log", interactive=False, lines=20)

    with gr.Tab("View Sent Messages"):
        refresh_messages_btn = gr.Button("Refresh Sent Messages")
        sent_messages_table = gr.DataFrame(fetch_sent_messages(), headers=["ID", "Email", "Template", "Subject", "Sent At", "Status"])

    with gr.Tab("Manual Search"):
        with gr.Row():
            manual_search_term = gr.Textbox(label="Manual Search Term")
            manual_num_results = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Number of Results")
            manual_campaign_id = gr.Dropdown(label="Campaign", choices=fetch_campaigns())
            manual_search_btn = gr.Button("Search")
        manual_search_results = gr.Dataframe(headers=["Email", "Source"])

    with gr.Tab("Manual Search Multiple"):
        with gr.Row():
            manual_search_terms = [gr.Textbox(label=f"Search Term {i+1}") for i in range(4)]
        with gr.Row():
            manual_num_results_multiple = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Number of Results per Term")
            manual_campaign_id_multiple = gr.Dropdown(label="Campaign", choices=fetch_campaigns())
        manual_search_multiple_btn = gr.Button("Search All Terms")
        manual_search_multiple_results = gr.Dataframe(headers=["Email", "Source"])

    with gr.Tab("View Leads"):
        refresh_leads_btn = gr.Button("Refresh Leads")
        leads_table = gr.Dataframe(fetch_leads(), headers=["ID", "Email", "First Name", "Last Name", "Company", "Job Title", "Search Term", "Source URL"])

    # Define button actions
    create_project_btn.click(create_project, inputs=[project_name], outputs=[project_status])
    create_campaign_btn.click(create_campaign, inputs=[campaign_name, project_id, campaign_type], outputs=[campaign_status])
    create_template_btn.click(create_message_template, inputs=[template_name, subject, body_content, campaign_id_for_template], outputs=[template_status])
    add_term_btn.click(add_search_term, inputs=[search_term, campaign_id_for_search], outputs=[search_term_status])
    preview_button.click(get_email_preview, inputs=[template_id, from_email, reply_to], outputs=email_preview)
    bulk_send_button.click(
        bulk_send_wrapper,
        inputs=[template_id, from_email, reply_to],
        outputs=[bulk_send_log],
        api_name="bulk_send"
    )
    manual_search_btn.click(
        manual_search_wrapper,
        inputs=[manual_search_term, manual_num_results, manual_campaign_id],
        outputs=manual_search_results
    )
    refresh_search_terms_btn.click(
        lambda: df_to_list(fetch_search_terms()),
        inputs=[],
        outputs=[search_term_list]
    )
    refresh_leads_btn.click(
        lambda: fetch_leads(),
        inputs=[],
        outputs=[leads_table]
    )
    refresh_messages_btn.click(
        lambda: fetch_sent_messages(),
        inputs=[],
        outputs=[sent_messages_table]
    )

    # New button actions for Bulk Search
    async def process_bulk_search(num_results):
        results = []
        logs = []
        async for output in bulk_search(num_results):
            if isinstance(output, str):
                logs.append(output)
            else:
                results = output
        return "\n".join(logs), results

    bulk_search_button.click(
        process_bulk_search,
        inputs=[num_results_bulk],
        outputs=[bulk_search_log, bulk_search_results]
    )

    # Add this new button action for Manual Search Multiple
    manual_search_multiple_btn.click(
        lambda *args: manual_search_multiple(args[:-2], args[-2], args[-1]),
        inputs=[*manual_search_terms, manual_num_results_multiple, manual_campaign_id_multiple],
        outputs=manual_search_multiple_results
    )

    # Update the bulk send button action
    async def bulk_send_wrapper(template_id, from_email, reply_to):
        logs = []
        async for log in bulk_send(template_id, from_email, reply_to):
            logs.append(log)
            yield "\n".join(logs)

    bulk_send_button.click(
        bulk_send_wrapper,
        inputs=[template_id, from_email, reply_to],
        outputs=[bulk_send_log],
        api_name="bulk_send"
    )

# Launch the app
if __name__ == "__main__":
    asyncio.run(gradio_app.launch(server_name="127.0.0.1", server_port=7872))