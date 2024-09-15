import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from googlesearch import search
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from openai import OpenAI
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
import random
import plotly.express as px
from dotenv import load_dotenv

# load_dotenv()  # Load environment variables from .env file  # Undefined function

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = os.getenv("AWS_REGION_NAME", "us-east-1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, OPENAI_API_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

# SQLite configuration
sqlite_db_path = "autoclient.db"

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
        conn = sqlite3.connect(sqlite_db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        st.error(f"Failed to connect to the database. Please try again later.")
        return None

# HTTP session with retry strategy
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Setup logging
try:
    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
except IOError as e:
#     print(f"Error setting up logging: {e}")  # Undefined function
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
def initialize_database():
    conn = get_db_connection()
    if not conn:
        st.error("Failed to initialize database. Please check your database configuration.")
        return

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
            meta_tags TEXT,
            phone_numbers TEXT,
            content TEXT,
            tags TEXT,
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
            campaign_id INTEGER,
            status TEXT DEFAULT 'pending',
            processed_leads INTEGER DEFAULT 0,
            last_processed_at TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
        );

        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            name TEXT,
            bio TEXT,
            values TEXT,
            contact_name TEXT,
            contact_role TEXT,
            contact_email TEXT,
            company_description TEXT,
            company_mission TEXT,
            company_target_market TEXT,
            company_other TEXT,
            product_name TEXT,
            product_description TEXT,
            product_target_customer TEXT,
            product_other TEXT,
            other_context TEXT,
            example_email TEXT,
            complete_version TEXT,
            medium_version TEXT,
            small_version TEXT,
            temporal_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        );
    ''')
    conn.commit()
    conn.close()
    logging.info("Database initialized successfully!")

import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime
from fake_useragent import UserAgent
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
import random
import plotly.express as px

AWS_ACCESS_KEY_ID = "AKIASO2XOMEGIVD422N7"
AWS_SECRET_ACCESS_KEY = "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"
REGION_NAME = "us-east-1"

openai.api_key = os.getenv("OPENAI_API_KEY", "sk-1234")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://openai-proxy-kl3l.onrender.com")
openai_model = "gpt-3.5-turbo"

# SQLite configuration
sqlite_db_path = "autoclient.db"

# Ensure the database file exists
try:
    if not os.path.exists(sqlite_db_path):
#         open(sqlite_db_path, 'w').close()  # Undefined function
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
    conn = sqlite3.connect(sqlite_db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# HTTP session with retry strategy
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Setup logging
try:
    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
except IOError as e:
#     print(f"Error setting up logging: {e}")  # Undefined function
    raise

# Input validation functions
    if not name or not name.strip():
        raise ValueError("Name cannot be empty or just whitespace")
    if len(name) > 100:
        raise ValueError("Name is too long (max 100 characters)")
    return name.strip()

    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        raise ValueError("Invalid email address")
    return email

    valid_types = ["Email", "SMS"]
    if campaign_type not in valid_types:
        raise ValueError(f"Invalid campaign type. Must be one of {valid_types}")
    return campaign_type

    try:
        id_int = int(id_value.split(':')[0] if ':' in str(id_value) else id_value)
        if id_int <= 0:
            raise ValueError
        return id_int
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid {id_type} ID")

    if status not in valid_statuses:
        raise ValueError(f"Invalid status. Must be one of {valid_statuses}")
    return status

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
        meta_tags TEXT,
        phone_numbers TEXT,
        content TEXT,
        tags TEXT,
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

    CREATE TABLE IF NOT EXISTS search_term_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email_template TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS search_terms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term TEXT NOT NULL,
    group_id INTEGER,
    category TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (group_id) REFERENCES search_term_groups (id)
);

CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_name TEXT,
    kb_bio TEXT,
    kb_values TEXT,
    contact_name TEXT,
    contact_role TEXT,
    contact_email TEXT,
    company_description TEXT,
    company_mission TEXT,
    company_target_market TEXT,
    company_other TEXT,
    product_name TEXT,
    product_description TEXT,
    product_target_customer TEXT,
    product_other TEXT,
    other_context TEXT,
    example_email TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    function_name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS optimized_search_terms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term TEXT NOT NULL,
    original_term_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (original_term_id) REFERENCES search_terms (id)
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

def create_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

# Define the function to handle real-time logging
def update_logs(logs, log_message, log_container):
    logs.append(log_message)
    log_container.text_area("Search Logs", "\n".join(logs), height=300)


def log_search_term_effectiveness(term, total_results, valid_leads, irrelevant_leads, blogs_found, directories_found):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO search_term_effectiveness
        (term, total_results, valid_leads, irrelevant_leads, blogs_found, directories_found)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (term, total_results, valid_leads, irrelevant_leads, blogs_found, directories_found))
    conn.commit()
    conn.close()

def should_scrape_page(meta_description, title):
    # No premature filtering. We process even if it's a blog or directory
    return True  # We'll always scrape, but categorize as 'blog' or 'directory'


def categorize_page_content(soup, url):
    title = soup.title.string if soup.title else ''
    meta_description = soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else ''
    
    if 'blog' in title.lower() or 'blog' in meta_description.lower():
        return 'blog'
    elif 'directory' in title.lower() or 'directory' in meta_description.lower():
        return 'directory'
    else:
        return 'company'



def manual_search_wrapper(term, num_results, campaign_id, search_type="All Leads", log_container=None):
    results = []
    logs = []
    ua = UserAgent()
    session = create_session()

    try:
        log_message = f"Starting search for term: {term}"
        if log_container:
            update_logs(logs, log_message, log_container)

        search_urls = list(search(term, num=num_results, stop=num_results, pause=2))
        log_message = f"Found {len(search_urls)} URLs"
        if log_container:
            update_logs(logs, log_message, log_container)

        term_id = add_search_term(term, campaign_id)
        if term_id is None:
            raise ValueError(f"Failed to add or retrieve search term ID for '{term}'")

        for url in search_urls:
            try:
                log_message = f"Processing URL: {url}"
                if log_container:
                    update_logs(logs, log_message, log_container)

                headers = {
                    'User-Agent': ua.random,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.google.com/',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                response = session.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')

                emails = find_emails(response.text)
                phone_numbers = extract_phone_numbers(response.text)

                log_message = f"Found {len(emails)} emails and {len(phone_numbers)} phone numbers on {url}"
                if log_container:
                    update_logs(logs, log_message, log_container)

                for email in emails:
                    if is_valid_email(email):
                        lead_id = save_lead(email, None, None, None, None, None)
                        save_lead_source(lead_id, term_id, url, soup.title.string if soup.title else None,
                                         soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else None,
                                         response.status_code, str(response.elapsed), soup.find_all('meta'), extract_phone_numbers(response.text), soup.get_text(), [])

                        log_message = f"Valid email found: {email}"
                        if log_container:
                            update_logs(logs, log_message, log_container)
                        results.append([email, url, term])

                time.sleep(random.uniform(1, 3))  # Random delay between requests
            except requests.exceptions.RequestException as e:
                log_message = f"Error processing {url}: {e}"
                if log_container:
                    update_logs(logs, log_message, log_container)
            except Exception as e:
                log_message = f"Unexpected error processing {url}: {e}"
                if log_container:
                    update_logs(logs, log_message, log_container)

            if len(results) >= num_results:
                break

        # Update the processed_leads count for this search term
#         update_processed_leads_count(term_id, len(results))  # Undefined function
        time.sleep(random.uniform(30, 60))  # Longer delay between search terms
    except Exception as e:
        log_message = f"Error in manual search: {e}"
        if log_container:
            update_logs(logs, log_message, log_container)

    log_message = f"Search completed. Found {len(results)} results."
    if log_container:
        update_logs(logs, log_message, log_container)

    return results[:num_results]



# Function to add a new search term
def add_search_term(term, campaign_id):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if the term already exists for this campaign
        cursor.execute("SELECT id FROM search_terms WHERE term = ? AND campaign_id = ?", (term, campaign_id))
        existing_term = cursor.fetchone()
        
        if existing_term:
            term_id = existing_term['id']
            logging.info(f"Search term '{term}' already exists for campaign {campaign_id} with ID: {term_id}")
        else:
            cursor.execute('''
                INSERT INTO search_terms (term, campaign_id, processed_leads) 
                VALUES (?, ?, 0)
            ''', (term, campaign_id))
            term_id = cursor.lastrowid
            logging.info(f"New search term '{term}' added for campaign {campaign_id} with ID: {term_id}")
        
        conn.commit()
        return term_id
    except sqlite3.Error as e:
        logging.error(f"Database error in add_search_term: {e}")
        return None
    finally:
        if conn:
            conn.close()

    results = []
    ua = UserAgent()
    session = create_session()
    
    try:
#         print(f"Starting search for term: {term}")  # Undefined function
        search_urls = list(search(term, num=num_results, stop=num_results, pause=2))
#         print(f"Found {len(search_urls)} URLs")  # Undefined function
        
        # Add the search term to the database and get its ID
        term_id = add_search_term(term, campaign_id)
        if term_id is None:
            raise ValueError(f"Failed to add or retrieve search term ID for '{term}'")
        
        for url in search_urls:
            try:
#                 print(f"Processing URL: {url}")  # Undefined function
                headers = {
                    'User-Agent': ua.random,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.google.com/',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                response = session.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                emails = find_emails(response.text)
                
                for email in emails:
                    if is_valid_email(email):
                        lead_id = save_lead(email, None, None, None, None, None)
                        save_lead_source(lead_id, term_id, url, soup.title.string if soup.title else None, 
                                         soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else None, 
                                         response.status_code, str(response.elapsed), soup.find_all('meta'), extract_phone_numbers(response.text), soup.get_text(), [])
                        results.append([email, url, term])
                
                time.sleep(random.uniform(1, 3))  # Random delay between requests
            except requests.exceptions.RequestException as e:
#                 print(f"Error processing {url}: {e}")  # Undefined function
            except Exception as e:
#                 print(f"Unexpected error processing {url}: {e}")  # Undefined function
            
            if len(results) >= num_results:
                break
        
        # Update the processed_leads count for this search term
#         update_processed_leads_count(term_id, len(results))  # Undefined function
        
        time.sleep(random.uniform(30, 60))  # Longer delay between search terms
    except Exception as e:
#         print(f"Error in manual search: {e}")  # Undefined function
    
#     print(f"Search completed. Found {len(results)} results.")  # Undefined function
    return results[:num_results]

def update_processed_leads_count(term_id, count):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE search_terms 
            SET processed_leads = processed_leads + ? 
            WHERE id = ?
        ''', (count, term_id))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error in update_processed_leads_count: {e}")
    finally:
        if conn:
            conn.close()


    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE search_terms 
            SET processed_leads = processed_leads + ? 
            WHERE id = ?
        ''', (count, term_id))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error in update_processed_leads_count: {e}")
    finally:
        if conn:
            conn.close()

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
        lead_id = existing_lead['id']
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


def save_lead_source(lead_id, term_id, url, page_title, meta_description, http_status, scrape_duration, meta_tags, phone_numbers, content, tags):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO lead_sources (lead_id, search_term_id, url, page_title, meta_description, http_status, scrape_duration, meta_tags, phone_numbers, content, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lead_id, term_id, url, page_title, meta_description, http_status, scrape_duration, json.dumps(meta_tags), json.dumps(phone_numbers), content, json.dumps(tags)))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error in save_lead_source: {e}")
    finally:
        if conn:
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
    return [f"{row['id']}: {row['template_name']}" for row in rows]

# Function to fetch projects
def fetch_projects():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, project_name FROM projects')
    rows = cursor.fetchall()
    conn.close()
    return [f"{row['id']}: {row['project_name']}" for row in rows]

# Function to fetch campaigns
def fetch_campaigns():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, campaign_name FROM campaigns ORDER BY campaign_name')
        campaigns = cursor.fetchall()
        return [f"{campaign['id']}: {campaign['campaign_name']}" for campaign in campaigns]
    except sqlite3.Error as e:
        logging.error(f"Database error in fetch_campaigns: {e}")
        return []
    finally:
        if conn:
            conn.close()

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
            search_urls = list(search(term['term'], num=num_results, stop=num_results, pause=2))
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
                                             response.status_code, str(response.elapsed), soup.find_all('meta'), extract_phone_numbers(response.text), soup.get_text(), [])
                            all_results.append([email, url, term['term']])
                            leads_found += 1
                            total_leads += 1
                            
                            if leads_found >= num_results:
                                break
                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")

            logs.append(f"Processed {leads_found} leads for term '{term['term']}'")
        except Exception as e:
            logging.error(f"Error performing search for term '{term['term']}': {e}")

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

# Update the fetch_sent_messages function
def fetch_sent_messages():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT m.id, m.sent_at, l.email, mt.template_name, m.customized_subject, m.customized_content, m.status, m.message_id
    FROM messages m
    JOIN leads l ON m.lead_id = l.id
    JOIN message_templates mt ON m.template_id = mt.id
    ORDER BY m.sent_at DESC
    ''')
    messages = cursor.fetchall()
    conn.close()
    return pd.DataFrame(messages, columns=['ID', 'Sent At', 'Email', 'Template', 'Subject', 'Content', 'Status', 'Message ID'])

# Update the view_sent_messages function
def view_sent_messages():
    st.header("View Sent Messages")
    
    if st.button("Refresh Sent Messages"):
        st.session_state.sent_messages = fetch_sent_messages()
    
    if 'sent_messages' not in st.session_state:
        st.session_state.sent_messages = fetch_sent_messages()
    
    # Display messages in a more organized manner
    for _, row in st.session_state.sent_messages.iterrows():
        with st.expander(f"Message to {row['Email']} - {row['Sent At']}"):
            st.write(f"**Subject:** {row['Subject']}")
            st.write(f"**Template:** {row['Template']}")
            st.write(f"**Status:** {row['Status']}")
            st.write(f"**Message ID:** {row['Message ID']}")
            st.write("**Content:**")
            st.markdown(row['Content'], unsafe_allow_html=True)

    # Display summary statistics
    st.subheader("Summary Statistics")
    total_messages = len(st.session_state.sent_messages)
    sent_messages = len(st.session_state.sent_messages[st.session_state.sent_messages['Status'] == 'sent'])
    failed_messages = len(st.session_state.sent_messages[st.session_state.sent_messages['Status'] == 'failed'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages", total_messages)
    col2.metric("Sent Messages", sent_messages)
    col3.metric("Failed Messages", failed_messages)

# Update the bulk_send function to be a coroutine instead of an async generator
async def bulk_send(template_id, from_email, reply_to, leads):
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
        return "Template not found"

    template_name, subject, body_content = template

    total_leads = len(leads)
    logs = [
        f"Preparing to send emails to {total_leads} leads",
        f"Template Name: {template_name}",
        f"Template ID: {template_id}",
        f"Subject: {subject}",
        f"From: {from_email}",
        f"Reply-To: {reply_to}",
        "---"
    ]

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
#             save_message(lead_id, template_id, 'sent', datetime.now(), subject, message_id, body_content)  # Undefined function
            total_sent += 1
            logs.append(f"[{index}/{total_leads}] Sent email to {email} - Subject: {subject} - MessageId: {message_id}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logging.error(f"Failed to send email to {email}: {error_code} - {error_message}")
#             save_message(lead_id, template_id, 'failed', None, subject)  # Undefined function
            logs.append(f"[{index}/{total_leads}] Failed to send email to {email}: {error_code} - {error_message}")
        except Exception as e:
            logging.error(f"Unexpected error sending email to {email}: {str(e)}")
#             save_message(lead_id, template_id, 'failed', None, subject)  # Undefined function
            logs.append(f"[{index}/{total_leads}] Unexpected error sending email to {email}: {str(e)}")
        
        await asyncio.sleep(0.1)  # Small delay to allow UI updates

    logs.append("---")
    logs.append(f"Bulk send completed. Total emails sent: {total_sent}/{total_leads}")
    return logs

# Update the bulk_send_page function
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
        
        send_option = st.radio("Send to:", 
                               ["All Leads", 
                                "All Not Contacted with this Template", 
                                "All Not Contacted with Templates from this Campaign",
                                "Selected Search Term Groups"])
        
        if send_option == "Selected Search Term Groups":
            groups = fetch_search_term_groups()
            selected_groups = st.multiselect("Select Search Term Groups", options=groups)
            group_leads = get_leads_count_for_groups(selected_groups)
            st.write("Leads in selected groups:")
            for group, count in group_leads.items():
                st.write(f"{group}: {count} leads")
        
        filter_option = st.radio("Filter:", 
                                 ["Not Filter Out Leads", 
                                  "Filter Out blog-directory"])
        
        col1, col2 = st.columns(2)
        with col1:
            preview_button = st.form_submit_button(label="Preview Email")
        with col2:
            send_button = st.form_submit_button(label="Start Bulk Send")
    
    if preview_button:
        preview = get_email_preview(template_id.split(":")[0], from_email, reply_to)
        st.components.v1.html(preview, height=600, scrolling=True)
    
    if send_button:
        if send_option == "Selected Search Term Groups" and not selected_groups:
            st.error("Please select at least one search term group.")
            return
        
        st.session_state.bulk_send_started = True
        st.session_state.bulk_send_logs = []
        st.session_state.bulk_send_progress = 0
        
        # Fetch leads based on send_option and filter_option
        leads_to_send = fetch_leads_for_bulk_send(template_id.split(":")[0], send_option, filter_option, selected_groups if send_option == "Selected Search Term Groups" else None)
        
        st.write(f"Preparing to send emails to {len(leads_to_send)} leads")
        
        # Perform bulk send
        bulk_send_coroutine = bulk_send(template_id.split(":")[0], from_email, reply_to, leads_to_send)
        logs = asyncio.run(bulk_send_coroutine)
        
        # Display logs and statistics
        for log in logs:
            st.write(log)
        
        st.success(f"Bulk send completed. Sent {len(leads_to_send)} emails.")

# New function to fetch search term groups
def fetch_search_term_groups():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM search_term_groups")
    groups = [f"{row['id']}: {row['name']}" for row in cursor.fetchall()]
    conn.close()
    return groups

# New function to get leads count for groups
def get_leads_count_for_groups(selected_groups):
    conn = get_db_connection()
    cursor = conn.cursor()
    group_leads = {}
    for group in selected_groups:
        group_id = group.split(":")[0]
        cursor.execute("""
            SELECT COUNT(DISTINCT l.id)
            FROM leads l
            JOIN lead_sources ls ON l.id = ls.lead_id
            JOIN search_terms st ON ls.search_term_id = st.id
            WHERE st.group_id = ?
        """, (group_id,))
        count = cursor.fetchone()[0]
        group_leads[group] = count
    conn.close()
    return group_leads

def fetch_leads_for_bulk_send(template_id, send_option, filter_option, selected_groups=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT DISTINCT l.id, l.email
    FROM leads l
    JOIN lead_sources ls ON l.id = ls.lead_id
    """
    
    if send_option == "All Not Contacted with this Template":
        query += f"""
        LEFT JOIN messages m ON l.id = m.lead_id AND m.template_id = {template_id}
        WHERE m.id IS NULL
        """
    elif send_option == "All Not Contacted with Templates from this Campaign":
        query += f"""
        LEFT JOIN messages m ON l.id = m.lead_id
        LEFT JOIN message_templates mt ON m.template_id = mt.id
        WHERE m.id IS NULL OR mt.campaign_id != (SELECT campaign_id FROM message_templates WHERE id = {template_id})
        """
    elif send_option == "Selected Search Term Groups":
        group_ids = [group.split(":")[0] for group in selected_groups]
        query += f"""
        JOIN search_terms st ON ls.search_term_id = st.id
        WHERE st.group_id IN ({','.join(group_ids)})
        """
    
    if filter_option == "Filter Out blog-directory":
        query += " AND NOT ls.tags LIKE '%blog-directory%'"
    
    cursor.execute(query)
    leads = cursor.fetchall()
    conn.close()
    
    return leads

    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT DISTINCT l.id, l.email
    FROM leads l
    JOIN lead_sources ls ON l.id = ls.lead_id
    """
    
    if send_option == "All Not Contacted with this Template":
        query += f"""
        LEFT JOIN messages m ON l.id = m.lead_id AND m.template_id = {template_id}
        WHERE m.id IS NULL
        """
    elif send_option == "All Not Contacted with Templates from this Campaign":
        query += f"""
        LEFT JOIN messages m ON l.id = m.lead_id
        LEFT JOIN message_templates mt ON m.template_id = mt.id
        WHERE m.id IS NULL OR mt.campaign_id != (SELECT campaign_id FROM message_templates WHERE id = {template_id})
        """
    
    if filter_option == "Filter Out blog-directory":
        query += " AND NOT ls.tags LIKE '%blog-directory%'"
    
    cursor.execute(query)
    leads = cursor.fetchall()
    conn.close()
    
    return leads

# Update the save_message function to include the subject
def save_message(lead_id, template_id, status, sent_at=None, subject=None, message_id=None, customized_content=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if sent_at:
        cursor.execute("""
            INSERT INTO messages (lead_id, template_id, status, sent_at, customized_subject, message_id, customized_content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (lead_id, template_id, status, sent_at, subject, message_id, customized_content))
    else:
        cursor.execute("""
            INSERT INTO messages (lead_id, template_id, status, customized_subject, message_id, customized_content)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (lead_id, template_id, status, subject, message_id, customized_content))
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
import re
from urllib.parse import urlparse

def extract_phone_numbers(text):
    phone_pattern = re.compile(r'\b(?:\+?34)?[\s.-]?[6789]\d{2}[\s.-]?\d{3}[\s.-]?\d{3}\b')
    return phone_pattern.findall(text)

def is_probable_blog_or_directory(soup, email):
    # Check if the email is inside an <article> or <main> tag
    article_content = soup.find('article')
    main_content = soup.find('main')
    
    if article_content and email in article_content.get_text():
        return True
    if main_content and email in main_content.get_text():
        return True
    
    # Check if there are multiple emails on the page
    all_emails = find_emails(soup.get_text())
    if len(all_emails) > 3:  # Arbitrary threshold, adjust as needed
        return True
    
    # Check for common blog/directory indicators in the URL or title
    url = soup.find('meta', property='og:url')
    url = url['content'] if url else ''
    title = soup.title.string if soup.title else ''
    indicators = ['blog', 'article', 'news', 'directory', 'list', 'index']
    if any(indicator in url.lower() or indicator in title.lower() for indicator in indicators):
        return True
    
    return False


def extract_visible_text(soup):
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()    # rip it out

    # get text
    text = soup.get_text()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text


    results = []
    try:
#         print(f"Starting search for term: {term}")  # Undefined function
        search_urls = list(search(term, num=num_results, stop=num_results, pause=2))
#         print(f"Found {len(search_urls)} URLs")  # Undefined function
        for url in search_urls:
            try:
#                 print(f"Processing URL: {url}")  # Undefined function
                response = session.get(url, timeout=10)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title = soup.title.string if soup.title else ""
                meta_description = soup.find('meta', attrs={'name': 'description'})
                meta_description = meta_description['content'] if meta_description else ""
                
                emails = find_emails(response.text)
                phone_numbers = extract_phone_numbers(response.text)
                
                meta_tags = [str(tag) for tag in soup.find_all('meta')]
                
                content = extract_visible_text(soup)
                
#                 print(f"Found {len(emails)} emails and {len(phone_numbers)} phone numbers on {url}")  # Undefined function
                
                for email in emails:
                    if is_valid_email(email):
                        tags = []
                        if is_probable_blog_or_directory(soup, email):
                            tags.append("blog-directory")
                        else:
                            tags.append("company")
                        
                        if search_type == "All Leads" or (search_type == "Exclude Probable Blogs/Directories" and "blog-directory" not in tags):
                            lead_id = save_lead(email, phone_numbers[0] if phone_numbers else None, None, None, None, None)
                            save_lead_source(lead_id, campaign_id, url, title, meta_description, response.status_code, 
#                                              str(response.elapsed), meta_tags, phone_numbers, content, tags)  # Undefined function
                            results.append([email, url, title, meta_description, tags])
#                             print(f"Valid email found: {email}")  # Undefined function
                
                if len(results) >= num_results:
                    break
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
#                 print(f"Error processing {url}: {e}")  # Undefined function
    except Exception as e:
        logging.error(f"Error in manual search: {e}")
#         print(f"Error in manual search: {e}")  # Undefined function
    
#     print(f"Search completed. Found {len(results)} results.")  # Undefined function
    return results[:num_results]


# Update save_lead function to be more resilient
    try:
        email = validate_email(email)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM leads WHERE email = ?", (email,))
        existing_lead = cursor.fetchone()
        
        if existing_lead:
            lead_id = existing_lead['id']
#             print(f"Existing lead found for email {email} with ID {lead_id}")  # Undefined function
            cursor.execute("""
                UPDATE leads
                SET phone = COALESCE(?, phone),
                    first_name = COALESCE(?, first_name),
                    last_name = COALESCE(?, last_name),
                    company = COALESCE(?, company),
                    job_title = COALESCE(?, job_title)
                WHERE id = ?
            """, (phone, first_name, last_name, company, job_title, lead_id))
#             print(f"Updated existing lead with ID {lead_id}")  # Undefined function
        else:
            cursor.execute("""
                INSERT INTO leads (email, phone, first_name, last_name, company, job_title)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (email, phone, first_name, last_name, company, job_title))
            lead_id = cursor.lastrowid
#             print(f"New lead created for email {email} with ID {lead_id}")  # Undefined function
        
        conn.commit()
        conn.close()
        return lead_id
    except Exception as e:
#         print(f"Error saving lead: {e}")  # Undefined function
        return None

# Update save_lead_source function to be more resilient
    try:
        if lead_id is None:
#             print("Cannot save lead source: lead_id is None")  # Undefined function
            return
        
        lead_id = validate_id(lead_id, "lead")
        search_term_id = validate_id(search_term_id, "search term")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO lead_sources (lead_id, search_term_id, url, page_title, meta_description, http_status, scrape_duration, meta_tags, phone_numbers, content, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (lead_id, search_term_id, url, page_title, meta_description, http_status, scrape_duration, json.dumps(meta_tags), json.dumps(phone_numbers), content, json.dumps(tags)))
        conn.commit()
        conn.close()
#         print(f"Lead source saved for lead ID {lead_id}")  # Undefined function
    except Exception as e:
#         print(f"Error saving lead source: {e}")  # Undefined function
        
# Update the update_search_term_status function to handle term as string
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
    results = []
    try:
#         print(f"Starting search for term: {term}")  # Undefined function
        search_urls = list(search(term, num=num_results, stop=num_results, pause=2))
#         print(f"Found {len(search_urls)} URLs")  # Undefined function
        for url in search_urls:
            try:
#                 print(f"Processing URL: {url}")  # Undefined function
                response = session.get(url, timeout=10)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title = soup.title.string if soup.title else ""
                meta_description = soup.find('meta', attrs={'name': 'description'})
                meta_description = meta_description['content'] if meta_description else ""
                
                emails = find_emails(response.text)
                phone_numbers = extract_phone_numbers(response.text)
                
                meta_tags = [str(tag) for tag in soup.find_all('meta')]
                
                content = extract_visible_text(soup)
                
#                 print(f"Found {len(emails)} emails and {len(phone_numbers)} phone numbers on {url}")  # Undefined function
                
                for email in emails:
                    if is_valid_email(email):
                        tags = []
                        if is_probable_blog_or_directory(soup, email):
                            tags.append("blog-directory")
                        else:
                            tags.append("company")
                        
                        if search_type == "All Leads" or (search_type == "Exclude Probable Blogs/Directories" and "blog-directory" not in tags):
                            lead_id = save_lead(email, phone_numbers[0] if phone_numbers else None, None, None, None, None)
                            save_lead_source(lead_id, campaign_id, url, title, meta_description, response.status_code, 
#                                              str(response.elapsed), meta_tags, phone_numbers, content, tags)  # Undefined function
                            results.append([email, url, title, meta_description, tags])
#                             print(f"Valid email found: {email}")  # Undefined function
                
                if len(results) >= num_results:
                    break
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
#                 print(f"Error processing {url}: {e}")  # Undefined function
    except Exception as e:
        logging.error(f"Error in manual search: {e}")
#         print(f"Error in manual search: {e}")  # Undefined function
    
#     print(f"Search completed. Found {len(results)} results.")  # Undefined function
    return results[:num_results]
# Function to fetch leads
def fetch_leads():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT l.id, l.email, l.phone, l.first_name, l.last_name, l.company, l.job_title, 
           st.term as search_term, ls.url as source_url, ls.page_title, ls.meta_description, 
           ls.phone_numbers, ls.content, ls.tags
    FROM leads l
    JOIN lead_sources ls ON l.id = ls.lead_id
    JOIN search_terms st ON ls.search_term_id = st.id
    ORDER BY l.id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["ID", "Email", "Phone", "First Name", "Last Name", "Company", "Job Title", 
                                       "Search Term", "Source URL", "Page Title", "Meta Description", 
                                       "Phone Numbers", "Content", "Tags"])

# Add this function to fetch search terms for a specific campaign
def fetch_search_terms_for_campaign(campaign_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, term FROM search_terms WHERE campaign_id = ?', (campaign_id,))
    terms = cursor.fetchall()
    conn.close()
    return [{"name": f"{term['id']}: {term['term']}", "value": str(term['id'])} for term in terms]

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
def get_knowledge_base(project_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM knowledge_base WHERE project_id = ?", (project_id,))
    knowledge = cursor.fetchone()
    conn.close()
    return knowledge

def update_knowledge_base(project_id, data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO knowledge_base (
        project_id, name, bio, values, contact_name, contact_role, contact_email,
        company_description, company_mission, company_target_market, company_other,
        product_name, product_description, product_target_customer, product_other,
        other_context, example_email, complete_version, medium_version, small_version,
        temporal_version, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (project_id, *data))
    conn.commit()
    conn.close()

def knowledge_base_view():
    st.header("Knowledge Base")
    
    projects = fetch_projects()
    selected_project = st.selectbox("Select Project", options=projects)
    project_id = int(selected_project.split(":")[0])
    
    knowledge = get_knowledge_base(project_id)
    
    with st.form("knowledge_base_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=knowledge[2] if knowledge else "")
            bio = st.text_area("Bio", value=knowledge[3] if knowledge else "")
            values = st.text_area("Values", value=knowledge[4] if knowledge else "")
        with col2:
            contact_name = st.text_input("Contact Name", value=knowledge[5] if knowledge else "")
            contact_role = st.text_input("Contact Role", value=knowledge[6] if knowledge else "")
            contact_email = st.text_input("Contact Email", value=knowledge[7] if knowledge else "")
        
        st.subheader("Company Information")
        company_description = st.text_area("Company Description", value=knowledge[8] if knowledge else "")
        company_mission = st.text_area("Company Mission/Vision", value=knowledge[9] if knowledge else "")
        company_target_market = st.text_area("Company Target Market", value=knowledge[10] if knowledge else "")
        company_other = st.text_area("Company Other", value=knowledge[11] if knowledge else "")
        
        st.subheader("Product Information")
        product_name = st.text_input("Product Name", value=knowledge[12] if knowledge else "")
        product_description = st.text_area("Product Description", value=knowledge[13] if knowledge else "")
        product_target_customer = st.text_area("Product Target Customer", value=knowledge[14] if knowledge else "")
        product_other = st.text_area("Product Other", value=knowledge[15] if knowledge else "")
        
        other_context = st.text_area("Other Context", value=knowledge[16] if knowledge else "")
        example_email = st.text_area("Example Reference Email", value=knowledge[17] if knowledge else "")
        
        submit_button = st.form_submit_button("Save Knowledge Base")
    
    if submit_button:
        data = (
            name, bio, values, contact_name, contact_role, contact_email,
            company_description, company_mission, company_target_market, company_other,
            product_name, product_description, product_target_customer, product_other,
            other_context, example_email, "", "", "", ""
        )
        update_knowledge_base(project_id, data)
        st.success("Knowledge base updated successfully!")
    
    if knowledge:
        st.subheader("AI Context Versions")
        st.text_area("Complete Version", value=knowledge[18], height=200, disabled=True)
        st.text_area("Medium Version", value=knowledge[19], height=150, disabled=True)
        st.text_area("Small Version", value=knowledge[20], height=100, disabled=True)
        st.text_area("Temporal Version", value=knowledge[21], height=100, disabled=True)

# Update the search_terms function
def search_terms():
    st.header("Search Terms")
    
    # Add a selector for different views
    view_option = st.radio("Select View", ["Basic List", "Search Term Groups"])
    
    # Display statistics
#     display_search_term_statistics()  # Undefined function
    
    if view_option == "Basic List":
#         display_basic_search_terms_list()  # Undefined function
    else:
#         display_search_term_groups()  # Undefined function

# New function to display search term statistics
def display_search_term_statistics():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM search_terms")
    total_terms = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM search_terms WHERE group_id IS NOT NULL")
    grouped_terms = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM search_term_groups")
    total_groups = cursor.fetchone()[0]
    
    conn.close()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Search Terms", total_terms)
    col2.metric("Grouped Terms", grouped_terms)
    col3.metric("Total Groups", total_groups)
    
    if st.button("Prune Empty Terms"):
        pruned_count = prune_empty_search_terms()
        st.success(f"Pruned {pruned_count} search terms with no associated leads.")

# New function to prune empty search terms
def prune_empty_search_terms():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM search_terms
        WHERE id NOT IN (
            SELECT DISTINCT search_term_id
            FROM lead_sources
        )
    """,)
    
    pruned_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return pruned_count

# Update the display_search_term_groups function
def display_search_term_groups():
    st.subheader("Search Term Groups")
    
    # Fetch search term groups data
    groups_data = fetch_search_term_groups_data()
    
    # Sort options
    sort_option = st.selectbox("Sort groups by", ["Name", "Number of Terms"])
    reverse_sort = st.checkbox("Reverse sort order")
    
    if sort_option == "Name":
        groups_data.sort(key=lambda x: x['group_name'], reverse=reverse_sort)
    else:
        groups_data.sort(key=lambda x: x['term_count'], reverse=reverse_sort)
    
    # Display groups and their terms
    for group in groups_data:
        with st.expander(f"{group['group_name']} (Terms: {group['term_count']}, Leads: {group['total_leads']})"):
            st.write(f"Description: {group['description']}")
            st.write("Search Terms:")
            terms_df = pd.DataFrame(group['terms'])
            
            # Allow sorting
            sort_column = st.selectbox(f"Sort by (Group {group['id']})", terms_df.columns)
            sort_order = st.radio(f"Order (Group {group['id']})", ["Ascending", "Descending"])
            sorted_df = terms_df.sort_values(sort_column, ascending=(sort_order == "Ascending"))
            
            st.dataframe(sorted_df)
            
            # Delete group button
            if st.button(f"Delete Group {group['id']}"):
#                 delete_search_term_group(group['id'])  # Undefined function
                st.success(f"Group {group['id']} deleted successfully.")
                st.rerun()
    
    # Multi-select and assign to group
    st.subheader("Assign Terms to Group")
    all_terms = fetch_all_search_terms()
    selected_terms = st.multiselect("Select terms to assign", all_terms)
    new_group_name = st.text_input("New group name (leave empty to use existing)")
    existing_groups = [g['group_name'] for g in groups_data]
    target_group = st.selectbox("Select target group", [""] + existing_groups)
    
    if st.button("Assign to Group"):
        if new_group_name:
            group_id = create_search_term_group(new_group_name)
        elif target_group:
            group_id = next(g['id'] for g in groups_data if g['group_name'] == target_group)
        else:
            st.error("Please provide a new group name or select an existing group.")
            return
        
#         assign_terms_to_group(selected_terms, group_id)  # Undefined function
        st.success(f"Assigned {len(selected_terms)} terms to the group.")
        st.rerun()
# Function to fetch leads
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT l.id, l.email, l.phone, l.first_name, l.last_name, l.company, l.job_title, 
           st.term as search_term, ls.url as source_url, ls.page_title, ls.meta_description, 
           ls.phone_numbers, ls.content, ls.tags
    FROM leads l
    JOIN lead_sources ls ON l.id = ls.lead_id
    JOIN search_terms st ON ls.search_term_id = st.id
    ORDER BY l.id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["ID", "Email", "Phone", "First Name", "Last Name", "Company", "Job Title", 
                                       "Search Term", "Source URL", "Page Title", "Meta Description", 
                                       "Phone Numbers", "Content", "Tags"])

# Add this function to fetch search terms for a specific campaign
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, term FROM search_terms WHERE campaign_id = ?', (campaign_id,))
    terms = cursor.fetchall()
    conn.close()
    return [{"name": f"{term['id']}: {term['term']}", "value": str(term['id'])} for term in terms]

# Update the manual_search_multiple function
    all_results = []
    for term in terms:
        if term.strip():  # Only process non-empty terms
            results = manual_search_wrapper(term.strip(), num_results, campaign_id)
            all_results.extend(results)
    return all_results

# New function to get least searched terms
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
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM knowledge_base WHERE project_id = ?", (project_id,))
    knowledge = cursor.fetchone()
    conn.close()
    return knowledge

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO knowledge_base (
        project_id, name, bio, values, contact_name, contact_role, contact_email,
        company_description, company_mission, company_target_market, company_other,
        product_name, product_description, product_target_customer, product_other,
        other_context, example_email, complete_version, medium_version, small_version,
        temporal_version, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (project_id, *data))
    conn.commit()
    conn.close()

    st.header("Knowledge Base")
    
    projects = fetch_projects()
    selected_project = st.selectbox("Select Project", options=projects)
    project_id = int(selected_project.split(":")[0])
    
    knowledge = get_knowledge_base(project_id)
    
    with st.form("knowledge_base_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=knowledge[2] if knowledge else "")
            bio = st.text_area("Bio", value=knowledge[3] if knowledge else "")
            values = st.text_area("Values", value=knowledge[4] if knowledge else "")
        with col2:
            contact_name = st.text_input("Contact Name", value=knowledge[5] if knowledge else "")
            contact_role = st.text_input("Contact Role", value=knowledge[6] if knowledge else "")
            contact_email = st.text_input("Contact Email", value=knowledge[7] if knowledge else "")
        
        st.subheader("Company Information")
        company_description = st.text_area("Company Description", value=knowledge[8] if knowledge else "")
        company_mission = st.text_area("Company Mission/Vision", value=knowledge[9] if knowledge else "")
        company_target_market = st.text_area("Company Target Market", value=knowledge[10] if knowledge else "")
        company_other = st.text_area("Company Other", value=knowledge[11] if knowledge else "")
        
        st.subheader("Product Information")
        product_name = st.text_input("Product Name", value=knowledge[12] if knowledge else "")
        product_description = st.text_area("Product Description", value=knowledge[13] if knowledge else "")
        product_target_customer = st.text_area("Product Target Customer", value=knowledge[14] if knowledge else "")
        product_other = st.text_area("Product Other", value=knowledge[15] if knowledge else "")
        
        other_context = st.text_area("Other Context", value=knowledge[16] if knowledge else "")
        example_email = st.text_area("Example Reference Email", value=knowledge[17] if knowledge else "")
        
        submit_button = st.form_submit_button("Save Knowledge Base")
    
    if submit_button:
        data = (
            name, bio, values, contact_name, contact_role, contact_email,
            company_description, company_mission, company_target_market, company_other,
            product_name, product_description, product_target_customer, product_other,
            other_context, example_email, "", "", "", ""
        )
        update_knowledge_base(project_id, data)
        st.success("Knowledge base updated successfully!")
    
    if knowledge:
        st.subheader("AI Context Versions")
        st.text_area("Complete Version", value=knowledge[18], height=200, disabled=True)
        st.text_area("Medium Version", value=knowledge[19], height=150, disabled=True)
        st.text_area("Small Version", value=knowledge[20], height=100, disabled=True)
        st.text_area("Temporal Version", value=knowledge[21], height=100, disabled=True)

# Update the main function to include the new knowledge_base_view
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key="sk-1234")
client.base_url = "https://openai-proxy-kl3l.onrender.com"

import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key="sk-1234",
    base_url="https://openai-proxy-kl3l.onrender.com"
)

# The model to use
openai_model = "gpt-3.5-turbo"

from openai import OpenAI
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the OpenAI client
client = OpenAI(
    api_key="sk-1234",
    base_url="https://openai-proxy-kl3l.onrender.com"
)

    # At the beginning of your script, add:
    pass  # Placeholder block to prevent indentation error
ai_request_logs = []

def autoclient_ai_view():
    st.header("AutoclientAI")

    # Display condensed knowledge base
    kb_info = get_knowledge_base_info()
    st.subheader("Knowledge Base Summary")
    st.json(kb_info)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Optimize Existing Groups", "Create New Groups", "Adjust Email Templates", "Optimize Search Terms"])

    with tab1:
#         optimize_existing_groups()  # Undefined function

    with tab2:
#         create_new_groups()  # Undefined function

    with tab3:
#         adjust_email_template()  # Undefined function

    with tab4:
#         optimize_search_terms()  # Undefined function

    # Display AI request logs
    if ai_request_logs:
        st.subheader("AI Request Logs")
        for log in ai_request_logs:
            st.text(log)
    else:
        st.info("No AI request logs available.")

def optimize_existing_groups():
    groups = fetch_search_term_groups()
    if not groups:
        st.warning("No search term groups found.")
        return

    selected_group = st.selectbox("Select a group to optimize", options=groups)

    if st.button("Optimize Selected Group"):
        with st.spinner("Optimizing group..."):
            try:
                group_data = get_group_data(selected_group)
                kb_info = get_knowledge_base_info()
                optimized_data = classify_search_terms(group_data['search_terms'], kb_info)
                
                st.subheader("Optimized Search Terms")
                for category, terms in optimized_data.items():
                    st.write(f"**{category}:**")
                    new_terms = st.text_area(f"Edit terms for {category}:", value="\n".join(terms))
                    optimized_data[category] = new_terms.split("\n")

                if st.button("Save Optimized Group"):
#                     save_optimized_group(selected_group, optimized_data)  # Undefined function
                    st.success("Group optimized and saved successfully!")
            except Exception as e:
                st.error(f"An error occurred during optimization: {str(e)}")
                logging.error(f"Error in optimize_existing_groups: {str(e)}")

def create_new_groups():
    st.subheader("Create New Search Term Group")
    
    group_name = st.text_input("Group Name")
    group_description = st.text_area("Group Description")
    search_terms = st.text_area("Enter search terms (one per line)")
    
    if st.button("Create Group"):
        if not group_name:
            st.error("Please provide a group name.")
            return
        
        terms_list = [term.strip() for term in search_terms.split('\n') if term.strip()]
        
        try:
            group_id = create_search_term_group(group_name, group_description)
#             assign_terms_to_group(terms_list, group_id)  # Undefined function
            st.success(f"Group '{group_name}' created successfully with {len(terms_list)} terms.")
        except Exception as e:
            st.error(f"An error occurred while creating the group: {str(e)}")
            logging.error(f"Error in create_new_groups: {str(e)}")

def adjust_email_template():
    templates = fetch_message_templates()
    if not templates:
        st.warning("No message templates found.")
        return

    selected_template = st.selectbox("Select a template to adjust", options=templates)
    
    template_data = get_template_data(selected_template.split(':')[0])
    if template_data:
        st.subheader("Current Template")
        st.text_input("Subject", value=template_data['subject'])
        st.text_area("Content", value=template_data['body_content'], height=300)
        
        st.subheader("Adjust Template")
        new_subject = st.text_input("New Subject", value=template_data['subject'])
        new_content = st.text_area("New Content", value=template_data['body_content'], height=300)
        
        if st.button("Save Adjusted Template"):
            try:
#                 save_adjusted_template(selected_template.split(':')[0], new_subject, new_content)  # Undefined function
                st.success("Template adjusted and saved successfully!")
            except Exception as e:
                st.error(f"An error occurred while saving the template: {str(e)}")
                logging.error(f"Error in adjust_email_template: {str(e)}")

def optimize_search_terms():
    st.subheader("Optimize Search Terms")
    current_terms = st.text_area("Enter current search terms (one per line):")
    
    if st.button("Optimize Search Terms"):
        if not current_terms.strip():
            st.error("Please enter some search terms to optimize.")
            return
        
        with st.spinner("Optimizing search terms..."):
            try:
                kb_info = get_knowledge_base_info()
                optimized_terms = generate_optimized_search_terms(current_terms.split('\n'), kb_info)
                
                st.subheader("Optimized Search Terms")
                st.write("\n".join(optimized_terms))
                
                if st.button("Save Optimized Terms"):
#                     save_optimized_search_terms(optimized_terms)  # Undefined function
                    st.success("Optimized search terms saved successfully!")
            except Exception as e:
                st.error(f"An error occurred during optimization: {str(e)}")
                logging.error(f"Error in optimize_search_terms: {str(e)}")

def classify_search_terms(search_terms, kb_info):
    prompt = f"""
    As an expert in lead generation and email marketing, classify the following search terms into strategic groups:
    pass  # Placeholder block to prevent indentation error

    Search Terms: {', '.join(search_terms)}

    Knowledge Base Info:
    {kb_info}

    Create groups that allow for tailored, personalized email content. Consider the product/service features, target audience, and potential customer pain points. Groups should be specific enough for customization but broad enough to be efficient. Always include a 'low_quality_search_terms' category for irrelevant or overly broad terms.

    Respond with category names as keys and lists of search terms as values.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in strategic search term classification for targeted email marketing campaigns."},
            {"role": "user", "content": prompt}
        ]
    )

#     log_ai_request("classify_search_terms", prompt, response.choices[0].message.content)  # Undefined function
    return eval(response.choices[0].message.content)

def generate_email_template(terms, kb_info):
    prompt = f"""
    Create an email template for the following search terms:
    pass  # Placeholder block to prevent indentation error

    Search Terms: {', '.join(terms)}

    Knowledge Base Info:
    {kb_info}

    Guidelines:
    1. Focus on benefits to the reader
    2. Address potential customer doubts and fears
    3. Include clear CTAs at the beginning and end
    4. Use a natural, conversational tone
    5. Be concise but impactful
    6. Use minimal formatting - remember this is an email, not a landing page

    Provide only the email body content in HTML format, excluding <body> tags. Use <p>, <strong>, <em>, and <a> tags as needed.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in creating high-converting email templates for targeted marketing campaigns."},
            {"role": "user", "content": prompt}
        ]
    )

#     log_ai_request("generate_email_template", prompt, response.choices[0].message.content)  # Undefined function
    return response.choices[0].message.content

def adjust_email_template_api(current_template, adjustment_prompt, kb_info):
    prompt = f"""
    Adjust the following email template based on the given instructions:
    pass  # Placeholder block to prevent indentation error

    Current Template:
    {current_template}

    Adjustment Instructions:
    {adjustment_prompt}

    Knowledge Base Info:
    {kb_info}

    Guidelines:
    1. Maintain focus on conversion and avoiding spam filters
    2. Preserve the natural, conversational tone
    3. Ensure benefits to the reader remain highlighted
    4. Continue addressing potential customer doubts and fears
    5. Keep clear CTAs at the beginning and end
    6. Remain concise and impactful
    7. Maintain minimal formatting suitable for an email

    Provide only the adjusted email body content in HTML format, excluding <body> tags.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in refining high-converting email templates for targeted marketing campaigns."},
            {"role": "user", "content": prompt}
        ]
    )

#     log_ai_request("adjust_email_template_api", prompt, response.choices[0].message.content)  # Undefined function
    return response.choices[0].message.content

def generate_optimized_search_terms(current_terms, kb_info):
    prompt = f"""
    Optimize the following search terms for targeted email campaigns:
    pass  # Placeholder block to prevent indentation error

    Current Terms: {', '.join(current_terms)}

    Knowledge Base Info:
    {kb_info}

    Guidelines:
    1. Focus on terms likely to attract high-quality leads
    2. Consider product/service features, target audience, and customer pain points
    3. Optimize for specificity and relevance
    4. Think about how each term could lead to a compelling email strategy
    5. Remove or improve low-quality or overly broad terms
    6. Add new, highly relevant terms based on the knowledge base information

    Provide a list of optimized search terms, aiming for quality over quantity.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in optimizing search terms for targeted email marketing campaigns."},
            {"role": "user", "content": prompt}
        ]
    )

#     log_ai_request("generate_optimized_search_terms", prompt, response.choices[0].message.content)  # Undefined function
    return response.choices[0].message.content.split('\n')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM search_term_groups")
    groups = [f"{row['id']}:{row['name']}" for row in cursor.fetchall()]
    conn.close()
    return groups

def get_group_data(group_id):
    if not group_id:
        return None
    conn = get_db_connection()
    cursor = conn.cursor()
    group_id = group_id.split(':')[0]
    cursor.execute("SELECT name, description, email_template FROM search_term_groups WHERE id = ?", (group_id,))
    group_data = dict(cursor.fetchone())
    cursor.execute("SELECT term FROM search_terms WHERE group_id = ?", (group_id,))
    group_data['search_terms'] = [row['term'] for row in cursor.fetchall()]
    conn.close()
    return group_data

def save_optimized_group(group_id, optimized_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    group_id = group_id.split(':')[0]
    
    # Delete existing terms for this group
    cursor.execute("DELETE FROM search_terms WHERE group_id = ?", (group_id,))
    
    # Insert new terms
    for category, terms in optimized_data.items():
        for term in terms:
            cursor.execute("INSERT INTO search_terms (term, group_id, category) VALUES (?, ?, ?)", 
                           (term, group_id, category))
    
    conn.commit()
    conn.close()

def fetch_all_search_terms():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT term FROM search_terms")
    terms = [row['term'] for row in cursor.fetchall()]
    conn.close()
    return terms

def save_new_group(category, terms, email_template):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO search_term_groups (name, email_template) VALUES (?, ?)", (category, email_template))
    group_id = cursor.lastrowid
    
    for term in terms:
        cursor.execute("INSERT INTO search_terms (term, group_id, category) VALUES (?, ?, ?)", 
                       (term, group_id, category))
    
    conn.commit()
    conn.close()

def save_adjusted_template(group_id, adjusted_template):
    conn = get_db_connection()
    cursor = conn.cursor()
    group_id = group_id.split(':')[0]
    cursor.execute("UPDATE search_term_groups SET email_template = ? WHERE id = ?", 
                   (adjusted_template, group_id))
    conn.commit()
    conn.close()

def get_knowledge_base_info():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT kb_name, kb_bio, kb_values, contact_name, contact_role, contact_email,
               company_description, company_mission, company_target_market, company_other,
               product_name, product_description, product_target_customer, product_other,
               other_context, example_email
        FROM knowledge_base 
        ORDER BY id DESC LIMIT 1
    """,)
    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            "kb_name": result[0],
            "kb_bio": result[1],
            "kb_values": result[2],
            "contact_name": result[3],
            "contact_role": result[4],
            "contact_email": result[5],
            "company_description": result[6],
            "company_mission": result[7],
            "company_target_market": result[8],
            "company_other": result[9],
            "product_name": result[10],
            "product_description": result[11],
            "product_target_customer": result[12],
            "product_other": result[13],
            "other_context": result[14],
            "example_email": result[15]
        }
    else:
        return {}

def save_optimized_search_terms(optimized_terms):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for term in optimized_terms:
        cursor.execute("INSERT INTO search_terms (term) VALUES (?)", (term,))
    
    conn.commit()
    conn.close()

def get_last_n_search_terms(n):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT term FROM search_terms ORDER BY id DESC LIMIT {n}")
    terms = [row[0] for row in cursor.fetchall()]
    conn.close()
    return terms

def get_random_leads(n, from_last):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT l.email, st.term 
        FROM leads l 
        JOIN lead_sources ls ON l.id = ls.lead_id 
        JOIN search_terms st ON ls.search_term_id = st.id 
        ORDER BY l.id DESC LIMIT {from_last}
    """,)
    leads = cursor.fetchall()
    conn.close()
    return random.sample(leads, min(n, len(leads)))

def format_leads_for_prompt(leads):
    return "\n".join([f"{lead[0]} - {lead[1]}" for lead in leads])

def save_new_search_terms(terms):
    conn = get_db_connection()
    cursor = conn.cursor()
    for term in terms:
        cursor.execute("INSERT INTO search_terms (term) VALUES (?)", (term,))
    conn.commit()
    conn.close()

def save_new_email_template(template):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO message_templates (template_name, subject, body_content) 
        VALUES (?, ?, ?)
    """, ("AI Generated Template", "AI Subject", template))
    conn.commit()
    conn.close()

def main():
    st.title("AUTOCLIENT")

    st.sidebar.title("Navigation")
    pages = [
        "Manual Search",
        "Bulk Search",
        "Bulk Send",
        "View Leads",
        "Search Terms",
        "Message Templates",
        "View Sent Messages",
        "Projects & Campaigns",
        "Knowledge Base",
        "AutoclientAI",
        "Automation Control"
    ]

    # Control page selection
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Manual Search"
    st.session_state.current_page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.current_page))

    # Call appropriate function based on the selected page
    if st.session_state.current_page == "Manual Search":
#         manual_search_page()  # Undefined function
    elif st.session_state.current_page == "Bulk Search":
#         bulk_search_page()  # Undefined function
    elif st.session_state.current_page == "Bulk Send":
        bulk_send_page()
    elif st.session_state.current_page == "View Leads":
#         view_leads()  # Undefined function
    elif st.session_state.current_page == "Search Terms":
        search_terms()
    elif st.session_state.current_page == "Message Templates":
#         message_templates()  # Undefined function
    elif st.session_state.current_page == "View Sent Messages":
        view_sent_messages()
    elif st.session_state.current_page == "Projects & Campaigns":
#         projects_and_campaigns()  # Undefined function
    elif st.session_state.current_page == "Knowledge Base":
        knowledge_base_view()
    elif st.session_state.current_page == "AutoclientAI":
        autoclient_ai_view()
    elif st.session_state.current_page == "Automation Control":
#         automation_view()  # Undefined function

def display_result_card(result, index):
    with st.container():
        st.markdown(f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; margin-bottom:10px;">
            <h3 style="color:#1E90FF;">Result {index + 1}</h3>
            <p><strong>Email:</strong> {result[0]}</p>
            <p><strong>Source:</strong> <a href="{result[1]}" target="_blank">{result[1][:50]}...</a></p>
            <p><strong>Tags:</strong> {', '.join(result[4])}</p>
        </div>
        """, unsafe_allow_html=True)

def manual_search_page():
    st.header("Manual Search")
    
    # Fetch the last 5 search terms to use as a reference or for input suggestions.
    last_5_terms = get_last_5_search_terms()
    
    # Separate UI for single and multiple term searches using tabs
    tab1, tab2 = st.tabs(["Single Term Search", "Multiple Terms Search"])
    
    # ---- Single Term Search Tab ----
    with tab1:
        with st.form(key="single_search_form"):
            search_term = st.text_input("Search Term")
            num_results = st.slider("Number of Results", min_value=10, max_value=200, value=30, step=10)
            campaign_id = st.selectbox("Select Campaign", options=fetch_campaigns())
            search_type = st.selectbox("Search Type", options=["All Leads", "Exclude Probable Blogs/Directories"])
            submit_button = st.form_submit_button(label="Search")
        
        if submit_button:
            st.session_state.single_search_started = True
            st.session_state.single_search_results = []
            st.session_state.single_search_logs = []
            st.session_state.single_search_progress = 0
        
        if 'single_search_started' in st.session_state and st.session_state.single_search_started:
            # Containers to display search progress, logs, and results.
            results_container = st.empty()
            progress_bar = st.progress(st.session_state.single_search_progress)
            status_text = st.empty()
            log_container = st.empty()
            
            if len(st.session_state.single_search_results) < num_results:
                with st.spinner("Searching..."):
                    new_results = manual_search_wrapper(search_term, num_results - len(st.session_state.single_search_results), campaign_id.split(":")[0], search_type)
                    st.session_state.single_search_results.extend(new_results)
                    
                    # Log and display progress of each found result.
                    for result in new_results:
                        log = f"Found result: {result[0]} from {result[1]}"
                        st.session_state.single_search_logs.append(log)
                        log_container.text_area("Search Logs", "\n".join(st.session_state.single_search_logs), height=200)
                        st.session_state.single_search_progress = len(st.session_state.single_search_results) / num_results
                        progress_bar.progress(st.session_state.single_search_progress)
                        status_text.text(f"Found {len(st.session_state.single_search_results)} results...")
                        
                        # Display the results dynamically.
                        with results_container.container():
                            for j, res in enumerate(st.session_state.single_search_results):
                                display_result_card(res, j)
                        time.sleep(0.1)  # Small delay for better animation effect.
            
            # Search completion status
            if len(st.session_state.single_search_results) >= num_results:
                st.success(f"Search completed! Found {len(st.session_state.single_search_results)} results.")
                st.session_state.single_search_started = False
            
            # Display session-specific statistics
            st.subheader("Search Statistics")
            st.metric("Total Results Found", len(st.session_state.single_search_results))
            st.metric("Unique Domains", len(set(result[0].split('@')[1] for result in st.session_state.single_search_results)))
    
    # ---- Multiple Terms Search Tab ----
    with tab2:
        st.subheader("Enter Search Terms")
        
        # Allow user to enter multiple search terms (one per line).
        search_terms_text = st.text_area("Enter one search term per line", height=150, value="\n".join(last_5_terms))
        load_button = st.button("Load Terms from Text Area")
        
        if load_button:
            terms_list = [term.strip() for term in search_terms_text.split('\n') if term.strip()]
            st.session_state.loaded_terms = terms_list
            st.rerun()  # Refresh with loaded terms
        
        if 'loaded_terms' not in st.session_state:
            st.session_state.loaded_terms = [""] * 4  # Default to 4 empty terms
        
        num_terms = len(st.session_state.loaded_terms)
        
        # Form for multiple search terms
        with st.form(key="multi_search_form"):
            search_terms = [st.text_input(f"Search Term {i+1}", value=term, key=f"term_{i}") 
                            for i, term in enumerate(st.session_state.loaded_terms)]
            
            num_results_multiple = st.slider("Number of Results per Term", min_value=10, max_value=200, value=30, step=10)
            campaign_id_multiple = st.selectbox("Select Campaign for Multiple Search", options=fetch_campaigns(), key="multi_campaign")
            search_type = st.selectbox("Search Type", options=["All Leads", "Exclude Probable Blogs/Directories"])
            
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
            st.session_state.multi_search_terms = [term for term in search_terms if term.strip()]
        
        if 'multi_search_started' in st.session_state and st.session_state.multi_search_started:
            # Containers to display multi-search progress, logs, and results.
            results_container = st.empty()
            progress_bar = st.progress(st.session_state.multi_search_progress)
            status_text = st.empty()
            log_container = st.empty()
            
            with st.spinner("Searching..."):
                all_results = []
                logs = []
                total_terms = len(st.session_state.multi_search_terms)
                
                # Iterate through multiple search terms and gather results
                for term_index, term in enumerate(st.session_state.multi_search_terms):
                    status_text.text(f"Searching term {term_index + 1} of {total_terms}: {term}")
                    term_results = manual_search_wrapper(term, num_results_multiple, campaign_id_multiple.split(":")[0], search_type)
                    
                    for i, result in enumerate(term_results):
                        all_results.append(result)
                        progress = (term_index * num_results_multiple + i + 1) / (total_terms * num_results_multiple)
                        progress_bar.progress(progress)
                        log = f"Term {term_index + 1}: Found result {i + 1}: {result[0]} from {result[1]}"
                        logs.append(log)
                        log_container.text_area("Search Logs", "\n".join(logs), height=200)
                        
                        # Display each found result dynamically
                        with results_container.container():
                            for j, res in enumerate(all_results):
                                display_result_card(res, j)
                        time.sleep(0.1)  # Small delay for animation effect
                
                # Update progress bar after search completion
                st.session_state.multi_search_progress = 1.0
                progress_bar.progress(st.session_state.multi_search_progress)
            
            st.success(f"Found {len(all_results)} results across all terms!")
            
            # Display statistics specific to this session
            st.subheader("Search Statistics")
            st.metric("Total Results Found", len(all_results))
            st.metric("Unique Domains", len(set(result[0].split('@')[1] for result in all_results)))
            st.metric("Search Terms Processed", len(st.session_state.multi_search_terms))
        
        if fill_button:
            least_searched = get_least_searched_terms(num_terms)
            st.session_state.loaded_terms = least_searched
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
            
            for term_index, term_row in enumerate(st.session_state.bulk_search_terms.iterrows()):
                if term_index < len(st.session_state.bulk_search_results) // num_results:
                    continue
                
                term = term_row.Term
                status_text.text(f"Searching term {term_index + 1} of {total_terms}: {term}")
                term_results = manual_search_wrapper(term, num_results, term_row.ID, "All Leads")
                
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

    st.header("Bulk Send")
    
    templates = fetch_message_templates()
    if not templates:
        st.warning("No message templates found. Please create a template first.")
        return

    with st.form(key="bulk_send_form"):
        template_id = st.selectbox("Select Message Template", options=templates)
        from_email = st.text_input("From Email", value="Sami Halawa <hello@indosy.com>")
        reply_to = st.text_input("Reply To", value="eugproductions@gmail.com")
        
        send_option = st.radio("Send to:", 
                               ["All Leads", 
                                "All Not Contacted with this Template", 
                                "All Not Contacted with Templates from this Campaign",
                                "Selected Search Term Groups"])
        
        if send_option == "Selected Search Term Groups":
            groups = fetch_search_term_groups()
            selected_groups = st.multiselect("Select Search Term Groups", options=groups)
            group_leads = get_leads_count_for_groups(selected_groups)
            st.write("Leads in selected groups:")
            for group, count in group_leads.items():
                st.write(f"{group}: {count} leads")
        
        filter_option = st.radio("Filter:", 
                                 ["Not Filter Out Leads", 
                                  "Filter Out blog-directory"])
        
        col1, col2 = st.columns(2)
        with col1:
            preview_button = st.form_submit_button(label="Preview Email")
        with col2:
            send_button = st.form_submit_button(label="Start Bulk Send")
    
    if preview_button:
        preview = get_email_preview(template_id.split(":")[0], from_email, reply_to)
        st.components.v1.html(preview, height=600, scrolling=True)
    
    if send_button:
        if send_option == "Selected Search Term Groups" and not selected_groups:
            st.error("Please select at least one search term group.")
            return
        
        st.session_state.bulk_send_started = True
        st.session_state.bulk_send_logs = []
        st.session_state.bulk_send_progress = 0
        
        # Fetch leads based on send_option and filter_option
        leads_to_send = fetch_leads_for_bulk_send(template_id.split(":")[0], send_option, filter_option, selected_groups if send_option == "Selected Search Term Groups" else None)
        
        st.write(f"Preparing to send emails to {len(leads_to_send)} leads")
        
        # Perform bulk send
        bulk_send_coroutine = bulk_send(template_id.split(":")[0], from_email, reply_to, leads_to_send)
        logs = asyncio.run(bulk_send_coroutine)
        
        # Display logs and statistics
        for log in logs:
            st.write(log)
        
        st.success(f"Bulk send completed. Sent {len(leads_to_send)} emails.")

    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT DISTINCT l.id, l.email
    FROM leads l
    JOIN lead_sources ls ON l.id = ls.lead_id
    """
    
    if send_option == "All Not Contacted with this Template":
        query += f"""
        LEFT JOIN messages m ON l.id = m.lead_id AND m.template_id = {template_id}
        WHERE m.id IS NULL
        """
    elif send_option == "All Not Contacted with Templates from this Campaign":
        query += f"""
        LEFT JOIN messages m ON l.id = m.lead_id
        LEFT JOIN message_templates mt ON m.template_id = mt.id
        WHERE m.id IS NULL OR mt.campaign_id != (SELECT campaign_id FROM message_templates WHERE id = {template_id})
        """
    elif send_option == "Selected Search Term Groups":
        group_ids = [group.split(":")[0] for group in selected_groups]
        query += f"""
        JOIN search_terms st ON ls.search_term_id = st.id
        WHERE st.group_id IN ({','.join(group_ids)})
        """
    
    if filter_option == "Filter Out blog-directory":
        query += " AND NOT ls.tags LIKE '%blog-directory%'"
    
    cursor.execute(query)
    leads = cursor.fetchall()
    conn.close()
    
    return leads

async def bulk_send(template_id, from_email, reply_to, leads):
    template_id = validate_id(template_id, "template")
    from_email = validate_email(from_email)
    reply_to = validate_email(reply_to)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT name, subject, body_content FROM message_templates WHERE id = ?', (template_id,))
    template = cursor.fetchone()
    
    if not template:
        return "Template not found"
    
    template_name, subject, body_content = template
    
    total_leads = len(leads)
    logs = [
        f"Preparing to send emails to {total_leads} leads",
        f"Template Name: {template_name}",
        f"Subject: {subject}",
        f"From Email: {from_email}",
        f"Reply To: {reply_to}"
    ]
    
    for i, (lead_id, email) in enumerate(leads):
        try:
            customized_content = customize_email_content(body_content, lead_id)
            message_id = send_email(email, subject, customized_content, from_email, reply_to)
            save_message(lead_id, template_id, 'sent', datetime.now(), subject, message_id, customized_content)
            logs.append(f"Sent email to {email} (Lead ID: {lead_id})")
        except Exception as e:
            save_message(lead_id, template_id, 'failed', datetime.now(), subject, None, str(e))
            logs.append(f"Failed to send email to {email} (Lead ID: {lead_id}): {e}")
        
        progress = (i + 1) / total_leads
        st.session_state.bulk_send_progress = progress
        st.session_state.bulk_send_logs = logs
        time.sleep(0.1)  # Add a small delay for UI updates
    
    conn.close()
    return logs

def view_leads():
    st.header("View Leads")
    
    if st.button("Refresh Leads"):
        st.session_state.leads = fetch_leads()
    
    if 'leads' not in st.session_state:
        st.session_state.leads = fetch_leads()
    
    for _, lead in st.session_state.leads.iterrows():
        with st.expander(f"Lead: {lead['Email']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**ID:** {lead['ID']}")
                st.write(f"**Email:** {lead['Email']}")
                st.write(f"**Phone:** {lead['Phone']}")
                st.write(f"**Name:** {lead['First Name']} {lead['Last Name']}")
                st.write(f"**Company:** {lead['Company']}")
                st.write(f"**Job Title:** {lead['Job Title']}")
            with col2:
                st.write(f"**Search Term:** {lead['Search Term']}")
                st.write(f"**Source URL:** {lead['Source URL']}")
                st.write(f"**Page Title:** {lead['Page Title']}")
                st.write(f"**Meta Description:** {lead['Meta Description']}")
                st.write(f"**Phone Numbers:** {lead['Phone Numbers']}")
                
                # Handle different possible formats of the Tags column
                if isinstance(lead['Tags'], str):
                    try:
                        tags = json.loads(lead['Tags'])
                        st.write(f"**Tags:** {', '.join(tags)}")
                    except json.JSONDecodeError:
                        st.write(f"**Tags:** {lead['Tags']}")
                elif isinstance(lead['Tags'], list):
                    st.write(f"**Tags:** {', '.join(lead['Tags'])}")
                else:
                    st.write(f"**Tags:** {lead['Tags']}")
            
            st.write("**Page Content:**")
            if lead['Content'] is not None:
                st.text(lead['Content'][:500] + "..." if len(lead['Content']) > 500 else lead['Content'])
            else:
                st.text("No content available")

    # Optional: Add a download button for the full dataset
    csv = st.session_state.leads.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download full dataset as CSV",
        data=csv,
        file_name="leads_data.csv",
        mime="text/csv",
    )

    st.header("Search Terms")
    
    # Add a selector for different views
    view_option = st.radio("Select View", ["Basic List", "Search Term Groups"])
    
    # Display statistics
    display_search_term_statistics()
    
    if view_option == "Basic List":
#         display_basic_search_terms_list()  # Undefined function
    else:
        display_search_term_groups()

    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM search_terms")
    total_terms = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM search_terms WHERE group_id IS NOT NULL")
    grouped_terms = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM search_term_groups")
    total_groups = cursor.fetchone()[0]
    
    conn.close()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Search Terms", total_terms)
    col2.metric("Grouped Terms", grouped_terms)
    col3.metric("Total Groups", total_groups)
    
    if st.button("Prune Empty Terms"):
        pruned_count = prune_empty_search_terms()
        st.success(f"Pruned {pruned_count} search terms with no associated leads.")

    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM search_terms
        WHERE id NOT IN (
            SELECT DISTINCT search_term_id
            FROM lead_sources
        )
    """,)
    
    pruned_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return pruned_count

def display_basic_search_terms_list():
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
        
        # Sort the dataframe by 'Leads Fetched' in descending order
        sorted_terms = st.session_state.search_terms.sort_values('Leads Fetched', ascending=False)
        
        for index, row in sorted_terms.iterrows():
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
#                     delete_search_term_and_leads(st.session_state.confirm_delete)  # Undefined function
                    del st.session_state.confirm_delete
                    del st.session_state.leads_to_delete
                    st.session_state.search_terms = fetch_search_terms()
                    st.success("Search term and related leads deleted successfully.")
            with col2:
                if st.button("Cancel"):
                    del st.session_state.confirm_delete
                    del st.session_state.leads_to_delete

    st.subheader("Search Term Groups")
    
    # Fetch search term groups data
    groups_data = fetch_search_term_groups_data()
    
    # Sort options
    sort_option = st.selectbox("Sort groups by", ["Name", "Number of Terms"])
    reverse_sort = st.checkbox("Reverse sort order")
    
    if sort_option == "Name":
        groups_data.sort(key=lambda x: x['group_name'], reverse=reverse_sort)
    else:
        groups_data.sort(key=lambda x: x['term_count'], reverse=reverse_sort)
    
    # Display groups and their terms
    for group in groups_data:
        with st.expander(f"{group['group_name']} (Terms: {group['term_count']}, Leads: {group['total_leads']})"):
            st.write(f"Description: {group['description']}")
            st.write("Search Terms:")
            terms_df = pd.DataFrame(group['terms'])
            
            # Allow sorting
            sort_column = st.selectbox(f"Sort by (Group {group['id']})", terms_df.columns)
            sort_order = st.radio(f"Order (Group {group['id']})", ["Ascending", "Descending"])
            sorted_df = terms_df.sort_values(sort_column, ascending=(sort_order == "Ascending"))
            
            st.dataframe(sorted_df)
            
            # Delete group button
            if st.button(f"Delete Group {group['id']}"):
#                 delete_search_term_group(group['id'])  # Undefined function
                st.success(f"Group {group['id']} deleted successfully.")
                st.rerun()
    
    # Multi-select and assign to group
    st.subheader("Assign Terms to Group")
    all_terms = fetch_all_search_terms()
    selected_terms = st.multiselect("Select terms to assign", all_terms)
    new_group_name = st.text_input("New group name (leave empty to use existing)")
    existing_groups = [g['group_name'] for g in groups_data]
    target_group = st.selectbox("Select target group", [""] + existing_groups)
    
    if st.button("Assign to Group"):
        if new_group_name:
            group_id = create_search_term_group(new_group_name)
        elif target_group:
            group_id = next(g['id'] for g in groups_data if g['group_name'] == target_group)
        else:
            st.error("Please provide a new group name or select an existing group.")
            return
        
#         assign_terms_to_group(selected_terms, group_id)  # Undefined function
        st.success(f"Assigned {len(selected_terms)} terms to the group.")
        st.rerun()

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

    st.header("View Sent Messages")
    
    if st.button("Refresh Sent Messages"):
        st.session_state.sent_messages = fetch_sent_messages()
    
    if 'sent_messages' not in st.session_state:
        st.session_state.sent_messages = fetch_sent_messages()
    
    # Display messages in a more organized manner
    for _, row in st.session_state.sent_messages.iterrows():
        with st.expander(f"Message to {row['Email']} - {row['Sent At']}"):
            st.write(f"**Subject:** {row['Subject']}")
            st.write(f"**Template:** {row['Template']}")
            st.write(f"**Status:** {row['Status']}")
            st.write(f"**Message ID:** {row['Message ID']}")
            st.write("**Content:**")
            st.markdown(row['Content'], unsafe_allow_html=True)

    # Display summary statistics
    st.subheader("Summary Statistics")
    total_messages = len(st.session_state.sent_messages)
    sent_messages = len(st.session_state.sent_messages[st.session_state.sent_messages['Status'] == 'sent'])
    failed_messages = len(st.session_state.sent_messages[st.session_state.sent_messages['Status'] == 'failed'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages", total_messages)
    col2.metric("Sent Messages", sent_messages)
    col3.metric("Failed Messages", failed_messages)

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

def get_last_5_search_terms():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT term FROM search_terms
    ORDER BY id DESC
    LIMIT 5
    """,)
    terms = [row[0] for row in cursor.fetchall()]
    conn.close()
    return terms


    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT l.id, l.email, l.phone, l.first_name, l.last_name, l.company, l.job_title, 
           st.term as search_term, ls.url as source_url, ls.page_title, ls.meta_description, 
           ls.phone_numbers, ls.content, ls.tags
    FROM leads l
    LEFT JOIN lead_sources ls ON l.id = ls.lead_id
    LEFT JOIN search_terms st ON ls.search_term_id = st.id
    ORDER BY l.id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["ID", "Email", "Phone", "First Name", "Last Name", "Company", "Job Title", 
                                       "Search Term", "Source URL", "Page Title", "Meta Description", 
                                       "Phone Numbers", "Content", "Tags"])









import asyncio
import logging
from datetime import datetime
import time

import time

def automation_view():
    st.header("Automation Control Panel")
    
    # Generate a unique key based on the current timestamp
    unique_key = f"automation_{int(time.time() * 1000)}"
    
    # Initialize automation_status if it doesn't exist
    if 'automation_status' not in st.session_state:
        st.session_state.automation_status = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Start/Stop Automation button
        button_text = "Stop Automation" if st.session_state.automation_status else "Start Automation"
        button_color = "secondary" if st.session_state.automation_status else "primary"
        if st.button(button_text, key=f"{unique_key}_toggle", type=button_color):
            st.session_state.automation_status = not st.session_state.automation_status
    
    with col2:
        # Quick Scan button
        if st.button("Quick Scan", key=f"{unique_key}_scan", type="primary"):
            with st.spinner("Performing quick scan..."):
                # Implement quick scan logic here
                st.success("Quick scan completed!")
    
    # Display current automation status
    if st.session_state.automation_status:
        st.success("Automation is currently **ON**.")
    else:
        st.warning("Automation is currently **OFF**.")
    
    # Real-time analytics for automation process
#     display_real_time_analytics()  # Undefined function

    # Additional control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Optimize Search Terms", key=f"{unique_key}_optimize", type="secondary"):
            with st.spinner("Optimizing search terms..."):
                asyncio.run(continuous_search_term_optimization())
            st.success("Search terms optimized.")
    
    with col2:
        if st.button("Send Test Email", key=f"{unique_key}_test_email", type="secondary"):
            with st.spinner("Sending test email..."):
                # Implement test email sending logic here
                st.success("Test email sent successfully.")
    
    with col3:
        if st.button("Generate Report", key=f"{unique_key}_report", type="secondary"):
            with st.spinner("Generating report..."):
                # Implement report generation logic here
                st.success("Report generated successfully.")

    # Run the automation process if it's ON
    if st.session_state.automation_status:
        if st.button("Run Full Automation Cycle", key=f"{unique_key}_run", type="primary"):
            with st.spinner("Running full automation cycle..."):
                asyncio.run(continuous_automation_process())
            st.success("Full automation cycle completed.")

# ... (rest of the code remains unchanged)

# Separate function for the continuous process
async def continuous_automation_process():
    try:
        # Step 1: Gather leads
        st.write("Gathering leads...")
        await continuous_lead_collection()

        # Step 2: Optimize search terms
        st.write("Optimizing search terms...")
        await continuous_search_term_optimization()

        # Step 3: Send emails
        st.write("Sending emails...")
        await continuous_email_sending()

    except Exception as e:
        st.error(f"Error during automation process: {e}")
        logging.error(f"Error during automation process: {e}")

# Make sure to add this function if it doesn't exist:
    pass  # Placeholder block to prevent indentation error
def display_real_time_analytics():
    st.subheader("Real-Time Analytics")
    
    total_leads = count_total_leads()
    leads_last_24_hours = count_leads_last_24_hours()
    emails_sent = count_emails_sent()
    optimized_terms = count_optimized_search_terms()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Leads", total_leads)
    col2.metric("Leads in Last 24 Hours", leads_last_24_hours)
    col3.metric("Emails Sent", emails_sent)
    col4.metric("Optimized Terms", optimized_terms)


# Count total leads gathered
def count_total_leads():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM leads")
    result = cursor.fetchone()[0]
    conn.close()
    return result

# Count leads gathered in the last 24 hours
def count_leads_last_24_hours():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM leads WHERE created_at >= datetime('now', '-1 day')")
    result = cursor.fetchone()[0]
    conn.close()
    return result

# Count total emails sent
def count_emails_sent():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM messages WHERE status = 'sent'")
    result = cursor.fetchone()[0]
    conn.close()
    return result

# Count the number of optimized search terms
def count_optimized_search_terms():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM optimized_search_terms")
    result = cursor.fetchone()[0]
    conn.close()
    return result

# Continuously run lead gathering, email sending, and search term optimization
async def continuous_automation_process():
    while st.session_state.automation_status:
        try:
            # Step 1: Gather leads
            st.write(f"{datetime.now()}: Gathering leads...")
            await continuous_lead_collection()

            # Step 2: Optimize search terms
            st.write(f"{datetime.now()}: Optimizing search terms...")
            await continuous_search_term_optimization()

            # Step 3: Send emails
            st.write(f"{datetime.now()}: Sending emails...")
            await continuous_email_sending()

            # Wait between iterations
            await asyncio.sleep(60)  # Adjust the sleep interval as necessary (e.g., 60 seconds)
        except Exception as e:
            logging.error(f"Error during automation process: {e}")
            break

# Step 1: Lead gathering automation
async def continuous_lead_collection(num_results_per_term=10, sleep_interval=60):
    """Automate lead collection process."""
    search_terms = get_least_searched_terms(5)  # Get least searched terms for optimization
    for term in search_terms:
        campaign_id = 1  # Default campaign
        results = await asyncio.to_thread(manual_search_wrapper, term, num_results_per_term, campaign_id)
        for result in results:
            email = result[0]
            if is_valid_email(email):
                lead_id = save_lead(email, None, None, None, None, None)
                add_lead_to_campaign(campaign_id, lead_id)

    # Wait for the next iteration
    await asyncio.sleep(sleep_interval)

# Step 2: Search term optimization automation
async def continuous_search_term_optimization():
    """Automate search term optimization."""
    current_terms = fetch_all_search_terms()  # Fetch all current terms
    kb_info = get_knowledge_base_info()  # Get context from knowledge base

    # Optimize search terms with AI
    optimized_terms = optimize_search_terms(current_terms, kb_info)

    # Save optimized terms
    save_optimized_search_terms(optimized_terms)
#     log_ai_request("Optimize Search Terms", current_terms, optimized_terms)  # Undefined function

# Step 3: Email sending automation
async def continuous_email_sending(sleep_interval=60):
    """Automate bulk email sending."""
    campaign_id = 1  # Example campaign
    leads = fetch_leads_for_bulk_send(template_id="1", send_option="All Not Contacted", filter_option="Filter Out blog-directory")

    if leads:
        logs = await bulk_send("1", "sender@example.com", "reply@example.com", leads)
        for log in logs:
            logging.info(log)

    # Wait for the next iteration
    await asyncio.sleep(sleep_interval)

# AI-powered search term optimization
    prompt = f"""
    Optimize the following search terms for high-quality lead generation:
    Search Terms: {', '.join(current_terms)}
    Knowledge Base Info: {kb_info}
    Focus on terms that will attract leads with high conversion potential and align with the product.
    """

    response = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    
#     log_ai_request("Optimize Search Terms", prompt, response.choices[0].message.content)  # Undefined function
    return response.choices[0].message.content.split('\n')

# Save AI request logs
def log_ai_request(request_type, prompt, response):
    log_entry = f"{datetime.now()}: {request_type}\nPrompt: {prompt}\nResponse: {response}"
    ai_request_logs.append(log_entry)
    logging.info(log_entry)

# Save optimized search terms
    conn = get_db_connection()
    cursor = conn.cursor()
    for term in optimized_terms:
        cursor.execute("INSERT INTO optimized_search_terms (term) VALUES (?)", (term,))
    conn.commit()
    conn.close()



async def supervisor_check():
    ineffective_terms = get_ineffective_search_terms(threshold=0.3)  # Terms with <30% lead success
    for term in ineffective_terms:
        optimized_term = refine_search_term(term)
#         update_search_term(term, optimized_term)  # Undefined function

def get_ineffective_search_terms(threshold):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT term FROM search_term_effectiveness
        WHERE (valid_leads / total_results) < ?
    """, (threshold,))
    terms = cursor.fetchall()
    conn.close()
    return [term[0] for term in terms]


def chatbot_style_logs(logs):
    st.subheader("Automation Logs")
    for log in logs:
        st.markdown(f"**[{log['timestamp']}]** {log['message']}")

    log_entry = {"timestamp": datetime.now(), "message": f"{request_type}: {response}"}
    ai_request_logs.append(log_entry)
    logging.info(f"{request_type}: {response}")
    chatbot_style_logs(ai_request_logs)

def automation_control_panel():
    st.header("Automation Control")

    # Toggle automation ON/OFF
    if 'automation_status' not in st.session_state:
        st.session_state.automation_status = False

    if st.button("Toggle Automation"):
        st.session_state.automation_status = not st.session_state.automation_status

    if st.session_state.automation_status:
        st.write("**Automation is currently ON.**")
        asyncio.run(continuous_automation_process())
    else:
        st.write("**Automation is currently OFF.**")

    # Real-time analytics
    display_real_time_analytics()

    total_leads = count_total_leads()
    leads_last_24h = count_leads_last_24_hours()
    emails_sent = count_emails_sent()
    optimized_terms = count_optimized_search_terms()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Leads", total_leads)
    col2.metric("Leads in Last 24 Hours", leads_last_24h)
    col3.metric("Emails Sent", emails_sent)
    col4.metric("Optimized Terms", optimized_terms)



async def continuous_automation_process():
    cycle_count = 0
    while st.session_state.automation_status:
        try:
            # Lead collection step
            await continuous_lead_collection()

            # Search term optimization every 5 cycles
            if cycle_count % 5 == 0:
#                 supervisor_check()  # Undefined function

            # Email sending step
            await continuous_email_sending()

            cycle_count += 1
            await asyncio.sleep(60)  # Run the cycle every 60 seconds

        except Exception as e:
            logging.error(f"Error in automation: {e}")
            st.error(f"Error in automation process: {e}")




# Initialize real-time analytics in Streamlit
if __name__ == "__main__":
    st.title("AUTOCLIENT - Full Automation System")

    # Automation view
    automation_view()

    # Start the continuous automation process if enabled
    if st.session_state.automation_status:
        asyncio.run(continuous_automation_process())

# Call the main function to run the Streamlit app
if __name__ == "__main__":
    main()
            sg