import streamlit as st
from streamlit_tags import st_tags
import pandas as pd
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
import logging
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from urllib.parse import urlparse
import re
from datetime import datetime, timezone
from googlesearch import search
from models import (
    Base, Lead, LeadSource, SearchTerm, EmailSettings, 
    EmailTemplate, Project, Campaign, KnowledgeBase
)
from utils.state import init_session_state

def get_domain_from_url(url):
    """Extract domain from URL."""
    try:
        return urlparse(url).netloc
    except:
        return ""

def is_valid_email(email):
    """Validate email format."""
    if not email:
        return False
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def extract_emails_from_html(html_content):
    """Extract email addresses from HTML content."""
    if not html_content:
        return []
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return list(set(re.findall(pattern, html_content)))

def extract_info_from_page(soup):
    """Extract information from page."""
    if not soup:
        return "", ""
        
    name = soup.find('meta', {'name': 'author'})
    name = name['content'] if name and 'content' in name.attrs else ''

    company = soup.find('meta', {'property': 'og:site_name'})
    company = company['content'] if company and 'content' in company.attrs else ''
    
    return name, company

@st.cache_data(ttl=3600)
def safe_google_search(term, num_results, lang):
    try:
        return list(search(term, num_results=num_results, lang=lang))
    except Exception as e:
        logging.error(f"Google search error: {str(e)}")
        return []

def main():
    try:
        init_session_state()
        
        st.title("🔍 Manual Search")
        
        if 'db' not in st.session_state:
            st.error("Database connection not initialized")
            return
            
        session = st.session_state.db
        
        # Set default project and campaign IDs
        st.session_state.current_project_id = st.session_state.get('current_project_id', 1)
        st.session_state.current_campaign_id = st.session_state.get('current_campaign_id', 1)
        
        # Get project and campaign info
        project = session.query(Project).get(st.session_state.current_project_id)
        campaign = session.query(Campaign).get(st.session_state.current_campaign_id)
        kb = session.query(KnowledgeBase).filter_by(project_id=st.session_state.current_project_id).first()
        
        if project and campaign:
            st.info(f"🎯 Current Project: {project.project_name} | Campaign: {campaign.campaign_name}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_terms = st_tags(
                label='Enter search terms:',
                text='Press enter to add more',
                value=['software developer'],
                maxtags=10,
                key='search_terms'
            )

        with col2:
            num_results = st.slider("Results per term", 5, 50, 10)
            language = st.selectbox("Language", ['ES', 'EN'], index=0)
            ignore_previous = st.checkbox("Ignore previously fetched", value=True)

        if st.button("🚀 Start Search"):
            if not search_terms:
                st.warning("Please enter at least one search term")
                return

            log_container = st.empty()
            progress_bar = st.progress(0)

            try:
                # Get email settings with proper error handling
                email_settings = session.query(EmailSettings).filter_by(is_active=True).first()
                email_template = session.query(EmailTemplate).first()
                
                results = manual_search(
                    session=session,
                    terms=search_terms,
                    num_results=num_results,
                    ignore_previously_fetched=ignore_previous,
                    language=language,
                    log_container=log_container,
                    progress_bar=progress_bar,
                    from_email=email_settings.email if email_settings else None,
                    email_template=f"{email_template.id}:{email_template.template_name}" if email_template else None,
                    reply_to=email_settings.email if email_settings else None
                )

                if results and results.get('results'):
                    df = pd.DataFrame(results['results'])
                    st.dataframe(
                        df,
                        column_config={
                            "Email": st.column_config.TextColumn("Email", width="medium"),
                            "URL": st.column_config.LinkColumn("URL"),
                            "Lead Source": st.column_config.TextColumn("Source", width="small"),
                            "Title": st.column_config.TextColumn("Page Title", width="large"),
                        },
                        hide_index=True
                    )
                    
                    st.success(f"✨ Found {results['total_leads']} new leads!")
                    
                    if st.button("📥 Download Results"):
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "search_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                else:
                    st.info("No results found. Try different search terms or settings.")
                    
            except Exception as e:
                logging.error(f"Search error: {str(e)}")
                st.error(f"An error occurred during search: {str(e)}")
                
    except Exception as e:
        logging.error(f"Page error: {str(e)}")
        st.error("An error occurred while loading the page. Please try refreshing.")

if __name__ == "__main__":
    main()
