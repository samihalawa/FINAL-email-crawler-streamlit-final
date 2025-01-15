import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker
from models import Lead, EmailCampaign, LeadSource, SearchTerm
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="View Leads",
    page_icon="📋",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def fetch_leads(session, date_filter, selected_companies, selected_statuses, selected_terms):
    query = session.query(
        Lead,
        EmailCampaign,
        SearchTerm.term
    ).outerjoin(
        EmailCampaign,
        Lead.id == EmailCampaign.lead_id
    ).outerjoin(
        SearchTerm,
        Lead.search_term_id == SearchTerm.id
    )
    
    if date_filter:
        query = query.filter(Lead.created_at >= date_filter)
    
    if selected_companies:
        query = query.filter(Lead.company.in_(selected_companies))
    
    if selected_statuses:
        query = query.filter(EmailCampaign.status.in_(selected_statuses))
    
    if selected_terms:
        query = query.filter(SearchTerm.term.in_(selected_terms))
    
    results = query.all()
    
    leads_data = []
    for lead, email_campaign, term in results:
        leads_data.append({
            'ID': lead.id,
            'Email': lead.email,
            'Company': lead.company,
            'Title': lead.title,
            'URL': lead.url,
            'Status': email_campaign.status if email_campaign else 'N/A',
            'Search Term': term if term else 'N/A',
            'Created At': lead.created_at,
            'Sent At': email_campaign.sent_at if email_campaign else None,
            'Opened At': email_campaign.opened_at if email_campaign else None,
            'Clicked At': email_campaign.clicked_at if email_campaign else None,
            'Opens': email_campaign.open_count if email_campaign else 0,
            'Clicks': email_campaign.click_count if email_campaign else 0
        })
    
    return leads_data

def main():
    st.title("📋 View Leads")
    
    try:
        with SessionLocal() as session:
            # Set default project and campaign
            st.session_state.current_project_id = 1
            st.session_state.current_campaign_id = 1
            
            # Sidebar filters
            st.sidebar.header("📊 Filters")
            
            # Date filter
            date_options = {
                "All time": None,
                "Last 24 hours": datetime.utcnow() - timedelta(days=1),
                "Last 7 days": datetime.utcnow() - timedelta(days=7),
                "Last 30 days": datetime.utcnow() - timedelta(days=30)
            }
            date_filter = st.sidebar.selectbox(
                "Time period",
                options=list(date_options.keys()),
                index=0
            )
            
            # Get unique values for filters
            companies = sorted(session.query(Lead.company).filter(Lead.company.isnot(None)).distinct())
            statuses = sorted(session.query(EmailCampaign.status).filter(EmailCampaign.status.isnot(None)).distinct())
            search_terms = sorted(session.query(SearchTerm.term).distinct())
            
            # Apply filters
            selected_companies = st.sidebar.multiselect("Filter by company", [c[0] for c in companies if c[0]])
            selected_statuses = st.sidebar.multiselect("Filter by status", [s[0] for s in statuses if s[0]])
            selected_terms = st.sidebar.multiselect("Filter by search term", [t[0] for t in search_terms if t[0]])
            
            # Fetch filtered leads
            results = fetch_leads(
                session,
                date_options[date_filter],
                selected_companies,
                selected_statuses,
                selected_terms
            )
            
            if results:
                df = pd.DataFrame(results)
                st.dataframe(
                    df,
                    column_config={
                        "ID": st.column_config.NumberColumn("ID", width="small"),
                        "Email": st.column_config.TextColumn("Email", width="medium"),
                        "Company": st.column_config.TextColumn("Company", width="medium"),
                        "Title": st.column_config.TextColumn("Title", width="medium"),
                        "URL": st.column_config.LinkColumn("URL"),
                        "Status": st.column_config.TextColumn("Status", width="small"),
                        "Search Term": st.column_config.TextColumn("Search Term", width="medium"),
                        "Created At": st.column_config.DatetimeColumn("Created At"),
                        "Sent At": st.column_config.DatetimeColumn("Sent At"),
                        "Opened At": st.column_config.DatetimeColumn("Opened At"),
                        "Clicked At": st.column_config.DatetimeColumn("Clicked At"),
                        "Opens": st.column_config.NumberColumn("Opens", width="small"),
                        "Clicks": st.column_config.NumberColumn("Clicks", width="small")
                    }
                )
            else:
                st.info("No leads found with the selected filters.")
    except Exception as e:
        st.error(f"Error loading leads: {str(e)}")
        st.error("Please check your database connection and try again")

if __name__ == "__main__":
    main()
