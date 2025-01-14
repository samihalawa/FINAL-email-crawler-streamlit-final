
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from models import *
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("📋 View Leads")

# Fetch and display leads
try:
    with SessionLocal() as session:
        query = session.query(
            Lead,
            EmailCampaign.status.label('Last Email Status'),
            EmailCampaign.sent_at.label('Last Contact')
        ).outerjoin(EmailCampaign)
        
        leads_data = []
        for lead, status, last_contact in query.all():
            leads_data.append({
                'email': lead.email,
                'company': lead.company,
                'first_name': lead.first_name,
                'last_name': lead.last_name,
                'job_title': lead.job_title,
                'Last Email Status': status,
                'Last Contact': last_contact,
                'created_at': lead.created_at
            })
        
        leads_df = pd.DataFrame(leads_data)
        
        if not leads_df.empty:
            # Add filters
            st.subheader("Filters")
            col1, col2 = st.columns(2)
            
            with col1:
                companies = sorted(leads_df['company'].dropna().unique())
                selected_companies = st.multiselect("Filter by company", companies)
            
            with col2:
                statuses = sorted(leads_df['Last Email Status'].dropna().unique())
                selected_statuses = st.multiselect("Filter by status", statuses)
            
            # Apply filters
            if selected_companies:
                leads_df = leads_df[leads_df['company'].isin(selected_companies)]
            if selected_statuses:
                leads_df = leads_df[leads_df['Last Email Status'].isin(selected_statuses)]
            
            # Display leads
            st.subheader("Leads")
            st.dataframe(
                leads_df,
                column_config={
                    "email": st.column_config.TextColumn("Email", width="medium"),
                    "company": st.column_config.TextColumn("Company", width="medium"),
                    "Last Contact": st.column_config.DatetimeColumn("Last Contact"),
                    "created_at": st.column_config.DatetimeColumn("Created At")
                }
            )
        else:
            st.info("No leads found in the database")
except Exception as e:
    st.error(f"Error loading leads: {str(e)}")
