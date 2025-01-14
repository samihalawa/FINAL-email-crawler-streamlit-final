import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("👥 View Leads")

# Fetch and display leads
try:
    with SessionLocal() as session:
        from streamlit_app import fetch_leads_with_sources
        
        leads_df = fetch_leads_with_sources(session)
        
        if not leads_df.empty:
            # Add filters
            st.subheader("Filters")
            col1, col2 = st.columns(2)
            
            with col1:
                companies = sorted(leads_df['company'].unique())
                selected_companies = st.multiselect("Filter by company", companies)
            
            with col2:
                statuses = sorted(leads_df['Last Email Status'].unique())
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
            
            st.download_button(
                "Download CSV",
                leads_df.to_csv(index=False).encode('utf-8'),
                "leads.csv",
                "text/csv"
            )
        else:
            st.info("No leads found in the database")
except Exception as e:
    st.error(f"Error loading leads: {str(e)}")
