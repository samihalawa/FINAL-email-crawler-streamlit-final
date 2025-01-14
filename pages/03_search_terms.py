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

st.title("🔑 Search Terms")

try:
    with SessionLocal() as session:
        from streamlit_app import fetch_search_terms_with_lead_count, SearchTerm
        
        # Display existing search terms
        st.subheader("Existing Search Terms")
        terms_df = fetch_search_terms_with_lead_count(session)
        if not terms_df.empty:
            st.dataframe(terms_df)
        else:
            st.info("No search terms found")
        
        # Add new search terms
        st.subheader("Add New Search Term")
        with st.form("new_search_term"):
            new_term = st.text_input("Enter new search term:")
            campaign_id = 1  # Default campaign ID
            submit = st.form_submit_button("Add Term")
            
            if submit and new_term:
                try:
                    new_search_term = SearchTerm(
                        term=new_term,
                        campaign_id=campaign_id,
                        language='ES'
                    )
                    session.add(new_search_term)
                    session.commit()
                    st.success(f"Added new search term: {new_term}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding search term: {str(e)}")
                    session.rollback()

except Exception as e:
    st.error(f"Error loading search terms: {str(e)}")
