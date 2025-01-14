import streamlit as st
from streamlit_tags import st_tags
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from attached_assets.streamlit_app import manual_search

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def perform_search(terms, num_results=10):
    results = manual_search(
        session=SessionLocal(),
        terms=terms,
        num_results=num_results,
        ignore_previously_fetched=True,
        optimize_english=False,
        optimize_spanish=False,
        shuffle_keywords_option=False,
        language='ES',
        enable_email_sending=False,
        log_container=st
    )
    return results

st.title("🔍 Manual Search")

# Search interface
search_terms = st_tags(
    label='Enter search terms:',
    text='Press enter to add more',
    value=['test organization'],
    maxtags=10,
    key='search_terms'
)

num_results = st.slider("Number of results per term", 5, 50, 10)

if st.button("Search"):
    if search_terms:
        with st.spinner("Searching..."):
            results = perform_search(search_terms, num_results)
            if results and 'results' in results:
                df = pd.DataFrame(results['results'])
                if not df.empty:
                    st.dataframe(df)
                else:
                    st.info("No results found")

                st.success(f"Found {results.get('total_leads', 0)} leads")
            else:
                st.warning("No results returned from search")
    else:
        st.warning("Please enter at least one search term")