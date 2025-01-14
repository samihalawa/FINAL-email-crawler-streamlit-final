import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, func, distinct
from sqlalchemy.orm import sessionmaker
from models import SearchTerm, Lead, LeadSource, EmailCampaign
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def main():
    st.title("🔑 Search Terms")

    try:
        with SessionLocal() as session:
            # Display existing search terms
            st.subheader("Existing Search Terms")

            # Fetch and display search terms with lead counts
            query = (session.query(SearchTerm.term, 
                               func.count(distinct(Lead.id)).label('lead_count'),
                               func.count(distinct(EmailCampaign.id)).label('email_count'))
                  .join(LeadSource, SearchTerm.id == LeadSource.search_term_id)
                  .join(Lead, LeadSource.lead_id == Lead.id)
                  .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
                  .group_by(SearchTerm.term))

            terms_df = pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

            if not terms_df.empty:
                st.dataframe(terms_df)
            else:
                st.info("No search terms found")

            # Add new search terms
            st.subheader("Add New Search Term")
            with st.form("new_search_term"):
                new_term = st.text_input("Enter new search term:")
                language = st.selectbox("Language", ['ES', 'EN'], index=0)
                campaign_id = st.session_state.get('current_campaign_id', 1)
                submit = st.form_submit_button("Add Term")

                if submit and new_term:
                    try:
                        # Check if term already exists
                        existing_term = session.query(SearchTerm).filter_by(term=new_term, campaign_id=campaign_id).first()
                        if existing_term:
                            st.warning(f"Search term '{new_term}' already exists")
                        else:
                            new_search_term = SearchTerm(
                                term=new_term,
                                campaign_id=campaign_id,
                                language=language
                            )
                            session.add(new_search_term)
                            session.commit()
                            st.success(f"Added new search term: {new_term}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error adding search term: {str(e)}")
                        session.rollback()

            # Delete search terms
            st.subheader("Delete Search Terms")
            terms = session.query(SearchTerm).all()
            if terms:
                term_to_delete = st.selectbox(
                    "Select term to delete",
                    options=[term.term for term in terms]
                )
                if st.button("Delete Term"):
                    try:
                        session.query(SearchTerm).filter_by(term=term_to_delete).delete()
                        session.commit()
                        st.success(f"Deleted search term: {term_to_delete}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting search term: {str(e)}")
                        session.rollback()
            else:
                st.info("No terms available to delete")

    except Exception as e:
        st.error(f"Error loading search terms: {str(e)}")

if __name__ == "__main__":
    main()
