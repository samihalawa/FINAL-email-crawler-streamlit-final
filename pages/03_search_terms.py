import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, func, distinct, desc
from sqlalchemy.orm import sessionmaker
from models import SearchTerm, Lead, LeadSource, EmailCampaign
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Search Terms",
    page_icon="🔑",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_term_stats(session):
    """Get statistics for search terms"""
    return (session.query(
        SearchTerm.term,
        SearchTerm.language,
        func.count(distinct(Lead.id)).label('lead_count'),
        func.count(distinct(EmailCampaign.id)).label('email_count'),
        func.min(Lead.created_at).label('first_lead'),
        func.max(Lead.created_at).label('last_lead'),
        func.count(distinct(LeadSource.domain)).label('domains_found')
    ).join(LeadSource, SearchTerm.id == LeadSource.search_term_id, isouter=True)
     .join(Lead, LeadSource.lead_id == Lead.id, isouter=True)
     .outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id)
     .group_by(SearchTerm.term, SearchTerm.language)
     .order_by(desc('lead_count')))

def add_search_term(session, term, language):
    """Add a new search term"""
    try:
        existing = session.query(SearchTerm).filter_by(term=term).first()
        if existing:
            return False, "Term already exists"
            
        new_term = SearchTerm(
            term=term,
            language=language,
            created_at=datetime.utcnow()
        )
        session.add(new_term)
        session.commit()
        return True, f"Added new search term: {term}"
    except Exception as e:
        session.rollback()
        return False, str(e)

def delete_search_term(session, term):
    """Delete a search term"""
    try:
        term_obj = session.query(SearchTerm).filter_by(term=term).first()
        if not term_obj:
            return False, "Term not found"
            
        # Get associated leads count
        leads_count = session.query(func.count(distinct(Lead.id))).join(
            LeadSource, Lead.id == LeadSource.lead_id
        ).filter(LeadSource.search_term_id == term_obj.id).scalar()
        
        if leads_count > 0:
            return False, f"Cannot delete term with {leads_count} associated leads"
            
        session.delete(term_obj)
        session.commit()
        return True, f"Deleted search term: {term}"
    except Exception as e:
        session.rollback()
        return False, str(e)

def main():
    st.title("🔑 Search Terms")
    
    try:
        with SessionLocal() as session:
            # Add new search term (in sidebar)
            st.sidebar.header("➕ Add New Term")
            with st.sidebar.form("new_search_term"):
                new_term = st.text_input("Enter search term:")
                language = st.selectbox("Language", ['ES', 'EN'], index=0)
                submitted = st.form_submit_button("Add Term")
                
                if submitted and new_term:
                    success, message = add_search_term(session, new_term, language)
                    if success:
                        st.sidebar.success(message)
                        st.rerun()
                    else:
                        st.sidebar.error(message)
            
            # Display term statistics
            terms_data = get_term_stats(session).all()
            if terms_data:
                df = pd.DataFrame(terms_data, columns=[
                    'Term', 'Language', 'Leads', 'Emails Sent', 
                    'First Lead', 'Last Lead', 'Unique Domains'
                ])
                
                # Summary metrics
                st.subheader("📊 Overview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Terms", len(df))
                with col2:
                    st.metric("Total Leads", df['Leads'].sum())
                with col3:
                    st.metric("Active Terms", len(df[df['Last Lead'] >= datetime.utcnow() - timedelta(days=7)]))
                with col4:
                    st.metric("Success Rate", f"{(df['Emails Sent'] / df['Leads'].sum() * 100):.1f}%" if df['Leads'].sum() > 0 else "0.0%")
                
                # Terms table
                st.subheader("📝 Search Terms")
                st.dataframe(
                    df,
                    column_config={
                        "Term": st.column_config.TextColumn("Term", width="large"),
                        "Language": st.column_config.TextColumn("Lang", width="small"),
                        "Leads": st.column_config.NumberColumn("Leads", format="%d"),
                        "Emails Sent": st.column_config.NumberColumn("Emails", format="%d"),
                        "First Lead": st.column_config.DatetimeColumn("First Lead", format="D MMM YYYY"),
                        "Last Lead": st.column_config.DatetimeColumn("Last Lead", format="D MMM YYYY"),
                        "Unique Domains": st.column_config.NumberColumn("Domains", format="%d")
                    },
                    hide_index=True
                )
                
                # Delete terms
                if st.checkbox("🗑️ Show Delete Options"):
                    st.warning("Only terms with no leads can be deleted")
                    term_to_delete = st.selectbox(
                        "Select term to delete",
                        options=df['Term'].tolist()
                    )
                    if st.button("Delete Term"):
                        success, message = delete_search_term(session, term_to_delete)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            else:
                st.info("No search terms found. Add your first search term using the sidebar.")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Please check your database connection and try again")

if __name__ == "__main__":
    main()