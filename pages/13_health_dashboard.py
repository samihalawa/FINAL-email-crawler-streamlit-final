import streamlit as st
from utils.db import SessionLocal, engine
from models import Lead, EmailCampaign, SearchTerm, Base
from sqlalchemy import func, inspect
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import os
from utils.migrations import safe_migrate

def get_schema_info():
    """Get database schema information from Supabase"""
    try:
        inspector = inspect(engine)
        schema_info = {}
        
        # Get existing tables and their structure (read-only operation)
        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable'],
                    'default': str(column['default']) if column.get('default') else 'NULL'
                })
            schema_info[table_name] = columns
        
        # Format as SQL-like output
        markdown = "```sql\n"
        markdown += f"-- Supabase DB Schema ({os.getenv('SUPABASE_DB_NAME')})\n"
        markdown += f"-- Host: {os.getenv('SUPABASE_DB_HOST')}\n\n"
        
        for table, columns in schema_info.items():
            markdown += f"\n-- Table: {table}\n"
            for col in columns:
                default = f"DEFAULT {col['default']}" if col['default'] != 'NULL' else ''
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                markdown += f"{col['name']} {col['type']} {nullable} {default}\n"
        
        markdown += "```"
        return markdown, None
        
    except Exception as e:
        return None, f"Error retrieving schema: {str(e)}"

def display_health_metrics():
    with SessionLocal() as session:
        # Get metrics
        total_leads = session.query(func.count(Lead.id)).scalar()
        total_emails = session.query(func.count(EmailCampaign.id)).scalar()
        
        # Calculate success rate safely
        if total_emails > 0:
            success_rate = session.query(
                func.count(EmailCampaign.id).filter(EmailCampaign.status == 'sent') * 100.0 / 
                func.nullif(func.count(EmailCampaign.id), 0)  # Use nullif to prevent division by zero
            ).scalar() or 0
        else:
            success_rate = 0
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Leads", total_leads)
        col2.metric("Total Emails", total_emails)
        col3.metric("Success Rate", f"{success_rate:.1f}%")

def display_lead_chart():
    with SessionLocal() as session:
        # Get lead data for last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        leads = session.query(
            func.date(Lead.created_at).label('date'),
            func.count(Lead.id).label('count')
        ).filter(Lead.created_at >= thirty_days_ago).group_by(func.date(Lead.created_at)).all()
        
        if leads:
            df = pd.DataFrame(leads, columns=['date', 'count'])
            fig = px.line(df, x='date', y='count', title='Leads Over Time')
            st.plotly_chart(fig)

def main():
    st.title("🏥 System Health Dashboard")
    
    display_health_metrics()
    display_lead_chart()
    
    # Add schema viewer
    st.subheader("�� Database Schema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Show Current Schema"):
            schema, error = get_schema_info()
            if error:
                st.error(error)
            else:
                st.code(schema, language="sql")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"schema_snapshot_{timestamp}.sql"
                st.download_button(
                    "Download Schema",
                    schema,
                    file_name=filename,
                    mime="text/plain"
                )
    
    with col2:
        if st.button("Run Safe Migrations"):
            success, message = safe_migrate()
            if success:
                st.success(message)
                st.rerun()  # Refresh to show new schema
            else:
                st.error(f"Migration failed: {message}")
    
    # System checks
    st.subheader("System Checks")
    with SessionLocal() as session:
        try:
            session.query(Lead).first()
            st.success("✅ Database Connection")
        except Exception as e:
            st.error(f"❌ Database Connection Failed: {str(e)}")

if __name__ == "__main__":
    main()
