import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("📨 Email Logs")

try:
    with SessionLocal() as session:
        from streamlit_app import EmailCampaign, Lead, EmailTemplate
        
        # Time range filter
        st.subheader("Filter Logs")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=7)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now()
            )
        
        # Status filter
        status_filter = st.multiselect(
            "Filter by Status",
            ["Sent", "Failed", "Opened", "Clicked"],
            default=["Sent", "Failed", "Opened", "Clicked"]
        )
        
        # Query email logs
        query = session.query(
            EmailCampaign,
            Lead.email,
            EmailTemplate.template_name
        ).join(
            Lead,
            EmailCampaign.lead_id == Lead.id
        ).join(
            EmailTemplate,
            EmailCampaign.template_id == EmailTemplate.id
        ).filter(
            func.date(EmailCampaign.sent_at) >= start_date,
            func.date(EmailCampaign.sent_at) <= end_date,
            EmailCampaign.status.in_(status_filter)
        ).order_by(EmailCampaign.sent_at.desc())
        
        results = query.all()
        
        if results:
            # Prepare data for display
            log_data = []
            for campaign, email, template_name in results:
                log_data.append({
                    'ID': campaign.id,
                    'Email': email,
                    'Template': template_name,
                    'Status': campaign.status,
                    'Sent At': campaign.sent_at,
                    'Opened At': campaign.opened_at,
                    'Clicked At': campaign.clicked_at,
                    'Opens': campaign.open_count,
                    'Clicks': campaign.click_count
                })
            
            # Display logs
            st.subheader("Email Logs")
            df = pd.DataFrame(log_data)
            st.dataframe(
                df,
                column_config={
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "Email": st.column_config.TextColumn("Email", width="medium"),
                    "Template": st.column_config.TextColumn("Template", width="medium"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Sent At": st.column_config.DatetimeColumn("Sent At"),
                    "Opened At": st.column_config.DatetimeColumn("Opened At"),
                    "Clicked At": st.column_config.DatetimeColumn("Clicked At"),
                    "Opens": st.column_config.NumberColumn("Opens", width="small"),
                    "Clicks": st.column_config.NumberColumn("Clicks", width="small")
                }
            )
            
            # Download option
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode('utf-8'),
                "email_logs.csv",
                "text/csv"
            )
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Emails", len(results))
            with col2:
                st.metric("Total Opens", sum(campaign.open_count or 0 for campaign, _, _ in results))
            with col3:
                st.metric("Total Clicks", sum(campaign.click_count or 0 for campaign, _, _ in results))
            with col4:
                successful = sum(1 for campaign, _, _ in results if campaign.status == "Sent")
                st.metric("Success Rate", f"{(successful/len(results)*100):.1f}%")
        else:
            st.info("No email logs found for the selected criteria")

except Exception as e:
    st.error(f"Error loading email logs: {str(e)}")
