import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("📨 Sent Campaigns")

try:
    with SessionLocal() as session:
        from streamlit_app import Campaign, EmailCampaign, Lead, EmailTemplate
        
        # Get all campaigns with email stats
        campaigns = session.query(
            Campaign,
            func.count(EmailCampaign.id).label('total_emails'),
            func.sum(func.case([(EmailCampaign.status == 'Sent', 1)], else_=0)).label('sent'),
            func.sum(func.case([(EmailCampaign.status == 'Failed', 1)], else_=0)).label('failed'),
            func.sum(EmailCampaign.open_count).label('opens'),
            func.sum(EmailCampaign.click_count).label('clicks')
        ).outerjoin(
            EmailCampaign,
            Campaign.id == EmailCampaign.campaign_id
        ).group_by(Campaign.id).all()
        
        if campaigns:
            # Prepare campaign data
            campaign_data = []
            for campaign, total, sent, failed, opens, clicks in campaigns:
                campaign_data.append({
                    'ID': campaign.id,
                    'Name': campaign.campaign_name,
                    'Type': campaign.campaign_type,
                    'Total Emails': total,
                    'Sent': sent or 0,
                    'Failed': failed or 0,
                    'Opens': opens or 0,
                    'Clicks': clicks or 0,
                    'Success Rate': f"{(sent/total*100 if total else 0):.1f}%" if total else "0.0%",
                    'Created': campaign.created_at
                })
            
            # Display campaigns
            st.dataframe(
                pd.DataFrame(campaign_data),
                column_config={
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "Name": st.column_config.TextColumn("Name", width="medium"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Total Emails": st.column_config.NumberColumn("Total", width="small"),
                    "Sent": st.column_config.NumberColumn("Sent", width="small"),
                    "Failed": st.column_config.NumberColumn("Failed", width="small"),
                    "Opens": st.column_config.NumberColumn("Opens", width="small"),
                    "Clicks": st.column_config.NumberColumn("Clicks", width="small"),
                    "Success Rate": st.column_config.TextColumn("Success", width="small"),
                    "Created": st.column_config.DatetimeColumn("Created")
                }
            )
            
            # Campaign Details
            st.subheader("Campaign Details")
            selected_campaign = st.selectbox(
                "Select Campaign",
                options=[(c.id, c.campaign_name) for c, _, _, _, _, _ in campaigns],
                format_func=lambda x: x[1]
            )
            
            if selected_campaign:
                campaign_id = selected_campaign[0]
                
                # Get email details for selected campaign
                emails = session.query(EmailCampaign).filter_by(campaign_id=campaign_id).all()
                if emails:
                    email_data = []
                    for email in emails:
                        lead = session.query(Lead).get(email.lead_id)
                        template = session.query(EmailTemplate).get(email.template_id)
                        email_data.append({
                            'Email': lead.email if lead else 'Unknown',
                            'Template': template.template_name if template else 'Unknown',
                            'Status': email.status,
                            'Sent At': email.sent_at,
                            'Opens': email.open_count,
                            'Clicks': email.click_count
                        })
                    
                    st.dataframe(pd.DataFrame(email_data))
                else:
                    st.info("No emails found for this campaign")
        else:
            st.info("No campaigns found")

except Exception as e:
    st.error(f"Error loading sent campaigns: {str(e)}")
