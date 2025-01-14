import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from attached_assets.streamlit_app import EmailSettings, Settings

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

st.title("⚙️ Settings")

try:
    with SessionLocal() as session:
        # Email Settings
        st.subheader("Email Settings")

        email_settings = session.query(EmailSettings).filter_by(is_active=True).first()

        with st.form("email_settings"):
            email = st.text_input("Email Address", value=email_settings.email if email_settings else "")
            provider = st.selectbox(
                "Email Provider",
                options=["ses", "smtp"],
                index=0 if email_settings and email_settings.provider == "ses" else 1
            )

            if provider == "smtp":
                smtp_server = st.text_input("SMTP Server", value=email_settings.smtp_server if email_settings else "")
                smtp_port = st.number_input("SMTP Port", value=email_settings.smtp_port if email_settings else 587)
                smtp_username = st.text_input("SMTP Username")
                smtp_password = st.text_input("SMTP Password", type="password")
            else:
                aws_access_key = st.text_input("AWS Access Key ID")
                aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
                aws_region = st.text_input("AWS Region", value="us-east-1")

            if st.form_submit_button("Save Email Settings"):
                try:
                    if not email_settings:
                        email_settings = EmailSettings()

                    email_settings.email = email
                    email_settings.provider = provider
                    email_settings.is_active = True

                    if provider == "smtp":
                        email_settings.smtp_server = smtp_server
                        email_settings.smtp_port = smtp_port
                        email_settings.smtp_username = smtp_username
                        if smtp_password:
                            email_settings.smtp_password = smtp_password
                    else:
                        email_settings.aws_access_key_id = aws_access_key
                        if aws_secret_key:
                            email_settings.aws_secret_access_key = aws_secret_key
                        email_settings.aws_region = aws_region

                    session.add(email_settings)
                    session.commit()
                    st.success("Email settings saved successfully!")
                except Exception as e:
                    st.error(f"Error saving email settings: {str(e)}")
                    session.rollback()

        # Search Settings
        st.subheader("Search Settings")

        search_settings = session.query(Settings).filter_by(
            setting_type='search',
            name='default_search_settings'
        ).first()

        with st.form("search_settings"):
            num_results = st.slider(
                "Default Number of Results",
                min_value=5,
                max_value=100,
                value=50
            )

            ignore_previous = st.checkbox(
                "Ignore Previously Fetched Results",
                value=True
            )

            if st.form_submit_button("Save Search Settings"):
                try:
                    if not search_settings:
                        search_settings = Settings(
                            name='default_search_settings',
                            setting_type='search'
                        )

                    search_settings.value = {
                        'num_results': num_results,
                        'ignore_previously_fetched': ignore_previous
                    }

                    session.add(search_settings)
                    session.commit()
                    st.success("Search settings saved successfully!")
                except Exception as e:
                    st.error(f"Error saving search settings: {str(e)}")
                    session.rollback()

except Exception as e:
    st.error(f"Error loading settings: {str(e)}")