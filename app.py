import streamlit as st
from dotenv import load_dotenv
from models import *
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Must be first Streamlit command
st.set_page_config(
    page_title="AutoclientAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def initialize_settings():
    try:
        load_dotenv()
        DATABASE_URL = os.getenv('DATABASE_URL')
        if not DATABASE_URL:
            return False

        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with engine.connect() as conn:
            return True

    except Exception as e:
        logging.exception(f"Error in initialize_settings: {str(e)}")
        return False

def main():
    # Initialize settings and check database state
    if not initialize_settings():
        st.error("Failed to initialize application. Please check the logs and configuration.")
        return

    st.title("Welcome to AutoclientAI")

if __name__ == "__main__":
    main()