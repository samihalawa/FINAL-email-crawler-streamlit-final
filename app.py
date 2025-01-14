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

from streamlit_option_menu import option_menu
from models import *

def initialize_settings():
    # Placeholder:  Replace with actual initialization logic from streamlit_app.py or elsewhere
    try:
        #Example: Load environment variables etc.
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

def initialize_session_state():
    # Placeholder: Replace with actual session state initialization logic
    st.session_state.update({"key1": "value1"})


def initialize_pages():
    return {
        "Home": lambda: st.title("Welcome to AutoclientAI")
    }

def main():
    # Initialize settings and check database state
    if not initialize_settings():
        st.error("Failed to initialize application. Please check the logs and configuration.")
        return

    # Initialize session state with defaults
    initialize_session_state()

    # Initialize pages
    pages = initialize_pages()

    # Create navigation menu
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=list(pages.keys()),
            icons=["search", "box-seam", "people", "key", "envelope", "book", "robot", 
                  "gear", "tools", "journal-text", "sliders", "envelope-paper", "rocket"],
            menu_icon="house",
            default_index=0
        )

    try:
        pages[selected]()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("An error occurred in the main function")

if __name__ == "__main__":
    main()