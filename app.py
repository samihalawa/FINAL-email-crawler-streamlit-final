import streamlit as st
from dotenv import load_dotenv
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

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

@st.cache_resource
def get_database():
    load_dotenv()
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise Exception("DATABASE_URL not found in environment variables")
    
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return engine

def get_session():
    engine = get_database()
    Session = sessionmaker(bind=engine)
    return Session()

def main():
    try:
        session = get_session()
        st.session_state['db'] = session
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return

    st.title("Welcome to AutoclientAI")
    st.write("Use the sidebar to navigate between different features.")

if __name__ == "__main__":
    main()