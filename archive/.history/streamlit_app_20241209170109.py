# Standard library imports
import os
import json
import re
import logging
import logging.config
import asyncio
import time
import random
import html
import smtplib
import uuid
import threading
import atexit
from datetime import datetime, timedelta
from contextlib import contextmanager, wraps
from threading import local, Lock
from urllib.parse import urlparse, urlencode
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict, Any, Union

# Third-party imports
import requests
import pandas as pd
import streamlit as st
import openai
import boto3
import aiohttp
import urllib3
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import (
    func, create_engine, Column, BigInteger, Text, DateTime, 
    ForeignKey, Boolean, JSON, select, text, distinct, and_, 
    Index, inspect, Float
)
from sqlalchemy.orm import (
    declarative_base, sessionmaker, relationship, 
    Session, joinedload
)
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import (
    retry, stop_after_attempt, wait_random_exponential, 
    wait_fixed, wait_exponential
)
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from openai import OpenAI
from streamlit_tags import st_tags
import plotly.express as px
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Load environment variables
load_dotenv()

# Initialize logging
LOGGING_CONFIG: Dict[str, Any] = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Disable urllib3 warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Database configuration
DB_CONFIG = {
    'host': os.getenv("SUPABASE_DB_HOST"),
    'name': os.getenv("SUPABASE_DB_NAME"),
    'user': os.getenv("SUPABASE_DB_USER"),
    'password': os.getenv("SUPABASE_DB_PASSWORD"),
    'port': os.getenv("SUPABASE_DB_PORT")
}

# Validate database configuration
if not all(DB_CONFIG.values()):
    raise ValueError("One or more required database environment variables are not set")

DATABASE_URL = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['name']}"

# Configure database engine with retry mechanism
def get_engine():
    """Create and return a database engine with optimized settings."""
    return create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        connect_args={
            'connect_timeout': 10,
            'application_name': 'autoclient_app',
            'options': '-c statement_timeout=30000'
        }
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def initialize_engine():
    """Initialize database engine with retry mechanism."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine

# Initialize engine
try:
    engine = initialize_engine()
except Exception as e:
    logger.error(f"Failed to initialize database engine: {str(e)}")
    raise

# Configure session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=Session
)

Base = declarative_base()

# Add cleanup handler
@atexit.register
def cleanup_engine() -> None:
    """Clean up database engine on application exit."""
    if 'engine' in globals():
        engine.dispose()
        logger.info("Database engine disposed")

# Thread-local storage
thread_local = local()

@contextmanager
def db_session():
    """Provide a transactional scope around a series of operations."""
    session = None
    try:
        session = SessionLocal()
        yield session
        session.commit()
    except SQLAlchemyError as e:
        if session:
            session.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Unexpected error: {str(e)}")
        raise
    finally:
        if session:
            session.close()
            if hasattr(thread_local, "session"):
                del thread_local.session

# Test database connection
try:
    with db_session() as session:
        session.execute(text("SELECT 1"))
except Exception as e:
    st.error(f"Failed to connect to database: {str(e)}")
    logger.error(f"Database connection error: {str(e)}")
    raise

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    __table_args__ = (
        Index('idx_term', 'term'),
        Index('idx_term_group', 'group_id'),
        Index('idx_term_campaign', 'campaign_id'),
        Index('idx_term_created', 'created_at'),
    )
    id = Column(BigInteger, primary_key=True)
    term = Column(Text, nullable=False)
    group_id = Column(BigInteger, ForeignKey('search_term_groups.id', ondelete='SET NULL'))
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id', ondelete='CASCADE'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    effectiveness_score = Column(Float, default=0.0)
    last_used = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    group = relationship("SearchTermGroup", back_populates="search_terms")
    campaign = relationship("Campaign", back_populates="search_terms")
    lead_sources = relationship("LeadSource", back_populates="search_term")

class SearchTermEffectiveness(Base):
    __tablename__ = 'search_term_effectiveness'
    id = Column(BigInteger, primary_key=True)
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    total_results = Column(BigInteger, default=0)
    valid_leads = Column(BigInteger, default=0)
    irrelevant_leads = Column(BigInteger, default=0)
    blogs_found = Column(BigInteger, default=0)
    directories_found = Column(BigInteger, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    search_term = relationship("SearchTerm")

class OptimizedSearchTerm(Base):
    __tablename__ = 'optimized_search_terms'
    id = Column(BigInteger, primary_key=True)
    original_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    term = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    original_term = relationship("SearchTerm")

# Update the fetch_recent_searches function
def fetch_recent_searches(session, limit=5):
    return session.query(SearchTerm).order_by(
        SearchTerm.created_at.desc()
    ).limit(limit).all()

# Update manual_search_page function
def manual_search_page():
    st.title("Manual Search")
    
    with db_session() as session:
        # Show active processes first
        active_processes = session.query(SearchProcess).filter(
            SearchProcess.status.in_(['running', 'completed'])
        ).order_by(SearchProcess.created_at.desc()).all()
        
        if active_processes:
            st.subheader("Active Search Processes")
            for process in active_processes:
                with st.expander(f"Process {process.id} - {process.status.title()} - Started at {process.created_at.strftime('%Y-%m-%d %H:%M:%S')}", expanded=True):
                    display_process_logs(process.id)
        
        # Main search interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            recent_searches = fetch_recent_searches(session)
            recent_terms = [term.term for term in recent_searches]
            
            search_terms = st_tags(
                label='Enter Search Terms',
                text='Press enter after each term',
                value=st.session_state.get('search_terms', []),
                suggestions=recent_terms,
                key='search_terms_input'
            )
            
            if search_terms != st.session_state.get('search_terms', []):
                st.session_state.search_terms = search_terms
            
            num_results = st.number_input('Results per term', min_value=1, max_value=100, value=10)

# ... rest of the code ...
