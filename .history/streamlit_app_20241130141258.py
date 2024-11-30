import os, json, re, logging, asyncio, time, requests, pandas as pd, streamlit as st, openai, boto3, uuid, aiohttp, urllib3, random, html, smtplib, threading
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from fake_useragent import UserAgent
from sqlalchemy import func, create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, select, text, distinct, and_, Index, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed, wait_exponential
from email_validator import validate_email, EmailNotValidError
from streamlit_option_menu import option_menu
from openai import OpenAI 
from typing import List, Optional
from urllib.parse import urlparse, urlencode
from streamlit_tags import st_tags
import plotly.express as px
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager, wraps

class EmailRateLimiter:
    def __init__(self, max_emails_per_minute=30):
        self.max_emails_per_minute = max_emails_per_minute
        self.sent_timestamps = []
        self._lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if we've exceeded our rate limit"""
        with self._lock:
            now = time.time()
            # Remove timestamps older than 1 minute
            self.sent_timestamps = [ts for ts in self.sent_timestamps if now - ts < 60]
            
            if len(self.sent_timestamps) >= self.max_emails_per_minute:
                sleep_time = 60 - (now - self.sent_timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.sent_timestamps.append(now)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    rate_limiter = EmailRateLimiter()
    rate_limiter.wait_if_needed()
    
    try:
        email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
        if not email_settings:
            logging.error(f"No email settings found for {from_email}")
            return

        # ... rest of the function ...