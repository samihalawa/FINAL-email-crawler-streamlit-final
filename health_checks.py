# health_checks.py
import streamlit as st
import logging
import sys
import time
from streamlit.web import cli as stcli
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from dotenv import load_dotenv
import os
import boto3
from botocore.exceptions import ClientError
import smtplib
import pandas as pd
from datetime import datetime, timedelta
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.web.server import Server
from models import (
    Project, Campaign, KnowledgeBase, Lead, EmailTemplate, 
    EmailCampaign, SearchTermGroup, SearchTerm, AutomationLog,
    EmailSettings
)

# Import page functions from streamlit_app
from streamlit_app import (
    manual_search_page,
    bulk_send_page,
    view_leads_page,
    knowledge_base_page,
    search_terms_page,
    email_templates_page,
    manual_search_worker_page,
    view_campaign_logs,
    settings_page,
    projects_campaigns_page,
    autoclient_ai_page
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('health_checks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

@contextmanager
def db_session():
    """Context manager for database sessions."""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        session.close()

def check_database_connection():
    """Check if database connection is working."""
    try:
        with db_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful.")
            print("✅ Database connection successful.")
            return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {str(e)}")
        print(f"❌ Database connection failed: {str(e)}")
        return False

def check_email_service():
    """Check if email service is properly configured."""
    try:
        with db_session() as session:
            email_settings = session.query(EmailSettings).first()
            if not email_settings:
                logger.error("❌ No email settings found in database.")
                print("❌ No email settings found in database.")
                return False
            logger.info("✅ Email settings found and validated.")
            print("✅ Email settings found and validated.")
            return True
    except Exception as e:
        logger.error(f"❌ Email service check failed: {str(e)}")
        print(f"❌ Email service check failed: {str(e)}")
        return False

def run_streamlit_app():
    """Run the Streamlit app and return the server instance."""
    try:
        # Get the path to the main app file
        app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
        if not os.path.exists(app_path):
            logger.error(f"❌ Streamlit app file not found at: {app_path}")
            return None

        # Initialize Streamlit runtime
        st.runtime.set_script_path(app_path)
        st.runtime.set_page_config(
            page_title="AutoclientAI Health Check",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Import the main script module
        import importlib.util
        spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Create a mock server object
        class MockServer:
            def __init__(self, module):
                self._main_script_module = module
                self.started = False
                self.script_path = app_path
                self.command_line = "streamlit run"

            def start(self):
                self.started = True
                return True

            def stop(self):
                self.started = False
                return True

        server = MockServer(module)
        
        # Start the server
        if server.start():
            logger.info("✅ Streamlit app loaded successfully.")
            print("✅ Streamlit app loaded successfully.")
            return server
        else:
            logger.error("❌ Failed to load Streamlit app")
            print("❌ Failed to load Streamlit app")
            return None
                
    except Exception as e:
        logger.error(f"❌ Failed to load Streamlit app: {str(e)}")
        print(f"❌ Failed to load Streamlit app: {str(e)}")
        return None

def check_page_health(server, page_name, page_func):
    """Check the health of a specific page with timeout."""
    try:
        # Set timeout for page checks
        start_time = datetime.now()
        timeout = timedelta(seconds=30)

        while True:
            try:
                # Reset session state for clean test
                for key in list(st.session_state.keys()):
                    del st.session_state[key]

                # Set the active page in session state
                st.session_state.active_page = page_name

                # Initialize Streamlit context for this page
                st.runtime.set_script_path(server.script_path)
                st.runtime.set_page_config(
                    page_title=f"AutoclientAI - {page_name}",
                    layout="wide",
                    initial_sidebar_state="expanded"
                )

                # Run the page
                page_func()
                logger.info(f"✅ Page '{page_name}' health check passed.")
                print(f"✅ Page '{page_name}' health check passed.")
                return True

            except Exception as e:
                if datetime.now() - start_time > timeout:
                    raise TimeoutError(f"Page check timed out for {page_name}")
                logger.warning(f"Retrying page '{page_name}' after error: {str(e)}")
                time.sleep(1)
                continue

    except Exception as e:
        logger.error(f"❌ Page '{page_name}' health check failed: {str(e)}")
        print(f"❌ Page '{page_name}' health check failed: {str(e)}")
        return False

def check_settings_page(server):
    """Check the settings page for basic functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "🔄 Settings"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()
        
        # Check email settings display
        with db_session() as session:
            email_settings = session.query(EmailSettings).first()
            if not email_settings:
                logger.warning("No email settings found to test with")
            else:
                # Test email settings display
                st.session_state.show_email_settings = True
                main_func()
                
                # Test email settings form
                st.session_state.show_add_email_form = True
                main_func()
                
                # Test edit functionality
                st.session_state.edit_email_setting = email_settings
                main_func()
                
                # Clean up session state
                del st.session_state.edit_email_setting
                st.session_state.show_add_email_form = False
                st.session_state.show_email_settings = False

        # Verify required components are present
        main_func()
        if 'email_settings' not in st.session_state:
            logger.error("❌ Settings page: email_settings not found in session state")
            return False

        logger.info("✅ Settings page health check passed")
        print("✅ Settings page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Settings page health check failed: {str(e)}")
        print(f"❌ Settings page health check failed: {str(e)}")
        return False

def check_autoclient_ai_page(server):
    """Check the AutoclientAI page for basic functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "🤖 AutoclientAI"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()

        # Initialize required session state variables
        st.session_state.total_leads_found = 0
        st.session_state.total_emails_sent = 0
        st.session_state.automation_logs = []
        st.session_state.email_logs = []
        
        # Test automation controls
        st.session_state.start_automation = True
        main_func()
        st.session_state.start_automation = False
        
        # Test campaign selection
        with db_session() as session:
            campaign = session.query(Campaign).first()
            if campaign:
                st.session_state.current_campaign_id = campaign.id
                main_func()
            else:
                logger.warning("No campaigns found to test with")

        # Verify all required components are present
        required_states = [
            'total_leads_found',
            'total_emails_sent',
            'automation_logs',
            'email_logs',
            'automation_status'
        ]
        
        for state in required_states:
            if state not in st.session_state:
                logger.error(f"❌ AutoclientAI page: {state} not found in session state")
                return False

        # Test automation log display
        st.session_state.automation_logs = [
            {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "Test automation log"}
        ]
        main_func()

        # Test email log display
        st.session_state.email_logs = [
            {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "Test email log"}
        ]
        main_func()

        logger.info("✅ AutoclientAI page health check passed")
        print("✅ AutoclientAI page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ AutoclientAI page health check failed: {str(e)}")
        print(f"❌ AutoclientAI page health check failed: {str(e)}")
        return False

def check_automation_control_panel_page(server):
    """Check the Automation Control Panel page for basic functionality."""
    try:
        st.session_state.active_page = "⚙️ Automation Control"
        main_func = server._main_script_module.main
        main_func()

        # Simulate toggling automation status
        st.session_state.automation_status = not st.session_state.get('automation_status', False)
        main_func()
        st.session_state.automation_status = not st.session_state.get('automation_status', False)
        main_func()

        # Simulate performing a quick scan
        with st.spinner():
            with db_session() as session:
                new_leads = session.query(Lead).filter(Lead.is_processed == False).count()
                session.query(Lead).filter(Lead.is_processed == False).update({Lead.is_processed: True})
                session.commit()
                logger.info(f"Quick scan completed! Found {new_leads} new leads.")

        logger.info("Automation Control Panel page health check passed.")
        return True
    except Exception as e:
        logger.error(f"Automation Control Panel page health check failed: {str(e)}")
        return False

def check_view_sent_email_campaigns_page(server):
    """Check the View Sent Email Campaigns page for basic functionality."""
    try:
        st.session_state.active_page = "📨 Sent Campaigns"
        main_func = server._main_script_module.main
        main_func()

        with db_session() as session:
            email_campaigns = fetch_sent_email_campaigns(session)
            if not email_campaigns.empty:
                # Simulate selecting a campaign
                selected_campaign = email_campaigns['ID'].iloc[0]
                st.session_state.selected_campaign = selected_campaign
                main_func()
                del st.session_state.selected_campaign

        logger.info("View Sent Email Campaigns page health check passed.")
        return True
    except Exception as e:
        logger.error(f"View Sent Email Campaigns page health check failed: {str(e)}")
        return False

def fetch_sent_email_campaigns(session):
    """Fetch sent email campaigns for health check."""
    try:
        query = session.query(
            EmailCampaign.id.label("ID"),
            Lead.email.label("Email"),
            EmailTemplate.subject.label("Subject"),
            EmailCampaign.sent_at.label("Sent At"),
            EmailCampaign.customized_content.label("Content"),
            EmailCampaign.status.label("Status")
        ).join(Lead, EmailCampaign.lead_id == Lead.id).join(EmailTemplate, EmailCampaign.template_id == EmailTemplate.id).filter(EmailCampaign.sent_at.isnot(None))
        df = pd.read_sql(query.statement, query.session.bind)
        return df
    except Exception as e:
        logger.error(f"Error fetching sent email campaigns: {str(e)}")
        return pd.DataFrame()

def check_manual_search_page(server):
    """Check the Manual Search page functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "🔍 Manual Search"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()

        # Test search form with valid inputs
        st.session_state.search_terms = "test engineer spain"
        st.session_state.num_results = 10
        st.session_state.ignore_previously_fetched = True
        st.session_state.optimize_english = True
        st.session_state.language = "ES"
        main_func()
        
        # Test search initiation
        st.session_state.start_search = True
        main_func()
        
        # Wait briefly for search to process
        time.sleep(2)
        
        # Verify search components
        if 'search_results' not in st.session_state:
            logger.error("❌ Manual Search page: No search results container")
            return False
            
        if 'search_logs' not in st.session_state:
            logger.error("❌ Manual Search page: No search logs container")
            return False
            
        # Test search logs
        st.session_state.search_logs = [
            {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "Processing search term: test engineer spain"}
        ]
        main_func()
        
        # Test invalid input handling
        st.session_state.search_terms = ""
        main_func()
        if 'search_error' not in st.session_state:
            logger.error("❌ Manual Search page: No error handling for invalid input")
            return False
            
        # Test search results display
        st.session_state.search_results = pd.DataFrame({
            'Company': ['Test Company'],
            'Name': ['John Doe'],
            'Email': ['test@example.com'],
            'Source': ['LinkedIn']
        })
        main_func()
        
        # Clean up
        st.session_state.start_search = False
        main_func()
        
        logger.info("✅ Manual Search page health check passed")
        print("✅ Manual Search page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Manual Search page health check failed: {str(e)}")
        print(f"❌ Manual Search page health check failed: {str(e)}")
        return False

def check_bulk_send_page(server):
    """Check the Bulk Send page functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "📦 Bulk Send"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()

        # Test template selection and email settings
        with db_session() as session:
            # Get test data
            template = session.query(EmailTemplate).first()
            email_settings = session.query(EmailSettings).first()
            
            if not template:
                logger.warning("No email templates found to test with")
            else:
                # Test template selection
                st.session_state.selected_template = template.id
                main_func()
                
                if not email_settings:
                    logger.warning("No email settings found to test with")
                else:
                    # Test email settings
                    st.session_state.from_email = email_settings.email
                    st.session_state.reply_to = email_settings.email
                    main_func()

                    # Test email preview
                    st.session_state.preview_email = True
                    main_func()
                    st.session_state.preview_email = False

                    # Test sending functionality
                    st.session_state.send_emails = True
                    main_func()
                    
                    # Verify sending components
                    if 'email_sent_success' not in st.session_state:
                        logger.error("❌ Bulk Send page: No email sending status indicator")
                        return False
                        
                    if 'sending_progress' not in st.session_state:
                        logger.error("❌ Bulk Send page: No sending progress indicator")
                        return False
                    
                    # Clean up
                    st.session_state.send_emails = False
                    main_func()

        # Verify required components
        required_components = [
            'template_list',
            'email_settings',
            'recipient_count',
            'sending_enabled'
        ]
        
        for component in required_components:
            if component not in st.session_state:
                logger.error(f"❌ Bulk Send page: {component} not found in session state")
                return False

        logger.info("✅ Bulk Send page health check passed")
        print("✅ Bulk Send page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Bulk Send page health check failed: {str(e)}")
        print(f"❌ Bulk Send page health check failed: {str(e)}")
        return False

def check_view_leads_page(server):
    """Check the View Leads page functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "👥 View Leads"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()

        # Test lead filtering
        filter_options = ["all", "new", "contacted", "responded", "converted"]
        for filter_option in filter_options:
            st.session_state.lead_filter = filter_option
            main_func()
            
            if 'leads_df' not in st.session_state:
                logger.error(f"❌ View Leads page: No leads displayed for filter '{filter_option}'")
                return False

        # Test deleted leads view
        st.session_state.show_deleted = True
        main_func()
        st.session_state.show_deleted = False
        
        # Test lead data with database
        with db_session() as session:
            leads = session.query(Lead).limit(5).all()
            if leads:
                # Test lead selection
                st.session_state.selected_lead = leads[0].id
                main_func()
                
                # Test lead details view
                st.session_state.show_lead_details = True
                main_func()
                
                # Test lead editing
                st.session_state.edit_lead = leads[0]
                main_func()
                
                # Clean up
                del st.session_state.edit_lead
                st.session_state.show_lead_details = False
            else:
                logger.warning("No leads found to test with")

        # Verify required components
        required_components = [
            'leads_df',
            'lead_filter',
            'show_deleted',
            'total_leads',
            'filtered_leads_count'
        ]
        
        for component in required_components:
            if component not in st.session_state:
                logger.error(f"❌ View Leads page: {component} not found in session state")
                return False

        # Test export functionality
        st.session_state.export_leads = True
        main_func()
        if 'export_data' not in st.session_state:
            logger.error("❌ View Leads page: Export functionality not working")
            return False
        st.session_state.export_leads = False

        logger.info("✅ View Leads page health check passed")
        print("✅ View Leads page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ View Leads page health check failed: {str(e)}")
        print(f"❌ View Leads page health check failed: {str(e)}")
        return False

def check_knowledge_base_page(server):
    """Check the Knowledge Base page functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "📚 Knowledge Base"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()

        # Test KB form functionality
        with db_session() as session:
            # Get test data
            project = session.query(Project).first()
            if not project:
                logger.warning("No projects found to test with")
                project = Project(project_name="Test Project")
                session.add(project)
                session.commit()
            
            kb = session.query(KnowledgeBase).filter_by(project_id=project.id).first()
            
            if kb:
                # Test KB display
                st.session_state.current_project_id = project.id
                main_func()
                
                # Test KB editing
                st.session_state.edit_kb = kb
                main_func()
                
                # Test form fields
                test_data = {
                    'kb_name': 'Updated KB Name',
                    'kb_bio': 'Test bio',
                    'kb_values': 'Test values',
                    'contact_name': 'John Doe',
                    'contact_role': 'Manager',
                    'contact_email': 'test@example.com',
                    'company_description': 'Test company',
                    'company_mission': 'Test mission',
                    'company_target_market': 'Test market',
                    'product_name': 'Test product',
                    'product_description': 'Test description',
                    'product_target_customer': 'Test customer'
                }
                
                for field, value in test_data.items():
                    setattr(st.session_state, field, value)
                main_func()
                
                # Clean up
                del st.session_state.edit_kb
            else:
                # Test creating new KB
                st.session_state.show_add_kb_form = True
                main_func()
                
                # Fill form with test data
                st.session_state.kb_name = "Test KB"
                st.session_state.kb_bio = "Test Bio"
                main_func()

        # Verify required components
        required_components = [
            'current_project_id',
            'kb_form_submitted',
            'kb_data',
            'form_valid'
        ]
        
        for component in required_components:
            if component not in st.session_state:
                logger.error(f"❌ Knowledge Base page: {component} not found in session state")
                return False

        logger.info("✅ Knowledge Base page health check passed")
        print("✅ Knowledge Base page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Knowledge Base page health check failed: {str(e)}")
        print(f"❌ Knowledge Base page health check failed: {str(e)}")
        return False

def check_search_terms_page(server):
    """Check the Search Terms page functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "🔑 Search Terms"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()

        # Test search term group functionality
        with db_session() as session:
            # Get test data
            group = session.query(SearchTermGroup).first()
            
            if group:
                # Test group selection
                st.session_state.selected_group = group.id
                main_func()
                
                # Test group editing
                st.session_state.edit_group = group
                main_func()
                
                # Test form fields
                test_data = {
                    'group_name': 'Updated Group Name',
                    'group_description': 'Test description',
                    'email_template': 'Test template'
                }
                
                for field, value in test_data.items():
                    setattr(st.session_state, field, value)
                main_func()
                
                # Test search terms
                search_terms = session.query(SearchTerm).filter_by(group_id=group.id).all()
                if search_terms:
                    # Test term editing
                    st.session_state.edit_term = search_terms[0]
                    main_func()
                    del st.session_state.edit_term
                
                # Clean up
                del st.session_state.edit_group
            
            # Test creating new group
            st.session_state.show_add_group_form = True
            main_func()
            
            # Fill form with test data
            st.session_state.new_group_name = "Test Group"
            st.session_state.new_group_description = "Test Description"
            main_func()
            
            # Test adding new search term
            st.session_state.show_add_term_form = True
            main_func()
            
            # Fill term form with test data
            st.session_state.new_term = "test engineer"
            st.session_state.new_term_category = "job_title"
            main_func()

        # Verify required components
        required_components = [
            'groups_list',
            'selected_group',
            'terms_list',
            'form_valid',
            'group_stats'
        ]
        
        for component in required_components:
            if component not in st.session_state:
                logger.error(f"❌ Search Terms page: {component} not found in session state")
                return False

        logger.info("✅ Search Terms page health check passed")
        print("✅ Search Terms page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Search Terms page health check failed: {str(e)}")
        print(f"❌ Search Terms page health check failed: {str(e)}")
        return False

def check_email_templates_page(server):
    """Check the Email Templates page functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "✉️ Email Templates"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()

        # Test template functionality
        with db_session() as session:
            # Get test data
            campaign = session.query(Campaign).first()
            if not campaign:
                logger.warning("No campaigns found to test with")
                campaign = Campaign(campaign_name="Test Campaign")
                session.add(campaign)
                session.commit()
            
            template = session.query(EmailTemplate).filter_by(campaign_id=campaign.id).first()
            
            if template:
                # Test template selection
                st.session_state.selected_template_id = template.id
                main_func()
                
                # Test template editing
                st.session_state.edit_template_id = template.id
                main_func()
                
                # Test form fields
                test_data = {
                    'template_name': 'Updated Template Name',
                    'subject': 'Test Subject',
                    'body_content': 'Test Content',
                    'is_ai_customizable': True,
                    'language': 'ES'
                }
                
                for field, value in test_data.items():
                    setattr(st.session_state, field, value)
                main_func()
                
                # Test template preview
                st.session_state.preview_template = True
                main_func()
                st.session_state.preview_template = False
                
                # Clean up
                del st.session_state.edit_template_id
            
            # Test creating new template
            st.session_state.show_add_template_form = True
            main_func()
            
            # Fill form with test data
            st.session_state.new_template_name = "Test Template"
            st.session_state.new_template_subject = "Test Subject"
            st.session_state.new_template_content = "Test Content"
            st.session_state.new_template_language = "ES"
            main_func()

        # Verify required components
        required_components = [
            'templates_list',
            'selected_template_id',
            'campaign_list',
            'form_valid',
            'template_stats'
        ]
        
        for component in required_components:
            if component not in st.session_state:
                logger.error(f"❌ Email Templates page: {component} not found in session state")
                return False

        logger.info("✅ Email Templates page health check passed")
        print("✅ Email Templates page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Email Templates page health check failed: {str(e)}")
        print(f"❌ Email Templates page health check failed: {str(e)}")
        return False

def check_manual_search_worker_page(server):
    """Check the Manual Search Worker page functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "⚙️ Manual Search Worker"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()

        # Test worker functionality
        with db_session() as session:
            # Get test data
            automation_log = session.query(AutomationLog).first()
            
            if not automation_log:
                logger.warning("No automation logs found to test with")
                # Create test automation log
                campaign = session.query(Campaign).first()
                if campaign:
                    automation_log = AutomationLog(
                        campaign_id=campaign.id,
                        leads_gathered=0,
                        emails_sent=0,
                        status="pending",
                        logs=[]
                    )
                    session.add(automation_log)
                    session.commit()
            
            if automation_log:
                # Test log selection
                st.session_state.automation_log_id = automation_log.id
                main_func()
                
                # Test search process
                if 'search_process' in st.session_state and st.session_state.search_process:
                    try:
                        st.session_state.search_process.terminate()
                    except:
                        pass
                    st.session_state.search_process = None

                # Test starting search
                st.session_state.start_search = True
                main_func()
                
                # Wait briefly for process to start
                time.sleep(2)
                
                # Verify search process
                if 'search_process' not in st.session_state:
                    logger.error("❌ Manual Search Worker page: Search process not created")
                    return False
                
                # Test stopping search
                st.session_state.stop_search = True
                main_func()
                
                # Clean up
                if 'search_process' in st.session_state and st.session_state.search_process:
                    try:
                        st.session_state.search_process.terminate()
                    except:
                        pass
                    st.session_state.search_process = None

        # Verify required components
        required_components = [
            'automation_log_id',
            'automation_logs',
            'search_status',
            'worker_stats',
            'process_running'
        ]
        
        for component in required_components:
            if component not in st.session_state:
                logger.error(f"❌ Manual Search Worker page: {component} not found in session state")
                return False

        # Test log display
        st.session_state.automation_logs = [
            {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "Test worker log"}
        ]
        main_func()

        logger.info("✅ Manual Search Worker page health check passed")
        print("✅ Manual Search Worker page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Manual Search Worker page health check failed: {str(e)}")
        print(f"❌ Manual Search Worker page health check failed: {str(e)}")
        return False

def check_email_logs_page(server):
    """Check the Email Logs page functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "📨 Email Logs"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()

        # Test log filtering
        filter_options = ["all", "sent", "opened", "clicked", "failed"]
        for filter_option in filter_options:
            st.session_state.selected_filter = filter_option
            main_func()
            
            if 'email_logs_df' not in st.session_state:
                logger.error(f"❌ Email Logs page: No logs displayed for filter '{filter_option}'")
                return False

        # Test date range filtering
        st.session_state.date_range = (
            datetime.now() - timedelta(days=7),
            datetime.now()
        )
        main_func()

        # Test campaign filtering
        with db_session() as session:
            campaign = session.query(Campaign).first()
            if campaign:
                st.session_state.selected_campaign = campaign.id
                main_func()
            else:
                logger.warning("No campaigns found to test with")

        # Test log details view
        with db_session() as session:
            email_campaign = session.query(EmailCampaign).first()
            if email_campaign:
                st.session_state.selected_log = email_campaign.id
                main_func()
                
                # Test engagement data
                st.session_state.show_engagement_data = True
                main_func()
                st.session_state.show_engagement_data = False
            else:
                logger.warning("No email campaigns found to test with")

        # Verify required components
        required_components = [
            'email_logs_df',
            'selected_filter',
            'date_range',
            'campaign_list',
            'log_stats'
        ]
        
        for component in required_components:
            if component not in st.session_state:
                logger.error(f"❌ Email Logs page: {component} not found in session state")
                return False

        # Test export functionality
        st.session_state.export_logs = True
        main_func()
        if 'export_data' not in st.session_state:
            logger.error("❌ Email Logs page: Export functionality not working")
            return False
        st.session_state.export_logs = False

        logger.info("✅ Email Logs page health check passed")
        print("✅ Email Logs page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Email Logs page health check failed: {str(e)}")
        print(f"❌ Email Logs page health check failed: {str(e)}")
        return False

def check_projects_campaigns_page(server):
    """Check the Projects & Campaigns page functionality."""
    try:
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.session_state.active_page = "🚀 Projects & Campaigns"
        main_func = server._main_script_module.main
        
        # First check - basic page load
        main_func()

        # Test project functionality
        with db_session() as session:
            # Get or create test project
            project = session.query(Project).first()
            if not project:
                logger.warning("No projects found to test with")
                project = Project(project_name="Test Project")
                session.add(project)
                session.commit()
            
            if project:
                # Test project selection
                st.session_state.current_project_id = project.id
                main_func()
                
                # Test project editing
                st.session_state.edit_project = project
                main_func()
                
                # Test form fields
                st.session_state.project_name = "Updated Project Name"
                main_func()
                
                # Clean up
                del st.session_state.edit_project
                
                # Test campaign functionality
                campaign = session.query(Campaign).filter_by(project_id=project.id).first()
                if campaign:
                    # Test campaign selection
                    st.session_state.current_campaign_id = campaign.id
                    main_func()
                    
                    # Test campaign editing
                    st.session_state.edit_campaign = campaign
                    main_func()
                    
                    # Test form fields
                    test_data = {
                        'campaign_name': 'Updated Campaign Name',
                        'campaign_type': 'Email',
                        'auto_send': True,
                        'loop_automation': False,
                        'ai_customization': True,
                        'max_emails_per_group': 50,
                        'loop_interval': 60
                    }
                    
                    for field, value in test_data.items():
                        setattr(st.session_state, field, value)
                    main_func()
                    
                    # Clean up
                    del st.session_state.edit_campaign
                else:
                    # Test creating new campaign
                    st.session_state.show_add_campaign_form = True
                    main_func()
                    
                    # Fill form with test data
                    st.session_state.new_campaign_name = "Test Campaign"
                    st.session_state.new_campaign_type = "Email"
                    main_func()

        # Test creating new project
        st.session_state.show_add_project_form = True
        main_func()
        
        # Fill form with test data
        st.session_state.new_project_name = "New Test Project"
        main_func()

        # Verify required components
        required_components = [
            'projects_list',
            'current_project_id',
            'campaigns_list',
            'project_stats',
            'campaign_stats'
        ]
        
        for component in required_components:
            if component not in st.session_state:
                logger.error(f"❌ Projects & Campaigns page: {component} not found in session state")
                return False

        logger.info("✅ Projects & Campaigns page health check passed")
        print("✅ Projects & Campaigns page controls verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Projects & Campaigns page health check failed: {str(e)}")
        print(f"❌ Projects & Campaigns page health check failed: {str(e)}")
        return False

def main():
    """Main function to run health checks."""
    print("\n=== Starting AutoclientAI Health Checks ===\n")
    logger.info("Starting health checks...")

    # Check database connection
    print("1. Checking Database Connection...")
    if not check_database_connection():
        logger.error("❌ Database connection failed. Health check aborted.")
        print("❌ Database connection failed. Health check aborted.")
        sys.exit(1)
    print("✅ Database connection successful.")

    # Check email service
    print("\n2. Checking Email Service...")
    if not check_email_service():
        logger.error("❌ Email service check failed. Health check aborted.")
        print("❌ Email service check failed. Health check aborted.")
        sys.exit(1)
    print("✅ Email service check successful.")

    # Import the main script module directly
    print("\n3. Loading Streamlit App...")
    try:
        app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
        if not os.path.exists(app_path):
            logger.error(f"❌ Streamlit app file not found at: {app_path}")
            sys.exit(1)
            
        import importlib.util
        spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✅ Streamlit app loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load Streamlit app: {str(e)}")
        print(f"❌ Failed to load Streamlit app: {str(e)}")
        sys.exit(1)

    # Create mock server
    class MockServer:
        def __init__(self, module):
            self._main_script_module = module
            self.started = False
            self.script_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")

    server = MockServer(module)

    # Define page check functions
    page_checks = {
        "🔍 Manual Search": lambda: check_page_health("Manual Search", manual_search_page),
        "📦 Bulk Send": lambda: check_page_health("Bulk Send", bulk_send_page),
        "👥 View Leads": lambda: check_page_health("View Leads", view_leads_page),
        "📚 Knowledge Base": lambda: check_page_health("Knowledge Base", knowledge_base_page),
        "🔑 Search Terms": lambda: check_page_health("Search Terms", search_terms_page),
        "✉️ Email Templates": lambda: check_page_health("Email Templates", email_templates_page),
        "⚙️ Manual Search Worker": lambda: check_page_health("Manual Search Worker", manual_search_worker_page),
        "📨 Email Logs": lambda: check_page_health("Email Logs", view_campaign_logs),
        "🔄 Settings": lambda: check_page_health("Settings", settings_page),
        "🚀 Projects & Campaigns": lambda: check_page_health("Projects & Campaigns", projects_campaigns_page),
        "🤖 AutoclientAI": lambda: check_page_health("AutoclientAI", autoclient_ai_page)
    }

    # Check each page
    print("\n4. Checking Individual Pages...")
    results = {}
    total_pages = len(page_checks)
    passed_pages = 0

    for i, (page_name, check_func) in enumerate(page_checks.items(), 1):
        print(f"\nChecking page {i}/{total_pages}: {page_name}")
        try:
            if check_func(server):
                passed_pages += 1
                results[page_name] = "✅ Passed"
            else:
                results[page_name] = "❌ Failed"
        except Exception as e:
            logger.error(f"Error checking {page_name}: {str(e)}")
            results[page_name] = f"❌ Error: {str(e)}"

    # Print results
    print("\n=== Health Check Results ===")
    print(f"\nTotal Pages: {total_pages}")
    print(f"Pages Passed: {passed_pages}")
    print(f"Success Rate: {(passed_pages/total_pages)*100:.1f}%\n")
    
    print("Detailed Results:")
    for page, result in results.items():
        print(f"{page}: {result}")

    if passed_pages == total_pages:
        logger.info("✅ All health checks passed successfully.")
        print("\n✅ All health checks passed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some health checks failed.")
        print("\n❌ Some health checks failed. See detailed results above.")
        sys.exit(1)

if __name__ == "__main__":
    main()