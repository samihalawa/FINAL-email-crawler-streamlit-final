import unittest
import streamlit as st
from streamlit.testing.v1 import AppTest
import time
import os
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit_app import Base, Project, Campaign, Lead, EmailTemplate, EmailSettings, Settings, SearchTerm, LeadSource, EmailCampaign, KnowledgeBase, main, db_session, send_email_ses, save_email_campaign
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import boto3

# Configuration for a test database
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create a test engine and session
test_engine = create_engine(TEST_DATABASE_URL)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Mocking external services
class MockSMTP:
    def __init__(self, *args, **kwargs):
        pass

    def starttls(self):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg):
        return {}

    def quit(self):
        pass

    def close(self):
        pass

class MockSESClient:
    def send_email(self, *args, **kwargs):
        return {'MessageId': 'mocked_message_id'}

class MockBoto3Client:
    def client(self, service_name, *args, **kwargs):
        if service_name == 'ses':
            return MockSESClient()
        return None

class StreamlitAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create all tables in the test database
        Base.metadata.create_all(bind=test_engine)

    @classmethod
    def tearDownClass(cls):
        # Drop all tables after tests are done
        Base.metadata.drop_all(bind=test_engine)

    def setUp(self):
        """Setup before each test"""
        self.at = AppTest.from_file("streamlit_app.py").run(timeout=120)
        self.session = TestSessionLocal()

        # Create a test project
        self.project = Project(project_name="Test Project")
        self.session.add(self.project)
        self.session.flush()

        # Create a test campaign
        self.campaign = Campaign(campaign_name="Test Campaign", project_id=self.project.id)
        self.session.add(self.campaign)
        self.session.flush()

        # Create a test lead
        self.lead = Lead(email="test@example.com", first_name="Test", last_name="User", company="Test Company", job_title="Test Job")
        self.session.add(self.lead)
        self.session.flush()

        # Create a test email template
        self.email_template = EmailTemplate(template_name="Test Template", subject="Test Subject", body_content="Test Body", campaign_id=self.campaign.id)
        self.session.add(self.email_template)
        self.session.flush()

        # Create test email settings
        self.email_settings = EmailSettings(
            name="Test Email Settings",
            email="test@example.com",
            provider="smtp",
            smtp_server="smtp.example.com",
            smtp_port=587,
            smtp_username="testuser",
            smtp_password="testpassword"
        )
        self.session.add(self.email_settings)
        self.session.flush()

        # Create test application settings
        self.app_settings = Settings(
            name="application_settings",
            setting_type="application",
            value={
                "enable_ai": True,
                "debug_mode": False,
                "default_language": "ES",
                "max_search_results": 50,
                "email_batch_size": 10
            }
        )
        self.session.add(self.app_settings)
        self.session.flush()

        # Create test AI settings
        self.ai_settings = Settings(
            name="ai_settings",
            setting_type="ai",
            value={
                "api_key": "test_api_key",
                "api_base_url": "https://api.example.com/v1",
                "model_name": "gpt-4",
                "max_tokens": 1500,
                "temperature": 0.7,
                "inference_api": False
            }
        )
        self.session.add(self.ai_settings)
        self.session.flush()

        # Create a test search term
        self.search_term = SearchTerm(term="Test Search Term", campaign_id=self.campaign.id, language="en")
        self.session.add(self.search_term)
        self.session.flush()

        # Create a test lead source
        self.lead_source = LeadSource(lead_id=self.lead.id, search_term_id=self.search_term.id, url="https://www.example.com", domain="example.com", page_title="Example Page", meta_description="Example Description", http_status=200)
        self.session.add(self.lead_source)
        self.session.flush()

        # Create a test email campaign
        self.email_campaign = EmailCampaign(campaign_id=self.campaign.id, lead_id=self.lead.id, template_id=self.email_template.id, status="sent", message_id="test_message_id", tracking_id="test_tracking_id")
        self.session.add(self.email_campaign)
        self.session.flush()

        # Create a test knowledge base
        self.knowledge_base = KnowledgeBase(project_id=self.project.id, kb_name="Test KB", kb_bio="Test Bio", contact_name="Test Contact", contact_email="contact@example.com")
        self.session.add(self.knowledge_base)
        self.session.commit()

    def tearDown(self):
        """Tear down after each test"""
        self.session.close()

    def test_manual_search_page(self):
        # Test Manual Search Page
        self.at.sidebar.radio[0].set_value("🔍 Manual Search").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "manual_search_page")

        # Test search terms input
        self.at.text_input[0].input("Test Search Term").run(timeout=60)
        self.assertEqual(self.at.text_input[0].value, "Test Search Term")

        # Test number of results slider
        self.at.slider[0].set_value(20).run(timeout=60)
        self.assertEqual(self.at.slider[0].value, 20)

        # Test checkboxes
        self.at.checkbox[0].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[0].value)
        self.at.checkbox[1].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[1].value)
        self.at.checkbox[2].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[2].value)
        self.at.checkbox[3].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[3].value)

        # Test language selection
        self.at.selectbox[0].set_value("English").run(timeout=60)
        self.assertEqual(self.at.selectbox[0].value, "English")

        # Test email sending functionality
        self.at.checkbox[4].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[4].value)
        self.at.selectbox[1].set_value("Test Template").run(timeout=60)
        self.assertEqual(self.at.selectbox[1].value, "Test Template")
        self.at.selectbox[2].set_value("test@example.com").run(timeout=60)
        self.assertEqual(self.at.selectbox[2].value, "test@example.com")
        self.at.text_input[1].input("reply@example.com").run(timeout=60)
        self.assertEqual(self.at.text_input[1].value, "reply@example.com")

        # Test start search button
        with patch('streamlit_app.manual_search', return_value={'results': [{'Email': 'test@example.com', 'URL': 'https://www.example.com', 'Company': 'Test Company'}]}):
            with patch('streamlit_app.send_email_ses', return_value=({'MessageId': 'mocked_message_id'}, 'mocked_tracking_id')):
                with patch('streamlit_app.save_email_campaign') as mock_save_email_campaign:
                    self.at.button[0].click().run(timeout=60)
                    self.assertTrue(self.at.session_state['search_running'])
                    self.assertEqual(self.at.status.text, "Running...")
                    self.at.run(timeout=60)
                    self.assertFalse(self.at.session_state['search_running'])
                    self.assertEqual(self.at.status.text, "Complete!")
                    mock_save_email_campaign.assert_called()

    def test_bulk_send_page(self):
        # Test Bulk Send Page
        self.at.sidebar.radio[0].set_value("📦 Bulk Send").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "bulk_send_page")

        # Test email template selection
        self.at.selectbox[0].set_value("Test Template").run(timeout=60)
        self.assertEqual(self.at.selectbox[0].value, "Test Template")

        # Test from email selection
        self.at.selectbox[1].set_value("test@example.com").run(timeout=60)
        self.assertEqual(self.at.selectbox[1].value, "test@example.com")

        # Test reply-to email input
        self.at.text_input[0].input("reply@example.com").run(timeout=60)
        self.assertEqual(self.at.text_input[0].value, "reply@example.com")

        # Test leads input
        self.at.text_area[0].input("test1@example.com\ntest2@example.com").run(timeout=60)
        self.assertEqual(self.at.text_area[0].value, "test1@example.com\ntest2@example.com")

        # Test send emails button
        with patch('streamlit_app.bulk_send_emails', return_value=(["Email sent to test1@example.com", "Email sent to test2@example.com"], 2)):
            self.at.button[0].click().run(timeout=60)
            self.assertEqual(self.at.success[0].label, "Emails sent successfully: 2")

    def test_view_leads_page(self):
        # Test View Leads Page
        self.at.sidebar.radio[0].set_value("👥 View Leads").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "view_leads_page")

        # Test leads table
        self.assertTrue(self.at.data_editor)

        # Test search functionality
        self.at.text_input[0].input("test@example.com").run(timeout=60)
        self.assertEqual(self.at.text_input[0].value, "test@example.com")

        # Test pagination
        self.at.button[1].click().run(timeout=60)
        self.at.button[2].click().run(timeout=60)

    def test_search_terms_page(self):
        # Test Search Terms Page
        self.at.sidebar.radio[0].set_value("🔑 Search Terms").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "search_terms_page")

        # Test search terms table
        self.assertTrue(self.at.data_editor)

        # Test add new search term button
        self.at.button[0].click().run(timeout=60)
        self.at.text_input[0].input("New Search Term").run(timeout=60)
        self.at.button[1].click().run(timeout=60)

        # Test delete selected terms button
        self.at.checkbox[0].check().run(timeout=60)
        self.at.button[2].click().run(timeout=60)

    def test_email_templates_page(self):
        # Test Email Templates Page
        self.at.sidebar.radio[0].set_value("✉️ Email Templates").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "email_templates_page")

        # Test email templates table
        self.assertTrue(self.at.data_editor)

        # Test add new template button
        self.at.button[0].click().run(timeout=60)
        self.at.text_input[0].input("New Template").run(timeout=60)
        self.at.text_input[1].input("New Subject").run(timeout=60)
        self.at.text_area[0].input("New Body").run(timeout=60)
        self.at.button[1].click().run(timeout=60)

        # Test delete selected templates button
        self.at.checkbox[0].check().run(timeout=60)
        self.at.button[2].click().run(timeout=60)

    def test_projects_campaigns_page(self):
        # Test Projects & Campaigns Page
        self.at.sidebar.radio[0].set_value("🚀 Projects & Campaigns").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "projects_campaigns_page")

        # Test projects table
        self.assertTrue(self.at.data_editor)

        # Test add new project button
        self.at.button[0].click().run(timeout=60)
        self.at.text_input[0].input("New Project").run(timeout=60)
        self.at.button[1].click().run(timeout=60)

        # Test delete selected projects button
        self.at.checkbox[0].check().run(timeout=60)
        self.at.button[2].click().run(timeout=60)

        # Test campaigns table
        self.at.expander[0].expand().run(timeout=60)
        self.assertTrue(self.at.data_editor)

        # Test add new campaign button
        self.at.button[3].click().run(timeout=60)
        self.at.text_input[1].input("New Campaign").run(timeout=60)
        self.at.selectbox[0].set_value("Email").run(timeout=60)
        self.at.button[4].click().run(timeout=60)

        # Test delete selected campaigns button
        self.at.checkbox[1].check().run(timeout=60)
        self.at.button[5].click().run(timeout=60)

    def test_knowledge_base_page(self):
        # Test Knowledge Base Page
        self.at.sidebar.radio[0].set_value("📚 Knowledge Base").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "knowledge_base_page")

        # Test knowledge base form
        self.at.text_input[0].input("Test KB").run(timeout=60)
        self.at.text_area[0].input("Test Bio").run(timeout=60)
        self.at.text_area[1].input("Test Values").run(timeout=60)
        self.at.text_input[1].input("Test Contact").run(timeout=60)
        self.at.text_input[2].input("Test Role").run(timeout=60)
        self.at.text_input[3].input("contact@example.com").run(timeout=60)
        self.at.text_area[2].input("Test Company Description").run(timeout=60)
        self.at.text_area[3].input("Test Company Mission").run(timeout=60)
        self.at.text_area[4].input("Test Target Market").run(timeout=60)
        self.at.text_area[5].input("Test Other Info").run(timeout=60)
        self.at.text_input[4].input("Test Product").run(timeout=60)
        self.at.text_area[6].input("Test Product Description").run(timeout=60)
        self.at.text_area[7].input("Test Product Target Customer").run(timeout=60)
        self.at.text_area[8].input("Test Product Other").run(timeout=60)
        self.at.text_area[9].input("Test Other Context").run(timeout=60)
        self.at.text_area[10].input("Test Example Email").run(timeout=60)
        self.at.form_submit_button[0].click().run(timeout=60)

    def test_autoclient_ai_page(self):
        # Test AutoclientAI Page
        self.at.sidebar.radio[0].set_value("🤖 AutoclientAI").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "autoclient_ai_page")

        # Test target leads input
        self.at.number_input[0].input(5).run(timeout=60)
        self.assertEqual(self.at.number_input[0].value, 5)

        # Test search terms input
        self.at.text_area[0].input("Test Search Term 1\nTest Search Term 2").run(timeout=60)
        self.assertEqual(self.at.text_area[0].value, "Test Search Term 1\nTest Search Term 2")

        # Test start button
        with patch('streamlit_app.manual_search', return_value={'results': [{'Email': 'test@example.com', 'URL': 'https://www.example.com', 'Company': 'Test Company'}]}):
            with patch('streamlit_app.send_email_ses', return_value=({'MessageId': 'mocked_message_id'}, 'mocked_tracking_id')):
                with patch('streamlit_app.save_email_campaign') as mock_save_email_campaign:
                    self.at.button[0].click().run(timeout=60)
                    self.assertTrue(self.at.session_state['search_running'])
                    self.at.run(timeout=60)
                    self.assertFalse(self.at.session_state['search_running'])
                    mock_save_email_campaign.assert_called()

    def test_automation_control_panel_page(self):
        # Test Automation Control Panel Page
        self.at.sidebar.radio[0].set_value("⚙️ Automation Control").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "automation_control_panel_page")

        # Test start/stop automation button
        self.at.button[0].click().run(timeout=60)
        self.assertTrue(self.at.session_state['automation_status'])
        self.at.button[0].click().run(timeout=60)
        self.assertFalse(self.at.session_state['automation_status'])

        # Test perform quick scan button
        with patch('streamlit_app.db_session') as mock_db_session:
            mock_session = MagicMock()
            mock_db_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.count.return_value = 5
            self.at.button[1].click().run(timeout=60)
            self.assertEqual(self.at.success[0].label, "Quick scan completed! Found 5 new leads.")

    def test_view_campaign_logs(self):
        # Test View Campaign Logs Page
        self.at.sidebar.radio[0].set_value("📨 Email Logs").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "view_campaign_logs")

        # Test filter selection
        self.at.selectbox[0].set_value("All").run(timeout=60)
        self.assertEqual(self.at.selectbox[0].value, "All")
        self.at.selectbox[0].set_value("Errors").run(timeout=60)
        self.assertEqual(self.at.selectbox[0].value, "Errors")
        self.at.selectbox[0].set_value("Successes").run(timeout=60)
        self.assertEqual(self.at.selectbox[0].value, "Successes")
        self.at.selectbox[0].set_value("Email Sent").run(timeout=60)
        self.assertEqual(self.at.selectbox[0].value, "Email Sent")
        self.at.selectbox[0].set_value("Search").run(timeout=60)
        self.assertEqual(self.at.selectbox[0].value, "Search")

        # Test auto-scroll checkbox
        self.at.checkbox[0].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[0].value)
        self.at.checkbox[0].uncheck().run(timeout=60)
        self.assertFalse(self.at.checkbox[0].value)

    def test_settings_page(self):
        # Test Settings Page
        self.at.sidebar.radio[0].set_value("🔄 Settings").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "settings_page")

        # Test add email setting form
        self.at.text_input[0].input("Test Email Setting").run(timeout=60)
        self.at.text_input[1].input("test@example.com").run(timeout=60)
        self.at.selectbox[0].set_value("SMTP").run(timeout=60)
        self.at.number_input[0].input(100).run(timeout=60)
        self.at.number_input[1].input(10).run(timeout=60)
        self.at.checkbox[0].check().run(timeout=60)
        self.at.text_input[2].input("smtp.example.com").run(timeout=60)
        self.at.number_input[2].input(587).run(timeout=60)
        self.at.text_input[3].input("testuser").run(timeout=60)
        self.at.text_input[4].input("testpassword").run(timeout=60)
        self.at.form_submit_button[0].click().run(timeout=60)

        # Test application settings form
        self.at.checkbox[1].uncheck().run(timeout=60)
        self.at.checkbox[2].check().run(timeout=60)
        self.at.selectbox[1].set_value("EN").run(timeout=60)
        self.at.number_input[3].input(20).run(timeout=60)
        self.at.number_input[4].input(5).run(timeout=60)
        self.at.form_submit_button[1].click().run(timeout=60)

        # Test AI settings form
        self.at.text_input[5].input("test_api_key").run(timeout=60)
        self.at.text_input[6].input("https://api.example.com/v1").run(timeout=60)
        self.at.selectbox[2].set_value("gpt-3.5-turbo").run(timeout=60)
        self.at.number_input[5].input(1000).run(timeout=60)
        self.at.slider[0].set_value(0.5).run(timeout=60)
        self.at.form_submit_button[2].click().run(timeout=60)

        # Test database maintenance buttons
        self.at.button[0].click().run(timeout=60)
        self.at.button[1].click().run(timeout=60)

    def test_view_sent_email_campaigns(self):
        # Test View Sent Email Campaigns Page
        self.at.sidebar.radio[0].set_value("📨 Sent Campaigns").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "view_sent_email_campaigns")

        # Test sent email campaigns table
        self.assertTrue(self.at.dataframe)

        # Test detailed content
        self.at.selectbox[0].set_value(self.email_campaign.id).run(timeout=60)
        self.assertEqual(self.at.selectbox[0].value, self.email_campaign.id)
        self.assertTrue(self.at.text_area[0].value)

    def test_manual_search_worker_page(self):
        # Test Manual Search Worker Page
        self.at.sidebar.radio[0].set_value("🔍 Manual Search Worker").run(timeout=60)
        self.assertEqual(self.at.session_state["selected_page"], "manual_search_worker_page")

        # Test AI generation
        self.at.form_submit_button[0].click().run(timeout=60)
        self.at.text_area[0].input("Test Product Description").run(timeout=60)
        with patch('streamlit_app.openai_chat_completion', side_effect=[
            "Search Term 1, Search Term 2, Search Term 3, Search Term 4, Search Term 5",
            "Subject: Test Subject\nTest Email Body"
        ]):
            with patch('streamlit_app.EmailSettings') as MockEmailSettings:
                mock_settings = MagicMock()
                mock_settings.email = "test@example.com"
                mock_settings.provider = "AWS SES"
                MockEmailSettings.query.filter_by.return_value.first.return_value = mock_settings
                self.at.button[0].click().run(timeout=60)
                self.assertTrue(self.at.session_state['search_terms_updated'])

        # Test search terms input
        self.at.text_input[0].input("Test Search Term").run(timeout=60)
        self.assertEqual(self.at.text_input[0].value, "Test Search Term")

        # Test number of results slider
        self.at.slider[0].set_value(20).run(timeout=60)
        self.assertEqual(self.at.slider[0].value, 20)

        # Test checkboxes
        self.at.checkbox[0].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[0].value)
        self.at.checkbox[1].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[1].value)
        self.at.checkbox[2].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[2].value)
        self.at.checkbox[3].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[3].value)

        # Test language selection
        self.at.selectbox[0].set_value("English").run(timeout=60)
        self.assertEqual(self.at.selectbox[0].value, "English")

        # Test email sending functionality
        self.at.checkbox[4].check().run(timeout=60)
        self.assertTrue(self.at.checkbox[4].value)
        self.at.selectbox[1].set_value("Test Template").run(timeout=60)
        self.assertEqual(self.at.selectbox[1].value, "Test Template")
        self.at.selectbox[2].set_value("test@example.com").run(timeout=60)
        self.assertEqual(self.at.selectbox[2].value, "test@example.com")
        self.at.text_input[1].input("reply@example.com").run(timeout=60)
        self.assertEqual(self.at.text_input[1].value, "reply@example.com")

        # Test start search button
        with patch('streamlit_app.run_automated_search') as mock_run_automated_search:
            self.at.form_submit_button[1].click().run(timeout=60)
            self.assertTrue(self.at.session_state['search_running'])
            mock_run_automated_search.assert_called_once()

        # Test clear completed searches button
        self.at.button[1].click().run(timeout=60)

    @patch('smtplib.SMTP', new=MockSMTP)
    def test_send_email_smtp(self):
        with db_session() as session:
            # Create test email settings for SMTP
            email_settings = EmailSettings(
                name="Test SMTP Settings",
                email="test@example.com",
                provider="smtp",
                smtp_server="smtp.example.com",
                smtp_port=587,
                smtp_username="testuser",
                smtp_password="testpassword"
            )
            session.add(email_settings)
            session.commit()

            # Test sending email via SMTP
            response, tracking_id = send_email_ses(session, "test@example.com", "recipient@example.com", "Test Subject", "Test Body")
            self.assertIsNotNone(response)
            self.assertIsNotNone(tracking_id)

    @patch('boto3.Session', new=lambda *args, **kwargs: MockBoto3Client())
    def test_send_email_ses(self):
        with db_session() as session:
            # Create test email settings for SES
            email_settings = EmailSettings(
                name="Test SES Settings",
                email="test@example.com",
                provider="ses",
                aws_access_key_id="test_access_key",
                aws_secret_access_key="test_secret_key",
                aws_region="us-east-1"
            )
            session.add(email_settings)
            session.commit()

            # Test sending email via SES
            response, tracking_id = send_email_ses(session, "test@example.com", "recipient@example.com", "Test Subject", "Test Body")
            self.assertIsNotNone(response)
            self.assertIsNotNone(tracking_id)

    def test_save_email_campaign(self):
        with db_session() as session:
            # Test saving email campaign
            campaign = save_email_campaign(session, "test@example.com", self.email_template.id, "sent", datetime.utcnow(), "Test Subject", "test_message_id", "Test Body")
            self.assertIsNotNone(campaign)
            self.assertEqual(campaign.lead.email, "test@example.com")
            self.assertEqual(campaign.template.id, self.email_template.id)
            self.assertEqual(campaign.status, "sent")
            self.assertEqual(campaign.message_id, "test_message_id")

if __name__ == "__main__":
    unittest.main()