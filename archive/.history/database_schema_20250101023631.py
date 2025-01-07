import os
from datetime import datetime
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON, func, UUID, Float, Integer, String, JSONB
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Database configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
    raise ValueError("One or more required database environment variables are not set")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine and session
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

@contextmanager
def db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
class AIRequestLogs(Base):
    __tablename__ = 'ai_request_logs'
    id = Column(BigInteger, primary_key=True)
    function_name = Column(Text)
    prompt = Column(Text)
    response = Column(Text)
    model_used = Column(Text)
    created_at = Column(DateTime(timezone=True))
    lead_id = Column(BigInteger)
    email_campaign_id = Column(BigInteger)

class AIRequests(Base):
    __tablename__ = 'ai_requests'
    id = Column(BigInteger, primary_key=True)
    function_name = Column(Text)
    prompt = Column(Text)
    response = Column(Text)
    lead_id = Column(BigInteger)
    email_campaign_id = Column(BigInteger)
    model_used = Column(Text)
    created_at = Column(DateTime(timezone=True))
    logs = Column(JSON)

class AlembicVersion(Base):
    __tablename__ = 'alembic_version'
    version_num = Column(String, primary_key=True)

class AutomationErrors(Base):
    __tablename__ = 'automation_errors'
    id = Column(BigInteger, primary_key=True)
    task_type = Column(Text)
    error_type = Column(Text)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True))

class AutomationJobs(Base):
    __tablename__ = 'automation_jobs'
    id = Column(UUID, primary_key=True)
    status = Column(String, nullable=False)
    current_group = Column(String)
    current_position = Column(Integer)
    total_emails_sent = Column(Integer)
    group_emails_sent = Column(Integer)
    distribution_method = Column(String)
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))
    loop_interval = Column(Integer)
    max_emails_per_group = Column(Integer)
    loop_automation = Column(Boolean)

class AutomationLogs(Base):
    __tablename__ = 'automation_logs'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger)
    search_term_id = Column(BigInteger)
    leads_gathered = Column(BigInteger)
    emails_sent = Column(BigInteger)
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    status = Column(Text)
    logs = Column(JSON)

class AutomationRules(Base):
    __tablename__ = 'automation_rules'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    rule_type = Column(Text)
    condition = Column(Text)
    threshold = Column(Float)
    action = Column(Text)
    notification_email = Column(Text)
    is_active = Column(Boolean)
    created_at = Column(DateTime(timezone=True))

class AutomationSchedules(Base):
    __tablename__ = 'automation_schedules'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text)
    task_type = Column(Text)
    frequency = Column(Text)
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    time_of_day = Column(Text)
    created_at = Column(DateTime(timezone=True))

class AutomationSettings(Base):
    __tablename__ = 'automation_settings'
    id = Column(Integer, primary_key=True)
    status = Column(Text)
    distribution_method = Column(Text)
    updated_at = Column(DateTime(timezone=True))

class AutomationState(Base):
    __tablename__ = 'automation_state'
    id = Column(BigInteger, primary_key=True)
    current_group_id = Column(BigInteger)
    current_term_id = Column(BigInteger)
    processed_groups = Column(JSONB)
    group_metrics = Column(JSONB)
    term_metrics = Column(JSONB)
    emails_sent = Column(BigInteger)
    leads_found = Column(BigInteger)
    last_updated = Column(DateTime(timezone=True))

class AutomationStatus(Base):
    __tablename__ = 'automation_status'
    id = Column(BigInteger, primary_key=True)
    status = Column(Text)
    started_at = Column(DateTime(timezone=True))
    stopped_at = Column(DateTime(timezone=True))
    paused_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True))

class AutomationTasks(Base):
    __tablename__ = 'automation_tasks'
    id = Column(BigInteger, primary_key=True)
    task_type = Column(Text)
    status = Column(Text)
    progress = Column(Integer)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    eta = Column(DateTime(timezone=True))
    logs = Column(JSON)
    created_at = Column(DateTime(timezone=True))

class CampaignLeads(Base):
    __tablename__ = 'campaign_leads'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger)
    lead_id = Column(BigInteger)
    status = Column(Text)
    created_at = Column(DateTime(timezone=True))

class Campaigns(Base):
    __tablename__ = 'campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_name = Column(Text)
    campaign_type = Column(Text)
    project_id = Column(BigInteger)
    created_at = Column(DateTime(timezone=True))
    auto_send = Column(Boolean)
    loop_automation = Column(Boolean)
    ai_customization = Column(Boolean)
    max_emails_per_group = Column(BigInteger)
    loop_interval = Column(BigInteger)
    schedule_config = Column(JSON)
    ab_test_config = Column(JSON)
    sequence_config = Column(JSON)
    status = Column(Text)
    updated_at = Column(DateTime(timezone=True))
    progress = Column(Integer)
    total_tasks = Column(Integer)
    completed_tasks = Column(Integer)

class EmailCampaigns(Base):
    __tablename__ = 'email_campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger)
    lead_id = Column(BigInteger)
    template_id = Column(BigInteger)
    customized_subject = Column(Text)
    customized_content = Column(Text)
    original_subject = Column(Text)
    original_content = Column(Text)
    status = Column(Text)
    engagement_data = Column(JSON)
    message_id = Column(Text)
    tracking_id = Column(Text)
    sent_at = Column(DateTime(timezone=True))
    ai_customized = Column(Boolean)
    opened_at = Column(DateTime(timezone=True))
    clicked_at = Column(DateTime(timezone=True))
    open_count = Column(BigInteger)
    click_count = Column(BigInteger)

class EmailQuotas(Base):
    __tablename__ = 'email_quotas'
    id = Column(BigInteger, primary_key=True)
    email_settings_id = Column(BigInteger)
    daily_sent = Column(BigInteger)
    last_reset = Column(DateTime(timezone=True))

class EmailSettings(Base):
    __tablename__ = 'email_settings'
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    email = Column(Text, nullable=False)
    provider = Column(Text, nullable=False)
    smtp_server = Column(Text)
    smtp_port = Column(BigInteger)
    smtp_username = Column(Text)
    smtp_password = Column(Text)
    aws_access_key_id = Column(Text)
    aws_secret_access_key = Column(Text)
    aws_region = Column(Text)
    created_at = Column(DateTime(timezone=True))

class EmailTemplates(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger)
    template_name = Column(Text)
    subject = Column(Text)
    body_content = Column(Text)
    created_at = Column(DateTime(timezone=True))
    is_ai_customizable = Column(Boolean)
    language = Column(Text)

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    id = Column(BigInteger, primary_key=True)
    project_id = Column(BigInteger, nullable=False)
    kb_name = Column(Text)
    kb_bio = Column(Text)
    kb_values = Column(Text)
    contact_name = Column(Text)
    contact_role = Column(Text)
    contact_email = Column(Text)
    company_description = Column(Text)
    company_mission = Column(Text)
    company_target_market = Column(Text)
    company_other = Column(Text)
    product_name = Column(Text)
    product_description = Column(Text)
    product_target_customer = Column(Text)
    product_other = Column(Text)
    other_context = Column(Text)
    example_email = Column(Text)
    tone_of_voice = Column(Text)
    communication_style = Column(Text)
    response_templates = Column(JSON)
    keywords = Column(JSON)
    context_variables = Column(JSON)
    ai_customization_rules = Column(JSON)
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))

class LeadSources(Base):
    __tablename__ = 'lead_sources'
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger)
    search_term_id = Column(BigInteger)
    url = Column(Text)
    domain = Column(Text)
    page_title = Column(Text)
    meta_description = Column(Text)
    scrape_duration = Column(Text)
    meta_tags = Column(Text)
    phone_numbers = Column(Text)
    content = Column(Text)
    tags = Column(Text)
    http_status = Column(BigInteger)
    domain_effectiveness = Column(JSON)
    correlation_data = Column(JSON)
    created_at = Column(DateTime(timezone=True))

class Leads(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text)
    phone = Column(Text)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text)
    job_title = Column(Text)
    lead_score = Column(BigInteger)
    status = Column(Text)
    source_category = Column(Text)
    created_at = Column(DateTime(timezone=True))
    is_processed = Column(Boolean)

class OptimizedSearchTerms(Base):
    __tablename__ = 'optimized_search_terms'
    id = Column(BigInteger, primary_key=True)
    original_term_id = Column(BigInteger)
    term = Column(Text)
    created_at = Column(DateTime(timezone=True))

class Projects(Base):
    __tablename__ = 'projects'
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text)
    created_at = Column(DateTime(timezone=True))

class SearchGroups(Base):
    __tablename__ = 'search_groups'
    id = Column(UUID, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    emails_sent = Column(Integer)
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))

class SearchProcesses(Base):
    __tablename__ = 'search_processes'
    id = Column(BigInteger, primary_key=True)
    search_terms = Column(JSON)

# Create or update tables
def init_db():
    Base.metadata.create_all(engine)