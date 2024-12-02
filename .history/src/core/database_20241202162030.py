from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, JSON, Text, Float, Enum, Index
from datetime import datetime
import enum
from typing import AsyncGenerator, List
from .config import settings

# Create async engine with optimized settings
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_size=20,
    max_overflow=30,
    pool_timeout=60,
    pool_pre_ping=True
)

# Create async session factory
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

class TaskStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class LeadStatus(enum.Enum):
    NEW = "new"
    QUALIFIED = "qualified"
    CONTACTED = "contacted"
    RESPONDED = "responded"
    CONVERTED = "converted"
    REJECTED = "rejected"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    api_key = Column(String, unique=True, nullable=True)
    rate_limit = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    projects = relationship("Project", back_populates="user")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(Text)
    settings = Column(JSON, default={
        "automation": {
            "enabled": False,
            "max_concurrent_tasks": 5,
            "search_delay": 2,
            "email_delay": 5,
            "ai_optimization": True,
            "auto_pause_threshold": 0.1
        },
        "email": {
            "daily_limit": 100,
            "retry_attempts": 3,
            "min_delay": 30,
            "max_delay": 300
        },
        "search": {
            "max_results_per_term": 50,
            "excluded_domains": [],
            "priority_domains": []
        }
    })
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    user = relationship("User", back_populates="projects")
    campaigns = relationship("Campaign", back_populates="project")
    knowledge_base = relationship("KnowledgeBase", back_populates="project", uselist=False)

class Campaign(Base):
    __tablename__ = "campaigns"
    __table_args__ = (
        Index('idx_campaign_status', 'status'),
        Index('idx_campaign_project', 'project_id')
    )
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    project_id = Column(Integer, ForeignKey("projects.id"))
    status = Column(String)
    settings = Column(JSON)
    stats = Column(JSON, default={
        "total_leads": 0,
        "qualified_leads": 0,
        "emails_sent": 0,
        "emails_opened": 0,
        "emails_replied": 0,
        "conversion_rate": 0.0,
        "bounce_rate": 0.0
    })
    created_at = Column(DateTime, default=datetime.utcnow)
    last_run = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=True)
    
    project = relationship("Project", back_populates="campaigns")
    search_terms = relationship("SearchTerm", back_populates="campaign")
    leads = relationship("Lead", back_populates="campaign")
    email_templates = relationship("EmailTemplate", back_populates="campaign")

class SearchTerm(Base):
    __tablename__ = "search_terms"
    __table_args__ = (
        Index('idx_search_term_campaign', 'campaign_id'),
        Index('idx_search_term_group', 'group_id')
    )
    
    id = Column(Integer, primary_key=True)
    term = Column(String)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    group_id = Column(Integer, ForeignKey("search_term_groups.id"), nullable=True)
    priority = Column(Integer, default=0)
    last_searched = Column(DateTime, nullable=True)
    results_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    campaign = relationship("Campaign", back_populates="search_terms")
    group = relationship("SearchTermGroup", back_populates="terms")

class SearchTermGroup(Base):
    __tablename__ = "search_term_groups"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(Text, nullable=True)
    priority = Column(Integer, default=0)
    
    terms = relationship("SearchTerm", back_populates="group")

class Lead(Base):
    __tablename__ = "leads"
    __table_args__ = (
        Index('idx_lead_status', 'status'),
        Index('idx_lead_campaign', 'campaign_id'),
        Index('idx_lead_email', 'email', unique=True)
    )
    
    id = Column(Integer, primary_key=True)
    email = Column(String, index=True)
    name = Column(String, nullable=True)
    company = Column(String, nullable=True)
    position = Column(String, nullable=True)
    source_url = Column(String)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    status = Column(Enum(LeadStatus), default=LeadStatus.NEW)
    quality_score = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    last_contacted = Column(DateTime, nullable=True)
    next_contact = Column(DateTime, nullable=True)
    
    campaign = relationship("Campaign", back_populates="leads")
    email_campaigns = relationship("EmailCampaign", back_populates="lead")

class EmailTemplate(Base):
    __tablename__ = "email_templates"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    subject = Column(String)
    body_content = Column(Text)
    is_ai_customizable = Column(Boolean, default=True)
    variables = Column(JSON, default=[])
    performance_stats = Column(JSON, default={
        "sent": 0,
        "opened": 0,
        "replied": 0,
        "conversion_rate": 0.0
    })
    created_at = Column(DateTime, default=datetime.utcnow)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    
    campaign = relationship("Campaign", back_populates="email_templates")
    email_campaigns = relationship("EmailCampaign", back_populates="template")

class EmailCampaign(Base):
    __tablename__ = "email_campaigns"
    __table_args__ = (
        Index('idx_email_campaign_status', 'status'),
        Index('idx_email_campaign_lead', 'lead_id'),
        Index('idx_email_campaign_template', 'template_id')
    )
    
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))
    template_id = Column(Integer, ForeignKey("email_templates.id"))
    status = Column(String)
    customized_subject = Column(Text, nullable=True)
    customized_content = Column(Text, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    opened_at = Column(DateTime, nullable=True)
    replied_at = Column(DateTime, nullable=True)
    bounce_info = Column(JSON, nullable=True)
    tracking_data = Column(JSON, default={})
    
    lead = relationship("Lead", back_populates="email_campaigns")
    template = relationship("EmailTemplate", back_populates="email_campaigns")

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    content = Column(JSON)
    embeddings = Column(JSON, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    version = Column(Integer, default=1)
    
    project = relationship("Project", back_populates="knowledge_base")

class AutomationTask(Base):
    __tablename__ = "automation_tasks"
    __table_args__ = (
        Index('idx_task_status', 'status'),
        Index('idx_task_project', 'project_id')
    )
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    task_type = Column(String)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    priority = Column(Integer, default=0)
    params = Column(JSON, default={})
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    next_retry = Column(DateTime, nullable=True)
    retry_count = Column(Integer, default=0)

# Database dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Initialize database
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Cleanup
async def cleanup_db():
    await engine.dispose() 