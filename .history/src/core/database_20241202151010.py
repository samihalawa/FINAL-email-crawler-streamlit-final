from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, JSON, Text, Float
from datetime import datetime
from typing import AsyncGenerator
from .config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True
)

# Create async session factory
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))

class Campaign(Base):
    __tablename__ = "campaigns"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    project_id = Column(Integer, ForeignKey("projects.id"))
    status = Column(String)
    settings = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class SearchTerm(Base):
    __tablename__ = "search_terms"
    
    id = Column(Integer, primary_key=True)
    term = Column(String)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    group_id = Column(Integer, ForeignKey("search_term_groups.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class SearchTermGroup(Base):
    __tablename__ = "search_term_groups"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(Text, nullable=True)

class Lead(Base):
    __tablename__ = "leads"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    company = Column(String, nullable=True)
    position = Column(String, nullable=True)
    source_url = Column(String)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_contacted = Column(DateTime, nullable=True)

class EmailTemplate(Base):
    __tablename__ = "email_templates"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    subject = Column(String)
    body_content = Column(Text)
    is_ai_customizable = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))

class EmailCampaign(Base):
    __tablename__ = "email_campaigns"
    
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))
    template_id = Column(Integer, ForeignKey("email_templates.id"))
    status = Column(String)
    sent_at = Column(DateTime, nullable=True)
    opened_at = Column(DateTime, nullable=True)
    replied_at = Column(DateTime, nullable=True)

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    content = Column(JSON)
    last_updated = Column(DateTime, default=datetime.utcnow)

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