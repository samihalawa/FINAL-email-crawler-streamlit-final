from sqlalchemy import create_engine, Column, BigInteger, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text, default="Default Project")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaigns = relationship("Campaign", back_populates="project", cascade="all, delete-orphan")

class Campaign(Base):
    __tablename__ = 'campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_name = Column(Text, default="Default Campaign")
    project_id = Column(BigInteger, ForeignKey('projects.id', ondelete='CASCADE'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    auto_send = Column(Boolean, default=False)
    project = relationship("Project", back_populates="campaigns")
    search_terms = relationship("SearchTerm", back_populates="campaign")

class SearchTerm(Base):
    __tablename__ = 'search_terms'
    id = Column(BigInteger, primary_key=True)
    term = Column(Text, nullable=False)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id', ondelete='CASCADE'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    campaign = relationship("Campaign", back_populates="search_terms")
    lead_sources = relationship("LeadSource", back_populates="search_term")

class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    company = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_contacted = Column(DateTime(timezone=True))
    status = Column(Text, default='New')
    lead_sources = relationship("LeadSource", back_populates="lead")

class LeadSource(Base):
    __tablename__ = 'lead_sources'
    id = Column(BigInteger, primary_key=True)
    lead_id = Column(BigInteger, ForeignKey('leads.id'))
    search_term_id = Column(BigInteger, ForeignKey('search_terms.id'))
    url = Column(Text)
    domain = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    lead = relationship("Lead", back_populates="lead_sources")
    search_term = relationship("SearchTerm", back_populates="lead_sources")

class SearchJob(Base):
    __tablename__ = 'search_jobs'
    id = Column(BigInteger, primary_key=True)
    status = Column(Text)  # running, completed, failed
    terms = Column(JSON)
    results = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    error = Column(Text)
