from sqlalchemy import create_engine, Column, BigInteger, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from sqlalchemy.sql import func

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    id = Column(BigInteger, primary_key=True)
    project_name = Column(Text, default="Default Project")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    campaigns = relationship("Campaign", back_populates="project")

class Campaign(Base):
    __tablename__ = 'campaigns'
    id = Column(BigInteger, primary_key=True)
    campaign_name = Column(Text, default="Default Campaign")
    campaign_type = Column(Text, default="Email")
    project_id = Column(BigInteger, ForeignKey('projects.id'), default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    auto_send = Column(Boolean, default=False)
    loop_automation = Column(Boolean, default=False)
    ai_customization = Column(Boolean, default=False)
    max_emails_per_group = Column(BigInteger, default=40)
    loop_interval = Column(BigInteger, default=60)
    project = relationship("Project", back_populates="campaigns")
    email_templates = relationship("EmailTemplate", back_populates="campaign")

class Lead(Base):
    __tablename__ = 'leads'
    id = Column(BigInteger, primary_key=True)
    email = Column(Text, unique=True)
    phone = Column(Text)
    first_name = Column(Text)
    last_name = Column(Text)
    company = Column(Text)
    job_title = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class EmailTemplate(Base):
    __tablename__ = 'email_templates'
    id = Column(BigInteger, primary_key=True)
    campaign_id = Column(BigInteger, ForeignKey('campaigns.id'))
    template_name = Column(Text)
    subject = Column(Text)
    body_content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_ai_customizable = Column(Boolean, default=False)
    language = Column(Text, default='ES')
    campaign = relationship("Campaign", back_populates="email_templates")

engine = create_engine('postgresql+psycopg2://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres')
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

with SessionLocal() as session:
    # Create project if it doesn't exist
    project = session.query(Project).filter_by(id=1).first()
    if not project:
        project = Project(id=1, project_name='Default Project')
        session.add(project)
        session.commit()
        print(f'Created project with ID: {project.id}')

    # Create campaign if it doesn't exist
    campaign = session.query(Campaign).filter_by(id=1).first()
    if not campaign:
        campaign = Campaign(
            id=1,
            campaign_name='Default Campaign',
            campaign_type='Email',
            project_id=project.id
        )
        session.add(campaign)
        session.commit()
        print(f'Created campaign with ID: {campaign.id}')

    # Create test leads
    test_leads = [
        {
            'email': 'info@empresa1.es',
            'first_name': 'Juan',
            'last_name': 'Pérez',
            'company': 'Empresa1',
            'job_title': 'Director Técnico'
        },
        {
            'email': 'contacto@softwaredev.es',
            'first_name': 'María',
            'last_name': 'García',
            'company': 'SoftwareDev',
            'job_title': 'CTO'
        },
        {
            'email': 'info@techcompany.es',
            'first_name': 'Carlos',
            'last_name': 'Rodríguez',
            'company': 'TechCompany',
            'job_title': 'Lead Developer'
        },
        {
            'email': 'contacto@consultoria.es',
            'first_name': 'Ana',
            'last_name': 'Martínez',
            'company': 'Consultoría IT',
            'job_title': 'Project Manager'
        },
        {
            'email': 'info@desarrollo.es',
            'first_name': 'Pablo',
            'last_name': 'Sánchez',
            'company': 'Desarrollo Software',
            'job_title': 'Software Architect'
        }
    ]

    for lead_data in test_leads:
        lead = session.query(Lead).filter_by(email=lead_data['email']).first()
        if not lead:
            lead = Lead(**lead_data)
            session.add(lead)
            print(f'Created lead: {lead_data["email"]}')
    session.commit()

    # Create email template
    template = EmailTemplate(
        campaign_id=campaign.id,
        template_name='Hello Template',
        subject='Hello from our team',
        body_content='Hi {first_name},\n\nI noticed your work at {company} and wanted to reach out.\n\nBest regards',
        is_ai_customizable=False,
        language='EN'
    )
    session.add(template)
    session.commit()
    print(f'Created template with ID: {template.id}') 