from models import Base, Project, Campaign
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_HOST = "aws-0-eu-central-1.pooler.supabase.com"
DB_NAME = "postgres"
DB_USER = "postgres.whwiyccyyfltobvqxiib"
DB_PASSWORD = "SamiHalawa1996"
DB_PORT = "6543"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

with SessionLocal() as session:
    # Create test project
    project = Project(project_name="Test Project")
    session.add(project)
    session.flush()
    
    # Create test campaign
    campaign = Campaign(
        project_id=project.id,
        campaign_name="Test Campaign",
        campaign_type="Email",
        auto_send=False,
        loop_automation=False,
        ai_customization=False,
        max_emails_per_group=40,
        loop_interval=60
    )
    session.add(campaign)
    session.commit()
    
    print(f"Created test project (ID: {project.id}) and campaign (ID: {campaign.id})")
