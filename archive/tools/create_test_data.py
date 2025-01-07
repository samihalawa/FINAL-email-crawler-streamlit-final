from models import Base, Project, Campaign
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql+psycopg2://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres")
SessionLocal = sessionmaker(bind=engine)

with SessionLocal() as session:
    project = Project(project_name="Test Project")
    session.add(project)
    session.flush()
    
    campaign = Campaign(
        project_id=project.id,
        campaign_name="Test Campaign",
        campaign_type="Email"
    )
    session.add(campaign)
    session.commit()
    
    print(f"Created test project (ID: {project.id}) and campaign (ID: {campaign.id})")
