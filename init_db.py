from models import Base, Project, Campaign, Lead, EmailTemplate, EmailCampaign, KnowledgeBase, CampaignLead, LeadSource, SearchTerm
from sqlalchemy import create_engine

DB_HOST = "aws-0-eu-central-1.pooler.supabase.com"
DB_NAME = "postgres"
DB_USER = "postgres.whwiyccyyfltobvqxiib"
DB_PASSWORD = "SamiHalawa1996"
DB_PORT = "6543"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)

# Create all tables
Base.metadata.create_all(engine)

print("Database tables created successfully!")
