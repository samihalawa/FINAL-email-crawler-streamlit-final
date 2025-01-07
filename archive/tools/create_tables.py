from models import Base
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres")

# Create all tables
Base.metadata.create_all(engine)

print("Database tables created successfully!")
