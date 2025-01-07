from sqlalchemy import create_engine, text
from models import Base

engine = create_engine("postgresql+psycopg2://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres")

# Create tables and columns if they don't exist
Base.metadata.create_all(engine, checkfirst=True)

print("Database tables and columns created successfully!")
