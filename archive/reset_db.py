from models import Base
from sqlalchemy import create_engine, text

DB_HOST = "aws-0-eu-central-1.pooler.supabase.com"
DB_NAME = "postgres"
DB_USER = "postgres.whwiyccyyfltobvqxiib"
DB_PASSWORD = "SamiHalawa1996"
DB_PORT = "6543"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)

# Drop all tables
Base.metadata.drop_all(engine)

# Create all tables
Base.metadata.create_all(engine)

print("Database tables dropped and recreated successfully!")
