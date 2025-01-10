from models import Base
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    DB_HOST = os.getenv('SUPABASE_DB_HOST')
    DB_NAME = os.getenv('SUPABASE_DB_NAME')
    DB_USER = os.getenv('SUPABASE_DB_USER')
    DB_PASSWORD = os.getenv('SUPABASE_DB_PASSWORD')
    DB_PORT = os.getenv('SUPABASE_DB_PORT')
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)

# Create tables and columns if they don't exist
Base.metadata.create_all(engine, checkfirst=True)

print("Database tables and columns created successfully!")
