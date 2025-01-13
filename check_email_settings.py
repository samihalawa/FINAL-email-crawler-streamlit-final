import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit_app import EmailSettings, Base

# Database configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

try:
    # Get all email settings
    settings = session.query(EmailSettings).all()
    
    print("\nCurrent Email Settings:")
    print("-" * 50)
    
    for setting in settings:
        print(f"\nSetting Name: {setting.name}")
        print(f"Email: {setting.email}")
        print(f"Provider: {setting.provider}")
        print(f"AWS Region: {setting.aws_region}")
        print(f"AWS Access Key ID: {'*' * len(setting.aws_access_key_id) if setting.aws_access_key_id else 'Not set'}")
        print(f"AWS Secret Key: {'*' * 20 if setting.aws_secret_access_key else 'Not set'}")
        print("-" * 50)

except Exception as e:
    print(f"Error: {str(e)}")
finally:
    session.close() 