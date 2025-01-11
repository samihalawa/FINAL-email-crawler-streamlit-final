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
    # Get existing settings to copy AWS credentials
    existing_settings = session.query(EmailSettings).first()
    
    if existing_settings:
        # Create new settings with same AWS credentials but different name
        new_settings = EmailSettings(
            name="Kitty Chino",
            email=existing_settings.email,  # Keep same email
            provider="AWS SES",
            aws_access_key_id=existing_settings.aws_access_key_id,
            aws_secret_access_key=existing_settings.aws_secret_access_key,
            aws_region=existing_settings.aws_region
        )
        
        session.add(new_settings)
        session.commit()
        print("Successfully added new email settings for Kitty Chino")
    else:
        print("No existing email settings found to copy from")

except Exception as e:
    print(f"Error: {str(e)}")
    session.rollback()
finally:
    session.close() 