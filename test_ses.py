import os
import boto3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def test_ses_connection():
    session = Session()
    try:
        # Get SES settings from database
        result = session.execute(text("""
            SELECT email, aws_access_key_id, aws_secret_access_key, aws_region 
            FROM email_settings 
            WHERE provider = 'AWS SES' 
            LIMIT 1
        """)).fetchone()
        
        if not result:
            print("No AWS SES settings found in database")
            return
            
        email, access_key, secret_key, region = result
        
        # Create SES client
        ses_client = boto3.client(
            'ses',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Send test email
        response = ses_client.send_email(
            Source=email,
            Destination={
                'ToAddresses': [email]  # Sending to same email for testing
            },
            Message={
                'Subject': {
                    'Data': 'Test Email from Python SES'
                },
                'Body': {
                    'Html': {
                        'Data': '<h1>Test Email</h1><p>This is a test email sent using AWS SES.</p>'
                    }
                }
            }
        )
        
        print(f"Email sent successfully! Message ID: {response['MessageId']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    test_ses_connection() 