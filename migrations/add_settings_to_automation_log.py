import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')

def migrate():
    """Add settings column to automation_logs table"""
    # Parse database URL
    from urllib.parse import urlparse
    result = urlparse(DATABASE_URL)
    db_params = {
        'dbname': result.path[1:],
        'user': result.username,
        'password': result.password,
        'host': result.hostname,
        'port': result.port
    }
    
    # Connect to database
    conn = psycopg2.connect(**db_params)
    conn.autocommit = True
    
    try:
        with conn.cursor() as cur:
            # Add settings column
            cur.execute("""
                ALTER TABLE automation_logs 
                ADD COLUMN IF NOT EXISTS settings JSONB;
            """)
            print("Migration completed successfully!")
    except Exception as e:
        print(f"Error during migration: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate() 