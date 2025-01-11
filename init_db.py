from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit_app import Base
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')

def init_db():
    """Initialize database with all tables"""
    engine = create_engine(DATABASE_URL)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    Session = sessionmaker(bind=engine)
    
    # Create a session
    session = Session()
    
    try:
        # Add default project if none exists
        from streamlit_app import Project
        if not session.query(Project).first():
            default_project = Project(project_name="Default Project")
            session.add(default_project)
            session.commit()
        
        print("Database initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    init_db() 