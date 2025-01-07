from RELEASE_streamlit_app import db_session, Project, Campaign, SearchTerm, Lead, EmailTemplate, KnowledgeBase
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time

def simulate_automation():
    print("Starting automation simulation...")
    
    with db_session() as session:
        # Get active project
        project = session.query(Project).first()
        if not project:
            print("No project found")
            return
            
        print(f"Using project: {project.project_name}")
        
        # Get knowledge base
        kb = session.query(KnowledgeBase).filter_by(project_id=project.id).first()
        if not kb:
            print("No knowledge base found")
            return
            
        print(f"Found knowledge base: {kb.kb_name}")
        
        # Get search terms
        terms = session.query(SearchTerm).filter_by(project_id=project.id).all()
        print(f"Found {len(terms)} search terms")
        
        # Get email template
        template = session.query(EmailTemplate).filter_by(project_id=project.id).first()
        if not template:
            print("No email template found")
            return
            
        print(f"Using template: {template.template_name}")
        
        # Print current state
        leads = session.query(Lead).all()
        print(f"Current leads in database: {len(leads)}")
        
        print("\nAutomation simulation complete")

if __name__ == "__main__":
    simulate_automation() 