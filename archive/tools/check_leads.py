import os
from RELEASE_streamlit_app import db_session, KnowledgeBase

# Set database URL
os.environ['DATABASE_URL'] = 'postgresql://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres'

with db_session() as session:
    # Create knowledge base entry
    kb = KnowledgeBase(
        project_id=1,
        kb_name="Test Knowledge Base",
        kb_bio="Test bio",
        contact_email="test@example.com",
        company_description="Test company description",
        company_mission="Test mission",
        company_target_market="Test market",
        tone_of_voice="Professional",
        communication_style="Direct"
    )
    session.add(kb)
    session.commit()
    
    print("Knowledge base created successfully")
    print(f"ID: {kb.id}")
    print(f"Project ID: {kb.project_id}")
    print(f"Name: {kb.kb_name}")
    print(f"Contact Email: {kb.contact_email}")
    print(f"Created: {kb.created_at}") 