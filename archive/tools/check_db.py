from RELEASE_streamlit_app import *

def check_db_state():
    with db_session() as session:
        projects = session.query(Project).all()
        campaigns = session.query(Campaign).all()
        templates = session.query(EmailTemplate).all()
        settings = session.query(EmailSettings).all()
        
        print("\nDatabase State:")
        print("-" * 50)
        print(f"Projects ({len(projects)}):")
        for p in projects:
            print(f"  - {p.id}: {p.project_name}")
            
        print(f"\nCampaigns ({len(campaigns)}):")
        for c in campaigns:
            print(f"  - {c.id}: {c.campaign_name}")
            
        print(f"\nEmail Templates ({len(templates)}):")
        for t in templates:
            print(f"  - {t.id}: {t.template_name}")
            
        print(f"\nEmail Settings ({len(settings)}):")
        for s in settings:
            print(f"  - {s.id}: {s.name} ({s.email})")

if __name__ == "__main__":
    check_db_state() 