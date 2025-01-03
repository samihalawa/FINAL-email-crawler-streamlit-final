from models import EmailTemplate
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql+psycopg2://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres")
SessionLocal = sessionmaker(bind=engine)

with SessionLocal() as session:
    template = EmailTemplate(
        campaign_id=1,
        template_name="Test Template",
        subject="Test Subject",
        body_content="Hello, this is a test email.",
        is_ai_customizable=True
    )
    session.add(template)
    session.commit()
    
    print(f"Created test email template (ID: {template.id})")
