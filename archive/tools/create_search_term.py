from models import SearchTerm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql+psycopg2://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres")
SessionLocal = sessionmaker(bind=engine)

with SessionLocal() as session:
    search_term = SearchTerm(
        campaign_id=1,
        term="software development company",
        category="Technology",
        language="EN"
    )
    session.add(search_term)
    session.commit()
    
    print(f"Created test search term (ID: {search_term.id})")
