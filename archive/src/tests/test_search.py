from RELEASE_streamlit_app import manual_search
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql+psycopg2://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres")
SessionLocal = sessionmaker(bind=engine)

with SessionLocal() as session:
    results = manual_search(
        session=session,
        terms=["software development company"],
        num_results=5,
        ignore_previously_fetched=True,
        optimize_english=True,
        optimize_spanish=False,
        shuffle_keywords_option=False,
        language="EN",
        enable_email_sending=False,
        log_container=None
    )
    
    if results:
        print(f"Found {results.get(\"total_leads\", 0)} leads")
        for result in results.get("results", []):
            print(f"Lead: {result.get(\"Email\")} from {result.get(\"URL\")}")
