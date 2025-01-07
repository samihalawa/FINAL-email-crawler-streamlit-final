from RELEASE_streamlit_app import manual_search
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres")
SessionLocal = sessionmaker(bind=engine)

with SessionLocal() as session:
    results = manual_search(session, ["software company"], 5, True, False, False, False, "EN", False, None)
    print("Search completed:", results)
