from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg2://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres")

with engine.connect() as conn:
    tables = ["projects", "campaigns", "email_templates", "search_terms", "leads"]
    for table in tables:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        count = result.scalar()
        print(f"Number of {table}: {count}")
