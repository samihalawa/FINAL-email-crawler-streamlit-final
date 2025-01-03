from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg2://postgres.whwiyccyyfltobvqxiib:SamiHalawa1996@aws-0-eu-central-1.pooler.supabase.com:6543/postgres")

with engine.connect() as conn:
    conn.execute(text("DROP SCHEMA public CASCADE"))
    conn.execute(text("CREATE SCHEMA public"))
    conn.commit()

print("Database reset successfully!")
