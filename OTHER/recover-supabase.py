import os
import pandas as pd
from datetime import datetime
import logging
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    filename=f'db_recovery_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# PostgreSQL connection parameters
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

# SQLite database path
SQLITE_DB_PATH = "autoclient.db"

def connect_to_postgres():
    """Connect to the PostgreSQL database using SQLAlchemy."""
    try:
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(DATABASE_URL)
        logging.info("Connected to PostgreSQL database")
        return engine
    except SQLAlchemyError as e:
        logging.error(f"Failed to connect to PostgreSQL: {e}")
        return None

def export_postgres_data():
    """Export data from PostgreSQL database using SQLAlchemy."""
    engine = connect_to_postgres()
    if not engine:
        return

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        os.makedirs('postgres_exports', exist_ok=True)

        for table_name in tables:
            df = pd.read_sql_table(table_name, engine)
            df.to_csv(f'postgres_exports/{table_name}.csv', index=False)
            logging.info(f"Exported {table_name} from PostgreSQL")
    except SQLAlchemyError as e:
        logging.error(f"Error exporting PostgreSQL data: {e}")

def export_sqlite_data():
    """Export data from SQLite database using SQLAlchemy."""
    if not os.path.exists(SQLITE_DB_PATH):
        logging.info("SQLite database not found")
        return

    try:
        sqlite_engine = create_engine(f'sqlite:///{SQLITE_DB_PATH}')
        inspector = inspect(sqlite_engine)
        tables = inspector.get_table_names()

        os.makedirs('sqlite_exports', exist_ok=True)

        for table_name in tables:
            df = pd.read_sql_table(table_name, sqlite_engine)
            df.to_csv(f'sqlite_exports/{table_name}.csv', index=False)
            logging.info(f"Exported {table_name} from SQLite")
    except SQLAlchemyError as e:
        logging.error(f"Error exporting SQLite data: {e}")

def main():
    """Main function to perform data recovery."""
    logging.info("Starting data recovery process")
    export_postgres_data()
    export_sqlite_data()
    logging.info("Data recovery process completed")

if __name__ == "__main__":
    main()