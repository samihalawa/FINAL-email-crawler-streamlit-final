import os
import psycopg2
from datetime import datetime
import json
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
import pandas as pd

# Setup logging
logging.basicConfig(
    filename=f'db_recovery_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Database connection parameters from original app
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

def connect_to_db():
    """Establish database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        logging.info("Successfully connected to database")
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        return None

def analyze_table_structure():
    """Analyze and log current table structure"""
    try:
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        inspector = inspect(engine)
        
        structure = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            structure[table_name] = {
                "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns],
                "primary_key": inspector.get_primary_keys(table_name),
                "foreign_keys": inspector.get_foreign_keys(table_name)
            }
        
        with open(f'db_structure_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(structure, f, indent=4)
        
        logging.info("Table structure analysis completed")
        return structure
    except Exception as e:
        logging.error(f"Structure analysis failed: {str(e)}")
        return None

def export_remaining_data():
    """Export any remaining data from all tables"""
    try:
        conn = connect_to_db()
        if not conn:
            return False

        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()

        # Create exports directory
        os.makedirs('db_exports', exist_ok=True)
        
        # Export each table
        for table in tables:
            table_name = table[0]
            try:
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql_query(query, conn)
                df.to_csv(f'db_exports/{table_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
                logging.info(f"Successfully exported table: {table_name}")
            except Exception as e:
                logging.error(f"Failed to export table {table_name}: {str(e)}")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Data export failed: {str(e)}")
        return False

def check_table_relationships():
    """Analyze and verify table relationships"""
    try:
        conn = connect_to_db()
        if not conn:
            return False

        cursor = conn.cursor()
        
        # Query to get foreign key relationships
        cursor.execute("""
            SELECT
                tc.table_schema, 
                tc.constraint_name, 
                tc.table_name, 
                kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name 
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
            WHERE constraint_type = 'FOREIGN KEY';
        """)
        
        relationships = cursor.fetchall()
        
        with open(f'table_relationships_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump([{
                "schema": rel[0],
                "constraint": rel[1],
                "table": rel[2],
                "column": rel[3],
                "foreign_table": rel[4],
                "foreign_column": rel[5]
            } for rel in relationships], f, indent=4)
        
        cursor.close()
        conn.close()
        logging.info("Relationship analysis completed")
        return True
    except Exception as e:
        logging.error(f"Relationship analysis failed: {str(e)}")
        return False

def main():
    """Main recovery process"""
    print("Starting database recovery and analysis process...")
    
    # 1. Analyze current table structure
    print("Analyzing table structure...")
    structure = analyze_table_structure()
    if structure:
        print("✅ Table structure analysis completed")
    else:
        print("❌ Table structure analysis failed")

    # 2. Export any remaining data
    print("Exporting remaining data...")
    if export_remaining_data():
        print("✅ Data export completed")
    else:
        print("❌ Data export failed")

    # 3. Check table relationships
    print("Analyzing table relationships...")
    if check_table_relationships():
        print("✅ Relationship analysis completed")
    else:
        print("❌ Relationship analysis failed")

    print("\nRecovery process completed. Please check the generated log files and exports directory.")
    print("Generated files:")
    print("- Log file: db_recovery_[timestamp].log")
    print("- Structure analysis: db_structure_[timestamp].json")
    print("- Table relationships: table_relationships_[timestamp].json")
    print("- Exported data: db_exports/[table_name]_[timestamp].csv")

if __name__ == "__main__":
    main()