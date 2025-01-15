from sqlalchemy import inspect, text
from models import Base
from utils.db import engine, get_db
import logging

logger = logging.getLogger(__name__)

def get_current_columns(table_name):
    """Get current columns in table"""
    inspector = inspect(engine)
    return {col['name']: col for col in inspector.get_columns(table_name)}

def get_model_columns(model):
    """Get columns defined in model"""
    return {c.name: c for c in model.__table__.columns}

def safe_migrate():
    """Run safe migrations - only adds missing columns"""
    try:
        with get_db() as session:
            # Get all models from Base
            models = Base._decl_class_registry.values()
            models = [m for m in models if hasattr(m, '__tablename__')]
            
            for Model in models:
                table_name = Model.__tablename__
                current_cols = get_current_columns(table_name)
                model_cols = get_model_columns(Model)
                
                # Find missing columns
                for col_name, col in model_cols.items():
                    if col_name not in current_cols:
                        # Generate safe ALTER TABLE statement
                        col_type = col.type.compile(engine.dialect)
                        nullable = "" if col.nullable else "NOT NULL"
                        default = f"DEFAULT {col.default.arg}" if col.default else ""
                        
                        sql = f"""
                        ALTER TABLE {table_name} 
                        ADD COLUMN IF NOT EXISTS {col_name} {col_type} {nullable} {default};
                        """
                        
                        logger.info(f"Adding column {col_name} to {table_name}")
                        session.execute(text(sql))
                
                session.commit()
            
            return True, "Migrations completed successfully"
            
    except Exception as e:
        logger.error(f"Migration error: {str(e)}")
        return False, str(e) 