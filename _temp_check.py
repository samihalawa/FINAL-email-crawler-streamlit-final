from app import *
from datetime import datetime
import json

def check_table(session, table, limit=10):
    print(f"\n=== {table.__name__} ===")
    rows = session.query(table).limit(limit).all()
    if not rows:
        print("No records found")
        return
    
    for row in rows:
        data = {}
        for column in table.__table__.columns:
            value = getattr(row, column.name)
            if isinstance(value, (datetime, bytes)):
                value = str(value)
            data[column.name] = value
        print(json.dumps(data, indent=2))
        print("-" * 40)

def main():
    with db_session() as session:
        tables = [
            Settings,
            EmailSettings,
            SearchTerm,
            Lead,
            EmailCampaign,
            SearchTermGroup,
            EmailTemplate,
            Project,
            Campaign,
            KnowledgeBase,
            CampaignLead,
            LeadSource,
            OptimizedSearchTerm,
            SearchTermEffectiveness,
            AIRequestLog,
            AutomationLog
        ]
        
        for table in tables:
            check_table(session, table)

if __name__ == "__main__":
    main() 