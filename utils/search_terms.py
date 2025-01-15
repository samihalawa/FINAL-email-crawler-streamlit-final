from models import SearchTerm
from datetime import datetime

def get_or_create_search_term(session, term, language='ES'):
    """Get existing search term or create new one"""
    existing = session.query(SearchTerm).filter_by(term=term).first()
    if existing:
        return existing
        
    new_term = SearchTerm(
        term=term,
        language=language,
        created_at=datetime.utcnow()
    )
    session.add(new_term)
    session.commit()
    return new_term 