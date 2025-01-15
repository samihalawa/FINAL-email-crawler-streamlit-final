from models import Campaign
from datetime import datetime

def get_active_campaign(session, project_id):
    """Get active campaign for project"""
    return session.query(Campaign).filter_by(
        project_id=project_id,
        is_active=True
    ).first()

def create_campaign(session, name, project_id, campaign_type="Email"):
    """Create new campaign"""
    campaign = Campaign(
        campaign_name=name,
        project_id=project_id,
        campaign_type=campaign_type,
        created_at=datetime.utcnow()
    )
    session.add(campaign)
    session.commit()
    return campaign 