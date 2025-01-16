from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

class ProjectBase(BaseModel):
    project_name: str

class ProjectCreate(ProjectBase):
    pass

class ProjectResponse(ProjectBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class CampaignBase(BaseModel):
    campaign_name: str
    campaign_type: str = "Email"
    auto_send: bool = False
    loop_automation: bool = False
    ai_customization: bool = False
    max_emails_per_group: int = 40
    loop_interval: int = 60

class CampaignCreate(CampaignBase):
    project_id: int

class CampaignResponse(CampaignBase):
    id: int
    created_at: datetime
    project_id: int
    
    class Config:
        from_attributes = True

class LeadBase(BaseModel):
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    phone: Optional[str] = None

class LeadCreate(LeadBase):
    pass

class LeadResponse(LeadBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class EmailTemplateBase(BaseModel):
    template_name: str
    subject: str
    body_content: str
    is_ai_customizable: bool = False
    language: str = 'ES'

class EmailTemplateCreate(EmailTemplateBase):
    campaign_id: int

class EmailTemplateResponse(EmailTemplateBase):
    id: int
    created_at: datetime
    campaign_id: int
    
    class Config:
        from_attributes = True

class SearchRequest(BaseModel):
    search_terms: List[str]
    num_results: int = 10
    ignore_previously_fetched: bool = True
    optimize_english: bool = False
    optimize_spanish: bool = False
    shuffle_keywords: bool = False
    language: str = 'ES'
    enable_email_sending: bool = False
    from_email: Optional[str] = None
    reply_to: Optional[str] = None
    email_template_id: Optional[int] = None

class SearchResponse(BaseModel):
    total_leads: int
    results: List[Dict[str, Any]]

class EmailSettingsBase(BaseModel):
    name: str
    email: EmailStr
    provider: str
    smtp_server: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None

class EmailSettingsCreate(EmailSettingsBase):
    pass

class EmailSettingsResponse(EmailSettingsBase):
    id: int
    
    class Config:
        from_attributes = True

class AutomationStatus(BaseModel):
    status: str
    leads_gathered: int
    emails_sent: int
    latest_logs: List[str] 