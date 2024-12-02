import logging
import smtplib
import boto3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import settings
from core.database import EmailTemplate, EmailCampaign, Lead

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_settings = {
            'server': settings.SMTP_SERVER,
            'port': settings.SMTP_PORT,
            'username': settings.SMTP_USERNAME,
            'password': settings.SMTP_PASSWORD
        }
        self.aws_settings = {
            'aws_access_key_id': settings.AWS_ACCESS_KEY_ID,
            'aws_secret_access_key': settings.AWS_SECRET_ACCESS_KEY,
            'region': settings.AWS_REGION
        }
        self.ses_client = None
        
    def _init_ses(self):
        """Initialize AWS SES client"""
        if not self.ses_client:
            self.ses_client = boto3.client(
                'ses',
                aws_access_key_id=self.aws_settings['aws_access_key_id'],
                aws_secret_access_key=self.aws_settings['aws_secret_access_key'],
                region_name=self.aws_settings['region']
            )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def send_email_smtp(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        from_email: str,
        reply_to: Optional[str] = None
    ) -> bool:
        """Send email using SMTP"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_email
            msg['To'] = to_email
            if reply_to:
                msg.add_header('Reply-To', reply_to)
            
            msg.attach(MIMEText(body_html, 'html'))
            
            with smtplib.SMTP(self.smtp_settings['server'], self.smtp_settings['port']) as server:
                server.starttls()
                server.login(self.smtp_settings['username'], self.smtp_settings['password'])
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def send_email_ses(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        from_email: str,
        reply_to: Optional[str] = None
    ) -> bool:
        """Send email using AWS SES"""
        try:
            self._init_ses()
            
            email_message = {
                'Subject': {'Data': subject},
                'Body': {'Html': {'Data': body_html}}
            }
            
            source = from_email
            destination = {'ToAddresses': [to_email]}
            
            if reply_to:
                source_arn = None  # Configure if using dedicated IP
                configuration_set = None  # Configure if using dedicated IP
                
                self.ses_client.send_email(
                    Source=source,
                    SourceArn=source_arn,
                    Destination=destination,
                    Message=email_message,
                    ReplyToAddresses=[reply_to],
                    ConfigurationSetName=configuration_set
                )
            else:
                self.ses_client.send_email(
                    Source=source,
                    Destination=destination,
                    Message=email_message
                )
            return True
        except ClientError as e:
            logger.error(f"Failed to send email via SES: {str(e)}")
            raise
    
    async def send_bulk_emails(
        self,
        leads: List[Lead],
        template: EmailTemplate,
        from_email: str,
        reply_to: Optional[str] = None,
        provider: str = 'smtp',
        update_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Send bulk emails with progress tracking"""
        results = {
            'total': len(leads),
            'sent': 0,
            'failed': 0,
            'errors': []
        }
        
        for i, lead in enumerate(leads):
            try:
                # Personalize email content
                personalized_subject = self._personalize_content(template.subject, lead)
                personalized_body = self._personalize_content(template.body_content, lead)
                
                # Send email using selected provider
                success = False
                if provider == 'smtp':
                    success = await self.send_email_smtp(
                        lead.email,
                        personalized_subject,
                        personalized_body,
                        from_email,
                        reply_to
                    )
                else:
                    success = await self.send_email_ses(
                        lead.email,
                        personalized_subject,
                        personalized_body,
                        from_email,
                        reply_to
                    )
                
                if success:
                    results['sent'] += 1
                    # Update campaign status
                    campaign = EmailCampaign(
                        lead_id=lead.id,
                        template_id=template.id,
                        status='sent',
                        sent_at=datetime.utcnow()
                    )
                    # Note: Session management should be handled by the caller
                
                # Update progress
                if update_callback:
                    progress = (i + 1) / len(leads) * 100
                    await update_callback(progress, results)
                    
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'email': lead.email,
                    'error': str(e)
                })
                logger.error(f"Failed to send email to {lead.email}: {str(e)}")
        
        return results
    
    def _personalize_content(self, content: str, lead: Lead) -> str:
        """Personalize email content with lead information"""
        replacements = {
            '{name}': lead.name or '',
            '{company}': lead.company or '',
            '{position}': lead.position or '',
            '{email}': lead.email,
        }
        
        for key, value in replacements.items():
            content = content.replace(key, value)
        
        return content
    
    async def track_email_open(self, campaign_id: int) -> None:
        """Track email open event"""
        # Implement email tracking logic
        pass
    
    async def track_email_reply(self, campaign_id: int) -> None:
        """Track email reply event"""
        # Implement reply tracking logic
        pass 