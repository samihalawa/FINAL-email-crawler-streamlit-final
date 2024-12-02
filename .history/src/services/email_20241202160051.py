from typing import Dict, List, Optional, Tuple
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import boto3
from botocore.exceptions import ClientError
import logging
from datetime import datetime
import asyncio

from core.database import get_db, EmailCampaign, Lead
from services.ai import AIService

ai_service = AIService()

class EmailService:
    def __init__(self):
        self.ses_client = None
        self.smtp_settings = None
        
    async def initialize_smtp(self, settings: Dict):
        """Initialize SMTP settings"""
        self.smtp_settings = {
            'hostname': settings['smtp_host'],
            'port': settings['smtp_port'],
            'username': settings['smtp_username'],
            'password': settings['smtp_password'],
            'use_tls': settings.get('smtp_use_tls', True)
        }
        
    def initialize_ses(self, settings: Dict):
        """Initialize AWS SES client"""
        self.ses_client = boto3.client(
            'ses',
            aws_access_key_id=settings['aws_access_key'],
            aws_secret_access_key=settings['aws_secret_key'],
            region_name=settings['aws_region']
        )
        
    async def send_email_smtp(self, to_email: str, subject: str, body: str, from_email: str, reply_to: str) -> bool:
        """Send email using SMTP"""
        if not self.smtp_settings:
            raise ValueError("SMTP not configured")
            
        try:
            message = MIMEMultipart()
            message['From'] = from_email
            message['To'] = to_email
            message['Subject'] = subject
            message['Reply-To'] = reply_to
            
            message.attach(MIMEText(body, 'html'))
            
            await aiosmtplib.send(
                message,
                hostname=self.smtp_settings['hostname'],
                port=self.smtp_settings['port'],
                username=self.smtp_settings['username'],
                password=self.smtp_settings['password'],
                use_tls=self.smtp_settings['use_tls']
            )
            
            return True
            
        except Exception as e:
            logging.error(f"SMTP error sending to {to_email}: {str(e)}")
            return False
            
    def send_email_ses(self, to_email: str, subject: str, body: str, from_email: str, reply_to: str) -> bool:
        """Send email using AWS SES"""
        if not self.ses_client:
            raise ValueError("AWS SES not configured")
            
        try:
            response = self.ses_client.send_email(
                Source=from_email,
                Destination={'ToAddresses': [to_email]},
                Message={
                    'Subject': {'Data': subject},
                    'Body': {'Html': {'Data': body}}
                },
                ReplyToAddresses=[reply_to]
            )
            
            return True
            
        except ClientError as e:
            logging.error(f"AWS SES error sending to {to_email}: {str(e)}")
            return False
            
    async def send_bulk_emails(
        self,
        leads: List[Lead],
        template: Dict,
        from_email: str,
        reply_to: str,
        provider: str = 'smtp',
        progress_callback = None
    ) -> Dict:
        """Send bulk emails with progress tracking"""
        results = {
            'sent': 0,
            'failed': 0,
            'errors': []
        }
        
        async with get_db() as session:
            # Create campaign
            campaign = EmailCampaign(
                template_id=template['id'],
                started_at=datetime.utcnow(),
                status='running'
            )
            session.add(campaign)
            await session.commit()
            
            total_leads = len(leads)
            for i, lead in enumerate(leads):
                try:
                    # Personalize email for lead
                    personalized = await ai_service.personalize_email(template, lead)
                    
                    # Send email based on provider
                    success = False
                    if provider == 'smtp':
                        success = await self.send_email_smtp(
                            lead.email,
                            personalized['subject'],
                            personalized['body'],
                            from_email,
                            reply_to
                        )
                    else:
                        success = self.send_email_ses(
                            lead.email,
                            personalized['subject'],
                            personalized['body'],
                            from_email,
                            reply_to
                        )
                        
                    if success:
                        results['sent'] += 1
                        lead.last_contacted = datetime.utcnow()
                        lead.campaign_id = campaign.id
                    else:
                        results['failed'] += 1
                        results['errors'].append({
                            'email': lead.email,
                            'error': 'Failed to send'
                        })
                        
                    # Update progress
                    if progress_callback:
                        await progress_callback((i + 1) / total_leads, results)
                        
                    # Avoid rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append({
                        'email': lead.email,
                        'error': str(e)
                    })
                    
            # Update campaign status
            campaign.completed_at = datetime.utcnow()
            campaign.status = 'completed'
            campaign.results = results
            await session.commit()
            
        return results
        
    async def test_email_config(self, settings: Dict, provider: str, test_email: str) -> Tuple[bool, str]:
        """Test email configuration"""
        try:
            if provider == 'smtp':
                await self.initialize_smtp(settings)
                success = await self.send_email_smtp(
                    test_email,
                    'Test Email',
                    'This is a test email from your application.',
                    settings['from_email'],
                    settings['reply_to']
                )
            else:
                self.initialize_ses(settings)
                success = self.send_email_ses(
                    test_email,
                    'Test Email',
                    'This is a test email from your application.',
                    settings['from_email'],
                    settings['reply_to']
                )
                
            return success, 'Email configuration test successful' if success else 'Failed to send test email'
            
        except Exception as e:
            return False, f'Email configuration test failed: {str(e)}' 