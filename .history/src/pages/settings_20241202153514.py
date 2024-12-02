from nicegui import ui
from typing import Dict, Any
import json
from datetime import datetime

from core.database import get_db
from core.auth import auth_handler
from core.config import settings
from services.ai import AIService
from services.email import EmailService

ai_service = AIService()
email_service = EmailService()

@ui.page('/settings')
@auth_handler.require_auth
async def page():
    # State management
    current_settings: Dict[str, Any] = {}
    
    async def load_settings():
        """Load all settings"""
        async with get_db() as session:
            # Load from database or config
            current_settings.update({
                'email': {
                    'smtp_server': settings.SMTP_SERVER,
                    'smtp_port': settings.SMTP_PORT,
                    'smtp_username': settings.SMTP_USERNAME,
                    'smtp_password': settings.SMTP_PASSWORD,
                    'aws_access_key_id': settings.AWS_ACCESS_KEY_ID,
                    'aws_secret_access_key': settings.AWS_SECRET_ACCESS_KEY,
                    'aws_region': settings.AWS_REGION
                },
                'ai': {
                    'openai_api_key': settings.OPENAI_API_KEY,
                    'model': 'gpt-4',
                    'temperature': 0.7
                },
                'automation': {
                    'max_concurrent_processes': 5,
                    'search_delay': 2,
                    'email_delay': 5,
                    'auto_optimize': True,
                    'auto_pause_on_bounce': True,
                    'daily_lead_limit': 100,
                    'daily_email_limit': 200
                },
                'search': {
                    'user_agent_rotation': True,
                    'proxy_enabled': False,
                    'proxy_list': [],
                    'blacklist_domains': [],
                    'min_delay': 1,
                    'max_delay': 5
                },
                'notifications': {
                    'email_enabled': True,
                    'slack_enabled': False,
                    'slack_webhook': '',
                    'notify_on_error': True,
                    'notify_on_completion': True
                }
            })
            update_settings_display()
    
    async def save_settings():
        """Save settings to database and update config"""
        async with get_db() as session:
            # Save to database
            settings_record = await session.get_settings()
            if settings_record:
                settings_record.content = current_settings
                settings_record.updated_at = datetime.utcnow()
            else:
                settings_record = Settings(
                    content=current_settings,
                    created_at=datetime.utcnow()
                )
                session.add(settings_record)
            await session.commit()
            
            # Update environment
            update_environment()
            
            ui.notify('Settings saved successfully')
    
    def update_environment():
        """Update environment variables with new settings"""
        # Update email settings
        settings.SMTP_SERVER = current_settings['email']['smtp_server']
        settings.SMTP_PORT = current_settings['email']['smtp_port']
        settings.SMTP_USERNAME = current_settings['email']['smtp_username']
        settings.SMTP_PASSWORD = current_settings['email']['smtp_password']
        settings.AWS_ACCESS_KEY_ID = current_settings['email']['aws_access_key_id']
        settings.AWS_SECRET_ACCESS_KEY = current_settings['email']['aws_secret_access_key']
        settings.AWS_REGION = current_settings['email']['aws_region']
        
        # Update AI settings
        settings.OPENAI_API_KEY = current_settings['ai']['openai_api_key']
    
    async def test_email_settings():
        """Test email configuration"""
        with ui.loading('Testing email settings...'):
            try:
                # Test SMTP
                if current_settings['email']['smtp_server']:
                    success = await email_service.send_email_smtp(
                        to_email=current_settings['email']['smtp_username'],
                        subject='Test Email',
                        body_html='This is a test email from AutoClient.ai',
                        from_email=current_settings['email']['smtp_username']
                    )
                    if success:
                        ui.notify('SMTP test successful', type='positive')
                    
                # Test AWS SES
                if current_settings['email']['aws_access_key_id']:
                    success = await email_service.send_email_ses(
                        to_email=current_settings['email']['smtp_username'],
                        subject='Test Email',
                        body_html='This is a test email from AutoClient.ai',
                        from_email=current_settings['email']['smtp_username']
                    )
                    if success:
                        ui.notify('AWS SES test successful', type='positive')
                
            except Exception as e:
                ui.notify(f'Email test failed: {str(e)}', type='negative')
    
    async def test_ai_settings():
        """Test AI configuration"""
        with ui.loading('Testing AI settings...'):
            try:
                response = await ai_service.test_connection()
                if response:
                    ui.notify('AI connection successful', type='positive')
            except Exception as e:
                ui.notify(f'AI test failed: {str(e)}', type='negative')
    
    def update_settings_display():
        """Update settings UI"""
        with settings_container:
            settings_container.clear()
            
            with ui.tabs().classes('w-full') as tabs:
                ui.tab('Email')
                ui.tab('AI & Automation')
                ui.tab('Search')
                ui.tab('Notifications')
                ui.tab('Advanced')
            
            with ui.tab_panels(tabs).classes('w-full'):
                # Email Settings Panel
                with ui.tab_panel('Email'):
                    with ui.card().classes('w-full q-ma-md'):
                        ui.label('SMTP Settings').classes('text-h6')
                        ui.input(
                            'SMTP Server',
                            value=current_settings['email']['smtp_server'],
                            on_change=lambda e: current_settings['email'].update({'smtp_server': e.value})
                        ).classes('w-full')
                        ui.number(
                            'SMTP Port',
                            value=current_settings['email']['smtp_port'],
                            on_change=lambda e: current_settings['email'].update({'smtp_port': e.value})
                        )
                        ui.input(
                            'SMTP Username',
                            value=current_settings['email']['smtp_username'],
                            on_change=lambda e: current_settings['email'].update({'smtp_username': e.value})
                        ).classes('w-full')
                        ui.input(
                            'SMTP Password',
                            value=current_settings['email']['smtp_password'],
                            password=True,
                            on_change=lambda e: current_settings['email'].update({'smtp_password': e.value})
                        ).classes('w-full')
                    
                    with ui.card().classes('w-full q-ma-md'):
                        ui.label('AWS SES Settings').classes('text-h6')
                        ui.input(
                            'AWS Access Key ID',
                            value=current_settings['email']['aws_access_key_id'],
                            on_change=lambda e: current_settings['email'].update({'aws_access_key_id': e.value})
                        ).classes('w-full')
                        ui.input(
                            'AWS Secret Access Key',
                            value=current_settings['email']['aws_secret_access_key'],
                            password=True,
                            on_change=lambda e: current_settings['email'].update({'aws_secret_access_key': e.value})
                        ).classes('w-full')
                        ui.input(
                            'AWS Region',
                            value=current_settings['email']['aws_region'],
                            on_change=lambda e: current_settings['email'].update({'aws_region': e.value})
                        ).classes('w-full')
                    
                    ui.button('Test Email Settings', on_click=test_email_settings).props('primary')
                
                # AI & Automation Panel
                with ui.tab_panel('AI & Automation'):
                    with ui.card().classes('w-full q-ma-md'):
                        ui.label('AI Settings').classes('text-h6')
                        ui.input(
                            'OpenAI API Key',
                            value=current_settings['ai']['openai_api_key'],
                            password=True,
                            on_change=lambda e: current_settings['ai'].update({'openai_api_key': e.value})
                        ).classes('w-full')
                        ui.select(
                            'Model',
                            options=['gpt-4', 'gpt-3.5-turbo'],
                            value=current_settings['ai']['model'],
                            on_change=lambda e: current_settings['ai'].update({'model': e.value})
                        )
                        ui.number(
                            'Temperature',
                            value=current_settings['ai']['temperature'],
                            min=0,
                            max=1,
                            step=0.1,
                            on_change=lambda e: current_settings['ai'].update({'temperature': e.value})
                        )
                        ui.button('Test AI Settings', on_click=test_ai_settings).props('primary')
                    
                    with ui.card().classes('w-full q-ma-md'):
                        ui.label('Automation Settings').classes('text-h6')
                        ui.number(
                            'Max Concurrent Processes',
                            value=current_settings['automation']['max_concurrent_processes'],
                            on_change=lambda e: current_settings['automation'].update({'max_concurrent_processes': e.value})
                        )
                        ui.number(
                            'Search Delay (seconds)',
                            value=current_settings['automation']['search_delay'],
                            on_change=lambda e: current_settings['automation'].update({'search_delay': e.value})
                        )
                        ui.number(
                            'Email Delay (seconds)',
                            value=current_settings['automation']['email_delay'],
                            on_change=lambda e: current_settings['automation'].update({'email_delay': e.value})
                        )
                        ui.checkbox(
                            'Auto-optimize Templates',
                            value=current_settings['automation']['auto_optimize'],
                            on_change=lambda e: current_settings['automation'].update({'auto_optimize': e.value})
                        )
                        ui.checkbox(
                            'Auto-pause on High Bounce Rate',
                            value=current_settings['automation']['auto_pause_on_bounce'],
                            on_change=lambda e: current_settings['automation'].update({'auto_pause_on_bounce': e.value})
                        )
                        ui.number(
                            'Daily Lead Limit',
                            value=current_settings['automation']['daily_lead_limit'],
                            on_change=lambda e: current_settings['automation'].update({'daily_lead_limit': e.value})
                        )
                        ui.number(
                            'Daily Email Limit',
                            value=current_settings['automation']['daily_email_limit'],
                            on_change=lambda e: current_settings['automation'].update({'daily_email_limit': e.value})
                        )
                
                # Search Settings Panel
                with ui.tab_panel('Search'):
                    with ui.card().classes('w-full q-ma-md'):
                        ui.label('Search Settings').classes('text-h6')
                        ui.checkbox(
                            'Enable User Agent Rotation',
                            value=current_settings['search']['user_agent_rotation'],
                            on_change=lambda e: current_settings['search'].update({'user_agent_rotation': e.value})
                        )
                        ui.checkbox(
                            'Enable Proxy',
                            value=current_settings['search']['proxy_enabled'],
                            on_change=lambda e: current_settings['search'].update({'proxy_enabled': e.value})
                        )
                        
                        with ui.expansion('Proxy List', value=True):
                            proxy_list = ui.textarea(
                                'One proxy per line',
                                value='\n'.join(current_settings['search']['proxy_list']),
                                on_change=lambda e: current_settings['search'].update({'proxy_list': e.value.split('\n')})
                            ).classes('w-full')
                        
                        with ui.expansion('Domain Blacklist', value=True):
                            blacklist = ui.textarea(
                                'One domain per line',
                                value='\n'.join(current_settings['search']['blacklist_domains']),
                                on_change=lambda e: current_settings['search'].update({'blacklist_domains': e.value.split('\n')})
                            ).classes('w-full')
                        
                        ui.number(
                            'Min Delay (seconds)',
                            value=current_settings['search']['min_delay'],
                            on_change=lambda e: current_settings['search'].update({'min_delay': e.value})
                        )
                        ui.number(
                            'Max Delay (seconds)',
                            value=current_settings['search']['max_delay'],
                            on_change=lambda e: current_settings['search'].update({'max_delay': e.value})
                        )
                
                # Notifications Panel
                with ui.tab_panel('Notifications'):
                    with ui.card().classes('w-full q-ma-md'):
                        ui.label('Notification Settings').classes('text-h6')
                        ui.checkbox(
                            'Enable Email Notifications',
                            value=current_settings['notifications']['email_enabled'],
                            on_change=lambda e: current_settings['notifications'].update({'email_enabled': e.value})
                        )
                        ui.checkbox(
                            'Enable Slack Notifications',
                            value=current_settings['notifications']['slack_enabled'],
                            on_change=lambda e: current_settings['notifications'].update({'slack_enabled': e.value})
                        )
                        ui.input(
                            'Slack Webhook URL',
                            value=current_settings['notifications']['slack_webhook'],
                            password=True,
                            on_change=lambda e: current_settings['notifications'].update({'slack_webhook': e.value})
                        ).classes('w-full')
                        ui.checkbox(
                            'Notify on Error',
                            value=current_settings['notifications']['notify_on_error'],
                            on_change=lambda e: current_settings['notifications'].update({'notify_on_error': e.value})
                        )
                        ui.checkbox(
                            'Notify on Completion',
                            value=current_settings['notifications']['notify_on_completion'],
                            on_change=lambda e: current_settings['notifications'].update({'notify_on_completion': e.value})
                        )
                
                # Advanced Panel
                with ui.tab_panel('Advanced'):
                    with ui.card().classes('w-full q-ma-md'):
                        ui.label('Advanced Settings').classes('text-h6')
                        with ui.expansion('Raw Settings', value=True):
                            ui.textarea(
                                'JSON Configuration',
                                value=json.dumps(current_settings, indent=2),
                                on_change=lambda e: current_settings.update(json.loads(e.value))
                            ).classes('w-full')
                    
                    with ui.card().classes('w-full q-ma-md'):
                        ui.label('Maintenance').classes('text-h6')
                        ui.button('Clear Cache', on_click=clear_cache).props('warning')
                        ui.button('Reset Settings', on_click=reset_settings).props('negative')
    
    async def clear_cache():
        """Clear application cache"""
        with ui.loading('Clearing cache...'):
            try:
                async with get_db() as session:
                    # Clear various caches
                    await session.execute("DELETE FROM cache")
                    await session.commit()
                ui.notify('Cache cleared successfully')
            except Exception as e:
                ui.notify(f'Failed to clear cache: {str(e)}', type='negative')
    
    async def reset_settings():
        """Reset settings to defaults"""
        if await ui.ask('Are you sure you want to reset all settings?'):
            async with get_db() as session:
                await session.execute("DELETE FROM settings")
                await session.commit()
            await load_settings()
            ui.notify('Settings reset to defaults')
    
    # UI Layout
    ui.label('Settings').classes('text-h4 q-ma-md')
    
    with ui.row().classes('w-full justify-end q-ma-md'):
        ui.button('Save All Settings', on_click=save_settings).props('primary')
        ui.button('Reload', on_click=load_settings).props('outline')
    
    # Settings Container
    settings_container = ui.column().classes('w-full')
    
    # Load initial settings
    await load_settings() 