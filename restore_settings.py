from app import *
import json

def restore_settings():
    try:
        with db_session() as session:
            # General settings
            general_settings = Settings(
                name='general_settings',
                setting_type='general',
                value={
                    'openai_api_key': '',
                    'openai_api_base': 'https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1/',
                    'openai_model': 'Qwen/Qwen2.5-72B-Instruct'
                },
                is_active=True
            )
            
            # Search settings
            search_settings = Settings(
                name='search_settings',
                setting_type='search',
                value={
                    'max_results_per_term': 50,
                    'rate_limit_delay': 1,
                    'ignore_previously_fetched': True,
                    'default_language': 'ES'
                },
                is_active=True
            )
            
            # Email settings
            email_settings = Settings(
                name='email_settings',
                setting_type='email',
                value={
                    'max_emails_per_day': 100,
                    'retry_attempts': 3,
                    'retry_delay': 60
                },
                is_active=True
            )
            
            # Delete existing settings
            session.query(Settings).delete()
            
            # Add new settings
            session.add(general_settings)
            session.add(search_settings)
            session.add(email_settings)
            
            session.commit()
            print('Settings restored successfully')
            
            # Verify settings
            settings = session.query(Settings).all()
            print('\nVerifying settings:')
            for s in settings:
                print(f'{s.setting_type}: {s.value}')
                
    except Exception as e:
        print(f'Error restoring settings: {str(e)}')

if __name__ == "__main__":
    restore_settings() 