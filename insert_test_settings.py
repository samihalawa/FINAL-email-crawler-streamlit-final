from app import *
import json

def insert_test_settings():
    try:
        with db_session() as session:
            # Test search settings
            search_settings = Settings(
                name='search_config',
                setting_type='search',
                value=json.dumps({
                    'max_results_per_term': 20,
                    'rate_limit_delay': 2,
                    'ignore_previously_fetched': True,
                    'default_language': 'EN'
                }),
                is_active=True
            )
            
            # Test email settings
            email_settings = Settings(
                name='email_config',
                setting_type='email',
                value=json.dumps({
                    'max_emails_per_day': 50,
                    'retry_attempts': 5,
                    'retry_delay': 30
                }),
                is_active=True
            )
            
            # Test AI settings
            ai_settings = Settings(
                name='ai_config',
                setting_type='ai',
                value=json.dumps({
                    'model_name': 'gpt-4',
                    'max_tokens': 2000,
                    'temperature': 0.8,
                    'openai_api_key': 'test_key'
                }),
                is_active=True
            )
            
            # Delete any existing settings first
            session.query(Settings).delete()
            
            # Add new settings
            session.add(search_settings)
            session.add(email_settings)
            session.add(ai_settings)
            session.commit()
            print('Test settings inserted successfully')
            
    except Exception as e:
        print(f'Error inserting test settings: {str(e)}')

if __name__ == "__main__":
    insert_test_settings() 