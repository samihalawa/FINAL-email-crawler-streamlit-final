from streamlit_app import db_session, Settings
import json

def update_ai_settings():
    with db_session() as session:
        try:
            # Get or create AI settings
            ai_settings = session.query(Settings).filter_by(setting_type='ai').first()
            if not ai_settings:
                ai_settings = Settings(
                    name='ai_settings',
                    setting_type='ai'
                )
                session.add(ai_settings)
            
            # Update with new values
            new_settings = {
                'model_name': 'Qwen/Qwen2.5-72B-Instruct',
                'api_key': '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',
                'api_base_url': 'https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct',
                'inference_api': True
            }
            
            ai_settings.value = new_settings
            session.commit()
            
            print("AI settings updated successfully:")
            print(json.dumps(new_settings, indent=2))
            
        except Exception as e:
            print(f"Error: {str(e)}")
            session.rollback()

if __name__ == "__main__":
    update_ai_settings() 