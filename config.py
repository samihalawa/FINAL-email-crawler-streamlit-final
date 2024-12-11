import os
from dotenv import load_dotenv

load_dotenv()

# Redis configuration with fallback to local Redis
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# If running on Hugging Face Spaces, use their Redis service
if os.getenv('SPACE_ID'):
    REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:6379/0"

# Celery configuration
CELERY_CONFIG = {
    'broker_url': REDIS_URL,
    'result_backend': REDIS_URL,
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
} 