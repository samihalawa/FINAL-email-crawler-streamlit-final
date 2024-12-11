from celery import Celery
from config import CELERY_CONFIG

celery_app = Celery('autoclient_tasks')
celery_app.conf.update(CELERY_CONFIG)

# Optional task routing
celery_app.conf.task_routes = {
    'tasks.background_search_task': {'queue': 'search'},
    'tasks.automation_loop_task': {'queue': 'automation'}
}

# Optional task time limits
celery_app.conf.task_time_limit = 3600  # 1 hour
celery_app.conf.task_soft_time_limit = 3300  # 55 minutes 