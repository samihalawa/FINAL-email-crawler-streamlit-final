from celery import Celery
import multiprocessing
from config import CELERY_CONFIG

# Configure Celery
celery_app = Celery('autoclient_tasks')
celery_app.conf.update(CELERY_CONFIG)

def start_worker():
    """Start Celery worker in a separate process"""
    argv = ['worker', '--loglevel=INFO', '--concurrency=1']
    celery_app.worker_main(argv)

def run_worker_process():
    """Run the worker in a separate process"""
    worker_process = multiprocessing.Process(target=start_worker)
    worker_process.start()
    return worker_process 