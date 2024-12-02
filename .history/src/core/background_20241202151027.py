import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)

class BackgroundTask:
    def __init__(self, func: Callable, *args, **kwargs):
        self.id = str(uuid4())
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = "pending"
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.progress = 0
        self.task: Optional[asyncio.Task] = None

    async def run(self):
        self.started_at = datetime.utcnow()
        self.status = "running"
        try:
            self.result = await self.func(*self.args, **self.kwargs)
            self.status = "completed"
        except Exception as e:
            self.error = str(e)
            self.status = "failed"
            logger.exception(f"Task {self.id} failed")
        finally:
            self.completed_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error
        }

class BackgroundTaskManager:
    def __init__(self):
        self.tasks: Dict[str, BackgroundTask] = {}
        self._running = False
        self._task_queue = asyncio.Queue()
        self._worker_task = None

    async def start(self):
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Background task manager started")

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Background task manager stopped")

    async def _worker(self):
        while self._running:
            try:
                task = await self._task_queue.get()
                asyncio.create_task(self._run_task(task))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in background worker")

    async def _run_task(self, task: BackgroundTask):
        task.task = asyncio.create_task(task.run())
        try:
            await task.task
        except asyncio.CancelledError:
            task.status = "cancelled"
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.exception(f"Task {task.id} failed")

    async def add_task(self, func: Callable, *args, **kwargs) -> str:
        task = BackgroundTask(func, *args, **kwargs)
        self.tasks[task.id] = task
        await self._task_queue.put(task)
        return task.id

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        return {task_id: task.to_dict() for task_id, task in self.tasks.items()}

    async def cancel_task(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        if task and task.task:
            task.task.cancel()
            return True
        return False

    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        now = datetime.utcnow()
        to_remove = []
        for task_id, task in self.tasks.items():
            if task.completed_at and (now - task.completed_at).total_seconds() > max_age_hours * 3600:
                to_remove.append(task_id)
        for task_id in to_remove:
            del self.tasks[task_id] 