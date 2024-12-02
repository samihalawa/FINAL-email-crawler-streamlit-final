import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import traceback
from functools import wraps

from core.config import settings

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            logs_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        )
    ]
)

# Create logger
logger = logging.getLogger("autoclient")

class StructuredLogger:
    """Structured logging with context and error tracking"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set context for all subsequent log messages"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear the current context"""
        self.context.clear()
    
    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Format message with context and extra data"""
        data = {
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            **self.context
        }
        if extra:
            data.update(extra)
        return json.dumps(data)
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """Log error message with context and optional exception info"""
        extra = kwargs.copy()
        if exc_info:
            extra.update({
                "exception_type": exc_info.__class__.__name__,
                "exception_message": str(exc_info),
                "traceback": traceback.format_exc()
            })
        self.logger.error(self._format_message(message, extra))
    
    def critical(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """Log critical error with context and optional exception info"""
        extra = kwargs.copy()
        if exc_info:
            extra.update({
                "exception_type": exc_info.__class__.__name__,
                "exception_message": str(exc_info),
                "traceback": traceback.format_exc()
            })
        self.logger.critical(self._format_message(message, extra))

# Custom exceptions
class AutoClientError(Exception):
    """Base exception for AutoClient"""
    pass

class ValidationError(AutoClientError):
    """Validation error"""
    pass

class RateLimitError(AutoClientError):
    """Rate limit exceeded"""
    pass

class AuthenticationError(AutoClientError):
    """Authentication error"""
    pass

class APIError(AutoClientError):
    """External API error"""
    pass

# Error tracking
def track_errors(logger: StructuredLogger):
    """Decorator to track function errors"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}",
                    exc_info=e,
                    function=func.__name__,
                    args=args,
                    kwargs=kwargs
                )
                raise
        return wrapper
    return decorator

# Create default logger instance
app_logger = StructuredLogger("autoclient")

# Example usage:
# @track_errors(app_logger)
# async def some_function():
#     app_logger.set_context(user_id="123", action="search")
#     app_logger.info("Starting search operation")
#     try:
#         # Do something
#         pass
#     except Exception as e:
#         app_logger.error("Search failed", exc_info=e)
#         raise
#     finally:
#         app_logger.clear_context() 