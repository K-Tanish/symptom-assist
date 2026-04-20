"""
error_handler.py
----------------
Centralized error handling for API calls with retry logic and user-friendly messages.

Supports:
  - Groq API errors (RateLimitError, APIError, APIConnectionError, APITimeoutError)
  - Graceful fallback messages for different error types
  - Exponential backoff retry logic
"""

import time
from typing import Callable, TypeVar, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Define error types for type checking
T = TypeVar('T')

class APIErrorHandler:
    """
    Centralized handler for API errors with user-friendly messages
    and smart retry logic.
    """
    
    # User-friendly error messages
    ERROR_MESSAGES = {
        "rate_limit": "I'm temporarily receiving too many requests. Please wait a moment and try again.",
        "authentication": "Authentication failed. Please check your API key configuration.",
        "connection": "I'm having trouble connecting to the diagnosis engine. Please check your internet connection and try again.",
        "timeout": "The request took too long. Please try again with a simpler query.",
        "server_error": "The diagnosis service is temporarily unavailable. Please try again in a moment.",
        "invalid_request": "There was an issue with your request. Please try rephrasing your symptoms.",
        "unknown": "An unexpected error occurred while processing your request. Please try again.",
    }
    
    @staticmethod
    def is_rate_limit_error(exception: Exception) -> bool:
        """
        Check if the exception is a rate limit error.
        Handles multiple error formats from Groq API.
        """
        error_str = str(exception).lower()
        error_type = type(exception).__name__.lower()
        
        # Check exception type (if using groq library)
        if "ratelimit" in error_type or "429" in error_type:
            return True
        
        # Check error message content
        rate_limit_keywords = ["429", "rate limit", "too many requests", "quota exceeded", "limited"]
        return any(keyword in error_str for keyword in rate_limit_keywords)
    
    @staticmethod
    def is_authentication_error(exception: Exception) -> bool:
        """Check if the exception is an authentication/API key error."""
        error_str = str(exception).lower()
        error_type = type(exception).__name__.lower()
        
        if "auth" in error_type or "unauthorized" in error_type:
            return True
        
        auth_keywords = ["401", "invalid api key", "unauthorized", "authentication failed"]
        return any(keyword in error_str for keyword in auth_keywords)
    
    @staticmethod
    def is_connection_error(exception: Exception) -> bool:
        """Check if the exception is a connection error."""
        error_type = type(exception).__name__.lower()
        error_str = str(exception).lower()
        
        connection_types = ["connectionerror", "connectionrefused", "timeout", "unreachable"]
        connection_keywords = ["connection", "refused", "network", "unreachable", "dns"]
        
        if any(ct in error_type for ct in connection_types):
            return True
        return any(keyword in error_str for keyword in connection_keywords)
    
    @staticmethod
    def is_server_error(exception: Exception) -> bool:
        """Check if the exception is a server error (5xx)."""
        error_str = str(exception).lower()
        error_type = type(exception).__name__.lower()
        
        if "500" in error_str or "502" in error_str or "503" in error_str:
            return True
        
        server_keywords = ["internal server error", "service unavailable", "bad gateway"]
        return any(keyword in error_str for keyword in server_keywords)
    
    @staticmethod
    def get_error_category(exception: Exception) -> str:
        """
        Classify the exception into a known category.
        Returns one of: rate_limit, authentication, connection, server_error, 
                        invalid_request, unknown
        """
        if APIErrorHandler.is_rate_limit_error(exception):
            return "rate_limit"
        elif APIErrorHandler.is_authentication_error(exception):
            return "authentication"
        elif APIErrorHandler.is_connection_error(exception):
            return "connection"
        elif APIErrorHandler.is_server_error(exception):
            return "server_error"
        elif "400" in str(exception):
            return "invalid_request"
        else:
            return "unknown"
    
    @staticmethod
    def get_user_message(exception: Exception) -> str:
        """Get user-friendly error message for the exception."""
        category = APIErrorHandler.get_error_category(exception)
        return APIErrorHandler.ERROR_MESSAGES.get(category, APIErrorHandler.ERROR_MESSAGES["unknown"])
    
    @staticmethod
    def log_error(exception: Exception, context: str = "") -> None:
        """Log error with full traceback for debugging."""
        category = APIErrorHandler.get_error_category(exception)
        logger.error(
            f"API Error [{category}] {context}: {type(exception).__name__}: {str(exception)}",
            exc_info=True
        )


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential calculation
    
    Usage:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_groq_api():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry on authentication errors
                    if APIErrorHandler.is_authentication_error(e):
                        APIErrorHandler.log_error(e, f"Attempt {attempt + 1}/{max_retries + 1}")
                        raise
                    
                    # Only retry on rate limit and connection errors
                    if not (APIErrorHandler.is_rate_limit_error(e) or 
                           APIErrorHandler.is_connection_error(e)):
                        APIErrorHandler.log_error(e, f"Attempt {attempt + 1}/{max_retries + 1}")
                        raise
                    
                    if attempt < max_retries:
                        # Calculate exponential backoff
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        logger.warning(
                            f"Retrying after {delay}s (attempt {attempt + 1}/{max_retries})... "
                            f"Error: {type(e).__name__}"
                        )
                        time.sleep(delay)
                    else:
                        APIErrorHandler.log_error(e, f"Final attempt {attempt + 1}/{max_retries + 1}")
            
            # If all retries exhausted, raise last exception
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator