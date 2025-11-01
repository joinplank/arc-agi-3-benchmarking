"""Custom exceptions for ARC-AGI-3 benchmarking"""


class ArcAgi3Error(Exception):
    """Base exception for all ARC-AGI-3 errors."""
    
    def __init__(self, message: str, cause: Exception = None, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message, **kwargs)
        self.message = message
        self.cause = cause
    
    def __str__(self):
        if self.cause:
            return f"{self.message} (caused by: {type(self.cause).__name__}: {str(self.cause)})"
        return self.message


class TokenMismatchError(ArcAgi3Error):
    """Raised when token counts do not add up correctly."""
    
    def __init__(self, message: str, prompt_tokens: int = None, 
                 completion_tokens: int = None, reason: str = None):
        """
        Initialize the token mismatch error.
        
        Args:
            message: Error message
            prompt_tokens: Number of prompt tokens counted
            completion_tokens: Number of completion tokens counted  
            reason: Reason for the mismatch
        """
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reason = reason


class GameClientError(ArcAgi3Error):
    """Raised when there's an error communicating with the ARC-AGI-3 API."""
    
    def __init__(self, message: str, status_code: int = None, 
                 response_body: str = None, endpoint: str = None):
        """
        Initialize the game client error.
        
        Args:
            message: Error message
            status_code: HTTP status code from the API
            response_body: Response body from failed request
            endpoint: API endpoint that failed
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.endpoint = endpoint
    
    @classmethod
    def from_http_error(cls, request_exception, endpoint: str = None):
        """Create a GameClientError from a requests HTTPError."""
        status_code = None
        response_body = ""
        
        if hasattr(request_exception, 'response') and request_exception.response is not None:
            status_code = request_exception.response.status_code
            try:
                response_body = request_exception.response.text
            except:
                response_body = "Could not read response"
        
        return cls(
            message=f"HTTP error communicating with ARC-AGI-3 API",
            status_code=status_code,
            response_body=response_body,
            endpoint=endpoint,
            cause=request_exception
        )


class ProviderError(ArcAgi3Error):
    """Raised when there's an error with an LLM provider."""
    
    def __init__(self, message: str, provider: str = None, 
                 model: str = None, operation: str = None):
        """
        Initialize the provider error.
        
        Args:
            message: Error message
            provider: Provider name (e.g., 'openai', 'anthropic')
            model: Model name that failed
            operation: Operation that failed (e.g., 'chat_completion')
        """
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.operation = operation
    
    @classmethod
    def from_provider_error(cls, provider_exception, provider: str = None,
                           model: str = None, operation: str = None):
        """Create a ProviderError from a provider-specific exception."""
        return cls(
            message=f"Provider error: {str(provider_exception)}",
            provider=provider,
            model=model,
            operation=operation,
            cause=provider_exception
        )

