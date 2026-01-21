"""
[ELITE ARCHITECTURE] rate_limiter.py
Sliding Window Rate Limiter for API stability.
"""

import time
from collections import deque
from loguru import logger

class RateLimiter:
    """
    Innovation: QoS Enforcement.
    Prevents LLM quota exhaustion and GPU thrashing by managing request 
    frequency per user or system-wide.
    """
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()

    def allow_request(self) -> bool:
        """
        Returns True if request is within limits, False otherwise.
        """
        now = time.time()
        
        # Clean expired requests
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()
            
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        else:
            logger.warning(f"Rate Limit Exceeded: {len(self.requests)} requests in {self.window_seconds}s")
            return False

if __name__ == "__main__":
    limiter = RateLimiter(max_requests=2, window_seconds=5)
    print(limiter.allow_request()) # True
    print(limiter.allow_request()) # True
    print(limiter.allow_request()) # False
