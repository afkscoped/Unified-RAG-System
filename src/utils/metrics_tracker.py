"""
Real-time Metrics Tracking
"""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class QueryMetric:
    """Single query metric entry"""
    timestamp: datetime
    query: str
    latency_ms: float
    num_results: int
    cache_hit: bool = False
    eval_score: Optional[float] = None


class MetricsTracker:
    """Track system metrics and performance in real-time"""
    
    def __init__(self):
        self.queries: List[QueryMetric] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.documents_processed: int = 0
        self.total_chunks: int = 0
        self.errors: List[Dict] = []
    
    def log_query(
        self,
        query: str,
        latency_ms: float,
        num_results: int,
        cache_hit: bool = False,
        eval_score: Optional[float] = None
    ) -> None:
        """Log search query metrics"""
        self.queries.append(QueryMetric(
            timestamp=datetime.now(),
            query=query,
            latency_ms=latency_ms,
            num_results=num_results,
            cache_hit=cache_hit,
            eval_score=eval_score
        ))
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def log_document(self, num_chunks: int) -> None:
        """Log document processing"""
        self.documents_processed += 1
        self.total_chunks += num_chunks
    
    def log_error(self, error_type: str, message: str) -> None:
        """Log error occurrence"""
        self.errors.append({
            'timestamp': datetime.now(),
            'type': error_type,
            'message': message
        })
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        total_queries = len(self.queries)
        total_cache_ops = self.cache_hits + self.cache_misses
        
        avg_latency = 0.0
        avg_results = 0.0
        avg_eval = 0.0
        
        if total_queries > 0:
            avg_latency = sum(q.latency_ms for q in self.queries) / total_queries
            avg_results = sum(q.num_results for q in self.queries) / total_queries
            eval_scores = [q.eval_score for q in self.queries if q.eval_score is not None]
            if eval_scores:
                avg_eval = sum(eval_scores) / len(eval_scores)
        
        hit_rate = self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0
        
        return {
            'total_queries': total_queries,
            'total_documents': self.documents_processed,
            'total_chunks': self.total_chunks,
            'cache_hit_rate': hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'avg_latency_ms': avg_latency,
            'avg_results': avg_results,
            'avg_eval_score': avg_eval,
            'total_errors': len(self.errors)
        }
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent queries"""
        return [
            {
                'query': q.query,
                'timestamp': q.timestamp.isoformat(),
                'latency_ms': q.latency_ms,
                'num_results': q.num_results,
                'cache_hit': q.cache_hit,
                'eval_score': q.eval_score
            }
            for q in reversed(self.queries[-limit:])
        ]
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.queries.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.documents_processed = 0
        self.total_chunks = 0
        self.errors.clear()
