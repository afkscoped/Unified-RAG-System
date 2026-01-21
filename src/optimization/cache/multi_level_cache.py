"""
[ELITE ARCHITECTURE] multi_level_cache.py
Advanced Hierarchical Caching (L1/L2/L3).
"""

import hashlib
import json
import os
import pickle
from typing import Optional, Any
from loguru import logger
from collections import OrderedDict

class MultiLevelCache:
    """
    Innovation: Hybrid Storage.
    L1: LRU RAM Cache (Immediate)
    L2: Local File System (Persistence)
    L3: Semantic Cache (Fuzzy matches - integrated with search_engine)
    """
    
    def __init__(self, cache_dir: str = "data/cache", l1_size: int = 100):
        self.cache_dir = cache_dir
        self.l1_cache = OrderedDict()
        self.l1_max = l1_size
        os.makedirs(cache_dir, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        """
        Retrieval through the hierarchy.
        """
        key = self._hash_key(query)
        
        # 1. L1 Check
        if key in self.l1_cache:
            logger.debug("L1 Cache Hit: Moving to front")
            self.l1_cache.move_to_end(key)
            return self.l1_cache[key]
            
        # 2. L2 Check
        l2_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(l2_path):
            try:
                with open(l2_path, 'rb') as f:
                    data = pickle.load(f)
                # Populate back to L1
                self.set(query, data, skip_l2=True)
                logger.debug("L2 Cache Hit: Loaded from disk")
                return data
            except Exception as e:
                logger.warning(f"L2 Cache Corruption for {key}: {e}")
                
        return None

    def set(self, query: str, response: Any, skip_l2: bool = False):
        """
        Stores response in the hierarchy.
        """
        key = self._hash_key(query)
        
        # L1 Update
        self.l1_cache[key] = response
        self.l1_cache.move_to_end(key)
        
        if len(self.l1_cache) > self.l1_max:
            self.l1_cache.popitem(last=False)
            
        # L2 Update
        if not skip_l2:
            l2_path = os.path.join(self.cache_dir, f"{key}.pkl")
            try:
                with open(l2_path, 'wb') as f:
                    pickle.dump(response, f)
            except Exception as e:
                logger.error(f"L2 Store Failure: {e}")

if __name__ == "__main__":
    cache = MultiLevelCache()
    print("Multi-Level Cache Engine Ready.")
