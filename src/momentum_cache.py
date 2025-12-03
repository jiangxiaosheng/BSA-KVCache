"""
MomentumDecay Cache Eviction Algorithm

This cache eviction algorithm tracks the moving momentum of attention scores
for each block. The momentum is updated as:
    x_t = beta * x_{t-1} + s_t

where:
    - x_t is the momentum score at time t
    - beta is the decay factor (default 0.9)
    - s_t is the normalized softmax score for this block

During eviction, the block with the lowest momentum score is evicted.
"""

import heapq
from typing import Dict, Tuple, Optional
from libcachesim.cache import CacheBase
from libcachesim import Request


class MomentumDecayCache(CacheBase):
    """
    Cache with momentum-decay based eviction.
    
    Evicts blocks with the lowest accumulated momentum score.
    Score update rule: x_t = beta * x_{t-1} + s_t
    
    Uses a min-heap with lazy deletion for O(log n) eviction.
    
    Inherits from CacheBase to ensure fair comparison with libcachesim algorithms.
    """
    
    def __init__(self, cache_size: int, beta: float = 0.9):
        """
        Args:
            cache_size: Maximum cache size in bytes
            beta: Decay factor for momentum (default 0.9)
        """
        # Don't call super().__init__() since we don't have a C++ cache object
        # Instead, we implement all required methods in pure Python
        self._cache_size = cache_size
        self._beta = beta
        
        # obj_id -> (momentum_score, obj_size, version)
        self._data: Dict[int, Tuple[float, int, int]] = {}
        
        # Min-heap of (momentum_score, version, obj_id)
        # We use version to handle lazy deletion when scores are updated
        self._heap: list = []
        
        self._occupied_bytes = 0
    
    def _evict_one(self) -> Optional[int]:
        """Evict the entry with the lowest momentum score. O(log n) amortized."""
        while self._heap:
            score, version, obj_id = heapq.heappop(self._heap)
            
            # Check if this entry is still valid (not stale)
            if obj_id in self._data:
                cached_score, obj_size, cached_version = self._data[obj_id]
                if version == cached_version:
                    # Valid entry - evict it
                    del self._data[obj_id]
                    self._occupied_bytes -= obj_size
                    return obj_id
            # Stale entry - continue to next
        
        return None
    
    def _ensure_capacity(self, required_size: int):
        """Evict entries until we have enough space."""
        while self._occupied_bytes + required_size > self._cache_size and self._data:
            self._evict_one()
    
    def get(self, req: Request) -> bool:
        """
        Process a cache request (CacheBase interface).
        
        This method is called by process_trace() and uses a default score of 1.0
        to ensure fair comparison with other cache algorithms.
        
        Args:
            req: Request object with obj_id and obj_size
            
        Returns:
            True if cache hit, False if cache miss
        """
        return self._access(req.obj_id, req.obj_size, score=1.0)
    
    def _access(self, obj_id: int, obj_size: int, score: float = 1.0) -> bool:
        """
        Internal access method with score support.
        
        Args:
            obj_id: Unique identifier for the object
            obj_size: Size of the object in bytes
            score: The score for this access (default 1.0 for fair comparison)
            
        Returns:
            True if cache hit, False if cache miss
        """
        if obj_id in self._data:
            # Cache hit - update momentum score
            old_score, obj_size, old_version = self._data[obj_id]
            new_score = self._beta * old_score + score
            new_version = old_version + 1
            
            # Update cache entry with new version
            self._data[obj_id] = (new_score, obj_size, new_version)
            
            # Push new entry to heap (old one becomes stale)
            heapq.heappush(self._heap, (new_score, new_version, obj_id))
            
            return True
        else:
            # Cache miss - need to insert
            # Ensure we have capacity
            self._ensure_capacity(obj_size)
            
            # Insert new entry with version 0
            self._data[obj_id] = (score, obj_size, 0)
            heapq.heappush(self._heap, (score, 0, obj_id))
            self._occupied_bytes += obj_size
            
            return False
    
    def get_occupied_byte(self) -> int:
        """Return the number of bytes currently in cache."""
        return self._occupied_bytes
    
    def get_n_obj(self) -> int:
        """Return the number of objects currently in cache."""
        return len(self._data)
    
    @property
    def cache_size(self) -> int:
        """Return the maximum cache size in bytes."""
        return self._cache_size
    
    @property
    def cache_name(self) -> str:
        """Return the name of this cache algorithm."""
        return f"MomentumDecay(beta={self._beta})"
    
    # Additional CacheBase methods - not all are needed for process_trace
    def find(self, req: Request, update_cache: bool = True) -> Optional[object]:
        """Find an object in cache without full get semantics."""
        if req.obj_id in self._data:
            return self._data[req.obj_id]
        return None
    
    def can_insert(self, req: Request) -> bool:
        """Check if an object can be inserted."""
        return True  # We can always insert after eviction
    
    def need_eviction(self, req: Request) -> bool:
        """Check if eviction is needed to insert this object."""
        return self._occupied_bytes + req.obj_size > self._cache_size
    
    def remove(self, obj_id: int) -> bool:
        """Remove an object from cache."""
        if obj_id in self._data:
            _, obj_size, _ = self._data[obj_id]
            del self._data[obj_id]
            self._occupied_bytes -= obj_size
            return True
        return False
    
    def set_cache_size(self, new_size: int) -> None:
        """Set a new cache size."""
        self._cache_size = new_size
        # Evict if necessary
        while self._occupied_bytes > self._cache_size and self._data:
            self._evict_one()
    
    def print_cache(self) -> str:
        """Return a string representation of the cache."""
        return f"MomentumDecayCache(size={self._cache_size}, occupied={self._occupied_bytes}, n_obj={len(self._data)})"
