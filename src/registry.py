import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import aiofiles

logger = logging.getLogger(__name__)

class Registry:
    """Registry for managing routing information and cache."""

    def __init__(self, cache_file: str = "routing_cache.json"):
        self.cache_file = cache_file
        self.routing_cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def _load_cache(self) -> None:
        """Load routing cache from file."""
        try:
            if os.path.exists(self.cache_file):
                async with aiofiles.open(self.cache_file, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    self.routing_cache = data.get('cache', {})
                    # Convert expiry timestamps back to datetime objects
                    self.cache_expiry = {}
                    for key, timestamp in data.get('expiry', {}).items():
                        self.cache_expiry[key] = datetime.fromisoformat(timestamp)
        except Exception as e:
            logger.warning(f"Failed to load routing cache: {e}")
            self.routing_cache = {}
            self.cache_expiry = {}

    async def _save_cache(self) -> None:
        """Save routing cache to file."""
        try:
            data = {
                'cache': self.routing_cache,
                'expiry': {k: v.isoformat() for k, v in self.cache_expiry.items()}
            }
            async with aiofiles.open(self.cache_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save routing cache: {e}")

    async def get_routing_info(self, key: str) -> Optional[Any]:
        """Get routing information from cache."""
        async with self._lock:
            await self._load_cache()  # Ensure cache is loaded

            if key in self.routing_cache:
                if key in self.cache_expiry and datetime.now() < self.cache_expiry[key]:
                    return self.routing_cache[key]
                else:
                    # Expired entry
                    del self.routing_cache[key]
                    if key in self.cache_expiry:
                        del self.cache_expiry[key]
                    await self._save_cache()
            return None

    async def set_routing_info(self, key: str, value: Any, ttl_minutes: int = 60) -> None:
        """Set routing information in cache with TTL."""
        async with self._lock:
            self.routing_cache[key] = value
            self.cache_expiry[key] = datetime.now() + timedelta(minutes=ttl_minutes)
            await self._save_cache()

    async def flush_routing_cache(self) -> bool:
        """Flush all routing cache entries."""
        try:
            async with self._lock:
                # Clear in-memory cache
                self.routing_cache.clear()
                self.cache_expiry.clear()

                # Remove cache file if it exists
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)

                logger.info("Routing cache flushed successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to flush routing cache: {e}")
            return False

    async def cleanup_expired_entries(self) -> int:
        """Clean up expired cache entries. Returns number of entries removed."""
        async with self._lock:
            await self._load_cache()
            now = datetime.now()
            expired_keys = [
                key for key, expiry in self.cache_expiry.items()
                if expiry < now
            ]

            for key in expired_keys:
                del self.routing_cache[key]
                del self.cache_expiry[key]

            if expired_keys:
                await self._save_cache()

            return len(expired_keys)