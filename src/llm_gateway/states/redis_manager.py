import json
import os
from contextlib import contextmanager
from typing import Any, cast

import redis


class RedisManager:
    """Generic Redis manager for shared state across multiple workers"""

    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis:
        """Get or create Redis client"""
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
            # Enable keyspace notifications for expired keys
            try:
                self._client.config_set("notify-keyspace-events", "Ex")
            except redis.ResponseError:
                # Redis might not allow config changes, log but continue
                print("Warning: Could not enable Redis keyspace notifications")
        return self._client

    def get_state(self, namespace: str, key: str) -> dict | None:
        """Get state for a specific namespace and key"""
        redis_key = f"{namespace}:{key}"
        data = cast(str, self.client.get(redis_key))
        if data:
            return json.loads(data)
        return None

    def set_state(
        self, namespace: str, key: str, state: dict, expiry_seconds: int = 600
    ) -> bool:
        """Set state with expiry"""
        redis_key = f"{namespace}:{key}"
        return cast(bool, self.client.setex(redis_key, expiry_seconds, json.dumps(state)))

    def extend_expiry(
        self, namespace: str, key: str, expiry_seconds: int = 600
    ) -> bool:
        """Extend state expiry time"""
        redis_key = f"{namespace}:{key}"
        return cast(bool, self.client.expire(redis_key, expiry_seconds))

    def delete_state(self, namespace: str, key: str) -> bool:
        """Delete state"""
        redis_key = f"{namespace}:{key}"
        return bool(self.client.delete(redis_key))

    def get_all_keys(self, namespace: str) -> list[str]:
        """Get all keys for a namespace"""
        pattern = f"{namespace}:*"
        keys = cast(list[str], self.client.keys(pattern))
        return [
            key.split(":", 1)[1] for key in keys if ":" in key and ":lock:" not in key
        ]

    def subscribe_to_expirations(self, namespace: str):
        """Subscribe to key expiration events for a namespace"""
        pubsub = self.client.pubsub()
        # Subscribe to expired events for all keys in the database
        pubsub.psubscribe("__keyevent@0__:expired")
        return pubsub

    @contextmanager
    def distributed_lock(self, namespace: str, key: str, timeout: int = 30) -> Any:
        """Distributed lock for operations"""
        lock_key = f"{namespace}:lock:{key}"
        lock = self.client.lock(lock_key, timeout=timeout)
        try:
            acquired = lock.acquire(blocking=True, blocking_timeout=timeout)
            if not acquired:
                raise TimeoutError(f"Could not acquire lock for {namespace}:{key}")
            yield
        finally:
            try:
                lock.release()
            except Exception:
                pass  # Lock already released or expired

    def health_check(self) -> bool:
        """Check Redis connectivity"""
        try:
            self.client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            return False
