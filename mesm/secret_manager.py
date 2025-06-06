from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import threading
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from enum import StrEnum, auto
from functools import cached_property, wraps
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    NoReturn,
    ParamSpec,
    Protocol,
    Self,
    TypedDict,
    TypeVar,
    cast,
    final,
    override,
)
from weakref import WeakSet

import structlog
from cachetools import TTLCache
from google.api_core.exceptions import GoogleAPICallError, GoogleAPIError
from google.auth import default as google_auth_default
from google.auth.credentials import Credentials as GoogleAuthCredentials
from google.cloud.secretmanager_v1 import (
    SecretManagerServiceAsyncClient,
    SecretManagerServiceClient,
)
from google.oauth2 import service_account
from prometheus_client import Counter, Histogram
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random,
)

__all__ = [
    "MultiEnvironmentSecretManager",
    "EnvironmentSpecificSecretManager",
    "SecretConfig",
    "CredentialConfig",
    "EnvironmentConfig",
    "MultiEnvironmentSecretManagerConfig",
    "CredentialMethod",
    "SecretManagerError",
    "SecretNotFoundError",
    "SecretAccessError",
    "ConfigurationError",
    "environment_context",
]

type GoogleCredentials = service_account.Credentials | GoogleAuthCredentials
type PathLike = str | Path
type JSONData = dict[str, Any]
type SecretValue = str | bytes
type ExcInfo = tuple[type[BaseException], BaseException, TracebackType | None] | None
type EnvironmentName = Literal["development", "staging", "production", "test"] | str
type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
type CircuitBreakerState = Literal["CLOSED", "OPEN", "HALF_OPEN"]

P = ParamSpec("P")
T = TypeVar("T")
_ClientT = TypeVar(
    "_ClientT", bound=SecretManagerServiceClient | SecretManagerServiceAsyncClient
)
_PoolItemT = TypeVar("_PoolItemT")

# Constants
MAX_RETRIES: Final[int] = 5
WAIT_MULTIPLIER: Final[float] = 1.0
MIN_WAIT_SECONDS: Final[float] = 2.0
MAX_WAIT_SECONDS: Final[float] = 60.0
DEFAULT_TIMEOUT: Final[float] = 30.0
MAX_SECRET_SIZE_BYTES: Final[int] = 65536  # 64KB GCP limit
HEALTH_CHECK_TIMEOUT: Final[float] = 5.0
DEFAULT_ENVIRONMENT: Final[EnvironmentName] = "production"
DEFAULT_CACHE_TTL: Final[int] = 300
DEFAULT_CACHE_SIZE: Final[int] = 1000
DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 3
DEFAULT_CIRCUIT_BREAKER_TIMEOUT: Final[float] = 30.0
DEFAULT_LOG_LEVEL: Final[int] = 20  # INFO level

RETRYABLE_ERRORS: Final[tuple[type[BaseException], ...]] = (
    GoogleAPIError,
    ConnectionError,
    OSError,
    TimeoutError,
)


# protocols with type safety
class Closeable(Protocol):
    """Protocol for resources that can be closed."""

    def close(self) -> None:
        """Close the resource."""
        ...


class AsyncCloseable(Protocol):
    """Protocol for async resources that can be closed."""

    async def close(self) -> None:
        """Close the resource asynchronously."""
        ...


class LoggingConfig:
    """Centralized logging configuration with thread safety."""

    _configured: bool = False
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def configure(cls, *, force_reconfigure: bool = False) -> None:
        """Configure structured logging with error handling."""
        if cls._configured and not force_reconfigure:
            return

        with cls._lock:
            if cls._configured and not force_reconfigure:
                return

            try:
                log_level = int(os.getenv("LOG_LEVEL", str(DEFAULT_LOG_LEVEL)))
            except (ValueError, TypeError):
                log_level = DEFAULT_LOG_LEVEL

            # renderer selection with fallback
            try:
                renderer = (
                    structlog.dev.ConsoleRenderer(colors=True)
                    if sys.stderr.isatty()
                    else structlog.processors.JSONRenderer()
                )
            except (OSError, AttributeError):
                renderer = structlog.processors.JSONRenderer()

            processors: list[structlog.types.Processor] = [
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                    ]
                ),
                structlog.processors.dict_tracebacks,
                renderer,
            ]

            structlog.configure(
                processors=processors,
                context_class=dict,
                wrapper_class=structlog.make_filtering_bound_logger(log_level),
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

            cls._configured = True


# context variables with defaults
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
environment_var: ContextVar[EnvironmentName] = ContextVar(
    "environment", default=DEFAULT_ENVIRONMENT
)

# metrics with buckets
secret_operations: Final[Counter] = Counter(
    "secret_manager_operations_total",
    "Total number of secret manager operations",
    ["operation", "project_id", "environment", "status", "error_type"],
)

secret_operation_duration: Final[Histogram] = Histogram(
    "secret_manager_operation_duration_seconds",
    "Duration of secret manager operations",
    ["operation", "project_id", "environment"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)


# custom exceptions with error context
class SecretManagerError(Exception):
    """Base exception for secret manager errors with context."""

    def __init__(
        self,
        message: str,
        *,
        secret_config: SecretConfig | None = None,
        original_error: Exception | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.secret_config = secret_config
        self.original_error = original_error
        self.error_code = error_code

    @override
    def __str__(self) -> str:
        """string representation with context."""
        base_msg = super().__str__()
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.secret_config:
            base_msg += f" (secret: {self.secret_config.secret_path})"
        return base_msg


class SecretNotFoundError(SecretManagerError):
    """Secret not found error."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, error_code="SECRET_NOT_FOUND", **kwargs)


class SecretAccessError(SecretManagerError):
    """Error accessing secret."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, error_code="SECRET_ACCESS_ERROR", **kwargs)


class ConfigurationError(SecretManagerError):
    """Configuration-related error."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, error_code="CONFIGURATION_ERROR", **kwargs)


# TypedDict definitions with structure
class CacheMetadata(TypedDict):
    """Metadata for cached secrets."""

    accessed_at: datetime
    version: str
    size_bytes: int
    encoding: str
    project_id: str
    secret_name: str
    ttl_seconds: int
    access_count: int


class CacheEntry(TypedDict):
    """Cache entry structure."""

    value: str
    metadata: CacheMetadata
    expires_at: datetime
    created_at: datetime


class PoolStats(TypedDict):
    """Connection pool statistics."""

    pooled: int
    in_use: int
    max_size: int
    created_total: int
    released_total: int
    errors_total: int


class CircuitBreakerStats(TypedDict):
    """Circuit breaker statistics."""

    state: CircuitBreakerState
    failure_count: int
    success_count: int
    last_failure_time: str | None
    last_success_time: str | None
    next_attempt_time: str | None


# metrics decorator with error categorization
def with_metrics(operation: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add metrics to operations."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            secret_config: SecretConfig | None = None
            if len(args) >= 2 and hasattr(args[1], "project_id"):
                secret_config = cast(SecretConfig, args[1])

            environment = environment_var.get()
            project_id = secret_config.project_id if secret_config else "unknown"

            start_time = datetime.now(UTC)
            try:
                with secret_operation_duration.labels(
                    operation, project_id, environment
                ).time():
                    result = func(*args, **kwargs)

                secret_operations.labels(
                    operation, project_id, environment, "success", "none"
                ).inc()
                return result
            except Exception as e:
                error_type = type(e).__name__
                secret_operations.labels(
                    operation, project_id, environment, "error", error_type
                ).inc()

                duration = (datetime.now(UTC) - start_time).total_seconds()
                LoggingConfig.configure()
                logger = structlog.get_logger(__name__)
                logger.error(
                    "operation_failed",
                    operation=operation,
                    duration_seconds=duration,
                    error_type=error_type,
                    project_id=project_id,
                    environment=environment,
                )
                raise

        return wrapper

    return decorator


# SecretCache with performance and monitoring
class SecretCache:
    """Thread-safe TTL cache for secrets with performance."""

    __slots__: tuple[str, ...] = (
        "_cache",
        "_sync_lock",
        "_async_lock",
        "_ttl",
        "_max_size",
        "_stats",
        "_logger",
    )

    def __init__(self, ttl_seconds: int, max_size: int) -> None:
        self._cache: TTLCache[str, CacheEntry] = TTLCache(
            maxsize=max_size, ttl=ttl_seconds
        )
        self._sync_lock: threading.RLock = threading.RLock()
        self._async_lock: asyncio.Lock = asyncio.Lock()
        self._ttl: int = ttl_seconds
        self._max_size: int = max_size
        self._stats: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "errors": 0,
        }

        LoggingConfig.configure()
        self._logger = structlog.get_logger(__name__).bind(component="SecretCache")

    def get(self, key: str) -> CacheEntry | None:
        """Get entry from cache (sync) with stats tracking."""
        try:
            with self._sync_lock:
                entry = self._cache.get(key)
                if entry and entry["expires_at"] > datetime.now(UTC):
                    self._stats["hits"] += 1
                    entry["metadata"]["access_count"] += 1
                    return entry

                if entry:
                    self._stats["evictions"] += 1

                self._stats["misses"] += 1
                return None
        except Exception as e:
            self._stats["errors"] += 1
            self._logger.error("cache_get_error", key=key, error=str(e))
            return None

    async def async_get(self, key: str) -> CacheEntry | None:
        """Get entry from cache (async) with stats tracking."""
        try:
            async with self._async_lock:
                entry = self._cache.get(key)
                if entry and entry["expires_at"] > datetime.now(UTC):
                    self._stats["hits"] += 1
                    entry["metadata"]["access_count"] += 1
                    return entry

                if entry:
                    self._stats["evictions"] += 1

                self._stats["misses"] += 1
                return None
        except Exception as e:
            self._stats["errors"] += 1
            self._logger.error("cache_async_get_error", key=key, error=str(e))
            return None

    def set(self, key: str, value: CacheEntry) -> None:
        """Set entry in cache (sync) with validation."""
        try:
            with self._sync_lock:
                if not value["value"] or len(value["value"]) > MAX_SECRET_SIZE_BYTES:
                    self._logger.warning(
                        "cache_set_invalid_size",
                        key=key,
                        size=len(value["value"]) if value["value"] else 0,
                    )
                    return

                value["created_at"] = datetime.now(UTC)
                self._cache[key] = value
        except Exception as e:
            self._stats["errors"] += 1
            self._logger.error("cache_set_error", key=key, error=str(e))

    async def async_set(self, key: str, value: CacheEntry) -> None:
        """Set entry in cache (async) with validation."""
        try:
            async with self._async_lock:
                if not value["value"] or len(value["value"]) > MAX_SECRET_SIZE_BYTES:
                    self._logger.warning(
                        "cache_async_set_invalid_size",
                        key=key,
                        size=len(value["value"]) if value["value"] else 0,
                    )
                    return

                value["created_at"] = datetime.now(UTC)
                self._cache[key] = value
        except Exception as e:
            self._stats["errors"] += 1
            self._logger.error("cache_async_set_error", key=key, error=str(e))

    def clear(self) -> None:
        """Clear all cache entries with stats reset."""
        with self._sync_lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "errors": 0,
            }
            self._logger.info("cache_cleared", cleared_count=cleared_count)

    def get_stats(self) -> dict[str, int | float]:
        """Get comprehensive cache statistics."""
        with self._sync_lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "errors": self._stats["errors"],
                "hit_rate": (
                    self._stats["hits"] / total_requests if total_requests > 0 else 0.0
                ),
                "error_rate": (
                    self._stats["errors"] / total_requests
                    if total_requests > 0
                    else 0.0
                ),
            }


# CircuitBreaker with improved state management
class CircuitBreaker:
    """Thread-safe circuit breaker with improved state management."""

    __slots__ = (
        "_failure_threshold",
        "_recovery_timeout",
        "_expected_exception",
        "_failure_count",
        "_success_count",
        "_state",
        "_last_failure_time",
        "_last_success_time",
        "_sync_lock",
        "_async_lock",
        "_logger",
    )

    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: float,
        expected_exception: type[BaseException] | tuple[type[BaseException], ...],
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._expected_exception = expected_exception

        self._failure_count = 0
        self._success_count = 0
        self._state: CircuitBreakerState = "CLOSED"
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None
        self._sync_lock = threading.RLock()
        self._async_lock = asyncio.Lock()

        LoggingConfig.configure()
        self._logger = structlog.get_logger(__name__).bind(component="CircuitBreaker")

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open with automatic state transitions."""
        with self._sync_lock:
            return self._check_and_update_state()

    async def async_is_open(self) -> bool:
        """Check if circuit breaker is open (async)."""
        async with self._async_lock:
            return self._check_and_update_state()

    def _check_and_update_state(self) -> bool:
        """Internal state check logic with automatic recovery."""
        current_time = datetime.now(UTC)

        if self._state == "OPEN":
            if self._last_failure_time and (
                current_time - self._last_failure_time
            ) > timedelta(seconds=self._recovery_timeout):
                self._state = "HALF_OPEN"
                self._logger.info(
                    "circuit_breaker_half_open",
                    failure_count=self._failure_count,
                    recovery_timeout=self._recovery_timeout,
                )
                return False
            return True
        return False

    def record_failure(self) -> None:
        """Record a failure with state transitions."""
        with self._sync_lock:
            self._record_failure_internal()

    async def async_record_failure(self) -> None:
        """Record a failure (async)."""
        async with self._async_lock:
            self._record_failure_internal()

    def _record_failure_internal(self) -> None:
        """Internal failure recording logic."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(UTC)

        if self._state == "HALF_OPEN":
            self._state = "OPEN"
            self._logger.warning(
                "circuit_breaker_opened_from_half_open",
                failure_count=self._failure_count,
            )
        elif self._failure_count >= self._failure_threshold and self._state == "CLOSED":
            self._state = "OPEN"
            self._logger.warning(
                "circuit_breaker_opened",
                failure_count=self._failure_count,
                threshold=self._failure_threshold,
            )

    def record_success(self) -> None:
        """Record a success with state reset."""
        with self._sync_lock:
            self._record_success_internal()

    async def async_record_success(self) -> None:
        """Record a success (async)."""
        async with self._async_lock:
            self._record_success_internal()

    def _record_success_internal(self) -> None:
        """Internal success recording logic."""
        self._success_count += 1
        self._last_success_time = datetime.now(UTC)

        if self._state == "HALF_OPEN":
            self._state = "CLOSED"
            self._failure_count = 0
            self._logger.info(
                "circuit_breaker_closed",
                success_count=self._success_count,
            )
        elif self._state == "CLOSED":
            self._failure_count = max(0, self._failure_count - 1)

    def get_state(self) -> CircuitBreakerStats:
        """Get comprehensive circuit breaker state."""
        with self._sync_lock:
            next_attempt_time = None
            if self._state == "OPEN" and self._last_failure_time:
                next_attempt = self._last_failure_time + timedelta(
                    seconds=self._recovery_timeout
                )
                next_attempt_time = next_attempt.isoformat()

            return CircuitBreakerStats(
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                last_failure_time=(
                    self._last_failure_time.isoformat()
                    if self._last_failure_time
                    else None
                ),
                last_success_time=(
                    self._last_success_time.isoformat()
                    if self._last_success_time
                    else None
                ),
                next_attempt_time=next_attempt_time,
            )


# ConnectionPool with improved resource management
class ConnectionPool(Generic[_PoolItemT]):
    """Thread-safe connection pool with improved resource management."""

    __slots__ = (
        "_factory",
        "_max_size",
        "_enable_pooling",
        "_pool_items",
        "_in_use_count",
        "_sync_lock",
        "_async_lock",
        "_created_count",
        "_released_count",
        "_error_count",
        "_logger",
        "_closed",
    )

    def __init__(
        self,
        factory: Callable[[], _PoolItemT],
        max_size: int,
        enable: bool = True,
    ) -> None:
        self._factory: Callable[[], _PoolItemT] = factory
        self._max_size: int = max_size
        self._enable_pooling: bool = enable

        self._pool_items: list[_PoolItemT] = []
        self._in_use_count: int = 0
        self._created_count: int = 0
        self._released_count: int = 0
        self._error_count: int = 0
        self._closed: bool = False
        self._sync_lock: threading.RLock = threading.RLock()
        self._async_lock: asyncio.Lock = asyncio.Lock()

        LoggingConfig.configure()
        self._logger: structlog.BoundLogger = structlog.get_logger(__name__).bind(
            component="ConnectionPool"
        )

    def _check_not_closed(self) -> None:
        """Check if pool is not closed."""
        if self._closed:
            raise RuntimeError(
                f"Connection pool is closed (created: {self._created_count}, "
                + f"released: {self._released_count})"
            )

    def acquire(self) -> _PoolItemT:
        """Acquire connection from pool (sync)."""
        self._check_not_closed()

        if not self._enable_pooling:
            try:
                self._created_count += 1
                return self._factory()
            except Exception as e:
                self._error_count += 1
                self._logger.error("factory_error_sync", error=str(e))
                raise

        with self._sync_lock:
            if self._pool_items:
                item = self._pool_items.pop()
                self._in_use_count += 1
                self._logger.debug(
                    "connection_acquired_from_pool",
                    pooled=len(self._pool_items),
                    in_use=self._in_use_count,
                )
                return item

            total_connections = self._in_use_count + len(self._pool_items)
            if total_connections < self._max_size:
                try:
                    self._in_use_count += 1
                    self._created_count += 1
                    item = self._factory()
                    self._logger.debug(
                        "connection_created",
                        created_total=self._created_count,
                        in_use=self._in_use_count,
                    )
                    return item
                except Exception as e:
                    self._error_count += 1
                    self._in_use_count = max(0, self._in_use_count - 1)
                    self._logger.error("factory_error_sync", error=str(e))
                    raise

            self._logger.warning(
                "connection_pool_exhausted",
                max_size=self._max_size,
                in_use=self._in_use_count,
                pooled=len(self._pool_items),
            )
            try:
                self._created_count += 1
                return self._factory()
            except Exception as e:
                self._error_count += 1
                self._logger.error("factory_error_exhausted_sync", error=str(e))
                raise

    def release(self, item: _PoolItemT) -> None:
        """Release connection back to pool (sync)."""
        self._released_count += 1

        if self._closed:
            self._close_item(item)
            return

        if not self._enable_pooling:
            self._close_item(item)
            return

        with self._sync_lock:
            if len(self._pool_items) < self._max_size and self._in_use_count > 0:
                self._pool_items.append(item)
                self._in_use_count = max(0, self._in_use_count - 1)
                self._logger.debug(
                    "connection_returned_to_pool",
                    pooled=len(self._pool_items),
                    in_use=self._in_use_count,
                )
            else:
                self._close_item(item)
                if self._in_use_count > 0:
                    self._in_use_count -= 1

    async def async_acquire(self) -> _PoolItemT:
        """Acquire connection from pool (async)."""
        self._check_not_closed()

        if not self._enable_pooling:
            try:
                self._created_count += 1
                return self._factory()
            except Exception as e:
                self._error_count += 1
                self._logger.error("factory_error_async", error=str(e))
                raise

        async with self._async_lock:
            if self._pool_items:
                item = self._pool_items.pop()
                self._in_use_count += 1
                self._logger.debug(
                    "connection_acquired_from_pool_async",
                    pooled=len(self._pool_items),
                    in_use=self._in_use_count,
                )
                return item

            total_connections = self._in_use_count + len(self._pool_items)

            if total_connections < self._max_size:
                try:
                    self._in_use_count += 1
                    self._created_count += 1
                    item = self._factory()
                    self._logger.debug(
                        "connection_created_async",
                        created_total=self._created_count,
                        in_use=self._in_use_count,
                    )
                    return item
                except Exception as e:
                    self._error_count += 1
                    self._in_use_count = max(0, self._in_use_count - 1)
                    self._logger.error("factory_error_async", error=str(e))
                    raise

            self._logger.warning(
                "connection_pool_exhausted_async",
                max_size=self._max_size,
                in_use=self._in_use_count,
                pooled=len(self._pool_items),
            )
            try:
                self._created_count += 1
                return self._factory()
            except Exception as e:
                self._error_count += 1
                self._logger.error("factory_error_exhausted_async", error=str(e))
                raise

    async def async_release(self, item: _PoolItemT) -> None:
        """Release connection back to pool (async)."""
        self._released_count += 1

        if self._closed:
            await self._safe_close_item_async(item)
            return

        if not self._enable_pooling:
            await self._safe_close_item_async(item)
            return

        async with self._async_lock:
            if len(self._pool_items) < self._max_size and self._in_use_count > 0:
                self._pool_items.append(item)
                self._in_use_count = max(0, self._in_use_count - 1)
                self._logger.debug(
                    "connection_returned_to_pool_async",
                    pooled=len(self._pool_items),
                    in_use=self._in_use_count,
                )
            else:
                await self._safe_close_item_async(item)
                if self._in_use_count > 0:
                    self._in_use_count -= 1

    def _close_item(self, item: _PoolItemT) -> None:
        """Close an item if it has a close method."""
        if hasattr(item, "close") and callable(getattr(item, "close")):
            try:
                cast(Closeable, item).close()
            except Exception as e:
                self._logger.warning("item_close_error", error=str(e))

    async def _safe_close_item_async(self, item: _PoolItemT) -> None:
        """Safely close an async item."""
        if hasattr(item, "close") and callable(getattr(item, "close")):
            try:
                close_method = getattr(item, "close")
                if asyncio.iscoroutinefunction(close_method):
                    await close_method()
                else:
                    close_method()
            except Exception as e:
                self._logger.warning("async_item_close_error", error=str(e))

    def close(self) -> None:
        """Close all connections in pool."""
        with self._sync_lock:
            if self._closed:
                return

            self._closed = True
            items_to_close = list(self._pool_items)
            self._pool_items.clear()
            in_use_before_close = self._in_use_count
            self._in_use_count = 0

        for item in items_to_close:
            self._close_item(item)

        self._logger.info(
            "connection_pool_closed",
            closed_count=len(items_to_close),
            in_use_before_close=in_use_before_close,
            created_total=self._created_count,
            released_total=self._released_count,
            error_total=self._error_count,
        )

    def get_stats(self) -> PoolStats:
        """Get comprehensive pool statistics."""
        with self._sync_lock:
            return PoolStats(
                pooled=len(self._pool_items),
                in_use=self._in_use_count,
                max_size=self._max_size,
                created_total=self._created_count,
                released_total=self._released_count,
                errors_total=self._error_count,
            )


# Configuration Models with validation
class CredentialMethod(StrEnum):
    """Credential method types."""

    APPLICATION_DEFAULT = auto()
    SERVICE_ACCOUNT_FILE = auto()
    SERVICE_ACCOUNT_JSON = auto()
    IMPERSONATION = auto()


class CredentialConfig(BaseModel):
    """credential configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        use_enum_values=True,
    )

    method: CredentialMethod = Field(default=CredentialMethod.APPLICATION_DEFAULT)
    service_account_path: Path | None = Field(default=None)
    service_account_json: JSONData | None = Field(default=None)
    impersonate_service_account: str | None = Field(default=None)
    scopes: list[str] = Field(
        default_factory=lambda: ["https://www.googleapis.com/auth/cloud-platform"]
    )
    quota_project_id: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_credential_config(self) -> Self:
        """credential configuration validation."""
        match self.method:
            case CredentialMethod.SERVICE_ACCOUNT_FILE:
                if not self.service_account_path:
                    raise ValueError(
                        "service_account_path required for SERVICE_ACCOUNT_FILE method"
                    )
                if not self.service_account_path.exists():
                    raise FileNotFoundError(
                        f"Service account file not found: {self.service_account_path}"
                    )
                if not self.service_account_path.is_file():
                    raise ValueError(
                        f"Service account path is not a file: {self.service_account_path}"
                    )
                try:
                    with self.service_account_path.open("r") as f:
                        json.load(f)
                except (json.JSONDecodeError, PermissionError) as e:
                    raise ValueError(
                        f"Invalid or unreadable service account file: {e}"
                    ) from e

            case CredentialMethod.SERVICE_ACCOUNT_JSON:
                if not self.service_account_json:
                    raise ValueError(
                        "service_account_json required for SERVICE_ACCOUNT_JSON method"
                    )
                required_fields = {
                    "type",
                    "project_id",
                    "private_key",
                    "client_email",
                    "private_key_id",
                }
                if not required_fields.issubset(self.service_account_json.keys()):
                    missing = required_fields - self.service_account_json.keys()
                    raise ValueError(
                        f"Missing required fields in service_account_json: {missing}"
                    )
                if self.service_account_json.get("type") != "service_account":
                    raise ValueError(
                        "service_account_json must be of type 'service_account'"
                    )

            case CredentialMethod.IMPERSONATION:
                if not self.impersonate_service_account:
                    raise ValueError(
                        "impersonate_service_account required for IMPERSONATION method"
                    )
                email = self.impersonate_service_account
                if "@" not in email or not email.endswith(".iam.gserviceaccount.com"):
                    raise ValueError(
                        "impersonate_service_account must be a valid service account email"
                    )

        if not self.scopes:
            raise ValueError("At least one scope must be specified")

        return self

    def get_credentials(self) -> GoogleCredentials:
        """Get Google credentials based on configuration."""
        LoggingConfig.configure()
        logger = structlog.get_logger(__name__).bind(component="CredentialConfig")

        try:
            match self.method:
                case CredentialMethod.SERVICE_ACCOUNT_FILE:
                    logger.debug(
                        "loading_file_credentials", path=str(self.service_account_path)
                    )
                    assert self.service_account_path
                    return service_account.Credentials.from_service_account_file(  # type: ignore
                        str(self.service_account_path),
                        scopes=self.scopes,
                        quota_project_id=self.quota_project_id,
                    )

                case CredentialMethod.SERVICE_ACCOUNT_JSON:
                    logger.debug("loading_json_credentials")
                    assert self.service_account_json
                    return service_account.Credentials.from_service_account_info(  # type: ignore
                        self.service_account_json,
                        scopes=self.scopes,
                        quota_project_id=self.quota_project_id,
                    )

                case CredentialMethod.IMPERSONATION:
                    logger.debug(
                        "loading_impersonation_credentials",
                        target=self.impersonate_service_account,
                    )
                    assert self.impersonate_service_account
                    source_credentials, _ = google_auth_default(  # type: ignore
                        scopes=self.scopes, quota_project_id=self.quota_project_id
                    )

                    from google.auth import impersonated_credentials

                    return impersonated_credentials.Credentials(  # type: ignore
                        source_credentials=source_credentials,
                        target_principal=self.impersonate_service_account,
                        target_scopes=self.scopes,
                        lifetime=3600,
                        quota_project_id=self.quota_project_id,
                    )

                case _:
                    logger.debug("loading_application_default_credentials")
                    credentials, _ = google_auth_default(  # type: ignore
                        scopes=self.scopes, quota_project_id=self.quota_project_id
                    )
                    return credentials

        except Exception as e:
            logger.error(
                "credential_loading_failed",
                method=self.method,
                error=str(e),
                exc_info=True,
            )
            raise ConfigurationError(
                f"Failed to load credentials using method {self.method}: {e}"
            ) from e


class EnvironmentConfig(BaseModel):
    """environment-specific configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True,
    )

    name: str = Field(..., examples=["development", "staging", "production"])
    default_credential: CredentialConfig = Field(default_factory=CredentialConfig)
    project_credentials: dict[str, CredentialConfig] = Field(default_factory=dict)
    allowed_projects: list[str] | None = Field(default=None)
    cache_ttl_seconds: int = Field(default=DEFAULT_CACHE_TTL, ge=0, le=86400)
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT, ge=1.0, le=300.0)
    max_retries: int = Field(default=MAX_RETRIES, ge=1, le=10)
    circuit_breaker_threshold: int = Field(
        default=DEFAULT_CIRCUIT_BREAKER_THRESHOLD, ge=1, le=20
    )
    circuit_breaker_timeout: float = Field(
        default=DEFAULT_CIRCUIT_BREAKER_TIMEOUT, ge=1.0, le=300.0
    )

    @classmethod
    @field_validator("name")
    def validate_environment_name(cls, v: str) -> str:
        """environment name validation."""
        if not v.strip():
            raise ValueError("Environment name cannot be empty")

        if not v.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError(
                "Environment name must contain only alphanumeric characters, "
                + "hyphens, underscores, and dots"
            )

        if len(v) > 50:
            raise ValueError("Environment name cannot exceed 50 characters")

        return v.lower()

    @classmethod
    @field_validator("allowed_projects")
    def validate_allowed_projects(cls, v: list[str] | None) -> list[str] | None:
        """Validate project IDs in allowed_projects."""
        if v is None:
            return v

        for project_id in v:
            if not project_id.strip():
                raise ValueError("Project ID cannot be empty")
            if not project_id.replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"Invalid project ID format: {project_id}")

        return v

    def get_credential_for_project(self, project_id: str) -> CredentialConfig:
        """Get credential configuration for a specific project."""
        if not project_id.strip():
            raise ValueError("Project ID cannot be empty")

        if self.allowed_projects and project_id not in self.allowed_projects:
            raise ValueError(
                f"Project {project_id} not allowed in environment {self.name}. "
                + f"Allowed projects: {self.allowed_projects}"
            )
        return self.project_credentials.get(project_id, self.default_credential)


class SecretConfig(BaseModel):
    """secret access configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
    )

    project_id: str = Field(
        ..., min_length=1, max_length=30, pattern=r"^[a-z][a-z0-9\-]*[a-z0-9]$"
    )
    secret_name: str = Field(
        ..., min_length=1, max_length=255, pattern=r"^[a-zA-Z][a-zA-Z0-9_\-]*$"
    )
    secret_version: str | int | None = Field(
        default="latest", union_mode="left_to_right"
    )
    environment: EnvironmentName | None = Field(default=None)
    credential_override: CredentialConfig | None = Field(default=None)

    @classmethod
    @field_validator("project_id")
    def validate_project_id(cls, v: str) -> str:
        """project ID validation."""
        if not v.strip():
            raise ValueError("Project ID cannot be empty")

        if v.startswith("-") or v.endswith("-"):
            raise ValueError("Project ID cannot start or end with hyphen")

        if "--" in v:
            raise ValueError("Project ID cannot contain consecutive hyphens")

        return v.lower()

    @classmethod
    @field_validator("secret_name")
    def validate_secret_name(cls, v: str) -> str:
        """secret name validation."""
        if not v.strip():
            raise ValueError("Secret name cannot be empty")

        reserved_names = {"latest", "versions"}
        if v.lower() in reserved_names:
            raise ValueError(f"Secret name cannot be a reserved word: {reserved_names}")

        return v

    @classmethod
    @field_validator("secret_version", mode="after")
    def validate_version_format(cls, v: str | int) -> str:
        """secret version validation."""
        if isinstance(v, int):
            if v < 1:
                raise ValueError("Version number must be positive")
            if v > 999999:
                raise ValueError("Version number too large")
            return str(v)

        if not v or not v.strip():
            raise ValueError("Secret version cannot be empty")

        if v.strip().lower() == "latest":
            return "latest"

        try:
            version_num = int(v.strip())
            if version_num < 1:
                raise ValueError("Version number must be positive")
            if version_num > 999999:
                raise ValueError("Version number too large")
            return str(version_num)
        except ValueError as e:
            if "positive" in str(e) or "large" in str(e):
                raise
            raise ValueError(
                "Secret version must be 'latest' or a positive integer"
            ) from e

    @cached_property
    def secret_path(self) -> str:
        """Get full secret path."""
        return f"projects/{self.project_id}/secrets/{self.secret_name}/versions/{self.secret_version}"

    @cached_property
    def cache_key(self) -> str:
        """Get cache key for this secret."""
        env = self.environment or environment_var.get()
        return f"{env}:{self.project_id}:{self.secret_name}:{self.secret_version}"


class MultiEnvironmentSecretManagerConfig(BaseModel):
    """multi-environment secret manager configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        validate_assignment=True, extra="forbid"
    )

    environments: dict[str, EnvironmentConfig] = Field(default_factory=dict)
    default_environment: EnvironmentName = Field(default=DEFAULT_ENVIRONMENT)
    enable_caching: bool = Field(default=True)
    enable_connection_pooling: bool = Field(default=True)
    max_connections_per_credential: int = Field(default=5, ge=1, le=50)
    strict_environment_isolation: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=DEFAULT_CACHE_TTL, ge=0, le=86400)
    cache_max_size: int = Field(default=DEFAULT_CACHE_SIZE, ge=1, le=10000)
    global_timeout_seconds: float = Field(default=DEFAULT_TIMEOUT, ge=1.0, le=300.0)
    enable_metrics: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_default_environment(self) -> Self:
        """Ensure default environment exists."""
        if self.default_environment not in self.environments:
            self.environments[self.default_environment] = EnvironmentConfig(
                name=self.default_environment
            )
        return self

    @classmethod
    @field_validator("environments")
    def validate_environments(
        cls, v: dict[str, EnvironmentConfig]
    ) -> dict[str, EnvironmentConfig]:
        """Validate environment configurations."""
        if not v:
            raise ValueError("At least one environment must be configured")

        for env_name, env_config in v.items():
            if env_name != env_config.name:
                raise ValueError(
                    f"Environment key '{env_name}' does not match config name '{env_config.name}'"
                )

        return v

    @classmethod
    def from_yaml(cls, path: PathLike) -> Self:
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load configuration from YAML. "
                + "Install with: pip install PyYAML"
            ) from e

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Configuration path is not a file: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}") from e
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot read configuration file {file_path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid YAML structure in {file_path}: expected dict, got {type(data)}"
            )

        try:
            return cls.model_validate(data)
        except Exception as e:
            raise ConfigurationError(
                f"Invalid configuration in {file_path}: {e}"
            ) from e

    @classmethod
    def from_env(cls) -> Self:
        """Load configuration from environment variables."""
        LoggingConfig.configure()
        logger = structlog.get_logger(__name__).bind(component="ConfigFromEnv")

        config = cls()

        detected_envs: list[str] = []
        for env_name in ["development", "staging", "production", "test"]:
            env_prefix = f"{env_name.upper()}_"
            sa_path_str = os.getenv(f"{env_prefix}SERVICE_ACCOUNT_PATH")
            sa_json_str = os.getenv(f"{env_prefix}SERVICE_ACCOUNT_JSON")
            impersonate_str = os.getenv(f"{env_prefix}IMPERSONATE_SERVICE_ACCOUNT")

            if sa_path_str or sa_json_str or impersonate_str:
                detected_envs.append(env_name)
                cred_config_data: dict[str, Any] = {
                    "method": CredentialMethod.APPLICATION_DEFAULT
                }

                if sa_path_str:
                    try:
                        sa_path = Path(sa_path_str)
                        if not sa_path.exists():
                            logger.warning(
                                "service_account_file_not_found",
                                env=env_name,
                                path=sa_path_str,
                            )
                            continue

                        cred_config_data.update(
                            {
                                "method": CredentialMethod.SERVICE_ACCOUNT_FILE,
                                "service_account_path": sa_path,
                            }
                        )
                    except Exception as e:
                        logger.error(
                            "invalid_service_account_path",
                            env=env_name,
                            path=sa_path_str,
                            error=str(e),
                        )
                        continue

                elif sa_json_str:
                    try:
                        sa_json = json.loads(sa_json_str)
                        cred_config_data.update(
                            {
                                "service_account_json": sa_json,
                                "method": CredentialMethod.SERVICE_ACCOUNT_JSON,
                            }
                        )
                    except json.JSONDecodeError as e:
                        logger.error(
                            "invalid_sa_json_env_var",
                            env_var=f"{env_prefix}SERVICE_ACCOUNT_JSON",
                            error=str(e),
                        )
                        continue

                elif impersonate_str:
                    cred_config_data.update(
                        {
                            "method": CredentialMethod.IMPERSONATION,
                            "impersonate_service_account": impersonate_str,
                        }
                    )

                quota_project = os.getenv(f"{env_prefix}QUOTA_PROJECT_ID")
                if quota_project:
                    cred_config_data["quota_project_id"] = quota_project

                try:
                    config.environments[env_name] = EnvironmentConfig(
                        name=env_name,
                        default_credential=CredentialConfig(**cred_config_data),
                    )
                except Exception as e:
                    logger.error(
                        "failed_to_create_env_config",
                        env=env_name,
                        error=str(e),
                    )

        if default_env := os.getenv("DEFAULT_ENVIRONMENT"):
            if default_env in config.environments:
                config.default_environment = default_env
            else:
                logger.warning(
                    "default_environment_not_found",
                    requested=default_env,
                    available=list(config.environments.keys()),
                )

        if cache_ttl := os.getenv("CACHE_TTL_SECONDS"):
            try:
                config.cache_ttl_seconds = int(cache_ttl)
            except ValueError:
                logger.warning("invalid_cache_ttl", value=cache_ttl)

        if max_conns := os.getenv("MAX_CONNECTIONS_PER_CREDENTIAL"):
            try:
                config.max_connections_per_credential = int(max_conns)
            except ValueError:
                logger.warning("invalid_max_connections", value=max_conns)

        logger.info(
            "configuration_loaded_from_env",
            detected_environments=detected_envs,
            default_environment=config.default_environment,
        )

        return config


# Client Pool Management
class ClientPool:
    """client pool with resource tracking."""

    __slots__ = (
        "_sync_pools",
        "_async_pools",
        "_credentials",
        "_max_per_credential",
        "_enable_pooling",
        "_sync_lock",
        "_async_lock",
        "_logger",
        "_closed",
        "_active_managers",
        "_pool_stats",
    )

    def __init__(self, max_per_credential: int, enable_pooling: bool) -> None:
        self._sync_pools: dict[str, ConnectionPool[SecretManagerServiceClient]] = {}
        self._async_pools: dict[
            str, ConnectionPool[SecretManagerServiceAsyncClient]
        ] = {}
        self._credentials: dict[str, GoogleCredentials] = {}

        self._max_per_credential: int = max_per_credential
        self._enable_pooling: bool = enable_pooling
        self._closed: bool = False

        self._sync_lock: threading.RLock = threading.RLock()
        self._async_lock: asyncio.Lock = asyncio.Lock()

        self._active_managers: WeakSet[MultiEnvironmentSecretManager] = WeakSet()
        self._pool_stats: dict[str, int] = {"credential_loads": 0, "pool_creates": 0}

        LoggingConfig.configure()
        self._logger = structlog.get_logger(__name__).bind(component="ClientPool")

    def _check_not_closed(self) -> None:
        """Check if client pool is not closed."""
        if self._closed:
            raise RuntimeError(
                f"Client pool is closed (managed {len(self._credentials)} credentials, "
                + f"created {self._pool_stats['pool_creates']} pools)"
            )

    @staticmethod
    def _get_pool_key(credential_config: CredentialConfig) -> str:
        """Generate unique key for credential configuration."""
        match credential_config.method:
            case CredentialMethod.SERVICE_ACCOUNT_FILE:
                path_str = str(credential_config.service_account_path)
                return f"file:{path_str}:{hash(path_str) % 10000}"
            case CredentialMethod.SERVICE_ACCOUNT_JSON:
                client_email = (
                    credential_config.service_account_json.get(
                        "client_email", "unknown"
                    )
                    if credential_config.service_account_json
                    else "unknown"
                )
                return f"json:{client_email}:{hash(client_email) % 10000}"
            case CredentialMethod.IMPERSONATION:
                target = credential_config.impersonate_service_account or "unknown"
                return f"impersonate:{target}:{hash(target) % 10000}"
            case _:
                scopes_str = ",".join(sorted(credential_config.scopes))
                return f"default:{hash(scopes_str) % 10000}"

    def _load_credentials_sync(
        self, pool_key: str, credential_config: CredentialConfig
    ) -> GoogleCredentials:
        """Load credentials synchronously."""
        if pool_key not in self._credentials:
            try:
                self._credentials[pool_key] = credential_config.get_credentials()
                self._pool_stats["credential_loads"] += 1
                self._logger.debug(
                    "credentials_loaded",
                    pool_key=pool_key,
                    method=credential_config.method,
                )
            except Exception as e:
                self._logger.error(
                    "credential_loading_failed",
                    pool_key=pool_key,
                    method=credential_config.method,
                    error=str(e),
                )
                raise
        return self._credentials[pool_key]

    async def _load_credentials_async(
        self, pool_key: str, credential_config: CredentialConfig
    ) -> GoogleCredentials:
        """Load credentials asynchronously."""
        if pool_key not in self._credentials:
            try:
                loop = asyncio.get_running_loop()
                self._credentials[pool_key] = await loop.run_in_executor(
                    None, credential_config.get_credentials
                )
                self._pool_stats["credential_loads"] += 1
                self._logger.debug(
                    "credentials_loaded_async",
                    pool_key=pool_key,
                    method=credential_config.method,
                )
            except Exception as e:
                self._logger.error(
                    "credential_loading_failed_async",
                    pool_key=pool_key,
                    method=credential_config.method,
                    error=str(e),
                )
                raise
        return self._credentials[pool_key]

    def get_sync_client(
        self, credential_config: CredentialConfig
    ) -> SecretManagerServiceClient:
        """Get synchronous client."""
        self._check_not_closed()
        pool_key = self._get_pool_key(credential_config)

        if not self._enable_pooling:
            with self._sync_lock:
                credentials = self._load_credentials_sync(pool_key, credential_config)
            try:
                return SecretManagerServiceClient(credentials=credentials)
            except Exception as e:
                self._logger.error(
                    "client_creation_failed",
                    pool_key=pool_key,
                    error=str(e),
                )
                raise

        with self._sync_lock:
            if pool_key not in self._sync_pools:
                credentials = self._load_credentials_sync(pool_key, credential_config)

                def factory() -> SecretManagerServiceClient:
                    try:
                        return SecretManagerServiceClient(credentials=credentials)
                    except Exception as e_in:
                        self._logger.error(
                            "pooled_client_creation_failed",
                            pool_key=pool_key,
                            error=str(e_in),
                        )
                        raise

                self._sync_pools[pool_key] = ConnectionPool(
                    factory, self._max_per_credential, True
                )
                self._pool_stats["pool_creates"] += 1
                self._logger.debug(
                    "sync_pool_created",
                    pool_key=pool_key,
                    max_size=self._max_per_credential,
                )

            return self._sync_pools[pool_key].acquire()

    def release_sync_client(
        self, client: SecretManagerServiceClient, credential_config: CredentialConfig
    ) -> None:
        """Release synchronous client."""
        if self._closed:
            self._safe_close_client(client)
            return

        if not self._enable_pooling:
            self._safe_close_client(client)
            return

        pool_key = self._get_pool_key(credential_config)
        with self._sync_lock:
            if pool_key in self._sync_pools:
                try:
                    self._sync_pools[pool_key].release(client)
                except Exception as e:
                    self._logger.warning(
                        "client_release_failed",
                        pool_key=pool_key,
                        error=str(e),
                    )
                    self._safe_close_client(client)
            else:
                self._safe_close_client(client)

    async def get_async_client(
        self, credential_config: CredentialConfig
    ) -> SecretManagerServiceAsyncClient:
        """Get asynchronous client."""
        self._check_not_closed()
        pool_key = self._get_pool_key(credential_config)

        if not self._enable_pooling:
            async with self._async_lock:
                credentials = await self._load_credentials_async(
                    pool_key, credential_config
                )
            try:
                return SecretManagerServiceAsyncClient(credentials=credentials)
            except Exception as e:
                self._logger.error(
                    "async_client_creation_failed",
                    pool_key=pool_key,
                    error=str(e),
                )
                raise

        async with self._async_lock:
            if pool_key not in self._async_pools:
                credentials = await self._load_credentials_async(
                    pool_key, credential_config
                )

                def factory() -> SecretManagerServiceAsyncClient:
                    try:
                        return SecretManagerServiceAsyncClient(credentials=credentials)
                    except Exception as e_in:
                        self._logger.error(
                            "pooled_async_client_creation_failed",
                            pool_key=pool_key,
                            error=str(e_in),
                        )
                        raise

                self._async_pools[pool_key] = ConnectionPool(
                    factory, self._max_per_credential, True
                )
                self._pool_stats["pool_creates"] += 1
                self._logger.debug(
                    "async_pool_created",
                    pool_key=pool_key,
                    max_size=self._max_per_credential,
                )

            return await self._async_pools[pool_key].async_acquire()

    async def release_async_client(
        self,
        client: SecretManagerServiceAsyncClient,
        credential_config: CredentialConfig,
    ) -> None:
        """Release asynchronous client."""
        if self._closed:
            await self._safe_close_client_async(client)
            return

        if not self._enable_pooling:
            await self._safe_close_client_async(client)
            return

        pool_key = self._get_pool_key(credential_config)
        async with self._async_lock:
            if pool_key in self._async_pools:
                try:
                    await self._async_pools[pool_key].async_release(client)
                except Exception as e:
                    self._logger.warning(
                        "async_client_release_failed",
                        pool_key=pool_key,
                        error=str(e),
                    )
                    await self._safe_close_client_async(client)
            else:
                await self._safe_close_client_async(client)

    def _safe_close_client(self, client: Any) -> None:
        """Safely close a client."""
        try:
            if hasattr(client, "close") and callable(client.close):
                client.close()
        except Exception as e:
            self._logger.debug("client_close_error", error=str(e))

    async def _safe_close_client_async(self, client: Any) -> None:
        """Safely close an async client."""
        try:
            if hasattr(client, "close") and callable(client.close):
                if asyncio.iscoroutinefunction(client.close):
                    await client.close()
                else:
                    client.close()
        except Exception as e:
            self._logger.debug("async_client_close_error", error=str(e))

    def register_manager(self, manager: MultiEnvironmentSecretManager) -> None:
        """Register a manager for cleanup tracking."""
        self._active_managers.add(manager)
        self._logger.debug(
            "manager_registered",
            active_managers=len(self._active_managers),
        )

    def close(self) -> None:
        """Close all synchronous resources."""
        with self._sync_lock:
            if self._closed:
                return
            self._closed = True

            self._logger.debug(
                "closing_client_pool",
                sync_pools=len(self._sync_pools),
                async_pools=len(self._async_pools),
                credentials=len(self._credentials),
            )

            sync_pools: list[ConnectionPool[SecretManagerServiceClient]] = list(
                self._sync_pools.values()
            )
            self._sync_pools.clear()

        for sync_pool in sync_pools:
            try:
                sync_pool.close()
            except Exception as e:
                self._logger.warning("sync_pool_close_error", error=str(e))

        async_pools: list[ConnectionPool[SecretManagerServiceAsyncClient]] = list(
            self._async_pools.values()
        )
        self._async_pools.clear()
        for async_pool in async_pools:
            try:
                async_pool.close()
            except Exception as e:
                self._logger.warning("async_pool_close_error", error=str(e))

        self._credentials.clear()
        self._logger.info(
            "client_pool_closed",
            stats=self._pool_stats,
            active_managers=len(self._active_managers),
        )

    async def close_async(self) -> None:
        """Close all resources asynchronously."""
        self._logger.debug(
            "closing_client_pool_async",
            sync_pools=len(self._sync_pools),
            async_pools=len(self._async_pools),
            credentials=len(self._credentials),
        )

        with self._sync_lock:
            if self._closed:
                return
            self._closed = True

            sync_pools: list[ConnectionPool[SecretManagerServiceClient]] = list(
                self._sync_pools.values()
            )
            self._sync_pools.clear()

        for sync_pool in sync_pools:
            try:
                sync_pool.close()
            except Exception as e:
                self._logger.warning("sync_pool_close_error_async", error=str(e))

        async with self._async_lock:
            async_pools: list[ConnectionPool[SecretManagerServiceAsyncClient]] = list(
                self._async_pools.values()
            )
            self._async_pools.clear()

        for async_pool in async_pools:
            try:
                async_pool.close()
            except Exception as e:
                self._logger.warning("async_pool_close_error_async", error=str(e))

        self._credentials.clear()
        self._logger.info(
            "client_pool_closed_async",
            stats=self._pool_stats,
            active_managers=len(self._active_managers),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive client pool statistics."""
        with self._sync_lock:
            sync_pool_stats = {
                key: pool.get_stats() for key, pool in self._sync_pools.items()
            }

        return {
            "sync_pools": sync_pool_stats,
            "async_pools_count": len(self._async_pools),
            "credentials_count": len(self._credentials),
            "active_managers": len(self._active_managers),
            "enable_pooling": self._enable_pooling,
            "max_per_credential": self._max_per_credential,
            "closed": self._closed,
            **self._pool_stats,
        }


# Main Secret Manager
@final
class MultiEnvironmentSecretManager:
    """multi-environment secret manager."""

    __slots__ = (
        "_config",
        "_logger",
        "_cache",
        "_client_pool",
        "_circuit_breakers",
        "_closed",
        "_stats",
        "__weakref__",
    )

    def __init__(
        self, config: MultiEnvironmentSecretManagerConfig | None = None
    ) -> None:
        self._config = config or MultiEnvironmentSecretManagerConfig.from_env()
        self._closed = False
        self._stats = {
            "created_at": datetime.now(UTC),
            "operations_total": 0,
            "errors_total": 0,
        }
        LoggingConfig.configure()
        self._logger = structlog.get_logger().bind(
            component="MultiEnvironmentSecretManager",
            request_id=request_id_var.get(),
            trace_id=trace_id_var.get(),
        )

        self._cache: SecretCache | None = None
        if self._config.enable_caching:
            self._cache = SecretCache(
                ttl_seconds=self._config.cache_ttl_seconds,
                max_size=self._config.cache_max_size,
            )

        self._client_pool = ClientPool(
            max_per_credential=self._config.max_connections_per_credential,
            enable_pooling=self._config.enable_connection_pooling,
        )
        self._client_pool.register_manager(self)

        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        for env_name, env_config in self._config.environments.items():
            self._circuit_breakers[env_name] = CircuitBreaker(
                failure_threshold=env_config.circuit_breaker_threshold,
                recovery_timeout=env_config.circuit_breaker_timeout,
                expected_exception=RETRYABLE_ERRORS,
            )

        self._logger.info(
            "multi_env_secret_manager_initialized",
            environments=list(self._config.environments.keys()),
            default_environment=self._config.default_environment,
            strict_isolation=self._config.strict_environment_isolation,
            caching_enabled=self._config.enable_caching,
            pooling_enabled=self._config.enable_connection_pooling,
            metrics_enabled=self._config.enable_metrics,
        )

    def _check_not_closed(self) -> None:
        """Check if manager is not closed."""
        if self._closed:
            created_at = self._stats["created_at"]
            if isinstance(created_at, datetime):
                uptime = datetime.now(UTC) - created_at
            else:
                uptime = timedelta(0)
            raise RuntimeError(
                f"Secret manager is closed (uptime: {uptime}, "
                + f"operations: {self._stats['operations_total']}, "
                + f"errors: {self._stats['errors_total']})"
            )

    def _get_environment_config(
        self, environment_name_override: EnvironmentName | None = None
    ) -> EnvironmentConfig:
        """Get environment configuration."""
        current_env: EnvironmentName = environment_var.get()
        effective_env: EnvironmentName = environment_name_override or current_env

        env_config = self._config.environments.get(effective_env)
        if not env_config:
            if self._config.strict_environment_isolation:
                self._logger.error(
                    "unknown_environment_strict",
                    environment=effective_env,
                    available_environments=list(self._config.environments.keys()),
                )
                raise ConfigurationError(
                    f"Unknown environment: {effective_env}. "
                    + f"Available: {list(self._config.environments.keys())}"
                )

            self._logger.warning(
                "unknown_environment_fallback",
                requested=effective_env,
                fallback=self._config.default_environment,
                available_environments=list(self._config.environments.keys()),
            )
            env_config = self._config.environments.get(self._config.default_environment)
            if not env_config:
                raise ConfigurationError(
                    f"Default environment '{self._config.default_environment}' not found. "
                    + f"Available: {list(self._config.environments.keys())}"
                )

        return env_config

    def _get_credential_config(
        self, secret_config: SecretConfig, env_config: EnvironmentConfig
    ) -> CredentialConfig:
        """Get credential configuration for secret."""
        if secret_config.credential_override:
            self._logger.debug(
                "using_credential_override",
                project_id=secret_config.project_id,
                method=secret_config.credential_override.method,
            )
            return secret_config.credential_override

        try:
            return env_config.get_credential_for_project(secret_config.project_id)
        except ValueError as e:
            if self._config.strict_environment_isolation:
                self._logger.error(
                    "project_not_allowed",
                    project_id=secret_config.project_id,
                    environment=env_config.name,
                    allowed_projects=env_config.allowed_projects,
                )
                raise ConfigurationError(str(e)) from e

            self._logger.warning(
                "project_not_allowed_fallback",
                project_id=secret_config.project_id,
                environment=env_config.name,
                allowed_projects=env_config.allowed_projects,
            )
            return CredentialConfig(method=CredentialMethod.APPLICATION_DEFAULT)

    @contextlib.contextmanager
    def environment_context(self, environment: EnvironmentName) -> Iterator[None]:
        """Set environment context."""
        self._check_not_closed()

        if environment not in self._config.environments:
            raise ConfigurationError(
                f"Unknown environment: {environment}. "
                + f"Available: {list(self._config.environments.keys())}"
            )

        token = environment_var.set(environment)
        self._logger.debug("environment_context_enter", environment=environment)
        try:
            yield
        finally:
            environment_var.reset(token)
            self._logger.debug("environment_context_exit", environment=environment)

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close_async()

    def close(self) -> None:
        """Close all resources."""
        if self._closed:
            return

        self._closed = True

        created_at = self._stats["created_at"]
        if isinstance(created_at, datetime):
            uptime = datetime.now(UTC) - created_at
            uptime_seconds = uptime.total_seconds()
        else:
            uptime_seconds = 0.0

        self._client_pool.close()
        if self._cache:
            self._cache.clear()

        self._logger.info(
            "secret_manager_closed",
            uptime_seconds=uptime_seconds,
            operations_total=self._stats["operations_total"],
            errors_total=self._stats["errors_total"],
        )

    async def close_async(self) -> None:
        """Close all resources asynchronously."""
        if self._closed:
            return

        self._closed = True

        created_at = self._stats["created_at"]
        if isinstance(created_at, datetime):
            uptime = datetime.now(UTC) - created_at
            uptime_seconds = uptime.total_seconds()
        else:
            uptime_seconds = 0.0

        await self._client_pool.close_async()
        if self._cache:
            self._cache.clear()

        self._logger.info(
            "secret_manager_closed_async",
            uptime_seconds=uptime_seconds,
            operations_total=self._stats["operations_total"],
            errors_total=self._stats["errors_total"],
        )

    @staticmethod
    def _create_retry_decorator(max_retries: int) -> Any:
        """Create retry decorator with environment-specific parameters."""
        return retry(
            stop=stop_after_attempt(max_retries) | stop_after_delay(MAX_WAIT_SECONDS),
            retry=retry_if_exception_type(RETRYABLE_ERRORS),
            wait=wait_exponential(
                multiplier=WAIT_MULTIPLIER, min=MIN_WAIT_SECONDS, max=MAX_WAIT_SECONDS
            )
            + wait_random(0, 1),
            reraise=True,
        )

    def _handle_access_error(
        self,
        error: Exception,
        secret_config: SecretConfig,
        logger: structlog.BoundLogger,
        is_async: bool = False,
    ) -> NoReturn:
        """Handle errors during secret access."""
        current_errors = self._stats.get("errors_total", 0)
        if isinstance(current_errors, int):
            self._stats["errors_total"] = current_errors + 1
        else:
            self._stats["errors_total"] = 1
        suffix = "_async" if is_async else "_sync"

        if isinstance(error, GoogleAPICallError):
            log_level = (
                "warning"
                if hasattr(error, "code") and error.code in {403, 404}
                else "error"
            )

            getattr(logger, log_level)(
                f"google_api_error{suffix}",
                error=str(error),
                code=getattr(error, "code", None),
                details=getattr(error, "details", None),
                secret_path=secret_config.secret_path,
            )

            if hasattr(error, "code"):
                if error.code == 404:
                    raise SecretNotFoundError(
                        f"Secret not found: {secret_config.secret_path}",
                        secret_config=secret_config,
                        original_error=error,
                    ) from error
                if error.code == 403:
                    raise SecretAccessError(
                        f"Permission denied: {secret_config.secret_path}",
                        secret_config=secret_config,
                        original_error=error,
                    ) from error
                if error.code == 429:
                    raise SecretAccessError(
                        f"Rate limit exceeded: {secret_config.secret_path}",
                        secret_config=secret_config,
                        original_error=error,
                    ) from error

        logger.error(
            f"unexpected_error{suffix}",
            error=str(error),
            error_type=type(error).__name__,
            secret_path=secret_config.secret_path,
            exc_info=True,
        )

        if not isinstance(error, SecretManagerError):
            raise SecretAccessError(
                f"Unexpected error accessing {secret_config.secret_path}: {error}",
                secret_config=secret_config,
                original_error=error,
            ) from error

        raise error

    @with_metrics("access_secret_sync")
    def access_secret_version(
        self, secret_config: SecretConfig, *, timeout: float | None = None
    ) -> str:
        """Access secret version synchronously."""
        self._check_not_closed()

        current_ops = self._stats.get("operations_total", 0)
        if isinstance(current_ops, int):
            self._stats["operations_total"] = current_ops + 1
        else:
            self._stats["operations_total"] = 1

        env_config = self._get_environment_config(secret_config.environment)
        env_name = env_config.name

        logger = self._logger.bind(
            operation="access_secret",
            environment=env_name,
            project_id=secret_config.project_id,
            secret_name=secret_config.secret_name,
            secret_version=str(secret_config.secret_version),
        )

        if self._cache and (cached := self._cache.get(secret_config.cache_key)):
            logger.debug("cache_hit", cache_key=secret_config.cache_key)
            return cached["value"]

        circuit_breaker = self._circuit_breakers.get(env_name)
        if circuit_breaker and circuit_breaker.is_open:
            logger.warning("circuit_breaker_open", environment=env_name)
            raise SecretAccessError(
                f"Circuit breaker open for environment: {env_name}",
                secret_config=secret_config,
            )

        credential_config = self._get_credential_config(secret_config, env_config)
        client = self._client_pool.get_sync_client(credential_config)
        effective_timeout = timeout or env_config.timeout_seconds

        retry_decorator = self._create_retry_decorator(env_config.max_retries)

        def _access_with_retry() -> str:
            try:
                logger.debug(
                    "accessing_secret",
                    timeout=effective_timeout,
                    secret_path=secret_config.secret_path,
                )
                response = client.access_secret_version(
                    name=secret_config.secret_path, timeout=effective_timeout
                )

                if not response.payload or not response.payload.data:
                    raise SecretAccessError(
                        f"Empty payload for secret: {secret_config.secret_path}",
                        secret_config=secret_config,
                    )

                payload_size = len(response.payload.data)
                if payload_size > MAX_SECRET_SIZE_BYTES:
                    raise SecretAccessError(
                        f"Secret too large: {payload_size} bytes (max: {MAX_SECRET_SIZE_BYTES})",
                        secret_config=secret_config,
                    )

                secret_value = response.payload.data.decode("utf-8")

                if circuit_breaker:
                    circuit_breaker.record_success()

                if self._cache:
                    cache_entry: CacheEntry = {
                        "value": secret_value,
                        "metadata": CacheMetadata(
                            accessed_at=datetime.now(UTC),
                            version=response.name.split("/")[-1],
                            size_bytes=payload_size,
                            encoding="utf-8",
                            project_id=secret_config.project_id,
                            secret_name=secret_config.secret_name,
                            ttl_seconds=env_config.cache_ttl_seconds,
                            access_count=1,
                        ),
                        "expires_at": datetime.now(UTC)
                        + timedelta(seconds=env_config.cache_ttl_seconds),
                        "created_at": datetime.now(UTC),
                    }
                    self._cache.set(secret_config.cache_key, cache_entry)

                logger.info(
                    "secret_accessed",
                    version=response.name.split("/")[-1],
                    size_bytes=payload_size,
                    cached=self._cache is not None,
                )
                return secret_value

            except Exception as e:
                if circuit_breaker:
                    circuit_breaker.record_failure()
                self._handle_access_error(e, secret_config, logger, False)

        # Apply retry decorator
        retry_decorated_func = retry_decorator(_access_with_retry)

        try:
            return retry_decorated_func()
        finally:
            self._client_pool.release_sync_client(client, credential_config)

    @with_metrics("access_secret_async")
    async def access_secret_version_async(
        self, secret_config: SecretConfig, *, timeout: float | None = None
    ) -> str:
        """Access secret version asynchronously."""
        self._check_not_closed()

        current_ops = self._stats.get("operations_total", 0)
        if isinstance(current_ops, int):
            self._stats["operations_total"] = current_ops + 1
        else:
            self._stats["operations_total"] = 1

        env_config = self._get_environment_config(secret_config.environment)
        env_name = env_config.name

        logger = self._logger.bind(
            operation="access_secret_async",
            environment=env_name,
            project_id=secret_config.project_id,
            secret_name=secret_config.secret_name,
            secret_version=str(secret_config.secret_version),
        )

        if self._cache and (
            cached := await self._cache.async_get(secret_config.cache_key)
        ):
            logger.debug("cache_hit", cache_key=secret_config.cache_key)
            return cached["value"]

        circuit_breaker = self._circuit_breakers.get(env_name)
        if circuit_breaker and await circuit_breaker.async_is_open():
            logger.warning("circuit_breaker_open", environment=env_name)
            raise SecretAccessError(
                f"Circuit breaker open for environment: {env_name}",
                secret_config=secret_config,
            )

        credential_config = self._get_credential_config(secret_config, env_config)
        client = await self._client_pool.get_async_client(credential_config)
        effective_timeout = timeout or env_config.timeout_seconds

        retry_decorator = self._create_retry_decorator(env_config.max_retries)

        async def _access_with_retry() -> str:
            try:
                logger.debug(
                    "accessing_secret",
                    timeout=effective_timeout,
                    secret_path=secret_config.secret_path,
                )
                response = await client.access_secret_version(
                    name=secret_config.secret_path, timeout=effective_timeout
                )

                if not response.payload or not response.payload.data:
                    raise SecretAccessError(
                        f"Empty payload for secret: {secret_config.secret_path}",
                        secret_config=secret_config,
                    )

                payload_size = len(response.payload.data)
                if payload_size > MAX_SECRET_SIZE_BYTES:
                    raise SecretAccessError(
                        f"Secret too large: {payload_size} bytes (max: {MAX_SECRET_SIZE_BYTES})",
                        secret_config=secret_config,
                    )

                secret_value = response.payload.data.decode("utf-8")

                if circuit_breaker:
                    await circuit_breaker.async_record_success()

                if self._cache:
                    cache_entry: CacheEntry = {
                        "value": secret_value,
                        "metadata": CacheMetadata(
                            accessed_at=datetime.now(UTC),
                            version=response.name.split("/")[-1],
                            size_bytes=payload_size,
                            encoding="utf-8",
                            project_id=secret_config.project_id,
                            secret_name=secret_config.secret_name,
                            ttl_seconds=env_config.cache_ttl_seconds,
                            access_count=1,
                        ),
                        "expires_at": datetime.now(UTC)
                        + timedelta(seconds=env_config.cache_ttl_seconds),
                        "created_at": datetime.now(UTC),
                    }
                    await self._cache.async_set(secret_config.cache_key, cache_entry)

                logger.info(
                    "secret_accessed",
                    version=response.name.split("/")[-1],
                    size_bytes=payload_size,
                    cached=self._cache is not None,
                )
                return secret_value

            except Exception as e:
                if circuit_breaker:
                    await circuit_breaker.async_record_failure()
                self._handle_access_error(e, secret_config, logger, True)

        # Apply retry decorator
        retry_decorated_func = retry_decorator(_access_with_retry)

        try:
            return await retry_decorated_func()
        finally:
            await self._client_pool.release_async_client(client, credential_config)

    def create_environment_specific_manager(
        self, environment: EnvironmentName
    ) -> EnvironmentSpecificSecretManager:
        """Create an environment-specific secret manager."""
        self._check_not_closed()

        if environment not in self._config.environments:
            self._logger.error(
                "unknown_environment_for_specific_manager",
                environment=environment,
                available_environments=list(self._config.environments.keys()),
            )
            raise ConfigurationError(
                f"Unknown environment: {environment}. "
                + f"Available: {list(self._config.environments.keys())}"
            )

        return EnvironmentSpecificSecretManager(self, environment)

    def get_stats(self) -> dict[str, int | float | bool | list[str] | dict[str, Any]]:
        """Get comprehensive manager statistics."""
        self._check_not_closed()

        created_at = self._stats["created_at"]
        if isinstance(created_at, datetime):
            uptime = datetime.now(UTC) - created_at
            uptime_seconds = uptime.total_seconds()
        else:
            uptime_seconds = 0.0

        operations_total = self._stats["operations_total"]
        errors_total = self._stats["errors_total"]

        # Ensure operations_total and errors_total are integers
        if not isinstance(operations_total, int):
            operations_total = 0
        if not isinstance(errors_total, int):
            errors_total = 0

        error_rate = errors_total / operations_total if operations_total > 0 else 0.0

        stats: dict[str, int | float | bool | list[str] | dict[str, Any]] = {
            "environments": list(self._config.environments.keys()),
            "cache_enabled": self._config.enable_caching,
            "pooling_enabled": self._config.enable_connection_pooling,
            "metrics_enabled": self._config.enable_metrics,
            "closed": self._closed,
            "uptime_seconds": uptime_seconds,
            "operations_total": operations_total,
            "errors_total": errors_total,
            "error_rate": error_rate,
        }

        if self._cache:
            stats["cache"] = self._cache.get_stats()

        stats["circuit_breakers"] = {
            env: cb.get_state() for env, cb in self._circuit_breakers.items()
        }

        stats["client_pool"] = self._client_pool.get_stats()

        return stats


@final
class EnvironmentSpecificSecretManager:
    """environment-specific view of the secret manager."""

    __slots__ = ("_parent", "_environment_name", "_logger")

    def __init__(
        self, parent: MultiEnvironmentSecretManager, environment_name: EnvironmentName
    ) -> None:
        self._parent = parent
        self._environment_name = environment_name

        LoggingConfig.configure()
        self._logger = structlog.get_logger(__name__).bind(
            component="EnvironmentSpecificSecretManager",
            environment=environment_name,
        )

    def access_secret(
        self,
        project_id: str,
        secret_name: str,
        version: str | int = "latest",
        *,
        timeout: float | None = None,
    ) -> str:
        """Access secret in this environment."""
        config = SecretConfig(
            project_id=project_id,
            secret_name=secret_name,
            secret_version=version,
            environment=self._environment_name,
        )
        return self._parent.access_secret_version(config, timeout=timeout)

    async def access_secret_async(
        self,
        project_id: str,
        secret_name: str,
        version: str | int = "latest",
        *,
        timeout: float | None = None,
    ) -> str:
        """Access secret in this environment asynchronously."""
        config = SecretConfig(
            project_id=project_id,
            secret_name=secret_name,
            secret_version=version,
            environment=self._environment_name,
        )
        return await self._parent.access_secret_version_async(config, timeout=timeout)

    def get_stats(
        self,
    ) -> dict[str, str | int | float | bool | list[str] | dict[str, Any] | None]:
        """Get statistics for this environment."""
        parent_stats = self._parent.get_stats()

        env_stats: dict[
            str, str | int | float | bool | list[str] | dict[str, Any] | None
        ] = {
            "environment": self._environment_name,
            "parent_closed": parent_stats["closed"],
        }

        circuit_breakers = parent_stats.get("circuit_breakers")
        if circuit_breakers and isinstance(circuit_breakers, dict):
            env_stats["circuit_breaker"] = circuit_breakers.get(self._environment_name)
        else:
            env_stats["circuit_breaker"] = None

        return env_stats


# convenience function
@contextlib.contextmanager
def environment_context(environment: EnvironmentName) -> Iterator[None]:
    """convenience function for setting environment context."""
    LoggingConfig.configure()
    token = environment_var.set(environment)
    logger = structlog.get_logger(__name__)
    logger.debug("environment_context_enter", environment=environment)
    try:
        yield
    finally:
        environment_var.reset(token)
        logger.debug("environment_context_exit", environment=environment)
