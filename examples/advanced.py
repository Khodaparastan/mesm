"""
Advanced async usage examples with real GCP integration.
Demonstrates async patterns, error handling, and performance monitoring.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, TypedDict

import structlog
from mesm.secret_manager import (
    CredentialConfig,
    CredentialMethod,
    EnvironmentConfig,
    MultiEnvironmentSecretManager,
    MultiEnvironmentSecretManagerConfig,
    SecretAccessError,
    SecretConfig,
    SecretNotFoundError,
    environment_context,
)
from pydantic import ValidationError

# Enhanced type definitions for Python 3.13+
type OperationResult = dict[str, Any]
type ErrorType = Literal[
    "SecretNotFoundError", "SecretAccessError", "TimeoutError", "UnknownError"
]
type EnvironmentName = Literal["development", "staging", "production", "test"]

# Constants with better organization
DEMO_TIMEOUT: Final[float] = 30.0
MAX_CONCURRENT_OPERATIONS: Final[int] = 10
PERFORMANCE_TEST_ITERATIONS: Final[int] = 20
CACHE_PERFORMANCE_THRESHOLD: Final[float] = 0.01  # 10ms for cache hit detection
PROGRESS_REPORT_INTERVAL: Final[int] = 5
DEMO_SESSION_TIMEOUT: Final[float] = 300.0  # 5 minutes


class OperationStats(TypedDict):
    """Statistics for secret operations with enhanced metrics."""

    total_operations: int
    successful_operations: int
    failed_operations: int
    average_duration: float
    min_duration: float
    max_duration: float
    cache_hits: int
    error_breakdown: dict[str, int]


class DemoResult(TypedDict):
    """Result of a demo operation with comprehensive metadata."""

    operation_type: str
    secret_path: str
    success: bool
    duration: float
    timestamp: datetime
    environment: str
    cached: bool
    error_type: str | None
    error_message: str | None
    value_length: int | None


@dataclass(frozen=True, slots=True)
class SecretTestCase:
    """Test case for secret access with enhanced validation."""

    name: str
    project_id: str
    secret_name: str
    secret_version: str | int = "latest"
    environment: EnvironmentName | None = None
    expected_error: type[Exception] | tuple[type[Exception], ...] | None = None
    timeout: float = 10.0
    skip_validation: bool = False

    def __post_init__(self) -> None:
        """Validate test case parameters."""
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if not self.name.strip():
            raise ValueError("Test case name cannot be empty")

    @property
    def expected_error_names(self) -> list[str]:
        """Get expected error names as a list of strings."""
        if not self.expected_error:
            return []
        if isinstance(self.expected_error, tuple):
            return [exc.__name__ for exc in self.expected_error]
        return [self.expected_error.__name__]


@dataclass(slots=True)
class DemoConfiguration:
    """Configuration for the demo with enhanced validation."""

    project_id: str = "dev-seotrack"
    test_secret_name: str = "app-config"
    environments: list[EnvironmentName] = field(
        default_factory=lambda: ["development", "staging", "production"]
    )
    enable_performance_tests: bool = True
    enable_error_tests: bool = True
    enable_concurrent_tests: bool = True
    max_concurrent_operations: int = MAX_CONCURRENT_OPERATIONS
    operation_timeout: float = DEMO_TIMEOUT

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_concurrent_operations <= 0:
            raise ValueError("max_concurrent_operations must be positive")
        if self.operation_timeout <= 0:
            raise ValueError("operation_timeout must be positive")
        if not self.environments:
            raise ValueError("At least one environment must be specified")


class SecretManagerDemo:
    """Enhanced demo class for advanced secret manager usage with comprehensive monitoring."""

    __slots__ = (
        "_config",
        "_demo_config",
        "_manager",
        "_results",
        "_logger",
        "_start_time",
        "_stats",
        "_session_id",
    )

    def __init__(
        self,
        config: MultiEnvironmentSecretManagerConfig,
        demo_config: DemoConfiguration | None = None,
    ) -> None:
        self._config = config
        self._demo_config = demo_config or DemoConfiguration()
        self._manager: MultiEnvironmentSecretManager | None = None
        self._results: list[DemoResult] = []
        self._start_time = datetime.now(UTC)
        self._stats: dict[str, Any] = defaultdict(int)
        self._session_id = self._start_time.strftime("%Y%m%d_%H%M%S")

        # Configure structured logging with enhanced processors
        self._configure_logging()

        self._logger = structlog.get_logger(__name__).bind(
            component="SecretManagerDemo",
            demo_session=self._session_id,
            start_time=self._start_time.isoformat(),
        )

    def _configure_logging(self) -> None:
        """Configure structured logging with enhanced settings."""
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]

        # Add appropriate renderer based on environment
        if sys.stderr.isatty():
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.processors.JSONRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    async def setup(self) -> None:
        """Setup the secret manager with enhanced initialization and validation."""
        self._logger.info(
            "initializing_secret_manager",
            config_summary={
                "environments": list(self._config.environments.keys()),
                "caching_enabled": self._config.enable_caching,
                "pooling_enabled": self._config.enable_connection_pooling,
                "strict_isolation": self._config.strict_environment_isolation,
            },
        )

        try:
            self._manager = MultiEnvironmentSecretManager(self._config)

            # Validate manager initialization
            if not self._manager:
                raise RuntimeError("Failed to create secret manager instance")

            # Test basic connectivity
            manager_stats = self._manager.get_stats()
            self._logger.info(
                "secret_manager_initialized",
                initial_stats=manager_stats,
            )
            print("✓ Secret manager initialized successfully")

        except Exception as e:
            self._logger.error(
                "secret_manager_initialization_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            print(f"❌ Failed to initialize secret manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Enhanced cleanup with comprehensive statistics logging."""
        if self._manager:
            try:
                # Get final statistics before closing
                final_stats = self._manager.get_stats()
                session_duration = datetime.now(UTC) - self._start_time

                self._logger.info(
                    "final_manager_stats",
                    stats=final_stats,
                    session_duration=session_duration.total_seconds(),
                    total_operations=len(self._results),
                )

                await self._manager.close_async()
                self._logger.info("secret_manager_closed")
                print("✓ Secret manager closed successfully")

            except Exception as e:
                self._logger.error(
                    "cleanup_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                print(f"⚠ Error during cleanup: {e}")

    def _record_result(
        self,
        operation_type: str,
        secret_path: str,
        success: bool,
        duration: float,
        environment: str = "unknown",
        cached: bool = False,
        error: Exception | None = None,
        value_length: int | None = None,
    ) -> None:
        """Record operation result with enhanced metadata and validation."""
        if duration < 0:
            self._logger.warning("negative_duration_detected", duration=duration)
            duration = 0.0

        result: DemoResult = {
            "operation_type": operation_type,
            "secret_path": secret_path,
            "success": success,
            "duration": duration,
            "timestamp": datetime.now(UTC),
            "environment": environment,
            "cached": cached,
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            "value_length": value_length,
        }

        self._results.append(result)

        # Update statistics with better categorization
        self._stats[f"{operation_type}_total"] += 1
        self._stats["total_operations"] += 1

        if success:
            self._stats[f"{operation_type}_success"] += 1
            self._stats["total_success"] += 1
            if cached:
                self._stats["cache_hits"] += 1
                self._stats[f"{operation_type}_cache_hits"] += 1
        else:
            self._stats[f"{operation_type}_errors"] += 1
            self._stats["total_errors"] += 1
            if error:
                error_key = f"error_{type(error).__name__}"
                self._stats[error_key] += 1

    async def _access_secret_with_timing(
        self,
        config: SecretConfig,
        operation_type: str = "access",
        timeout: float | None = None,
    ) -> DemoResult:
        """Access a secret with comprehensive timing and error handling."""
        if not self._manager:
            raise RuntimeError("Manager not initialized")

        start_time = time.perf_counter()
        environment = config.environment or "unknown"
        effective_timeout = timeout or self._demo_config.operation_timeout

        try:
            self._logger.debug(
                "accessing_secret",
                secret_path=config.secret_path,
                environment=environment,
                timeout=effective_timeout,
                operation_type=operation_type,
            )

            # Use timeout wrapper for better control
            async with asyncio.timeout(effective_timeout):
                secret_value = await self._manager.access_secret_version_async(
                    config, timeout=effective_timeout
                )

            duration = time.perf_counter() - start_time
            value_length = len(secret_value)

            # Enhanced cache detection with multiple criteria
            cached = duration < CACHE_PERFORMANCE_THRESHOLD and operation_type in {
                "cache_test",
                "performance_test",
            }

            self._logger.info(
                "secret_access_success",
                secret_path=config.secret_path,
                duration=duration,
                value_length=value_length,
                cached=cached,
                operation_type=operation_type,
            )

            self._record_result(
                operation_type=operation_type,
                secret_path=config.secret_path,
                success=True,
                duration=duration,
                environment=environment,
                cached=cached,
                value_length=value_length,
            )

            return {
                "operation_type": operation_type,
                "secret_path": config.secret_path,
                "success": True,
                "duration": duration,
                "timestamp": datetime.now(UTC),
                "environment": environment,
                "cached": cached,
                "error_type": None,
                "error_message": None,
                "value_length": value_length,
            }

        except Exception as e:
            duration = time.perf_counter() - start_time

            self._logger.warning(
                "secret_access_failed",
                secret_path=config.secret_path,
                duration=duration,
                error_type=type(e).__name__,
                error=str(e),
                operation_type=operation_type,
            )

            self._record_result(
                operation_type=operation_type,
                secret_path=config.secret_path,
                success=False,
                duration=duration,
                environment=environment,
                error=e,
            )

            return {
                "operation_type": operation_type,
                "secret_path": config.secret_path,
                "success": False,
                "duration": duration,
                "timestamp": datetime.now(UTC),
                "environment": environment,
                "cached": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "value_length": None,
            }

    async def demonstrate_concurrent_access(self) -> None:
        """Demonstrate concurrent secret access with enhanced monitoring and safety."""
        if not self._demo_config.enable_concurrent_tests:
            self._logger.info("concurrent_tests_disabled")
            return

        print("\n=== Concurrent Secret Access Demo ===")
        self._logger.info("starting_concurrent_access_demo")

        if not self._manager:
            raise RuntimeError("Manager not initialized")

        # Create balanced test configurations
        secret_configs = []
        ops_per_env = max(
            1,
            self._demo_config.max_concurrent_operations
            // len(self._demo_config.environments),
        )

        for env in self._demo_config.environments:
            for i in range(ops_per_env):
                secret_configs.append(
                    SecretConfig(
                        project_id=self._demo_config.project_id,
                        secret_name=self._demo_config.test_secret_name,
                        secret_version="latest",
                        environment=env,
                    )
                )

        print(f"Accessing {len(secret_configs)} secrets concurrently...")
        start_time = time.perf_counter()

        try:
            # Use asyncio.timeout for better timeout handling
            async with asyncio.timeout(DEMO_TIMEOUT):
                # Create semaphore to limit concurrent operations
                semaphore = asyncio.Semaphore(min(10, len(secret_configs)))

                async def limited_access(config: SecretConfig) -> DemoResult:
                    async with semaphore:
                        return await self._access_secret_with_timing(
                            config, "concurrent_access"
                        )

                tasks = [limited_access(config) for config in secret_configs]
                results = await asyncio.gather(*tasks, return_exceptions=True)

        except TimeoutError:
            self._logger.error("concurrent_access_timeout")
            print("❌ Concurrent access demo timed out")
            return

        total_duration = time.perf_counter() - start_time

        # Enhanced result processing
        successful_results = [
            r for r in results if isinstance(r, dict) and r.get("success")
        ]
        failed_results = [
            r for r in results if isinstance(r, dict) and not r.get("success")
        ]
        exceptions = [r for r in results if not isinstance(r, dict)]

        # Calculate performance metrics
        ops_per_second = len(results) / total_duration if total_duration > 0 else 0

        self._logger.info(
            "concurrent_access_completed",
            total_duration=total_duration,
            total_operations=len(results),
            successful=len(successful_results),
            failed=len(failed_results),
            exceptions=len(exceptions),
            operations_per_second=ops_per_second,
        )

        print(f"✓ Total duration: {total_duration:.2f}s")
        print(f"✓ Operations per second: {ops_per_second:.1f}")
        print(f"✓ Successful accesses: {len(successful_results)}")
        print(f"✓ Failed accesses: {len(failed_results)}")
        print(f"✓ Exceptions: {len(exceptions)}")

        # Test cache performance if we had success
        if successful_results:
            await self._test_cache_performance(secret_configs[0])

    async def _test_cache_performance(self, config: SecretConfig) -> None:
        """Test cache performance with detailed metrics and validation."""
        print("\nTesting cache performance...")
        self._logger.info("testing_cache_performance", secret_path=config.secret_path)

        cache_tests = []
        test_count = 5

        for i in range(test_count):
            try:
                result = await self._access_secret_with_timing(config, "cache_test")
                cache_tests.append(result)

                # Small delay between tests to avoid overwhelming the system
                if i < test_count - 1:
                    await asyncio.sleep(0.1)

            except Exception as e:
                self._logger.warning(
                    "cache_test_failed",
                    test_number=i + 1,
                    error=str(e),
                )

        successful_cache_tests = [r for r in cache_tests if r["success"]]

        if successful_cache_tests:
            durations = [r["duration"] for r in successful_cache_tests]
            avg_cache_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)

            # Count likely cache hits
            cache_hits = [r for r in successful_cache_tests if r["cached"]]

            print(
                f"✓ Cache tests completed: {len(successful_cache_tests)}/{test_count}"
            )
            print(f"✓ Average duration: {avg_cache_duration:.4f}s")
            print(f"✓ Min/Max duration: {min_duration:.4f}s / {max_duration:.4f}s")
            print(f"✓ Likely cache hits: {len(cache_hits)}")

            self._logger.info(
                "cache_performance_results",
                tests_total=test_count,
                tests_successful=len(successful_cache_tests),
                average_duration=avg_cache_duration,
                min_duration=min_duration,
                max_duration=max_duration,
                cache_hits=len(cache_hits),
            )
        else:
            print("⚠ No successful cache tests")
            self._logger.warning("no_successful_cache_tests")

    async def demonstrate_error_handling(self) -> None:
        """Demonstrate comprehensive error handling with various scenarios."""
        if not self._demo_config.enable_error_tests:
            self._logger.info("error_tests_disabled")
            return

        print("\n=== Error Handling Demo ===")
        self._logger.info("starting_error_handling_demo")

        if not self._manager:
            raise RuntimeError("Manager not initialized")

        # Enhanced error scenarios with better validation
        error_scenarios = [
            SecretTestCase(
                name="Non-existent secret",
                project_id=self._demo_config.project_id,
                secret_name="non-existent-secret-12345",
                expected_error=SecretNotFoundError,
            ),
            SecretTestCase(
                name="Invalid project ID format",
                project_id="invalid-project-123",
                secret_name=self._demo_config.test_secret_name,
                expected_error=(SecretAccessError, ValidationError),
            ),
            SecretTestCase(
                name="Invalid version number",
                project_id=self._demo_config.project_id,
                secret_name=self._demo_config.test_secret_name,
                secret_version="999999",
                expected_error=SecretNotFoundError,
            ),
            SecretTestCase(
                name="Very short timeout",
                project_id=self._demo_config.project_id,
                secret_name=self._demo_config.test_secret_name,
                timeout=0.001,  # 1ms timeout
                expected_error=(asyncio.TimeoutError, SecretAccessError),
            ),
            SecretTestCase(
                name="Empty secret name",
                project_id=self._demo_config.project_id,
                secret_name="",
                expected_error=ValidationError,
                skip_validation=False,
            ),
        ]

        for scenario in error_scenarios:
            print(f"\nTesting: {scenario.name}")
            self._logger.info(
                "testing_error_scenario",
                scenario_name=scenario.name,
                expected_errors=scenario.expected_error_names,
            )

            await self._test_error_scenario(scenario)

    async def _test_error_scenario(self, scenario: SecretTestCase) -> None:
        """Test a single error scenario with proper error handling."""
        try:
            # Handle potential validation errors during config creation
            if not scenario.skip_validation:
                config = SecretConfig(
                    project_id=scenario.project_id,
                    secret_name=scenario.secret_name,
                    secret_version=scenario.secret_version,
                    environment=scenario.environment,
                )

                result = await self._access_secret_with_timing(
                    config,
                    "error_test",
                    timeout=scenario.timeout,
                )

                # Check if we got the expected result
                if scenario.expected_error and result["success"]:
                    print(f"⚠ Unexpected success for {scenario.name}")
                    self._logger.warning(
                        "unexpected_success",
                        scenario=scenario.name,
                        result=result,
                    )
                elif scenario.expected_error and not result["success"]:
                    # Check if the error type matches expectations
                    error_type = result.get("error_type")
                    expected_names = scenario.expected_error_names

                    if error_type in expected_names:
                        print(
                            f"✓ Caught expected {error_type}: {result.get('error_message', 'No message')}"
                        )
                        self._logger.info(
                            "caught_expected_error_in_result",
                            scenario=scenario.name,
                            error_type=error_type,
                        )
                    else:
                        print(
                            f"⚠ Caught unexpected {error_type}, expected one of: {expected_names}"
                        )
                        self._logger.warning(
                            "caught_unexpected_error_in_result",
                            scenario=scenario.name,
                            error_type=error_type,
                            expected=expected_names,
                        )
                else:
                    print(f"✓ Success for {scenario.name}")

        except ValidationError as e:
            print(f"✓ Caught validation error: {e}")
            self._logger.info(
                "caught_validation_error",
                scenario=scenario.name,
                error=str(e),
            )

        except Exception as e:
            expected_types = scenario.expected_error
            if expected_types:
                if not isinstance(expected_types, tuple):
                    expected_types = (expected_types,)

                if any(
                    isinstance(e, expected_type) for expected_type in expected_types
                ):
                    print(f"✓ Caught expected {type(e).__name__}: {e}")
                    self._logger.info(
                        "caught_expected_error",
                        scenario=scenario.name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                else:
                    print(f"⚠ Caught unexpected {type(e).__name__}: {e}")
                    self._logger.warning(
                        "caught_unexpected_error",
                        scenario=scenario.name,
                        error_type=type(e).__name__,
                        expected=scenario.expected_error_names,
                        error_message=str(e),
                    )
            else:
                print(f"✓ Caught {type(e).__name__}: {e}")
                self._logger.info(
                    "caught_error",
                    scenario=scenario.name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )

    async def demonstrate_environment_switching(self) -> None:
        """Demonstrate environment switching patterns with comprehensive testing."""
        print("\n=== Environment Switching Demo ===")
        self._logger.info("starting_environment_switching_demo")

        if not self._manager:
            raise RuntimeError("Manager not initialized")

        for env in self._demo_config.environments:
            print(f"\n--- Testing {env.upper()} environment ---")
            self._logger.info("testing_environment", environment=env)

            # Method 1: Using environment context
            await self._test_environment_context(env)

            # Method 2: Using environment-specific manager
            await self._test_environment_specific_manager(env)

    async def _test_environment_context(self, env: EnvironmentName) -> None:
        """Test environment context method with enhanced error handling."""
        print(f"Method 1: Environment context for {env}")

        try:
            async with asyncio.timeout(15):
                with environment_context(env):
                    config = SecretConfig(
                        project_id=self._demo_config.project_id,
                        secret_name=self._demo_config.test_secret_name,
                        secret_version="latest",
                    )
                    result = await self._access_secret_with_timing(
                        config,
                        "env_context",
                    )

                    if result["success"]:
                        print(
                            f"✓ Context method: Retrieved {env} secret "
                            f"(length: {result['value_length']}, "
                            f"duration: {result['duration']:.3f}s)"
                        )
                    else:
                        print(f"⚠ Context method failed: {result['error_message']}")

        except TimeoutError:
            print(f"⚠ Context method timed out for {env}")
            self._logger.warning("environment_context_timeout", environment=env)
        except Exception as e:
            print(f"⚠ Context method error in {env}: {e}")
            self._logger.error(
                "environment_context_error",
                environment=env,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _test_environment_specific_manager(self, env: EnvironmentName) -> None:
        """Test environment-specific manager method with enhanced monitoring."""
        print(f"Method 2: Environment-specific manager for {env}")

        if not self._manager:
            return

        try:
            env_manager = self._manager.create_environment_specific_manager(env)

            start_time = time.perf_counter()
            secret_value = await env_manager.access_secret_async(
                project_id=self._demo_config.project_id,
                secret_name=self._demo_config.test_secret_name,
                version="latest",
                timeout=10.0,
            )
            duration = time.perf_counter() - start_time

            print(
                f"✓ Env manager: Retrieved {env} secret "
                f"(length: {len(secret_value)}, duration: {duration:.3f}s)"
            )

            self._record_result(
                operation_type="env_manager",
                secret_path=f"projects/{self._demo_config.project_id}/secrets/{self._demo_config.test_secret_name}/versions/latest",
                success=True,
                duration=duration,
                environment=env,
                value_length=len(secret_value),
            )

        except Exception as e:
            print(f"⚠ Environment manager error in {env}: {e}")
            self._logger.error(
                "environment_manager_error",
                environment=env,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def demonstrate_performance_monitoring(self) -> None:
        """Demonstrate performance monitoring with comprehensive metrics."""
        if not self._demo_config.enable_performance_tests:
            self._logger.info("performance_tests_disabled")
            return

        print("\n=== Performance Monitoring Demo ===")
        self._logger.info("starting_performance_monitoring_demo")

        if not self._manager:
            raise RuntimeError("Manager not initialized")

        # Create varied test operations for realistic performance testing
        operations = self._create_performance_test_operations()

        print(f"Performing {len(operations)} operations for performance analysis...")
        start_time = time.perf_counter()

        # Execute operations with progress reporting
        performance_results = await self._execute_performance_operations(operations)
        total_time = time.perf_counter() - start_time

        # Analyze and report results
        await self._analyze_performance_results(performance_results, total_time)

    def _create_performance_test_operations(self) -> list[SecretConfig]:
        """Create varied test operations for performance testing."""
        operations = []

        for i in range(PERFORMANCE_TEST_ITERATIONS):
            # Vary secrets to test cache behavior
            secret_name = (
                self._demo_config.test_secret_name
                if i % 3 == 0
                else f"perf-test-secret-{i % 3}"
            )
            env = self._demo_config.environments[
                i % len(self._demo_config.environments)
            ]

            operations.append(
                SecretConfig(
                    project_id=self._demo_config.project_id,
                    secret_name=secret_name,
                    secret_version="latest",
                    environment=env,
                )
            )

        return operations

    async def _execute_performance_operations(
        self, operations: list[SecretConfig]
    ) -> list[DemoResult]:
        """Execute performance operations with progress reporting."""
        performance_results = []

        for i, config in enumerate(operations):
            try:
                result = await self._access_secret_with_timing(
                    config, "performance_test"
                )
                performance_results.append(result)

                # Report progress
                if (i + 1) % PROGRESS_REPORT_INTERVAL == 0:
                    print(f"  Progress: {i + 1}/{len(operations)} operations completed")

            except Exception as e:
                self._logger.error(
                    "performance_test_error",
                    operation=i,
                    error=str(e),
                    error_type=type(e).__name__,
                )

        return performance_results

    async def _analyze_performance_results(
        self, performance_results: list[DemoResult], total_time: float
    ) -> None:
        """Analyze and report performance results with comprehensive metrics."""
        successful_ops = [r for r in performance_results if r["success"]]
        failed_ops = [r for r in performance_results if not r["success"]]

        stats: OperationStats | None = None

        if successful_ops:
            durations = [r["duration"] for r in successful_ops]
            cache_hits = [r for r in successful_ops if r["cached"]]

            stats = {
                "total_operations": len(performance_results),
                "successful_operations": len(successful_ops),
                "failed_operations": len(failed_ops),
                "average_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "cache_hits": len(cache_hits),
                "error_breakdown": {},
            }

            # Calculate error breakdown
            for result in failed_ops:
                error_type = result.get("error_type", "Unknown")
                stats["error_breakdown"][error_type] = (
                    stats["error_breakdown"].get(error_type, 0) + 1
                )

        # Get manager statistics
        manager_stats = self._manager.get_stats() if self._manager else {}

        # Display results
        self._display_performance_results(
            performance_results, total_time, stats, manager_stats
        )

        # Save detailed results
        await self._save_performance_results(performance_results, manager_stats)

    def _display_performance_results(
        self,
        performance_results: list[DemoResult],
        total_time: float,
        stats: OperationStats | None,
        manager_stats: dict[str, Any],
    ) -> None:
        """Display comprehensive performance results."""
        successful_ops = [r for r in performance_results if r["success"]]
        failed_ops = [r for r in performance_results if not r["success"]]

        print(f"\n📊 Performance Analysis Results:")
        print(f"  Total execution time: {total_time:.2f}s")
        print(f"  Operations per second: {len(performance_results) / total_time:.2f}")
        print(
            f"  Successful operations: {len(successful_ops)}/{len(performance_results)}"
        )

        if successful_ops and stats:
            print(f"  Average operation duration: {stats['average_duration']:.3f}s")
            print(
                f"  Min/Max duration: {stats['min_duration']:.3f}s / {stats['max_duration']:.3f}s"
            )
            print(f"  Cache hits: {stats['cache_hits']}")
            print(
                f"  Cache hit rate: {(stats['cache_hits'] / len(successful_ops)) * 100:.1f}%"
            )

        if failed_ops:
            print(f"  Failed operations: {len(failed_ops)}")
            if stats and stats["error_breakdown"]:
                print(f"  Error breakdown: {stats['error_breakdown']}")

        # Log detailed statistics
        self._logger.info(
            "performance_analysis_completed",
            total_time=total_time,
            operations_per_second=len(performance_results) / total_time,
            performance_stats=stats,
            manager_stats=manager_stats,
        )

    async def _save_performance_results(
        self,
        results: list[DemoResult],
        manager_stats: dict[str, Any],
    ) -> None:
        """Save performance results to a JSON file for detailed analysis."""
        try:
            results_file = Path(f"secret_manager_demo_results_{self._session_id}.json")

            output_data = {
                "demo_session": self._session_id,
                "start_time": self._start_time.isoformat(),
                "demo_config": {
                    "project_id": self._demo_config.project_id,
                    "environments": self._demo_config.environments,
                    "max_concurrent_operations": self._demo_config.max_concurrent_operations,
                    "operation_timeout": self._demo_config.operation_timeout,
                },
                "performance_results": [
                    {**result, "timestamp": result["timestamp"].isoformat()}
                    for result in results
                ],
                "manager_stats": manager_stats,
                "demo_stats": dict(self._stats),
                "session_duration": (
                    datetime.now(UTC) - self._start_time
                ).total_seconds(),
            }

            with results_file.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, default=str)

            print(f"📁 Detailed results saved to: {results_file}")
            self._logger.info("performance_results_saved", file=str(results_file))

        except Exception as e:
            self._logger.error(
                "failed_to_save_results",
                error=str(e),
                error_type=type(e).__name__,
            )
            print(f"⚠ Failed to save results: {e}")

    def print_comprehensive_summary(self) -> None:
        """Print comprehensive summary of all operations with detailed analytics."""
        print("\n=== Comprehensive Demo Summary ===")

        if not self._results:
            print("No results to summarize")
            self._logger.info("no_results_to_summarize")
            return

        # Calculate comprehensive statistics
        total_ops = len(self._results)
        successful_ops = [r for r in self._results if r["success"]]
        failed_ops = [r for r in self._results if not r["success"]]

        # Group by operation type and environment
        ops_by_type = defaultdict(list)
        ops_by_env = defaultdict(list)

        for result in self._results:
            ops_by_type[result["operation_type"]].append(result)
            ops_by_env[result["environment"]].append(result)

        # Display overall statistics
        self._display_overall_statistics(total_ops, successful_ops, failed_ops)

        # Display operation type breakdown
        self._display_operation_breakdown(ops_by_type)

        # Display environment breakdown
        self._display_environment_breakdown(ops_by_env)

        # Display error analysis
        self._display_error_analysis(failed_ops)

        # Display session information
        self._display_session_info(total_ops)

    def _display_overall_statistics(
        self,
        total_ops: int,
        successful_ops: list[DemoResult],
        failed_ops: list[DemoResult],
    ) -> None:
        """Display overall operation statistics."""
        if successful_ops:
            durations = [r["duration"] for r in successful_ops]
            cache_hits = [r for r in successful_ops if r["cached"]]

            print(f"📈 Overall Statistics:")
            print(f"  Total operations: {total_ops}")
            print(
                f"  Successful: {len(successful_ops)} ({len(successful_ops) / total_ops * 100:.1f}%)"
            )
            print(
                f"  Failed: {len(failed_ops)} ({len(failed_ops) / total_ops * 100:.1f}%)"
            )
            print(f"  Average duration: {sum(durations) / len(durations):.3f}s")
            print(f"  Min/Max duration: {min(durations):.3f}s / {max(durations):.3f}s")
            print(
                f"  Cache hits: {len(cache_hits)} ({len(cache_hits) / len(successful_ops) * 100:.1f}%)"
            )

    def _display_operation_breakdown(
        self, ops_by_type: dict[str, list[DemoResult]]
    ) -> None:
        """Display operation type breakdown."""
        print(f"\n📊 Operation Type Breakdown:")
        for op_type, ops in ops_by_type.items():
            successful = [r for r in ops if r["success"]]
            if successful:
                avg_duration = sum(r["duration"] for r in successful) / len(successful)
                print(
                    f"  {op_type}: {len(successful)}/{len(ops)} successful (avg: {avg_duration:.3f}s)"
                )
            else:
                print(f"  {op_type}: {len(successful)}/{len(ops)} successful")

    def _display_environment_breakdown(
        self, ops_by_env: dict[str, list[DemoResult]]
    ) -> None:
        """Display environment breakdown."""
        print(f"\n🌍 Environment Breakdown:")
        for env, ops in ops_by_env.items():
            successful = [r for r in ops if r["success"]]
            if successful:
                avg_duration = sum(r["duration"] for r in successful) / len(successful)
                print(
                    f"  {env}: {len(successful)}/{len(ops)} successful (avg: {avg_duration:.3f}s)"
                )
            else:
                print(f"  {env}: {len(successful)}/{len(ops)} successful")

    def _display_error_analysis(self, failed_ops: list[DemoResult]) -> None:
        """Display error analysis."""
        if failed_ops:
            error_types = defaultdict(int)
            for result in failed_ops:
                error_type = result.get("error_type", "Unknown")
                error_types[error_type] += 1

            print(f"\n❌ Error Analysis:")
            for error_type, count in error_types.items():
                print(f"  {error_type}: {count} occurrences")

    def _display_session_info(self, total_ops: int) -> None:
        """Display demo session information."""
        session_duration = datetime.now(UTC) - self._start_time
        print(f"\n⏱ Demo Session Info:")
        print(f"  Session ID: {self._session_id}")
        print(f"  Session duration: {session_duration.total_seconds():.1f}s")
        print(
            f"  Operations per minute: {total_ops / (session_duration.total_seconds() / 60):.1f}"
        )

        # Log final summary
        self._logger.info(
            "demo_summary_completed",
            session_id=self._session_id,
            total_operations=total_ops,
            successful_operations=len([r for r in self._results if r["success"]]),
            failed_operations=len([r for r in self._results if not r["success"]]),
            session_duration=session_duration.total_seconds(),
            demo_stats=dict(self._stats),
        )


async def create_demo_configuration() -> MultiEnvironmentSecretManagerConfig:
    """Create enhanced demo configuration with environment-specific settings."""
    return MultiEnvironmentSecretManagerConfig(
        environments={
            "development": EnvironmentConfig(
                name="development",
                default_credential=CredentialConfig(
                    method=CredentialMethod.APPLICATION_DEFAULT
                ),
                cache_ttl_seconds=30,
                timeout_seconds=10.0,
                max_retries=3,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=30.0,
            ),
            "staging": EnvironmentConfig(
                name="staging",
                default_credential=CredentialConfig(
                    method=CredentialMethod.APPLICATION_DEFAULT
                ),
                cache_ttl_seconds=60,
                timeout_seconds=15.0,
                max_retries=3,
                circuit_breaker_threshold=3,
                circuit_breaker_timeout=60.0,
            ),
            "production": EnvironmentConfig(
                name="production",
                default_credential=CredentialConfig(
                    method=CredentialMethod.APPLICATION_DEFAULT
                ),
                cache_ttl_seconds=300,
                timeout_seconds=30.0,
                max_retries=5,
                circuit_breaker_threshold=2,
                circuit_breaker_timeout=120.0,
            ),
        },
        default_environment="development",
        enable_caching=True,
        enable_connection_pooling=False,  # Disable pooling to avoid event loop issues
        cache_max_size=100,
        cache_ttl_seconds=300,
        max_connections_per_credential=5,
        strict_environment_isolation=False,  # Allow fallback for demo
        enable_metrics=True,
    )


async def main() -> None:
    """Enhanced main async demo function with comprehensive error handling."""
    print("🚀 Multi-Environment Secret Manager - Advanced Async Examples\n")

    logger = structlog.get_logger(__name__)
    logger.info("starting_demo_session")

    demo: SecretManagerDemo | None = None

    try:
        # Create configurations with environment variable support
        config = await create_demo_configuration()
        demo_config = DemoConfiguration(
            project_id=os.getenv("DEMO_PROJECT_ID", "dev-seotrack"),
            test_secret_name=os.getenv("DEMO_SECRET_NAME", "app-config"),
            enable_performance_tests=os.getenv("ENABLE_PERF_TESTS", "true").lower()
            == "true",
            enable_error_tests=os.getenv("ENABLE_ERROR_TESTS", "true").lower()
            == "true",
            enable_concurrent_tests=os.getenv("ENABLE_CONCURRENT_TESTS", "true").lower()
            == "true",
        )

        # Initialize and run demo with timeout
        demo = SecretManagerDemo(config, demo_config)

        async with asyncio.timeout(DEMO_SESSION_TIMEOUT):
            await demo.setup()

            # Run all demonstrations
            logger.info("running_demo_phases")

            if demo_config.enable_concurrent_tests:
                await demo.demonstrate_concurrent_access()

            if demo_config.enable_error_tests:
                await demo.demonstrate_error_handling()

            await demo.demonstrate_environment_switching()

            if demo_config.enable_performance_tests:
                await demo.demonstrate_performance_monitoring()

            demo.print_comprehensive_summary()

        print("\n🎉 All advanced async examples completed successfully!")
        logger.info("demo_session_completed_successfully")

    except KeyboardInterrupt:
        print("\n⚠ Demo interrupted by user")
        logger.warning("demo_interrupted_by_user")
    except TimeoutError:
        print("\n⏰ Demo timed out")
        logger.error("demo_session_timeout")
    except Exception as e:
        print(f"\n❌ Unexpected error in demo: {e}")
        logger.error(
            "unexpected_demo_error",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise
    finally:
        # Ensure cleanup happens
        if demo:
            try:
                await demo.cleanup()
            except Exception as e:
                logger.error(
                    "cleanup_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )


if __name__ == "__main__":
    # Set up environment with better defaults
    os.environ.setdefault("LOG_LEVEL", "20")  # INFO level
    os.environ.setdefault("DEMO_PROJECT_ID", "dev-seotrack")
    os.environ.setdefault("DEMO_SECRET_NAME", "app-config")

    # Run async demo with proper error handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
