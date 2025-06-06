#!/usr/bin/env python3
"""
Batch operations and performance testing example.
Demonstrates bulk secret access, performance optimization, and load testing.
"""

import asyncio
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mesm import (
    CredentialConfig,
    CredentialMethod,
    EnvironmentConfig,
    MultiEnvironmentSecretManager,
    MultiEnvironmentSecretManagerConfig,
    SecretConfig,
)


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""

    operation_count: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration: float = 0.0
    operation_durations: list[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.operation_count == 0:
            return 0.0
        return self.successful_operations / self.operation_count

    @property
    def average_duration(self) -> float:
        """Calculate average operation duration."""
        if not self.operation_durations:
            return 0.0
        return statistics.mean(self.operation_durations)

    @property
    def median_duration(self) -> float:
        """Calculate median operation duration."""
        if not self.operation_durations:
            return 0.0
        return statistics.median(self.operation_durations)

    @property
    def p95_duration(self) -> float:
        """Calculate 95th percentile duration."""
        if not self.operation_durations:
            return 0.0
        return statistics.quantiles(self.operation_durations, n=20)[
            18
        ]  # 95th percentile

    def add_operation(
        self, duration: float, success: bool, error_type: str | None = None
    ) -> None:
        """Add an operation result."""
        self.operation_count += 1
        self.operation_durations.append(duration)

        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
            if error_type:
                self.errors_by_type[error_type] = (
                    self.errors_by_type.get(error_type, 0) + 1
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "operation_count": self.operation_count,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.success_rate,
            "total_duration": self.total_duration,
            "average_duration": self.average_duration,
            "median_duration": self.median_duration,
            "p95_duration": self.p95_duration,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "errors_by_type": self.errors_by_type,
        }


class BatchSecretManager:
    """Batch operations manager for secrets."""

    def __init__(self, manager: MultiEnvironmentSecretManager) -> None:
        self.manager = manager
        self.metrics = PerformanceMetrics()

    async def access_secrets_batch_async(
        self,
        secret_configs: list[SecretConfig],
        max_concurrent: int = 10,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Access multiple secrets concurrently."""
        print(
            f"Accessing {len(secret_configs)} secrets with max {max_concurrent} concurrent operations..."
        )

        semaphore = asyncio.Semaphore(max_concurrent)
        start_time = time.time()

        async def access_single_secret(config: SecretConfig) -> dict[str, Any]:
            """Access a single secret with metrics tracking."""
            async with semaphore:
                operation_start = time.time()
                try:
                    secret_value = await self.manager.access_secret_version_async(
                        config, timeout=timeout
                    )
                    duration = time.time() - operation_start
                    self.metrics.add_operation(duration, True)

                    return {
                        "secret_path": config.secret_path,
                        "success": True,
                        "duration": duration,
                        "value_length": len(secret_value),
                    }
                except Exception as e:
                    duration = time.time() - operation_start
                    error_type = type(e).__name__
                    self.metrics.add_operation(duration, False, error_type)

                    return {
                        "secret_path": config.secret_path,
                        "success": False,
                        "duration": duration,
                        "error": str(e),
                        "error_type": error_type,
                    }

        # Execute all operations
        tasks = [access_single_secret(config) for config in secret_configs]
        results = await asyncio.gather(*tasks)

        self.metrics.total_duration = time.time() - start_time

        return {
            "results": results,
            "metrics": self.metrics.to_dict(),
            "total_duration": self.metrics.total_duration,
        }

    def access_secrets_batch_sync(
        self,
        secret_configs: list[SecretConfig],
        max_workers: int = 5,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Access multiple secrets using thread pool."""
        print(
            f"Accessing {len(secret_configs)} secrets with {max_workers} worker threads..."
        )

        start_time = time.time()
        results = []

        def access_single_secret(config: SecretConfig) -> dict[str, Any]:
            """Access a single secret with metrics tracking."""
            operation_start = time.time()
            try:
                secret_value = self.manager.access_secret_version(
                    config, timeout=timeout
                )
                duration = time.time() - operation_start
                self.metrics.add_operation(duration, True)

                return {
                    "secret_path": config.secret_path,
                    "success": True,
                    "duration": duration,
                    "value_length": len(secret_value),
                }
            except Exception as e:
                duration = time.time() - operation_start
                error_type = type(e).__name__
                self.metrics.add_operation(duration, False, error_type)

                return {
                    "secret_path": config.secret_path,
                    "success": False,
                    "duration": duration,
                    "error": str(e),
                    "error_type": error_type,
                }

        # Execute with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {
                executor.submit(access_single_secret, config): config
                for config in secret_configs
            }

            for future in as_completed(future_to_config):
                result = future.result()
                results.append(result)

        self.metrics.total_duration = time.time() - start_time

        return {
            "results": results,
            "metrics": self.metrics.to_dict(),
            "total_duration": self.metrics.total_duration,
        }

    async def load_test(
        self,
        secret_configs: list[SecretConfig],
        duration_seconds: int = 60,
        requests_per_second: int = 10,
    ) -> dict[str, Any]:
        """Perform load testing on secret access."""
        print(
            f"Starting load test: {requests_per_second} RPS for {duration_seconds} seconds..."
        )

        start_time = time.time()
        end_time = start_time + duration_seconds
        interval = 1.0 / requests_per_second

        load_metrics = PerformanceMetrics()

        async def make_request() -> None:
            """Make a single request."""
            config = secret_configs[
                int(time.time()) % len(secret_configs)
            ]  # Rotate configs
            operation_start = time.time()

            try:
                await self.manager.access_secret_version_async(config, timeout=10.0)
                duration = time.time() - operation_start
                load_metrics.add_operation(duration, True)
            except Exception as e:
                duration = time.time() - operation_start
                error_type = type(e).__name__
                load_metrics.add_operation(duration, False, error_type)

        # Generate load
        tasks = []
        next_request_time = start_time

        while time.time() < end_time:
            if time.time() >= next_request_time:
                tasks.append(asyncio.create_task(make_request()))
                next_request_time += interval

            # Clean up completed tasks periodically
            if len(tasks) > 100:
                done_tasks = [task for task in tasks if task.done()]
                for task in done_tasks:
                    tasks.remove(task)

            await asyncio.sleep(0.01)  # Small sleep to prevent busy waiting

        # Wait for remaining tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        load_metrics.total_duration = time.time() - start_time

        return {
            "load_test_config": {
                "duration_seconds": duration_seconds,
                "target_rps": requests_per_second,
                "actual_rps": load_metrics.operation_count
                / load_metrics.total_duration,
            },
            "metrics": load_metrics.to_dict(),
        }


async def demonstrate_batch_operations() -> None:
    """Demonstrate batch operations."""
    print("=== Batch Operations Demo ===")

    # Create configuration
    config = MultiEnvironmentSecretManagerConfig(
        environments={
            "development": EnvironmentConfig(
                name="development",
                default_credential=CredentialConfig(
                    method=CredentialMethod.APPLICATION_DEFAULT
                ),
                cache_ttl_seconds=30,
                timeout_seconds=10.0,
            ),
        },
        default_environment="development",
        enable_caching=True,
        enable_connection_pooling=True,
        cache_max_size=100,
    )

    async with MultiEnvironmentSecretManager(config) as manager:
        batch_manager = BatchSecretManager(manager)

        # Create test secret configurations
        secret_configs = [
            SecretConfig(
                project_id="demo-project",
                secret_name=f"batch-test-secret-{i}",
                secret_version="latest",
            )
            for i in range(1, 21)  # 20 secrets
        ]

        # Test async batch operations
        print("\n1. Testing async batch operations...")
        async_results = await batch_manager.access_secrets_batch_async(
            secret_configs, max_concurrent=5, timeout=10.0
        )

        print(f"Async batch results:")
        print(f"  Total operations: {async_results['metrics']['operation_count']}")
        print(f"  Success rate: {async_results['metrics']['success_rate']:.2%}")
        print(
            f"  Average duration: {async_results['metrics']['average_duration']:.3f}s"
        )
        print(f"  Total time: {async_results['total_duration']:.2f}s")

        # Test sync batch operations
        print("\n2. Testing sync batch operations...")
        sync_results = batch_manager.access_secrets_batch_sync(
            secret_configs[:10],
            max_workers=3,
            timeout=10.0,  # Smaller batch for sync
        )

        print(f"Sync batch results:")
        print(f"  Total operations: {sync_results['metrics']['operation_count']}")
        print(f"  Success rate: {sync_results['metrics']['success_rate']:.2%}")
        print(f"  Average duration: {sync_results['metrics']['average_duration']:.3f}s")
        print(f"  Total time: {sync_results['total_duration']:.2f}s")

        # Test cache performance
        print("\n3. Testing cache performance...")
        cache_test_configs = secret_configs[:5]  # Use subset for cache test

        # First access (cache miss)
        first_access = await batch_manager.access_secrets_batch_async(
            cache_test_configs, max_concurrent=5, timeout=10.0
        )

        # Second access (should hit cache)
        batch_manager.metrics = PerformanceMetrics()  # Reset metrics
        second_access = await batch_manager.access_secrets_batch_async(
            cache_test_configs, max_concurrent=5, timeout=10.0
        )

        print(f"Cache performance comparison:")
        print(f"  First access avg: {first_access['metrics']['average_duration']:.3f}s")
        print(
            f"  Second access avg: {second_access['metrics']['average_duration']:.3f}s"
        )

        # Mini load test
        print("\n4. Running mini load test...")
        load_results = await batch_manager.load_test(
            secret_configs[:3], duration_seconds=10, requests_per_second=5
        )

        print(f"Load test results:")
        print(f"  Target RPS: {load_results['load_test_config']['target_rps']}")
        print(f"  Actual RPS: {load_results['load_test_config']['actual_rps']:.2f}")
        print(f"  Success rate: {load_results['metrics']['success_rate']:.2%}")
        print(f"  P95 duration: {load_results['metrics']['p95_duration']:.3f}s")


def save_performance_report(
    results: dict[str, Any], filename: str = "performance_report.json"
) -> None:
    """Save performance results to file."""
    report_path = Path(filename)

    report = {
        "timestamp": time.time(),
        "results": results,
    }

    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Performance report saved to: {report_path}")


async def comprehensive_performance_test() -> None:
    """Run comprehensive performance tests."""
    print("=== Comprehensive Performance Test ===")

    # Test different configurations
    test_configs = [
        {
            "name": "No Cache, No Pooling",
            "config": MultiEnvironmentSecretManagerConfig(
                enable_caching=False,
                enable_connection_pooling=False,
            ),
        },
        {
            "name": "Cache Only",
            "config": MultiEnvironmentSecretManagerConfig(
                enable_caching=True,
                enable_connection_pooling=False,
                cache_ttl_seconds=60,
            ),
        },
        {
            "name": "Pooling Only",
            "config": MultiEnvironmentSecretManagerConfig(
                enable_caching=False,
                enable_connection_pooling=True,
                max_connections_per_credential=5,
            ),
        },
        {
            "name": "Cache + Pooling",
            "config": MultiEnvironmentSecretManagerConfig(
                enable_caching=True,
                enable_connection_pooling=True,
                cache_ttl_seconds=60,
                max_connections_per_credential=5,
            ),
        },
    ]

    # Test secrets
    test_secrets = [
        SecretConfig(
            project_id="perf-test-project",
            secret_name=f"perf-secret-{i}",
            secret_version="latest",
        )
        for i in range(1, 11)
    ]

    all_results = {}

    for test_config in test_configs:
        print(f"\nTesting configuration: {test_config['name']}")

        async with MultiEnvironmentSecretManager(test_config["config"]) as manager:
            batch_manager = BatchSecretManager(manager)

            # Run batch test
            results = await batch_manager.access_secrets_batch_async(
                test_secrets, max_concurrent=3, timeout=10.0
            )

            all_results[test_config["name"]] = results

            print(f"  Success rate: {results['metrics']['success_rate']:.2%}")
            print(f"  Average duration: {results['metrics']['average_duration']:.3f}s")
            print(f"  Total time: {results['total_duration']:.2f}s")

    # Save comprehensive report
    save_performance_report(all_results, "comprehensive_performance_report.json")

    # Print summary
    print("\n=== Performance Summary ===")
    for config_name, results in all_results.items():
        metrics = results["metrics"]
        print(f"{config_name}:")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Avg Duration: {metrics['average_duration']:.3f}s")
        print(f"  P95 Duration: {metrics['p95_duration']:.3f}s")


async def main() -> None:
    """Main function for batch operations demo."""
    print("Multi-Environment Secret Manager - Batch Operations & Performance Testing\n")

    try:
        await demonstrate_batch_operations()
        await comprehensive_performance_test()

        print("\n🎉 All batch operations and performance tests completed!")

    except KeyboardInterrupt:
        print("\n⚠ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in tests: {e}")
        raise


if __name__ == "__main__":
    import os

    # Set up environment
    os.environ.setdefault("LOG_LEVEL", "30")  # WARNING level to reduce noise

    # Run the demo
    asyncio.run(main())
