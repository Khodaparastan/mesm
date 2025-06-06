#!/usr/bin/env python3
"""
Basic usage examples for the Multi-Environment Secret Manager.
Demonstrates configuration, basic secret access, and error handling.
"""

import os
import sys
from pathlib import Path

from mesm.secret_manager import (
    ConfigurationError,
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


def basic_configuration_example() -> None:
    """Demonstrate basic configuration setup."""
    _ = sys.stdout.write("=== Basic Configuration Example ===")

    # Method 1: Configuration from environment variables
    _ = sys.stdout.write("1. Loading configuration from environment variables...")
    try:
        config = MultiEnvironmentSecretManagerConfig.from_env()
        _ = sys.stdout.write(
            f"✓ Loaded config with environments: {list(config.environments.keys())}"
        )
    except Exception as e:
        _ = sys.stdout.write(f"⚠ Error loading from env: {e}")
        _ = sys.stdout.write("Creating default configuration...")
        config = MultiEnvironmentSecretManagerConfig()

    # Method 2: Manual configuration
    _ = sys.stdout.write("\n2. Creating manual configuration...")
    manual_config = MultiEnvironmentSecretManagerConfig(
        environments={
            "development": EnvironmentConfig(
                name="development",
                default_credential=CredentialConfig(
                    method=CredentialMethod.APPLICATION_DEFAULT
                ),
                cache_ttl_seconds=60,  # Shorter cache for dev
                timeout_seconds=10.0,
            ),
            "production": EnvironmentConfig(
                name="production",
                default_credential=CredentialConfig(
                    method=CredentialMethod.APPLICATION_DEFAULT
                ),
                allowed_projects=["my-prod-project"],  # Restrict projects
                cache_ttl_seconds=300,
                timeout_seconds=30.0,
            ),
        },
        default_environment="development",
        enable_caching=True,
        enable_connection_pooling=True,
        strict_environment_isolation=True,
    )
    _ = sys.stdout.write(
        f"✓ Created manual config with {len(manual_config.environments)} environments"
    )

    # Method 3: YAML configuration (if PyYAML is available)
    _ = sys.stdout.write("\n3. YAML configuration example...")
    yaml_content = """
environments:
  development:
    name: development
    default_credential:
      method: application_default
    cache_ttl_seconds: 60
    timeout_seconds: 10.0
  staging:
    name: staging
    default_credential:
      method: application_default
    allowed_projects:
      - my-staging-project
    cache_ttl_seconds: 180
    timeout_seconds: 20.0
  production:
    name: production
    default_credential:
      method: application_default
    allowed_projects:
      - my-prod-project
    cache_ttl_seconds: 300
    timeout_seconds: 30.0

default_environment: development
enable_caching: true
enable_connection_pooling: true
strict_environment_isolation: true
cache_ttl_seconds: 300
cache_max_size: 1000
"""

    # Save to temporary file and load
    yaml_file = Path("xsecret_manager_config.yaml")
    try:
        if yaml_file.write_text(yaml_content):
            yaml_config = MultiEnvironmentSecretManagerConfig.from_yaml(yaml_file)
            _ = sys.stdout.write(
                f"✓ Loaded YAML config with {len(yaml_config.environments)} environments"
            )
            yaml_file.unlink()
        else:
            _ = sys.stdout.write("⚠ Failed to write YAML config")
    except ImportError:
        _ = sys.stdout.write("⚠ PyYAML not available, skipping YAML example")
    except Exception as e:
        _ = sys.stdout.write(f"⚠ Error with YAML config: {e}")

    _ = sys.stdout.write("✓ Configuration examples completed\n")


def basic_secret_access_example() -> None:
    """Demonstrate basic secret access patterns."""
    _ = sys.stdout.write("=== Basic Secret Access Example ===")

    # Create manager with default configuration
    config = MultiEnvironmentSecretManagerConfig()

    with MultiEnvironmentSecretManager(config) as manager:
        # Example secret configurations
        secret_configs = [
            SecretConfig(
                project_id="my-project",
                secret_name="database-password",
                secret_version="latest",
            ),
            SecretConfig(
                project_id="my-project",
                secret_name="api-key",
                secret_version="2",  # Specific version
            ),
            SecretConfig(
                project_id="another-project",
                secret_name="service-account-key",
                secret_version="latest",
                environment="production",  # Specific environment
            ),
        ]

        for secret_config in secret_configs:
            try:
                _ = sys.stdout.write(f"Accessing secret: {secret_config.secret_path}")

                # Access secret (this will fail in demo without real GCP setup)
                secret_value = manager.access_secret_version(secret_config, timeout=5.0)
                _ = sys.stdout.write(
                    f"✓ Retrieved secret (length: {len(secret_value)})"
                )

            except SecretNotFoundError as e:
                _ = sys.stdout.write(f"⚠ Secret not found: {e}")
            except SecretAccessError as e:
                _ = sys.stdout.write(f"⚠ Access denied: {e}")
            except ConfigurationError as e:
                _ = sys.stdout.write(f"⚠ Configuration error: {e}")
            except Exception as e:
                _ = sys.stdout.write(f"⚠ Unexpected error: {e}")

        # Show manager statistics
        stats = manager.get_stats()
        _ = sys.stdout.write(f"\nManager stats: {stats}")

    _ = sys.stdout.write("✓ Basic secret access examples completed\n")


def environment_context_example() -> None:
    """Demonstrate environment context usage."""
    _ = sys.stdout.write("=== Environment Context Example ===")

    config = MultiEnvironmentSecretManagerConfig(
        environments={
            "dev": EnvironmentConfig(name="dev"),
            "prod": EnvironmentConfig(name="prod"),
        },
        default_environment="dev",
    )

    with MultiEnvironmentSecretManager(config) as manager:
        # Method 1: Using environment context manager
        _ = sys.stdout.write("1. Using environment context manager...")

        with environment_context("prod"):
            secret_config = SecretConfig(
                project_id="my-project",
                secret_name="prod-secret",
            )
            _ = sys.stdout.write(f"In prod context: {secret_config.cache_key}")

        with environment_context("dev"):
            secret_config = SecretConfig(
                project_id="my-project",
                secret_name="dev-secret",
            )
            _ = sys.stdout.write(f"In dev context: {secret_config.cache_key}")

        # Method 2: Using manager's environment context
        _ = sys.stdout.write("\n2. Using manager's environment context...")

        with manager.environment_context("prod"):
            secret_config = SecretConfig(
                project_id="my-project",
                secret_name="another-secret",
            )
            _ = sys.stdout.write(f"Manager prod context: {secret_config.cache_key}")

        # Method 3: Environment-specific manager
        _ = sys.stdout.write("\n3. Using environment-specific manager...")

        prod_manager = manager.create_environment_specific_manager("prod")
        try:
            # This would access secret in production environment
            secret_value = prod_manager.access_secret(
                project_id="my-project",
                secret_name="prod-only-secret",
                version="latest",
                timeout=5.0,
            )
            _ = sys.stdout.write(
                f"✓ Retrieved prod secret (length: {len(secret_value)})"
            )
        except Exception as e:
            _ = sys.stdout.write(f"⚠ Error accessing prod secret: {e}")

    _ = sys.stdout.write("✓ Environment context examples completed\n")


if __name__ == "__main__":
    _ = sys.stdout.write("Multi-Environment Secret Manager - Basic Examples\n")

    # Set up some demo environment variables
    _ = os.environ.setdefault("DEFAULT_ENVIRONMENT", "development")
    _ = os.environ.setdefault("LOG_LEVEL", "20")  # INFO level

    try:
        basic_configuration_example()
        basic_secret_access_example()
        environment_context_example()

        _ = sys.stdout.write("🎉 All basic examples completed successfully!")

    except KeyboardInterrupt:
        _ = sys.stdout.write("\n⚠ Examples interrupted by user")
    except Exception as e:
        _ = sys.stdout.write(f"\n❌ Unexpected error in examples: {e}")
        raise
