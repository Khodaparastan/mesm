"""
Simple async script to get a secret from Google Cloud Secret Manager.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import NoReturn

from mesm.secret_manager import (
    CredentialConfig,
    CredentialMethod,
    EnvironmentConfig,
    MultiEnvironmentSecretManager,
    MultiEnvironmentSecretManagerConfig,
    SecretConfig,
)


async def get_secret_async() -> str | NoReturn:
    """Get a secret asynchronously."""

    # Configuration
    project_id = os.getenv("GCP_PROJECT_ID", "dev-seotrack")
    secret_name = os.getenv("SECRET_NAME", "app-config")
    secret_version = os.getenv("SECRET_VERSION", "latest")

    print(f"🔍 Getting secret: {secret_name} from project: {project_id}")

    # Create simple configuration
    config = MultiEnvironmentSecretManagerConfig(
        environments={
            "production": EnvironmentConfig(
                name="production",
                default_credential=CredentialConfig(
                    method=CredentialMethod.APPLICATION_DEFAULT
                ),
                timeout_seconds=10.0,
            )
        },
        default_environment="production",
        enable_caching=True,
        enable_connection_pooling=False,  # Keep it simple
    )

    # Create secret configuration
    secret_config = SecretConfig(
        project_id=project_id,
        secret_name=secret_name,
        secret_version=secret_version,
    )

    # Get the secret
    async with MultiEnvironmentSecretManager(config) as manager:
        try:
            secret_value = await manager.access_secret_version_async(secret_config)
            print(f"✅ Secret retrieved successfully!")
            print(f"📏 Secret length: {len(secret_value)} characters")
            print(f"🔒 Secret value: {secret_value}")
            if len(secret_value) > 50:
                preview = secret_value[:50] + "..."
            else:
                preview = secret_value
            print(f"🔍 Preview: {preview}")

            return secret_value

        except Exception as e:
            print(f"❌ Error getting secret: {e}")
            raise


async def main() -> None:
    """Main async function."""
    try:
        _ = await get_secret_async()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set default environment variables if not set
    _ = os.environ.setdefault("GCP_PROJECT_ID", "dev-seotrack")
    _ = os.environ.setdefault("SECRET_NAME", "app-config")
    _ = os.environ.setdefault("SECRET_VERSION", "latest")

    print("🚀 Simple Async Secret Retrieval")
    asyncio.run(main())
