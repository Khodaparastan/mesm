"""
Simple sync script to get a secret from Google Cloud Secret Manager.
"""

import os
import sys

from mesm.secret_manager import (
    CredentialConfig,
    CredentialMethod,
    EnvironmentConfig,
    MultiEnvironmentSecretManager,
    MultiEnvironmentSecretManagerConfig,
    SecretConfig,
)


def get_secret_sync() -> str:
    """Get a secret synchronously."""

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
        enable_connection_pooling=False,
    )

    # Create secret configuration
    secret_config = SecretConfig(
        project_id=project_id,
        secret_name=secret_name,
        secret_version=secret_version,
    )

    # Get the secret
    with MultiEnvironmentSecretManager(config) as manager:
        try:
            secret_value = manager.access_secret_version(secret_config)
            print(f"✅ Secret retrieved successfully!")
            print(f"📏 Secret length: {len(secret_value)} characters")

            if len(secret_value) > 50:
                preview = secret_value[:50] + "..."
            else:
                preview = secret_value
            print(f"🔍 Preview: {preview}")

            return secret_value

        except Exception as e:
            print(f"❌ Error getting secret: {e}")
            raise


def main() -> None:
    """Main function."""
    try:
        secret = get_secret_sync()
        # Do something with the secret here
        print(f"🎉 Successfully retrieved secret!")

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

    print("🚀 Simple Sync Secret Retrieval")
    main()
