#!/usr/bin/env python3
"""
Production web application integration example.
Demonstrates FastAPI integration, dependency injection, and production patterns.
"""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

try:
    from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    print("This example requires FastAPI. Install with: pip install fastapi uvicorn")
    exit(1)

from mesm import (
    ConfigurationError,
    CredentialConfig,
    CredentialMethod,
    EnvironmentConfig,
    MultiEnvironmentSecretManager,
    MultiEnvironmentSecretManagerConfig,
    SecretAccessError,
    SecretConfig,
    SecretNotFoundError,
)

# Global secret manager instance
secret_manager: MultiEnvironmentSecretManager | None = None


class SecretRequest(BaseModel):
    """Request model for secret access."""

    project_id: str
    secret_name: str
    version: str = "latest"
    environment: str | None = None
    timeout: float = 30.0


class SecretResponse(BaseModel):
    """Response model for secret access."""

    success: bool
    value: str | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    environment: str
    cache_enabled: bool
    stats: dict[str, Any]


async def get_secret_manager() -> MultiEnvironmentSecretManager:
    """Dependency to get the secret manager instance."""
    if secret_manager is None:
        raise HTTPException(status_code=500, detail="Secret manager not initialized")
    return secret_manager


def create_production_config() -> MultiEnvironmentSecretManagerConfig:
    """Create production-ready configuration."""

    # Load service account from environment or file
    sa_file_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    sa_json_str = os.getenv("SERVICE_ACCOUNT_JSON")

    credential_config = CredentialConfig(method=CredentialMethod.APPLICATION_DEFAULT)

    if sa_file_path:
        credential_config = CredentialConfig(
            method=CredentialMethod.SERVICE_ACCOUNT_FILE,
            service_account_path=Path(sa_file_path),
        )
    elif sa_json_str:
        try:
            sa_json = json.loads(sa_json_str)
            credential_config = CredentialConfig(
                method=CredentialMethod.SERVICE_ACCOUNT_JSON,
                service_account_json=sa_json,
            )
        except json.JSONDecodeError:
            print("Warning: Invalid SERVICE_ACCOUNT_JSON, using default credentials")

    # Create environment-specific configurations
    environments = {
        "development": EnvironmentConfig(
            name="development",
            default_credential=credential_config,
            cache_ttl_seconds=60,  # Short cache for dev
            timeout_seconds=10.0,
        ),
        "staging": EnvironmentConfig(
            name="staging",
            default_credential=credential_config,
            allowed_projects=os.getenv("STAGING_PROJECTS", "").split(",") or None,
            cache_ttl_seconds=180,
            timeout_seconds=20.0,
        ),
        "production": EnvironmentConfig(
            name="production",
            default_credential=credential_config,
            allowed_projects=os.getenv("PRODUCTION_PROJECTS", "").split(",") or None,
            cache_ttl_seconds=600,  # Longer cache for prod
            timeout_seconds=30.0,
        ),
    }

    return MultiEnvironmentSecretManagerConfig(
        environments=environments,
        default_environment=os.getenv("DEFAULT_ENVIRONMENT", "production"),
        enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
        enable_connection_pooling=os.getenv("ENABLE_POOLING", "true").lower() == "true",
        max_connections_per_credential=int(os.getenv("MAX_CONNECTIONS", "10")),
        strict_environment_isolation=os.getenv("STRICT_ISOLATION", "true").lower()
        == "true",
        cache_ttl_seconds=int(os.getenv("CACHE_TTL", "300")),
        cache_max_size=int(os.getenv("CACHE_SIZE", "1000")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    global secret_manager

    print("🚀 Starting secret manager service...")

    try:
        # Initialize secret manager
        config = create_production_config()
        secret_manager = MultiEnvironmentSecretManager(config)

        print("✅ Secret manager initialized successfully")

        # Startup health check
        stats = secret_manager.get_stats()
        print(f"📊 Initial stats: {stats}")

        yield

    except Exception as e:
        print(f"❌ Failed to initialize secret manager: {e}")
        raise
    finally:
        # Cleanup
        print("🛑 Shutting down secret manager...")
        if secret_manager:
            await secret_manager.close_async()
            secret_manager = None
        print("✅ Secret manager shut down successfully")


# Create FastAPI app
app = FastAPI(
    title="Secret Manager Service",
    description="Production secret management service with multi-environment support",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check(
    manager: MultiEnvironmentSecretManager = Depends(get_secret_manager),
) -> HealthResponse:
    """Health check endpoint."""
    try:
        stats = manager.get_stats()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            environment=os.getenv("ENVIRONMENT", "unknown"),
            cache_enabled=stats.get("cache_enabled", False),
            stats=stats,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.post("/secrets/access", response_model=SecretResponse)
async def access_secret(
    request: SecretRequest,
    background_tasks: BackgroundTasks,
    manager: MultiEnvironmentSecretManager = Depends(get_secret_manager),
) -> SecretResponse:
    """Access a secret with full error handling."""

    def log_access_attempt() -> None:
        """Background task to log access attempt."""
        print(f"Secret access: {request.project_id}/{request.secret_name}")

    background_tasks.add_task(log_access_attempt)

    try:
        # Create secret configuration
        config = SecretConfig(
            project_id=request.project_id,
            secret_name=request.secret_name,
            secret_version=request.version,
            environment=request.environment,
        )

        # Access secret
        secret_value = await manager.access_secret_version_async(
            config, timeout=request.timeout
        )

        # Return success response (don't log the actual secret value)
        return SecretResponse(
            success=True,
            value=secret_value,
            metadata={
                "secret_path": config.secret_path,
                "cache_key": config.cache_key,
                "environment": request.environment or "default",
            },
        )

    except SecretNotFoundError as e:
        return SecretResponse(
            success=False,
            error=f"Secret not found: {e}",
        )
    except SecretAccessError as e:
        return SecretResponse(
            success=False,
            error=f"Access denied: {e}",
        )
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/secrets/{project_id}/{secret_name}")
async def get_secret_simple(
    project_id: str,
    secret_name: str,
    version: str = "latest",
    environment: str | None = None,
    timeout: float = 30.0,
    manager: MultiEnvironmentSecretManager = Depends(get_secret_manager),
) -> JSONResponse:
    """Simple GET endpoint for secret access."""

    request = SecretRequest(
        project_id=project_id,
        secret_name=secret_name,
        version=version,
        environment=environment,
        timeout=timeout,
    )

    response = await access_secret(request, BackgroundTasks(), manager)

    if response.success:
        return JSONResponse(
            content={"value": response.value, "metadata": response.metadata}
        )
    raise HTTPException(status_code=404, detail=response.error)


@app.get("/environments/{environment}/secrets/{project_id}/{secret_name}")
async def get_secret_environment_specific(
    environment: str,
    project_id: str,
    secret_name: str,
    version: str = "latest",
    timeout: float = 30.0,
    manager: MultiEnvironmentSecretManager = Depends(get_secret_manager),
) -> JSONResponse:
    """Environment-specific secret access endpoint."""

    try:
        # Create environment-specific manager
        env_manager = manager.create_environment_specific_manager(environment)

        # Access secret
        secret_value = await env_manager.access_secret_async(
            project_id=project_id,
            secret_name=secret_name,
            version=version,
            timeout=timeout,
        )

        return JSONResponse(
            content={
                "value": secret_value,
                "metadata": {
                    "environment": environment,
                    "project_id": project_id,
                    "secret_name": secret_name,
                    "version": version,
                },
            }
        )

    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SecretNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SecretAccessError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/stats")
async def get_stats(
    manager: MultiEnvironmentSecretManager = Depends(get_secret_manager),
) -> dict[str, Any]:
    """Get manager statistics."""
    return manager.get_stats()


@app.post("/cache/clear")
async def clear_cache(
    manager: MultiEnvironmentSecretManager = Depends(get_secret_manager),
) -> JSONResponse:
    """Clear the secret cache."""
    try:
        # Access private cache to clear it
        if hasattr(manager, "_cache") and manager._cache:
            manager._cache.clear()
            return JSONResponse(content={"message": "Cache cleared successfully"})
        return JSONResponse(content={"message": "Cache not enabled"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")


# Example client code
class SecretManagerClient:
    """Client for the secret manager service."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    async def get_secret(
        self,
        project_id: str,
        secret_name: str,
        version: str = "latest",
        environment: str | None = None,
        timeout: float = 30.0,
    ) -> str:
        """Get a secret from the service."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/secrets/{project_id}/{secret_name}",
                params={
                    "version": version,
                    "environment": environment,
                    "timeout": timeout,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["value"]

    async def get_health(self) -> dict[str, Any]:
        """Get service health status."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()


async def demo_client_usage() -> None:
    """Demonstrate client usage."""
    print("=== Client Demo ===")

    client = SecretManagerClient()

    try:
        # Check health
        health = await client.get_health()
        print(f"Service health: {health['status']}")

        # Get a secret
        secret_value = await client.get_secret(
            project_id="demo-project",
            secret_name="test-secret",
            environment="development",
        )
        print(f"Retrieved secret (length: {len(secret_value)})")

    except Exception as e:
        print(f"Client error: {e}")


if __name__ == "__main__":
    import uvicorn

    print("Starting Secret Manager Service...")
    print("API documentation available at: http://localhost:8000/docs")

    # Set up environment
    os.environ.setdefault("LOG_LEVEL", "20")
    os.environ.setdefault("ENVIRONMENT", "development")

    # Run the service
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,  # Set to True for development
    )
