# Multi-Environment Secret Manager

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy.readthedocs.io/)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

A Python library for secure, scalable, and efficient access to Google Cloud Secret Manager across multiple environments. Built with modern Python 3.13+ features, enterprise-grade patterns, and built-in observability.

## 🚀 Features

### Core Capabilities
- **Multi-Environment Support**: Seamless secret management across development, staging, and production
- **Multiple Authentication Methods**: Service accounts, application default credentials, and impersonation
- **High Performance**: Connection pooling, intelligent caching, and async/await support
- **Reliability**: Circuit breaker pattern, retry logic, and graceful error handling
- **Security**: Environment isolation, credential management, and audit logging
- **Observability**: Prometheus metrics, structured logging, and distributed tracing

### Advanced Features
- **Smart Caching**: TTL-based caching with environment isolation
- **Connection Pooling**: Efficient resource management and connection reuse
- **Circuit Breaking**: Automatic failure detection and recovery
- **Batch Operations**: High-throughput bulk secret access
- **Type Safety**: Full type annotations and strict typing support
- **Configuration Management**: YAML, environment variables, and programmatic configuration

## 📋 Requirements

- **Python**: 3.13 or higher
- **Google Cloud**: Secret Manager API enabled
- **Dependencies**: See [pyproject.toml](pyproject.toml)

## 🛠 Installation

### Installation

```bash
git clone github.com/khodaparastan/mesm && cd mesm
poetry install
```


## 🚀 Quick Start

### Basic Usage

```python
from mesm import (
    MultiEnvironmentSecretManager,
    SecretConfig,
    environment_context
)

# Initialize with default configuration
async with MultiEnvironmentSecretManager() as manager:
    # Access a secret
    config = SecretConfig(
        project_id="my-project",
        secret_name="database-password",
        secret_version="latest"
    )

    secret_value = await manager.access_secret_version_async(config)
    print(f"Retrieved secret: {len(secret_value)} characters")
```

### Environment-Specific Access

```python
# Using environment context
with environment_context("production"):
    config = SecretConfig(
        project_id="prod-project",
        secret_name="api-key"
    )
    secret = await manager.access_secret_version_async(config)

# Using environment-specific manager
prod_manager = manager.create_environment_specific_manager("production")
secret = await prod_manager.access_secret_async(
    project_id="prod-project",
    secret_name="api-key"
)
```

### Configuration Examples

#### YAML Configuration

```yaml
# config.yaml
environments:
  development:
    name: development
    default_credential:
      method: APPLICATION_DEFAULT
    cache_ttl_seconds: 60
    timeout_seconds: 10.0

  production:
    name: production
    default_credential:
      method: SERVICE_ACCOUNT_FILE
      service_account_path: /path/to/service-account.json
    allowed_projects:
      - my-prod-project
    cache_ttl_seconds: 300
    timeout_seconds: 30.0

default_environment: development
enable_caching: true
enable_connection_pooling: true
strict_environment_isolation: true
```

```python
from mesm import MultiEnvironmentSecretManagerConfig

config = MultiEnvironmentSecretManagerConfig.from_yaml("config.yaml")
manager = MultiEnvironmentSecretManager(config)
```

#### Environment Variables

```bash
# Set environment-specific service accounts
export PRODUCTION_SERVICE_ACCOUNT_PATH="/path/to/prod-sa.json"
export STAGING_SERVICE_ACCOUNT_PATH="/path/to/staging-sa.json"
export DEFAULT_ENVIRONMENT="production"
export ENABLE_CACHING="true"
export CACHE_TTL="300"
```

```python
# Load from environment variables
config = MultiEnvironmentSecretManagerConfig.from_env()
manager = MultiEnvironmentSecretManager(config)
```

#### Programmatic Configuration

```python
from mesm import (
    MultiEnvironmentSecretManagerConfig,
    EnvironmentConfig,
    CredentialConfig,
    CredentialMethod
)

config = MultiEnvironmentSecretManagerConfig(
    environments={
        "development": EnvironmentConfig(
            name="development",
            default_credential=CredentialConfig(
                method=CredentialMethod.APPLICATION_DEFAULT
            ),
            cache_ttl_seconds=60,
            timeout_seconds=10.0,
        ),
        "production": EnvironmentConfig(
            name="production",
            default_credential=CredentialConfig(
                method=CredentialMethod.SERVICE_ACCOUNT_FILE,
                service_account_path=Path("/path/to/service-account.json")
            ),
            allowed_projects=["my-prod-project"],
            cache_ttl_seconds=300,
            timeout_seconds=30.0,
        ),
    },
    default_environment="development",
    enable_caching=True,
    enable_connection_pooling=True,
    strict_environment_isolation=True,
)

manager = MultiEnvironmentSecretManager(config)
```

## 📚 Advanced Usage

### Batch Operations

```python
from mesm import SecretConfig

# Define multiple secrets
secret_configs = [
    SecretConfig(project_id="project1", secret_name="secret1"),
    SecretConfig(project_id="project2", secret_name="secret2"),
    SecretConfig(project_id="project3", secret_name="secret3"),
]

# Access concurrently
async def access_secrets_batch():
    tasks = [
        manager.access_secret_version_async(config)
        for config in secret_configs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

secrets = await access_secrets_batch()
```

### Error Handling

```python
from mesm import (
    SecretNotFoundError,
    SecretAccessError,
    ConfigurationError
)

try:
    secret = await manager.access_secret_version_async(config)
except SecretNotFoundError as e:
    print(f"Secret not found: {e}")
    # Handle missing secret
except SecretAccessError as e:
    print(f"Access denied: {e}")
    # Handle permission issues
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle configuration problems
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

### Performance Monitoring

```python
# Get manager statistics
stats = manager.get_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
print(f"Circuit breaker states: {stats['circuit_breakers']}")

# Access cache statistics
if manager._cache:
    cache_stats = manager._cache.get_stats()
    print(f"Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"Cache hits: {cache_stats['hits']}")
    print(f"Cache misses: {cache_stats['misses']}")
```

### Custom Credential Configuration

```python
# Service Account File
credential_config = CredentialConfig(
    method=CredentialMethod.SERVICE_ACCOUNT_FILE,
    service_account_path=Path("/path/to/service-account.json"),
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Service Account JSON
credential_config = CredentialConfig(
    method=CredentialMethod.SERVICE_ACCOUNT_JSON,
    service_account_json={
        "type": "service_account",
        "project_id": "my-project",
        "private_key_id": "key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\n...",
        "client_email": "service-account@my-project.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
)

# Service Account Impersonation
credential_config = CredentialConfig(
    method=CredentialMethod.IMPERSONATION,
    impersonate_service_account="target-sa@my-project.iam.gserviceaccount.com",
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
```

## 🌐 Web Service Integration

### FastAPI Integration

```python
from fastapi import FastAPI, Depends, HTTPException
from mesm import MultiEnvironmentSecretManager

app = FastAPI()
manager: MultiEnvironmentSecretManager = None

@app.on_event("startup")
async def startup():
    global manager
    manager = MultiEnvironmentSecretManager()

@app.on_event("shutdown")
async def shutdown():
    if manager:
        await manager.close_async()

async def get_secret_manager() -> MultiEnvironmentSecretManager:
    if manager is None:
        raise HTTPException(status_code=500, detail="Secret manager not initialized")
    return manager

@app.get("/secrets/{project_id}/{secret_name}")
async def get_secret(
    project_id: str,
    secret_name: str,
    version: str = "latest",
    environment: str = None,
    manager: MultiEnvironmentSecretManager = Depends(get_secret_manager)
):
    try:
        config = SecretConfig(
            project_id=project_id,
            secret_name=secret_name,
            secret_version=version,
            environment=environment
        )
        secret_value = await manager.access_secret_version_async(config)
        return {"value": secret_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Django Integration

```python
# settings.py
from mesm import MultiEnvironmentSecretManager, SecretConfig

# Initialize secret manager
SECRET_MANAGER = MultiEnvironmentSecretManager()

# Helper function for Django settings
def get_secret(project_id: str, secret_name: str, default=None):
    try:
        config = SecretConfig(
            project_id=project_id,
            secret_name=secret_name
        )
        return SECRET_MANAGER.access_secret_version(config)
    except Exception:
        if default is not None:
            return default
        raise

# Use in settings
DATABASE_PASSWORD = get_secret("my-project", "database-password")
SECRET_KEY = get_secret("my-project", "django-secret-key")
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
---
services:
  secret-manager:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=20
      - DEFAULT_ENVIRONMENT=production
      - ENABLE_CACHING=true
      - GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/service-account.json
    volumes:
      - ./secrets:/app/secrets:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## ☸️ Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secret-manager
  labels:
    app: secret-manager
spec:
  replicas: 3
  selector:
    matchLabels:
      app: secret-manager
  template:
    metadata:
      labels:
        app: secret-manager
    spec:
      serviceAccountName: secret-manager-sa
      containers:
      - name: secret-manager
        image: secret-manager:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "20"
        - name: DEFAULT_ENVIRONMENT
          value: "production"
        - name: ENABLE_CACHING
          value: "true"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: secret-manager-service
spec:
  selector:
    app: secret-manager
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Service Account Setup

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: secret-manager-sa
  annotations:
    iam.gke.io/gcp-service-account: secret-manager@my-project.iam.gserviceaccount.com
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: secret-manager-role
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: secret-manager-binding
subjects:
- kind: ServiceAccount
  name: secret-manager-sa
  namespace: default
roleRef:
  kind: ClusterRole
  name: secret-manager-role
  apiGroup: rbac.authorization.k8s.io
```

## 📊 Monitoring & Observability

### Prometheus Metrics

The library exposes comprehensive metrics for monitoring:

```python
# Operation metrics
secret_operations_total{operation, project_id, environment, status, error_type}
secret_operation_duration_seconds{operation, project_id, environment}

# Cache metrics (available via /stats endpoint)
cache_size
cache_hits
cache_misses
cache_hit_rate

# Circuit breaker metrics
circuit_breaker_state{environment}
circuit_breaker_failures{environment}
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Secret Manager Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(secret_operations_total[5m])"
          }
        ]
      },
      {
        "title": "Success Rate",
        "targets": [
          {
            "expr": "rate(secret_operations_total{status=\"success\"}[5m]) / rate(secret_operations_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(secret_operation_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Structured Logging

```python
import structlog

# Configure structured logging
logger = structlog.get_logger()

# Logs include context
logger.info(
    "secret_accessed",
    project_id="my-project",
    secret_name="database-password",
    environment="production",
    duration=0.123,
    cache_hit=True
)
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mesm --cov-report=html

# Run specific test categories
pytest -m "not integration"  # Skip integration tests
pytest -m "async"            # Run only async tests
```

### Integration Tests

```bash
# Set up test environment
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/test-service-account.json"
export TEST_PROJECT_ID="my-test-project"

# Run integration tests
pytest -m integration
```

### Load Testing

```python
import asyncio
from mesm import MultiEnvironmentSecretManager, SecretConfig

async def load_test():
    manager = MultiEnvironmentSecretManager()

    # Create test configurations
    configs = [
        SecretConfig(
            project_id="test-project",
            secret_name=f"test-secret-{i}"
        )
        for i in range(100)
    ]

    # Measure performance
    start_time = time.time()

    tasks = [
        manager.access_secret_version_async(config)
        for config in configs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start_time
    successful = sum(1 for r in results if not isinstance(r, Exception))

    print(f"Processed {len(configs)} secrets in {duration:.2f}s")
    print(f"Success rate: {successful/len(configs):.2%}")
    print(f"Throughput: {len(configs)/duration:.2f} secrets/second")

asyncio.run(load_test())
```

## 🔧 Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_ENVIRONMENT` | Default environment name | `production` |
| `LOG_LEVEL` | Logging level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR) | `20` |
| `ENABLE_CACHING` | Enable secret caching | `true` |
| `ENABLE_POOLING` | Enable connection pooling | `true` |
| `CACHE_TTL` | Cache TTL in seconds | `300` |
| `CACHE_SIZE` | Maximum cache size | `1000` |
| `MAX_CONNECTIONS` | Max connections per credential | `5` |
| `STRICT_ISOLATION` | Enforce strict environment isolation | `true` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON file | - |
| `{ENV}_SERVICE_ACCOUNT_PATH` | Environment-specific service account path | - |
| `{ENV}_SERVICE_ACCOUNT_JSON` | Environment-specific service account JSON | - |

### Configuration Schema

```python
class MultiEnvironmentSecretManagerConfig(BaseModel):
    environments: dict[str, EnvironmentConfig] = Field(default_factory=dict)
    default_environment: str = Field(default="production")
    enable_caching: bool = Field(default=True)
    enable_connection_pooling: bool = Field(default=True)
    max_connections_per_credential: int = Field(default=5, ge=1, le=50)
    strict_environment_isolation: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
    cache_max_size: int = Field(default=1000, ge=1, le=10000)

class EnvironmentConfig(BaseModel):
    name: str
    default_credential: CredentialConfig = Field(default_factory=CredentialConfig)
    project_credentials: dict[str, CredentialConfig] = Field(default_factory=dict)
    allowed_projects: list[str] | None = Field(default=None)
    cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)

class CredentialConfig(BaseModel):
    method: CredentialMethod = Field(default=CredentialMethod.APPLICATION_DEFAULT)
    service_account_path: Path | None = Field(default=None)
    service_account_json: dict[str, Any] | None = Field(default=None)
    impersonate_service_account: str | None = Field(default=None)
    scopes: list[str] = Field(default_factory=lambda: ["https://www.googleapis.com/auth/cloud-platform"])

class SecretConfig(BaseModel):
    project_id: str = Field(..., pattern=r"^[a-z][a-z0-9\-]*[a-z0-9]$")
    secret_name: str = Field(..., pattern=r"^[a-zA-Z][a-zA-Z0-9_\-]*$")
    secret_version: str | int = Field("latest")
    environment: str | None = Field(default=None)
    credential_override: CredentialConfig | None = Field(default=None)
```

## 🔒 Security Considerations

### Authentication & Authorization

1. **Service Account Permissions**: Ensure service accounts have minimal required permissions:
   ```bash
   # Grant Secret Manager Secret Accessor role
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:SA_EMAIL" \
     --role="roles/secretmanager.secretAccessor"
   ```

2. **Environment Isolation**: Use separate service accounts for different environments:
   ```yaml
   environments:
     production:
       default_credential:
         method: SERVICE_ACCOUNT_FILE
         service_account_path: /secrets/prod-sa.json
     staging:
       default_credential:
         method: SERVICE_ACCOUNT_FILE
         service_account_path: /secrets/staging-sa.json
   ```

3. **Project Restrictions**: Limit access to specific projects:
   ```yaml
   environments:
     production:
       allowed_projects:
         - my-prod-project-1
         - my-prod-project-2
   ```

### Best Practices

1. **Credential Rotation**: Regularly rotate service account keys
2. **Audit Logging**: Enable Cloud Audit Logs for Secret Manager
3. **Network Security**: Use VPC Service Controls when possible
4. **Secret Rotation**: Implement automatic secret rotation
5. **Monitoring**: Set up alerts for unusual access patterns

## 🚨 Troubleshooting

### Common Issues

#### Authentication Errors

```python
# Error: DefaultCredentialsError
# Solution: Set up application default credentials
gcloud auth application-default login

# Or set service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

#### Permission Denied

```python
# Error: 403 Permission denied
# Solution: Grant appropriate IAM roles
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:SA_EMAIL" \
  --role="roles/secretmanager.secretAccessor"
```

#### Secret Not Found

```python
# Error: 404 Secret not found
# Solution: Verify secret exists and name is correct
gcloud secrets list --project=PROJECT_ID
gcloud secrets versions list SECRET_NAME --project=PROJECT_ID
```

#### Circuit Breaker Open

```python
# Error: Circuit breaker open
# Solution: Check underlying issues and wait for recovery
stats = manager.get_stats()
print(stats['circuit_breakers'])

# Or manually reset (if needed)
# manager._circuit_breakers[environment].record_success()
```

### Debugging

#### Enable Debug Logging

```python
import logging
import os

# Set debug level
os.environ["LOG_LEVEL"] = "10"  # DEBUG

# Or programmatically
logging.getLogger().setLevel(logging.DEBUG)
```

#### Performance Analysis

```python
# Get detailed statistics
stats = manager.get_stats()
print(json.dumps(stats, indent=2, default=str))

# Monitor cache performance
if manager._cache:
    cache_stats = manager._cache.get_stats()
    print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
```

#### Connection Pool Analysis

```python
# Check pool statistics
for pool_key, pool in manager._client_pool._sync_pools.items():
    pool_stats = pool.get_stats()
    print(f"Pool {pool_key}: {pool_stats}")
```

## 📈 Performance Tuning

### Cache Optimization

```python
# Tune cache settings based on usage patterns
config = MultiEnvironmentSecretManagerConfig(
    cache_ttl_seconds=600,      # Longer TTL for stable secrets
    cache_max_size=5000,        # Larger cache for high-volume applications
)

# Environment-specific cache settings
environments = {
    "production": EnvironmentConfig(
        cache_ttl_seconds=1800,  # 30 minutes for production
    ),
    "development": EnvironmentConfig(
        cache_ttl_seconds=60,    # 1 minute for development
    ),
}
```

### Connection Pool Tuning

```python
# Optimize connection pool size
config = MultiEnvironmentSecretManagerConfig(
    max_connections_per_credential=10,  # Higher for high-concurrency apps
    enable_connection_pooling=True,
)
```

### Async Optimization

```python
# Use semaphores to limit concurrent requests
semaphore = asyncio.Semaphore(50)  # Limit to 50 concurrent requests

async def access_secret_with_limit(config):
    async with semaphore:
        return await manager.access_secret_version_async(config)
```

## 🤝 Contributing

We welcome contributions!

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/multi-environment-secret-manager.git
cd multi-environment-secret-manager

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black .
mypy .
bandit -r mesm/
```

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **MyPy**: Type checking
- **Bandit**: Security analysis
- **Pytest**: Testing
- **Pre-commit**: Git hooks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation

- [Architecture Documentation](docs/architecture.md)
- API Reference
- [Examples](examples/)
- Deployment Guidee

### Community

- [GitHub Issues](https://github.com/khodaparastan/mesm/issues)
- [Discussions](https://github.com/github.com/khodaparastan/mesm/discussions)

### Enterprise Support

For enterprise support, custom integrations, or consulting services, please contact [mesm@mkh.contact](mailto:mesm@mkh.contact).

## 🗺 Roadmap

### Version 0.1.0 (Q2 2025)
- [ ] Redis-based distributed caching
- [ ] Vault integration support
- [ ] Enhanced metrics and dashboards
- [ ] Secret rotation automation

### Version 0.2.0 (Q3 2025)
- [ ] AWS Secrets Manager support
- [ ] Azure Key Vault integration
- [ ] Multi-cloud secret federation
- [ ] Advanced RBAC features

### Version 1.0.0 (Q4 2025)
- [ ] GraphQL API support
- [ ] Event-driven secret updates
- [ ] Machine learning-based anomaly detection
- [ ] Zero-trust security model

## 📊 Changelog

### [0.1.0] - 2025-6-5

#### Added
- Initial release with full multi-environment support
- Google Cloud Secret Manager integration
- Connection pooling and caching
- Circuit breaker pattern implementation
- Comprehensive monitoring and observability
- FastAPI and Django integration examples
- Docker and Kubernetes deployment support

#### Security
- Environment isolation and strict access controls
- Comprehensive audit logging
- Service account impersonation support

#### Performance
- Async/await support throughout
- Intelligent caching with TTL management
- Connection pooling for optimal resource usage
- Batch operations for high-throughput scenarios

---

**Built with ❤️ for the Python community**
