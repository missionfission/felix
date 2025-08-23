# Felix Enterprise Suite - Deployment Guide

## Overview

Felix Enterprise Suite is a production-ready, hardware-aware deep learning optimization platform designed for enterprise deployment. This guide covers installation, configuration, security, and operational best practices.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [Security Setup](#security-setup)
5. [Hardware Integration](#hardware-integration)
6. [Monitoring & Observability](#monitoring--observability)
7. [Scaling & High Availability](#scaling--high-availability)
8. [Backup & Disaster Recovery](#backup--disaster-recovery)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Ubuntu 20.04 LTS, RHEL 8+, or CentOS 8+
- **CPU**: 8 cores, 2.5 GHz
- **Memory**: 32 GB RAM
- **Storage**: 500 GB SSD (for system and models)
- **Network**: 1 Gbps network connection
- **Python**: 3.8 or higher

### Recommended for Production

- **Operating System**: Ubuntu 22.04 LTS
- **CPU**: 16+ cores, 3.0+ GHz (Intel Xeon or AMD EPYC)
- **Memory**: 64+ GB RAM
- **Storage**: 2+ TB NVMe SSD
- **Network**: 10 Gbps network connection
- **GPU**: AMD Instinct MI100/MI200 or NVIDIA A100/H100

### Hardware-Specific Requirements

#### AMD ROCm Setup
```bash
# Install ROCm (Ubuntu)
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.4/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms rocm-dev rocm-libs rocm-utils
```

#### NVIDIA CUDA Setup
```bash
# Install CUDA (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

## Installation Methods

### Method 1: Docker Deployment (Recommended)

#### Prerequisites
```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### Quick Start
```bash
# Clone the repository
git clone https://github.com/felix-technologies/felix-enterprise.git
cd felix-enterprise

# Copy and customize environment configuration
cp .env.example .env
vim .env  # Edit configuration

# Start the platform
docker-compose up -d

# Check status
docker-compose ps
```

#### Production Deployment
```bash
# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale workers based on workload
docker-compose up -d --scale felix-worker=4
```

### Method 2: Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (1.23+)
- Helm 3.0+
- kubectl configured

#### Installation
```bash
# Add Felix Helm repository
helm repo add felix-enterprise https://charts.felix-enterprise.com
helm repo update

# Install with custom values
helm install felix-enterprise felix-enterprise/felix-enterprise \
  --namespace felix-system \
  --create-namespace \
  --values values-production.yaml

# Check deployment status
kubectl get pods -n felix-system
```

#### Sample Kubernetes Values (values-production.yaml)
```yaml
# Felix Enterprise Helm Values
global:
  environment: production
  imageRegistry: registry.felix-enterprise.com
  
api:
  replicas: 3
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
  
worker:
  replicas: 5
  resources:
    requests:
      cpu: "4"
      memory: "8Gi"
      nvidia.com/gpu: 1
    limits:
      cpu: "8"
      memory: "16Gi"
      nvidia.com/gpu: 1

database:
  enabled: true
  type: postgresql
  size: 100Gi
  replicas: 3
  
redis:
  enabled: true
  cluster:
    enabled: true
    nodes: 6
```

### Method 3: Native Installation

#### System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3-pip \
    postgresql-13 redis-server nginx \
    build-essential cmake git

# RHEL/CentOS
sudo dnf install -y python38 python38-pip python38-devel \
    postgresql13-server redis nginx \
    gcc-c++ cmake git
```

#### Felix Enterprise Installation
```bash
# Create application user
sudo useradd -m -s /bin/bash felix
sudo usermod -aG sudo felix

# Switch to felix user
sudo su - felix

# Create virtual environment
python3.8 -m venv /home/felix/felix-enterprise
source /home/felix/felix-enterprise/bin/activate

# Install Felix Enterprise
pip install felix-enterprise[amd,cuda]

# Initialize database
felix-enterprise init-db

# Start services
felix-server --config /etc/felix/production.yaml
```

## Configuration

### Environment Variables

Create `/etc/felix/config.env`:
```bash
# Database Configuration
DATABASE_URL=postgresql://felix:secure_password@localhost:5432/felix_enterprise
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1

# Security
JWT_SECRET_KEY=your-super-secure-secret-key-here
API_KEY_ENCRYPTION_KEY=another-secure-key-for-api-keys
ALLOWED_HOSTS=api.yourcompany.com,felix.yourcompany.com

# Hardware Configuration
ROCM_PATH=/opt/rocm
CUDA_VISIBLE_DEVICES=0,1,2,3
MAX_CONCURRENT_OPTIMIZATIONS=4

# Monitoring
PROMETHEUS_ENABLED=true
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Storage
MODEL_STORAGE_PATH=/data/models
ARTIFACT_STORAGE_BACKEND=s3
AWS_S3_BUCKET=felix-enterprise-artifacts
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/felix/felix.log
```

### Application Configuration

Create `/etc/felix/production.yaml`:
```yaml
# Felix Enterprise Production Configuration
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  worker_class: "uvicorn.workers.UvicornWorker"
  
database:
  pool_size: 20
  max_overflow: 30
  pool_pre_ping: true
  echo: false
  
optimization:
  max_concurrent_jobs: 10
  default_timeout_minutes: 60
  cleanup_completed_jobs_days: 30
  
hardware:
  auto_detect: true
  preferred_backends:
    - amd_rocm
    - nvidia_cuda
    - intel_gpu
  
security:
  cors_origins:
    - "https://dashboard.yourcompany.com"
    - "https://api.yourcompany.com"
  rate_limiting:
    enabled: true
    requests_per_minute: 1000
  
monitoring:
  metrics_enabled: true
  health_check_interval: 30
  log_level: "INFO"
```

## Security Setup

### SSL/TLS Configuration

#### Generate SSL Certificates
```bash
# Using Let's Encrypt (recommended)
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourcompany.com

# Or use your enterprise CA
# Copy certificates to /etc/felix/ssl/
```

#### Nginx Configuration
Create `/etc/nginx/sites-available/felix-enterprise`:
```nginx
# Felix Enterprise Nginx Configuration
upstream felix_api {
    least_conn;
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name api.yourcompany.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourcompany.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/api.yourcompany.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourcompany.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Proxy Configuration
    location / {
        proxy_pass http://felix_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # File upload limits
        client_max_body_size 1G;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://felix_api/health;
    }
}
```

### Authentication & Authorization

#### Enterprise SSO Integration
```python
# config/auth.py
AUTHENTICATION_BACKENDS = [
    'felix_enterprise.auth.backends.SAMLBackend',
    'felix_enterprise.auth.backends.OIDCBackend',
    'felix_enterprise.auth.backends.LDAPBackend',
]

SAML_CONFIG = {
    'sp_entity_id': 'https://api.yourcompany.com',
    'idp_sso_url': 'https://idp.yourcompany.com/sso',
    'idp_x509_cert': '/etc/felix/ssl/idp-cert.pem',
}

OIDC_CONFIG = {
    'client_id': 'felix-enterprise',
    'client_secret': 'your-oidc-secret',
    'discovery_url': 'https://auth.yourcompany.com/.well-known/openid_configuration',
}
```

### Network Security

#### Firewall Configuration
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP (redirect to HTTPS)
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL (internal)
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis (internal)
sudo ufw enable

# iptables (RHEL/CentOS)
firewall-cmd --permanent --add-service=http
firewall-cmd --permanent --add-service=https
firewall-cmd --permanent --add-port=22/tcp
firewall-cmd --reload
```

## Hardware Integration

### AMD ROCm Configuration

#### System Setup
```bash
# Add user to render group
sudo usermod -aG render,video felix

# Configure ROCm environment
echo 'export ROCM_PATH=/opt/rocm' >> /etc/environment
echo 'export HIP_PLATFORM=amd' >> /etc/environment
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> /etc/environment

# Verify installation
rocm-smi
rocminfo
```

#### Felix Configuration
```yaml
# AMD-specific configuration
hardware:
  amd_rocm:
    enabled: true
    devices: "auto"  # or specific device IDs: [0, 1, 2, 3]
    memory_fraction: 0.9
    optimization_presets:
      conservative:
        n_measurements: 512
        n_grad_steps: 128
      balanced:
        n_measurements: 1024
        n_grad_steps: 256
      aggressive:
        n_measurements: 2048
        n_grad_steps: 512
```

### NVIDIA CUDA Configuration

#### System Setup
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Install additional libraries
sudo apt install cuda-nvml-dev-11-8 cuda-nvrtc-11-8
```

#### Felix Configuration
```yaml
hardware:
  nvidia_cuda:
    enabled: true
    devices: "auto"
    enable_tensorcore: true
    memory_fraction: 0.8
    optimization_presets:
      tensorcore_optimized:
        precision: "fp16"
        enable_tensorcore: true
        batch_size_optimization: true
```

## Monitoring & Observability

### Prometheus Metrics

Felix Enterprise exposes comprehensive metrics:

```yaml
# Prometheus scrape configuration
scrape_configs:
  - job_name: 'felix-enterprise'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
    
  - job_name: 'felix-workers'
    static_configs:
      - targets: ['localhost:8001', 'localhost:8002']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Key Metrics to Monitor

- `felix_optimization_jobs_total` - Total optimization jobs
- `felix_optimization_duration_seconds` - Job completion time
- `felix_hardware_utilization_percent` - GPU/CPU utilization
- `felix_memory_usage_bytes` - Memory consumption
- `felix_api_requests_total` - API request count
- `felix_errors_total` - Error rates

### Grafana Dashboards

Import pre-built dashboards:
- Felix Enterprise Overview
- Hardware Performance
- Optimization Job Analytics
- System Health

### Log Management

#### Structured Logging Configuration
```yaml
logging:
  version: 1
  formatters:
    json:
      class: pythonjsonlogger.jsonlogger.JsonFormatter
      format: '%(asctime)s %(name)s %(levelname)s %(message)s'
  handlers:
    file:
      class: logging.handlers.RotatingFileHandler
      filename: /var/log/felix/felix.log
      maxBytes: 100000000  # 100MB
      backupCount: 10
      formatter: json
    elasticsearch:
      class: felix_enterprise.logging.ElasticsearchHandler
      hosts: ['elasticsearch:9200']
      index_name: felix-logs
      formatter: json
  root:
    level: INFO
    handlers: [file, elasticsearch]
```

## Scaling & High Availability

### Horizontal Scaling

#### API Servers
```bash
# Scale API servers
docker-compose up -d --scale felix-api=3

# Or with Kubernetes
kubectl scale deployment felix-api --replicas=5 -n felix-system
```

#### Worker Scaling
```bash
# Auto-scaling based on queue length
docker-compose up -d --scale felix-worker=8

# Kubernetes HPA
kubectl autoscale deployment felix-worker \
  --cpu-percent=70 \
  --min=2 \
  --max=20 \
  -n felix-system
```

### Database High Availability

#### PostgreSQL Cluster
```yaml
# PostgreSQL HA configuration
postgresql:
  replication:
    enabled: true
    master:
      replicas: 1
    slave:
      replicas: 2
  persistence:
    size: 500Gi
    storageClass: fast-ssd
  backup:
    enabled: true
    schedule: "0 2 * * *"
    retention: "30d"
```

### Load Balancing

#### HAProxy Configuration
```
# HAProxy load balancer
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend felix_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/felix.pem
    redirect scheme https if !{ ssl_fc }
    default_backend felix_backend

backend felix_backend
    balance roundrobin
    option httpchk GET /health
    server felix-api-1 10.0.1.10:8000 check
    server felix-api-2 10.0.1.11:8000 check
    server felix-api-3 10.0.1.12:8000 check
```

## Backup & Disaster Recovery

### Database Backup

#### Automated Backups
```bash
#!/bin/bash
# backup-felix-db.sh

BACKUP_DIR="/backup/felix"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="felix_enterprise"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -h localhost -U felix $DB_NAME | gzip > $BACKUP_DIR/felix_db_$DATE.sql.gz

# Model artifacts backup
tar -czf $BACKUP_DIR/models_$DATE.tar.gz /data/models/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

# Upload to S3
aws s3 cp $BACKUP_DIR/ s3://felix-backups/ --recursive --exclude "*" --include "*$DATE*"
```

#### Automated Schedule
```bash
# Add to crontab
0 2 * * * /usr/local/bin/backup-felix-db.sh
```

### Disaster Recovery

#### Recovery Procedures
```bash
# 1. Restore database
gunzip -c felix_db_YYYYMMDD_HHMMSS.sql.gz | psql -h localhost -U felix felix_enterprise

# 2. Restore model artifacts
tar -xzf models_YYYYMMDD_HHMMSS.tar.gz -C /

# 3. Restart services
systemctl restart felix-enterprise
```

#### RTO/RPO Targets
- **Recovery Time Objective (RTO)**: < 4 hours
- **Recovery Point Objective (RPO)**: < 1 hour

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check ROCm installation
rocm-smi
lsmod | grep amdgpu

# Check CUDA installation
nvidia-smi
lsmod | grep nvidia

# Verify Felix can access GPU
felix-enterprise check-hardware
```

#### Performance Issues
```bash
# Check resource usage
htop
nvidia-smi
rocm-smi

# Review logs
tail -f /var/log/felix/felix.log

# Check database performance
SELECT * FROM pg_stat_activity WHERE state = 'active';
```

#### Connection Issues
```bash
# Check service status
systemctl status felix-enterprise
docker-compose ps

# Test API connectivity
curl -k https://api.yourcompany.com/health

# Check database connectivity
psql -h localhost -U felix felix_enterprise -c "SELECT 1;"
```

### Debug Mode

Enable debug logging:
```yaml
logging:
  level: DEBUG
  debug_modules:
    - felix_enterprise.optimization
    - felix_enterprise.hardware
    - felix_enterprise.api
```

### Support Channels

- **Enterprise Support**: support@felix-enterprise.com
- **Documentation**: https://docs.felix-enterprise.com
- **Community Forum**: https://community.felix-enterprise.com
- **Emergency Hotline**: +1-800-FELIX-911

## Maintenance

### Regular Maintenance Tasks

#### Weekly
- Review system logs
- Check disk usage
- Validate backups
- Update security patches

#### Monthly
- Database maintenance (VACUUM, REINDEX)
- Performance review
- Capacity planning
- Security audit

#### Quarterly
- Disaster recovery testing
- Performance benchmarking
- Configuration review
- License compliance check

### Updates & Upgrades

#### Rolling Updates
```bash
# Docker deployment
docker-compose pull
docker-compose up -d --no-deps felix-api

# Kubernetes deployment
kubectl set image deployment/felix-api \
  felix-api=felix-enterprise:v1.1.0 \
  -n felix-system
```

#### Database Migrations
```bash
# Run migrations
felix-enterprise migrate

# Verify migration
felix-enterprise check-migrations
```

---

For additional support and enterprise features, contact Felix Technologies at enterprise@felix-technologies.com