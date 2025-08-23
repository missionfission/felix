"""
Felix Enterprise Suite API Gateway

Main FastAPI application providing RESTful API for optimization services.
"""

from contextlib import asynccontextmanager
from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ..auth import get_current_user, get_api_key
from ..config import get_settings
from ..database import get_db_session
from ..exceptions import FelixEnterpriseError, OptimizationError, HardwareError
from ..models import (
    OptimizationJob,
    JobStatus,
    OptimizationResult,
    HardwareTarget,
    SystemHealth,
    TenantInfo,
    UserInfo,
)
from ..services import (
    OptimizationService,
    HardwareService,
    MonitoringService,
    TenantService,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Felix Enterprise Suite API")
    
    # Initialize services
    app.state.optimization_service = OptimizationService()
    app.state.hardware_service = HardwareService()
    app.state.monitoring_service = MonitoringService()
    app.state.tenant_service = TenantService()
    
    # Start background services
    await app.state.monitoring_service.start()
    
    yield
    
    # Cleanup
    logger.info("Shutting down Felix Enterprise Suite API")
    await app.state.monitoring_service.stop()


# Create FastAPI application
app = FastAPI(
    title="Felix Enterprise Suite API",
    description="Hardware-aware deep learning optimization platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Settings
settings = get_settings()


# Exception handlers
@app.exception_handler(FelixEnterpriseError)
async def felix_exception_handler(request, exc: FelixEnterpriseError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "type": type(exc).__name__}
    )


@app.exception_handler(OptimizationError)
async def optimization_exception_handler(request, exc: OptimizationError):
    return JSONResponse(
        status_code=422,
        content={"error": str(exc), "type": "OptimizationError"}
    )


@app.exception_handler(HardwareError)
async def hardware_exception_handler(request, exc: HardwareError):
    return JSONResponse(
        status_code=503,
        content={"error": str(exc), "type": "HardwareError"}
    )


# Health check endpoints
@app.get("/health", response_model=SystemHealth)
async def health_check():
    """System health check."""
    return await app.state.monitoring_service.get_system_health()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


# Authentication endpoints
@app.post("/auth/login")
async def login(credentials: dict):
    """User authentication."""
    # Implementation would integrate with enterprise auth systems
    pass


@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    """Refresh authentication token."""
    pass


# Optimization job endpoints
@app.post("/jobs", response_model=OptimizationJob, status_code=status.HTTP_201_CREATED)
async def create_optimization_job(
    job_data: dict,
    background_tasks: BackgroundTasks,
    current_user: UserInfo = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """Create a new optimization job."""
    try:
        job = await app.state.optimization_service.create_job(
            job_data, current_user.tenant_id, current_user.id, db_session
        )
        
        # Start optimization in background
        background_tasks.add_task(
            app.state.optimization_service.execute_job, job.id
        )
        
        logger.info("Created optimization job", job_id=str(job.id), user_id=current_user.id)
        return job
        
    except Exception as e:
        logger.error("Failed to create optimization job", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/jobs", response_model=List[OptimizationJob])
async def list_jobs(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
    current_user: UserInfo = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """List optimization jobs for the current tenant."""
    return await app.state.optimization_service.list_jobs(
        current_user.tenant_id, skip, limit, status_filter, db_session
    )


@app.get("/jobs/{job_id}", response_model=OptimizationJob)
async def get_job(
    job_id: UUID,
    current_user: UserInfo = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """Get optimization job details."""
    job = await app.state.optimization_service.get_job(
        job_id, current_user.tenant_id, db_session
    )
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(
    job_id: UUID,
    current_user: UserInfo = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """Get job status."""
    status = await app.state.optimization_service.get_job_status(
        job_id, current_user.tenant_id, db_session
    )
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.get("/jobs/{job_id}/result", response_model=OptimizationResult)
async def get_job_result(
    job_id: UUID,
    current_user: UserInfo = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """Get optimization results."""
    result = await app.state.optimization_service.get_job_result(
        job_id, current_user.tenant_id, db_session
    )
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: UUID,
    current_user: UserInfo = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """Cancel a running job."""
    success = await app.state.optimization_service.cancel_job(
        job_id, current_user.tenant_id, db_session
    )
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    return {"message": "Job cancelled successfully"}


# Hardware management endpoints
@app.get("/hardware/targets", response_model=List[HardwareTarget])
async def list_hardware_targets(
    current_user: UserInfo = Depends(get_current_user)
):
    """List available hardware targets."""
    return await app.state.hardware_service.get_available_targets()


@app.get("/hardware/backends")
async def list_hardware_backends(
    current_user: UserInfo = Depends(get_current_user)
):
    """List available optimization backends."""
    return await app.state.hardware_service.get_available_backends()


@app.post("/hardware/benchmark")
async def benchmark_hardware(
    hardware_target: HardwareTarget,
    current_user: UserInfo = Depends(get_current_user)
):
    """Run hardware benchmark."""
    return await app.state.hardware_service.run_benchmark(hardware_target)


# Monitoring endpoints
@app.get("/monitoring/jobs/stats")
async def get_job_statistics(
    current_user: UserInfo = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """Get job execution statistics."""
    return await app.state.monitoring_service.get_job_stats(
        current_user.tenant_id, db_session
    )


@app.get("/monitoring/performance")
async def get_performance_metrics(
    current_user: UserInfo = Depends(get_current_user)
):
    """Get system performance metrics."""
    return await app.state.monitoring_service.get_performance_metrics()


# Tenant management endpoints (admin only)
@app.get("/admin/tenants", response_model=List[TenantInfo])
async def list_tenants(
    current_user: UserInfo = Depends(get_current_user)
):
    """List all tenants (admin only)."""
    if "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Admin access required")
    return await app.state.tenant_service.list_tenants()


@app.post("/admin/tenants", response_model=TenantInfo)
async def create_tenant(
    tenant_data: dict,
    current_user: UserInfo = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """Create new tenant (admin only)."""
    if "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Admin access required")
    return await app.state.tenant_service.create_tenant(tenant_data, db_session)


# API key management
@app.post("/api-keys")
async def create_api_key(
    key_data: dict,
    current_user: UserInfo = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """Create new API key."""
    return await app.state.tenant_service.create_api_key(
        key_data, current_user.tenant_id, current_user.id, db_session
    )


@app.get("/api-keys")
async def list_api_keys(
    current_user: UserInfo = Depends(get_current_user),
    db_session = Depends(get_db_session)
):
    """List API keys for current user."""
    return await app.state.tenant_service.list_api_keys(
        current_user.tenant_id, current_user.id, db_session
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "felix_enterprise.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )