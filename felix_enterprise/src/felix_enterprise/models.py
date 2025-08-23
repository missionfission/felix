"""
Pydantic models for Felix Enterprise Suite API.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class JobState(str, Enum):
    """Job execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HardwareType(str, Enum):
    """Supported hardware types."""
    AMD_ROCM = "amd_rocm"
    NVIDIA_CUDA = "nvidia_cuda"
    INTEL_GPU = "intel_gpu"
    CPU = "cpu"
    EDGE_TPU = "edge_tpu"


class OptimizationLevel(str, Enum):
    """Optimization intensity levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class ModelFormat(str, Enum):
    """Supported model formats."""
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    TENSORRT = "tensorrt"


class HardwareTarget(BaseModel):
    """Hardware target specification."""
    type: HardwareType
    device_id: Optional[int] = None
    compute_capability: Optional[str] = None
    memory_gb: Optional[float] = None
    cores: Optional[int] = None
    frequency_mhz: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class OptimizationConfig(BaseModel):
    """Optimization configuration parameters."""
    level: OptimizationLevel = OptimizationLevel.BALANCED
    target_latency_ms: Optional[float] = None
    target_throughput: Optional[float] = None
    batch_size: int = 1
    precision: str = "fp32"  # fp32, fp16, int8
    enable_tensorcore: bool = False
    enable_dynamic_shapes: bool = False
    max_optimization_time_minutes: int = 60
    custom_params: Dict[str, Any] = Field(default_factory=dict)

    @validator('precision')
    def validate_precision(cls, v):
        allowed = ['fp32', 'fp16', 'int8', 'mixed']
        if v not in allowed:
            raise ValueError(f'precision must be one of {allowed}')
        return v


class PerformanceMetrics(BaseModel):
    """Performance metrics for optimized models."""
    latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    gpu_utilization_percent: Optional[float] = None
    power_usage_watts: Optional[float] = None
    accuracy_drop_percent: float = 0.0
    speedup_factor: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class JobStatus(BaseModel):
    """Job status information."""
    job_id: UUID
    state: JobState
    progress_percent: int = Field(ge=0, le=100)
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Optional[PerformanceMetrics] = None

    class Config:
        use_enum_values = True


class OptimizationJob(BaseModel):
    """Optimization job specification."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    model_path: str
    model_format: ModelFormat
    hardware_target: HardwareTarget
    optimization_config: OptimizationConfig
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    tenant_id: str
    status: JobStatus = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        if self.status is None:
            self.status = JobStatus(
                job_id=self.id,
                state=JobState.PENDING,
                progress_percent=0,
                message="Job created"
            )

    class Config:
        use_enum_values = True


class OptimizationResult(BaseModel):
    """Results of an optimization job."""
    job_id: UUID
    optimized_model_path: str
    original_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    optimization_log: List[str] = Field(default_factory=list)
    artifacts: Dict[str, str] = Field(default_factory=dict)  # artifact_name -> path
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SystemHealth(BaseModel):
    """System health status."""
    status: str  # healthy, degraded, unhealthy
    version: str
    uptime_seconds: int
    active_jobs: int
    queued_jobs: int
    failed_jobs_last_hour: int
    available_backends: List[str]
    resource_usage: Dict[str, float]  # cpu, memory, gpu percentages
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TenantInfo(BaseModel):
    """Tenant information."""
    id: str
    name: str
    created_at: datetime
    is_active: bool = True
    resource_limits: Dict[str, int] = Field(default_factory=dict)
    usage_stats: Dict[str, float] = Field(default_factory=dict)


class UserInfo(BaseModel):
    """User information."""
    id: str
    username: str
    email: str
    tenant_id: str
    roles: List[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class APIKeyInfo(BaseModel):
    """API key information."""
    key_id: str
    name: str
    tenant_id: str
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    permissions: List[str] = Field(default_factory=list)
    last_used: Optional[datetime] = None