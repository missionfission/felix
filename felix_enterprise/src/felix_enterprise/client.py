"""
Felix Enterprise Suite Python Client SDK

Provides a simple, enterprise-friendly interface for interacting with the
Felix optimization platform.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from uuid import UUID
import time

import httpx
import structlog
from .models import (
    OptimizationJob,
    JobStatus,
    JobState,
    HardwareTarget,
    HardwareType,
    OptimizationConfig,
    OptimizationLevel,
    ModelFormat,
    PerformanceMetrics,
    OptimizationResult,
)
from .exceptions import FelixEnterpriseError, OptimizationError, HardwareError

logger = structlog.get_logger()


class Client:
    """Felix Enterprise Suite Python Client."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize Felix Enterprise client.
        
        Args:
            base_url: Base URL of the Felix Enterprise API
            api_key: API key for authentication
            tenant_id: Tenant ID (if not provided, will be extracted from API key)
            timeout: Request timeout in seconds
            max_retries: Maximum number of request retries
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.timeout = timeout
        self.max_retries = max_retries
        
        # HTTP client configuration
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )
        
        logger.info("Felix Enterprise client initialized", base_url=base_url)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling and retries."""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = await self._client.get(url, params=params)
                elif method.upper() == "POST":
                    response = await self._client.post(url, json=data, params=params)
                elif method.upper() == "PUT":
                    response = await self._client.put(url, json=data, params=params)
                elif method.upper() == "DELETE":
                    response = await self._client.delete(url, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Handle HTTP errors
                if response.status_code >= 400:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("error", f"HTTP {response.status_code}")
                    
                    if response.status_code == 401:
                        raise FelixEnterpriseError("Authentication failed")
                    elif response.status_code == 403:
                        raise FelixEnterpriseError("Access denied")
                    elif response.status_code == 404:
                        raise FelixEnterpriseError("Resource not found")
                    elif response.status_code == 422:
                        raise OptimizationError(error_msg)
                    elif response.status_code == 503:
                        raise HardwareError(error_msg)
                    else:
                        raise FelixEnterpriseError(error_msg)
                
                return response.json()
                
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise FelixEnterpriseError(f"Request failed after {self.max_retries} retries: {str(e)}")
                
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.warning(f"Request failed, retrying in {wait_time}s", error=str(e))
                await asyncio.sleep(wait_time)
    
    # Health and system information
    async def health_check(self) -> Dict[str, Any]:
        """Check system health status."""
        return await self._request("GET", "/health")
    
    async def get_version(self) -> str:
        """Get API version information."""
        health = await self.health_check()
        return health.get("version", "unknown")
    
    # Hardware management
    async def list_hardware_targets(self) -> List[HardwareTarget]:
        """List available hardware targets."""
        data = await self._request("GET", "/hardware/targets")
        return [HardwareTarget(**item) for item in data]
    
    async def list_hardware_backends(self) -> List[str]:
        """List available optimization backends."""
        data = await self._request("GET", "/hardware/backends")
        return data.get("backends", [])
    
    async def benchmark_hardware(self, hardware_target: HardwareTarget) -> PerformanceMetrics:
        """Run hardware benchmark."""
        data = await self._request(
            "POST", 
            "/hardware/benchmark", 
            data=hardware_target.dict()
        )
        return PerformanceMetrics(**data)
    
    # Optimization jobs
    async def optimize_model(
        self,
        model_path: Union[str, Path],
        hardware_target: Union[HardwareTarget, str, HardwareType],
        name: Optional[str] = None,
        description: Optional[str] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
        model_format: Optional[ModelFormat] = None,
        **optimization_params
    ) -> OptimizationJob:
        """
        Submit a model for optimization.
        
        Args:
            model_path: Path to the model file
            hardware_target: Target hardware for optimization
            name: Job name (optional, will be auto-generated if not provided)
            description: Job description
            optimization_level: Optimization intensity level
            model_format: Model format (auto-detected if not provided)
            **optimization_params: Additional optimization parameters
            
        Returns:
            OptimizationJob instance
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Auto-detect model format if not provided
        if model_format is None:
            model_format = self._detect_model_format(model_path)
        
        # Convert hardware target to HardwareTarget object if needed
        if isinstance(hardware_target, str):
            hardware_target = await self._resolve_hardware_target(hardware_target)
        elif isinstance(hardware_target, HardwareType):
            hardware_target = await self._resolve_hardware_target(hardware_target.value)
        
        # Generate job name if not provided
        if name is None:
            name = f"optimization_{model_path.stem}_{int(time.time())}"
        
        # Create optimization configuration
        optimization_config = OptimizationConfig(
            level=optimization_level,
            **optimization_params
        )
        
        # Prepare job data
        job_data = {
            "name": name,
            "description": description,
            "model_path": str(model_path),
            "model_format": model_format.value,
            "hardware_target": hardware_target.dict(),
            "optimization_config": optimization_config.dict(),
        }
        
        data = await self._request("POST", "/jobs", data=job_data)
        job = OptimizationJob(**data)
        
        logger.info("Optimization job submitted", job_id=str(job.id), name=name)
        return job
    
    def _detect_model_format(self, model_path: Path) -> ModelFormat:
        """Auto-detect model format from file extension."""
        suffix = model_path.suffix.lower()
        
        if suffix == ".onnx":
            return ModelFormat.ONNX
        elif suffix in [".pt", ".pth"]:
            return ModelFormat.PYTORCH
        elif suffix in [".pb", ".savedmodel"]:
            return ModelFormat.TENSORFLOW
        elif suffix == ".trt":
            return ModelFormat.TENSORRT
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
    
    async def _resolve_hardware_target(self, target_name: str) -> HardwareTarget:
        """Resolve hardware target name to HardwareTarget object."""
        targets = await self.list_hardware_targets()
        
        # Try exact match first
        for target in targets:
            if target.type.value == target_name:
                return target
        
        # Try partial match
        for target in targets:
            if target_name.lower() in target.type.value.lower():
                return target
        
        raise ValueError(f"Hardware target not found: {target_name}")
    
    async def get_job(self, job_id: Union[str, UUID]) -> OptimizationJob:
        """Get optimization job details."""
        data = await self._request("GET", f"/jobs/{job_id}")
        return OptimizationJob(**data)
    
    async def get_job_status(self, job_id: Union[str, UUID]) -> JobStatus:
        """Get job status."""
        data = await self._request("GET", f"/jobs/{job_id}/status")
        return JobStatus(**data)
    
    async def get_job_result(self, job_id: Union[str, UUID]) -> OptimizationResult:
        """Get optimization results."""
        data = await self._request("GET", f"/jobs/{job_id}/result")
        return OptimizationResult(**data)
    
    async def cancel_job(self, job_id: Union[str, UUID]) -> bool:
        """Cancel a running job."""
        try:
            await self._request("POST", f"/jobs/{job_id}/cancel")
            return True
        except FelixEnterpriseError:
            return False
    
    async def list_jobs(
        self,
        limit: int = 100,
        status_filter: Optional[JobState] = None,
        skip: int = 0,
    ) -> List[OptimizationJob]:
        """List optimization jobs."""
        params = {"limit": limit, "skip": skip}
        if status_filter:
            params["status_filter"] = status_filter.value
            
        data = await self._request("GET", "/jobs", params=params)
        return [OptimizationJob(**item) for item in data]
    
    async def wait_for_completion(
        self,
        job_id: Union[str, UUID],
        polling_interval: float = 5.0,
        timeout: Optional[float] = None,
        progress_callback: Optional[callable] = None,
    ) -> OptimizationResult:
        """
        Wait for job completion and return results.
        
        Args:
            job_id: Job ID to wait for
            polling_interval: Polling interval in seconds
            timeout: Maximum wait time in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            OptimizationResult when job completes
        """
        start_time = time.time()
        
        while True:
            status = await self.get_job_status(job_id)
            
            if progress_callback:
                progress_callback(status)
            
            if status.state == JobState.COMPLETED:
                return await self.get_job_result(job_id)
            elif status.state == JobState.FAILED:
                raise OptimizationError(f"Job failed: {status.error_message}")
            elif status.state == JobState.CANCELLED:
                raise OptimizationError("Job was cancelled")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job did not complete within {timeout} seconds")
            
            await asyncio.sleep(polling_interval)
    
    # Convenience methods
    async def quick_optimize(
        self,
        model_path: Union[str, Path],
        hardware_type: str = "amd_rocm",
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
        wait_for_completion: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> OptimizationResult:
        """
        Convenience method for quick model optimization.
        
        Args:
            model_path: Path to model file
            hardware_type: Target hardware type
            optimization_level: Optimization level
            wait_for_completion: Whether to wait for completion
            progress_callback: Progress callback function
            
        Returns:
            OptimizationResult if wait_for_completion=True, else OptimizationJob
        """
        # Submit optimization job
        job = await self.optimize_model(
            model_path=model_path,
            hardware_target=hardware_type,
            optimization_level=optimization_level,
        )
        
        if not wait_for_completion:
            return job
        
        # Wait for completion
        return await self.wait_for_completion(
            job.id, 
            progress_callback=progress_callback
        )
    
    async def batch_optimize(
        self,
        model_paths: List[Union[str, Path]],
        hardware_target: Union[HardwareTarget, str],
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
        max_concurrent: int = 3,
    ) -> List[OptimizationJob]:
        """
        Submit multiple models for optimization concurrently.
        
        Args:
            model_paths: List of model file paths
            hardware_target: Target hardware
            optimization_level: Optimization level
            max_concurrent: Maximum concurrent jobs
            
        Returns:
            List of OptimizationJob instances
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def optimize_single(model_path):
            async with semaphore:
                return await self.optimize_model(
                    model_path=model_path,
                    hardware_target=hardware_target,
                    optimization_level=optimization_level,
                )
        
        tasks = [optimize_single(path) for path in model_paths]
        return await asyncio.gather(*tasks)
    
    # Monitoring and statistics
    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get job execution statistics."""
        return await self._request("GET", "/monitoring/jobs/stats")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return await self._request("GET", "/monitoring/performance")


# Synchronous wrapper for easier use
class SyncClient:
    """Synchronous wrapper for the Felix Enterprise Client."""
    
    def __init__(self, *args, **kwargs):
        self._async_client = Client(*args, **kwargs)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self._async_client.close())
    
    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(coro)
    
    def optimize_model(self, *args, **kwargs) -> OptimizationJob:
        """Synchronous version of optimize_model."""
        return self._run_async(self._async_client.optimize_model(*args, **kwargs))
    
    def get_job_status(self, job_id: Union[str, UUID]) -> JobStatus:
        """Synchronous version of get_job_status."""
        return self._run_async(self._async_client.get_job_status(job_id))
    
    def get_job_result(self, job_id: Union[str, UUID]) -> OptimizationResult:
        """Synchronous version of get_job_result."""
        return self._run_async(self._async_client.get_job_result(job_id))
    
    def wait_for_completion(self, *args, **kwargs) -> OptimizationResult:
        """Synchronous version of wait_for_completion."""
        return self._run_async(self._async_client.wait_for_completion(*args, **kwargs))
    
    def quick_optimize(self, *args, **kwargs) -> OptimizationResult:
        """Synchronous version of quick_optimize."""
        return self._run_async(self._async_client.quick_optimize(*args, **kwargs))
    
    def list_hardware_targets(self) -> List[HardwareTarget]:
        """Synchronous version of list_hardware_targets."""
        return self._run_async(self._async_client.list_hardware_targets())
    
    def health_check(self) -> Dict[str, Any]:
        """Synchronous version of health_check."""
        return self._run_async(self._async_client.health_check())