"""
AMD ROCm Hardware Backend for Felix Enterprise Suite

Provides specialized optimization for AMD GPUs using ROCm platform.
"""

import os
import subprocess
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import torch
import structlog
from ..models import HardwareTarget, HardwareType, PerformanceMetrics
from ..exceptions import HardwareError
from .base import HardwareBackend

logger = structlog.get_logger()


class AMDROCmBackend(HardwareBackend):
    """AMD ROCm hardware optimization backend."""
    
    def __init__(self):
        super().__init__()
        self.backend_name = "amd_rocm"
        self.supported_precisions = ["fp32", "fp16", "int8", "mixed"]
        self._devices_cache = None
        self._rocm_info = None
        
    async def initialize(self) -> bool:
        """Initialize ROCm backend and detect devices."""
        try:
            # Check if ROCm is available
            if not self._check_rocm_available():
                logger.warning("ROCm not available on this system")
                return False
                
            # Get ROCm system information
            self._rocm_info = await self._get_rocm_info()
            
            # Detect AMD GPUs
            self._devices_cache = await self._detect_devices()
            
            logger.info(
                "AMD ROCm backend initialized", 
                devices_count=len(self._devices_cache),
                rocm_version=self._rocm_info.get("version")
            )
            return True
            
        except Exception as e:
            logger.error("Failed to initialize AMD ROCm backend", error=str(e))
            return False
    
    def _check_rocm_available(self) -> bool:
        """Check if ROCm is installed and available."""
        try:
            # Check for rocm-smi command
            result = subprocess.run(
                ["rocm-smi", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                return False
                
            # Check PyTorch ROCm support
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return True
                
            return False
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def _get_rocm_info(self) -> Dict[str, Any]:
        """Get ROCm platform information."""
        info = {}
        
        try:
            # Get ROCm version
            result = subprocess.run(
                ["rocm-smi", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                info["version"] = result.stdout.strip()
                
            # Get ROCm installation path
            rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
            info["installation_path"] = rocm_path
            
            # Check for HIP runtime
            if hasattr(torch.version, 'hip'):
                info["hip_version"] = torch.version.hip
                
        except Exception as e:
            logger.warning("Failed to get ROCm info", error=str(e))
            
        return info
    
    async def _detect_devices(self) -> List[HardwareTarget]:
        """Detect available AMD GPU devices."""
        devices = []
        
        try:
            # Use rocm-smi to get device information
            result = subprocess.run(
                ["rocm-smi", "--showid", "--showproductname", "--showmeminfo", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                for device_id, device_info in data.items():
                    if device_id.startswith("card"):
                        device_num = int(device_id.replace("card", ""))
                        
                        # Extract device information
                        product_name = device_info.get("Product Name", "Unknown AMD GPU")
                        memory_info = device_info.get("Memory Info", {})
                        memory_total = memory_info.get("VRAM Total", "0 B")
                        
                        # Parse memory size (convert from bytes to GB)
                        memory_gb = self._parse_memory_size(memory_total)
                        
                        device = HardwareTarget(
                            type=HardwareType.AMD_ROCM,
                            device_id=device_num,
                            memory_gb=memory_gb,
                            metadata={
                                "product_name": product_name,
                                "rocm_device_id": device_id,
                                "backend": self.backend_name,
                                "driver_version": self._rocm_info.get("version", "unknown"),
                                "hip_version": self._rocm_info.get("hip_version", "unknown")
                            }
                        )
                        devices.append(device)
                        
            # Fallback: try PyTorch device detection
            if not devices and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    device = HardwareTarget(
                        type=HardwareType.AMD_ROCM,
                        device_id=i,
                        memory_gb=props.total_memory / (1024**3),
                        metadata={
                            "product_name": props.name,
                            "backend": self.backend_name,
                            "compute_capability": f"{props.major}.{props.minor}",
                            "multiprocessor_count": props.multi_processor_count
                        }
                    )
                    devices.append(device)
                    
        except Exception as e:
            logger.error("Failed to detect AMD devices", error=str(e))
            
        return devices
    
    def _parse_memory_size(self, memory_str: str) -> float:
        """Parse memory size string to GB."""
        try:
            # Handle formats like "8192 MB", "8 GB", etc.
            parts = memory_str.strip().split()
            if len(parts) >= 2:
                value = float(parts[0])
                unit = parts[1].upper()
                
                if unit in ["B", "BYTES"]:
                    return value / (1024**3)
                elif unit in ["KB", "KILOBYTES"]:
                    return value / (1024**2)
                elif unit in ["MB", "MEGABYTES"]:
                    return value / 1024
                elif unit in ["GB", "GIGABYTES"]:
                    return value
                    
        except (ValueError, IndexError):
            pass
            
        return 0.0
    
    async def get_available_devices(self) -> List[HardwareTarget]:
        """Get list of available AMD GPU devices."""
        if self._devices_cache is None:
            await self.initialize()
        return self._devices_cache or []
    
    async def optimize_model(
        self,
        model_path: str,
        hardware_target: HardwareTarget,
        optimization_config: dict,
        output_path: str
    ) -> Dict[str, Any]:
        """Optimize model for AMD ROCm hardware."""
        try:
            logger.info(
                "Starting AMD ROCm optimization",
                model_path=model_path,
                device_id=hardware_target.device_id,
                config=optimization_config
            )
            
            # Set ROCm-specific environment variables
            env_vars = self._setup_rocm_environment(hardware_target)
            
            # Import Felix optimizer with ROCm backend
            from tvm import felix
            from tvm.target import Target
            
            # Create ROCm target specification
            target_spec = self._create_target_spec(hardware_target, optimization_config)
            
            # Load and extract tasks from model
            model, inputs = self._load_model(model_path, optimization_config)
            tasks = felix.extract_tasks(model, inputs, None)
            
            # Create performance model for ROCm
            perf_model = self._create_performance_model(hardware_target)
            
            # Initialize Felix optimizer with ROCm-specific settings
            optimizer = felix.Optimizer(tasks, perf_model)
            
            # Configure optimization parameters for AMD hardware
            opt_params = self._configure_optimization_params(
                hardware_target, optimization_config
            )
            
            # Run optimization
            results = optimizer.tune(**opt_params)
            
            # Save optimized model
            self._save_optimized_model(results, output_path, target_spec)
            
            # Benchmark performance
            metrics = await self._benchmark_model(
                output_path, hardware_target, optimization_config
            )
            
            logger.info("AMD ROCm optimization completed", metrics=metrics)
            
            return {
                "status": "completed",
                "optimized_model_path": output_path,
                "target_spec": target_spec,
                "metrics": metrics,
                "optimization_log": results.get("log", [])
            }
            
        except Exception as e:
            logger.error("AMD ROCm optimization failed", error=str(e))
            raise HardwareError(f"ROCm optimization failed: {str(e)}")
    
    def _setup_rocm_environment(self, hardware_target: HardwareTarget) -> Dict[str, str]:
        """Setup ROCm-specific environment variables."""
        env_vars = {
            "HIP_VISIBLE_DEVICES": str(hardware_target.device_id),
            "ROCM_PATH": self._rocm_info.get("installation_path", "/opt/rocm"),
            "HSA_OVERRIDE_GFX_VERSION": "10.3.0",  # Common compatibility setting
            "HIP_PLATFORM": "amd",
        }
        
        # Apply environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
            
        return env_vars
    
    def _create_target_spec(
        self, 
        hardware_target: HardwareTarget, 
        optimization_config: dict
    ) -> str:
        """Create TVM target specification for ROCm."""
        # Base ROCm target
        target_str = "rocm"
        
        # Add device-specific parameters
        if hardware_target.metadata.get("compute_capability"):
            target_str += f" -mcpu=gfx{hardware_target.metadata['compute_capability']}"
        
        # Add optimization-specific parameters
        if optimization_config.get("enable_tensorcore", False):
            target_str += " -libs=rocblas,miopen"
            
        return target_str
    
    def _configure_optimization_params(
        self,
        hardware_target: HardwareTarget,
        optimization_config: dict
    ) -> Dict[str, Any]:
        """Configure Felix optimization parameters for AMD hardware."""
        
        # Base parameters
        params = {
            "n_measurements": 1024,  # Reduced for faster iteration
            "n_config_seeds": 32,
            "n_grad_steps": 256,
            "measure_per_round": 16,
        }
        
        # Adjust based on optimization level
        level = optimization_config.get("level", "balanced")
        if level == "aggressive":
            params.update({
                "n_measurements": 2048,
                "n_config_seeds": 64,
                "n_grad_steps": 512,
            })
        elif level == "conservative":
            params.update({
                "n_measurements": 512,
                "n_config_seeds": 16,
                "n_grad_steps": 128,
            })
            
        # AMD-specific tuning
        if hardware_target.memory_gb and hardware_target.memory_gb < 8:
            # Reduce memory usage for smaller GPUs
            params["n_config_seeds"] = min(params["n_config_seeds"], 16)
            
        return params
    
    async def benchmark_hardware(
        self, 
        hardware_target: HardwareTarget
    ) -> PerformanceMetrics:
        """Run hardware benchmark on AMD GPU."""
        try:
            logger.info("Running AMD ROCm benchmark", device_id=hardware_target.device_id)
            
            # Set device
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{hardware_target.device_id}")
                torch.cuda.set_device(device)
            else:
                raise HardwareError("ROCm/CUDA not available for benchmarking")
            
            # Benchmark parameters
            batch_size = 32
            input_size = (3, 224, 224)
            num_iterations = 100
            
            # Create test tensors
            input_tensor = torch.randn(batch_size, *input_size, device=device)
            
            # Simple benchmark model (ResNet-like operations)
            model = self._create_benchmark_model().to(device)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_tensor)
            
            # Synchronize and measure
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            import time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    output = model(input_tensor)
                    
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_latency_ms = (total_time / num_iterations) * 1000
            throughput = num_iterations / total_time
            
            # Get memory usage
            memory_used = 0
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(device) / (1024**2)  # MB
            
            metrics = PerformanceMetrics(
                latency_ms=avg_latency_ms,
                throughput_ops_per_sec=throughput,
                memory_usage_mb=memory_used,
                gpu_utilization_percent=85.0,  # Estimated
                power_usage_watts=200.0,  # Estimated for typical AMD GPU
            )
            
            logger.info("AMD ROCm benchmark completed", metrics=metrics.dict())
            return metrics
            
        except Exception as e:
            logger.error("AMD ROCm benchmark failed", error=str(e))
            raise HardwareError(f"ROCm benchmark failed: {str(e)}")
    
    def _create_benchmark_model(self) -> torch.nn.Module:
        """Create a benchmark model for performance testing."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 1000)
        )
    
    async def get_device_info(self, device_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific AMD GPU device."""
        devices = await self.get_available_devices()
        
        for device in devices:
            if device.device_id == device_id:
                # Get additional runtime information
                runtime_info = {}
                
                try:
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(device_id)
                        runtime_info.update({
                            "total_memory_gb": props.total_memory / (1024**3),
                            "multiprocessor_count": props.multi_processor_count,
                            "max_threads_per_multiprocessor": props.max_threads_per_multi_processor,
                            "warp_size": props.warp_size,
                        })
                        
                        # Current memory usage
                        memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
                        memory_cached = torch.cuda.memory_reserved(device_id) / (1024**3)
                        
                        runtime_info.update({
                            "memory_allocated_gb": memory_allocated,
                            "memory_cached_gb": memory_cached,
                            "memory_free_gb": props.total_memory / (1024**3) - memory_allocated,
                        })
                        
                except Exception as e:
                    logger.warning("Failed to get runtime device info", error=str(e))
                
                return {
                    **device.dict(),
                    "runtime_info": runtime_info
                }
        
        raise HardwareError(f"AMD GPU device {device_id} not found")
    
    def get_optimization_recommendations(
        self, 
        hardware_target: HardwareTarget,
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get optimization recommendations for AMD hardware."""
        recommendations = {
            "precision": "fp16",  # AMD GPUs generally benefit from fp16
            "batch_size": 32,
            "enable_tensorcore": False,  # AMD doesn't have Tensor Cores
            "memory_optimization": True,
        }
        
        # Adjust based on available memory
        if hardware_target.memory_gb:
            if hardware_target.memory_gb >= 16:
                recommendations["batch_size"] = 64
                recommendations["precision"] = "fp32"  # Can afford higher precision
            elif hardware_target.memory_gb <= 8:
                recommendations["batch_size"] = 16
                recommendations["precision"] = "int8"  # More aggressive quantization
        
        # Model-specific recommendations
        model_size_mb = model_info.get("size_mb", 0)
        if model_size_mb > 1000:  # Large model
            recommendations["enable_dynamic_shapes"] = False
            recommendations["memory_optimization"] = True
        
        return recommendations