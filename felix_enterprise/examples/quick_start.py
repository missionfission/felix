#!/usr/bin/env python3
"""
Felix Enterprise Suite - Quick Start Example

This example demonstrates how to use Felix Enterprise Suite to optimize
deep learning models for AMD ROCm hardware in an enterprise environment.
"""

import asyncio
import logging
from pathlib import Path

# Felix Enterprise imports
import felix_enterprise as fe
from felix_enterprise.models import OptimizationLevel, HardwareType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example demonstrating Felix Enterprise usage."""
    
    print("🚀 Felix Enterprise Suite - Quick Start Example")
    print("=" * 60)
    
    # Initialize Felix Enterprise client
    async with fe.Client(
        base_url="http://localhost:8000",
        api_key="your-api-key-here",  # Replace with actual API key
    ) as client:
        
        # 1. Check system health
        print("\n1. Checking system health...")
        health = await client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Active jobs: {health['active_jobs']}")
        
        # 2. List available hardware targets
        print("\n2. Discovering available hardware...")
        hardware_targets = await client.list_hardware_targets()
        
        if not hardware_targets:
            print("   ❌ No hardware targets available!")
            return
            
        # Find AMD ROCm hardware
        amd_target = None
        for target in hardware_targets:
            if target.type == HardwareType.AMD_ROCM:
                amd_target = target
                break
                
        if amd_target:
            print(f"   ✅ Found AMD ROCm GPU: {amd_target.metadata.get('product_name', 'Unknown')}")
            print(f"      Memory: {amd_target.memory_gb:.1f} GB")
            print(f"      Device ID: {amd_target.device_id}")
        else:
            print("   ⚠️  No AMD ROCm hardware found, using first available target")
            amd_target = hardware_targets[0]
            
        # 3. Benchmark hardware performance
        print(f"\n3. Benchmarking hardware performance...")
        try:
            benchmark_metrics = await client.benchmark_hardware(amd_target)
            print(f"   Latency: {benchmark_metrics.latency_ms:.2f} ms")
            print(f"   Throughput: {benchmark_metrics.throughput_ops_per_sec:.1f} ops/sec")
            print(f"   Memory usage: {benchmark_metrics.memory_usage_mb:.1f} MB")
            print(f"   GPU utilization: {benchmark_metrics.gpu_utilization_percent:.1f}%")
        except Exception as e:
            print(f"   ⚠️  Benchmark failed: {e}")
            
        # 4. Prepare a sample model for optimization
        print(f"\n4. Preparing sample model...")
        
        # For this example, we'll create a simple PyTorch model
        # In practice, you would use your existing model files
        sample_model_path = create_sample_model()
        print(f"   Sample model created: {sample_model_path}")
        
        # 5. Submit optimization job
        print(f"\n5. Submitting optimization job...")
        
        def progress_callback(status):
            print(f"   Progress: {status.progress_percent}% - {status.message}")
        
        try:
            # Option A: Quick optimization (fire and forget)
            print("   Using quick optimization...")
            result = await client.quick_optimize(
                model_path=sample_model_path,
                hardware_type="amd_rocm",
                optimization_level=OptimizationLevel.BALANCED,
                wait_for_completion=True,
                progress_callback=progress_callback
            )
            
            print(f"   ✅ Optimization completed!")
            print(f"   Original latency: {result.original_metrics.latency_ms:.2f} ms")
            print(f"   Optimized latency: {result.optimized_metrics.latency_ms:.2f} ms")
            print(f"   Speedup: {result.optimized_metrics.speedup_factor:.2f}x")
            print(f"   Optimized model: {result.optimized_model_path}")
            
        except Exception as e:
            print(f"   ❌ Optimization failed: {e}")
            
            # Option B: Manual job submission and monitoring
            print("   Trying manual job submission...")
            try:
                job = await client.optimize_model(
                    model_path=sample_model_path,
                    hardware_target=amd_target,
                    name="Quick Start Example",
                    description="Sample optimization for quick start guide",
                    optimization_level=OptimizationLevel.CONSERVATIVE,  # Faster for demo
                )
                
                print(f"   Job submitted: {job.id}")
                print(f"   Job name: {job.name}")
                print(f"   Status: {job.status.state}")
                
                # Monitor job progress
                print("   Monitoring job progress...")
                result = await client.wait_for_completion(
                    job.id,
                    polling_interval=2.0,
                    timeout=300,  # 5 minutes timeout
                    progress_callback=progress_callback
                )
                
                print(f"   ✅ Job completed successfully!")
                display_results(result)
                
            except Exception as e2:
                print(f"   ❌ Manual optimization also failed: {e2}")
        
        # 6. List recent jobs
        print(f"\n6. Recent optimization jobs...")
        jobs = await client.list_jobs(limit=5)
        for job in jobs:
            status_icon = "✅" if job.status.state == "completed" else "⏳" if job.status.state == "running" else "❌"
            print(f"   {status_icon} {job.name} ({job.status.state}) - {job.created_at.strftime('%Y-%m-%d %H:%M')}")
        
        # 7. Get system statistics
        print(f"\n7. System statistics...")
        try:
            stats = await client.get_job_statistics()
            print(f"   Total jobs: {stats.get('total_jobs', 'N/A')}")
            print(f"   Completed jobs: {stats.get('completed_jobs', 'N/A')}")
            print(f"   Average optimization time: {stats.get('avg_optimization_time_minutes', 'N/A')} min")
        except Exception as e:
            print(f"   ⚠️  Could not fetch statistics: {e}")
    
    print(f"\n🎉 Quick start example completed!")
    print("=" * 60)


def create_sample_model():
    """Create a sample PyTorch model for demonstration."""
    import torch
    import torch.nn as nn
    
    # Simple CNN model
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(128, 10)
            
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = SampleModel()
    model.eval()
    
    # Create sample input for tracing
    sample_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    model_path = Path("sample_model.onnx")
    torch.onnx.export(
        model,
        sample_input,
        model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    return model_path


def display_results(result):
    """Display optimization results in a nice format."""
    print(f"\n📊 Optimization Results:")
    print(f"   Job ID: {result.job_id}")
    print(f"   Optimized model: {result.optimized_model_path}")
    
    print(f"\n   Performance Comparison:")
    print(f"   {'Metric':<20} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
    print(f"   {'-'*65}")
    
    orig = result.original_metrics
    opt = result.optimized_metrics
    
    print(f"   {'Latency (ms)':<20} {orig.latency_ms:<15.2f} {opt.latency_ms:<15.2f} {opt.speedup_factor:<15.2f}x")
    print(f"   {'Throughput (ops/s)':<20} {orig.throughput_ops_per_sec:<15.1f} {opt.throughput_ops_per_sec:<15.1f} {(opt.throughput_ops_per_sec/orig.throughput_ops_per_sec):<15.2f}x")
    print(f"   {'Memory (MB)':<20} {orig.memory_usage_mb:<15.1f} {opt.memory_usage_mb:<15.1f} {(orig.memory_usage_mb/opt.memory_usage_mb):<15.2f}x")
    
    if opt.accuracy_drop_percent > 0:
        print(f"   {'Accuracy drop (%)':<20} {'-':<15} {opt.accuracy_drop_percent:<15.2f} {'-':<15}")


# Synchronous version for easier testing
def quick_start_sync():
    """Synchronous version of the quick start example."""
    print("🚀 Felix Enterprise Suite - Synchronous Quick Start")
    print("=" * 60)
    
    # Using synchronous client wrapper
    with fe.SyncClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here",
    ) as client:
        
        # Check health
        print("\nChecking system health...")
        health = client.health_check()
        print(f"Status: {health['status']}")
        
        # List hardware
        print("\nListing hardware targets...")
        targets = client.list_hardware_targets()
        for target in targets:
            print(f"  - {target.type.value}: {target.metadata.get('product_name', 'Unknown')}")
        
        if targets:
            # Quick optimization
            print("\nRunning quick optimization...")
            sample_model = create_sample_model()
            
            try:
                result = client.quick_optimize(
                    model_path=sample_model,
                    hardware_type="amd_rocm",
                    optimization_level=OptimizationLevel.CONSERVATIVE
                )
                print(f"✅ Optimization completed! Speedup: {result.optimized_metrics.speedup_factor:.2f}x")
                
            except Exception as e:
                print(f"❌ Optimization failed: {e}")
        
    print("🎉 Synchronous quick start completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        # Run synchronous version
        quick_start_sync()
    else:
        # Run async version
        asyncio.run(main())