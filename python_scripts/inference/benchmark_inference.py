"""
Performance Benchmark Script
============================

Compare inference performance across different configurations.
"""

import time
import numpy as np
import torch
import json
from pathlib import Path
import sys

try:
    from local_inference import LocalModelInference
    ORIGINAL_AVAILABLE = True
except:
    ORIGINAL_AVAILABLE = False

try:
    from local_inference_optimized import OptimizedModelInference
    OPTIMIZED_AVAILABLE = True
except:
    OPTIMIZED_AVAILABLE = False


def create_benchmark_states(num_states=100, num_nodes=10):
    """Create benchmark network states."""
    states = []
    for _ in range(num_states):
        state = {
            'nodes': [
                {
                    'node_id': f'node_{i}',
                    'cpu_util': np.random.uniform(0.2, 0.8),
                    'mem_util': np.random.uniform(0.2, 0.7),
                    'energy': np.random.uniform(100, 150),
                    'latency': np.random.uniform(5, 30),
                    'bandwidth': np.random.uniform(100, 200),
                    'queue_len': np.random.randint(0, 15),
                    'node_type': np.random.randint(0, 3)
                }
                for i in range(num_nodes)
            ]
        }
        states.append(state)
    return states


def benchmark_model(model, states, name, warmup=10):
    """Benchmark a model with given states."""
    print(f"\n{'='*70}")
    print(f"🔬 Benchmarking: {name}")
    print(f"{'='*70}")
    
    # Warmup
    print(f"   Warming up ({warmup} iterations)...")
    for state in states[:warmup]:
        try:
            model.predict(state)
        except:
            pass
    
    # Benchmark
    print(f"   Running benchmark ({len(states)} predictions)...")
    timings = []
    errors = 0
    
    start_total = time.time()
    for state in states:
        try:
            start = time.time()
            result = model.predict(state)
            elapsed = (time.time() - start) * 1000
            timings.append(elapsed)
        except Exception as e:
            errors += 1
            print(f"   ⚠️  Error: {e}")
    
    total_time = (time.time() - start_total) * 1000
    
    # Results
    if timings:
        print(f"\n   ✅ Results:")
        print(f"      Total Time: {total_time:.2f}ms")
        print(f"      Successful: {len(timings)}/{len(states)}")
        print(f"      Mean: {np.mean(timings):.2f}ms")
        print(f"      Median: {np.median(timings):.2f}ms")
        print(f"      Std: {np.std(timings):.2f}ms")
        print(f"      Min: {np.min(timings):.2f}ms")
        print(f"      Max: {np.max(timings):.2f}ms")
        print(f"      P95: {np.percentile(timings, 95):.2f}ms")
        print(f"      P99: {np.percentile(timings, 99):.2f}ms")
        print(f"      Throughput: {len(timings)/(total_time/1000):.2f} pred/sec")
    
    return {
        'name': name,
        'total_time': total_time,
        'mean': np.mean(timings) if timings else None,
        'median': np.median(timings) if timings else None,
        'std': np.std(timings) if timings else None,
        'min': np.min(timings) if timings else None,
        'max': np.max(timings) if timings else None,
        'p95': np.percentile(timings, 95) if timings else None,
        'p99': np.percentile(timings, 99) if timings else None,
        'throughput': len(timings)/(total_time/1000) if timings else None,
        'successful': len(timings),
        'errors': errors
    }


def main():
    print("\n" + "="*70)
    print("⚡ AI EDGE ALLOCATOR - PERFORMANCE BENCHMARK")
    print("="*70)
    
    # Configuration
    num_states = 100
    num_nodes = 10
    model_type = 'dqn'  # Change to 'ppo' or 'hybrid' as needed
    
    print(f"\n📋 Configuration:")
    print(f"   Model Type: {model_type}")
    print(f"   Test States: {num_states}")
    print(f"   Nodes per State: {num_nodes}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Generate test data
    print(f"\n📊 Generating test data...")
    states = create_benchmark_states(num_states, num_nodes)
    print(f"   ✅ Generated {len(states)} test states")
    
    results = []
    
    # Check model availability
    model_path = Path(f"models/{model_type}/best_model")
    if model_type == "hybrid":
        model_path = Path(f"models/{model_type}/best_model.pt")
    else:
        model_path = model_path / "best_model.zip"
    
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print(f"   Please train the {model_type} model first.")
        return
    
    # Benchmark 1: Original Implementation (CPU)
    if ORIGINAL_AVAILABLE:
        try:
            print(f"\n🔹 Test 1: Original Implementation")
            model = LocalModelInference(model_type=model_type)
            result = benchmark_model(model, states, "Original (CPU)")
            results.append(result)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Benchmark 2: Optimized Implementation (CPU)
    if OPTIMIZED_AVAILABLE:
        try:
            print(f"\n🔹 Test 2: Optimized Implementation (CPU)")
            model = OptimizedModelInference(
                model_type=model_type,
                device='cpu',
                enable_quantization=False
            )
            result = benchmark_model(model, states, "Optimized (CPU)")
            results.append(result)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Benchmark 3: Optimized with Quantization (CPU)
    if OPTIMIZED_AVAILABLE and model_type == "hybrid":
        try:
            print(f"\n🔹 Test 3: Optimized + Quantization (CPU)")
            model = OptimizedModelInference(
                model_type=model_type,
                device='cpu',
                enable_quantization=True
            )
            result = benchmark_model(model, states, "Optimized + Quantization (CPU)")
            results.append(result)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Benchmark 4: Optimized Implementation (GPU)
    if OPTIMIZED_AVAILABLE and torch.cuda.is_available():
        try:
            print(f"\n🔹 Test 4: Optimized Implementation (GPU)")
            model = OptimizedModelInference(
                model_type=model_type,
                device='cuda'
            )
            result = benchmark_model(model, states, "Optimized (GPU)")
            results.append(result)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Benchmark 5: Batch Processing
    if OPTIMIZED_AVAILABLE:
        try:
            print(f"\n🔹 Test 5: Batch Processing")
            model = OptimizedModelInference(model_type=model_type)
            
            print(f"\n{'='*70}")
            print(f"🔬 Benchmarking: Batch Processing")
            print(f"{'='*70}")
            
            start = time.time()
            batch_results = model.predict_batch(states)
            elapsed = (time.time() - start) * 1000
            
            print(f"\n   ✅ Results:")
            print(f"      Total Time: {elapsed:.2f}ms")
            print(f"      Per Prediction: {elapsed/len(states):.2f}ms")
            print(f"      Throughput: {len(states)/(elapsed/1000):.2f} pred/sec")
            
            results.append({
                'name': 'Batch Processing',
                'total_time': elapsed,
                'mean': elapsed/len(states),
                'throughput': len(states)/(elapsed/1000)
            })
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Summary
    if len(results) > 1:
        print(f"\n\n{'='*70}")
        print("📊 PERFORMANCE COMPARISON")
        print(f"{'='*70}\n")
        
        # Find baseline (original or first result)
        baseline = results[0]
        baseline_mean = baseline.get('mean', baseline.get('total_time', 1))
        
        print(f"{'Configuration':<35} {'Mean (ms)':<12} {'Throughput':<15} {'Speedup':<10}")
        print("-" * 70)
        
        for result in results:
            name = result['name']
            mean = result.get('mean')
            throughput = result.get('throughput')
            
            if mean:
                speedup = baseline_mean / mean
                print(f"{name:<35} {mean:>10.2f}  {throughput:>12.1f}/s  {speedup:>8.2f}x")
            else:
                print(f"{name:<35} {'N/A':<12} {'N/A':<15} {'N/A':<10}")
        
        # Best configuration
        best = min(results, key=lambda x: x.get('mean', float('inf')))
        print(f"\n🏆 Best Configuration: {best['name']}")
        print(f"   Mean Latency: {best.get('mean', 'N/A'):.2f}ms")
        print(f"   Throughput: {best.get('throughput', 'N/A'):.1f} predictions/sec")
    
    # Save results
    output_file = 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("✅ Benchmark complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
