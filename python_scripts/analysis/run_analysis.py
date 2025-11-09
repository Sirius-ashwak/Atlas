"""
Run Complete Phase 2 Analysis

This script runs all analysis and visualization tasks.
"""

from src.analysis.model_comparison import ModelAnalyzer
from src.analysis.network_visualization import NetworkVisualizer

def main():
    print("\n" + "="*80)
    print("🚀 PHASE 2: COMPREHENSIVE ANALYSIS & VISUALIZATION")
    print("="*80 + "\n")
    
    print("Part 1: Model Performance Analysis")
    print("-" * 80)
    analyzer = ModelAnalyzer()
    analyzer.run_full_analysis()
    
    print("\n" + "="*80)
    print("Part 2: Network Topology Visualization")
    print("-" * 80)
    visualizer = NetworkVisualizer()
    visualizer.run_full_visualization()
    
    print("\n" + "="*80)
    print("🎉 PHASE 2 COMPLETE!")
    print("="*80)
    print("\n📊 Results Summary:")
    print("   - Performance comparison charts ✅")
    print("   - Variance and stability analysis ✅")
    print("   - Training efficiency metrics ✅")
    print("   - Network topology visualization ✅")
    print("   - Resource utilization heatmaps ✅")
    print("   - Allocation pattern analysis ✅")
    print("\n📁 Check the 'reports/figures/' folder for all visualizations!")
    print("📄 Check 'reports/model_comparison.csv' for detailed metrics!")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
