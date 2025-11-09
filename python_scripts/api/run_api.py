"""
Run the FastAPI inference server.

Usage:
    python python_scripts/api/run_api.py
    python python_scripts/api/run_api.py --port 8080
    python python_scripts/api/run_api.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run AI Edge Allocator API Server")
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to (default: 8000)')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload on code changes')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')
    parser.add_argument('--log-level', type=str, default='info',
                       choices=['debug', 'info', 'warning', 'error'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 AI EDGE ALLOCATOR API SERVER")
    print("="*80)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Auto-reload: {args.reload}")
    print(f"\n📖 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"❤️  Health Check: http://{args.host}:{args.port}/health")
    print("="*80 + "\n")
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
