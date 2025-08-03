#!/usr/bin/env python3
"""
Auto Thievia API Server

Launch script for the FastAPI REST API server.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    import uvicorn
    from auto_thievia.api import app

    def main():
        """Run the API server."""
        print("🚗 Starting Auto Thievia API Server...")
        print("📍 API Documentation: http://localhost:8000/docs")
        print("🗺️  Interactive API: http://localhost:8000/redoc")
        print("🌐 Main Interface: http://localhost:8000/")
        print("-" * 50)

        # Ensure output directory exists
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run server
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure FastAPI dependencies are installed:")
    print("   pip install fastapi uvicorn pydantic")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n🛑 Server stopped by user")
    sys.exit(0)
