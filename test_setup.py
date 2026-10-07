#!/usr/bin/env python3
"""
Root verification script for generative-ai-engineering workspace.
Validates dependencies, environment settings, and local infrastructure.
"""

import os
import sys

def check_python_version():
    print("Checking Python version...", end=" ")
    if sys.version_info >= (3, 10):
        print(f"OK (Python {sys.version_info.major}.{sys.version_info.minor})")
    else:
        print(f"FAIL (Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor})")

def check_dependencies():
    print("Checking core dependencies...", end=" ")
    required_packages = ["openai", "sentence_transformers", "fastapi", "qdrant_client", "pypdf"]
    missing = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
            
    if not missing:
        print("OK (All core packages installed)")
    else:
        print(f"FAIL (Missing packages: {', '.join(missing)})")

def check_env_file():
    print("Checking environment variables...", end=" ")
    if os.path.exists(".env"):
        print("OK (.env file found)")
    else:
        print("WARNING (.env file missing. Please copy .env.example to .env)")

def main():
    print("==================================================")
    print(" Generative AI Engineering Workspace Healthcheck ")
    print("==================================================\n")
    
    check_python_version()
    check_dependencies()
    check_env_file()
    
    print("\nWorkspace verification complete.")

if __name__ == "__main__":
    main()