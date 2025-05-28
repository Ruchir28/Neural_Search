#!/usr/bin/env python3
"""
Simple script to run the orchestrator with environment configuration
"""

import os
import sys
from pathlib import Path

# Add the deployment_scripts directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv not installed. Install with: pip install python-dotenv")
    sys.exit(1)

def main():
    """Load environment and run orchestrator"""
    
    # Load environment from config file
    config_file = Path(__file__).parent / "orchestrator_config.env"
    
    if config_file.exists():
        print(f"Loading configuration from {config_file}")
        load_dotenv(config_file)
    else:
        print(f"WARNING: Configuration file {config_file} not found.")
        print("Please copy orchestrator_config.env.example to orchestrator_config.env and configure it.")
        
        # Check if required environment variables are set
        required_vars = ["EBS_VOLUME_ID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"ERROR: Required environment variables not set: {', '.join(missing_vars)}")
            sys.exit(1)
    
    # Import and run orchestrator
    try:
        from orchestrator import main as orchestrator_main
        orchestrator_main()
    except ImportError as e:
        print(f"ERROR: Failed to import orchestrator: {e}")
        print("Make sure you're in the deployment_scripts directory and all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 