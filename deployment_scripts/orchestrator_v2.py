#!/usr/bin/env python3
"""
Semantic Search Orchestrator Server
This server manages EC2 instances for the semantic search service.
It can spin up instances, attach EBS volumes, and route requests.
"""
import sys
import asyncio
from datetime import datetime, timedelta

try:
    import boto3
    from fastapi import FastAPI
    import uvicorn
except ImportError:
    print("ERROR: Required dependencies not installed. Install with: pip install boto3 fastapi uvicorn httpx")
    sys.exit(1)

from config import OrchestratorConfig
from models import QueryRequest, QueryResponse, OrchestrationStatus
from aws_manager import AWSManager
from instance_manager import InstanceManager
from api import OrchestrationAPI

class SemanticSearchOrchestrator:
    """Main orchestrator class that coordinates all components"""
    
    def __init__(self):
        # Load configuration
        self.config = OrchestratorConfig.from_env()
        
        # Initialize components
        self.aws_manager = AWSManager(self.config)
        self.instance_manager = InstanceManager(self.config, self.aws_manager)
        self.api = OrchestrationAPI(self.instance_manager)
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Semantic Search Orchestrator")
        self._setup_routes()
        
        # Initialize AWS resources
        self.aws_manager.initialize_resources()
        
        # Discover existing instances
        self.instance_manager.discover_existing_instances()
        
        # Setup startup event
        @self.app.on_event("startup")
        async def startup_event():
            # Validate discovered instances
            await self.instance_manager.validate_discovered_instances()
            
            # Start background tasks
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._cleanup_loop())
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/query", response_model=QueryResponse)
        async def handle_query(request: QueryRequest):
            return await self.api.handle_query(request)
        
        @self.app.get("/status", response_model=OrchestrationStatus)
        async def get_status():
            return await self.api.get_status()
        
        @self.app.post("/scale-up")
        async def scale_up(count: int = 1):
            return await self.api.scale_up(count)
        
        @self.app.post("/scale-down")
        async def scale_down(count: int = 1):
            return await self.api.scale_down(count)
        
        @self.app.delete("/instance/{instance_id}")
        async def terminate_instance(instance_id: str):
            return await self.api.terminate_instance(instance_id)
    
    async def _health_check_loop(self):
        """Periodic health check for all instances"""
        while True:
            await asyncio.sleep(self.config.health_check_interval)
            
            # Health check all active instances
            health_check_tasks = []
            for instance in self.instance_manager.instances.values():
                if instance.status.value in ['ready']:
                    task = asyncio.create_task(
                        self.instance_manager.health_check_instance(instance)
                    )
                    health_check_tasks.append(task)
            
            if health_check_tasks:
                await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _cleanup_loop(self):
        """Periodic cleanup of inactive and failed instances"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            current_time = datetime.now()
            
            # Clean up failed instances
            self.instance_manager.cleanup_failed_instances()
            
            # Terminate instances that have been running for 30+ minutes (1800 seconds)
            max_runtime = self.config.instance_max_runtime
            long_running_instances = []
            
            for instance in self.instance_manager.instances.values():
                runtime = (current_time - instance.created_at).total_seconds()
                if runtime > max_runtime:
                    long_running_instances.append(instance)
            
            for instance in long_running_instances:
                runtime_minutes = int((current_time - instance.created_at).total_seconds() / 60)
                print(f"Terminating long-running instance {instance.instance_id} (running for {runtime_minutes} minutes)")
                try:
                    await self.instance_manager.terminate_instance(instance.instance_id)
                except Exception as e:
                    print(f"Failed to terminate long-running instance {instance.instance_id}: {e}")
            
            # Terminate instances that are not processing requests and have been inactive too long
            inactive_instances = self.instance_manager.get_idle_instances()
            
            for instance in inactive_instances:
                inactive_time = (current_time - instance.last_used).total_seconds()
                if inactive_time > self.config.instance_idle_timeout:
                    inactive_minutes = int(inactive_time / 60)
                    print(f"Terminating inactive instance {instance.instance_id} (inactive for {inactive_minutes} minutes)")
                    try:
                        await self.instance_manager.terminate_instance(instance.instance_id)
                    except Exception as e:
                        print(f"Failed to terminate inactive instance {instance.instance_id}: {e}")

def main():
    """Main function to run the orchestrator"""
    print("Starting Semantic Search Orchestrator (v2)...")
    
    # Validate AWS credentials
    try:
        boto3.Session().get_credentials()
    except Exception as e:
        print(f"AWS credentials not configured: {e}")
        sys.exit(1)
    
    # Create and run orchestrator
    try:
        orchestrator = SemanticSearchOrchestrator()
        
        uvicorn.run(
            orchestrator.app,
            host="0.0.0.0",
            port=8080,
            log_level="info"
        )
    except Exception as e:
        print(f"Failed to start orchestrator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 