"""
FastAPI routes and HTTP handlers for the semantic search orchestrator
"""
import asyncio
from datetime import datetime
from typing import Dict, Any
import httpx
from fastapi import HTTPException

from models import QueryRequest, QueryResponse, OrchestrationStatus, InstanceInfo, InstanceStatus
from instance_manager import InstanceManager

class OrchestrationAPI:
    """Handles HTTP API endpoints for the orchestrator"""
    
    def __init__(self, instance_manager: InstanceManager):
        self.instance_manager = instance_manager
    
    async def handle_query(self, request: QueryRequest) -> QueryResponse:
        """Handle a semantic search query by routing to an available instance"""
        
        # Find an available instance
        instance = await self.instance_manager.get_available_instance()
        if not instance:
            raise HTTPException(status_code=503, detail="No available instances. Scaling up...")
        
        # Track request count and update last used time
        instance.current_requests += 1
        instance.last_used = datetime.now()
        
        try:
            # Forward request to the instance
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{instance.public_ip}:8000/query",
                    json=request.dict()
                )
                response.raise_for_status()
                result = response.json()
                
            # Add instance info to response
            result["instance_id"] = instance.instance_id
            
            return QueryResponse(**result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
        
        finally:
            # Decrement request count
            instance.current_requests -= 1
    
    async def get_status(self) -> OrchestrationStatus:
        """Get current orchestration status"""
        instances = self.instance_manager.instances
        
        total = len(instances)
        ready = sum(1 for i in instances.values() if i.is_ready_to_serve)
        processing = sum(1 for i in instances.values() if i.is_processing_requests)
        launching = sum(1 for i in instances.values() if i.status == InstanceStatus.LAUNCHING)
        
        instance_infos = [
            InstanceInfo(
                instance_id=i.instance_id,
                public_ip=i.public_ip,
                status=i.status.value,
                created_at=i.created_at.isoformat(),
                last_used=i.last_used.isoformat(),
                current_requests=i.current_requests,
                is_ready_to_serve=i.is_ready_to_serve,
                is_processing_requests=i.is_processing_requests
            )
            for i in instances.values()
        ]
        
        return OrchestrationStatus(
            total_instances=total,
            ready_instances=ready,
            processing_requests=processing,
            launching_instances=launching,
            instances=instance_infos
        )
    
    async def scale_up(self, count: int = 1) -> Dict[str, Any]:
        """Scale up by launching additional instances"""
        
        # Check if a launch is already in progress
        if self.instance_manager._launching_lock.locked():
            # Get detailed information about current instances
            instances = self.instance_manager.instances
            launching_instances = [
                {
                    "instance_id": i.instance_id,
                    "status": i.status.value,
                    "created_at": i.created_at.isoformat(),
                    "public_ip": i.public_ip or "pending"
                }
                for i in instances.values() 
                if i.status in [InstanceStatus.LAUNCHING, InstanceStatus.STARTING]
            ]
            
            all_instances = [
                {
                    "instance_id": i.instance_id,
                    "status": i.status.value,
                    "created_at": i.created_at.isoformat(),
                    "public_ip": i.public_ip or "pending",
                    "last_used": i.last_used.isoformat() if i.last_used else None
                }
                for i in instances.values()
            ]
            
            return {
                "error": "Cannot scale up: instance launch already in progress. Please wait for current launch to complete.",
                "launching_instances": launching_instances,
                "all_instances": all_instances,
                "total_instances": len(instances)
            }
        
        # Check current instance count
        current_count = len(self.instance_manager.instances)
        
        if current_count >= 1:
            # Get information about existing instances
            existing_instances = [
                {
                    "instance_id": i.instance_id,
                    "status": i.status.value,
                    "created_at": i.created_at.isoformat(),
                    "public_ip": i.public_ip,
                    "last_used": i.last_used.isoformat() if i.last_used else None
                }
                for i in self.instance_manager.instances.values()
            ]
            
            return {
                "error": "Cannot scale up: already have an instance. EBS volume can only attach to one instance.",
                "existing_instances": existing_instances,
                "total_instances": current_count
            }
        
        if count > 1:
            return {
                "error": "Cannot launch multiple instances: EBS volume can only attach to one instance at a time.",
                "total_instances": current_count
            }
        
        # Try to launch the instance
        try:
            instance_id = await self.instance_manager.launch_instance()
            
            # Get information about the newly launched instance
            launched_instance = self.instance_manager.instances.get(instance_id)
            launched_instance_info = {
                "instance_id": instance_id,
                "status": launched_instance.status.value if launched_instance else "launching",
                "created_at": launched_instance.created_at.isoformat() if launched_instance else None,
                "public_ip": launched_instance.public_ip if launched_instance else "pending"
            }
            
            return {
                "message": "Instance launched successfully",
                "launched_instances": [launched_instance_info],
                "total_instances": len(self.instance_manager.instances)
            }
        except ValueError as e:
            return {
                "error": str(e),
                "total_instances": len(self.instance_manager.instances)
            }
        except Exception as e:
            print(f"Failed to launch instance: {e}")
            return {
                "error": f"Failed to launch instance: {str(e)}",
                "total_instances": len(self.instance_manager.instances)
            }
    
    async def scale_down(self, count: int = 1) -> Dict[str, Any]:
        """Scale down by terminating instances (in single-instance setup, this shuts down the service)"""
        
        current_instances = list(self.instance_manager.instances.values())
        
        if not current_instances:
            return {
                "message": "No instances to terminate",
                "total_instances": 0
            }
        
        # In single-instance setup, scale-down means shutting down the service
        if len(current_instances) == 1:
            instance = current_instances[0]
            
            # Check if instance is currently processing requests
            if instance.current_requests > 0:
                return {
                    "error": f"Cannot terminate instance {instance.instance_id}: currently processing {instance.current_requests} requests. Wait for requests to complete or force terminate via DELETE /instance/{instance.instance_id}",
                    "total_instances": 1
                }
            
            # Terminate the single instance (shuts down service)
            try:
                await self.instance_manager.terminate_instance(instance.instance_id)
                return {
                    "message": "Service shut down: terminated the only instance",
                    "terminated_instances": [instance.instance_id],
                    "total_instances": 0
                }
            except Exception as e:
                return {
                    "error": f"Failed to terminate instance {instance.instance_id}: {str(e)}",
                    "total_instances": 1
                }
        
        # This shouldn't happen in single-instance setup, but handle gracefully
        return {
            "error": "Unexpected state: multiple instances found in single-instance setup",
            "total_instances": len(current_instances)
        }
    
    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        """Terminate a specific instance"""
        try:
            await self.instance_manager.terminate_instance(instance_id)
            return {
                "message": f"Instance {instance_id} terminated successfully",
                "terminated_instances": [instance_id],
                "total_instances": len(self.instance_manager.instances)
            }
        except ValueError as e:
            return {
                "error": f"Instance not found: {str(e)}",
                "total_instances": len(self.instance_manager.instances)
            }
        except Exception as e:
            return {
                "error": f"Failed to terminate instance: {str(e)}",
                "total_instances": len(self.instance_manager.instances)
            } 