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
        
        # Mark instance as busy
        instance.status = InstanceStatus.BUSY
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
            # Mark instance as available again
            instance.current_requests -= 1
            if instance.current_requests == 0:
                instance.status = InstanceStatus.IDLE
    
    async def get_status(self) -> OrchestrationStatus:
        """Get current orchestration status"""
        instances = self.instance_manager.instances
        
        total = len(instances)
        ready = sum(1 for i in instances.values() if i.status == InstanceStatus.READY)
        busy = sum(1 for i in instances.values() if i.status == InstanceStatus.BUSY)
        launching = sum(1 for i in instances.values() if i.status == InstanceStatus.LAUNCHING)
        
        instance_infos = [
            InstanceInfo(
                instance_id=i.instance_id,
                public_ip=i.public_ip,
                status=i.status.value,
                created_at=i.created_at.isoformat(),
                last_used=i.last_used.isoformat(),
                current_requests=i.current_requests
            )
            for i in instances.values()
        ]
        
        return OrchestrationStatus(
            total_instances=total,
            ready_instances=ready,
            busy_instances=busy,
            launching_instances=launching,
            instances=instance_infos
        )
    
    async def scale_up(self, count: int = 1) -> Dict[str, Any]:
        """Scale up by launching additional instances"""
        launched = []
        
        for _ in range(count):
            if len(self.instance_manager.instances) >= self.instance_manager.config.max_instances:
                break
            
            try:
                instance_id = await self.instance_manager.launch_instance()
                launched.append(instance_id)
            except Exception as e:
                print(f"Failed to launch instance: {e}")
        
        return {
            "launched_instances": launched, 
            "total_instances": len(self.instance_manager.instances)
        }
    
    async def scale_down(self, count: int = 1) -> Dict[str, Any]:
        """Scale down by terminating idle instances"""
        terminated = []
        
        # Find idle instances to terminate
        idle_instances = self.instance_manager.get_idle_instances()
        
        for instance in idle_instances[:count]:
            try:
                await self.instance_manager.terminate_instance(instance.instance_id)
                terminated.append(instance.instance_id)
            except Exception as e:
                print(f"Failed to terminate instance {instance.instance_id}: {e}")
        
        return {
            "terminated_instances": terminated, 
            "total_instances": len(self.instance_manager.instances)
        }
    
    async def terminate_instance(self, instance_id: str) -> Dict[str, str]:
        """Terminate a specific instance"""
        try:
            await self.instance_manager.terminate_instance(instance_id)
            return {"message": f"Instance {instance_id} terminated successfully"}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 