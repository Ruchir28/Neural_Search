"""
Data models and enums for the semantic search orchestrator
"""
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any
from pydantic import BaseModel

class InstanceStatus(Enum):
    LAUNCHING = "launching"      # EC2 instance starting up
    STARTING = "starting"        # Instance running, service initializing
    READY = "ready"             # Service ready and responding to health checks
    TERMINATING = "terminating"  # Being shut down
    FAILED = "failed"           # Health checks failed

@dataclass
class ManagedInstance:
    instance_id: str
    public_ip: str
    private_ip: str
    status: InstanceStatus
    created_at: datetime
    last_used: datetime
    current_requests: int = 0
    health_check_failures: int = 0
    
    @property
    def is_ready_to_serve(self) -> bool:
        """Returns True if instance is ready to serve requests (responds to health checks)"""
        return self.status == InstanceStatus.READY
    
    @property
    def is_processing_requests(self) -> bool:
        """Returns True if instance is currently processing requests"""
        return self.current_requests > 0

# API Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 15

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    timings: Dict[str, float]
    instance_id: str

class InstanceInfo(BaseModel):
    instance_id: str
    public_ip: str
    status: str
    created_at: str
    last_used: str
    current_requests: int
    is_ready_to_serve: bool
    is_processing_requests: bool

class OrchestrationStatus(BaseModel):
    total_instances: int
    ready_instances: int
    processing_requests: int  # Instances currently handling requests
    launching_instances: int
    instances: List[InstanceInfo] 