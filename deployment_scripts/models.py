"""
Data models and enums for the semantic search orchestrator
"""
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any
from pydantic import BaseModel

class InstanceStatus(Enum):
    LAUNCHING = "launching"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    TERMINATING = "terminating"
    FAILED = "failed"

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

class OrchestrationStatus(BaseModel):
    total_instances: int
    ready_instances: int
    busy_instances: int
    launching_instances: int
    instances: List[InstanceInfo] 