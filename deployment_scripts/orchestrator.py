#!/usr/bin/env python3
"""
Semantic Search Orchestrator Server
This server manages EC2 instances for the semantic search service.
It can spin up instances, attach EBS volumes, and route requests.
"""
import os
import sys
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import boto3
    from botocore.exceptions import ClientError, WaiterError
    from botocore.config import Config
except ImportError:
    print("ERROR: boto3 not installed. Install with: pip install boto3")
    sys.exit(1)

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel
    import uvicorn
    import httpx
except ImportError:
    print("ERROR: FastAPI dependencies not installed. Install with: pip install fastapi uvicorn httpx")
    sys.exit(1)

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EBS_VOLUME_ID = os.getenv("EBS_VOLUME_ID")  # The volume with embedding data
EC2_INSTANCE_TYPE = os.getenv("EC2_INSTANCE_TYPE", "g4dn.4xlarge")  # GPU instance for CUDA
# Deep Learning AMI (Amazon Linux 2) - user specified AMI with NVIDIA drivers, CUDA, and PyTorch pre-installed
EC2_AMI_ID = os.getenv("EC2_AMI_ID", "ami-074d9c327b5296aaa")  # Deep Learning AMI (Amazon Linux 2) x86
EC2_KEY_PAIR = os.getenv("EC2_KEY_PAIR", "ec2_key")
EC2_SECURITY_GROUP = os.getenv("EC2_SECURITY_GROUP", "semantic-search-sg")
EC2_SUBNET_ID = os.getenv("EC2_SUBNET_ID")  # Optional, will use default VPC if not set

# Semantic search configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Required for semantic search functionality
LOAD_EMBEDDINGS_TO_RAM = os.getenv("LOAD_EMBEDDINGS_TO_RAM", "false").lower() in ("true", "1", "yes")  # Performance option

# Instance management settings
MAX_INSTANCES = int(os.getenv("MAX_INSTANCES", "1"))
INSTANCE_IDLE_TIMEOUT = int(os.getenv("INSTANCE_IDLE_TIMEOUT", "1800"))  # 30 minutes
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))  # 1 minute

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

class SemanticSearchOrchestrator:
    def __init__(self):
        self.instances: Dict[str, ManagedInstance] = {}
        self.ec2_client = self._create_ec2_client()
        self.app = FastAPI(title="Semantic Search Orchestrator")
        self._setup_routes()
        
        # Validate configuration
        if not EBS_VOLUME_ID:
            raise ValueError("EBS_VOLUME_ID environment variable is required")
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable is required for semantic search functionality")
        
        # Initialize AWS resources
        self.ebs_availability_zone = None
        self.target_subnet_id = None
        self.security_group_id = None
        self._initialize_aws_resources()
        
        # Add startup event to create background tasks
        @self.app.on_event("startup")
        async def startup_event():
            # Start background tasks after event loop is running
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._cleanup_loop())

    def _create_ec2_client(self):
        """Create EC2 client with optimized configuration"""
        config = Config(
            region_name=AWS_REGION,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        return boto3.client('ec2', config=config)

    def _initialize_aws_resources(self):
        """Initialize and validate AWS resources"""
        print("Initializing AWS resources...")
        
        # Get EBS volume info to determine AZ constraint
        self._get_ebs_volume_info()
        
        # Find or create security group
        self._ensure_security_group()
        
        # Find appropriate subnet in the same AZ as EBS volume
        self._find_target_subnet()
        
        # Validate key pair exists
        self._validate_key_pair()
        
        print(f"✓ EBS Volume AZ: {self.ebs_availability_zone}")
        print(f"✓ Target Subnet: {self.target_subnet_id}")
        print(f"✓ Security Group: {self.security_group_id}")

    def _get_ebs_volume_info(self):
        """Get EBS volume information including its availability zone"""
        try:
            response = self.ec2_client.describe_volumes(VolumeIds=[EBS_VOLUME_ID])
            volume = response['Volumes'][0]
            self.ebs_availability_zone = volume['AvailabilityZone']
            
            if volume['State'] != 'available':
                print(f"WARNING: EBS volume {EBS_VOLUME_ID} is in state '{volume['State']}', not 'available'")
                
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidVolume.NotFound':
                raise ValueError(f"EBS volume {EBS_VOLUME_ID} not found")
            raise

    def _ensure_security_group(self):
        """Find existing security group or create a new one"""
        try:
            # Try to find existing security group
            if EC2_SECURITY_GROUP:
                response = self.ec2_client.describe_security_groups(
                    Filters=[
                        {'Name': 'group-name', 'Values': [EC2_SECURITY_GROUP]}
                    ]
                )
                if response['SecurityGroups']:
                    self.security_group_id = response['SecurityGroups'][0]['GroupId']
                    print(f"✓ Found existing security group: {EC2_SECURITY_GROUP}")
                    return
            
            # Create new security group
            print("Creating new security group...")
            vpc_response = self.ec2_client.describe_vpcs(
                Filters=[{'Name': 'is-default', 'Values': ['true']}]
            )
            
            if not vpc_response['Vpcs']:
                raise ValueError("No default VPC found. Please specify EC2_SUBNET_ID.")
            
            default_vpc_id = vpc_response['Vpcs'][0]['VpcId']
            
            sg_name = EC2_SECURITY_GROUP or f"semantic-search-sg-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            response = self.ec2_client.create_security_group(
                GroupName=sg_name,
                Description='Security group for Semantic Search instances',
                VpcId=default_vpc_id
            )
            
            self.security_group_id = response['GroupId']
            
            # Add inbound rules
            self.ec2_client.authorize_security_group_ingress(
                GroupId=self.security_group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 8000,
                        'ToPort': 8000,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'Semantic Search API'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access'}]
                    }
                ]
            )
            
            print(f"✓ Created security group: {sg_name} ({self.security_group_id})")
            
        except ClientError as e:
            raise ValueError(f"Failed to setup security group: {e}")

    def _find_target_subnet(self):
        """Find a subnet in the same AZ as the EBS volume"""
        if EC2_SUBNET_ID:
            # Validate provided subnet is in correct AZ
            response = self.ec2_client.describe_subnets(SubnetIds=[EC2_SUBNET_ID])
            subnet = response['Subnets'][0]
            
            if subnet['AvailabilityZone'] != self.ebs_availability_zone:
                raise ValueError(
                    f"Provided subnet {EC2_SUBNET_ID} is in AZ {subnet['AvailabilityZone']}, "
                    f"but EBS volume is in AZ {self.ebs_availability_zone}. "
                    f"They must be in the same AZ."
                )
            
            self.target_subnet_id = EC2_SUBNET_ID
            print(f"✓ Using provided subnet: {EC2_SUBNET_ID}")
            return
        
        # Find a subnet in the correct AZ
        response = self.ec2_client.describe_subnets(
            Filters=[
                {'Name': 'availability-zone', 'Values': [self.ebs_availability_zone]},
                {'Name': 'default-for-az', 'Values': ['true']}
            ]
        )
        
        if not response['Subnets']:
            # Try any subnet in the AZ
            response = self.ec2_client.describe_subnets(
                Filters=[
                    {'Name': 'availability-zone', 'Values': [self.ebs_availability_zone]}
                ]
            )
        
        if not response['Subnets']:
            raise ValueError(f"No subnets found in AZ {self.ebs_availability_zone}")
        
        self.target_subnet_id = response['Subnets'][0]['SubnetId']
        print(f"✓ Found subnet in correct AZ: {self.target_subnet_id}")

    def _validate_key_pair(self):
        """Validate that the key pair exists"""
        if not EC2_KEY_PAIR:
            print("WARNING: No key pair specified. You won't be able to SSH to instances.")
            return
        
        try:
            self.ec2_client.describe_key_pairs(KeyNames=[EC2_KEY_PAIR])
            print(f"✓ Key pair exists: {EC2_KEY_PAIR}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidKeyPair.NotFound':
                raise ValueError(f"Key pair '{EC2_KEY_PAIR}' not found. Please create it first.")
            raise

    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/query", response_model=QueryResponse)
        async def handle_query(request: QueryRequest):
            return await self._handle_query(request)
        
        @self.app.get("/status", response_model=OrchestrationStatus)
        async def get_status():
            return await self._get_status()
        
        @self.app.post("/scale-up")
        async def scale_up(count: int = 1):
            return await self._scale_up(count)
        
        @self.app.post("/scale-down")
        async def scale_down(count: int = 1):
            return await self._scale_down(count)
        
        @self.app.delete("/instance/{instance_id}")
        async def terminate_instance(instance_id: str):
            return await self._terminate_instance(instance_id)

    async def _handle_query(self, request: QueryRequest) -> QueryResponse:
        """Handle a semantic search query by routing to an available instance"""
        
        # Find an available instance
        instance = await self._get_available_instance()
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

    async def _get_available_instance(self) -> Optional[ManagedInstance]:
        """Get an available instance, launching one if necessary"""
        
        # Check for idle instances first
        for instance in self.instances.values():
            if instance.status == InstanceStatus.IDLE:
                return instance
        
        # Check for ready instances with low load
        for instance in self.instances.values():
            if instance.status == InstanceStatus.READY and instance.current_requests == 0:
                return instance
        
        # If no available instances and under limit, launch a new one
        if len(self.instances) < MAX_INSTANCES:
            await self._launch_instance()
            # Wait a bit for the instance to start launching
            await asyncio.sleep(1)
            
            # Return the launching instance (caller will need to wait)
            for instance in self.instances.values():
                if instance.status == InstanceStatus.LAUNCHING:
                    return await self._wait_for_instance_ready(instance)
        
        return None

    async def _wait_for_instance_ready(self, instance: ManagedInstance, timeout: int = 300) -> Optional[ManagedInstance]:
        """Wait for an instance to become ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if instance.status == InstanceStatus.READY:
                return instance
            elif instance.status == InstanceStatus.FAILED:
                return None
            
            await asyncio.sleep(5)
        
        # Timeout - mark as failed
        instance.status = InstanceStatus.FAILED
        return None

    async def _launch_instance(self) -> str:
        """Launch a new EC2 instance with the EBS volume attached"""
        
        # Create user data script to setup the instance
        user_data = self._create_user_data_script()
        
        try:
            # Launch instance
            launch_params = {
                'ImageId': EC2_AMI_ID,
                'MinCount': 1,
                'MaxCount': 1,
                'InstanceType': EC2_INSTANCE_TYPE,
                'SecurityGroupIds': [self.security_group_id],
                'SubnetId': self.target_subnet_id,
                'UserData': user_data,
                'TagSpecifications': [
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'semantic-search-{datetime.now().strftime("%Y%m%d-%H%M%S")}'},
                            {'Key': 'Service', 'Value': 'semantic-search'},
                            {'Key': 'ManagedBy', 'Value': 'orchestrator'},
                            {'Key': 'EBSVolumeId', 'Value': EBS_VOLUME_ID}
                        ]
                    }
                ]
            }
            
            # Add key pair if specified
            if EC2_KEY_PAIR:
                launch_params['KeyName'] = EC2_KEY_PAIR
            
            response = self.ec2_client.run_instances(**launch_params)
            
            instance_id = response['Instances'][0]['InstanceId']
            
            # Wait for instance to be running
            waiter = self.ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id], WaiterConfig={'Delay': 15, 'MaxAttempts': 20})
            
            # Get instance details
            instances = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            instance_data = instances['Reservations'][0]['Instances'][0]
            
            # Attach EBS volume
            await self._attach_ebs_volume(instance_id)
            
            # Create managed instance
            managed_instance = ManagedInstance(
                instance_id=instance_id,
                public_ip=instance_data.get('PublicIpAddress', ''),
                private_ip=instance_data.get('PrivateIpAddress', ''),
                status=InstanceStatus.STARTING,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            self.instances[instance_id] = managed_instance
            
            # Start monitoring the instance
            asyncio.create_task(self._monitor_instance_startup(instance_id))
            
            return instance_id
            
        except Exception as e:
            print(f"Failed to launch instance: {e}")
            raise

    async def _attach_ebs_volume(self, instance_id: str):
        """Attach the EBS volume to the instance"""
        try:
            self.ec2_client.attach_volume(
                VolumeId=EBS_VOLUME_ID,
                InstanceId=instance_id,
                Device='/dev/sdf'  # Will appear as /dev/xvdf on the instance
            )
            
            # Wait for volume to be attached
            waiter = self.ec2_client.get_waiter('volume_in_use')
            waiter.wait(VolumeIds=[EBS_VOLUME_ID], WaiterConfig={'Delay': 15, 'MaxAttempts': 20})
            
        except Exception as e:
            print(f"Failed to attach EBS volume to {instance_id}: {e}")
            raise

    def _create_user_data_script(self) -> str:
        """Create user data script for instance initialization"""
        return f"""#!/bin/bash
set -e  # Exit on any error

# Logging function
log() {{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /var/log/semantic-search-setup.log
}}

log "Starting semantic search server setup..."

# Update system (minimal updates since Deep Learning AMI is pre-configured)
log "Updating system packages..."
yum update -y

# Install git and curl (Python and pip are already available in Deep Learning AMI)
log "Installing git and curl..."
yum install git curl -y

# Mount EBS volume
log "Setting up EBS volume mount..."
mkdir -p /data
while [ ! -e /dev/xvdf ]; do 
    log "Waiting for EBS volume to be available..."
    sleep 5
done
mount /dev/xvdf /data
log "EBS volume mounted successfully"

# Verify critical data files exist
log "Verifying data files on EBS volume..."
required_files=(
    "/data/trained_indices/trained_ivfpq_index_full.bin"
    "/data/lists_info/embeddings_listwise.memmap"
    "/data/lists_info/layout_sorted.npz"
    "/data/lists_info/vec_id_to_sorted_row.bin.npy"
    "/data/lists_info/metadata_25gb.lmdb"
)

for file in "${{required_files[@]}}"; do
    if [ ! -e "$file" ]; then
        log "ERROR: Required data file not found: $file"
        echo "Setup failed: Missing data file $file" > /tmp/setup_failed
        exit 1
    fi
done
log "All required data files verified"

# Clone repository and setup
log "Cloning repository..."
cd /home/ec2-user
if ! git clone https://github.com/Ruchir28/Neural_Search.git; then
    log "ERROR: Failed to clone repository"
    echo "Setup failed: Repository clone failed" > /tmp/setup_failed
    exit 1
fi
cd Neural_Search
log "Repository cloned successfully"

# Activate the pre-installed conda environment with PyTorch and CUDA
log "Activating PyTorch conda environment..."
source /opt/miniconda3/bin/activate pytorch

# Verify conda environment is active
if [ "$CONDA_DEFAULT_ENV" != "pytorch" ]; then
    log "ERROR: Failed to activate pytorch conda environment"
    echo "Setup failed: Conda environment activation failed" > /tmp/setup_failed
    exit 1
fi
log "PyTorch environment activated: $CONDA_DEFAULT_ENV"

# Install Python dependencies with error checking
log "Installing Python dependencies..."
if ! pip install -r requirements.txt; then
    log "ERROR: Failed to install Python dependencies"
    echo "Setup failed: Pip install failed" > /tmp/setup_failed
    exit 1
fi
log "Python dependencies installed successfully"

# Verify critical packages are installed
log "Verifying critical packages..."
python -c "import cupy, cuvs, cohere, fastapi, lmdb" || {{
    log "ERROR: Critical packages not properly installed"
    echo "Setup failed: Package verification failed" > /tmp/setup_failed
    exit 1
}}
log "Critical packages verified"

# Create config.env file with environment variables
log "Creating config.env file..."
cat > config.env << 'EOF'
# Data Directory Configuration
DATA_DIR=/data

# Cohere API Configuration
COHERE_API_KEY={COHERE_API_KEY}

# Load Embeddings to RAM Configuration
LOAD_EMBEDDINGS_TO_RAM={LOAD_EMBEDDINGS_TO_RAM}
EOF

# Set environment variables for current session and future sessions
export DATA_DIR="/data"
export COHERE_API_KEY="{COHERE_API_KEY}"
export LOAD_EMBEDDINGS_TO_RAM="{LOAD_EMBEDDINGS_TO_RAM}"
echo 'export DATA_DIR="/data"' >> /home/ec2-user/.bashrc
echo 'export COHERE_API_KEY="{COHERE_API_KEY}"' >> /home/ec2-user/.bashrc
echo 'export LOAD_EMBEDDINGS_TO_RAM="{LOAD_EMBEDDINGS_TO_RAM}"' >> /home/ec2-user/.bashrc

# Ensure the conda environment is activated for future sessions
echo 'source /opt/miniconda3/bin/activate pytorch' >> /home/ec2-user/.bashrc

# Start the semantic search server with data paths pointing to EBS volume
log "Starting semantic search server..."
# Use the conda environment's Python and set environment variables
DATA_DIR="/data" COHERE_API_KEY="{COHERE_API_KEY}" LOAD_EMBEDDINGS_TO_RAM="{LOAD_EMBEDDINGS_TO_RAM}" nohup /opt/miniconda3/envs/pytorch/bin/python server.py > server.log 2>&1 &

# Wait for server to be ready with timeout
log "Waiting for server to be ready..."
timeout=300  # 5 minutes
elapsed=0
while [ $elapsed -lt $timeout ]; do
    # Check if the HTTP endpoint is responding
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log "Semantic search server is ready and responding"
        echo "Instance ready" > /tmp/instance_ready
        break
    fi
    
    sleep 10
    elapsed=$((elapsed + 10))
    log "Waiting for server... ($elapsed/$timeout seconds)"
done

if [ $elapsed -ge $timeout ]; then
    log "ERROR: Server failed to start within timeout"
    log "Server logs:"
    tail -50 server.log
    echo "Setup failed: Server startup timeout" > /tmp/setup_failed
    exit 1
fi

log "Semantic search server setup completed successfully"
"""

    async def _monitor_instance_startup(self, instance_id: str):
        """Monitor instance startup and update status"""
        instance = self.instances.get(instance_id)
        if not instance:
            return
        
        # Wait for the service to be ready
        max_attempts = 60  # 10 minutes
        attempt = 0
        
        while attempt < max_attempts:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"http://{instance.public_ip}:8000/status")
                    if response.status_code == 200:
                        instance.status = InstanceStatus.READY
                        print(f"Instance {instance_id} is ready")
                        return
            except:
                pass
            
            await asyncio.sleep(10)
            attempt += 1
        
        # If we get here, the instance failed to start
        instance.status = InstanceStatus.FAILED
        print(f"Instance {instance_id} failed to start")

    async def _health_check_loop(self):
        """Periodic health check for all instances"""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            
            for instance in list(self.instances.values()):
                if instance.status in [InstanceStatus.READY, InstanceStatus.IDLE, InstanceStatus.BUSY]:
                    await self._health_check_instance(instance)

    async def _health_check_instance(self, instance: ManagedInstance):
        """Perform health check on a single instance"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://{instance.public_ip}:8000/health")
                if response.status_code == 200:
                    instance.health_check_failures = 0
                    if instance.status == InstanceStatus.BUSY and instance.current_requests == 0:
                        instance.status = InstanceStatus.IDLE
                else:
                    instance.health_check_failures += 1
        except:
            instance.health_check_failures += 1
        
        # If too many failures, mark as failed
        if instance.health_check_failures >= 3:
            instance.status = InstanceStatus.FAILED
            print(f"Instance {instance.instance_id} marked as failed due to health check failures")

    async def _cleanup_loop(self):
        """Periodic cleanup of idle instances"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            current_time = datetime.now()
            
            for instance_id, instance in list(self.instances.items()):
                # Remove failed instances
                if instance.status == InstanceStatus.FAILED:
                    await self._terminate_instance(instance_id)
                
                # Terminate idle instances that have been idle too long
                elif (instance.status == InstanceStatus.IDLE and 
                      (current_time - instance.last_used).seconds > INSTANCE_IDLE_TIMEOUT):
                    print(f"Terminating idle instance {instance_id}")
                    await self._terminate_instance(instance_id)

    async def _get_status(self) -> OrchestrationStatus:
        """Get current orchestration status"""
        total = len(self.instances)
        ready = sum(1 for i in self.instances.values() if i.status == InstanceStatus.READY)
        busy = sum(1 for i in self.instances.values() if i.status == InstanceStatus.BUSY)
        launching = sum(1 for i in self.instances.values() if i.status == InstanceStatus.LAUNCHING)
        
        instances = [
            InstanceInfo(
                instance_id=i.instance_id,
                public_ip=i.public_ip,
                status=i.status.value,
                created_at=i.created_at.isoformat(),
                last_used=i.last_used.isoformat(),
                current_requests=i.current_requests
            )
            for i in self.instances.values()
        ]
        
        return OrchestrationStatus(
            total_instances=total,
            ready_instances=ready,
            busy_instances=busy,
            launching_instances=launching,
            instances=instances
        )

    async def _scale_up(self, count: int = 1) -> Dict[str, Any]:
        """Scale up by launching additional instances"""
        launched = []
        
        for _ in range(count):
            if len(self.instances) >= MAX_INSTANCES:
                break
            
            try:
                instance_id = await self._launch_instance()
                launched.append(instance_id)
            except Exception as e:
                print(f"Failed to launch instance: {e}")
        
        return {"launched_instances": launched, "total_instances": len(self.instances)}

    async def _scale_down(self, count: int = 1) -> Dict[str, Any]:
        """Scale down by terminating idle instances"""
        terminated = []
        
        # Find idle instances to terminate
        idle_instances = [i for i in self.instances.values() if i.status == InstanceStatus.IDLE]
        idle_instances.sort(key=lambda x: x.last_used)  # Terminate least recently used first
        
        for instance in idle_instances[:count]:
            try:
                await self._terminate_instance(instance.instance_id)
                terminated.append(instance.instance_id)
            except Exception as e:
                print(f"Failed to terminate instance {instance.instance_id}: {e}")
        
        return {"terminated_instances": terminated, "total_instances": len(self.instances)}

    async def _terminate_instance(self, instance_id: str) -> Dict[str, str]:
        """Terminate a specific instance"""
        instance = self.instances.get(instance_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Instance not found")
        
        try:
            # Detach EBS volume first
            self.ec2_client.detach_volume(VolumeId=EBS_VOLUME_ID, InstanceId=instance_id)
            
            # Terminate instance
            self.ec2_client.terminate_instances(InstanceIds=[instance_id])
            
            # Remove from our tracking
            del self.instances[instance_id]
            
            return {"message": f"Instance {instance_id} terminated successfully"}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to terminate instance: {str(e)}")

def main():
    """Main function to run the orchestrator"""
    print("Starting Semantic Search Orchestrator...")
    
    # Validate AWS credentials
    try:
        boto3.Session().get_credentials()
    except Exception as e:
        print(f"AWS credentials not configured: {e}")
        sys.exit(1)
    
    # Create and run orchestrator
    orchestrator = SemanticSearchOrchestrator()
    
    uvicorn.run(
        orchestrator.app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )

if __name__ == "__main__":
    main() 