"""
Instance lifecycle management for the semantic search orchestrator
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, Optional, List
import httpx

from config import OrchestratorConfig
from models import ManagedInstance, InstanceStatus
from aws_manager import AWSManager

class InstanceManager:
    """Manages the lifecycle of semantic search instances"""
    
    def __init__(self, config: OrchestratorConfig, aws_manager: AWSManager):
        self.config = config
        self.aws_manager = aws_manager
        self.instances: Dict[str, ManagedInstance] = {}
        self._launching_lock = asyncio.Lock()  # Prevent race conditions during launch
    
    def discover_existing_instances(self):
        """Discover and reconnect to existing instances using AWS tags"""
        print("Discovering existing instances...")
        
        try:
            aws_instances = self.aws_manager.discover_existing_instances()
            
            discovered_count = 0
            for instance_data in aws_instances:
                try:
                    instance_id = instance_data['InstanceId']
                    public_ip = instance_data.get('PublicIpAddress', '')
                    private_ip = instance_data.get('PrivateIpAddress', '')
                    instance_state = instance_data['State']['Name']
                    
                    # Skip instances without public IP (they won't be accessible)
                    if not public_ip:
                        print(f"⚠ Skipping instance {instance_id}: no public IP")
                        continue
                    
                    # Determine initial status based on EC2 state
                    if instance_state == 'running':
                        status = InstanceStatus.STARTING  # Will be health-checked
                    elif instance_state == 'pending':
                        status = InstanceStatus.LAUNCHING
                    else:
                        print(f"⚠ Skipping instance {instance_id}: unexpected state {instance_state}")
                        continue
                    
                    # Parse launch time
                    launch_time = instance_data['LaunchTime']
                    if hasattr(launch_time, 'timestamp'):
                        created_at = datetime.fromtimestamp(launch_time.timestamp())
                    else:
                        created_at = launch_time.replace(tzinfo=None)
                    
                    managed_instance = ManagedInstance(
                        instance_id=instance_id,
                        public_ip=public_ip,
                        private_ip=private_ip,
                        status=status,
                        created_at=created_at,
                        last_used=datetime.now()
                    )
                    
                    self.instances[instance_id] = managed_instance
                    discovered_count += 1
                    print(f"✓ Discovered instance: {instance_id} ({public_ip}) - {instance_state}")
                    
                except Exception as e:
                    print(f"⚠ Error processing discovered instance: {e}")
                    continue
            
            if discovered_count == 0:
                print("No existing instances found")
            else:
                print(f"✓ Discovered {discovered_count} existing instances")
                
        except Exception as e:
            print(f"⚠ Error discovering existing instances: {e}")
            # Don't fail startup if discovery fails - just continue with empty state
    
    async def validate_discovered_instances(self):
        """Validate discovered instances by health-checking them"""
        validation_tasks = []
        for instance_id, instance in self.instances.items():
            if instance.status == InstanceStatus.STARTING:
                task = asyncio.create_task(self._validate_discovered_instance(instance_id))
                validation_tasks.append(task)
        
        if validation_tasks:
            await asyncio.gather(*validation_tasks, return_exceptions=True)
    
    async def _validate_discovered_instance(self, instance_id: str):
        """Validate a discovered instance by health-checking it"""
        instance = self.instances.get(instance_id)
        if not instance:
            return
        
        print(f"Validating discovered instance {instance_id}...")
        
        # Give the instance some time to be ready if it was just starting
        max_attempts = 12  # 2 minutes
        attempt = 0
        
        while attempt < max_attempts:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"http://{instance.public_ip}:8000/health")
                    if response.status_code == 200:
                        instance.status = InstanceStatus.READY
                        print(f"✓ Instance {instance_id} is healthy and ready")
                        return
                    else:
                        print(f"⚠ Instance {instance_id} health check returned {response.status_code}")
            except Exception as e:
                print(f"⚠ Instance {instance_id} health check failed: {e}")
            
            await asyncio.sleep(10)
            attempt += 1
        
        # If we get here, the instance failed validation
        instance.status = InstanceStatus.FAILED
        print(f"✗ Instance {instance_id} failed validation - marking as failed")
    
    async def launch_instance(self) -> str:
        """Launch a new EC2 instance with the EBS volume attached"""
        
        async with self._launching_lock:
            if len(self.instances) >= 1:
                raise ValueError(f"Cannot launch instance: already have {len(self.instances)} instances (EBS volume can only attach to one instance)")
            
            # Check if any instance is currently launching
            for instance in self.instances.values():
                if instance.status in [InstanceStatus.LAUNCHING, InstanceStatus.STARTING]:
                    raise ValueError("Cannot launch instance: another instance is already launching/starting")
            
            # Create user data script to setup the instance
            user_data = self._create_user_data_script()
            
            try:
                # Launch instance
                instance_data = self.aws_manager.launch_instance(user_data)
                instance_id = instance_data['InstanceId']
                
                # Wait for instance to be running
                self.aws_manager.wait_for_instance_running(instance_id)
                
                # Get updated instance details
                instance_data = self.aws_manager.get_instance_details(instance_id)
                
                self.aws_manager.attach_ebs_volume(instance_id)
                
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
yum install git curl -y --allowerasing

# Mount EBS volume
log "Setting up EBS volume mount..."
mkdir -p /data

# Ensure log file has proper permissions
touch /var/log/semantic-search-setup.log
chmod 644 /var/log/semantic-search-setup.log

# Use volume ID to find the exact EBS device (more reliable than device name guessing)
VOLUME_ID="{self.config.ebs_volume_id}"
# Remove "vol-" prefix if present
VOLUME_ID_CLEAN=${{VOLUME_ID#vol-}}

log "Looking for EBS volume with ID: $VOLUME_ID_CLEAN"

# Wait for EBS volume to be available and find it using volume ID
max_attempts=48  # 4 minutes
attempt=0
ebs_device=""

while [ $attempt -lt $max_attempts ]; do
    # Find device using volume ID in /dev/disk/by-id/ (support both NVMe and SCSI)
    DEV_NAME=$(ls /dev/disk/by-id/ 2>/dev/null \
        | grep -x "nvme-Amazon_Elastic_Block_Store_vol${{VOLUME_ID_CLEAN}}" \
        || ls /dev/disk/by-id/ 2>/dev/null \
        | grep -x "scsi-0Amazon_Elastic_Block_Store_vol${{VOLUME_ID_CLEAN}}" \
        || true)
    
    if [ -n "$DEV_NAME" ]; then
        ebs_device="/dev/disk/by-id/$DEV_NAME"
        log "Found EBS volume at: $ebs_device"
        break
    fi
    
    log "Waiting for EBS volume to be available... (attempt $((attempt + 1))/$max_attempts)"
    sleep 5
    attempt=$((attempt + 1))
done

if [ -z "$ebs_device" ]; then
    log "ERROR: EBS volume with ID $VOLUME_ID_CLEAN not found after waiting"
    echo "Setup failed: EBS volume not found" > /tmp/setup_failed
    exit 1
fi

# Check if /data is already mounted to avoid double-mounting
if mountpoint -q /data; then
    log "EBS volume already mounted at /data"
else
    # Mount the EBS volume
    log "Mounting EBS volume from $ebs_device to /data..."
    if ! mount -t ext4 "$ebs_device" /data; then
        log "ERROR: Failed to mount EBS volume"
        echo "Setup failed: EBS volume mount failed" > /tmp/setup_failed
        exit 1
    fi
    log "EBS volume mounted successfully"
fi

# Verify mount and show some info
log "EBS volume mount verification:"
df -h /data | tee -a /var/log/semantic-search-setup.log

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
if ! sudo -u ec2-user git clone https://github.com/Ruchir28/Neural_Search.git; then
    log "ERROR: Failed to clone repository"
    echo "Setup failed: Repository clone failed" > /tmp/setup_failed
    exit 1
fi
cd Neural_Search
log "Repository cloned successfully"

# Activate the PyTorch virtual environment (PyTorch 2.6+ uses venv instead of conda)
log "Activating PyTorch virtual environment..."
source /opt/pytorch/bin/activate

# Verify virtual environment is active
if [ -z "$VIRTUAL_ENV" ] || [ ! -f "/opt/pytorch/bin/python" ]; then
    log "ERROR: Failed to activate PyTorch virtual environment"
    echo "Setup failed: Virtual environment activation failed" > /tmp/setup_failed
    exit 1
fi
log "PyTorch virtual environment activated: $VIRTUAL_ENV"

# Install Python dependencies with error checking
log "Installing Python dependencies..."
if ! pip install --no-cache-dir -r requirements.txt; then
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
COHERE_API_KEY={self.config.cohere_api_key}

# Load Embeddings to RAM Configuration
LOAD_EMBEDDINGS_TO_RAM={self.config.load_embeddings_to_ram}
EOF

# Set environment variables for current session and future sessions
export DATA_DIR="/data"
export COHERE_API_KEY="{self.config.cohere_api_key}"
export LOAD_EMBEDDINGS_TO_RAM="{self.config.load_embeddings_to_ram}"

# Add environment variables to ec2-user's bashrc (ensure proper ownership)
sudo -u ec2-user bash -c 'echo "export DATA_DIR=\"/data\"" >> /home/ec2-user/.bashrc'
sudo -u ec2-user bash -c 'echo "export COHERE_API_KEY=\"{self.config.cohere_api_key}\"" >> /home/ec2-user/.bashrc'
sudo -u ec2-user bash -c 'echo "export LOAD_EMBEDDINGS_TO_RAM=\"{self.config.load_embeddings_to_ram}\"" >> /home/ec2-user/.bashrc'

# Ensure the PyTorch virtual environment is activated for future sessions
sudo -u ec2-user bash -c 'echo "source /opt/pytorch/bin/activate" >> /home/ec2-user/.bashrc'

# Start the semantic search server with data paths pointing to EBS volume
log "Starting semantic search server..."
# Run the server as ec2-user for proper permissions
sudo -u ec2-user bash -c 'cd /home/ec2-user/Neural_Search && source /opt/pytorch/bin/activate && DATA_DIR="/data" COHERE_API_KEY="{self.config.cohere_api_key}" LOAD_EMBEDDINGS_TO_RAM="{self.config.load_embeddings_to_ram}" nohup python server.py > server.log 2>&1 &'

# Wait for server to be ready with timeout
log "Waiting for server to be ready..."
timeout=900  # 15 minutes (increased from 5 minutes for large embedding loading)
elapsed=0
last_progress=-1
while [ $elapsed -lt $timeout ]; do
    # Check if the HTTP endpoint is responding
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        # Get the health response
        health_response=$(curl -s http://localhost:8000/health 2>/dev/null)
        
        # Check if server is ready
        if echo "$health_response" | grep -q '"status".*"ready"'; then
            log "Semantic search server is ready and responding"
            echo "Instance ready" > /tmp/instance_ready
            break
        fi
        
        # Check if server is initializing and extract progress
        if echo "$health_response" | grep -q '"status".*"initializing"'; then
            # Extract progress percentage (simple grep approach)
            progress=$(echo "$health_response" | grep -o '"progress"[[:space:]]*:[[:space:]]*[0-9]*' | grep -o '[0-9]*$' || echo "0")
            stage=$(echo "$health_response" | grep -o '"stage"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*"\\([^"]*\\)".*/\\1/' || echo "unknown")
            
            # Only log progress updates to avoid spam
            if [ "$progress" != "$last_progress" ] && [ -n "$progress" ]; then
                log "Server initializing: $progress% - $stage"
                last_progress="$progress"
            fi
        elif echo "$health_response" | grep -q '"status".*"error"'; then
            log "ERROR: Server initialization failed"
            echo "Setup failed: Server initialization error" > /tmp/setup_failed
            exit 1
        else
            log "Server responding but status unclear"
        fi
    else
        log "Waiting for server to start responding... ($elapsed/$timeout seconds)"
    fi
    
    sleep 30  # Check every 30 seconds
    elapsed=$((elapsed + 30))
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
        
        # Wait for the service to be ready - increased timeout for large embedding loading
        max_attempts = 90  # 15 minutes (increased from 10 minutes)
        attempt = 0
        
        while attempt < max_attempts:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"http://{instance.public_ip}:8000/health")
                    
                    if response.status_code == 200:
                        # Server is ready
                        response_data = response.json()
                        if response_data.get("status") == "ready":
                            instance.status = InstanceStatus.READY
                            print(f"✓ Instance {instance_id} is ready")
                            return
                        else:
                            # Server is still initializing, show progress
                            progress = response_data.get("progress", 0)
                            stage = response_data.get("stage", "unknown")
                            message = response_data.get("message", "Initializing...")
                            elapsed = response_data.get("elapsed_time", 0)
                            print(f"Instance {instance_id} initializing: {progress}% - {stage} - {message} (elapsed: {elapsed:.0f}s)")
                    
                    elif response.status_code == 503:
                        # Server returned error or still initializing
                        try:
                            error_data = response.json()
                            if isinstance(error_data, dict) and "detail" in error_data:
                                detail = error_data["detail"]
                                if isinstance(detail, dict):
                                    if detail.get("status") == "error":
                                        print(f"✗ Instance {instance_id} initialization failed: {detail.get('error', 'Unknown error')}")
                                        instance.status = InstanceStatus.FAILED
                                        return
                                    else:
                                        # Still initializing
                                        progress = detail.get("progress", 0)
                                        stage = detail.get("stage", "unknown")
                                        print(f"Instance {instance_id} initializing: {progress}% - {stage}")
                                else:
                                    print(f"Instance {instance_id} returned 503: {detail}")
                            else:
                                # Handle case where response is initializing status
                                if error_data.get("status") == "initializing":
                                    progress = error_data.get("progress", 0)
                                    stage = error_data.get("stage", "unknown")
                                    message = error_data.get("message", "Initializing...")
                                    elapsed = error_data.get("elapsed_time", 0)
                                    print(f"Instance {instance_id} initializing: {progress}% - {stage} - {message} (elapsed: {elapsed:.0f}s)")
                                else:
                                    print(f"Instance {instance_id} health check returned 503: {error_data}")
                        except:
                            print(f"Instance {instance_id} health check returned 503 (could not parse response)")
                    else:
                        print(f"Instance {instance_id} health check returned {response.status_code}")
                        
            except Exception as e:
                print(f"Instance {instance_id} health check failed: {e}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
            attempt += 1
        
        # If we get here, the instance failed to start within timeout
        instance.status = InstanceStatus.FAILED
        print(f"✗ Instance {instance_id} failed to start within {max_attempts * 10 // 60} minutes")
    
    async def get_available_instance(self) -> Optional[ManagedInstance]:
        """Get an available instance, launching one if necessary"""
        
        # Check for ready instances that can serve requests
        for instance in self.instances.values():
            if instance.is_ready_to_serve:
                return instance
        
        # If no available instances and no instances exist, launch one
        if len(self.instances) == 0:
            try:
                await self.launch_instance()
                # Wait a bit for the instance to start launching
                await asyncio.sleep(1)
                
                # Return the launching instance (caller will need to wait)
                for instance in self.instances.values():
                    if instance.status in [InstanceStatus.LAUNCHING, InstanceStatus.STARTING]:
                        return await self._wait_for_instance_ready(instance)
            except ValueError as e:
                print(f"Cannot launch instance: {e}")
                return None
        
        # If we have an instance but it's not ready, wait for it
        for instance in self.instances.values():
            if instance.status in [InstanceStatus.LAUNCHING, InstanceStatus.STARTING]:
                return await self._wait_for_instance_ready(instance)
        
        return None
    
    async def _wait_for_instance_ready(self, instance: ManagedInstance, timeout: int = 1800) -> Optional[ManagedInstance]:
        """Wait for an instance to become ready with enhanced status reporting"""
        start_time = time.time()
        last_progress_report = 0
        
        while time.time() - start_time < timeout:
            if instance.status == InstanceStatus.READY:
                return instance
            elif instance.status == InstanceStatus.FAILED:
                return None
            
            # Try to get status update from health check
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"http://{instance.public_ip}:8000/health")
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        if response_data.get("status") == "ready":
                            instance.status = InstanceStatus.READY
                            return instance
                        else:
                            # Show progress updates
                            progress = response_data.get("progress", 0)
                            stage = response_data.get("stage", "unknown")
                            message = response_data.get("message", "Initializing...")
                            elapsed = response_data.get("elapsed_time", 0)
                            
                            # Only print progress updates every 10% or stage change
                            if progress >= last_progress_report + 10 or stage != getattr(instance, '_last_stage', ''):
                                print(f"Instance {instance.instance_id} progress: {progress}% - {stage} - {message} (elapsed: {elapsed:.0f}s)")
                                last_progress_report = progress
                                instance._last_stage = stage
                                
                    elif response.status_code == 503:
                        # Handle initialization status
                        try:
                            error_data = response.json()
                            if isinstance(error_data, dict):
                                if "detail" in error_data:
                                    detail = error_data["detail"]
                                    if isinstance(detail, dict) and detail.get("status") == "error":
                                        print(f"✗ Instance {instance.instance_id} initialization failed: {detail.get('error', 'Unknown error')}")
                                        instance.status = InstanceStatus.FAILED
                                        return None
                                elif error_data.get("status") == "initializing":
                                    progress = error_data.get("progress", 0)
                                    stage = error_data.get("stage", "unknown")
                                    message = error_data.get("message", "Initializing...")
                                    elapsed = error_data.get("elapsed_time", 0)
                                    
                                    if progress >= last_progress_report + 10 or stage != getattr(instance, '_last_stage', ''):
                                        print(f"Instance {instance.instance_id} progress: {progress}% - {stage} - {message} (elapsed: {elapsed:.0f}s)")
                                        last_progress_report = progress
                                        instance._last_stage = stage
                        except:
                            pass  # Continue waiting
                            
            except:
                pass  # Continue waiting
            
            await asyncio.sleep(5)
        
        # Timeout - mark as failed
        elapsed_minutes = timeout // 60
        print(f"✗ Instance {instance.instance_id} failed to become ready within {elapsed_minutes} minutes")
        instance.status = InstanceStatus.FAILED
        return None
    
    async def health_check_instance(self, instance: ManagedInstance):
        """Perform health check on a single instance with enhanced status reporting"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"http://{instance.public_ip}:8000/health")
                
                if response.status_code == 200:
                    # Server is ready and healthy
                    response_data = response.json()
                    if response_data.get("status") == "ready":
                        instance.health_check_failures = 0
                        # Update last_used to current time since it's responding
                        instance.last_used = datetime.now()
                    else:
                        # Server is still initializing
                        progress = response_data.get("progress", 0)
                        stage = response_data.get("stage", "unknown")
                        print(f"Instance {instance.instance_id} still initializing: {progress}% - {stage}")
                        instance.health_check_failures = 0  # Don't count initialization as failure
                        
                elif response.status_code == 503:
                    # Server returned error or still initializing
                    try:
                        error_data = response.json()
                        if isinstance(error_data, dict):
                            if "detail" in error_data:
                                detail = error_data["detail"]
                                if isinstance(detail, dict) and detail.get("status") == "error":
                                    print(f"Instance {instance.instance_id} reported error: {detail.get('error', 'Unknown error')}")
                                    instance.health_check_failures += 1
                                else:
                                    # Still initializing
                                    instance.health_check_failures = 0
                            elif error_data.get("status") == "initializing":
                                # Server is initializing
                                progress = error_data.get("progress", 0)
                                stage = error_data.get("stage", "unknown")
                                print(f"Instance {instance.instance_id} initializing: {progress}% - {stage}")
                                instance.health_check_failures = 0
                            else:
                                instance.health_check_failures += 1
                        else:
                            instance.health_check_failures += 1
                    except:
                        instance.health_check_failures += 1
                else:
                    instance.health_check_failures += 1
                    
        except Exception as e:
            print(f"Health check failed for instance {instance.instance_id}: {e}")
            instance.health_check_failures += 1
        
        # If too many failures, mark as failed
        if instance.health_check_failures >= 3:
            instance.status = InstanceStatus.FAILED
            print(f"✗ Instance {instance.instance_id} marked as failed due to {instance.health_check_failures} consecutive health check failures")
    
    async def terminate_instance(self, instance_id: str):
        """Terminate a specific instance"""
        instance = self.instances.get(instance_id)
        if not instance:
            raise ValueError("Instance not found")
        
        try:
            # Detach EBS volume first
            self.aws_manager.detach_ebs_volume(instance_id)
            
            # Terminate instance
            self.aws_manager.terminate_instance(instance_id)
            
            # Remove from our tracking
            del self.instances[instance_id]
            
        except Exception as e:
            raise ValueError(f"Failed to terminate instance: {str(e)}")
    
    async def cleanup_failed_instances(self):
        """Properly terminate failed instances instead of just removing them from tracking"""
        failed_instances = [
            instance_id for instance_id, instance in self.instances.items()
            if instance.status == InstanceStatus.FAILED
        ]
        
        for instance_id in failed_instances:
            print(f"Terminating failed instance {instance_id} (will detach EBS volume and terminate EC2 instance)")
            try:
                await self.terminate_instance(instance_id)
                print(f"✓ Successfully terminated failed instance {instance_id}")
            except Exception as e:
                print(f"✗ Failed to terminate failed instance {instance_id}: {e}")
                # Still remove from tracking even if termination failed to avoid infinite loops
                if instance_id in self.instances:
                    del self.instances[instance_id]
                    print(f"Removed {instance_id} from tracking despite termination failure")
    
    def get_idle_instances(self) -> List[ManagedInstance]:
        """Get list of ready instances that are not processing requests, sorted by last used time"""
        idle_instances = [
            i for i in self.instances.values() 
            if i.is_ready_to_serve and not i.is_processing_requests
        ]
        idle_instances.sort(key=lambda x: x.last_used)  # Least recently used first
        return idle_instances 