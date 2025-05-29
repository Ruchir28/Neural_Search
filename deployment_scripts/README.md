# Semantic Search Orchestrator

This orchestrator manages EC2 instances for the semantic search service, providing automatic scaling, health monitoring, and instance discovery.

## Key Features

### Instance Discovery on Restart
The orchestrator can now **automatically discover and reconnect** to existing instances when it restarts. This prevents:
- Resource waste from orphaned instances
- Loss of running instances when the orchestrator restarts
- Need to manually track instance state

### How Instance Discovery Works

1. **On startup**, the orchestrator queries AWS for existing instances using tags:
   - `Service: semantic-search`
   - `ManagedBy: orchestrator` 
   - `EBSVolumeId: <your-volume-id>`

2. **Health validation** - Discovered instances are health-checked to ensure they're responsive

3. **State restoration** - Healthy instances are added back to the orchestrator's management

### Instance Tagging
All launched instances are automatically tagged with:
```
Name: semantic-search-YYYYMMDD-HHMMSS
Service: semantic-search
ManagedBy: orchestrator
EBSVolumeId: vol-xxxxxxxxxxxxxxxxx
```

## Configuration

Copy `orchestrator_config.env.example` to `orchestrator_config.env` and configure:

```bash
# Required
EBS_VOLUME_ID=vol-xxxxxxxxxxxxxxxxx
COHERE_API_KEY=your_cohere_api_key_here

# Optional
AWS_REGION=us-east-1
EC2_INSTANCE_TYPE=g4dn.4xlarge
INSTANCE_IDLE_TIMEOUT=1800

# Spot Instance Configuration (Cost Savings)
USE_SPOT_INSTANCES=true
# SPOT_MAX_PRICE=0.50  # Optional, leave empty to use current spot price
```

## Usage

```bash
# Install dependencies
pip install boto3 fastapi uvicorn httpx python-dotenv

# Run orchestrator
python run_orchestrator.py
```

## API Endpoints

- `POST /query` - Submit semantic search queries
- `GET /status` - View orchestration status and instance health
- `POST /scale-up?count=N` - Launch a replacement instance (max 1 due to EBS)
- `POST /scale-down?count=N` - Shut down the service (terminates the instance)
- `DELETE /instance/{instance_id}` - Force terminate specific instance

## Instance Lifecycle

1. **LAUNCHING** - EC2 instance starting up
2. **STARTING** - Instance running, semantic search service initializing
3. **READY** - Available for queries
4. **BUSY** - Processing queries
5. **IDLE** - Ready but not processing queries
6. **FAILED** - Health checks failed
7. **TERMINATING** - Being shut down

## Monitoring

The orchestrator provides:
- Automatic health checking every 60 seconds
- Idle instance cleanup after 30 minutes
- Failed instance detection and marking
- Real-time status via `/status` endpoint

## Recovery Scenarios

### Orchestrator Restart
✅ **Automatic** - Discovers and reconnects to existing instances

### Instance Failure
✅ **Automatic** - Failed instances marked and excluded from routing

### Network Issues
✅ **Resilient** - Health check failures tracked, instances marked failed after 3 consecutive failures

### EBS Volume Issues
⚠️ **Manual** - Requires investigation of volume attachment and data integrity

## Features

- **Dynamic Scaling**: Automatically launches EC2 instances when needed
- **EBS Volume Management**: Attaches your data volume to instances in the correct AZ
- **Health Monitoring**: Continuous health checks and automatic failure recovery
- **Resource Management**: Auto-creates security groups and finds appropriate subnets
- **Load Balancing**: Routes requests to available instances
- **Cost Optimization**: Terminates idle instances after configurable timeout
- **Spot Instance Support**: Use spot instances for up to 90% cost savings

## Prerequisites

1. **AWS Credentials**: Configure AWS CLI or set environment variables
2. **EBS Volume**: Your semantic search data must be on an EBS volume
3. **Python 3.8+**: Required for the orchestrator

## Quick Setup

### 1. Install Dependencies

```bash
cd deployment_scripts
pip install -r orchestrator_requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example configuration
cp orchestrator_config.env.example orchestrator_config.env

# Edit the configuration file
nano orchestrator_config.env
```

**Required Configuration:**
```bash
# Your EBS volume ID (REQUIRED)
EBS_VOLUME_ID=vol-xxxxxxxxxxxxxxxxx

# AWS region where your EBS volume exists
AWS_REGION=us-east-1

# Instance type (GPU recommended for semantic search)
EC2_INSTANCE_TYPE=g4dn.xlarge

# Optional: Specify subnet if you have preferences
EC2_SUBNET_ID=subnet-xxxxxxxxxxxxxxxxx
```

### 3. Run the Orchestrator

```bash
python run_orchestrator.py
```

The orchestrator will:
- ✅ Validate your EBS volume exists
- ✅ Find the correct Availability Zone
- ✅ Create security group if needed (ports 8000, 22)
- ✅ Find appropriate subnet in the same AZ
- ✅ Start the orchestrator server on port 8080

## Usage

### Query the Semantic Search

```bash
# Send a search query
curl -X POST "http://localhost:8080/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 10}'
```

### Monitor Status

```bash
# Check orchestrator status
curl http://localhost:8080/status

# Response example:
{
  "total_instances": 1,
  "ready_instances": 1,
  "busy_instances": 0,
  "launching_instances": 0,
  "instances": [
    {
      "instance_id": "i-1234567890abcdef0",
      "public_ip": "54.123.45.67",
      "status": "ready",
      "created_at": "2024-01-15T10:30:00",
      "last_used": "2024-01-15T10:35:00",
      "current_requests": 0
    }
  ]
}
```

### Manual Scaling

```bash
# Scale up (launch replacement instance if none exists)
curl -X POST "http://localhost:8080/scale-up?count=1"

# Scale down (shut down service by terminating instance)
curl -X POST "http://localhost:8080/scale-down?count=1"

# Force terminate specific instance
curl -X DELETE "http://localhost:8080/instance/i-1234567890abcdef0"
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `EBS_VOLUME_ID` | *Required* | Your EBS volume with embedding data |
| `AWS_REGION` | `us-east-1` | AWS region |
| `EC2_INSTANCE_TYPE` | `g4dn.xlarge` | Instance type (GPU recommended) |
| `EC2_AMI_ID` | `ami-0c02fb55956c7d316` | Amazon Linux 2 AMI |
| `EC2_KEY_PAIR` | `semantic-search-key` | Key pair for SSH access |
| `EC2_SECURITY_GROUP` | `semantic-search-sg` | Security group name |
| `EC2_SUBNET_ID` | *Auto-detected* | Subnet ID (must be in same AZ as EBS) |
| `USE_SPOT_INSTANCES` | `false` | Use spot instances for cost savings |
| `SPOT_MAX_PRICE` | *Current price* | Maximum price for spot instances (optional) |
| `INSTANCE_IDLE_TIMEOUT` | `1800` | Seconds before terminating idle instances |
| `INSTANCE_MAX_RUNTIME` | `1800` | Maximum seconds an instance can run (30 min default) |
| `HEALTH_CHECK_INTERVAL` | `60` | Seconds between health checks |

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Request  │───▶│   Orchestrator   │───▶│  EC2 Instance   │
└─────────────────┘    │   (Port 8080)    │    │  (Port 8000)    │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Auto Scaling    │    │   EBS Volume    │
                       │  Health Checks   │    │ (Embedding Data)│
                       └──────────────────┘    └─────────────────┘
```

## Important Notes

### EBS Volume and Availability Zones

⚠️ **Critical**: Your EC2 instances must be in the same Availability Zone as your EBS volume. The orchestrator automatically:

1. Detects your EBS volume's AZ
2. Finds a subnet in that AZ
3. Launches instances in the correct location

If you specify `EC2_SUBNET_ID`, ensure it's in the same AZ as your EBS volume.

### Single Instance Limitation

🔒 **Important**: Due to EBS volume constraints, only **one instance can run at a time**:

- EBS volumes can only be attached to one EC2 instance
- The orchestrator enforces single-instance operation
- Scale-up requests are limited to single instance replacement
- Multiple scale-up calls are protected by locks to prevent race conditions

This means the orchestrator provides **high availability through replacement**, not horizontal scaling.

### Security Groups

The orchestrator automatically creates a security group with:
- **Port 8000**: For semantic search API
- **Port 22**: For SSH access (if key pair specified)
- **Source**: `0.0.0.0/0` (adjust for production)

### Instance Startup Time

Instances take 5-10 minutes to become ready because they need to:
1. Boot and install dependencies
2. Mount the EBS volume
3. Copy/link embedding data
4. Load the semantic search model

### Cost Management

- Instances auto-terminate after 30 minutes of inactivity
- Instances are force-terminated after 30 minutes of total runtime (configurable)
- Monitor with `GET /status` endpoint

### Spot Instances

💰 **Cost Savings**: Enable spot instances for up to 90% cost reduction:

```bash
USE_SPOT_INSTANCES=true
# SPOT_MAX_PRICE=0.50  # Optional price limit
```

**Benefits:**
- Significant cost savings (typically 50-90% off on-demand price)
- Same performance as on-demand instances
- Automatic fallback if spot capacity unavailable

**Considerations:**
- Instances can be interrupted with 2-minute notice
- Best for fault-tolerant workloads
- Orchestrator automatically handles interruptions by launching replacements

## Troubleshooting

### Common Issues

**"EBS volume not found"**
```bash
# Check if volume exists and you have permissions
aws ec2 describe-volumes --volume-ids vol-xxxxxxxxxxxxxxxxx
```

**"No subnets found in AZ"**
```bash
# Check available subnets in your EBS volume's AZ
aws ec2 describe-subnets --filters "Name=availability-zone,Values=us-east-1a"
```

**"Key pair not found"**
```bash
# Create a key pair or update config
aws ec2 create-key-pair --key-name semantic-search-key
```

**Instance fails to start**
```bash
# Check instance logs
aws ec2 get-console-output --instance-id i-xxxxxxxxxxxxxxxxx
```

### Logs

The orchestrator logs to stdout. For production, redirect to a file:
```bash
python run_orchestrator.py > orchestrator.log 2>&1 &
```

## Production Deployment

For production use:

1. **Use a dedicated VPC** with private subnets
2. **Restrict security group** to specific IP ranges
3. **Use IAM roles** instead of access keys
4. **Set up CloudWatch monitoring**
5. **Use Application Load Balancer** for high availability
6. **Configure backup** for your EBS volume

## API Reference

### POST /query
Search the semantic index
- **Body**: `{"query": "search text", "top_k": 10}`
- **Response**: Search results with timings

### GET /status
Get orchestrator status
- **Response**: Instance counts and details

### POST /scale-up
Launch additional instances
- **Query**: `?count=1`
- **Success Response**: 
  ```json
  {
    "message": "Instance launched successfully", 
    "launched_instances": [
      {
        "instance_id": "i-1234567890abcdef0",
        "status": "launching",
        "created_at": "2024-01-15T10:30:00.123456",
        "public_ip": "pending"
      }
    ], 
    "total_instances": 1
  }
  ```
- **Error Response (already have instance)**: 
  ```json
  {
    "error": "Cannot scale up: already have an instance", 
    "existing_instances": [
      {
        "instance_id": "i-1234567890abcdef0",
        "status": "ready",
        "created_at": "2024-01-15T10:25:00.123456",
        "public_ip": "54.123.45.67",
        "last_used": "2024-01-15T10:35:00.123456"
      }
    ],
    "total_instances": 1
  }
  ```
- **Error Response (launch in progress)**: 
  ```json
  {
    "error": "Cannot scale up: instance launch already in progress", 
    "launching_instances": [
      {
        "instance_id": "i-0987654321fedcba0",
        "status": "launching",
        "created_at": "2024-01-15T10:32:00.123456",
        "public_ip": "pending"
      }
    ],
    "all_instances": [
      {
        "instance_id": "i-0987654321fedcba0",
        "status": "launching",
        "created_at": "2024-01-15T10:32:00.123456",
        "public_ip": "pending",
        "last_used": null
      }
    ],
    "total_instances": 1
  }
  ```

### POST /scale-down
Terminate idle instances
- **Query**: `?count=1`
- **Success Response**: `{"message": "Service shut down", "terminated_instances": ["i-123"], "total_instances": 0}`
- **Error Response**: `{"error": "Cannot terminate: instance is busy", "total_instances": 1}`

### DELETE /instance/{instance_id}
Terminate specific instance
- **Response**: Termination confirmation 