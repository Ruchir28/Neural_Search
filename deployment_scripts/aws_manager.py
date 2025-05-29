"""
AWS resource management for the semantic search orchestrator
"""
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from typing import Dict, Any, Optional, List
from datetime import datetime

from config import OrchestratorConfig

class AWSManager:
    """Manages AWS resources for the orchestrator"""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.ec2_client = self._create_ec2_client()
        
        # AWS resource info (populated during initialization)
        self.ebs_availability_zone: Optional[str] = None
        self.target_subnet_id: Optional[str] = None
        self.security_group_id: Optional[str] = None
        
    def _create_ec2_client(self):
        """Create EC2 client with optimized configuration"""
        config = Config(
            region_name=self.config.aws_region,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        return boto3.client('ec2', config=config)
    
    def initialize_resources(self):
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
            response = self.ec2_client.describe_volumes(VolumeIds=[self.config.ebs_volume_id])
            volume = response['Volumes'][0]
            self.ebs_availability_zone = volume['AvailabilityZone']
            
            if volume['State'] != 'available':
                print(f"WARNING: EBS volume {self.config.ebs_volume_id} is in state '{volume['State']}', not 'available'")
                
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidVolume.NotFound':
                raise ValueError(f"EBS volume {self.config.ebs_volume_id} not found")
            raise
    
    def _ensure_security_group(self):
        """Find existing security group or create a new one"""
        try:
            # Try to find existing security group
            if self.config.ec2_security_group:
                response = self.ec2_client.describe_security_groups(
                    Filters=[
                        {'Name': 'group-name', 'Values': [self.config.ec2_security_group]}
                    ]
                )
                if response['SecurityGroups']:
                    self.security_group_id = response['SecurityGroups'][0]['GroupId']
                    print(f"✓ Found existing security group: {self.config.ec2_security_group}")
                    return
            
            # Create new security group
            print("Creating new security group...")
            vpc_response = self.ec2_client.describe_vpcs(
                Filters=[{'Name': 'is-default', 'Values': ['true']}]
            )
            
            if not vpc_response['Vpcs']:
                raise ValueError("No default VPC found. Please specify EC2_SUBNET_ID.")
            
            default_vpc_id = vpc_response['Vpcs'][0]['VpcId']
            
            sg_name = self.config.ec2_security_group or f"semantic-search-sg-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
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
        if self.config.ec2_subnet_id:
            # Validate provided subnet is in correct AZ
            response = self.ec2_client.describe_subnets(SubnetIds=[self.config.ec2_subnet_id])
            subnet = response['Subnets'][0]
            
            if subnet['AvailabilityZone'] != self.ebs_availability_zone:
                raise ValueError(
                    f"Provided subnet {self.config.ec2_subnet_id} is in AZ {subnet['AvailabilityZone']}, "
                    f"but EBS volume is in AZ {self.ebs_availability_zone}. "
                    f"They must be in the same AZ."
                )
            
            self.target_subnet_id = self.config.ec2_subnet_id
            print(f"✓ Using provided subnet: {self.config.ec2_subnet_id}")
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
        if not self.config.ec2_key_pair:
            print("WARNING: No key pair specified. You won't be able to SSH to instances.")
            return
        
        try:
            self.ec2_client.describe_key_pairs(KeyNames=[self.config.ec2_key_pair])
            print(f"✓ Key pair exists: {self.config.ec2_key_pair}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidKeyPair.NotFound':
                raise ValueError(f"Key pair '{self.config.ec2_key_pair}' not found. Please create it first.")
            raise
    
    def launch_instance(self, user_data: str) -> Dict[str, Any]:
        """Launch a new EC2 instance, trying multiple instance types if needed"""
        
        # Try each instance type in order until one succeeds
        last_error = None
        for i, instance_type in enumerate(self.config.ec2_instance_types):
            try:
                print(f"Attempting to launch {instance_type} (attempt {i+1}/{len(self.config.ec2_instance_types)})...")
                return self._launch_instance_with_type(instance_type, user_data)
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                
                # Check if this is a capacity/availability error
                if error_code in ['InsufficientInstanceCapacity', 'InstanceLimitExceeded', 'Unsupported']:
                    print(f"✗ {instance_type} not available: {error_message}")
                    last_error = e
                    continue
                else:
                    # For other errors (permissions, invalid params, etc.), don't retry
                    print(f"✗ Failed to launch {instance_type}: {error_message}")
                    raise
            except Exception as e:
                print(f"✗ Unexpected error launching {instance_type}: {e}")
                last_error = e
                continue
        
        # If we get here, all instance types failed
        if last_error:
            raise RuntimeError(f"Failed to launch any instance type. Tried: {', '.join(self.config.ec2_instance_types)}. Last error: {last_error}")
        else:
            raise RuntimeError(f"Failed to launch any instance type. Tried: {', '.join(self.config.ec2_instance_types)}")
    
    def _launch_instance_with_type(self, instance_type: str, user_data: str) -> Dict[str, Any]:
        """Launch an EC2 instance with a specific instance type"""
        launch_params = {
            'ImageId': self.config.ec2_ami_id,
            'MinCount': 1,
            'MaxCount': 1,
            'InstanceType': instance_type,
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
                        {'Key': 'EBSVolumeId', 'Value': self.config.ebs_volume_id},
                        {'Key': 'InstanceType', 'Value': instance_type}
                    ]
                }
            ]
        }
        
        # Add key pair if specified
        if self.config.ec2_key_pair:
            launch_params['KeyName'] = self.config.ec2_key_pair
        
        # Handle spot instances
        if self.config.use_spot_instances:
            spot_spec = {
                'SpotPrice': self.config.spot_max_price or '',  # Empty string means current spot price
                'Type': 'one-time'
            }
            launch_params['InstanceMarketOptions'] = {
                'MarketType': 'spot',
                'SpotOptions': spot_spec
            }
        
        response = self.ec2_client.run_instances(**launch_params)
        instance_data = response['Instances'][0]
        
        print(f"✓ Successfully launched {instance_type}: {instance_data['InstanceId']}")
        return instance_data
    
    def wait_for_instance_running(self, instance_id: str):
        """Wait for instance to be in running state"""
        waiter = self.ec2_client.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id], WaiterConfig={'Delay': 15, 'MaxAttempts': 20})
    
    def get_instance_details(self, instance_id: str) -> Dict[str, Any]:
        """Get instance details"""
        instances = self.ec2_client.describe_instances(InstanceIds=[instance_id])
        return instances['Reservations'][0]['Instances'][0]
    
    def attach_ebs_volume(self, instance_id: str):
        """Attach the EBS volume to the instance"""
        self.ec2_client.attach_volume(
            VolumeId=self.config.ebs_volume_id,
            InstanceId=instance_id,
            Device='/dev/sdf'  # Will appear as /dev/xvdf on the instance
        )
        
        # Wait for volume to be attached
        waiter = self.ec2_client.get_waiter('volume_in_use')
        waiter.wait(VolumeIds=[self.config.ebs_volume_id], WaiterConfig={'Delay': 15, 'MaxAttempts': 20})
    
    def detach_ebs_volume(self, instance_id: str):
        """Detach the EBS volume from the instance"""
        self.ec2_client.detach_volume(VolumeId=self.config.ebs_volume_id, InstanceId=instance_id)
    
    def terminate_instance(self, instance_id: str):
        """Terminate an instance"""
        self.ec2_client.terminate_instances(InstanceIds=[instance_id])
    
    def discover_existing_instances(self) -> List[Dict[str, Any]]:
        """Discover existing instances using tags"""
        response = self.ec2_client.describe_instances(
            Filters=[
                {'Name': 'tag:Service', 'Values': ['semantic-search']},
                {'Name': 'tag:ManagedBy', 'Values': ['orchestrator']},
                {'Name': 'tag:EBSVolumeId', 'Values': [self.config.ebs_volume_id]},
                {'Name': 'instance-state-name', 'Values': ['running', 'pending']}
            ]
        )
        
        instances = []
        for reservation in response['Reservations']:
            instances.extend(reservation['Instances'])
        
        return instances 