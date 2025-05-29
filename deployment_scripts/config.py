"""
Configuration management for the semantic search orchestrator
"""
import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class OrchestratorConfig:
    """Configuration settings for the orchestrator"""
    
    # AWS Configuration
    aws_region: str
    ebs_volume_id: str
    ec2_instance_types: List[str]
    ec2_ami_id: str
    ec2_key_pair: Optional[str]
    ec2_security_group: str
    ec2_subnet_id: Optional[str]
    use_spot_instances: bool
    spot_max_price: Optional[str]
    
    # Semantic Search Configuration
    cohere_api_key: str
    load_embeddings_to_ram: bool
    
    # Instance Management Settings
    instance_idle_timeout: int
    instance_max_runtime: int
    health_check_interval: int
    
    @classmethod
    def from_env(cls) -> 'OrchestratorConfig':
        """Load configuration from environment variables"""
        
        # Required variables
        ebs_volume_id = os.getenv("EBS_VOLUME_ID")
        if not ebs_volume_id:
            raise ValueError("EBS_VOLUME_ID environment variable is required")
        
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")
        
        # Parse instance types - support both single type and comma-separated list
        instance_types_str = os.getenv("EC2_INSTANCE_TYPES", os.getenv("EC2_INSTANCE_TYPE", "g4dn.4xlarge"))
        ec2_instance_types = [t.strip() for t in instance_types_str.split(",") if t.strip()]
        
        if not ec2_instance_types:
            raise ValueError("At least one EC2 instance type must be specified")
        
        return cls(
            # AWS Configuration
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            ebs_volume_id=ebs_volume_id,
            ec2_instance_types=ec2_instance_types,
            ec2_ami_id=os.getenv("EC2_AMI_ID", "ami-074d9c327b5296aaa"),
            ec2_key_pair=os.getenv("EC2_KEY_PAIR"),
            ec2_security_group=os.getenv("EC2_SECURITY_GROUP", "semantic-search-sg"),
            ec2_subnet_id=os.getenv("EC2_SUBNET_ID"),
            use_spot_instances=os.getenv("USE_SPOT_INSTANCES", "false").lower() in ("true", "1", "yes"),
            spot_max_price=os.getenv("SPOT_MAX_PRICE"),
            
            # Semantic Search Configuration
            cohere_api_key=cohere_api_key,
            load_embeddings_to_ram=os.getenv("LOAD_EMBEDDINGS_TO_RAM", "false").lower() in ("true", "1", "yes"),
            
            # Instance Management Settings
            instance_idle_timeout=int(os.getenv("INSTANCE_IDLE_TIMEOUT", "1800")),
            instance_max_runtime=int(os.getenv("INSTANCE_MAX_RUNTIME", "1800")),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))
        ) 