"""
API Key Authentication and Management System
"""

import secrets
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"  # Should be in env vars
ALGORITHM = "HS256"
API_KEY_EXPIRY_DAYS = 365

# Security scheme
security = HTTPBearer()


class APIKey(BaseModel):
    """API Key model"""
    key_id: str
    key_hash: str
    name: str
    created_at: str
    expires_at: Optional[str]
    is_active: bool
    usage_count: int
    rate_limit: int  # requests per minute
    allowed_models: List[str]
    metadata: Dict


class APIKeyManager:
    """Manages API keys for the service"""
    
    def __init__(self, storage_path: str = "data/api_keys.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.api_keys: Dict[str, APIKey] = {}
        self.load_keys()
    
    def load_keys(self):
        """Load API keys from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.api_keys = {
                        k: APIKey(**v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self.api_keys)} API keys")
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
                self.api_keys = {}
        else:
            # Create default API key for testing
            self.create_default_key()
    
    def save_keys(self):
        """Save API keys to storage"""
        try:
            data = {
                k: v.dict() for k, v in self.api_keys.items()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("API keys saved successfully")
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
    
    def create_default_key(self):
        """Create a default API key for testing"""
        key = self.generate_api_key(
            name="Default Test Key",
            allowed_models=["dqn", "ppo", "hybrid", "hybrid_gat", "hybrid_attention"],
            rate_limit=100
        )
        logger.info(f"Created default API key: {key}")
        print(f"\n{'='*60}")
        print(f"🔑 DEFAULT API KEY CREATED")
        print(f"{'='*60}")
        print(f"API Key: {key}")
        print(f"Use this key in the Authorization header:")
        print(f"Authorization: Bearer {key}")
        print(f"{'='*60}\n")
        return key
    
    def generate_api_key(
        self,
        name: str,
        allowed_models: List[str],
        rate_limit: int = 60,
        expires_in_days: Optional[int] = API_KEY_EXPIRY_DAYS
    ) -> str:
        """Generate a new API key"""
        # Generate random key
        key = f"sk-iot-{secrets.token_urlsafe(32)}"
        key_id = secrets.token_hex(8)
        
        # Hash the key for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Calculate expiry
        created_at = datetime.utcnow()
        expires_at = None
        if expires_in_days:
            expires_at = created_at + timedelta(days=expires_in_days)
        
        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=created_at.isoformat(),
            expires_at=expires_at.isoformat() if expires_at else None,
            is_active=True,
            usage_count=0,
            rate_limit=rate_limit,
            allowed_models=allowed_models,
            metadata={}
        )
        
        # Store the key
        self.api_keys[key_hash] = api_key
        self.save_keys()
        
        return key
    
    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key"""
        # Hash the provided key
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Check if key exists
        if key_hash not in self.api_keys:
            return None
        
        api_key = self.api_keys[key_hash]
        
        # Check if key is active
        if not api_key.is_active:
            return None
        
        # Check expiry
        if api_key.expires_at:
            expires = datetime.fromisoformat(api_key.expires_at)
            if datetime.utcnow() > expires:
                return None
        
        # Update usage count
        api_key.usage_count += 1
        self.save_keys()
        
        return api_key
    
    def revoke_key(self, key: str) -> bool:
        """Revoke an API key"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        if key_hash in self.api_keys:
            self.api_keys[key_hash].is_active = False
            self.save_keys()
            return True
        return False
    
    def list_keys(self) -> List[Dict]:
        """List all API keys (without exposing the actual keys)"""
        return [
            {
                "key_id": key.key_id,
                "name": key.name,
                "created_at": key.created_at,
                "expires_at": key.expires_at,
                "is_active": key.is_active,
                "usage_count": key.usage_count,
                "rate_limit": key.rate_limit,
                "allowed_models": key.allowed_models
            }
            for key in self.api_keys.values()
        ]


# Global instance
api_key_manager = APIKeyManager()


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests: Dict[str, List[datetime]] = {}
    
    def check_rate_limit(self, key_hash: str, limit: int) -> bool:
        """Check if request is within rate limit"""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        if key_hash not in self.requests:
            self.requests[key_hash] = []
        
        # Remove old requests
        self.requests[key_hash] = [
            req_time for req_time in self.requests[key_hash]
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[key_hash]) >= limit:
            return False
        
        # Add current request
        self.requests[key_hash].append(now)
        return True


# Global rate limiter
rate_limiter = RateLimiter()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> APIKey:
    """Verify API key from request"""
    token = credentials.credentials
    
    # Validate the API key
    api_key = api_key_manager.validate_api_key(token)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check rate limit
    key_hash = hashlib.sha256(token.encode()).hexdigest()
    if not rate_limiter.check_rate_limit(key_hash, api_key.rate_limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {api_key.rate_limit} requests per minute.",
        )
    
    return api_key


def check_model_access(api_key: APIKey, model_type: str) -> bool:
    """Check if API key has access to a specific model"""
    return model_type in api_key.allowed_models
