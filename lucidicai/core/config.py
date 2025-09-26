"""Centralized configuration management for Lucidic SDK.

This module provides a single source of truth for all SDK configuration,
including environment variables, defaults, and runtime settings.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class Environment(Enum):
    """SDK environment modes"""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    DEBUG = "debug"


@dataclass
class NetworkConfig:
    """Network and connection settings"""
    base_url: str = "https://api.lucidic.ai/api"
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.5
    connection_pool_size: int = 20
    connection_pool_maxsize: int = 100
    
    @classmethod
    def from_env(cls) -> 'NetworkConfig':
        """Load network configuration from environment variables"""
        debug = os.getenv("LUCIDIC_DEBUG", "False").lower() == "true"
        return cls(
            base_url="http://localhost:8000/api" if debug else "https://api.lucidic.ai/api",
            timeout=int(os.getenv("LUCIDIC_TIMEOUT", "30")),
            max_retries=int(os.getenv("LUCIDIC_MAX_RETRIES", "3")),
            backoff_factor=float(os.getenv("LUCIDIC_BACKOFF_FACTOR", "0.5")),
            connection_pool_size=int(os.getenv("LUCIDIC_CONNECTION_POOL_SIZE", "20")),
            connection_pool_maxsize=int(os.getenv("LUCIDIC_CONNECTION_POOL_MAXSIZE", "100"))
        )


@dataclass
class EventQueueConfig:
    """Event queue processing settings"""
    max_queue_size: int = 100000
    flush_interval_ms: int = 100
    flush_at_count: int = 100
    blob_threshold: int = 65536
    daemon_mode: bool = True
    max_parallel_workers: int = 10
    retry_failed: bool = True
    
    @classmethod
    def from_env(cls) -> 'EventQueueConfig':
        """Load event queue configuration from environment variables"""
        return cls(
            max_queue_size=int(os.getenv("LUCIDIC_MAX_QUEUE_SIZE", "100000")),
            flush_interval_ms=int(os.getenv("LUCIDIC_FLUSH_INTERVAL", "1000")),
            flush_at_count=int(os.getenv("LUCIDIC_FLUSH_AT", "50")),
            blob_threshold=int(os.getenv("LUCIDIC_BLOB_THRESHOLD", "65536")),
            daemon_mode=os.getenv("LUCIDIC_DAEMON_QUEUE", "true").lower() == "true",
            max_parallel_workers=int(os.getenv("LUCIDIC_MAX_PARALLEL", "10")),
            retry_failed=os.getenv("LUCIDIC_RETRY_FAILED", "true").lower() == "true"
        )


@dataclass
class ErrorHandlingConfig:
    """Error handling and suppression settings"""
    suppress_errors: bool = True
    cleanup_on_error: bool = True
    log_suppressed: bool = True
    capture_uncaught: bool = True
    
    @classmethod
    def from_env(cls) -> 'ErrorHandlingConfig':
        """Load error handling configuration from environment variables"""
        return cls(
            suppress_errors=os.getenv("LUCIDIC_SUPPRESS_ERRORS", "true").lower() == "true",
            cleanup_on_error=os.getenv("LUCIDIC_CLEANUP_ON_ERROR", "true").lower() == "true",
            log_suppressed=os.getenv("LUCIDIC_LOG_SUPPRESSED", "true").lower() == "true",
            capture_uncaught=os.getenv("LUCIDIC_CAPTURE_UNCAUGHT", "true").lower() == "true"
        )


@dataclass
class TelemetryConfig:
    """Telemetry and instrumentation settings"""
    providers: List[str] = field(default_factory=list)
    verbose: bool = False
    
    @classmethod
    def from_env(cls) -> 'TelemetryConfig':
        """Load telemetry configuration from environment variables"""
        return cls(
            providers=[],  # Set during initialization
            verbose=os.getenv("LUCIDIC_VERBOSE", "False").lower() == "true"
        )


@dataclass
class SDKConfig:
    """Main SDK configuration container"""
    # Required settings
    api_key: Optional[str] = None
    agent_id: Optional[str] = None
    
    # Feature flags
    auto_end: bool = True
    production_monitoring: bool = False
    
    # Sub-configurations
    network: NetworkConfig = field(default_factory=NetworkConfig)
    event_queue: EventQueueConfig = field(default_factory=EventQueueConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    
    # Runtime settings
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    
    @classmethod
    def from_env(cls, **overrides) -> 'SDKConfig':
        """Create configuration from environment variables with optional overrides"""
        from dotenv import load_dotenv
        load_dotenv()
        
        debug = os.getenv("LUCIDIC_DEBUG", "False").lower() == "true"
        
        config = cls(
            api_key=os.getenv("LUCIDIC_API_KEY"),
            agent_id=os.getenv("LUCIDIC_AGENT_ID"),
            auto_end=os.getenv("LUCIDIC_AUTO_END", "true").lower() == "true",
            production_monitoring=False,
            network=NetworkConfig.from_env(),
            event_queue=EventQueueConfig.from_env(),
            error_handling=ErrorHandlingConfig.from_env(),
            telemetry=TelemetryConfig.from_env(),
            environment=Environment.DEBUG if debug else Environment.PRODUCTION,
            debug=debug
        )
        
        # Apply any overrides
        config.update(**overrides)
        return config
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            # Only update if value is not None (to preserve env defaults)
            if value is not None:
                if hasattr(self, key):
                    setattr(self, key, value)
                elif key == "providers" and hasattr(self.telemetry, "providers"):
                    self.telemetry.providers = value
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.api_key:
            errors.append("API key is required (LUCIDIC_API_KEY)")
        
        if not self.agent_id:
            errors.append("Agent ID is required (LUCIDIC_AGENT_ID)")
        
        if self.event_queue.max_parallel_workers < 1:
            errors.append("Max parallel workers must be at least 1")
        
        if self.event_queue.flush_interval_ms < 10:
            errors.append("Flush interval must be at least 10ms")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "api_key": self.api_key[:8] + "..." if self.api_key else None,
            "agent_id": self.agent_id,
            "environment": self.environment.value,
            "debug": self.debug,
            "auto_end": self.auto_end,
            "network": {
                "base_url": self.network.base_url,
                "timeout": self.network.timeout,
                "max_retries": self.network.max_retries,
                "connection_pool_size": self.network.connection_pool_size
            },
            "event_queue": {
                "max_workers": self.event_queue.max_parallel_workers,
                "flush_interval_ms": self.event_queue.flush_interval_ms,
                "flush_at_count": self.event_queue.flush_at_count
            },
            "error_handling": {
                "suppress": self.error_handling.suppress_errors,
                "cleanup": self.error_handling.cleanup_on_error
            }
        }


# Global configuration instance
_config: Optional[SDKConfig] = None


def get_config() -> SDKConfig:
    """Get the current SDK configuration"""
    global _config
    if _config is None:
        _config = SDKConfig.from_env()
    return _config


def set_config(config: SDKConfig):
    """Set the SDK configuration"""
    global _config
    _config = config


def reset_config():
    """Reset configuration to defaults"""
    global _config
    _config = None