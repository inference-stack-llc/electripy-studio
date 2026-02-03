# Core Module

The core module provides essential utilities for configuration, logging, error handling, and type definitions.

## Configuration

The `Config` class manages application configuration with environment variable support.

### Usage

```python
from electripy import Config

# Create with defaults
config = Config()

# Load from environment
config = Config.from_env()

# Access configuration
print(config.log_level)  # "INFO"
print(config.config_dir)  # Path to config directory
```

### Environment Variables

- `ELECTRIPY_LOG_LEVEL`: Logging level (default: INFO)
- `ELECTRIPY_LOG_FORMAT`: Log format (default: json)
- `ELECTRIPY_CONFIG_DIR`: Configuration directory (default: ~/.electripy)

## Logging

Simple logging setup with structured output support.

### Usage

```python
from electripy import get_logger, setup_logging

# Setup logging
setup_logging(level="DEBUG", format_type="json")

# Get logger
logger = get_logger(__name__)
logger.info("Application started")
logger.error("An error occurred", exc_info=True)
```

## Error Handling

Custom exception hierarchy for ElectriPy.

### Exception Types

- `ElectriPyError`: Base exception for all ElectriPy errors
- `ConfigError`: Configuration-related errors
- `ValidationError`: Validation failures
- `RetryError`: Retry exhaustion
- `RateLimitError`: Rate limit exceeded

### Usage

```python
from electripy.core.errors import ValidationError

def validate_data(data):
    if not data:
        raise ValidationError("Data cannot be empty")
```

## Type Utilities

Common type definitions for JSON data.

```python
from electripy.core.typing import JSONDict, JSONValue

def process_json(data: JSONDict) -> JSONValue:
    return data.get("result")
```
