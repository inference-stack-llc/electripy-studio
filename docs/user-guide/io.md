# I/O Module

Utilities for reading and writing JSONL (JSON Lines) files.

## JSONL Format

JSONL is a convenient format for storing structured data where each line is a valid JSON object. It's perfect for:

- Logs and event streams
- Large datasets that don't fit in memory
- Append-only data structures
- Line-by-line processing

## Writing JSONL

### Write Entire File

```python
from electripy.io import write_jsonl

data = [
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25},
    {"id": 3, "name": "Charlie", "age": 35},
]

write_jsonl("users.jsonl", data)
```

### Append Single Record

```python
from electripy.io import append_jsonl

# Append log entries
append_jsonl("events.jsonl", {
    "timestamp": "2024-01-01T10:00:00",
    "event": "user_login",
    "user_id": 123,
})
```

## Reading JSONL

### Iterate Over Records

```python
from electripy.io import read_jsonl

for record in read_jsonl("users.jsonl"):
    print(f"User {record['name']} is {record['age']} years old")
```

### Load All Records

```python
from electripy.io import read_jsonl

# Read all records into a list
records = list(read_jsonl("users.jsonl"))
print(f"Total records: {len(records)}")
```

### Filter While Reading

```python
from electripy.io import read_jsonl

# Process only matching records
for record in read_jsonl("events.jsonl"):
    if record.get("event") == "error":
        handle_error(record)
```

## Features

- **Memory Efficient**: Reads line by line, suitable for large files
- **Unicode Support**: Full UTF-8 support for international characters
- **Automatic Directory Creation**: Creates parent directories if needed
- **Empty Line Handling**: Gracefully skips empty lines

## Example: ETL Pipeline

```python
from electripy.io import read_jsonl, write_jsonl

def transform_users(input_file, output_file):
    """Transform user data from input to output."""
    transformed = []
    
    for user in read_jsonl(input_file):
        # Transform the record
        transformed_user = {
            "user_id": user["id"],
            "full_name": user["name"],
            "age_group": "adult" if user["age"] >= 18 else "minor",
        }
        transformed.append(transformed_user)
    
    write_jsonl(output_file, transformed)

transform_users("input.jsonl", "output.jsonl")
```
