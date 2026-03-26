# Sensitive Data Scanner

The **Sensitive Data Scanner** detects PII and secrets in text before
it reaches an LLM, enabling pre-flight redaction, blocking, or logging.

## When to use it

- You accept user input that will be sent to an LLM and need to check
  for PII (emails, SSNs, phone numbers) or leaked secrets (API keys).
- You want a fast, deterministic, regex-based scanner with zero network
  calls or ML dependencies.
- You need to extend detection with your own patterns.

## Core concepts

| Symbol | Role |
|--------|------|
| `scan_text(text)` | Module-level convenience — uses all built-in patterns. |
| `SensitiveDataScanner` | Configurable scanner — add/remove patterns. |
| `SensitiveMatch` | Frozen result: `category`, `matched_text`, `start`, `end`. |
| `SensitivePattern` | Frozen pattern: `category`, `pattern`, `description`. |

## Built-in patterns

| Category | Example match |
|----------|---------------|
| `email` | `user@example.com` |
| `phone_us` | `555-123-4567` |
| `ssn` | `123-45-6789` |
| `credit_card` | `4111 1111 1111 1111` |
| `api_key_generic` | `sk-...` / `key-...` (32+ chars) |
| `api_key_openai` | `sk-proj-...` |
| `api_key_anthropic` | `sk-ant-...` |
| `aws_access_key` | `AKIA...` (20 uppercase chars) |
| `ipv4` | `192.168.1.1` |

## Quick start

```python
from electripy.ai.sensitive_data_scanner import scan_text

matches = scan_text("Email alice@corp.io, key sk-proj-abc123xyz")
for m in matches:
    print(m.category, m.matched_text)
# email  alice@corp.io
# api_key_openai  sk-proj-abc123xyz
```

## Custom patterns

```python
from electripy.ai.sensitive_data_scanner import SensitiveDataScanner, SensitivePattern
import re

scanner = SensitiveDataScanner()
scanner.add_pattern(SensitivePattern(
    category="internal_id",
    pattern=re.compile(r"PROJ-\d{6}"),
    description="Internal project ID",
))

matches = scanner.scan("See PROJ-123456 for details.")
print(matches[0].category)  # "internal_id"
```

## Boolean check

```python
from electripy.ai.sensitive_data_scanner import SensitiveDataScanner

scanner = SensitiveDataScanner()
if scanner.has_sensitive_data("Send to alice@corp.io"):
    print("Blocked — contains PII")
```

## Scanner without built-in patterns

If you only want your custom patterns and none of the defaults:

```python
scanner = SensitiveDataScanner(include_builtins=False)
scanner.add_pattern(my_custom_pattern)
```
