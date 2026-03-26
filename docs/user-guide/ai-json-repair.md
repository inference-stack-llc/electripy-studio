# JSON Repair

`json_repair()` fixes the most common JSON breakage patterns produced
by LLMs and returns a parsed `dict` in one call.

## When to use it

- An LLM returns JSON wrapped in markdown fences, with trailing commas,
  single-quoted keys, or gets cut off mid-object by token limits.
- You want a single function that handles all of these cases without
  chaining regex hacks yourself.

## Repair strategies (applied in order)

1. **Strip markdown fences** — `` ```json … ``` ``
2. **Extract the outermost `{…}` block** from surrounding prose.
3. **Remove trailing commas** before `}` or `]`.
4. **Replace single-quoted strings** with double quotes.
5. **Quote bare (unquoted) keys** — JavaScript-style `name:` → `"name":`.
6. **Fix mismatched brackets** — inserts missing `]` or `}` when
   a closer matches a deeper bracket (e.g. `{"items": [1,2,3}` →
   `{"items": [1,2,3]}`).
7. **Close truncated JSON** — appends missing braces/brackets for
   objects that were cut off by token limits.

## Basic example

```python
from electripy.ai.json_repair import json_repair

text = '''Here is the result:
```json
{"name": "Alice", "age": 30,}
```'''

data = json_repair(text)
print(data)  # {"name": "Alice", "age": 30}
```

## Raw string variant

If you need the repaired JSON as a string (e.g. for logging or storage)
rather than a parsed dict:

```python
from electripy.ai.json_repair import json_repair_raw

raw = json_repair_raw(text)
print(type(raw))  # <class 'str'>
```

## Truncated JSON

Token limits frequently cut off LLM output mid-object.  `json_repair`
handles this automatically:

```python
data = json_repair('{"users": [{"name": "Alice"')
# {"users": [{"name": "Alice"}]}
```

## Error handling

If no JSON object can be recovered at all, `ValueError` is raised.
The error message includes the first 200 characters of the input for
debugging.
