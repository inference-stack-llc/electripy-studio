# AI Skills

The skills package provides a **reusable skills packaging system** for AI
applications.  A skill bundles instructions, metadata, assets, execution
hints, validation rules, and supporting files into a versioned, portable
skill unit.

## Why it exists

Agent and prompt pipelines accumulate structured knowledge — style
guides, review checklists, code-generation templates — that needs to be
packaged, validated, and resolved at runtime.  This component gives
those artefacts a well-defined shape.

## Core concepts

| Symbol | Role |
|--------|------|
| `SkillManifest` | Declarative metadata read from `manifest.json`. |
| `SkillPackage` | Loaded skill — manifest + instructions + root path. |
| `SkillVersion` | Semantic version with comparison operators. |
| `SkillExecutionContext` | Runtime variables supplied during resolution. |
| `SkillResolverResult` | Resolved instructions and rendered templates. |
| `SkillValidationResult` | Diagnostics from validating a skill. |

## Skill directory layout

```
my-skill/
  manifest.json            # required
  instructions/
    main.md                # entry instruction
    style.md               # extra fragment
  templates/
    report.md              # renderable template
  assets/
    config.json            # data / config files
```

## Manifest example

```json
{
  "name": "code-review",
  "version": "1.2.0",
  "description": "Automated code review skill",
  "entry_instruction": "instructions/main.md",
  "variables": ["code", "reviewer", "findings"],
  "assets": [
    { "name": "style-guide", "kind": "instruction", "path": "instructions/style.md" },
    { "name": "report-template", "kind": "template", "path": "templates/report.md" },
    { "name": "config", "kind": "config", "path": "assets/config.json" }
  ],
  "dependencies": [
    { "name": "base-reviewer", "version": ">=1.0.0" }
  ],
  "metadata": {
    "author": "ElectriPy Team",
    "license": "MIT",
    "capabilities": ["code_generation"],
    "tags": ["review", "quality"]
  }
}
```

## Loading a skill

```python
from electripy.ai.skills import load_skill

pkg = load_skill("./skills/code-review")
print(pkg.manifest.name)           # "code-review"
print(pkg.manifest.version)        # 1.2.0
print(pkg.instructions.entry_instruction[:60])
```

## Validating before use

```python
from electripy.ai.skills import validate_skill

result = validate_skill("./skills/code-review")
if result.valid:
    print("All checks passed")
else:
    for diag in result.errors:
        print(f"[{diag.code}] {diag.message}")
```

To raise on error-level diagnostics:

```python
from electripy.ai.skills import validate_skill, SkillValidationError

try:
    validate_skill("./skills/code-review", fail_on_error=True)
except SkillValidationError as exc:
    print(exc)
```

## Resolving with variables

Resolution replaces `{{variable}}` placeholders in instructions and
templates with runtime values.

```python
from electripy.ai.skills import load_skill, resolve_skill
from electripy.ai.skills import SkillExecutionContext

pkg = load_skill("./skills/code-review")
ctx = SkillExecutionContext(
    variables=(
        ("code", "def add(a, b): return a + b"),
        ("reviewer", "Alice"),
        ("findings", "- Consider type hints"),
    ),
)
result = resolve_skill(pkg, ctx)
print(result.instructions.entry_instruction)
print(result.rendered_templates)   # tuple of (name, content) pairs
print(result.unresolved_variables) # empty when all vars supplied
```

## Discovering skills

```python
from electripy.ai.skills import list_skills

manifests = list_skills("./skills")
for m in manifests:
    print(f"{m.name} v{m.version}")
```

## Using the service directly

For full lifecycle control, inject your own adapters into `SkillService`:

```python
from electripy.ai.skills import SkillService

svc = SkillService()
pkg = svc.load("./skills/code-review")
result = svc.validate("./skills/code-review", fail_on_error=True)
resolved = svc.resolve(pkg, ctx)
registered = svc.get_registered("code-review")
```

## Observability

Implement `SkillObserverPort` to hook into load, validate, and resolve
events:

```python
from electripy.ai.skills import SkillObserverPort

class MyObserver:
    def on_load(self, package):
        print(f"Loaded {package.manifest.name}")

    def on_validate(self, manifest, result):
        print(f"Validated {manifest.name}: valid={result.valid}")

    def on_resolve(self, package, result):
        print(f"Resolved {package.manifest.name}")

svc = SkillService(observer=MyObserver())
```

## Validation diagnostics

The validator emits diagnostics at three severity levels:

- **error** — the skill cannot be used (e.g. missing entry instruction)
- **warning** — the skill works but is incomplete (e.g. missing description)
- **info** — informational notes

Each diagnostic carries a `code`, `message`, and `severity`.
