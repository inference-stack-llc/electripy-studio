"""Ports for the Structured Output Engine."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OutputModelPort(Protocol):
    """Contract for an output model that can validate parsed JSON.

    Any class implementing this protocol must be constructable from a dict
    and expose a ``model_json_schema`` class method (compatible with Pydantic).
    """

    @classmethod
    def model_validate(cls, obj: Any) -> Any:
        """Validate and construct an instance from a dict.

        Raises:
          Exception: Implementation-specific validation errors.
        """
        ...

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Return the JSON schema for this model.

        Returns:
          A JSON-serialisable dict representing the schema.
        """
        ...


@runtime_checkable
class SchemaRendererPort(Protocol):
    """Renders a model class into a prompt-friendly JSON-schema string."""

    def render(self, model: type) -> str:
        """Return a prompt-injectable string describing the schema.

        Args:
          model: The output model class to render.

        Returns:
          A human-readable schema string suitable for LLM prompts.

        Raises:
          SchemaGenerationError: If the schema cannot be generated.
        """
        ...
