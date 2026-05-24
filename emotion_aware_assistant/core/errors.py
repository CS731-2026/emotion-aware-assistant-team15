"""Project-specific exceptions."""


class AssistantError(Exception):
    """Base class for assistant errors."""


class MissingOptionalDependency(AssistantError):
    """Raised when an optional dependency is needed for a requested feature."""


class DocumentLoadError(AssistantError):
    """Raised when a paper document cannot be loaded."""


class ModelLoadError(AssistantError):
    """Raised when an optional model adapter cannot load its files."""
