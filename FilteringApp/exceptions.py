"""
Custom exception hierarchy for the Data Filtering Platform.

All domain-specific errors inherit from DFPError so they can be caught
uniformly in the API error handlers.
"""


class DFPError(Exception):
    """Base exception for the Data Filtering Platform."""

    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class ValidationError(DFPError):
    """Raised when uploaded data fails validation checks."""
    pass


class CatalogError(DFPError):
    """Raised when a requested dataset is not found in the catalog."""
    pass


class FilterError(DFPError):
    """Raised when a filtering operation fails."""
    pass


class DeliveryError(DFPError):
    """Raised when packaging or delivery fails."""
    pass


class FileNotFoundInRunError(DFPError):
    """Raised when a file matching the catalog pattern is not found in the prod run."""
    pass
