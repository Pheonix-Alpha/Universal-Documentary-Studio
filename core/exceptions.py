"""Custom exception hierarchy for Universal Documentary Studio."""
from __future__ import annotations


class UDSError(Exception):
    """Base class for all UDS errors."""


class ResourceError(UDSError):
    """Raised when hardware/resource requirements cannot be satisfied."""


class ModelUnavailableError(UDSError):
    """Raised when no compatible model can be found in the registry."""


class CheckpointError(UDSError):
    """Raised when checkpoint read/write fails or is corrupt."""


class StateTransitionError(UDSError):
    """Raised when an illegal project state transition is attempted."""


class ValidationError(UDSError):
    """Raised when a data model fails validation (facts, sources, assets...)."""


class LicenseError(UDSError):
    """Raised when an asset/model license cannot be verified or is disallowed."""


class SchedulerError(UDSError):
    """Raised for job scheduling failures."""


class RenderError(UDSError):
    """Raised when rendering (FFmpeg or otherwise) fails."""


class QAFailure(UDSError):
    """Raised when QA gates a project from proceeding."""
