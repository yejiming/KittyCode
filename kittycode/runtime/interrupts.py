"""Cancellation helpers for cooperative CLI interrupts."""


class CancellationRequested(Exception):
    """Raised when the current agent run has been cancelled."""
