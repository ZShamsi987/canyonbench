"""Domain-specific exceptions with actionable messages."""


class CanyonBenchError(Exception):
    """Base class for expected CanyonBench failures."""


class DataValidationError(CanyonBenchError):
    """Input data violates a documented benchmark contract."""


class ExternalToolError(CanyonBenchError):
    """An external executable is missing or failed."""


class BudgetExceededError(CanyonBenchError):
    """A run would exceed its configured request or cost budget."""
