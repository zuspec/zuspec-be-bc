"""Lowering error type."""


class LoweringError(Exception):
    """Raised when Scenario IR cannot be lowered to ZBC.

    Carries an optional ``loc`` (source location) so out-of-scope constructs
    (M1 §1.1) produce a clear, source-anchored diagnostic rather than a crash or
    silent miscompile.
    """

    def __init__(self, message, loc=None):
        super().__init__(message)
        self.loc = loc
