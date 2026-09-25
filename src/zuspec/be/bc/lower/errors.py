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


class PssSemanticError(LoweringError):
    """The PSS itself is wrong (a rule the LRM says "shall" be an error).

    Distinct from a plain :class:`LoweringError`, which means *bc* cannot lower
    something legal: a caller reports this one as a compile error, the other as
    unsupported.
    """
