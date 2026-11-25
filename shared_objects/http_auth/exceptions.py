"""Auth errors for both the btauth/1 path and the legacy v10 compat path, so callers can
catch one AuthError type regardless of which scheme rejected the request."""


class AuthError(Exception):
    """Base class for any signed-request verification failure, new scheme or legacy."""


class MalformedAuth(AuthError):
    pass


class WrongReceiver(AuthError):
    pass


class StaleRequest(AuthError):
    pass


class ReplayedRequest(AuthError):
    pass


class BadSignature(AuthError):
    pass


class NoRecognizedScheme(AuthError):
    """Neither btauth/1 nor (if enabled) legacy v10 headers were present."""
