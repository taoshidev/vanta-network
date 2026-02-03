class BracketOrderException(Exception):
    """Exception raised when bracket order (SLTP) creation or validation fails."""
    def __init__(self, message):
        super().__init__(message)
