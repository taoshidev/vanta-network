from enum import Enum


class ExecutionType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    LIMIT_CANCEL = "LIMIT_CANCEL"
    LIMIT_EDIT = "LIMIT_EDIT"
    BRACKET = "BRACKET"
    STOP_LIMIT = "STOP_LIMIT"
    FLAT_ALL = "FLAT_ALL"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(execution_type_value: str):
        # Handle None or missing execution_type - default to MARKET
        if execution_type_value is None:
            return ExecutionType.MARKET

        try:
            return ExecutionType(execution_type_value.upper())
        except ValueError:
            raise ValueError(f"No matching execution type found for value '{execution_type_value}'. "
                             f"Valid values are: {', '.join([e.value for e in ExecutionType])}")

    def __json__(self):
        # Provide a dictionary representation for JSON serialization
        return self.__str__()


