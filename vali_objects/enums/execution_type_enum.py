from enum import Enum


class ExecutionType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    LIMIT_CANCEL = "LIMIT_CANCEL"
    LIMIT_EDIT = "LIMIT_EDIT"
    BRACKET = "BRACKET"
    FLAT_ALL = "FLAT_ALL"

    def __str__(self):
        return self.value

    @staticmethod
    def execution_type_map():
        return {e.value: e for e in ExecutionType}

    @staticmethod
    def from_string(execution_type_value: str):
        # Handle None or missing execution_type - default to MARKET
        if execution_type_value is None:
            return ExecutionType.MARKET

        e_map = ExecutionType.execution_type_map()
        execution_type_upper = execution_type_value.upper()
        if execution_type_upper in e_map:
            return e_map[execution_type_upper]  # Use uppercase version for lookup
        else:
            raise ValueError(f"No matching execution type found for value '{execution_type_value}'. "
                             f"Valid values are: {', '.join(e_map.keys())}")

    def __json__(self):
        # Provide a dictionary representation for JSON serialization
        return self.__str__()


