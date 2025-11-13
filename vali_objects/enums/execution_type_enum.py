from enum import Enum


class ExecutionType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    LIMIT_CANCEL = "LIMIT_CANCEL"

    def __str__(self):
        return self.value

    @staticmethod
    def execution_type_map():
        return {e.value: e for e in ExecutionType}

    @staticmethod
    def from_string(execution_type_value: str):
        e_map = ExecutionType.execution_type_map()
        if execution_type_value in e_map:
            return e_map[execution_type_value]
        else:
            raise ValueError(f"No matching order type found for value '{execution_type_value}'. Please check the input "
                             f"and try again.")

    def __json__(self):
        # Provide a dictionary representation for JSON serialization
        return self.__str__()


