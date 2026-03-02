from enum import Enum


class OrderType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

    def __str__(self):
        return self.value

    @staticmethod
    def order_type_map():
        return {ote.value: ote for ote in OrderType}

    @staticmethod
    def from_string(order_type_value: str):
        """
        Converts an order_type string into a OrderType object.

        Args:
            order_type_value (str): The ID of the order type

        Returns:
            OrderType: The corresponding OrderType object.

        Raises:
            ValueError: If no matching order type is found.
        """
        otm = OrderType.order_type_map()
        if order_type_value in otm:
            return otm[order_type_value]
        else:
            raise ValueError(f"No matching order type found for value '{order_type_value}'. Please check the input "
                             f"and try again.")

    @staticmethod
    def opposite_order_type(order_type):
        if order_type == OrderType.LONG:
            return OrderType.SHORT
        elif order_type == OrderType.SHORT:
            return OrderType.LONG
        else:
            return None

    def __json__(self):
        # Provide a dictionary representation for JSON serialization
        return self.__str__()


class StopCondition(Enum):
    GTE = "GTE"  # trigger when price >= stop_price
    LTE = "LTE"  # trigger when price <= stop_price

    def __str__(self):
        return self.value

    @staticmethod
    def stop_condition_map():
        return {sc.value: sc for sc in StopCondition}

    @staticmethod
    def from_string(stop_condition_value: str):
        scm = StopCondition.stop_condition_map()
        upper_value = stop_condition_value.upper() if stop_condition_value else None
        if upper_value in scm:
            return scm[upper_value]
        else:
            raise ValueError(f"No matching stop condition found for value '{stop_condition_value}'. "
                             f"Valid values are: {', '.join(scm.keys())}")

    def __json__(self):
        return self.__str__()
