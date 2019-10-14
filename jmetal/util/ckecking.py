class NoneParameterException(Exception):
    def __init__(self, message = ""):
        self.error_message = message


class InvalidConditionException(Exception):
    def __init__(self, message):
        self.error_message = message


class Check:
    @staticmethod
    def is_not_null(o, message = ""):
        if o is None:
            raise NoneParameterException(message)

    @staticmethod
    def that(expression, message = ""):
        if not expression:
            raise InvalidConditionException(message)
