class EngineException(Exception):
    pass


class ConfigurationError(EngineException):
    pass


class NotFoundError(EngineException):
    pass


class InputError(EngineException):
    pass


class ModelError(EngineException):
    pass


class RunawayError(EngineException):
    pass


class DelegationLoopError(EngineException):
    pass


class DelegationDepthError(EngineException):
    pass


class StorageError(EngineException):
    pass
