class EngineException(Exception):
    pass


class ConfigurationError(EngineException):
    pass


class _Unpausable(ConfigurationError):
    pass


_Unpausable.__name__ = _Unpausable.__qualname__ = ConfigurationError.__name__


class _AskPaused(EngineException):
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


class WorkerError(EngineException):
    pass


def error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
