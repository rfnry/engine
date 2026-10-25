from rfnry_engine.errors import (
    ConfigurationError,
    DelegationDepthError,
    DelegationLoopError,
    EngineException,
    InputError,
    ModelError,
    NotFoundError,
    RunawayError,
    StorageError,
    WorkerError,
)


def test_every_error_is_an_engine_exception() -> None:
    for cls in (
        ConfigurationError,
        NotFoundError,
        InputError,
        ModelError,
        RunawayError,
        DelegationLoopError,
        DelegationDepthError,
        StorageError,
        WorkerError,
    ):
        assert issubclass(cls, EngineException)
        assert issubclass(cls, Exception)
