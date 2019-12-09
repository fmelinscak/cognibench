import logging
from enum import IntEnum


ALL = logging.NOTSET + 1
INFO = logging.INFO
VERBOSE = logging.DEBUG
NO_LOGGING = logging.CRITICAL + 1


def logger():
    if not hasattr(logger, "_logger_ready"):
        _logger_ready = False

    log_obj = logging.getLogger("ldmunit")
    if not _logger_ready:
        _logger_ready = True
        _set_logger(log_obj)
    return log_obj


def set_logging_level(level=logging.INFO):
    logger().setLevel(level)


def _set_logger(log_obj):
    logging.basicConfig(
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d-%y %H:%M",
        level=INFO,
    )
