import logging
from enum import IntEnum


# Logging levels.
# Written in decreasing level of logging. Check python3 logging documentation for more details.
ALL = logging.NOTSET + 1
VERBOSE = logging.DEBUG
INFO = logging.INFO
NO_LOGGING = logging.CRITICAL + 1


def logger():
    """
    Return cognibench logger object.

    See Also
    --------
    `<https://docs.python.org/3/library/logging.html>`_.
    """
    if not hasattr(logger, "_logger_ready"):
        _logger_ready = False

    log_obj = logging.getLogger("cognibench")
    if not _logger_ready:
        _logger_ready = True
        _set_logger(log_obj)
    return log_obj


def set_logging_level(level=logging.INFO):
    """
    Set cognibench logging level. See top level definitions.
    """
    logger().setLevel(level)


def _set_logger(log_obj):
    """
    Configure logger object.
    """
    logging.basicConfig(
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d-%y %H:%M",
        level=INFO,
    )
