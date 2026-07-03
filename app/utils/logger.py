import logging
import sys
from functools import lru_cache

def setup_logging(log_level:str=logging.INFO):
    log_format=logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup Root Logger
    root_logger=logging.getLogger()
    root_logger.setLevel(logging,log_level.upper(),logging.INFO)

    # Remove existing handlers
    for handlers in root_logger.handlers[:]:
        root_logger.removeHandler(handlers)
    
    # Setup Console Handler
    console_handler=logging.StreamHandler(sys.stdout)
    console_handler.formatter(log_format)
    root_logger.addHandler(console_handler)

    # Avoid noise from 3rd party Libraries
    logging.getLogger('httpx').setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@lru_cache
def get_logger(name:str)->logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

class LoggingMixin():
    """Mixin class to add logging capability to classes."""
    @property
    def logger(self)->logging.Logger:
        return get_logger(self.__class__.__name__)

    