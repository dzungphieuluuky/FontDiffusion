import logging
import os
import sys
import colorlog
def get_rank():
    """Get distributed rank for logging (if available)."""
    try:
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return int(os.environ.get("LOCAL_RANK", 0))

class RankFilter(logging.Filter):
    """Injects rank info into log records."""
    def filter(self, record):
        record.rank = get_rank()
        return True

def setup_logging(level=logging.INFO, name=None):
    """
    Unified logging setup for scripts and notebooks.
    Args:
        level: Logging level (default: INFO)
        name: Logger name (default: root)
    Returns:
        Configured logger
    """
    # Remove all handlers (prevents duplicate logs in notebooks/reloads)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Try colorlog for notebook friendliness
    try:
        from colorlog import ColoredFormatter
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] [Rank %(rank)d] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
            }
        )
    except ImportError:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [Rank %(rank)d] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # StreamHandler for stdout (works in notebooks and scripts)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addFilter(RankFilter())
    logger.propagate = False  # Prevent double logging

    return logger

# Usage example (put this at the top of your script/module):
# logger = setup_logging()
# logger.info("Logging is configured!")