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

def setup_logging(level=logging.INFO, name="app"):
    # 1. Clean up the root logger to prevent duplicates
    root_logger = logging.getLogger()
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])

    # 2. Define the format (handling potential missing 'rank' attribute)
    log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    try:
        from colorlog import ColoredFormatter
        formatter = ColoredFormatter(
            f"%(log_color)s{log_format}",
            datefmt=date_format,
            log_colors={
                'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
                'ERROR': 'red', 'CRITICAL': 'bold_red',
            }
        )
    except ImportError:
        formatter = logging.Formatter(log_format, datefmt=date_format)

    # 3. Create the specific named logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Standard StreamHandler for notebook/console output
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Important: Remove any old handlers from *this* specific logger too
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
        
    logger.addHandler(handler)
    logger.propagate = False  # Avoid sending logs to the root logger

    return logger

# Usage example (put this at the top of your script/module):
# logger = setup_logging()
# logger.info("Logging is configured!")