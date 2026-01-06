import logging
import os
import sys

def get_rank():
    try:
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return int(os.environ.get("LOCAL_RANK", 0))

class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = get_rank()
        return True

class TqdmLoggingHandler(logging.Handler):
    """Handler that writes log records using tqdm.write (safe with progress bars)."""
    def emit(self, record):
        try:
            from tqdm.auto import tqdm
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(level=logging.INFO, name=None, use_tqdm=False):
    # Remove all handlers from root logger
    root_logger = logging.getLogger()
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])

    log_format = "%(asctime)s [%(levelname)s] [Rank %(rank)d] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    try:
        from colorlog import ColoredFormatter
        formatter = ColoredFormatter(
            "%(log_color)s" + log_format,
            datefmt=date_format,
            log_colors={
                'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
                'ERROR': 'red', 'CRITICAL': 'bold_red',
            }
        )
    except ImportError:
        formatter = logging.Formatter(log_format, datefmt=date_format)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addFilter(RankFilter())

    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    if use_tqdm:
        handler = TqdmLoggingHandler()
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger

# Usage example (put this at the top of your script/module):
# logger = setup_logging()
# logger.info("Logging is configured!")