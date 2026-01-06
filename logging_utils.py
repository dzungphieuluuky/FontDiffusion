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

# ANSI escape code helper for 24-bit color
def hex_to_ansi(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"\033[38;2;{r};{g};{b}m"

RESET = "\033[0m"
LEVEL_COLORS = {
    'DEBUG':    hex_to_ansi('#B31E6F'),
    'INFO':     hex_to_ansi('#22EAAA'),
    'WARNING':  hex_to_ansi('#FFB174'),
    'ERROR':    hex_to_ansi('#9E2A3A'),
    'CRITICAL': hex_to_ansi('#FF0000'),
}

class HexColorFormatter(logging.Formatter):
    def format(self, record):
        color = LEVEL_COLORS.get(record.levelname, "")
        msg = super().format(record)
        if color and sys.stdout.isatty():
            msg = f"{color}{msg}{RESET}"
        return msg

def setup_logging(level=logging.INFO, name=None, use_tqdm=False):
    # Remove all handlers from root logger
    root_logger = logging.getLogger()
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])

    log_format = "%(asctime)s [%(levelname)s] [Rank %(rank)d] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = HexColorFormatter(log_format, datefmt=date_format)

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

# Usage example:
# logger = setup_logging()
# logger.info("Logging is configured!")