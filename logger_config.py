import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evita duplicar handlers
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
