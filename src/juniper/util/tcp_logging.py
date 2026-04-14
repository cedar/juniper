import logging
import os
from pathlib import Path


_LOGGER_CACHE = {}


def get_tcp_logger(name: str) -> logging.Logger:
    logger = _LOGGER_CACHE.get(name)
    if logger is not None:
        return logger

    log_path = Path(os.getcwd()) / "juniper_tcp.log"
    logger = logging.getLogger(f"juniper.tcp.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _LOGGER_CACHE[name] = logger
    return logger
