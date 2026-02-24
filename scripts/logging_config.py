import logging
import os
from logging.handlers import RotatingFileHandler


def configure_logging(level='INFO'):
    """Configure root logger with console and rotating file handler.

    - human-friendly format with timestamp, level, module
    - RotatingFileHandler -> logs/app.log (5MB, 3 backups)
    - sets root logger level to given level (str or int)
    """
    # Resolve numeric level if string provided
    if isinstance(level, str):
        level_name = level.upper()
        logging_level = getattr(logging, level_name, logging.INFO)
    else:
        logging_level = level

    # Ensure logs directory exists (create in project cwd)
    logs_dir = os.path.join(os.getcwd(), 'logs')
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        # best-effort: proceed even if cannot create
        pass

    log_file = os.path.join(logs_dir, 'app.log')

    # Console-friendly format
    fmt = '%(asctime)s %(levelname)-8s [%(module)s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    # Configure basic logging to console
    logging.basicConfig(level=logging_level, format=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging_level)

    # Add RotatingFileHandler if not already present
    try:
        fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        fh.setLevel(logging_level)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        # Avoid duplicate handlers on repeated calls
        if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
            root.addHandler(fh)
    except Exception:
        root.debug('Failed to create RotatingFileHandler for %s', log_file)
import logging
import os


def configure_logging(level: str | None = None) -> None:
    """Configure root logging for CLI and subprocess runs.

    - Uses `LOG_LEVEL` env var if `level` is None.
    - Leaves existing handlers in place to avoid duplicate output when called multiple times.
    """
    lvl = level or os.environ.get("LOG_LEVEL", "INFO")
    try:
        numeric = getattr(logging, lvl.upper(), logging.INFO)
    except Exception:
        numeric = logging.INFO

    root = logging.getLogger()
    # Simple human-friendly formatter; consumers can replace with JSON formatter if desired.
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))

    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(numeric)
