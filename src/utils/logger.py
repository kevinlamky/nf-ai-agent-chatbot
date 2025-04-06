import os
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)


# Configure the root logger with a file handler
def setup_logging(level=logging.INFO):
    """Setup basic logging configuration."""
    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create file handler that logs all messages
    current_date = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(f"logs/app_{current_date}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Create console handler with the same log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Configure the root logger
    logging.basicConfig(
        level=level, handlers=[file_handler, console_handler], force=True
    )


# Get a logger for the specified module
def get_logger(name):
    """Get a logger with the given name."""
    return logging.getLogger(name)


# Initialize logging with default settings
setup_logging(level=logging.INFO)
