"""Logging configuration for HPI-FHFA."""

import structlog
from structlog.stdlib import LoggerFactory
import logging
import sys
from typing import Optional


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output logs as JSON
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )
    
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    
    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
    ]
    
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stdout.isatty()
        )
    
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib ProcessorFormatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a logger instance.
    
    Args:
        name: Logger name (uses caller's module if not provided)
        
    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)