from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

from ..config import settings

_configured = False


def configure_audit_logger() -> logging.Logger:
    global _configured
    logger = logging.getLogger("audit")
    if _configured:
        return logger

    logger.setLevel(getattr(logging, settings.AUDIT_LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )
        logger.addHandler(stream_handler)

    try:
        log_path = settings.AUDIT_LOG_PATH
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
            file_handler = RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
            )
            logger.addHandler(file_handler)
    except Exception:
        # Keep audit logging on stdout even if file handler cannot be initialised.
        pass

    _configured = True
    return logger


def audit_event(
    event: str,
    *,
    actor: Optional[Any] = None,
    actor_username: Optional[str] = None,
    target_type: Optional[str] = None,
    target_id: Optional[Any] = None,
    status: str = "success",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    logger = configure_audit_logger()
    username = actor_username or getattr(actor, "username", None) or "system"
    actor_id = getattr(actor, "id", None)
    payload = {
        "event": event,
        "status": status,
        "actor_username": username,
        "actor_id": actor_id,
        "target_type": target_type,
        "target_id": target_id,
        "details": details or {},
    }
    logger.info(json.dumps(payload, default=str, ensure_ascii=True))

