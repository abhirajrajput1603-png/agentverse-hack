from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
AUDIT_LOG_PATH = LOGS_DIR / "audit.jsonl"


class AuditLogger:
    """
    Append-only deterministic JSONL audit logger.
    Logs every safety check, circuit breaker evaluation, and user decision.
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = log_path or AUDIT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Appends a structured event to the audit trail with UTC timestamp."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "details": details,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def get_recent_entries(self, limit: int = 50) -> list[Dict[str, Any]]:
        """Reads recent audit log entries."""
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        return entries[-limit:]
