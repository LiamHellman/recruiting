import hashlib
import json
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import AuditLog


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any) -> str:
    """Hash of a JSON-serialisable object, independent of key order."""
    return sha256_hex(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))


def record(
    session: Session,
    *,
    tenant_id: uuid.UUID,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str | uuid.UUID,
    payload: dict[str, Any],
) -> AuditLog:
    """Append an audit entry to the caller's transaction.

    Payloads hold IDs, hashes, versions and non-personal settings (skill IDs, weights).
    Never put CV text, names, emails or other personal data here.
    """
    entry = AuditLog(
        tenant_id=tenant_id,
        actor=actor,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id),
        payload=payload,
    )
    session.add(entry)
    return entry
