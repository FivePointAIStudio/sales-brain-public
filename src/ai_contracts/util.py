import hashlib
import json
from datetime import datetime, timezone


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def utcnow():
    return datetime.now(timezone.utc)


def canonical_region(region):
    result = str(region).strip().upper()
    result = {"HANOI": "HN", "HÀ NỘI": "HN", "BANGKOK": "BKK", "HOCHIMINH": "HCM", "INDONESIA": "IDK"}.get(result, result)
    if result not in {"IDK", "BKK", "HCM", "HN"}:
        raise ValueError("unsupported_region")
    return result
