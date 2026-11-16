import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
from fastapi import Depends, HTTPException, Request
from .util import canonical_region


@dataclass(frozen=True)
class Actor:
    subject: str
    tenant_id: str
    scopes: frozenset[str]
    regions: frozenset[str]

    def require_region(self, region):
        try:
            value = canonical_region(region)
        except ValueError:
            raise HTTPException(422, detail="unsupported_region")
        if value not in self.regions:
            raise HTTPException(403, detail="region_forbidden")
        return value

    def has_scope(self, scope):
        return "admin" in self.scopes or scope in self.scopes


def authenticated_actor(request: Request) -> Actor:
    authorization = request.headers.get("authorization", "")
    if not authorization.startswith("Bearer ") or not 24 <= len(authorization[7:]) <= 512:
        raise HTTPException(401, detail="authentication_required")
    try:
        cfg = json.loads(Path(os.environ["SA_AUTH_TOKENS_FILE"]).read_text())
        registrations = cfg["tokens"]
        fingerprint = hashlib.sha256(authorization[7:].encode()).hexdigest()
        item = next((r for r in registrations if r.get("active", True) and hmac.compare_digest(r.get("sha256", ""), fingerprint)), None)
        if item is None or item["tenant_id"] != os.environ.get("SA_TENANT_ID", "example-tenant"):
            raise HTTPException(401, detail="invalid_credentials")
        actor = Actor(item["subject"], item["tenant_id"], frozenset(item["scopes"]), frozenset(item["regions"]))
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(503, detail="authentication_unavailable")
    request.state.actor = actor
    return actor


def require_scope(scope: str):
    def dependency(actor: Actor = Depends(authenticated_actor)):
        if not actor.has_scope(scope):
            raise HTTPException(403, detail="scope_forbidden")
        return actor
    return dependency
