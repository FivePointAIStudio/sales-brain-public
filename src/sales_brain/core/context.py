"""Read-only customer context and an explicit adapter for the audited preview DB.

The adapter does not create identities, infer an ERP customer, advance cursors or
save memory. The old database has no tenant column: its tenant is a mandatory,
trusted constructor binding. It cannot be used as a live customer data source.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
import hashlib
import json
from typing import Any, Callable, Mapping, Protocol


REGIONS = frozenset({"BKK", "HN", "HCM", "IDK"})
LANGUAGES = frozenset({"zh", "en", "th", "vi", "id"})


class ContextError(ValueError):
    def __init__(self, code: str, message: str, *, details: Mapping | None = None):
        super().__init__(message)
        self.code, self.message = code, message
        self.details = dict(details or {})

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


def canonical_region(value: str) -> str:
    if not isinstance(value, str):
        raise ContextError("invalid_region", "Business region must be a string")
    region = "HN" if value in {"Hanoi", "HANOI", "hanoi"} else value
    if region not in REGIONS:
        raise ContextError("invalid_region", "Unsupported business region")
    return region


def version_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     default=str, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RequestScope:
    """Constructed by authentication, never from model output or request JSON.

    A wildcard customer grant is meaningful only with an explicitly tenant-bound
    repository. A region grant never grants another customer's personal records.
    """
    tenant_ref: str
    environment: str
    allowed_regions: frozenset[str]
    caller_ref: str
    allowed_customer_refs: frozenset[str]

    def __post_init__(self):
        if not all(isinstance(x, str) and 0 < len(x) <= 160 for x in (self.tenant_ref, self.caller_ref)):
            raise ContextError("invalid_scope", "Tenant and caller are required")
        if self.environment not in {"sandbox", "shadow", "live"}:
            raise ContextError("invalid_environment", "Unknown environment")
        if isinstance(self.allowed_regions, (str, bytes)) or isinstance(self.allowed_customer_refs, (str, bytes)):
            raise ContextError("invalid_scope", "Region and customer grants must be explicit collections")
        object.__setattr__(self, "allowed_regions", frozenset(
            canonical_region(x) for x in self.allowed_regions))
        object.__setattr__(self, "allowed_customer_refs", frozenset(
            str(x) for x in self.allowed_customer_refs))
        if not self.allowed_regions or not self.allowed_customer_refs:
            raise ContextError("empty_scope", "An explicit customer and region grant is required")

    def require(self, customer_ref: str, region: str | None = None) -> None:
        if "*" not in self.allowed_customer_refs and customer_ref not in self.allowed_customer_refs:
            raise ContextError("customer_scope_denied", "Customer is outside the caller's scope")
        if region is not None and canonical_region(region) not in self.allowed_regions:
            raise ContextError("region_scope_denied", "Region is outside the caller's scope")


@dataclass(frozen=True)
class ContextSnapshot:
    customer_ref: str
    region: str
    messages: tuple[dict, ...]
    agent_config: dict
    pending_intents: tuple[dict, ...] = ()
    displayed_products: tuple[dict, ...] = ()
    facts: tuple[dict, ...] = ()
    cases: tuple[dict, ...] = ()
    receipts: tuple[dict, ...] = ()
    context_version: str = ""
    config_version: str = ""
    selected_event_refs: tuple[str, ...] = ()
    customer: dict = field(default_factory=dict)
    control: dict = field(default_factory=dict)
    source_versions: dict = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    private_snapshot: dict = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict:
        # Do NOT use dataclasses.asdict here: the private legacy source includes
        # unredacted text and tool-only identities, and must never reach a model.
        return {f.name: deepcopy(getattr(self, f.name)) for f in fields(self)
                if f.name != "private_snapshot"}


class ContextProvider(Protocol):
    def load(self, customer_ref: str, scope: RequestScope,
             event_ref: str | None = None) -> ContextSnapshot: ...


