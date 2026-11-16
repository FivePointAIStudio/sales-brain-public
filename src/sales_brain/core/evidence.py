"""Extracted source-bound evidence verifier; no ORM or model transport."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import unicodedata
from .context import ContextSnapshot
from .contracts import PLAN_SCHEMA
from .diagnostics import field_failure

class RunError(ValueError):
    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(message)

@dataclass(frozen=True)
class Plan:
    data: dict
    def to_dict(self):
        return deepcopy(self.data)

@dataclass
class Evidence:
    products: set[str] = field(default_factory=set)
    quotes: set[str] = field(default_factory=set)
    facts: set[str] = field(default_factory=set)
    rankings: set[str] = field(default_factory=set)
    events: set[str] = field(default_factory=set)
    cases: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    recipients: set[str] = field(default_factory=set)
    routes: set[str] = field(default_factory=set)
    commands: set[str] = field(default_factory=set)
    product_codes: dict[str, set[str]] = field(default_factory=dict)
    quote_rows: dict[str, dict] = field(default_factory=dict)

    def ingest(self, value, *, capabilities: bool = False):
        # Only server-side ContextProvider/registered read adapters call ingest.
        # Model drafts and model-written call arguments are never evidence.
        if isinstance(value, dict):
            if value.get("is_demo") is True or value.get("valid") is False or value.get("verified") is False:
                return
            for key, bucket in (("product_ref", self.products), ("quote_ref", self.quotes),
                                ("fact_ref", self.facts), ("ranking_evidence_ref", self.rankings),
                                ("event_ref", self.events), ("case_ref", self.cases),
                                ("source_ref", self.sources), ("recipient_ref", self.recipients),
                                ("channel_route_ref", self.routes)):
                if isinstance(value.get(key), str) and value[key]:
                    if key == "quote_ref" and value.get("valid") is not True:
                        continue  # Historical references need a fresh applicable-price read.
                    bucket.add(value[key])
            if isinstance(value.get("product_ref"), str):
                codes = [value.get(key) for key in ("dcode", "public_code", "product_code")]
                from .products import normalize_product_reference
                normalized = {normalize_product_reference(code) for code in codes if isinstance(code, str)} - {None}
                if normalized:
                    self.product_codes.setdefault(value["product_ref"], set()).update(normalized)
            if isinstance(value.get("quote_ref"), str) and value["quote_ref"] in self.quotes:
                self.quote_rows[value["quote_ref"]] = deepcopy(value)
            if capabilities and isinstance(value.get("command_ref"), str) and value.get("command_ref"):
                self.commands.add(value["command_ref"])
            for key, bucket in (("source_refs", self.sources), ("source_event_refs", self.events),
                                ("ranking_evidence_refs", self.rankings)):
                refs = value.get(key, [])
                if isinstance(refs, (list, tuple)):
                    bucket.update(x for x in refs if isinstance(x, str))
            for child in value.values():
                self.ingest(child, capabilities=capabilities)
        elif isinstance(value, (tuple, list)):
            for child in value:
                self.ingest(child, capabilities=capabilities)

    def verify(self, plan: Plan, snapshot: ContextSnapshot, *, now: datetime | None = None):
        data = plan.to_dict()
        now = now or datetime.now(timezone.utc)
        explicit_codes = _explicit_customer_codes(snapshot)
        covered = {r["intent_ref"] for r in data["intent_coverage"]}
        pending = {r["intent_ref"] for r in snapshot.pending_intents
                   if r.get("intent_ref") and r.get("status") not in {"completed", "cancelled", "answered"}}
        if pending - covered:
            raise RunError("uncovered_prior_intent", "Final plan omitted a pending customer intent")
        covered_events = {e for row in data["intent_coverage"] for e in row["source_event_refs"]}
        if data["actions"][0]["type"] != "NO_ACTION" and set(snapshot.selected_event_refs) - covered_events:
            raise RunError("uncovered_trigger_event", "Final plan omitted the selected customer event")
        for row in data["intent_coverage"]:
            if set(row["source_event_refs"]) - self.events:
                raise RunError("unknown_source_event", "Intent cites an unverified customer event")

        def verify_product(ref: str):
            for code in self.product_codes.get(ref, ()):
                if _conflicts_with_explicit(code, explicit_codes):
                    raise RunError("explicit_product_prefix_mismatch", "Product identity conflicts with the customer's explicit prefix")

        def verify_quote(ref: str):
            row = self.quote_rows.get(ref, {})
            from .context import canonical_region, ContextError
            try:
                if canonical_region(row.get("region", "")) != canonical_region(snapshot.region):
                    raise RunError("quote_scope_mismatch", "Quote is not applicable to this business region")
            except ContextError as exc:
                raise RunError("quote_scope_mismatch", "Quote has no verified business region") from exc
            if row.get("valid_until"):
                try:
                    expiry = datetime.fromisoformat(str(row["valid_until"]).replace("Z", "+00:00"))
                    if expiry.tzinfo is None or expiry <= now:
                        raise ValueError("expired")
                except (TypeError, ValueError):
                    raise RunError("quote_expired", "Quote expiry is invalid or has already passed")

        def reject_reference(key, path, bucket, count):
            error = RunError("unverified_reference", f"Unknown evidence in {key}")
            error.diagnostics = field_failure(error.code, path, PLAN_SCHEMA, stage='evidence_validation',
                reference_kind=key, verified_candidate_count=len(bucket), unknown_reference_count=count)
            raise error

        def walk(value, key="", *, excluded=False, path=()):
            if isinstance(value, dict):
                if "from_action_id" in value:
                    return  # Structural and producer checks belong to contracts.
                excluded = excluded or (key == "product_interests" and value.get("status") == "excluded")
                for k, v in value.items():
                    walk(v, k, excluded=excluded, path=(*path, k))
            elif isinstance(value, list):
                buckets = {"product_refs": self.products, "quote_refs": self.quotes,
                           "fact_refs": self.facts | self.sources,
                           "ranking_evidence_refs": self.rankings,
                           "source_event_refs": self.events}
                if key in buckets and set(value) - buckets[key]:
                    index = next(index for index, ref in enumerate(value) if ref not in buckets[key])
                    reject_reference(key, (*path, index), buckets[key], len(set(value) - buckets[key]))
                if key == "product_refs" and not excluded:
                    for ref in value:
                        verify_product(ref)
                if key == "quote_refs":
                    for ref in value:
                        verify_quote(ref)
                for index, v in enumerate(value):
                    walk(v, key, excluded=excluded, path=(*path, index))
            elif isinstance(value, str):
                buckets = {"product_ref": self.products, "recipient_ref": self.recipients,
                           "channel_route_ref": self.routes, "case_ref": self.cases,
                           "command_ref": self.commands}
                if key in buckets and value not in buckets[key]:
                    reject_reference(key, path, buckets[key], 1)
                if key == "product_ref" and not excluded:
                    verify_product(value)
        for index, action in enumerate(data["actions"]):
            walk(action["payload"], path=('actions', index, 'payload'))
            if action["type"] == "PROPOSE_CUSTOMER_FACT" and action["payload"]["subject_ref"] != snapshot.customer_ref:
                raise RunError("customer_scope_denied", "Fact belongs to another customer")


_LITERAL_PRODUCT = re.compile(r"(?<![A-Z0-9])([A-Z]{1,12}[-_]?\d{3,}(?:[-_][A-Z0-9]+)*)(?![A-Z0-9])", re.I)


def _explicit_customer_codes(snapshot: ContextSnapshot) -> set[str]:
    """Protect literal prefixes, not infer a purchase intent from prose.

    This intentionally narrow check addresses AP/FP-style substitution. Broader
    product selection, negation and alternatives remain Agent/catalog decisions.
    """
    from .products import normalize_product_reference
    selected = set(snapshot.selected_event_refs)
    found = set()
    for message in snapshot.messages:
        if (message.get("event_ref") not in selected or message.get("role") != "customer"
                or message.get("removed") or message.get("body_truncated")):
            continue
        text = unicodedata.normalize("NFKC", message.get("text", "")).upper()
        found.update(normalize_product_reference(match) for match in _LITERAL_PRODUCT.findall(text))
    return found - {None}


def _number_part(reference: str) -> str | None:
    match = re.fullmatch(r"(?:[A-Z]+[-_]?)?(\d+(?:[-_][A-Z0-9]+)*)", reference)
    return match.group(1) if match else None


def _conflicts_with_explicit(reference: str, explicit: set[str]) -> bool:
    suffix = _number_part(reference)
    return bool(suffix and reference not in explicit and any(_number_part(code) == suffix for code in explicit))

