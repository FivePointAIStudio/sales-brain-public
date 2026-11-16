"""Source-bound sales policy, with no network calls or message sending.

Trust boundary: VerifiedContactIntent is created by the host's IntentResolver,
never directly from model JSON. Callers must supply a verifier backed by the
original, scoped source event. This module is an executable reference policy,
not an ERP adapter or an end-to-end notification implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping


ACTION_TYPES = frozenset({
    "REPLY_TEXT", "ASK_CLARIFICATION", "RECOMMEND_PRODUCTS",
    "PROPOSE_PROJECT_PATCH", "HANDOFF_SALES", "PROPOSE_CUSTOMER_FACT",
    "NOTIFY_SALES", "NO_ACTION",
})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 300:
        raise ValueError(f"invalid_{name}")
    return value


def _json_normal(value: Any) -> Any:
    """Canonicalize typed domain facts, not free-form model summaries."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non_finite_material_fact")
        return int(value) if value.is_integer() else value
    if isinstance(value, (list, tuple)):
        return [_json_normal(x) for x in value]
    if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
        return {k: _json_normal(v) for k, v in value.items()}
    raise ValueError("unsupported_material_fact")


def _digest(value: Any) -> str:
    raw = json.dumps(_json_normal(value), sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class CustomerScope:
    tenant_ref: str
    environment: str
    customer_ref: str
    region: str

    def __post_init__(self):
        for name in ("tenant_ref", "environment", "customer_ref", "region"):
            _text(getattr(self, name), name)

    def matches(self, value: Mapping[str, Any]) -> bool:
        return all(value.get(k) == getattr(self, k) for k in
                   ("tenant_ref", "environment", "customer_ref", "region"))


@dataclass(frozen=True)
class VerifiedContactIntent:
    scope: CustomerScope
    intent_ref: str
    source_event_refs: tuple[str, ...]
    product_refs: tuple[str, ...] = ()
    kind: str = "contact_request"

    def __post_init__(self):
        _text(self.intent_ref, "intent_ref")
        if self.kind != "contact_request" or not self.source_event_refs:
            raise ValueError("contact_intent_needs_verified_sources")
        for ref in self.source_event_refs + self.product_refs:
            _text(ref, "source_or_product_ref")
        if len(set(self.source_event_refs)) != len(self.source_event_refs):
            raise ValueError("duplicate_source_events")


@dataclass(frozen=True)
class PolicyContext:
    """Authority/configuration from the host; never accepted from a draft."""
    customer_send_enabled: bool = True
    business_writes_enabled: bool = True
    sales_notify_enabled: bool = True
    manual_takeover: bool = False
    internal_assist_during_takeover: bool = False
    sales_owner_available: bool = True
    no_outbound_contact: bool = False


@dataclass(frozen=True)
class ActionRequirement:
    name: str
    action_type: str
    required: bool
    blocked_reason: str | None = None
    request_type: str | None = None


@dataclass(frozen=True)
class CoverageIssue:
    code: str
    requirement: str
    message: str


IntentVerifier = Callable[[VerifiedContactIntent], bool]


def required_contact_actions(
    intent: VerifiedContactIntent,
    context: PolicyContext,
    *, verify_intent: IntentVerifier,
) -> tuple[ActionRequirement, ...]:
    """Derive four requirements; automatic callback is never implied.

    A no-outbound preference can still be recorded internally. The responsible
    salesperson must see that preference, and outbound contact remains blocked
    by the contact/channel adapter. This function does not authorize a call.
    """
    if verify_intent(intent) is not True:
        raise ValueError("unverified_contact_intent")
    customer_allowed = context.customer_send_enabled and not context.manual_takeover
    internal_allowed = (context.business_writes_enabled and
                        (not context.manual_takeover or context.internal_assist_during_takeover))
    notice_allowed = (internal_allowed and context.sales_notify_enabled and
                      context.sales_owner_available)
    internal_reason = None if internal_allowed else "internal_assist_not_authorized"
    notice_reason = (None if notice_allowed else
                     internal_reason or
                     ("sales_notify_disabled" if not context.sales_notify_enabled
                      else "sales_owner_unavailable"))
    return (
        ActionRequirement("answer_contact", "REPLY_TEXT", customer_allowed,
                          None if customer_allowed else "customer_send_suppressed"),
        ActionRequirement("save_interest", "PROPOSE_PROJECT_PATCH", internal_allowed,
                          internal_reason),
        ActionRequirement("sales_followup", "HANDOFF_SALES", internal_allowed,
                          internal_reason, "sales_followup"),
        ActionRequirement("notify_sales", "NOTIFY_SALES", notice_allowed, notice_reason),
    )


def contact_reply_is_independent(
    action: Mapping[str, Any], all_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    """The public answer must be deliverable even when an internal write stalls.

    A second, receipt-bound acknowledgement is allowed after the independent
    answer. Checking the transitive graph prevents a reply dependency from
    concealing a business-write dependency farther upstream.
    """
    if action.get("type") != "REPLY_TEXT":
        return False
    visited: set[str] = set()
    pending = [d.get("action_id") for d in action.get("depends_on", [])]
    while pending:
        aid = pending.pop()
        if aid in visited:
            continue
        visited.add(aid)
        dep = all_by_id.get(aid, {})
        if dep.get("type") in {"PROPOSE_PROJECT_PATCH", "HANDOFF_SALES", "NOTIFY_SALES", "PROPOSE_CUSTOMER_FACT"}:
            return False
        pending.extend(d.get("action_id") for d in dep.get("depends_on", []))
    return True


def check_contact_coverage(
    plan: Mapping[str, Any], intent: VerifiedContactIntent,
    context: PolicyContext, *, verify_intent: IntentVerifier,
) -> tuple[CoverageIssue, ...]:
    """Bridge into validate_candidate(plan, context, observations).

    Checks scoped coverage, all four actions, callback misclassification, and
    nonblocking contact reply. The shared contract must additionally validate
    typed payloads, source ownership, resource bindings, DAG shape, permissions,
    company facts, responsible-recipient resolution and current source window.
    Blocked requirements returned by required_contact_actions must be preserved
    as structured unresolved reasons by the plan compiler; they are not success.
    """
    requirements = required_contact_actions(intent, context, verify_intent=verify_intent)
    issues: list[CoverageIssue] = []
    if not intent.scope.matches(plan):
        return (CoverageIssue("scope_mismatch", "plan", "Plan does not match trusted customer scope"),)
    rows = [r for r in plan.get("intent_coverage", [])
            if isinstance(r, Mapping) and r.get("intent_ref") == intent.intent_ref]
    if len(rows) != 1:
        return (CoverageIssue("missing_or_duplicate_intent", "coverage", "One contact intent coverage row is required"),)
    row = rows[0]
    if (row.get("kind") != "contact_request" or row.get("confirmation") != "confirmed"
            or not set(intent.source_event_refs).issubset(row.get("source_event_refs", []))):
        issues.append(CoverageIssue("unverified_coverage", "coverage", "Confirmed contact sources must be retained"))
    linked = set(row.get("action_ids", []))
    actions = [a for a in plan.get("actions", []) if isinstance(a, Mapping)]
    all_by_id = {a.get("action_id"): a for a in actions}
    # One saved purchase case can cover both product_interest and contact_request.
    # Coverage links plus real sources, rather than one exclusive intent label,
    # establish that this shared action actually addresses the contact request.
    covered = [a for a in actions if a.get("action_id") in linked]
    for req in requirements:
        matches = [a for a in covered if a.get("type") == req.action_type
                   and (req.request_type is None or
                        a.get("payload", {}).get("request_type") == req.request_type)]
        if req.required and not matches:
            issues.append(CoverageIssue("missing_required_action", req.name, req.action_type))
        if not req.required and matches:
            issues.append(CoverageIssue("action_not_authorized", req.name, req.blocked_reason or "blocked"))
        if (not req.required and not matches and req.blocked_reason
                and req.blocked_reason not in str(row.get("reason", ""))):
            issues.append(CoverageIssue("missing_blocked_reason", req.name, req.blocked_reason))
    for action in covered:
        payload = action.get("payload", {})
        if (action.get("type") in {"PROPOSE_PROJECT_PATCH", "HANDOFF_SALES", "NOTIFY_SALES"}
                and not set(intent.source_event_refs).issubset(payload.get("source_event_refs", []))):
            issues.append(CoverageIssue("action_sources_missing", "coverage",
                                        str(action.get("action_id"))))
        if action.get("type") == "HANDOFF_SALES" and payload.get("request_type") == "callback":
            issues.append(CoverageIssue("callback_needs_separate_intent", "sales_followup",
                                        "A contact_request must not be used as callback consent"))
        if (context.no_outbound_contact and action.get("type") == "HANDOFF_SALES"
                and payload.get("request_type") == "sales_followup"
                and payload.get("parameters", {}).get("no_outbound_contact") is not True):
            issues.append(CoverageIssue("outbound_restriction_missing", "sales_followup",
                                        "The follow-up must carry the verified no-outbound preference"))
    if (any(req.name == "answer_contact" and req.required for req in requirements)
            and not any(contact_reply_is_independent(action, all_by_id) for action in covered)):
        issues.append(CoverageIssue("contact_reply_blocked_by_internal_action", "answer_contact",
                                    "At least one contact answer must be independent of internal writes"))
    return tuple(issues)


MATERIAL_FIELDS = frozenset({
    "selected_display_keys", "quantity", "color", "width_cm", "weight_gsm",
    "lead_time_days", "urgency", "contact_preference", "purpose", "destination_region",
})


@dataclass(frozen=True)
class MaterialFact:
    key: str
    value: Any
    source_event_refs: tuple[str, ...]


def material_snapshot(facts: Iterable[MaterialFact]) -> dict[str, Any]:
    """Facts must be verified and normalized to domain units upstream.

    Provenance is mandatory but deliberately absent from the material digest:
    repeating the same fact in another message is not a new sales opportunity.
    Product values use stable display identities, not versioned product_ref.
    """
    result: dict[str, Any] = {}
    for fact in facts:
        if fact.key not in MATERIAL_FIELDS or fact.key in result:
            raise ValueError("unknown_or_duplicate_material_field")
        if not fact.source_event_refs:
            raise ValueError("material_fact_requires_source")
        for ref in fact.source_event_refs:
            _text(ref, "fact_source")
        value = _json_normal(fact.value)
        if fact.key == "selected_display_keys":
            if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
                raise ValueError("invalid_selected_display_keys")
            value = sorted(set(value))
        result[fact.key] = value
    return result


def material_fingerprint(facts: Iterable[MaterialFact]) -> str:
    return _digest(material_snapshot(facts))


def changed_material_fields(previous: Iterable[MaterialFact], current: Iterable[MaterialFact]) -> tuple[str, ...]:
    old, new = material_snapshot(previous), material_snapshot(current)
    return tuple(sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k)
                        or (k in old) != (k in new)))


def semantic_notification_key(
    scope: CustomerScope, *, followup_ref: str, semantic_revision: int,
    recipient_ref: str, route_ref: str, notice_reason: str,
) -> str:
    """Revision is atomically assigned by sales storage after material diff.

    Do not use the current snapshot digest alone as the message key: A -> B -> A
    is a real later change and needs a new semantic revision. Merely adding a
    source event or rewording a summary must not increment that revision.
    """
    for name, value in (("followup_ref", followup_ref), ("recipient_ref", recipient_ref),
                        ("route_ref", route_ref), ("notice_reason", notice_reason)):
        _text(value, name)
    if type(semantic_revision) is not int or semantic_revision < 1:
        raise ValueError("semantic_revision_must_be_persisted_positive_integer")
    return "sales-notice:" + _digest([
        scope.tenant_ref, scope.environment, scope.customer_ref, scope.region,
        followup_ref, semantic_revision, recipient_ref, route_ref, notice_reason,
    ])


def receipt_allows_claim(
    claim: str, receipt: Mapping[str, Any], *, scope: CustomerScope,
    expected_action_id: str, expected_resource_ref: str,
    verify_receipt: Callable[[Mapping[str, Any]], bool],
) -> bool:
    """Validate state AND action/resource/scope binding before success wording.

    A nonempty status string is not evidence. The verifier must read the actual
    persisted source receipt and check its provider/resource binding. This
    function does not generate customer text or treat business completion as a
    universal boolean.
    """
    _text(expected_action_id, "expected_action_id")
    _text(expected_resource_ref, "expected_resource_ref")
    if (verify_receipt(receipt) is not True or not scope.matches(receipt)
            or receipt.get("action_id") != expected_action_id
            or not receipt.get("source_receipt_ref")
            or receipt.get("execution_status") != "succeeded"):
        return False
    if claim == "interest_recorded":
        return (receipt.get("action_type") == "PROPOSE_PROJECT_PATCH"
                and receipt.get("resource_kind") == "purchase_case"
                and receipt.get("resource_ref") == expected_resource_ref
                and receipt.get("resource_version") is not None)
    if claim == "sales_notice_sent":
        return (receipt.get("action_type") == "NOTIFY_SALES"
                and receipt.get("request_ref") == expected_resource_ref
                and receipt.get("delivery_status") in {"sent", "delivered", "read"})
    if claim == "sales_followup_accepted":
        return (receipt.get("action_type") == "HANDOFF_SALES"
                and receipt.get("resource_kind") in {"sales_request", "sales_followup"}
                and receipt.get("resource_ref") == expected_resource_ref
                and receipt.get("business_status") == "accepted"
                and receipt.get("business_event") == "owner_accepted")
    return False
