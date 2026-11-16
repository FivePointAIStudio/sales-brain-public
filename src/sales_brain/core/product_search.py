"""Request/identity validation used by the actual v2.1 product tool adapters.

This module does NOT replace the company's matching algorithm. ``SearchScope``
and source messages must come from the authorized server-side context assembler,
never directly from model tool arguments. Quote checks prove provenance only;
the brain/requirements service must separately verify requirement semantics.

The owner advertises ``matcher.plan_source.v1`` and ``catalog.search.v1`` only
when the actual service/store implementations support them. Clients negotiate
these capabilities and reject an older service instead of dropping conditions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Mapping, Sequence
import unicodedata


class SearchValidationError(ValueError):
    """A stable category suitable for a sanitized tool error."""


REGIONS = frozenset({"BKK", "HN", "HCM", "IDK"})
SEARCH_MODES = frozenset({"resolve_exact", "resolve_candidates", "recommend", "reorder"})
PLAN_SOURCES = frozenset({"original_text", "confirmed_conditions"})
SOURCE_KINDS = frozenset({"customer", "research", "sales", "unknown"})
NECESSITIES = frozenset({"required", "preferred", "option", "to_confirm", "context"})
CONDITION_KINDS = frozenset({
    "usage", "audience", "construction", "fiber", "weight", "width", "property",
    "color", "pattern", "price", "quantity", "delivery", "quality", "stock", "sku", "other",
})
CONDITION_FIELDS = frozenset({
    "id", "kind", "name", "values", "alternatives", "min", "max", "unit",
    "negated", "necessity", "source_kind", "clause_ids", "quote",
})
CODE = re.compile(r"[A-Z0-9][A-Z0-9_-]{0,63}\Z")
# Shared by the app client, owner endpoint and bounded worker transport.
# Never truncate a source window just to fit this budget.
MAX_REQUEST_BYTES = 256 * 1024


def _json_hash(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _text(value: Any, code: str, max_length: int = 240) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise SearchValidationError(code)
    return value


def normalize_reference(value: str) -> str:
    """Normalize one identified product-reference field, not a whole utterance."""
    if not isinstance(value, str):
        raise SearchValidationError("invalid_product_reference")
    result = "".join(unicodedata.normalize("NFKC", value).upper().split())
    if not CODE.fullmatch(result):
        raise SearchValidationError("invalid_product_reference")
    return result


@dataclass(frozen=True)
class SourceMessage:
    event_ref: str
    text: str
    source_kind: str = "customer"


@dataclass(frozen=True)
class SearchScope:
    """Trusted, already authorized scope, including an optional verified ERP code."""
    tenant_ref: str
    customer_ref: str
    region: str
    confirmed_customer_code: str | None = None

    def __post_init__(self) -> None:
        _text(self.tenant_ref, "invalid_tenant")
        _text(self.customer_ref, "invalid_customer")
        if not isinstance(self.region, str) or self.region not in REGIONS:
            raise SearchValidationError("invalid_region")
        if self.confirmed_customer_code is not None:
            _text(self.confirmed_customer_code, "invalid_confirmed_customer_code", 160)


@dataclass(frozen=True)
class ValidatedConditions:
    conditions: tuple[dict[str, Any], ...]
    hard_expression: dict[str, Any] | None
    condition_sources: dict[str, tuple[str, ...]]
    source_window_sha256: str


@dataclass(frozen=True)
class SearchRequest:
    service: str
    path: str
    body: dict[str, Any]
    context: dict[str, Any]
    required_capabilities: frozenset[str]

    def __post_init__(self) -> None:
        try:
            encoded = json.dumps(self.body, ensure_ascii=False, allow_nan=False).encode()
        except (TypeError, ValueError):
            raise SearchValidationError("invalid_serializable_request") from None
        if len(encoded) > MAX_REQUEST_BYTES:
            raise SearchValidationError("request_byte_budget_exceeded")

    def require_capabilities(self, available: Sequence[str]) -> None:
        if not self.required_capabilities <= set(available):
            raise SearchValidationError("upstream_contract_upgrade_required")


def _source_window(messages: Sequence[SourceMessage]) -> tuple[dict[str, SourceMessage], str]:
    if not messages or len(messages) > 100:
        raise SearchValidationError("invalid_source_window")
    result: dict[str, SourceMessage] = {}
    total = 0
    for message in messages:
        if not isinstance(message, SourceMessage):
            raise SearchValidationError("invalid_source_message")
        _text(message.event_ref, "invalid_source_event_ref", 200)
        _text(message.text, "invalid_source_text", 20_000)
        if (message.event_ref in result or not isinstance(message.source_kind, str)
                or message.source_kind not in SOURCE_KINDS):
            raise SearchValidationError("invalid_source_identity")
        total += len(message.text)
        result[message.event_ref] = message
    if total + len(messages) - 1 > 20_000:
        raise SearchValidationError("source_window_too_large")
    digest = _json_hash([
        {"event_ref": m.event_ref, "text": m.text, "source_kind": m.source_kind}
        for m in messages
    ])
    return result, digest


def _string_list(value: Any, code: str, *, maximum: int, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum or (not value and not allow_empty):
        raise SearchValidationError(code)
    for item in value:
        _text(item, code, 600)
    return list(value)


def _validate_expression(expression: Any, hard_ids: set[str]) -> dict[str, Any] | None:
    if not hard_ids:
        if expression is not None:
            raise SearchValidationError("nonhard_expression")
        return None
    if expression is None:
        # Do not guess AND when the confirmed customer requirement may be OR.
        raise SearchValidationError("explicit_hard_expression_required")
    nodes = 0
    referenced: set[str] = set()

    def visit(node: Any, depth: int) -> dict[str, Any]:
        nonlocal nodes
        nodes += 1
        if depth > 8 or nodes > 150 or not isinstance(node, dict):
            raise SearchValidationError("invalid_hard_expression")
        if set(node) == {"condition"}:
            ident = node["condition"]
            if not isinstance(ident, str) or ident not in hard_ids:
                raise SearchValidationError("nonhard_or_unknown_condition_reference")
            referenced.add(ident)
            return {"condition": ident}
        if (set(node) != {"op", "children"} or not isinstance(node["op"], str)
                or node["op"] not in {"and", "or"}):
            raise SearchValidationError("invalid_hard_expression")
        children = node["children"]
        if not isinstance(children, list) or not 1 <= len(children) <= 60:
            raise SearchValidationError("invalid_hard_expression")
        return {"op": node["op"], "children": [visit(c, depth + 1) for c in children]}

    clean = visit(expression, 1)
    if referenced != hard_ids:
        raise SearchValidationError("hard_condition_coverage_gap")
    return clean


def validate_conditions(
    conditions: Sequence[Mapping[str, Any]],
    hard_expression: Mapping[str, Any] | None,
    source_messages: Sequence[SourceMessage],
    condition_sources: Mapping[str, Sequence[str]],
) -> ValidatedConditions:
    """Check shape, exact quote provenance, owner and hard-tree coverage.

    This deliberately does not claim that ``min=150`` semantically follows from
    a quoted sentence. A requirements service must verify that before approving
    a confirmed plan. Condition bindings stay outside the native Condition so
    unknown per-condition fields are not silently sent to its strict schema.
    """
    sources, digest = _source_window(source_messages)
    if (not isinstance(conditions, (list, tuple)) or not 1 <= len(conditions) <= 60
            or not isinstance(condition_sources, Mapping)):
        raise SearchValidationError("invalid_conditions")
    clean: list[dict[str, Any]] = []
    ids: set[str] = set()
    bindings: dict[str, tuple[str, ...]] = {}
    for raw in conditions:
        if not isinstance(raw, Mapping) or set(raw) - CONDITION_FIELDS:
            raise SearchValidationError("invalid_condition_fields")
        ident = _text(raw.get("id"), "invalid_condition_id", 160)
        if ident in ids:
            raise SearchValidationError("duplicate_condition_id")
        ids.add(ident)
        if (not isinstance(raw.get("kind"), str) or raw["kind"] not in CONDITION_KINDS
                or not isinstance(raw.get("necessity"), str) or raw["necessity"] not in NECESSITIES):
            raise SearchValidationError("invalid_condition_kind_or_necessity")
        if not isinstance(raw.get("source_kind"), str) or raw["source_kind"] not in SOURCE_KINDS:
            raise SearchValidationError("invalid_condition_source_kind")
        name = _text(raw.get("name"), "invalid_condition_name", 160)
        quote = _text(raw.get("quote"), "invalid_condition_quote", 20_000)
        refs = condition_sources.get(ident)
        if (not isinstance(refs, (list, tuple)) or not refs or len(refs) > 24
                or any(not isinstance(r, str) for r in refs) or len(set(refs)) != len(refs)):
            raise SearchValidationError("invalid_condition_sources")
        if any(ref not in sources for ref in refs):
            raise SearchValidationError("condition_source_outside_context")
        if not any(quote in sources[ref].text for ref in refs):
            raise SearchValidationError("quote_not_in_bound_source")
        if any(sources[ref].source_kind != raw["source_kind"] for ref in refs):
            raise SearchValidationError("condition_source_owner_mismatch")
        values = _string_list(raw.get("values", []), "invalid_condition_values", maximum=20)
        alternatives = raw.get("alternatives", [])
        if not isinstance(alternatives, list) or len(alternatives) > 10:
            raise SearchValidationError("invalid_condition_alternatives")
        alternatives = [_string_list(x, "invalid_condition_alternatives", maximum=20,
                                     allow_empty=False) for x in alternatives]
        clause_ids = _string_list(raw.get("clause_ids"), "invalid_clause_ids", maximum=30,
                                  allow_empty=False)
        if len(set(clause_ids)) != len(clause_ids):
            raise SearchValidationError("duplicate_clause_ids")
        low, high = raw.get("min"), raw.get("max")
        for value in (low, high):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise SearchValidationError("invalid_numeric_bound")
        if low is not None and high is not None and low > high:
            raise SearchValidationError("reversed_numeric_bounds")
        if not values and not alternatives and low is None and high is None:
            raise SearchValidationError("empty_condition")
        unit = raw.get("unit")
        if unit is not None:
            _text(unit, "invalid_condition_unit", 20)
        negated = raw.get("negated", False)
        if type(negated) is not bool:
            raise SearchValidationError("invalid_condition_negation")
        clean.append({
            "id": ident, "kind": raw["kind"], "name": name, "values": values,
            "alternatives": alternatives, "min": low, "max": high, "unit": unit,
            "negated": negated, "necessity": raw["necessity"], "source_kind": raw["source_kind"],
            "clause_ids": clause_ids, "quote": quote,
        })
        bindings[ident] = tuple(refs)
    if set(condition_sources) != ids:
        raise SearchValidationError("condition_binding_coverage_gap")
    hard = {c["id"] for c in clean if c["source_kind"] == "customer" and c["necessity"] == "required"}
    tree = _validate_expression(deepcopy(hard_expression), hard)
    return ValidatedConditions(tuple(clean), tree, bindings, digest)


def validate_plan_payload(payload: Mapping[str, Any]) -> ValidatedConditions | None:
    """Validate the host-bound v2.1 source envelope before a real service call.

    This checks provenance/shape, not semantic equivalence. The owner service
    independently audits confirmed conditions against the original source.
    """
    mode = payload.get("plan_source")
    window = payload.get("source_window")
    if mode not in PLAN_SOURCES or not isinstance(window, list):
        raise SearchValidationError("invalid_plan_source_envelope")
    messages = []
    for row in window:
        if not isinstance(row, dict) or set(row) != {"event_ref", "text", "source_kind"}:
            raise SearchValidationError("invalid_source_window_row")
        messages.append(SourceMessage(row["event_ref"], row["text"], row["source_kind"]))
    _, digest = _source_window(messages)
    if payload.get("source_window_sha256") != digest or payload.get("text") != "\n".join(m.text for m in messages):
        raise SearchValidationError("source_window_digest_or_text_mismatch")
    if mode == "original_text":
        if payload.get("conditions") or payload.get("hard_expression") is not None or payload.get("condition_sources"):
            raise SearchValidationError("explicit_plan_source_conflict")
        if any(m.source_kind != "customer" for m in messages):
            raise SearchValidationError("original_text_requires_customer_window")
        return None
    return validate_conditions(payload.get("conditions"), payload.get("hard_expression"), messages,
                               payload.get("condition_sources"))


def build_search_request(
    *, mode: str, scope: SearchScope, source_messages: Sequence[SourceMessage],
    reference: str | None = None, generation: str | None = None,
    plan_source: str = "original_text", conditions: Sequence[Mapping[str, Any]] = (),
    hard_expression: Mapping[str, Any] | None = None,
    condition_sources: Mapping[str, Sequence[str]] | None = None,
    exclude_product_codes: Sequence[str] = (), limit: int = 10,
    cursor: str | None = None,
) -> SearchRequest:
    """Build a route/payload; the caller must negotiate required capabilities.

    Exclusions are server-resolved, verified full code groups from the display
    ledger, not a model's guessed aliases. They only apply to recommendations.
    Exact lookup must still find an already displayed product.
    """
    if (not isinstance(mode, str) or mode not in SEARCH_MODES
            or not isinstance(plan_source, str) or plan_source not in PLAN_SOURCES):
        raise SearchValidationError("invalid_search_mode")
    if not isinstance(scope, SearchScope) or type(limit) is not int or not 1 <= limit <= 10:
        raise SearchValidationError("invalid_search_scope_or_limit")
    sources, digest = _source_window(source_messages)
    if cursor is not None:
        _text(cursor, "invalid_cursor", 2000)
        if mode != "resolve_candidates":
            raise SearchValidationError("cursor_not_supported_for_route")
    context: dict[str, Any] = {
        "tenant_ref": scope.tenant_ref, "customer_ref": scope.customer_ref,
        "source_event_refs": list(sources), "source_window_sha256": digest,
    }
    if mode == "resolve_exact":
        if conditions or hard_expression is not None or condition_sources or plan_source != "original_text":
            raise SearchValidationError("exact_lookup_conditions_require_separate_verification")
        code = normalize_reference(reference)  # type: ignore[arg-type]
        if code.isdigit():
            raise SearchValidationError("bare_code_requires_candidate_route")
        _text(generation, "catalog_generation_required", 160)
        # Exact existing endpoint: keep its precise three-key request contract.
        return SearchRequest("catalog", "/v4/catalog/by-code",
                             {"region": scope.region, "generation": generation, "dcode": code},
                             context, frozenset())
    if plan_source == "original_text":
        if conditions or hard_expression is not None or condition_sources:
            raise SearchValidationError("explicit_plan_source_conflict")
        if any(m.source_kind != "customer" for m in source_messages):
            raise SearchValidationError("original_text_requires_customer_window")
        plan: dict[str, Any] = {"plan_source": "original_text"}
    else:
        checked = validate_conditions(conditions, hard_expression, source_messages, condition_sources or {})
        plan = {"plan_source": "confirmed_conditions", "conditions": list(checked.conditions),
                "hard_expression": checked.hard_expression,
                "condition_sources": {k: list(v) for k, v in checked.condition_sources.items()}}
    # The proposed plan_source contract retains original evidence in both modes.
    plan.update(text="\n".join(m.text for m in source_messages),
                source_window=[{"event_ref": m.event_ref, "text": m.text, "source_kind": m.source_kind}
                               for m in source_messages], source_window_sha256=digest)
    if mode == "resolve_candidates":
        if exclude_product_codes:
            raise SearchValidationError("candidate_identity_search_must_not_hide_seen_products")
        code = normalize_reference(reference)  # type: ignore[arg-type]
        _text(generation, "catalog_generation_required", 160)
        body = {"region": scope.region, "generation": generation, "reference": code,
                "limit": limit, **plan}
        if cursor is not None:
            body["cursor"] = cursor
        return SearchRequest("catalog", "/v4/catalog/search", body, context,
                             frozenset({"catalog.search.v1", "matcher.plan_source.v1"}))
    if reference is not None:
        raise SearchValidationError("recommendation_reference_requires_resolve_first")
    if isinstance(exclude_product_codes, (str, bytes)) or len(exclude_product_codes) > 1000:
        raise SearchValidationError("invalid_product_exclusions")
    excluded = list(dict.fromkeys(normalize_reference(code) for code in exclude_product_codes))
    body = {"region": scope.region, "scene": "customer_service",
            "mode": "reorder" if mode == "reorder" else "new_opportunity",
            "result_contract": "formal_top10", "limit": limit,
            "exclude_product_codes": excluded, **plan}
    if scope.confirmed_customer_code is not None:
        body["customer_code"] = scope.confirmed_customer_code
    return SearchRequest("matcher", "/v4/match/products", body, context,
                         frozenset({"matcher.plan_source.v1"}))


def _positive_id(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise SearchValidationError("invalid_goods_id")
    return value


def stable_product_key(
    *, tenant_ref: str, source_namespace: str, goods_id: int,
    verified_alias: Mapping[str, Any] | None = None,
) -> str:
    """Use Store.identity semantics; never include generation, query or prices."""
    _text(tenant_ref, "invalid_tenant")
    _text(source_namespace, "invalid_source_namespace")
    canonical = _positive_id(goods_id)
    if verified_alias is not None:
        if (verified_alias.get("kind") != "same_goods_alias" or verified_alias.get("verified") is not True
                or verified_alias.get("goods_id") != goods_id or not verified_alias.get("source")):
            raise SearchValidationError("unverified_alias_mapping")
        canonical = _positive_id(verified_alias.get("canonical_goods_id"))
    return "product:" + _json_hash([tenant_ref, source_namespace, canonical])


def stable_display_key(
    *, tenant_ref: str, source_namespace: str, goods_id: int,
    verified_alias: Mapping[str, Any] | None = None,
    verified_color_mapping: Mapping[str, Any] | None = None,
    expand_colors: bool = False,
) -> str:
    """Mirror verified display grouping; do not infer families from codes/images."""
    if type(expand_colors) is not bool:
        raise SearchValidationError("invalid_expand_colors")
    product_key = stable_product_key(tenant_ref=tenant_ref, source_namespace=source_namespace,
                                     goods_id=goods_id, verified_alias=verified_alias)
    if verified_color_mapping is not None:
        mapping = verified_color_mapping
        family = mapping.get("family_id")
        if (mapping.get("kind") != "same_style_color_variant" or mapping.get("verified") is not True
                or mapping.get("goods_id") != goods_id or not mapping.get("source")
                or type(family) not in (str, int) or not str(family).strip()):
            raise SearchValidationError("unverified_color_mapping")
        if not expand_colors:
            return "display:" + _json_hash([tenant_ref, source_namespace, "color_family", str(family)])
    return "display:" + _json_hash([product_key])
