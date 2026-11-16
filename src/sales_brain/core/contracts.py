"""Strict v2.1 planning contracts, using only the Python standard library.

This is an executable design reference, not a deployed adapter. A plan proposes
actions; neither validation nor assembly executes a business write. Authority,
intent confirmation and immutable operation identities come from trusted runtime
context, never from the planner's JSON. The validator intentionally supports a
documented JSON Schema subset and rejects unsupported schema keywords.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
from typing import Any, Mapping


class ContractError(ValueError):
    def __init__(self, code: str, message: str, path: str = "$"):
        self.code, self.message, self.path = code, message, path
        super().__init__(f"{code} at {path}: {message}")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


REF = {"type": "string", "minLength": 1, "maxLength": 240}
TEXT = {"type": "string", "maxLength": 6000}
NONEMPTY_TEXT = {**TEXT, "minLength": 1}
REFS = {"type": "array", "items": REF, "maxItems": 64, "uniqueItems": True}
SOURCES = {**REFS, "minItems": 1}
NULL = {"type": "null"}
VERSION = {"anyOf": [REF, {"type": "integer", "minimum": 0}]}
TIMESTAMP = {"type": "string", "format": "date-time", "maxLength": 64}
BINDING = obj({"from_action_id": REF,
               "result_path": {"enum": ["resource_ref", "resource_kind", "resource_version"]}})
BOUND_REF = {"anyOf": [REF, BINDING]}
BOUND_VERSION = {"anyOf": [VERSION, BINDING]}
OPTIONAL_REF = {"anyOf": [BOUND_REF, NULL]}
OPTIONAL_VERSION = {"anyOf": [BOUND_VERSION, NULL]}
DEPENDENCY = obj({"action_id": REF, "condition": {"enum": ["success", "terminal"]}})
EXECUTION_STATUSES = ["pending", "running", "succeeded", "failed", "unknown", "blocked", "cancelled"]
GUARD_SCHEMA = {"anyOf": [
    obj({"kind": {"const": "context_version"}, "expected": REF}),
    obj({"kind": {"const": "config_version"}, "expected": REF}),
    obj({"kind": {"const": "permission"}, "permission": REF}),
    obj({"kind": {"const": "not_cancelled"}}),
    obj({"kind": {"const": "not_taken_over"}}),
    obj({"kind": {"const": "resource_version"}, "resource_ref": BOUND_REF,
         "resource_kind": BOUND_REF, "expected_version": BOUND_VERSION}),
    obj({"kind": {"const": "source_current"}, "source_ref": REF, "expected_version": VERSION}),
]}
WHEN = obj({"action_id": REF,
            "execution_status": {"type": "array", "items": {"enum": EXECUTION_STATUSES},
                                 "minItems": 1, "uniqueItems": True, "maxItems": 7},
            "business_status": SOURCES, "delivery_status": SOURCES},
           ["action_id", "execution_status"])
SEGMENT = obj({"text": NONEMPTY_TEXT, "when": WHEN})
RANGE = obj({"min": {"type": "number", "minimum": 0}, "max": {"type": "number", "minimum": 0}})
QUANTITY = obj({"value": {"type": "number", "minimum": 0}, "unit": REF})
REQUIREMENT_FIELDS = {
    "need_text": TEXT, "purpose": TEXT, "material": TEXT, "composition": TEXT,
    "weight_gsm": RANGE, "width_cm": RANGE, "quantity": QUANTITY,
    "color": TEXT, "budget": QUANTITY, "lead_time_days": {"type": "integer", "minimum": 0},
    "excluded_materials": REFS, "excluded_product_refs": REFS, "alternatives": REFS,
}
INTEREST = obj({"product_ref": REF, "source_event_refs": SOURCES,
                "status": {"enum": ["interested", "selected", "excluded"]}})
DISPLAY_MODES = ["new_card", "reference_existing", "update_existing", "explicit_redisplay", "media_only"]
PRODUCT_ITEM = obj({
    "product_ref": REF, "product_key": REF, "display_key": REF,
    "mode": {"enum": DISPLAY_MODES}, "source_event_refs": SOURCES,
    "existing_presentation_ref": {"anyOf": [REF, NULL]}, "media_refs": REFS,
})
REQUEST_PARAMETERS = {
    "sales_followup": obj({"summary": NONEMPTY_TEXT, "reason": NONEMPTY_TEXT, "product_refs": REFS,
                           "preferred_contact_ref": {"anyOf": [REF, NULL]},
                           "no_outbound_contact": {"type": "boolean"},
                           "priority": {"enum": ["low", "normal", "high"]}},
                          ["summary", "reason", "product_refs"]),
    "callback": obj({"contact_ref": REF, "channel_route_ref": REF, "summary": TEXT,
                     "preferred_time": TEXT}, []),
    "formal_quote": obj({"product_refs": SOURCES, "quote_refs": REFS, "quantity": QUANTITY,
                         "currency": REF, "terms_ref": REF, "summary": TEXT}, []),
    "sample": obj({"product_refs": SOURCES, "quantity": QUANTITY, "recipient_ref": REF,
                   "address_ref": REF, "shipping_method_ref": REF, "summary": TEXT}, []),
    "order": obj({"product_refs": SOURCES, "quote_refs": REFS, "quantity": QUANTITY,
                  "terms_ref": REF, "customer_confirmation_event_ref": REF,
                  "address_ref": REF, "summary": TEXT}, []),
    "group_invite": obj({"group_ref": REF, "channel_identity_ref": REF,
                         "consent_event_ref": REF, "summary": TEXT}, []),
    "payment": obj({"order_ref": REF, "payment_method_ref": REF, "summary": TEXT}, []),
    "special_price": obj({"quote_ref": REF, "request_event_ref": REF, "summary": TEXT}, []),
    "human_support": obj({"contact_ref": REF, "summary": TEXT}, []),
    "responsibility": obj({"contact_ref": REF, "summary": TEXT}, []),
}
FACT_CONTENT = obj({"kind": {"enum": ["profile", "preference", "relationship"]},
                    "key": REF, "value": TEXT, "source_quote": NONEMPTY_TEXT})
FACT_SCOPE = obj({"kind": {"enum": ["personal", "region"]}, "region": {"anyOf": [REF, NULL]}})
PAYLOADS = {
    "REPLY_TEXT": obj({"body": TEXT, "fact_refs": REFS, "reply_group": REF,
                       "conditional_segments": {"type": "array", "items": SEGMENT, "maxItems": 32}}),
    "ASK_CLARIFICATION": obj({"question": NONEMPTY_TEXT, "missing_fields": SOURCES,
                              "intent_refs": SOURCES, "reply_group": REF, "fact_refs": REFS},
                             ["question", "missing_fields", "intent_refs", "reply_group"]),
    "RECOMMEND_PRODUCTS": obj({
        "items": {"type": "array", "items": PRODUCT_ITEM, "minItems": 1, "maxItems": 10},
        "product_refs": {**SOURCES, "maxItems": 10}, "quote_refs": REFS,
        "ranking_evidence_refs": SOURCES, "display_reason": TEXT, "reply_group": REF,
        "case_ref": OPTIONAL_REF, "recommendation_ref": REF,
    }, ["items", "product_refs", "quote_refs", "ranking_evidence_refs", "display_reason", "reply_group"]),
    "PROPOSE_PROJECT_PATCH": obj({
        "operation": {"enum": ["create", "update", "cancel", "close", "reopen"]},
        "case_ref": OPTIONAL_REF, "expected_version": OPTIONAL_VERSION,
        "product_interests": {"type": "array", "items": INTEREST, "maxItems": 32},
        "requirements": obj(REQUIREMENT_FIELDS, []), "open_intents": REFS,
        "clear_fields": {"type": "array", "items": {"enum": list(REQUIREMENT_FIELDS) + ["product_interests", "open_intents"]},
                         "maxItems": 24, "uniqueItems": True}, "source_event_refs": SOURCES,
    }),
    "HANDOFF_SALES": {"anyOf": [obj({
        "request_type": {"const": kind}, "operation": REF, "command_ref": REF,
        "case_ref": BOUND_REF, "expected_version": BOUND_VERSION,
        "request_ref": OPTIONAL_REF, "target_result": REF,
        "parameters": parameters, "source_event_refs": SOURCES,
    }, ["request_type", "operation", "command_ref", "case_ref", "expected_version",
        "target_result", "parameters", "source_event_refs"]) for kind, parameters in REQUEST_PARAMETERS.items()]},
    "PROPOSE_CUSTOMER_FACT": obj({
        "operation": {"enum": ["create", "revise", "reject", "retire", "delete", "forbid", "allow"]},
        "subject_ref": REF, "scope": FACT_SCOPE, "fact_ref": OPTIONAL_REF,
        "expected_version": OPTIONAL_VERSION, "content": {"anyOf": [FACT_CONTENT, NULL]},
        "confirmation": {"enum": ["explicit", "inferred", "reviewed"]},
        "source_event_refs": SOURCES, "suppression_ref": OPTIONAL_REF,
    }, ["operation", "subject_ref", "scope", "fact_ref", "expected_version", "content",
        "confirmation", "source_event_refs"]),
    "NOTIFY_SALES": obj({"request_ref": BOUND_REF, "recipient_ref": REF, "channel_route_ref": REF,
                         "body": NONEMPTY_TEXT, "source_event_refs": SOURCES, "fact_refs": REFS}),
    "NO_ACTION": obj({"reason_code": {"enum": ["no_pending_event", "no_response_needed", "manual_takeover",
                                                  "opted_out", "already_handled", "blocked"]},
                      "covered_event_refs": REFS, "unresolved_intent_refs": REFS}),
}
ACTION_TYPES = frozenset(PAYLOADS)
CUSTOMER_SEND_ACTIONS = frozenset({"REPLY_TEXT", "ASK_CLARIFICATION", "RECOMMEND_PRODUCTS"})
ACTION_SCHEMA = {"anyOf": [obj({
    "action_id": REF, "type": {"const": kind}, "intent_ref": REF, "operation_key": REF,
    "depends_on": {"type": "array", "items": DEPENDENCY, "maxItems": 32},
    "guards": {"type": "array", "items": GUARD_SCHEMA, "maxItems": 32}, "payload": payload,
}) for kind, payload in PAYLOADS.items()]}
DRAFT_ACTION_SCHEMA = deepcopy(ACTION_SCHEMA)
for _variant in DRAFT_ACTION_SCHEMA["anyOf"]:
    _variant["properties"].pop("operation_key")
    _variant["required"].remove("operation_key")
INTENT_KINDS = ["contact_request", "product_search", "product_interest", "product_compare", "product_reshow", "product_media_request", "quote_request", "sample_request",
                "order_request", "callback_request", "customer_fact", "general_question", "stop_contact"]
INTENT_SCHEMA = obj({
    "intent_ref": REF, "kind": {"enum": INTENT_KINDS},
    "confirmation": {"enum": ["confirmed", "candidate"]}, "source_event_refs": SOURCES,
    "status": {"enum": ["answered", "planned", "needs_clarification", "blocked", "cancelled"]},
    "action_ids": REFS, "reason": TEXT,
})
DRAFT_INTENT_SCHEMA = deepcopy(INTENT_SCHEMA)
DRAFT_INTENT_SCHEMA["properties"]["confirmation"] = {"const": "candidate"}
DRAFT_PLAN_SCHEMA = obj({
    "status": {"enum": ["ready", "needs_clarification"]},
    "intent_coverage": {"type": "array", "items": DRAFT_INTENT_SCHEMA, "maxItems": 64},
    "actions": {"type": "array", "items": DRAFT_ACTION_SCHEMA, "minItems": 1, "maxItems": 32},
})
CONTACT_POLICY_PROJECTION_SCHEMA = obj({
    "projection_ref": REF, "policy_kind": {"const": "contact_restriction"},
    "contact_scope": {"enum": ["outbound_only", "all_contact"]},
    "tenant_ref": REF, "customer_ref": REF, "region": REF,
    "environment": {"enum": ["sandbox", "shadow", "live"]},
    "intent_ref": REF, "source_event_refs": SOURCES, "source_context_version": REF,
    "status": {"const": "active"},
})
PLAN_SCHEMA = obj({
    "schema_version": {"const": "2.1"}, "run_id": REF, "decision_id": REF,
    "tenant_ref": REF, "customer_ref": REF, "region": REF,
    "environment": {"enum": ["sandbox", "shadow", "live"]}, "conversation_ref": REF,
    "task_ref": REF, "plan_revision": {"type": "integer", "minimum": 1},
    "context_version": REF, "config_version": REF, "trigger_event_refs": REFS,
    "expires_at": TIMESTAMP, "status": {"enum": ["ready", "needs_clarification"]},
    "contact_policy_projection": {"type": "array", "items": CONTACT_POLICY_PROJECTION_SCHEMA, "maxItems": 64},
    "intent_coverage": {"type": "array", "items": INTENT_SCHEMA, "maxItems": 64},
    "actions": {"type": "array", "items": ACTION_SCHEMA, "minItems": 1, "maxItems": 32},
})
PLAN_SCHEMA["$schema"] = "https://json-schema.org/draft/2020-12/schema"


def model_step_schema(tools: list[Mapping[str, Any]]) -> dict:
    variants = [obj({"call_id": REF, "name": {"const": t["name"]}, "arguments": t["parameters"]}) for t in tools]
    branches = [obj({"kind": {"const": "final"}, "plan": DRAFT_PLAN_SCHEMA})]
    if variants:
        branches.insert(0, obj({"kind": {"const": "read_tools"}, "calls": {
            "type": "array", "items": {"anyOf": variants}, "minItems": 1, "maxItems": 4}}))
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "anyOf": branches}


# Portable envelope only. At runtime model_step_schema() replaces the generic
# name/arguments with the authorized tool names and their exact input schemas.
MODEL_STEP_SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema", "anyOf": [
    obj({"kind": {"const": "read_tools"}, "calls": {"type": "array", "minItems": 1, "maxItems": 4,
         "items": obj({"call_id": REF, "name": REF, "arguments": {"type": "object"}})}}),
    obj({"kind": {"const": "final"}, "plan": DRAFT_PLAN_SCHEMA}),
]}


def parse_timestamp(value: str) -> datetime:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", value):
        raise ValueError("RFC3339 timestamp with timezone required")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


_KEYWORDS = {"$schema", "$id", "title", "description", "type", "properties", "required", "additionalProperties",
             "enum", "const", "anyOf", "oneOf", "items", "minItems", "maxItems", "uniqueItems",
             "minLength", "maxLength", "minimum", "maximum", "pattern", "format"}


def check_schema(schema: dict) -> None:
    if not isinstance(schema, dict) or set(schema) - _KEYWORDS:
        raise ContractError("unsupported_schema", "only the documented strict JSON Schema subset is supported")
    if schema.get("format") not in {None, "date-time"}:
        raise ContractError("unsupported_schema", "unsupported string format")
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if not isinstance(kinds, list) or set(kinds) - {"object", "array", "string", "integer", "number", "boolean", "null"}:
        raise ContractError("unsupported_schema", "invalid type")
    for name in ("anyOf", "oneOf"):
        if name in schema:
            if not isinstance(schema[name], list) or not schema[name]:
                raise ContractError("unsupported_schema", "empty schema branches")
            for child in schema[name]:
                check_schema(child)
    for child in schema.get("properties", {}).values():
        check_schema(child)
    if "items" in schema:
        check_schema(schema["items"])
    extra = schema.get("additionalProperties", True)
    if isinstance(extra, dict):
        check_schema(extra)
    elif type(extra) is not bool:
        raise ContractError("unsupported_schema", "invalid additionalProperties")


def _validate(value: Any, schema: dict, path: str) -> None:
    def fail(message: str):
        raise ContractError("invalid_contract", message, path)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            successes, errors = 0, []
            for child in schema[key]:
                try:
                    _validate(value, child, path)
                    successes += 1
                except ContractError as exc:
                    errors.append(exc)
            if not successes:
                raise max(errors, key=lambda e: len(e.path))
            if key == "oneOf" and successes != 1:
                fail("oneOf must match exactly one branch")
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        fail("constant mismatch")
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        fail("value outside enum")
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    matches = {"object": type(value) is dict, "array": type(value) is list, "string": type(value) is str,
               "integer": type(value) is int, "number": type(value) in (int, float) and math.isfinite(value),
               "boolean": type(value) is bool, "null": value is None}
    if kinds and not any(matches[k] for k in kinds):
        fail("incorrect JSON type")
    if type(value) is dict:
        if set(schema.get("required", [])) - set(value):
            fail("missing required fields")
        props, extra = schema.get("properties", {}), schema.get("additionalProperties", True)
        for name, child in value.items():
            if name in props:
                _validate(child, props[name], f"{path}.{name}")
            elif extra is False:
                fail(f"unexpected field: {name}")
            elif isinstance(extra, dict):
                _validate(child, extra, f"{path}.{name}")
    if type(value) is list:
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            fail("array size outside bounds")
        if schema.get("uniqueItems") and len({canonical(v) for v in value}) != len(value):
            fail("duplicate array items")
        if "items" in schema:
            for i, child in enumerate(value):
                _validate(child, schema["items"], f"{path}[{i}]")
    if type(value) is str:
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
            fail("string length outside bounds")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            fail("pattern mismatch")
        if schema.get("format") == "date-time":
            try:
                parse_timestamp(value)
            except ValueError:
                fail("invalid RFC3339 timestamp")
    if type(value) in (int, float):
        if not math.isfinite(value) or not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            fail("number outside bounds")


def validate_json(value: Any, schema: dict) -> None:
    check_schema(schema)
    def json_only(item: Any, depth: int = 0):
        if depth > 80:
            raise ContractError("invalid_json", "JSON nesting is too deep")
        if type(item) is dict:
            if not all(type(key) is str for key in item):
                raise ContractError("invalid_json", "JSON object keys must be strings")
            for child in item.values():
                json_only(child, depth + 1)
        elif type(item) is list:
            for child in item:
                json_only(child, depth + 1)
        elif type(item) not in {str, int, float, bool, type(None)}:
            raise ContractError("invalid_json", "only JSON values are allowed")
    json_only(value)
    try:
        canonical(value)  # Reject non-JSON values and non-finite numbers, including inside free-form objects.
    except (TypeError, ValueError) as exc:
        raise ContractError("invalid_json", "only finite JSON values are allowed") from exc
    _validate(value, schema, "$")


def validate_model_step(step: dict, tools: list[Mapping[str, Any]]) -> None:
    validate_json(step, model_step_schema(tools))
    if step["kind"] == "read_tools" and len({c["call_id"] for c in step["calls"]}) != len(step["calls"]):
        raise ContractError("duplicate_call_id", "tool call IDs must be distinct in one response")


def _bindings(value: Any, field: str = ""):
    if isinstance(value, dict):
        if "from_action_id" in value:
            yield field, value
        else:
            for key, child in value.items():
                yield from _bindings(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from _bindings(child, field)


def validate_plan(plan: Mapping[str, Any]) -> dict:
    data = deepcopy(dict(plan))
    validate_json(data, PLAN_SCHEMA)
    actions = data["actions"]
    projections = data["contact_policy_projection"]
    if len({p["projection_ref"] for p in projections}) != len(projections):
        raise ContractError("duplicate_policy_projection", "contact policy projections must be unique")
    for projection in projections:
        if any(projection[key] != data[key] for key in ("tenant_ref", "customer_ref", "region", "environment")):
            raise ContractError("policy_projection_scope", "contact policy projection differs from plan scope")
    by_id = {a["action_id"]: a for a in actions}
    if len(by_id) != len(actions) or len({a["operation_key"] for a in actions}) != len(actions):
        raise ContractError("duplicate_action", "action and operation IDs must be unique")
    if any(a["type"] == "NO_ACTION" for a in actions) and len(actions) != 1:
        raise ContractError("invalid_no_action", "NO_ACTION must stand alone")
    graph = {}
    producers = {"case_ref": "PROPOSE_PROJECT_PATCH", "fact_ref": "PROPOSE_CUSTOMER_FACT",
                 "suppression_ref": "PROPOSE_CUSTOMER_FACT", "request_ref": "HANDOFF_SALES"}
    for action_index, a in enumerate(actions):
        aid, p, kind = a["action_id"], a["payload"], a["type"]
        deps = {d["action_id"]: d["condition"] for d in a["depends_on"]}
        if len(deps) != len(a["depends_on"]) or aid in deps or set(deps) - set(by_id):
            raise ContractError("invalid_dependency", "unknown, duplicate or self dependency")
        graph[aid] = set(deps)
        for field, binding in _bindings({"payload": p, "guards": a["guards"]}):
            source = binding["from_action_id"]
            if deps.get(source) != "success":
                raise ContractError("invalid_binding", "a receipt binding needs a success dependency")
            expected = "resource_version" if field == "expected_version" else "resource_kind" if field == "resource_kind" else "resource_ref"
            if binding["result_path"] != expected:
                raise ContractError("invalid_binding", "receipt field type mismatch")
            if field in producers and by_id[source]["type"] != producers[field]:
                raise ContractError("invalid_binding", "wrong producer action type")
        if kind == "REPLY_TEXT":
            if not p["body"].strip() and not p["conditional_segments"]:
                raise ContractError("empty_reply", "reply must have content")
            for segment_index, segment in enumerate(p["conditional_segments"]):
                if segment["when"]["action_id"] not in deps:
                    error = ContractError("invalid_condition", "conditional text needs a receipt dependency")
                    error.diagnostics = {'code': error.code,
                        'field_path': ['actions', action_index, 'payload', 'conditional_segments', segment_index, 'when', 'action_id']}
                    raise error
        if kind == "RECOMMEND_PRODUCTS":
            items = p["items"]
            if [x["product_ref"] for x in items] != p["product_refs"]:
                raise ContractError("product_index_mismatch", "product_refs must equal ordered items.product_ref")
            if len({x["display_key"] for x in items}) != len(items):
                raise ContractError("duplicate_display_key", "a plan cannot repeat one presentation identity")
            for item in items:
                previous = item["existing_presentation_ref"]
                if item["mode"] == "new_card" and previous is not None:
                    raise ContractError("invalid_presentation", "new_card cannot reference an old presentation")
                if item["mode"] in {"reference_existing", "update_existing", "explicit_redisplay"} and previous is None:
                    raise ContractError("invalid_presentation", "this mode requires an existing presentation")
                if item["mode"] == "media_only" and not item["media_refs"]:
                    raise ContractError("invalid_presentation", "media_only requires media references")
        if kind == "PROPOSE_PROJECT_PATCH":
            if p["operation"] == "create":
                if p["case_ref"] is not None or p["expected_version"] is not None or p["clear_fields"]:
                    raise ContractError("invalid_create", "case create needs null identity/version and no clears")
            elif p["case_ref"] is None or p["expected_version"] is None:
                raise ContractError("version_required", "case mutation requires identity/version")
            for field in ("width_cm", "weight_gsm"):
                value = p["requirements"].get(field)
                if value and value["min"] > value["max"]:
                    raise ContractError("invalid_range", "range endpoints reversed")
        if kind == "PROPOSE_CUSTOMER_FACT":
            if p["subject_ref"] != data["customer_ref"]:
                raise ContractError("subject_mismatch", "personal fact subject differs from plan customer")
            if (p["scope"]["kind"] == "personal") != (p["scope"]["region"] is None):
                raise ContractError("fact_scope", "personal/region scope fields disagree")
            if p["scope"]["kind"] == "region" and p["scope"]["region"] != data["region"]:
                raise ContractError("fact_scope", "fact region differs from plan scope")
            op = p["operation"]
            if op == "create" and (p["fact_ref"] is not None or p["expected_version"] is not None or p["content"] is None):
                raise ContractError("invalid_create", "fact create requires content and null existing identity/version")
            if op in {"revise", "reject", "retire", "delete"} and (p["fact_ref"] is None or p["expected_version"] is None):
                raise ContractError("version_required", "existing fact operation needs identity/version")
            if op == "revise" and p["content"] is None:
                raise ContractError("fact_content", "fact revision requires content")
            if op == "allow" and not p.get("suppression_ref"):
                raise ContractError("suppression_required", "allow requires an existing suppression")
            if op == "forbid" and p["fact_ref"] is None and p["content"] is None:
                raise ContractError("fact_predicate", "forbid needs a fact or predicate")
    remaining = set(graph)
    while remaining:
        ready = {a for a in remaining if not (graph[a] & remaining)}
        if not ready:
            raise ContractError("cyclic_dependency", "action graph must be acyclic")
        remaining -= ready
    rows = {r["intent_ref"]: r for r in data["intent_coverage"]}
    if len(rows) != len(data["intent_coverage"]):
        raise ContractError("duplicate_intent", "coverage intents must be unique")
    for row in rows.values():
        if set(row["action_ids"]) - set(by_id):
            raise ContractError("coverage_action", "coverage references a missing action")
        if row["status"] in {"answered", "planned", "needs_clarification"} and not row["action_ids"]:
            raise ContractError("unbacked_coverage", "active coverage needs an action")
        if row["status"] in {"blocked", "cancelled"} and not row["reason"].strip():
            raise ContractError("coverage_reason", "blocked/cancelled coverage needs an explanation")
    for action_index, a in enumerate(actions):
        if a["type"] != "NO_ACTION" and (a["intent_ref"] not in rows or a["action_id"] not in rows[a["intent_ref"]]["action_ids"]):
            error = ContractError("unbound_action", "action must belong to its covered intent")
            error.diagnostics = {'code': error.code, 'field_path': ['actions', action_index, 'intent_ref'],
                'intent_coverage_row_exists': a['intent_ref'] in rows,
                'action_listed_in_own_coverage': a['intent_ref'] in rows and a['action_id'] in rows[a['intent_ref']]['action_ids']}
            raise error
    if actions[0]["type"] == "NO_ACTION" and any(r["status"] in {"answered", "planned", "needs_clarification"} for r in rows.values()):
        raise ContractError("invalid_no_action", "NO_ACTION cannot fulfill an active request")
    return data


def _source_lists(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"source_event_refs", "covered_event_refs"}:
                yield child
            elif key in {"customer_confirmation_event_ref", "request_event_ref", "consent_event_ref"}:
                yield [child]
            else:
                yield from _source_lists(child)
    elif isinstance(value, list):
        for child in value:
            yield from _source_lists(child)


def assemble_plan(draft: dict, context: Mapping[str, Any], *, run_id: str, decision_id: str,
                  now: datetime | None = None, ttl_seconds: int = 120) -> dict:
    """Bind a planner draft to runtime metadata and source-verified intent kinds.

    ``context`` is supplied by the authenticated brain runtime, never request JSON.
    The executor must still perform current-state, capability and business checks.
    """
    validate_json(draft, DRAFT_PLAN_SCHEMA)
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None or not 1 <= ttl_seconds <= 600:
        raise ContractError("invalid_assembly", "timezone and bounded TTL are required")
    result = deepcopy(draft)
    for key in ("tenant_ref", "customer_ref", "region", "environment", "conversation_ref", "task_ref",
                "plan_revision", "context_version", "config_version"):
        result[key] = deepcopy(context[key])
    result.update(schema_version="2.1", run_id=run_id, decision_id=decision_id,
                  trigger_event_refs=list(context["trigger_event_refs"]),
                  contact_policy_projection=deepcopy(list(context.get("contact_policy_projection", []))),
                  expires_at=(current_time + timedelta(seconds=ttl_seconds)).isoformat())
    allowed = {m["event_ref"] for m in context["messages"] if m.get("role") == "customer"
               and not m.get("removed") and not m.get("body_truncated")}
    if not set(result["trigger_event_refs"]) <= allowed:
        raise ContractError("source_out_of_scope", "trigger events are not available customer sources")
    for refs in _source_lists(result):
        if not set(refs) <= allowed:
            raise ContractError("source_out_of_scope", "plan cites a missing, removed or non-customer event")
    confirmed = {i["intent_ref"]: i for i in context.get("confirmed_intents", [])}
    covered = {i["intent_ref"] for i in result["intent_coverage"]}
    if set(confirmed) - covered:
        raise ContractError("missing_intent_coverage", "a confirmed pending intent was omitted")
    for row in result["intent_coverage"]:
        trusted = confirmed.get(row["intent_ref"])
        if trusted is not None:
            if row["kind"] != trusted["kind"] or set(row["source_event_refs"]) != set(trusted["source_event_refs"]):
                raise ContractError("intent_binding_mismatch", "planner cannot reclassify or replace confirmed intent sources")
            row["confirmation"] = "confirmed"
    for a in result["actions"]:
        control = context.get("control", {})
        takeover = control.get("manual_takeover") is True
        internal_assist = control.get("internal_assist_during_takeover") is True
        customer_suppressed = takeover or control.get("opted_out") is True or control.get("customer_send_enabled") is False
        customer_outreach_request = (a["type"] == "HANDOFF_SALES"
                                     and a["payload"]["request_type"] in {"callback", "group_invite"})
        if control.get("no_outbound_contact") is True and customer_outreach_request:
            raise ContractError("outbound_contact_suppressed", "verified customer policy prohibits proactive outreach")
        if (control.get("no_outbound_contact") is True and a["type"] == "HANDOFF_SALES"
                and a["payload"]["request_type"] == "sales_followup"):
            # Host-bound command constraint, not a planner-selected opt-in.
            a["payload"]["parameters"]["no_outbound_contact"] = True
        if customer_suppressed and (a["type"] in CUSTOMER_SEND_ACTIONS or customer_outreach_request):
            raise ContractError("customer_send_suppressed", "trusted control disallows autonomous customer delivery")
        if a["type"] != "NO_ACTION" and (control.get("opted_out") is True or (takeover and not internal_assist)):
            raise ContractError("internal_assist_not_authorized", "this reference runtime pauses all proposals under this control state")
        # Stable across transport retries of one task revision. Executors must
        # reject the same operation_key with a different immutable payload hash.
        a["operation_key"] = "op:" + digest([result[k] for k in (
            "tenant_ref", "environment", "customer_ref", "conversation_ref", "task_ref", "plan_revision")]
            + [a["action_id"]])
        fixed = [{"kind": "context_version", "expected": result["context_version"]},
                 {"kind": "config_version", "expected": result["config_version"]},
                 {"kind": "not_cancelled"}]
        if a["type"] != "NO_ACTION" and (a["type"] in CUSTOMER_SEND_ACTIONS or not internal_assist):
            fixed.append({"kind": "not_taken_over"})
        elif a["type"] != "NO_ACTION" and internal_assist:
            fixed.append({"kind": "permission", "permission": "sales.internal_assist_during_takeover"})
        # Authority-bearing guards come from context, not an LLM-provided version.
        a["guards"] = fixed + [g for g in a["guards"] if g["kind"] not in {
            "context_version", "config_version", "not_cancelled", "not_taken_over"}]
    return validate_plan(result)
